"""CLIP clustering and Equation (1), with explicit build/save/load lifecycle."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from config_tuner.records import TuningResult


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(np.dot(a, b) / denominator, 0, 1)) if denominator else 0.0


def similarity_weight(sim, eps=1e-6):
    return max(0.0, -math.log(1.0 - float(np.clip(sim, 0, 1)) + eps))


def video_feature(frame_features):
    features = np.asarray(frame_features, dtype=float)
    if features.ndim != 2 or features.shape[0] != 16:
        raise ValueError('Knowledge-base CLIP input must contain 16 frame features')
    feature = features.mean(axis=0)
    return feature / max(float(np.linalg.norm(feature)), 1e-12)


def _kmeans(x, k, rng, restarts=5):
    best = None
    for _ in range(restarts):
        centers = x[rng.choice(len(x), k, replace=False)].copy()
        for _ in range(100):
            labels = ((x[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
            new = np.stack([x[labels == j].mean(axis=0) if np.any(labels == j)
                            else x[int(rng.integers(len(x)))] for j in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        labels = ((x[:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels, centers)
    return best[1:]


def _silhouette(distances, labels):
    clusters = set(labels.tolist())
    if len(clusters) < 2:
        return -1.0
    values = []
    for i, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        if len(same) <= 1:
            values.append(0.0)
            continue
        a = float(distances[i, same].sum() / (len(same) - 1))
        b = min(float(distances[i, labels == c].mean()) for c in clusters if c != label)
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values))


class VideoKnowledgeBase:
    def __init__(self, context, features=None):
        self.context = dict(context)
        self.features = {str(k): np.asarray(v, dtype=float) for k, v in (features or {}).items()}
        self.video_ids = []
        self.results = {}
        self.n_clusters = 0
        self.similarity_threshold = None

    def build(self, fraction=.1, seed=42):
        ids = sorted(self.features)
        if not ids or not 0 < fraction <= 1:
            raise ValueError('Empty features or invalid knowledge-base fraction')
        x = np.stack([self.features[v] for v in ids])
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Invalid video features')
        count = max(1, min(len(ids), math.ceil(len(ids) * fraction)))
        rng = np.random.default_rng(seed)
        if len(ids) < 3 or count == 1:
            labels, centers = np.zeros(len(ids), dtype=int), x.mean(axis=0, keepdims=True)
        else:
            distances = np.linalg.norm(x[:, None] - x[None, :], axis=2)
            candidates = []
            for k in range(2, min(count, len(ids) - 1) + 1):
                labels, centers = _kmeans(x, k, rng)
                candidates.append((_silhouette(distances, labels), -k, labels, centers))
            _, _, labels, centers = max(candidates, key=lambda c: (c[0], c[1]))
        self.n_clusters = len(centers)
        sizes = np.bincount(labels, minlength=len(centers))
        raw = sizes / len(ids) * count
        quotas = np.floor(raw).astype(int)
        for j in np.argsort(-(raw - quotas))[:count - int(quotas.sum())]:
            quotas[j] += 1
        picks = []
        for j, quota in enumerate(quotas):
            members = np.flatnonzero(labels == j)
            order = sorted(members, key=lambda i: (float(np.linalg.norm(x[i] - centers[j])), ids[i]))
            picks.extend(ids[i] for i in order[:quota])
        self.video_ids = picks
        return picks

    def add_result(self, video_id, result):
        if video_id not in self.video_ids:
            raise ValueError('Only representative videos may contribute to this knowledge base')
        self.results[video_id] = result

    def mc_value_ranking(self, target_video_id):
        totals, counts = {}, {}
        target = self.features[target_video_id]
        for vid, result in self.results.items():
            if vid == target_video_id:
                continue
            scores = {}
            for trial in result.history:
                if set(trial.scores) == set(result.train_ids):
                    scores[trial.config.mc] = max(scores.get(trial.config.mc, -math.inf), trial.quality)
            weight = similarity_weight(cosine(target, self.features[vid]))
            for mc, score in scores.items():
                totals[mc] = totals.get(mc, 0.0) + weight * score
                counts[mc] = counts.get(mc, 0) + 1
        return {mc: totals[mc] / counts[mc] for mc in totals}

    def warm_config(self, target_video_id, mc):
        neighbors = sorted((v for v in self.results if v != target_video_id),
                           key=lambda v: cosine(self.features[target_video_id], self.features[v]), reverse=True)
        for vid in neighbors:
            result = self.results[vid]
            trials = [t for t in result.history if t.config.mc == tuple(mc)
                      and set(t.scores) == set(result.train_ids)]
            if trials:
                return max(trials, key=lambda t: (t.quality, -t.ttft)).config
        return None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'schema_version': 1, 'context': self.context,
                                   'features': {v: f.tolist() for v, f in self.features.items()},
                                   'video_ids': self.video_ids, 'n_clusters': self.n_clusters,
                                   'similarity_threshold': self.similarity_threshold,
                                   'results': {v: r.to_dict() for v, r in self.results.items()}}, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path, context):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or data['context'] != context:
            raise ValueError('Knowledge-base model/dataset/SLO/seed/config context mismatch')
        kb = cls(context, data['features'])
        kb.video_ids, kb.n_clusters = data['video_ids'], data['n_clusters']
        kb.similarity_threshold = data['similarity_threshold']
        kb.results = {v: TuningResult.from_dict(r) for v, r in data['results'].items()}
        return kb
