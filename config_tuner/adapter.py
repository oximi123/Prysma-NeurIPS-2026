"""Equation (4): query filtering, configuration scoring and weighted ensemble."""
from __future__ import annotations
from statistics import mean
import numpy as np
from config_tuner.knowledgebase import cosine, similarity_weight


class OnlineSemanticAdapter:
    def __init__(self, questions, tuning_result, embedder, similarity_threshold=.6):
        # IDs, not question text, identify observations (duplicate question strings are valid).
        self.questions = {q: questions[q] for q in tuning_result.train_ids}
        self.result = tuning_result
        self.embedder = embedder
        self.threshold = float(similarity_threshold)
        self.ids = list(self.questions)
        self.embeddings = np.asarray(embedder([self.questions[q] for q in self.ids]), dtype=float)
        self.configs = {}
        for record in tuning_result.history:
            entry = self.configs.setdefault(record.config.key, {'config': record.config, 'scores': {}, 'latencies': []})
            for observation in record.observations:
                if observation.question_id in self.questions:
                    entry['scores'].setdefault(observation.question_id, []).append(observation.quality(record.slo))
                    entry['latencies'].append(observation.ttft)
        # Repeated measurements are averaged, not selected using final evaluation answers.
        for entry in self.configs.values():
            entry['scores'] = {q: mean(v) for q, v in entry['scores'].items()}
            entry['ttft'] = mean(entry['latencies'])

    def adapt(self, question, exclude_id=None):
        vector = np.asarray(self.embedder([question])[0], dtype=float)
        sims = {q: cosine(vector, emb) for q, emb in zip(self.ids, self.embeddings)
                if q != exclude_id and cosine(vector, emb) > self.threshold}
        best, best_value, best_ttft = None, -float('inf'), float('inf')
        for entry in self.configs.values():
            matched = [q for q in sims if q in entry['scores']]
            if not matched:
                continue
            value = sum(similarity_weight(sims[q]) * entry['scores'][q] for q in matched) / len(matched)
            if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and entry['ttft'] < best_ttft):
                best, best_value, best_ttft = entry['config'], value, entry['ttft']
        return best if best is not None else self.result.fallback


def select_similarity_threshold(kb_results, questions_by_video, embedder, thresholds=None):
    """Leave-one-tuning-query-out grid search, using KB observations only.

    Validation compares configurations with a measured score for the validation query;
    absent entries are excluded rather than imputed or evaluated on held-out labels.
    """
    thresholds = list(thresholds if thresholds is not None else np.round(np.arange(0, 1.01, .05), 2))
    totals = {float(t): [] for t in thresholds}
    for video_id, result in kb_results.items():
        questions = questions_by_video[video_id]
        measured = {}
        for record in result.history:
            for q, score in record.scores.items():
                measured.setdefault((q, record.config.key), []).append(score)
        for threshold in totals:
            adapter = OnlineSemanticAdapter(questions, result, embedder, threshold)
            for q in result.train_ids:
                # Restrict candidates to configs measured on q, while excluding q from voting.
                candidates = adapter.configs
                adapter.configs = {key: value for key, value in candidates.items() if (q, key) in measured}
                selected = adapter.adapt(questions[q], exclude_id=q)
                adapter.configs = candidates
                value = measured.get((q, selected.key))
                if value is not None:
                    totals[threshold].append(mean(value))
    if not any(totals.values()):
        raise ValueError('No knowledge-base tuning observations for threshold calibration')
    return max(totals, key=lambda t: (mean(totals[t]) if totals[t] else -1, -t))
