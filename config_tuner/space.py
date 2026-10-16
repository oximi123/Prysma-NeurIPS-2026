"""Finite hybrid space: modality masks, categorical extractors and numeric grids."""
from __future__ import annotations
import copy
import itertools
import numpy as np
from config_tuner.records import Configuration


class SearchSpace:
    def __init__(self, modalities, retrieval_models=("BERT", "LLM", "None"), retrieval_thresholds=None):
        self.modalities = copy.deepcopy(modalities)
        self.retrieval_models = list(retrieval_models)
        self.retrieval_thresholds = list(retrieval_thresholds if retrieval_thresholds is not None
                                         else np.round(np.arange(0, 1.01, .1), 2))
        if not modalities or any(not v for knobs in modalities.values() for v in knobs.values()):
            raise ValueError("Empty configuration space")
        self.fields = [(m, k, list(v)) for m, knobs in sorted(modalities.items()) for k, v in sorted(knobs.items())]

    def combinations(self):
        names = sorted(self.modalities)
        return [tuple(c) for n in range(1, len(names) + 1) for c in itertools.combinations(names, n)]

    def random(self, rng, mc=None):
        if mc is None:
            combos = self.combinations()
            mc = combos[int(rng.integers(len(combos)))]
        mods = {m: {k: copy.deepcopy(v[int(rng.integers(len(v)))]) for k, v in self.modalities[m].items()} for m in mc}
        return self.normalize(Configuration(mods, self.retrieval_models[int(rng.integers(len(self.retrieval_models)))],
                                            float(rng.choice(self.retrieval_thresholds))))

    def normalize(self, config):
        if config.modalities.get("caption", {}).get("model") == "llava":
            config.modalities["caption"]["frames_per_caption"] = 1
        if not set(config.modalities).intersection(("object", "ocr", "asr")):
            config.retrieval_model, config.retrieval_threshold = "None", 0.0
        return config

    def lhs(self, rng, count, mc):
        configs = [self.random(rng, mc) for _ in range(count)]
        for m, k, values in self.fields:
            if m not in mc:
                continue
            indices = np.floor((rng.permutation(count) + rng.random(count)) / count * len(values)).astype(int)
            for c, i in zip(configs, indices):
                c.modalities[m][k] = copy.deepcopy(values[int(i)])
        return [self.normalize(c) for c in configs]

    def suggest(self, trial, mc=None):
        if mc is None:
            mc = tuple(trial.suggest_categorical("mc", ["+".join(c) for c in self.combinations()]).split("+"))
        mods = {m: {k: trial.suggest_categorical(f"{m}.{k}", list(v))
                    for k, v in self.modalities[m].items()} for m in mc}
        if not set(mc).intersection(('object', 'ocr', 'asr')):
            return self.normalize(Configuration(mods, 'None', 0.0))
        retrieval = trial.suggest_categorical("retrieval_model", self.retrieval_models)
        threshold = trial.suggest_categorical("retrieval_threshold", self.retrieval_thresholds)
        return self.normalize(Configuration(mods, retrieval, float(threshold)))

    def parameters(self, config, all_mc=False):
        params = {f"{m}.{k}": v for m, knobs in config.modalities.items() for k, v in knobs.items()}
        if set(config.mc).intersection(('object', 'ocr', 'asr')):
            params.update(retrieval_model=config.retrieval_model, retrieval_threshold=config.retrieval_threshold)
        if all_mc:
            params["mc"] = "+".join(config.mc)
        return params

    def encode(self, config):
        result = [float(m in config.modalities) for m in sorted(self.modalities)]
        for m, k, values in self.fields:
            value = config.modalities.get(m, {}).get(k)
            if all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in values):
                result.append(0.0 if value is None else (float(value) - min(values)) / max(1e-12, max(values) - min(values)))
            else:
                result.extend(float(value == v) for v in values)
        result.extend(float(config.retrieval_model == v) for v in self.retrieval_models)
        result.append(config.retrieval_threshold)
        return np.asarray(result, dtype=float)

    def crossover(self, a, b, rng, mutation=.15):
        mc = tuple(m for m in self.modalities if m in (a if rng.random() < .5 else b).modalities)
        if not mc:
            mc = (str(rng.choice(list(self.modalities))),)
        child = self.random(rng, mc)
        for m in mc:
            for k in child.modalities[m]:
                parent = a if rng.random() < .5 else b
                if m in parent.modalities and rng.random() >= mutation:
                    child.modalities[m][k] = parent.modalities[m][k]
        parent = a if rng.random() < .5 else b
        if rng.random() >= mutation:
            child.retrieval_model, child.retrieval_threshold = parent.retrieval_model, parent.retrieval_threshold
        if rng.random() < mutation:
            m = str(rng.choice(list(self.modalities)))
            if m in child.modalities and len(child.modalities) > 1:
                child.modalities.pop(m)
            elif m not in child.modalities:
                child.modalities[m] = self.random(rng, (m,)).modalities[m]
        return self.normalize(child)


def default_space(dataset):
    frames = [4, 8, 16, 32, 64, 128]
    thresholds = [round(i / 10, 1) for i in range(1, 10)]
    mods = {
        "visiontoken": {"num_frame": frames, "sampling_method": ["uniform", "keyframe"]},
        "caption": {"seconds_per_caption": [1, 2, 4, 8, 16, 30], "frames_per_caption": [1, 2, 4], "model": ["llava", "lavila"]},
        "ocr": {"num_frame": frames, "sampling_method": ["uniform", "keyframe"], "text_threshold": thresholds},
        "object": {"fps": [1, 2, 4, 8, 16, 30], "confidence_threshold": thresholds, "model": ["APELD", "RTDETR"]},
        "asr": {"model": ["whisper-small", "whisper-medium", "whisper-large"],
                "chunk_length": [3, 6, 9, 15, 30], "sampling_rate": [8000, 16000]},
    }
    if dataset == "MSVD-QA":
        mods.pop("asr")
    return SearchSpace(mods)
