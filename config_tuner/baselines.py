"""VQA adaptations of Fig. 5 baselines, not replicas of upstream serving systems.

SCOOT: RBF GP expected improvement with bootstrap classification-forest feasibility.
LLMConf: steady-state GA, with Chebyshev-distance diversity in survivor selection.
OPPerTune: categorical exponential-weights bandits and numeric two-point feedback.
All receive the same subset-only objective and per-request SLO-penalized quality.
"""
from __future__ import annotations
import copy
import math
from statistics import NormalDist
import numpy as np
from config_tuner.records import Configuration, TuningResult, best_full_config


class GeneticSearch:
    def __init__(self, space, rng, population_size=16, mutation=.15):
        self.space, self.rng = space, rng
        self.population_size, self.mutation = population_size, mutation

    def propose(self, history, subset):
        # Re-score history on the CURRENT subset; no zero-filled missing entries.
        ranked = []
        for record in history:
            scores = record.scores
            if set(subset).issubset(scores):
                ranked.append((sum(scores[q] for q in subset) / len(subset), record.config))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        population, seen = [], set()
        for score, config in ranked:
            if config.key not in seen:
                population.append((score, config))
                seen.add(config.key)
        population = population[:self.population_size]
        if len(population) < 2:
            return self.space.random(self.rng)
        def tournament():
            choices = self.rng.choice(len(population), min(3, len(population)), replace=False)
            return max((population[i] for i in choices), key=lambda item: item[0])[1]
        return self.space.crossover(tournament(), tournament(), self.rng, self.mutation)


class _Tree:
    def __init__(self, x, y, rng, depth=0):
        self.probability = float(y.mean())
        self.left = self.right = None
        if depth >= 6 or len(y) < 4 or np.all(y == y[0]):
            return
        best = None
        for feature in rng.choice(x.shape[1], max(1, int(math.sqrt(x.shape[1]))), replace=False):
            values = np.unique(x[:, feature])
            if len(values) < 2:
                continue
            for threshold in rng.choice((values[:-1] + values[1:]) / 2, min(8, len(values) - 1), replace=False):
                mask = x[:, feature] <= threshold
                if not mask.any() or mask.all():
                    continue
                impurity = sum(len(part) * part.mean() * (1 - part.mean()) for part in (y[mask], y[~mask]))
                if best is None or impurity < best[0]:
                    best = (impurity, int(feature), float(threshold), mask)
        if best is not None:
            _, self.feature, self.threshold, mask = best
            self.left = _Tree(x[mask], y[mask], rng, depth + 1)
            self.right = _Tree(x[~mask], y[~mask], rng, depth + 1)

    def predict(self, x):
        if self.left is None:
            return self.probability
        return (self.left if x[self.feature] <= self.threshold else self.right).predict(x)


class SCOOTSearch:
    def __init__(self, space, rng, startup=5, candidates=128, forest_size=24):
        self.space, self.rng = space, rng
        self.startup, self.candidates, self.forest_size = startup, candidates, forest_size

    def propose(self, history, subset):
        if len(history) < self.startup:
            return self.space.random(self.rng)
        x = np.stack([self.space.encode(t.config) for t in history])
        y = np.asarray([t.quality for t in history])
        # Hidden feasibility models mean tuning latency, while the objective still
        # applies the paper's penalty independently to every violating request.
        feasible = np.asarray([float(t.ttft <= t.slo) for t in history])
        trees = []
        for _ in range(self.forest_size):
            indices = self.rng.integers(len(x), size=len(x))
            trees.append(_Tree(x[indices], feasible[indices], self.rng))
        configs = [self.space.random(self.rng) for _ in range(self.candidates)]
        z = np.stack([self.space.encode(c) for c in configs])
        probability = np.asarray([np.mean([tree.predict(row) for tree in trees]) for row in z])
        # RBF GP posterior, with jitter for repeated configurations.
        kernel = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2).sum(axis=2) / 2)
        k = kernel(x, x) + np.eye(len(x)) * 1e-5
        center, scale = float(y.mean()), max(float(y.std()), 1e-6)
        cholesky = np.linalg.cholesky(k)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, (y - center) / scale))
        cross = kernel(x, z)
        mu = center + scale * cross.T @ alpha
        solved = np.linalg.solve(cholesky, cross)
        sigma = scale * np.sqrt(np.maximum(1e-12, 1 - (solved ** 2).sum(axis=0)))
        improvement = mu - float(y.max())
        normal = NormalDist()
        standardized = improvement / sigma
        ei = np.asarray([d * normal.cdf(v) + s * math.exp(-v * v / 2) / math.sqrt(2 * math.pi)
                         for d, s, v in zip(improvement, sigma, standardized)])
        allowed = probability >= .5
        if not allowed.any():
            return configs[int(probability.argmax())]
        acquisition = np.where(allowed, ei * probability, -math.inf)
        return configs[int(acquisition.argmax())]


class LLMConfSearch(GeneticSearch):
    def propose(self, history, subset):
        if len(history) < 5:
            return self.space.random(self.rng)
        # Quality ranks candidates; Chebyshev distance retains distinct parents
        # rather than substituting an NSGA sampler for the entire baseline.
        ranked = sorted(history, key=lambda t: t.quality, reverse=True)
        selected = [ranked.pop(0)]
        while ranked and len(selected) < self.population_size:
            def priority(t):
                distance = min(float(np.max(np.abs(self.space.encode(t.config) - self.space.encode(s.config))))
                               for s in selected)
                return (t.quality + .05 * distance, distance)
            winner = max(ranked, key=priority)
            ranked.remove(winner)
            selected.append(winner)
        return super().propose(selected, subset)


class OPPerTuneSearch:
    def __init__(self, space, rng, learning_rate=.1, perturbation=.1, exploration=.1):
        self.space, self.rng = space, rng
        self.lr, self.delta, self.exploration = learning_rate, perturbation, exploration
        self.categorical = {}
        self.numeric = {}
        self.pending = None
        self.first_reward = None
        self.last = None
        choices = {"mc": space.combinations(), "retrieval_model": space.retrieval_models}
        choices.update({f'{m}.{k}': values for m, k, values in space.fields})
        choices['retrieval_threshold'] = space.retrieval_thresholds
        for name, values in choices.items():
            if name not in ('mc', 'retrieval_model') and all(isinstance(v, (int, float)) for v in values) and len(values) > 1:
                self.numeric[name] = {'values': values, 'center': .5}
            else:
                self.categorical[name] = {'values': values, 'weights': np.zeros(len(values))}

    def propose(self, history, subset):
        if self.pending is None:
            categorical, arms, probabilities = {}, {}, {}
            for name, state in self.categorical.items():
                exp = np.exp(state['weights'] - state['weights'].max())
                p = (1 - self.exploration) * exp / exp.sum() + self.exploration / len(exp)
                i = int(self.rng.choice(len(p), p=p))
                categorical[name], arms[name], probabilities[name] = state['values'][i], i, float(p[i])
            direction = {name: float(self.rng.choice([-1, 1])) for name in self.numeric}
            self.pending = (categorical, arms, probabilities, direction)
            self.first_reward = None
        categorical, arms, probabilities, direction = self.pending
        sign = 1 if self.first_reward is None else -1
        values = dict(categorical)
        for name, state in self.numeric.items():
            position = float(np.clip(state['center'] + sign * self.delta * direction[name], 0, 1))
            grid = state['values']
            target = min(grid) + position * (max(grid) - min(grid))
            values[name] = min(grid, key=lambda v: abs(v - target))
        mods = {m: {k: values[f'{m}.{k}'] for k in self.space.modalities[m]} for m in values['mc']}
        config = self.space.normalize(Configuration(mods, values['retrieval_model'], float(values['retrieval_threshold'])))
        self.last = config
        return config

    def observe(self, record):
        reward = record.quality / 5.0
        if self.first_reward is None:
            self.first_reward = reward
            return
        _, arms, probabilities, direction = self.pending
        gradient = (self.first_reward - reward) / (2 * self.delta)
        active = {'mc'} | {f'{m}.{k}' for m, knobs in self.last.modalities.items() for k in knobs}
        if set(self.last.mc).intersection(('object', 'asr', 'ocr')):
            active.update(('retrieval_model', 'retrieval_threshold'))
        for name, state in self.numeric.items():
            if name in active:
                state['center'] = float(np.clip(state['center'] + self.lr * gradient * direction[name], 0, 1))
        combined = (self.first_reward + reward) / 2
        for name, arm in arms.items():
            if name in active:
                self.categorical[name]['weights'][arm] += self.lr * combined / max(probabilities[name], 1e-12)
        self.pending = None


def tune_baseline(method, objective, space, budget, seed):
    rng = np.random.default_rng(seed)
    if method == 'VT':
        for frame in space.modalities['visiontoken']['num_frame'][:int(budget)]:
            config = Configuration({'visiontoken': {'num_frame': frame, 'sampling_method': 'uniform'}}, 'None', 0.0)
            objective.evaluate_config(config, stage='VT')
        feasible = [t for t in objective.history if all(o.ttft <= objective.slo for o in t.observations)]
        selected = max(feasible, key=lambda t: (t.quality, -t.ttft)) if feasible else min(objective.history, key=lambda t: t.ttft)
        return TuningResult(objective.train_ids, objective.history, selected.config)
    searches = {'Random': None, 'SCOOT': SCOOTSearch, 'LLMConf': LLMConfSearch, 'OPPerTune': OPPerTuneSearch}
    if method not in searches:
        raise ValueError(f'Unknown baseline {method}')
    search = searches[method](space, rng) if searches[method] else None
    for _ in range(int(budget)):
        config = search.propose(objective.history, objective.train_ids) if search else space.random(rng)
        record = objective.evaluate_config(config, stage=method)
        if record is None:
            raise ValueError('Prysma latency pruning must not be attached to baseline objectives')
        if hasattr(search, 'observe'):
            search.observe(record)
    return TuningResult(objective.train_ids, objective.history, best_full_config(objective.history, objective.train_ids))
