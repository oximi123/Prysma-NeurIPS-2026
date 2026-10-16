"""Paper Algorithms 1/2: restricted TPE, dynamic MC budgets, EMMR, subset GA."""
from __future__ import annotations
import math
import numpy as np
from config_tuner.baselines import GeneticSearch
from config_tuner.records import TuningResult, best_full_config


def allocate_budget(budget, values):
    if not values:
        return {}
    keys = list(values)
    weights = np.asarray([max(0., values[k]) for k in keys])
    if weights.sum() <= 0:
        weights[:] = 1
    raw = weights / weights.sum() * int(budget)
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[:int(budget) - int(counts.sum())]:
        counts[i] += 1
    return dict(zip(keys, map(int, counts)))


class TPESearch:
    def __init__(self, space, seed):
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError('Prysma requires Optuna; install requirements-core.txt') from exc
        self.optuna, self.space, self.seed = optuna, space, seed
        self.studies = {}
        self.pending = None

    def _study(self, mc):
        if mc not in self.studies:
            self.studies[mc] = self.optuna.create_study(direction='maximize',
                sampler=self.optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=3))
        return self.studies[mc]

    def enqueue(self, config, mc):
        self._study(mc).enqueue_trial(self.space.parameters(config, all_mc=mc is None))

    def propose(self, mc):
        study = self._study(mc)
        trial = study.ask()
        config = self.space.suggest(trial, mc)
        self.pending = (study, trial)
        return config

    def observe(self, record):
        study, trial = self.pending
        if record is None:
            study.tell(trial, state=self.optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, record.quality)


class EMMRMonitor:
    def __init__(self, space, seed):
        try:
            import optuna
            from optuna.terminator import EMMREvaluator, MedianErrorEvaluator, Terminator
        except ImportError as exc:
            raise RuntimeError('Install Optuna with public EMMR/MedianError/Terminator support') from exc
        self.space = space
        self.study = optuna.create_study(direction='maximize')
        evaluator = EMMREvaluator(seed=seed, min_n_trials=3)
        self.terminator = Terminator(improvement_evaluator=evaluator,
            error_evaluator=MedianErrorEvaluator(evaluator, warm_up_trials=3, n_initial_trials=5), min_n_trials=8)

    def update(self, record):
        # EMMR's GP requires a fixed numerical search space. Feed it the same
        # observed quality in a one-hot/numeric encoding, never future outcomes.
        values = {f'x{i}': float(v) for i, v in enumerate(self.space.encode(record.config))}
        self.study.enqueue_trial(values)
        trial = self.study.ask()
        for key in values:
            trial.suggest_float(key, 0., 1.)
        self.study.tell(trial, record.quality)
        return bool(self.terminator.should_terminate(self.study))


def estimate_mc_ei(space, mc, history, rng, probes=8):
    """Monte Carlo mean l(x)/g(x), using good/bad Parzen densities.

    This explicit KDE avoids depending on private Optuna estimator internals.
    Fixed categorical choices have one-hot coordinates in the density space.
    """
    records = [t for t in history if t.stage == 'video' and t.config.mc == mc]
    if len(records) < 3:
        return 1.0
    ranked = sorted(records, key=lambda t: t.quality, reverse=True)
    cut = max(1, math.ceil(len(ranked) * .2))
    good = np.stack([space.encode(t.config) for t in ranked[:cut]])
    bad = np.stack([space.encode(t.config) for t in ranked[cut:]])
    samples = np.stack([space.encode(space.random(rng, mc)) for _ in range(probes)])
    def log_density(data):
        bandwidth = max(.1, len(data) ** (-1 / (data.shape[1] + 4)))
        logs = -((samples[:, None] - data[None, :]) ** 2).sum(axis=2) / (2 * bandwidth ** 2)
        logs -= data.shape[1] * math.log(bandwidth)
        largest = logs.max(axis=1)
        return largest + np.log(np.exp(logs - largest[:, None]).mean(axis=1))
    return float(np.exp(np.clip(log_density(good) - log_density(bad), -30, 30)).mean())


class TwostageTuner:
    def __init__(self, space, seed=42, knowledge_base=None, target_video_id=None,
                 tpe=None, monitor=None, population_size=16, lhs_samples=3):
        self.space, self.seed = space, seed
        self.kb, self.video_id = knowledge_base, target_video_id
        self.rng = np.random.default_rng(seed)
        self.tpe = tpe if tpe is not None else TPESearch(space, seed)
        self.monitor = monitor if monitor is not None else EMMRMonitor(space, seed)
        self.population_size, self.lhs_samples = population_size, lhs_samples
        self.allocation_events = []

    def tune(self, objective, total_budget=30):
        if total_budget < 1:
            raise ValueError('At least one full-set trial is required')
        values = self.kb.mc_value_ranking(self.video_id) if self.kb is not None else {}
        legal = set(self.space.combinations())
        values = {mc: v for mc, v in values.items() if mc in legal}
        # KB representatives skip MC scoring and tune the entire hybrid space.
        budgets = allocate_budget(total_budget, values) if values else {None: int(total_budget)}
        initialized, last_update_best = set(), {}
        used, rejected, switch_reason = 0., 0, 'budget_exhausted'
        enter_stage2 = False
        current = None
        while used + 1 <= total_budget + 1e-12:
            active = [mc for mc, b in budgets.items() if b > 0]
            if not active:
                break
            if current not in active:
                current = max(active, key=lambda mc: budgets[mc])
                last_update_best[current] = max((t.quality for t in objective.history), default=-math.inf)
            if current not in initialized:
                if current is not None:
                    warm = self.kb.warm_config(self.video_id, current)
                    initialization = []
                    if warm is not None:
                        initialization.append(warm)
                    initialization.extend(self.space.lhs(self.rng, min(self.lhs_samples, budgets[current]), current))
                    for c in initialization[:budgets[current]]:
                        self.tpe.enqueue(c, current)
                initialized.add(current)
            candidate = self.tpe.propose(current)
            if current is not None and candidate.mc != current:
                raise ValueError('TPE proposal escaped its selected modality combination')
            record = objective.evaluate_config(candidate, stage='video')
            self.tpe.observe(record)
            if record is None:
                rejected += 1
                if rejected >= 128:
                    budgets[current] = 0
                    rejected = 0
                continue
            rejected = 0
            used += record.budget_cost
            budgets[current] -= 1
            if self.monitor.update(record):
                switch_reason, enter_stage2 = 'emmr', True
                break
            if current is None:
                continue
            global_best = max(t.quality for t in objective.history)
            remaining = [mc for mc, b in budgets.items() if mc != current and b > 0]
            if not remaining:
                continue
            ei = {mc: estimate_mc_ei(self.space, mc, objective.history, self.rng)
                  for mc in remaining + [current]}
            winner = max(remaining, key=lambda mc: ei[mc])
            explored = {m for t in objective.history for m in t.config.mc}
            if budgets[current] > 0 and ei[winner] > ei[current] and set(winner).issubset(explored):
                moved = budgets[current]
                budgets[winner] += moved
                budgets[current] = 0
                self.allocation_events.append({'from': current, 'to': winner, 'budget': moved, 'reason': 'EI'})
            elif budgets[current] == 0 and global_best > last_update_best.get(current, -math.inf):
                donor = min(remaining, key=lambda mc: values[mc])
                moved = budgets[donor]
                budgets[current] += moved
                budgets[donor] = 0
                last_update_best[current] = global_best
                self.allocation_events.append({'from': donor, 'to': current, 'budget': moved, 'reason': 'improvement'})
        if enter_stage2:
            ga = GeneticSearch(self.space, self.rng, self.population_size)
            rejected = 0
            while used < total_budget - 1e-12:
                subset = objective.poor_queries()
                if not subset:
                    switch_reason += ':no_poor_queries'
                    break
                cost = len(subset) / len(objective.train_ids)
                if used + cost > total_budget + 1e-12:
                    break
                candidate = ga.propose(objective.history, subset)
                record = objective.evaluate_config(candidate, subset, stage='query')
                if record is None:
                    rejected += 1
                    if rejected >= 128:
                        switch_reason += ':space_exhausted'
                        break
                    continue
                rejected = 0
                used += record.budget_cost
        fallback = best_full_config(objective.history, objective.train_ids)
        return TuningResult(objective.train_ids, objective.history, fallback, switch_reason)
