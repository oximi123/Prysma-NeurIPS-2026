"""Subset-only objective. Evaluation IDs are explicit and never positional."""
from config_tuner.records import TrialRecord


class Objective:
    def __init__(self, backend, train_ids, slo, pruner=None):
        self.backend = backend
        self.train_ids = list(map(str, train_ids))
        if not self.train_ids or len(set(self.train_ids)) != len(self.train_ids):
            raise ValueError('Tuning IDs must be nonempty and unique')
        self.slo = float(slo)
        self.pruner = pruner
        self.history = []

    def evaluate_config(self, config, question_ids=None, stage='video'):
        ids = list(self.train_ids if question_ids is None else question_ids)
        if not ids or not set(ids).issubset(self.train_ids):
            raise ValueError('Only nonempty subsets of tuning queries may be evaluated')
        if self.pruner is not None and self.pruner.should_prune(config, ids):
            return None
        observations = self.backend.evaluate(config, ids)
        actual = [o.question_id for o in observations]
        if len(actual) != len(ids) or set(actual) != set(ids):
            raise ValueError('Backend must return exactly the requested query IDs')
        record = TrialRecord(config, observations, stage, len(ids) / len(self.train_ids), self.slo)
        self.history.append(record)
        if self.pruner is not None:
            self.pruner.observe(record)
        return record

    def best_scores(self):
        best = {q: 0.0 for q in self.train_ids}
        for trial in self.history:
            for q, score in trial.scores.items():
                best[q] = max(best[q], score)
        return best

    def poor_queries(self, good_cutoff=4.5):
        scores = self.best_scores()
        average = sum(scores.values()) / len(scores)
        return [q for q in self.train_ids if scores[q] < average and scores[q] < good_cutoff]
