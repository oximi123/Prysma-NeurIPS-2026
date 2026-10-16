"""Calibrated token budgets and query-scoped monotone dominance pruning (Appendix G)."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
from config_tuner.records import Configuration


@dataclass(frozen=True)
class KnobSpec:
    modality: str
    name: str
    direction: str


DEFAULT_MONOTONE = [KnobSpec('visiontoken', 'num_frame', 'increasing'),
                    KnobSpec('object', 'fps', 'increasing'),
                    KnobSpec('object', 'confidence_threshold', 'decreasing')]


@dataclass
class TokenLatencyModel:
    model: str
    serving_context: dict
    token_counts: list[int]
    ttfts: list[float]
    vt_tokens_per_frame: float
    overheads: dict[str, float]
    monotone_knobs: list[KnobSpec]

    def __post_init__(self):
        if len(self.token_counts) < 2 or len(self.token_counts) != len(self.ttfts):
            raise ValueError('A measured profile needs at least two token/TTFT samples')
        if (any(b <= a for a, b in zip(self.token_counts, self.token_counts[1:]))
                or any(b < a for a, b in zip(self.ttfts, self.ttfts[1:]))
                or any(not math.isfinite(v) or v < 0 for v in self.ttfts)
                or self.vt_tokens_per_frame <= 0):
            raise ValueError('Profile must have increasing token counts and monotone finite TTFTs')
        if any(not math.isfinite(v) or v < 0 for v in self.overheads.values()):
            raise ValueError('Retrieval overhead must be measured, finite and nonnegative')
        if any(s.direction not in ('increasing', 'decreasing') for s in self.monotone_knobs):
            raise ValueError('Invalid monotonic direction')

    def overhead(self, config):
        key = '+'.join(config.mc) + '|' + config.retrieval_model
        if key not in self.overheads:
            raise ValueError(f'Missing measured retrieval overhead: {key}')
        return self.overheads[key]

    def tokens_to_ttft(self, tokens):
        if tokens > self.token_counts[-1]:
            return math.inf  # outside calibrated range, never extrapolate a feasible prediction
        return float(np.interp(tokens, self.token_counts, self.ttfts))

    def max_tokens_under_slo(self, slo, config):
        allowance = slo - self.overhead(config)
        if allowance < self.ttfts[0]:
            return -1.0
        if allowance >= self.ttfts[-1]:
            # Monotonicity alone cannot identify an upper bound beyond the last
            # measured prompt length. Do not declare unmeasured lengths infeasible.
            return math.inf
        # Invert a monotone piecewise-linear profile, including flat segments.
        feasible = [(n, t) for n, t in zip(self.token_counts, self.ttfts) if t <= allowance]
        n, t = feasible[-1]
        i = self.token_counts.index(n)
        if i == len(self.ttfts) - 1:
            return float(n)
        return n + (allowance - t) * (self.token_counts[i + 1] - n) / (self.ttfts[i + 1] - t)

    def to_dict(self):
        return {'schema_version': 1, 'model': self.model, 'serving_context': self.serving_context,
                'token_counts': self.token_counts, 'ttfts': self.ttfts,
                'vt_tokens_per_frame': self.vt_tokens_per_frame, 'overheads': self.overheads,
                'monotone_knobs': [vars(s) for s in self.monotone_knobs]}

    @classmethod
    def load(cls, path, model, serving_context):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if d.get('schema_version') != 1 or d['model'] != model or d['serving_context'] != serving_context:
            raise ValueError('Latency profile does not match model/hardware/serving conditions')
        d.pop('schema_version')
        d['monotone_knobs'] = [KnobSpec(**s) for s in d['monotone_knobs']]
        return cls(**d)


def profile_monotonic_knobs(measure_tokens, base_config, knob_space):
    """Check ALL adjacent sweep values with other knobs fixed, not only endpoints."""
    import copy
    specs = []
    for (modality, knob), values in knob_space.items():
        counts = []
        for value in sorted(values):
            candidate = copy.deepcopy(base_config)
            candidate.modalities[modality][knob] = value
            counts.append(float(measure_tokens(candidate)))
        if len(counts) < 2 or not all(math.isfinite(n) for n in counts) or max(counts) == min(counts):
            continue
        if all(b >= a for a, b in zip(counts, counts[1:])):
            specs.append(KnobSpec(modality, knob, 'increasing'))
        elif all(b <= a for a, b in zip(counts, counts[1:])):
            specs.append(KnobSpec(modality, knob, 'decreasing'))
    return specs


class LatencyAwareSpacePruner:
    """One instance per video/context, shared by both stages; bans are per query.

    Partial-query violations must not remove configurations useful to other queries.
    Token observations are reusable only with matched nonmonotone/retrieval knobs.
    """
    def __init__(self, slo_seconds, token_latency_model, monotone_knobs=None):
        self.slo = float(slo_seconds)
        self.profile = token_latency_model
        specs = monotone_knobs if monotone_knobs is not None else self.profile.monotone_knobs
        self.monotone = {(s.modality, s.name): s.direction for s in specs}
        self.banned = []
        self.token_observations = []
        self.pruned_count = 0

    def dominates(self, lower, upper):
        if not set(lower.mc).issubset(upper.mc):
            return False
        if (lower.retrieval_model, lower.retrieval_threshold) != (upper.retrieval_model, upper.retrieval_threshold):
            return False
        for m, knobs in lower.modalities.items():
            if set(knobs) != set(upper.modalities[m]):
                return False
            for k, value in knobs.items():
                other = upper.modalities[m][k]
                direction = self.monotone.get((m, k))
                if direction == 'increasing' and other < value:
                    return False
                if direction == 'decreasing' and other > value:
                    return False
                if direction is None and other != value:
                    return False
        return True

    def token_lower_bound(self, config, question_id):
        lower = float(config.modalities.get('visiontoken', {}).get('num_frame', 0)) * self.profile.vt_tokens_per_frame
        # Prompt-token observations include query-specific retrieval: never transfer across queries.
        for old, observation in self.token_observations:
            if observation.question_id == question_id and self.dominates(old, config):
                lower = max(lower, observation.token_count)
        return lower

    def pre_prune(self, config, question_ids):
        budget = self.profile.max_tokens_under_slo(self.slo, config)
        return all(self.token_lower_bound(config, q) > budget for q in question_ids)

    def should_prune(self, config, question_ids):
        if not question_ids:
            return True
        # Reject only when every query is known infeasible. Unknown information is not a zero score.
        result = all(self.token_lower_bound(config, q) > self.profile.max_tokens_under_slo(self.slo, config)
                     or any(q == old_q and self.dominates(old, config)
                            and self.profile.overhead(config) >= self.profile.overhead(old)
                            for old, old_q in self.banned) for q in question_ids)
        self.pruned_count += int(result)
        return result

    def observe(self, record):
        for observation in record.observations:
            self.token_observations.append((record.config, observation))
            if observation.ttft > self.slo:
                self.banned.append((record.config, observation.question_id))
