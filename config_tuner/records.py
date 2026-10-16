"""Model-independent records shared by tuning, adaptation and Fig. 5 evaluation."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from statistics import mean
from typing import Protocol


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class Configuration:
    modalities: dict[str, dict]
    retrieval_model: str = "None"
    retrieval_threshold: float = 0.3

    def __post_init__(self):
        if not self.modalities:
            raise ValueError("A configuration must contain at least one modality")
        if self.retrieval_model not in ("BERT", "LLM", "None"):
            raise ValueError("Unknown retrieval model")
        if not 0 <= self.retrieval_threshold <= 1:
            raise ValueError("retrieval_threshold must lie in [0, 1]")

    @property
    def mc(self):
        return tuple(sorted(self.modalities))

    @property
    def key(self):
        return hashlib.sha256(canonical(asdict(self)).encode()).hexdigest()[:24]

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass
class Observation:
    question_id: str
    answer: str
    raw_score: float
    ttft: float
    token_count: int = 0
    modality_tokens: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.question_id = str(self.question_id)
        if not math.isfinite(self.raw_score) or not 0 <= self.raw_score <= 5:
            raise ValueError("Judge score must be a finite number in [0, 5]")
        if not math.isfinite(self.ttft) or self.ttft < 0:
            raise ValueError("TTFT must be finite and nonnegative")
        if self.token_count < 0 or any(v < 0 for v in self.modality_tokens.values()):
            raise ValueError("Token counts must be nonnegative")

    def quality(self, slo):
        return self.raw_score if self.ttft <= slo else 0.0


@dataclass
class TrialRecord:
    config: Configuration
    observations: list[Observation]
    stage: str
    budget_cost: float
    slo: float

    @property
    def scores(self):
        return {o.question_id: o.quality(self.slo) for o in self.observations}

    @property
    def quality(self):
        return mean(self.scores.values()) if self.observations else 0.0

    @property
    def ttft(self):
        return mean(o.ttft for o in self.observations) if self.observations else math.inf

    def to_dict(self):
        result = asdict(self)
        result["scores"] = self.scores
        return result

    @classmethod
    def from_dict(cls, value):
        return cls(Configuration.from_dict(value["config"]),
                   [Observation(**o) for o in value["observations"]],
                   value["stage"], value["budget_cost"], value["slo"])


@dataclass
class TuningResult:
    train_ids: list[str]
    history: list[TrialRecord]
    fallback: Configuration
    switch_reason: str = "budget_exhausted"

    @property
    def budget_used(self):
        return sum(t.budget_cost for t in self.history)

    def to_dict(self):
        return {"schema_version": 1, "train_ids": self.train_ids,
                "history": [t.to_dict() for t in self.history],
                "fallback": asdict(self.fallback), "switch_reason": self.switch_reason}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported history schema; legacy pickle histories are not tuning inputs")
        return cls(value["train_ids"], [TrialRecord.from_dict(t) for t in value["history"]],
                   Configuration.from_dict(value["fallback"]), value["switch_reason"])


class EvaluationBackend(Protocol):
    def evaluate(self, config: Configuration, question_ids: list[str]) -> list[Observation]: ...


def best_full_config(history, train_ids):
    full = [t for t in history if set(t.scores) == set(train_ids)]
    if not full:
        raise RuntimeError("No full tuning-set trial was evaluated; no video-level fallback exists")
    return max(full, key=lambda t: (t.quality, -t.ttft)).config
