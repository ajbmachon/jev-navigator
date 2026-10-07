"""Search in groups of 16 through a caller-supplied oracle, with explicit missing observations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .graph import CodeGraph, random_walk


@dataclass(frozen=True)
class Observation:
    probability: float
    confirmed: bool

    def __post_init__(self):
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Oracle probability must be between zero and one")


class Oracle(Protocol):
    def judge(self, candidates: tuple[str, ...]) -> Mapping[str, Observation]: ...


@dataclass(frozen=True)
class ActivePolicy:
    max_requests: int = 48
    propagation_weight: float = 1.0
    min_expected_gain: float = 0.1
    min_requests: int = 1
    prior_positive: float = 1.0
    prior_negative: float = 15.0
    propagation_iterations: int = 2

    def __post_init__(self):
        if self.max_requests < 1 or not 0 <= self.min_requests <= self.max_requests:
            raise ValueError("Invalid active-search request limits")
        if min(self.prior_positive, self.prior_negative) <= 0:
            raise ValueError("Marginal-value beta prior parameters must be positive")
        if self.propagation_weight < 0 or self.min_expected_gain < 0:
            raise ValueError("Active-search weights must be nonnegative")
        if self.propagation_iterations < 1:
            raise ValueError("Propagation must take at least one graph step")


@dataclass(frozen=True)
class ActiveResult:
    batches: tuple[tuple[str, ...], ...]
    observations: Mapping[str, Observation]
    unjudged: tuple[str, ...]
    pending: tuple[str, ...]
    stopped_by: str
    expected_gain: float


DEFAULT_ACTIVE_POLICY = ActivePolicy()


def active_search(
    candidates: Sequence[str],
    scores: Mapping[str, float],
    graph: CodeGraph,
    oracle: Oracle,
    *,
    policy: ActivePolicy = DEFAULT_ACTIVE_POLICY,
) -> ActiveResult:
    """Judge 16, propagate confirmed relevance, re-rank, then estimate the next batch's gain.

    The marginal rule uses posterior observed yield times the next batch's mean normalized scent,
    plus its propagated relevance, bounded by one per candidate. It is a declared heuristic, not
    calibrated Jev confidence. Missing answers stay unjudged and never count as negatives. A caller
    can disable marginal stopping with min_expected_gain=0 and fit the rule on independent cases.
    """
    if len(set(candidates)) != len(candidates):
        raise ValueError("Active search needs distinct candidate identities")
    pending = list(candidates)
    observations: dict[str, Observation] = {}
    batches = []
    unjudged = []
    maximum = max((scores[id] for id in candidates), default=0) or 1
    normalized = {id: max(0.0, scores[id] / maximum) for id in candidates}
    expected = 0.0
    stopped = "exhausted"
    while pending:
        if len(batches) >= policy.max_requests:
            stopped = "request guard"
            break
        confirmed = {
            id: observation.probability for id, observation in observations.items() if observation.confirmed
        }
        propagation = (
            random_walk(graph, confirmed, max_iterations=policy.propagation_iterations) if confirmed else {}
        )
        propagation_max = max((propagation.get(id, 0) for id in pending), default=0) or 1
        boost = {id: propagation.get(id, 0) / propagation_max for id in pending}
        pending.sort(key=lambda id: -(normalized[id] + policy.propagation_weight * boost[id]))
        batch = tuple(pending[:16])
        yield_rate = (policy.prior_positive + sum(item.probability for item in observations.values())) / (
            policy.prior_positive + policy.prior_negative + len(observations)
        )
        expected = sum(
            min(1, yield_rate * normalized[id] + policy.propagation_weight * boost[id]) for id in batch
        )
        if len(batches) >= policy.min_requests and expected < policy.min_expected_gain:
            stopped = "marginal value"
            break
        answers = dict(oracle.judge(batch))
        if not answers.keys() <= set(batch):
            raise ValueError("Oracle returned observations for candidates outside its batch")
        batches.append(batch)
        observations.update(answers)
        unjudged.extend(id for id in batch if id not in answers)
        del pending[: len(batch)]
        if len(answers) != len(batch):
            stopped = "missing oracle answers"
            break
    return ActiveResult(tuple(batches), observations, tuple(unjudged), tuple(pending), stopped, expected)
