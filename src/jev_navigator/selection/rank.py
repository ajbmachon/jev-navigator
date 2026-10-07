"""Few explicit weights over normalized independent signals; stable candidate-order ties."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .graph import CodeGraph, random_walk
from .scent import ScentIndex


@dataclass(frozen=True)
class RankFeatures:
    scent: float = 0.0
    walk: float = 0.0
    path: float = 0.0
    prior: float = 0.0
    hub: float = 0.0
    test: float = 0.0


@dataclass(frozen=True)
class RankWeights:
    scent: float = 1.0
    walk: float = 1.0
    path: float = 2.0
    prior: float = 0.0
    hub: float = 0.0
    test: float = 0.0

    def score(self, features: RankFeatures) -> float:
        return (
            self.scent * features.scent
            + self.walk * features.walk
            + self.path * features.path
            + self.prior * features.prior
            - self.hub * features.hub
            - self.test * features.test
        )


def rank_features(
    index: ScentIndex,
    text: str,
    graph: CodeGraph,
    seeds: Mapping[str, float],
    *,
    priors: Mapping[str, float] | None = None,
) -> dict[str, RankFeatures]:
    """Each signal scaled by its query maximum. Priors are explicit caller data, never labels."""
    scent, filename = index.signals(text)
    walk = random_walk(graph, seeds)
    degrees = graph.degrees()
    ids = index.documents
    normalized = [_normalize({id: values.get(id, 0) for id in ids}) for values in (scent, filename, walk)]
    hub = _normalize({id: math.log1p(degrees.get(id, 0)) for id in ids})
    return {
        id: RankFeatures(
            normalized[0][id],
            normalized[2][id],
            normalized[1][id],
            (priors or {}).get(id, 0),
            hub[id],
            float(document.test),
        )
        for id, document in ids.items()
    }


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0) or 1
    return {id: value / maximum for id, value in values.items()}


def rank(candidates: Sequence[str], features: Mapping[str, RankFeatures], weights: RankWeights) -> list[str]:
    """Sort only caller-admitted candidates. Equal scores retain the supplied search order."""
    return sorted(candidates, key=lambda id: -weights.score(features[id]))
