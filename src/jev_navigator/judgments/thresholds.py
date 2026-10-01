"""When a Jev answer counts as yes, no, unsure or confident enough to act on.

Precedence, lowest first: library defaults, then environment variables (read once at the edge with
``Thresholds.from_env``), then a directive's own defaults, then per-call overrides. Each later layer
is a partial mapping applied with ``Thresholds.updated``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

ENVIRONMENT_NAMES = {
    "choice_min_confidence": "JEV_NAVIGATOR_CHOICE_MIN_CONFIDENCE",
    "noul_yes_at": "JEV_NAVIGATOR_NOUL_YES_AT",
    "noul_no_at": "JEV_NAVIGATOR_NOUL_NO_AT",
}


class NoulVerdict(StrEnum):
    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


@dataclass(frozen=True)
class Thresholds:
    choice_min_confidence: float = 0.70
    noul_yes_at: float = 0.80
    noul_no_at: float = 0.20

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.noul_no_at >= self.noul_yes_at:
            raise ValueError(f"noul_no_at ({self.noul_no_at}) must be below noul_yes_at ({self.noul_yes_at})")

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> Thresholds:
        environment = os.environ if environment is None else environment
        found = {
            field: float(environment[name])
            for field, name in ENVIRONMENT_NAMES.items()
            if name in environment
        }
        return cls().updated(found)

    def updated(self, overrides: Mapping[str, float] | None) -> Thresholds:
        return replace(self, **overrides) if overrides else self

    def noul_verdict(self, probability: float) -> NoulVerdict:
        if probability >= self.noul_yes_at:
            return NoulVerdict.YES
        if probability <= self.noul_no_at:
            return NoulVerdict.NO
        return NoulVerdict.UNSURE

    def choice_is_confident(self, confidence: float) -> bool:
        return confidence >= self.choice_min_confidence

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Calibration:
    """Maps one model's probabilities onto the shared scale, for a model that runs hotter or
    colder than the one the thresholds were chosen for.

    ``own`` holds the bars that model needs, ``shared`` the thresholds everything else uses. The
    model's own bars land exactly on the shared bars, and values between them move linearly, so
    a verdict under the shared thresholds is the verdict under the model's own, and every other
    threshold, ranking and report compares that model's answers like with like. Noul
    probabilities move by ``noul_no_at`` and ``noul_yes_at``, and a choice's confidence by
    ``choice_min_confidence``; choice probabilities and scores are kept as they are.
    """

    own: Thresholds
    shared: Thresholds

    def __post_init__(self) -> None:
        self._noul_curve()
        self._confidence_curve()

    def noul_probability(self, probability: float) -> float:
        return _along(self._noul_curve(), probability)

    def choice_confidence(self, confidence: float) -> float:
        return _along(self._confidence_curve(), confidence)

    def _noul_curve(self) -> tuple[tuple[float, float], ...]:
        return _curve(
            (self.own.noul_no_at, self.shared.noul_no_at), (self.own.noul_yes_at, self.shared.noul_yes_at)
        )

    def _confidence_curve(self) -> tuple[tuple[float, float], ...]:
        return _curve((self.own.choice_min_confidence, self.shared.choice_min_confidence))


def _curve(*bars: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    points = sorted({(0.0, 0.0), *bars, (1.0, 1.0)})
    if len({x for x, _ in points}) != len(points):
        raise ValueError("a bar at 0 or 1 cannot move: 0 and 1 keep their meaning on every scale")
    return tuple(points)


def _along(points: tuple[tuple[float, float], ...], value: float) -> float:
    """``value`` moved along the curve. A bar lands exactly on its shared bar, and a value strictly
    between two bars stays strictly between theirs, so rounding never moves a verdict."""
    if value <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if value == x1:
            return y1
        if value < x1:
            moved = y0 + (value - x0) * (y1 - y0) / (x1 - x0)
            return min(max(moved, math.nextafter(y0, y1)), math.nextafter(y1, y0))
    return points[-1][1]
