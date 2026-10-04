"""When a Jev answer counts as yes, no, unsure or confident enough to act on.

Precedence, lowest first: library defaults, then environment variables (read once at the edge with
``Thresholds.from_env``), then a directive's own defaults, then per-call overrides. Each later layer
is a partial mapping applied with ``Thresholds.updated``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

from ..errors import UsageError

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
                raise UsageError(f"{name} must be between 0 and 1, got {value}")
        if self.noul_no_at >= self.noul_yes_at:
            raise UsageError(f"noul_no_at ({self.noul_no_at}) must be below noul_yes_at ({self.noul_yes_at})")

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
