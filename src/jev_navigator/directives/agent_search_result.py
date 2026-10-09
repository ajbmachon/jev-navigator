"""What ``agent_search`` returns: per point, where to look and whether it is there, and what was examined.

Every value is a fact the search measured: J1-3 probabilities rank places, an existence probability
and its band say whether a point's shortlist holds it, and file counts say how many distinct files
hold definite or possible places. The calling agent applies its own "at least N" and decides whether a
hypothesis holds. Conflicts, the places matching a refuting point, come before support.

Code travels only for the places a reader needs first, within ``max_code_chars``: conflicts, then each
point's shortlist, best places first across points. Every other place is a location, so one result
cannot flood the agent's context. ``to_json`` gives the whole result as plain JSON values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import zip_longest
from typing import Any

from ..index.units import Item, UnresolvedAnchor
from ..sources import anchor_text
from .agent_search_request import SearchScope
from .existence import ExistenceAnswer
from .search_coverage import PointResult

CODE_OMITTED = "omitted: the result's code allowance was spent on higher-ranked places"


class Band(StrEnum):
    """Where a point's existence probability falls: act on it, gather more, or do not act."""

    HIGH = "high"
    MIDDLE = "middle"
    LOW = "low"


class Outcome(StrEnum):
    """``established``: the shortlist holds the point (high band). ``not_found_in_scope``: low band
    once nothing was left to judge for the point; its shortlist is still where to look.
    ``undecided``: anything else, such as a middle band or a stop by budget."""

    ESTABLISHED = "established"
    NOT_FOUND_IN_SCOPE = "not_found_in_scope"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class RankedPlace:
    """One judged place for one point: a whole unit, or a cut unit's best piece. ``roles`` holds the
    role labels' raw probabilities when the place was labelled for this point."""

    place: Item
    symbol: str
    probability: float
    request_sha256: str
    from_store: bool
    roles: Mapping[str, float] | None = None

    def to_json(self) -> dict[str, Any]:
        entry = {
            "place": self.place.id,
            "file": self.place.file,
            "lines": [list(lines) for lines in self.place.ranges],
            "symbol": self.symbol,
            "probability": round(self.probability, 4),
            "request_sha256": self.request_sha256,
            "from_store": self.from_store,
        }
        if self.roles is not None:
            entry["roles"] = {role: round(value, 4) for role, value in self.roles.items()}
        return entry


@dataclass(frozen=True)
class PointOutcome:
    """One point's result. ``shortlist`` is its best places (its beam), ``next`` the following ones as
    locations; ``existence`` was asked over ``existence.shown``, which may be an earlier shortlist than
    the final one. ``definite_files`` hold a place at or above the Judge's yes bar,
    ``possible_files`` (other files) one at or above 0.5. ``coverage`` is the search_coverage fact for
    the point. ``labels`` says whether its shortlist was role-labelled, and why not."""

    id: str
    hypothesis: str
    kind: str
    point: str
    outcome: Outcome
    stopped_by: str
    band: Band | None
    existence: ExistenceAnswer | None
    shortlist: tuple[RankedPlace, ...]
    next: tuple[RankedPlace, ...]
    definite_files: tuple[str, ...]
    possible_files: tuple[str, ...]
    labels: str
    coverage: PointResult

    def to_json(self, bar: float) -> dict[str, Any]:
        existence = self.existence
        return {
            "id": self.id,
            "kind": self.kind,
            "point": self.point,
            "outcome": self.outcome.value,
            "stopped_by": self.stopped_by,
            "band": None if self.band is None else self.band.value,
            "exists": None if existence is None else round(existence.probability, 4),
            "exists_over": [] if existence is None else list(existence.shown),
            "exists_answered_by": None
            if existence is None or existence.answered_by is None
            else existence.answered_by.to_json(),
            "definite_files": list(self.definite_files),
            "possible_files": list(self.possible_files),
            "shortlist": [place.to_json() for place in self.shortlist],
            "next": [place.to_json() for place in self.next],
            "labels": self.labels,
            "coverage": self.coverage.render(bar),
        }


@dataclass(frozen=True)
class HypothesisOutcome:
    """A hypothesis's mechanism, echoed as the agent wrote it, and its points' results."""

    id: str
    mechanism: str
    points: tuple[PointOutcome, ...]


@dataclass(frozen=True)
class Conflict:
    """A place at or above the yes bar for a refuting point."""

    point: str
    place: RankedPlace


@dataclass(frozen=True)
class RequestUse:
    """Requests sent against ``budget``, by step; ``replayed_answers`` came from the answer store free."""

    budget: int
    used: int
    ranking: int
    existence: int
    labels: int
    replayed_answers: int


@dataclass(frozen=True)
class SearchCoverage:
    """What the search reached and examined. ``not_judged`` counts reached units left unjudged by the
    source that reached them (its label) or by the reason they could not be judged; ``unlisted`` files
    gave no units, ``unresolved`` anchors named none, ``outside_scope`` files were named but left out by
    the request's scope, and ``evicted`` places had to leave an existence request for its size."""

    units_reached: int
    units_judged: int
    not_judged: Mapping[str, int]
    unlisted: Mapping[str, str]
    unresolved: tuple[UnresolvedAnchor, ...]
    outside_scope: tuple[str, ...]
    evicted: tuple[str, ...]
    scope: SearchScope

    def to_json(self) -> dict[str, Any]:
        return {
            "units_reached": self.units_reached,
            "units_judged": self.units_judged,
            "not_judged": dict(self.not_judged),
            "unlisted": dict(self.unlisted),
            "unresolved_anchors": [
                {"anchor": anchor_text(gap.anchor), "problem": gap.problem} for gap in self.unresolved
            ],
            "outside_scope": list(self.outside_scope),
            "evicted_from_existence": list(self.evicted),
            "scope": {
                "include": list(self.scope.include),
                "exclude": list(self.scope.exclude),
                "with_tests": self.scope.with_tests,
            },
        }


@dataclass(frozen=True)
class AgentSearchResult:
    """The search's result. ``code`` maps each place id the result names to its code, or to None when
    the code allowance was spent first; ``failure`` holds the error a failed search stopped on."""

    hypotheses: tuple[HypothesisOutcome, ...]
    conflicts: tuple[Conflict, ...]
    code: Mapping[str, str | None]
    stopped_by: str
    requests: RequestUse
    coverage: SearchCoverage
    yes_bar: float
    failure: Exception | None = None

    def point(self, point_id: str) -> PointOutcome:
        return next(
            point for hypothesis in self.hypotheses for point in hypothesis.points if point.id == point_id
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "stopped_by": self.stopped_by,
            "failure": None if self.failure is None else f"{type(self.failure).__name__}: {self.failure}",
            "requests": vars(self.requests),
            "conflicts": [
                {"point": conflict.point, **conflict.place.to_json()} for conflict in self.conflicts
            ],
            "hypotheses": [
                {
                    "id": hypothesis.id,
                    "mechanism": hypothesis.mechanism,
                    "points": [point.to_json(self.yes_bar) for point in hypothesis.points],
                }
                for hypothesis in self.hypotheses
            ],
            "code": {place_id: _code_entry(code) for place_id, code in self.code.items()},
            "coverage": self.coverage.to_json(),
        }


def code_within(
    conflicts: Sequence[Conflict],
    points: Sequence[PointOutcome],
    read: Callable[[Item], str],
    max_code_chars: int,
) -> dict[str, str | None]:
    """Each place the result names, with its code while the allowance lasts: conflicts first, then the
    points' shortlists one rank at a time across points, so every point's best place comes before any
    point's second; a place that would overrun the allowance gets None and smaller later ones still fit."""
    code: dict[str, str | None] = {}
    spent = 0
    for place in _reading_order(conflicts, points):
        if place.id in code:
            continue
        text = read(place)
        fits = spent + len(text) <= max_code_chars
        code[place.id] = text if fits else None
        spent += len(text) if fits else 0
    return code


def _reading_order(conflicts: Sequence[Conflict], points: Sequence[PointOutcome]) -> list[Item]:
    by_rank = zip_longest(*(point.shortlist for point in points))
    shortlisted = [place.place for rank in by_rank for place in rank if place is not None]
    return [*(conflict.place.place for conflict in conflicts), *shortlisted]


def _code_entry(code: str | None) -> dict[str, Any]:
    return {"code": code} if code is not None else {"code": None, "omitted": CODE_OMITTED}
