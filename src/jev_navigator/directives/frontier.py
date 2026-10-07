"""The frontier: the order in which a search judges the units its sources reached.

Under any call cap, whatever is ranked last is what gets lost, so the order is a policy, named and
compared. ``STAGE_ORDER`` is the order find_all always had: each source's units in the order the
search lists its sources (by default the units the anchors name, then the units of the files, then
the units holding each name's hits, rarest name first), each wave's batches sorted by place.
``VALUE`` (B1) scores every unit by code before any call and judges the best first, batches in that
order. Its features are facts any language has: which of the request's names a unit's code contains
and how rare each is, whether the unit is named like one, whether its file is, how far the source
that reached it puts it from the anchors (``sources.Reach.distance``), and whether it is a test. A
test is ranked by the same score, never dropped. Ties go to the unit's content hash, never to its
path, and code that repeats a unit already in the queue is judged once.

Under ``VALUE`` each target has its own queue (B3), ranked by the names its description spells out,
and a share of the item slots in every batch: equal by default, set by the caller. A target settles
once a unit clears the Judge's yes bar for it and the units the search's hop sources reach from that
unit (by default its callers and callees), one step deep, have been judged; a settled target draws no
more slots, so its share flows to the targets still open, and the search stops when every target has
settled.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from ..index.units import Item, Unit


@dataclass(frozen=True)
class Weights:
    """What each code feature adds to a unit's value. The name rarities add as they are; distance and a
    test file subtract. Fixed before the first trial (speed-bench's code-first weights, plus the test
    weight); a simulator tunes them as data."""

    defines: float = 2.0
    file_named: float = 1.0
    distance: float = 0.1
    test: float = 1.0


@dataclass(frozen=True)
class Policy:
    """``ranked`` False keeps the stage order. True gives each target a queue ordered by value under
    ``weights``, keeps that order in the batches, splits each batch's slots by the targets' shares, and
    judges repeated code once. ``settles`` (ranked only) lets a target settle and its units push the
    places a search's hop sources reach from them (see the module docstring); without it every unit is
    judged until the call cap. ``role_coverage`` adds a caller-owned prerequisite: every required
    role must be observed among the target's judged units before it can settle."""

    name: str
    ranked: bool
    settles: bool = False
    weights: Weights = field(default_factory=Weights)
    role_coverage: RoleCoverage | None = None
    expands: bool = False
    search_order: bool = False

    def __post_init__(self) -> None:
        if self.settles and not self.ranked:
            raise ValueError("only a ranked policy settles: the stage order judges every unit")


@dataclass(frozen=True)
class RoleCoverage:
    """Caller-required roles and observed roles of judged units. Missing observations never cover a role.

    The reader receives the target and judged unit ids, not predictions about unread units.
    The caller supplies role observations, including any semantic judgments, outside this policy.
    """

    required: Mapping[str, Collection[str]]
    observed: Callable[[str, Collection[str]], Collection[str]]

    def complete(self, target: str, unit_ids: Collection[str]) -> bool:
        roles = self.required.get(target)
        return roles is not None and set(roles) <= set(self.observed(target, unit_ids))


STAGE_ORDER = Policy("stage_order", ranked=False)
VALUE = Policy("value", ranked=True, settles=True)
WHOLE_FRONTIER = Policy("whole_frontier", ranked=False, expands=True, search_order=True)


@dataclass(frozen=True)
class Features:
    """A unit's code features for one target. ``names`` are the target's names its code contains,
    ``rarity`` their sum of ``1 / log2(2 + places)``, ``defines`` whether the unit is named like one,
    ``file_named`` whether its file is, ``distance`` the smallest distance a source reached it at,
    ``test`` whether its file is a test."""

    names: tuple[str, ...]
    rarity: float
    defines: bool
    file_named: bool
    distance: int
    test: bool

    def value(self, weights: Weights) -> float:
        return (
            self.rarity
            + weights.defines * self.defines
            + weights.file_named * self.file_named
            - weights.distance * self.distance
            - weights.test * self.test
        )


def name_rarities(place_counts: Mapping[str, int]) -> dict[str, float]:
    """What each name adds to a unit whose code contains it: a name the sources reached fewer places
    by is rarer and adds more."""
    return {name: 1 / math.log2(2 + count) for name, count in place_counts.items()}


def target_rarities(description: str, rarities: Mapping[str, float]) -> dict[str, float]:
    """The rarities of the names ``description`` spells out as whole words, so each target ranks by its
    own names; all of them when it spells out none."""
    own = {name: rarity for name, rarity in rarities.items() if _holds_word(description, name)}
    return own or dict(rarities)


def features_of(unit: Unit, code: str, distance: int, rarities: Mapping[str, float]) -> Features:
    """``unit``'s features, from its ``code``, the ``distance`` a source reached it at and each name's
    rarity. A name counts only as a whole word: ``limit`` is not in ``limits``."""
    names = tuple(name for name in rarities if _holds_word(code, name))
    return Features(
        names=names,
        rarity=sum(rarities[name] for name in names),
        defines=_last_part(unit.symbol) in rarities,
        file_named=_normalised(PurePosixPath(unit.path).stem) in {_normalised(name) for name in rarities},
        distance=distance,
        test=unit.test,
    )


def value_key(unit: Unit, features: Features, weights: Weights) -> tuple[float, str, str]:
    """The queue order: best value first, then the content hash; identical code last by id."""
    return -features.value(weights), unit.content_sha256, unit.id


class Frontier:
    """The places a ranked search still has to judge: one queue per target, and the hops each target's
    clearing units pushed. A wave's slots go to the targets still drawing in proportion to their
    shares, by smooth weighted round robin, so a share holds over consecutive slots and the same queues
    always draw the same wave. A target draws its hops before its queue, and a settling target only its
    hops. A place drawn for one target leaves every queue. The Judge closes a batch early when the next
    item would not fit, so a share is exact over a wave's slots, not inside every request."""

    def __init__(self, queues: Mapping[str, Sequence[Item]], shares: Mapping[str, float]) -> None:
        self._queues = {target: deque(places) for target, places in queues.items()}
        self._hops: dict[str, deque[Item]] = {target: deque() for target in queues}
        self._shares = {target: shares.get(target, 1.0) for target in queues}
        self._credit = dict.fromkeys(queues, 0.0)
        self._drawn: set[str] = set()

    def push_hops(self, target: str, places: Iterable[Item]) -> None:
        self._hops[target].extend(places)

    def wave(self, size: int, drawing: Mapping[str, bool]) -> list[Item]:
        """Up to ``size`` places for the targets in ``drawing``, each mapped to whether it draws only
        its hops."""
        wave = []
        while len(wave) < size and (target := self._next_target(drawing)) is not None:
            wave.append(self._draw(target))
        return wave

    def _next_target(self, drawing: Mapping[str, bool]) -> str | None:
        able = [target for target, hops_only in drawing.items() if self._has_next(target, hops_only)]
        if not able:
            return None
        for target in able:
            self._credit[target] += self._shares[target]
        chosen = max(able, key=lambda target: self._credit[target])
        self._credit[chosen] -= sum(self._shares[target] for target in able)
        return chosen

    def _has_next(self, target: str, hops_only: bool) -> bool:
        if self._front(self._hops[target]) is not None:
            return True
        return not hops_only and self._front(self._queues[target]) is not None

    def _draw(self, target: str) -> Item:
        queue = self._hops[target] if self._front(self._hops[target]) is not None else self._queues[target]
        place = queue.popleft()
        self._drawn.add(place.id)
        return place

    def _front(self, queue: deque[Item]) -> Item | None:
        while queue and queue[0].id in self._drawn:
            queue.popleft()
        return queue[0] if queue else None


def checked_shares(shares: Mapping[str, float], targets: Collection[str], policy: Policy) -> dict[str, float]:
    """``shares`` as a ranked search uses them: each names a target and is a positive number."""
    if not shares:
        return {}
    if not policy.ranked:
        raise ValueError("shares split a ranked policy's batches; the stage order has one queue")
    if unknown := sorted(set(shares) - set(targets)):
        raise ValueError(f"shares name targets the search does not have: {unknown}")
    if bad := sorted(target for target, share in shares.items() if not (math.isfinite(share) and share > 0)):
        raise ValueError(f"a share must be a positive number: {bad}")
    return dict(shares)


def _holds_word(code: str, name: str) -> bool:
    return re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", code) is not None


def _last_part(symbol: str) -> str:
    """``OrderService.place`` is named ``place``; a schema block ``model Website`` is named ``Website``."""
    return re.split(r"[.\s]", symbol)[-1]


def _normalised(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())
