"""Judge every unit of a population against described targets: one Noul per unit per target.

The population is units (``index/units.py``), each one a listing lists, that the search's sources
(``sources.py``) reach from the caller's seeds: by default the units the caller's line and range
anchors name, the units of the named files, and the units holding each hit of the named texts. A
policy (``frontier``) orders them: ``STAGE_ORDER``, the default, judges them source by source in that
order, names with fewer hits first, so a common word never decides which hits of a rare name are
seen; ``VALUE`` judges the units worth most by code first, each target from its own queue and share,
and stops spending on a target once it has settled. Code finds the units; Jev judges each one whole,
or a unit larger than its room in a request by its pieces. The Judge's call cap is the only budget: no
code step is capped. The result keeps every raw answer with its place; any bar belongs to the caller.

``find_all`` judges code units only. ``find_all_text`` judges the text units of the files JVN does
not parse, the same way and with the same question, and never a code unit; ``find_text`` is the text
search that stops once a unit is found. A caller gives a text search its own Judge, so it never
spends code search's budget.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import groupby
from typing import TypeVar

from ..index.code_index import CodeIndex
from ..index.units import (
    Anchor,
    Item,
    Piece,
    RangeAnchor,
    Reading,
    Unit,
    UnitReader,
    UnresolvedAnchor,
    best_piece,
    items_to_judge,
    read_ranges,
)
from ..judgments.judge import CallCapReachedError, CheckResult, Judge, Refusal
from ..judgments.questions import Check, item_path, serialized_chars
from ..judgments.secrets import DEFAULT_MASKER
from ..judgments.thresholds import NoulVerdict
from ..mentions import code_names_in, literal_names_in
from ..sources import ANCHORS, CALLEES, CALLERS, FILES, NAMES, TEXT_NAMES, Reach, Seeds, Source
from .find_code import search_failure
from .frontier import (
    STAGE_ORDER,
    Features,
    Frontier,
    Policy,
    checked_shares,
    features_of,
    name_rarities,
    target_rarities,
    value_key,
)

ITEMS = "items"
TARGETS = "targets"
DELIVERED = "already delivered by the caller"
TOO_LARGE = "too large to judge"
NOT_REACHED = "not reached: the search stopped first"
REFUSED = "refused: the provider or the final secret scan refused its request, so its answer is unknown"
FOUND = "found"
SETTLED = "settled"
FIND_TEXT_TARGET = "target"
BATCHES_PER_WAVE = 16
"""How many requests' worth of places one wave hands the Judge, in the population's order. The Judge
sends a wave's batches together and orders them by place, so the population's order holds between
waves, and exactly only at one batch per wave. Like ``items_per_request`` it shapes the batches, and
so the answer store's keys; it does not follow the Judge's concurrency, so that setting never moves
them."""
CODE_SOURCES: tuple[Source, ...] = (ANCHORS, FILES, NAMES)
"""find_all's sources by default: the units the anchors name, the units of the files, then the units
holding each name's hits."""
TEXT_SOURCES: tuple[Source, ...] = (ANCHORS, FILES, TEXT_NAMES)
"""find_all_text's and find_text's sources by default: as ``CODE_SOURCES``, with a name's hits only in
text files and never in a lockfile."""
HOP_SOURCES: tuple[Source, ...] = (CALLERS, CALLEES)
"""What a unit that clears a target's bar pushes under a settling policy, by default: its callers and
callees."""

T = TypeVar("T")


def match_check(target: str) -> Check:
    """The one question asked of every unit for ``target``: no criteria, the description in the
    shared state, as jgrep asks it."""
    return Check(
        f"match_{target}",
        f"Look only at `{{item}}`. Does that code match the description in `{TARGETS}.{target}`?",
    )


@dataclass(frozen=True)
class NameHits:
    """A request name's distinct places: how many the search's sources reached by the name, how many
    the search resolved into units before it stopped, and how many of those named no listed unit (a line
    of a file the reading leaves out, or of top-level code a listing leaves out, such as imports)."""

    found: int
    reached: int
    without_unit: int


@dataclass(frozen=True)
class UnitScore:
    """A unit's answer for one target: its own, or for a cut unit its best judged piece's."""

    unit: Unit
    answer: CheckResult
    piece: Piece | None

    @property
    def probability(self) -> float:
        return self.answer.probability


@dataclass(frozen=True)
class FindAllResult:
    """``units`` are every unit of the population, in place order; ``judged`` each target's answers,
    each naming its place; ``not_judged`` each unit or piece id left unjudged with the reason.
    ``room`` is what one unit's code had in a request and ``batches_per_wave`` the wave size, both
    None when the search never started. ``unlisted`` files gave no units, and ``unresolved`` anchors
    named none: the caller's, and any other a source reached other than by a name (one reached by a
    name counts in ``names``). ``refusals`` keeps each refused place's error; the place is
    ``not_judged`` as ``REFUSED`` and the search goes on past it. ``sources`` are the sources the
    search used, its hop sources last, and ``entered_by`` gives the name of the source each unit
    entered the population by: the first that reached it, or under a ranked policy the one that put it
    nearest. ``policy`` ordered the population; under a ranked policy ``features`` holds each target's
    code features of each unit, and ``repeat_of`` names, for a unit whose code repeats another's, the
    unit judged in its place, whose answers and reasons it shares. Under a settling policy ``pushed``
    gives each unit that cleared a target's bar the units its hop sources reached from it, and
    ``settled`` the targets that settled, in the order they did."""

    targets: Mapping[str, str]
    room: int | None
    batches_per_wave: int | None
    units: tuple[Unit, ...]
    judged: Mapping[str, tuple[CheckResult, ...]]
    not_judged: Mapping[str, str]
    unlisted: Mapping[str, str]
    unresolved: tuple[UnresolvedAnchor, ...]
    names: Mapping[str, NameHits]
    unparsed_files: frozenset[str]
    stopped_by: str
    calls: int
    failure: Exception | None = None
    refusals: tuple[Refusal, ...] = ()
    sources: tuple[Source, ...] = ()
    entered_by: Mapping[str, str] = field(default_factory=dict)
    policy: Policy = STAGE_ORDER
    features: Mapping[str, Mapping[str, Features]] = field(default_factory=dict)
    repeat_of: Mapping[str, str] = field(default_factory=dict)
    pushed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    settled: tuple[str, ...] = ()

    @classmethod
    def not_started(cls, targets: Mapping[str, str], stopped_by: str) -> FindAllResult:
        """The result of a search that stopped before it listed anything."""
        return cls(
            dict(targets),
            None,
            None,
            (),
            {target: () for target in targets},
            {},
            {},
            (),
            {},
            frozenset(),
            stopped_by,
            0,
        )

    def scores(self, target: str) -> tuple[UnitScore, ...]:
        """Every judged unit's answer for ``target``, in place order; a repeat has its copy's answer."""
        answers = {answer.place.id: answer for answer in self.judged[target] if answer.place is not None}
        probabilities = {place_id: answer.probability for place_id, answer in answers.items()}
        by_id = {unit.id: unit for unit in self.units}
        scored = (
            _unit_score(unit, by_id[self.repeat_of.get(unit.id, unit.id)], answers, probabilities)
            for unit in self.units
        )
        return tuple(score for score in scored if score is not None)

    def ranked(self, target: str) -> tuple[UnitScore, ...]:
        """Every judged unit's answer for ``target``, best first. A tie goes to the unit worth more by
        code under a ranked policy, then to the content hash, never to the path."""
        features = self.features.get(target, {})
        return tuple(sorted(self.scores(target), key=lambda score: self._rank(score, features)))

    def _rank(self, score: UnitScore, features: Mapping[str, Features]) -> tuple[float, float, str, str]:
        unit_features = features.get(score.unit.id)
        value = 0.0 if unit_features is None else unit_features.value(self.policy.weights)
        return -score.probability, -value, score.unit.content_sha256, score.unit.id

    @property
    def coverage(self) -> str:
        """Examination coverage, never proof that the semantic answers are correct."""
        reasons = set(self.not_judged.values())
        if self.stopped_by != "scope_examined" or NOT_REACHED in reasons:
            return "partial"
        if self.unlisted or self.unresolved or self.unparsed_files or reasons & {TOO_LARGE, REFUSED}:
            return "scope_incomplete"
        return "units_examined"


def find_all(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
    policy: Policy = STAGE_ORDER,
    shares: Mapping[str, float] | None = None,
    sources: Sequence[Source] = CODE_SOURCES,
    reading: Reading = Reading.CODE,
    hops: Sequence[Source] = HOP_SOURCES,
) -> FindAllResult:
    """Judge the units ``sources`` reach from ``anchors``, ``files``, ``names`` and the targets'
    descriptions (by default the units ``anchors`` name, the units of ``files`` and the units holding
    each hit of ``names``), in the order ``policy`` gives (see ``frontier``), in waves of
    ``batches_per_wave`` requests' worth, until the judge's call cap stops it.

    ``targets`` maps a name (an identifier) to a description; each unit is asked one question per
    target, all in the same request. ``delivered`` names the lines the caller already shows: a unit
    or piece whose every line lies in them is not judged, one with a line outside them is. ``completed``
    holds each target's answers from an earlier run over the same code; a place answered for every
    target is not asked again. A failed request ends the search ``failed`` with ``failure`` holding
    the error (see ``search_failure``), Ctrl-C ends it ``cancelled``, and either way ``judged`` keeps
    every answer that arrived.

    Under a ranked policy ``shares`` sets each target's share of the item slots in every batch (a
    target it does not name has 1). Under a settling policy (``VALUE``) a unit clears a target's bar
    when its answer is yes by the Judge's thresholds; that target then draws only the units ``hops``
    reach from it (by default its callers and callees), and settles once none of them is left to
    judge. Every unit drawn is still asked every target's question, so a settled target spends no call
    of its own. Once every target has settled the search ends ``settled``, with the units it never
    reached ``not_judged``.
    """
    composition = _Composition(tuple(sources), tuple(hops), policy, shares or {})
    search = _begin(
        index, judge, targets, delivered, completed, cancelled, batches_per_wave, reading, composition
    )
    return _finished(search, _seeds(targets, files, anchors, names))


def find_all_text(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
    policy: Policy = STAGE_ORDER,
    shares: Mapping[str, float] | None = None,
    sources: Sequence[Source] = TEXT_SOURCES,
    hops: Sequence[Source] = HOP_SOURCES,
) -> FindAllResult:
    """``find_all`` over text units (``Reading.TEXT``): the files JVN does not parse, each in the blocks
    its format gives, a code file named ``unlisted``. By default a name's hits in code and in lockfiles
    are left out (``TEXT_SOURCES``), so a common word never floods the search with a lockfile's
    pieces; a lockfile ``files`` or ``anchors`` name is judged."""
    composition = _Composition(tuple(sources), tuple(hops), policy, shares or {})
    search = _begin(
        index, judge, targets, delivered, completed, cancelled, batches_per_wave, Reading.TEXT, composition
    )
    return _finished(search, _seeds(targets, files, anchors, names))


def find_text(
    index: CodeIndex,
    judge: Judge,
    description: str,
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
    policy: Policy = STAGE_ORDER,
    sources: Sequence[Source] = TEXT_SOURCES,
) -> FindAllResult:
    """``find_all_text`` for the one target ``description``, named ``FIND_TEXT_TARGET``, ending
    ``found`` after the first wave in which a unit's answer is yes by the judge's thresholds."""
    targets = {FIND_TEXT_TARGET: description}
    composition = _Composition(tuple(sources), HOP_SOURCES, policy, {})
    search = _begin(
        index, judge, targets, delivered, None, cancelled, batches_per_wave, Reading.TEXT, composition
    )
    search.stops_when_found = True
    return _finished(search, _seeds(targets, files, anchors, names))


def _seeds(
    targets: Mapping[str, str], files: Sequence[str], anchors: Sequence[Anchor], names: Sequence[str]
) -> Seeds:
    return Seeds(tuple(dict.fromkeys(names)), tuple(targets.values()), tuple(files), tuple(anchors))


def _finished(search: _Search, seeds: Seeds) -> FindAllResult:
    try:
        search.run(seeds)
    except (KeyboardInterrupt, Exception) as error:  # noqa: BLE001 - _stop_by owns how an error ends a search
        return search.ended(error)
    return search.ended(None)


async def find_all_async(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
    policy: Policy = STAGE_ORDER,
    shares: Mapping[str, float] | None = None,
    sources: Sequence[Source] = CODE_SOURCES,
    reading: Reading = Reading.CODE,
    hops: Sequence[Source] = HOP_SOURCES,
) -> FindAllResult:
    """``find_all`` with each wave's requests sent concurrently through the Judge's async form, for an
    async client. Reaching, listing, resolving and reading code run in a worker thread, so the event
    loop stays free. ``cancelled`` is read before each parse and between waves, since the Judge's async
    form reads none; a cancelled task's ``CancelledError`` is never caught."""
    composition = _Composition(tuple(sources), tuple(hops), policy, shares or {})
    search = _begin(
        index, judge, targets, delivered, completed, cancelled, batches_per_wave, reading, composition
    )
    return await _finished_async(search, _seeds(targets, files, anchors, names))


async def find_all_text_async(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
    policy: Policy = STAGE_ORDER,
    shares: Mapping[str, float] | None = None,
    sources: Sequence[Source] = TEXT_SOURCES,
    hops: Sequence[Source] = HOP_SOURCES,
) -> FindAllResult:
    """``find_all_text`` the way ``find_all_async`` runs ``find_all``."""
    composition = _Composition(tuple(sources), tuple(hops), policy, shares or {})
    search = _begin(
        index, judge, targets, delivered, completed, cancelled, batches_per_wave, Reading.TEXT, composition
    )
    return await _finished_async(search, _seeds(targets, files, anchors, names))


async def _finished_async(search: _Search, seeds: Seeds) -> FindAllResult:
    try:
        await search.run_async(seeds)
    except (KeyboardInterrupt, Exception) as error:  # noqa: BLE001 - _stop_by owns how an error ends a search
        return search.ended(error)
    return search.ended(None)


@dataclass(frozen=True)
class _Composition:
    """What a search is composed of beside its seeds and Judge: the sources that start it, the hop
    sources a clearing unit pushes through, the policy that orders the units and the targets' shares."""

    sources: tuple[Source, ...]
    hops: tuple[Source, ...]
    policy: Policy
    shares: Mapping[str, float]


def _begin(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    delivered: Sequence[RangeAnchor],
    completed: Mapping[str, Sequence[CheckResult]] | None,
    cancelled: Callable[[], bool] | None,
    batches_per_wave: int,
    reading: Reading,
    composition: _Composition,
) -> _Search:
    _require_identifiers(targets)
    if batches_per_wave < 1:
        raise ValueError("batches_per_wave must be at least 1")
    shares = checked_shares(composition.shares, targets, composition.policy)
    scoped = judge.scope()
    if scoped.masker is DEFAULT_MASKER:
        scoped.masker = _SearchMasker()
    search = _Search(
        index, scoped, targets, delivered, cancelled, batches_per_wave, reading, composition, shares
    )
    search.resume(completed or {})
    return search


class _SearchMasker:
    """Reuse deterministic built-in scans for this search, with bounded string retention."""

    def __init__(self) -> None:
        self.mask = lru_cache(maxsize=256)(DEFAULT_MASKER.mask)
        self.masked_values = lru_cache(maxsize=256)(DEFAULT_MASKER.masked_values)


def _stop_by(error: KeyboardInterrupt | Exception | None) -> tuple[str, Exception | None]:
    """How an error ends a search: a call cap ``budget``, Ctrl-C ``cancelled``, any other error
    ``failed`` holding it (``search_failure`` decides), and no error ``scope_examined``, or ``found``
    for a search that stops once found (``_Search.ended``)."""
    if error is None:
        return "scope_examined", None
    if isinstance(error, CallCapReachedError):
        return "budget", None
    if isinstance(error, KeyboardInterrupt):
        return "cancelled", None
    return "failed", search_failure(error)


def _require_identifiers(targets: Mapping[str, str]) -> None:
    if not targets:
        raise ValueError("find_all needs at least one target")
    if bad := [target for target in targets if not target.isidentifier()]:
        raise ValueError(f"target names must be identifiers, so a question can point at them: {bad}")


class _Search:
    def __init__(
        self,
        index: CodeIndex,
        judge: Judge,
        targets: Mapping[str, str],
        delivered: Sequence[RangeAnchor],
        cancelled: Callable[[], bool] | None,
        batches_per_wave: int,
        reading: Reading,
        composition: _Composition,
        shares: Mapping[str, float],
    ) -> None:
        self.index = index
        self.reading = reading
        self.sources = composition.sources
        self.hops = composition.hops
        self.policy = composition.policy
        self.shares = shares
        self.seeds = Seeds()
        self.stops_when_found = False
        self.found_one = False
        self.batches_per_wave = batches_per_wave
        self.judge = judge
        self.targets = dict(targets)
        self.checks = [match_check(target) for target in targets]
        self.target_of = {check.name: target for check, target in zip(self.checks, targets, strict=True)}
        self.shared = {TARGETS: self.targets}
        self.room = _room(judge, index, self.checks, self.shared)
        self.reader = UnitReader(index, self.room, listed_only=True, reading=reading)
        self.delivered = _lines_by_file(delivered)
        self.delivered_units: set[str] = set()
        self.cancelled = cancelled
        self.answered: set[tuple[str, str]] = set()
        self.judged: dict[str, list[CheckResult]] = {target: [] for target in targets}
        self.units: dict[str, Unit] = {}
        self.resolved_units: dict[str | Anchor, tuple[Unit, ...]] = {}
        self.entered_by: dict[str, str] = {}
        self.features: dict[str, dict[str, Features]] = {target: {} for target in targets}
        self.repeat_of: dict[str, str] = {}
        self.first_with_content: dict[str, str] = {}
        self.unit_of_place: dict[str, str] = {}
        self.places_of: dict[str, list[Item]] = {}
        self.rarities: dict[str, dict[str, float]] = {}
        self.pushed: dict[str, tuple[str, ...]] = {}
        self.hopped: set[str] = set()
        self.expanded: dict[str, set[str]] = {target: set() for target in targets}
        self.awaiting: dict[str, set[str]] = {target: set() for target in targets}
        self.settled: list[str] = []
        self.not_judged: dict[str, str] = {}
        self.unlisted: dict[str, str] = {}
        self.unresolved: list[UnresolvedAnchor] = []
        self.refusals: list[Refusal] = []
        self.found: dict[str, set[str | Anchor]] = {}
        self.reached: defaultdict[str, set[str | Anchor]] = defaultdict(set)
        self.without_unit: defaultdict[str, set[str | Anchor]] = defaultdict(set)

    def stopped(self) -> bool:
        return self.cancelled is not None and self.cancelled()

    def resume(self, completed: Mapping[str, Sequence[CheckResult]]) -> None:
        for target in self.targets:
            for answer in completed.get(target, ()):
                self._record(target, answer)

    def run(self, seeds: Seeds) -> None:
        for places in self._waves(seeds):
            self._judge(places)
            if self.found_one:
                return

    async def run_async(self, seeds: Seeds) -> None:
        waves = self._waves(seeds)
        while (wave := await asyncio.to_thread(self._next_wave, waves)) is not None:
            places, entries = wave
            if self.stopped():
                return
            async for name, answer in self.judge.iter_check_every_async(
                self.checks,
                entries,
                self.shared,
                list_name=ITEMS,
                places=places,
                refusals=self.refusals,
                keep_order=self.policy.ranked or self.policy.search_order,
            ):
                self._record(self.target_of[name], answer)

    def ended(self, error: KeyboardInterrupt | Exception | None) -> FindAllResult:
        stop, failure = _stop_by(error)
        if self.stopped() and failure is None:
            stop = "cancelled"
        elif error is None and self.found_one:
            stop = FOUND
        elif error is None and self._settled_early():
            stop = SETTLED
        return self.result(stop, failure)

    def _settled_early(self) -> bool:
        """Every target settled while units were still waiting to be judged."""
        return len(self.settled) == len(self.targets) and NOT_REACHED in self.not_judged.values()

    def _waves(self, seeds: Seeds) -> Iterator[list[Item]]:
        self.seeds = seeds
        size = self.judge.items_per_request * self.batches_per_wave
        if self.policy.ranked:
            return self._ranked_waves(size)
        if self.policy.expands:
            return self._expanding_waves(size)
        return _chunks(self._staged_population(), size)

    def _expanding_waves(self, size: int) -> Iterator[list[Item]]:
        """Register the entire reached frontier before judging, then expand newly judged units.

        Answers control neither admission nor expansion. Every unit is expanded once, and fresh
        terms are searched once. A spent call budget leaves the full pending population visible.
        """
        pending = list(self._staged_population())
        expanded: set[str] = set()
        terms = set(self.seeds.names)
        searched_literals: set[str] = set()
        while not self.stopped():
            yield from _chunks(pending, size)
            judged = self.delivered_units | {
                self.unit_of_place[answer.place.id]
                for answers in self.judged.values()
                for answer in answers
                if answer.place is not None and answer.place.id in self.unit_of_place
            }
            fresh_ids = judged - expanded
            fresh = [unit for unit in self.units.values() if unit.id in fresh_ids]
            if not fresh:
                return
            expanded.update(unit.id for unit in fresh)
            codes = tuple(read_ranges(self.index, unit.path, unit.ranges) for unit in fresh)
            names = tuple(
                dict.fromkeys(name for code in codes for name in code_names_in(code) if name not in terms)
            )
            terms.update(names)
            literals = tuple(
                dict.fromkeys(
                    literal
                    for code in codes
                    for literal in literal_names_in(code)
                    if literal not in searched_literals
                )
            )
            searched_literals.update(literals)
            seeds = Seeds(
                names=names,
                texts=codes,
                files=tuple(dict.fromkeys(unit.path for unit in fresh)),
                units=tuple(fresh),
                literals=literals,
            )
            pending = []
            for source in self.hops:
                reaches = list(source.reach(self.index, seeds))
                pending.extend(self._admitted(self._units_reached(reaches)))

    def _next_wave(self, waves: Iterator[list[Item]]) -> tuple[list[Item], list[dict]] | None:
        """The next wave's places and the entries Jev reads for them; None when the population is spent."""
        places = next(waves, None)
        return None if places is None else (places, self._entries(places))

    def _entries(self, places: Sequence[Item]) -> list[dict]:
        return [
            {"file": place.file, "code": read_ranges(self.index, place.file, place.ranges)}
            for place in places
        ]

    def _staged_population(self) -> Iterator[Item]:
        """Every place still to judge, source by source in the order the search lists them. The files
        a source reached are listed together; a request's worth of anchors is resolved together, and
        only when the next wave needs it."""
        if self.stopped():
            return
        for reaches in self._start_reaches():
            for group in self._resolution_groups(reaches):
                if self.stopped():
                    return
                yield from self._admitted(self._units_reached(group))

    def _start_reaches(self) -> list[list[Reach]]:
        """Each start source's places, in the order the search lists the sources. Every request name
        counts distinct places over all of them, so a name no source reached counts 0."""
        reaches = [list(source.reach(self.index, self.seeds)) for source in self.sources]
        self.found = {name: set() for name in self.seeds.names}
        for reach in (reach for source_reaches in reaches for reach in source_reaches):
            for name in reach.names:
                self.found.setdefault(name, set()).add(reach.at)
        return reaches

    def _resolution_groups(self, reaches: Sequence[Reach]) -> Iterator[list[Reach]]:
        for is_file, run in groupby(reaches, key=_is_file_reach):
            group = list(run)
            yield from ([group] if is_file else _chunks(group, self.judge.items_per_request))

    def _ranked_waves(self, size: int) -> Iterator[list[Item]]:
        """Waves of ``size`` places drawn from the targets' queues (``frontier.Frontier``). Every source
        is reached, every place resolved and every unit admitted before the first wave, so a unit the
        search never reaches is still counted; a unit whose code repeats one already admitted shares
        that unit's answers instead of being judged. Before each wave the targets settle."""
        if self.stopped():
            return
        frontier = self._frontier(self._candidates())
        while not self.stopped() and (drawing := self._drawing(frontier)):
            wave = frontier.wave(size, drawing)
            if not wave:
                return
            yield wave

    def _frontier(self, candidates: Sequence[tuple[Unit, Reach]]) -> Frontier:
        """Each target's queue: every admitted place, its unit's value for that target first."""
        rarities = name_rarities({name: len(places) for name, places in self.found.items()})
        self.rarities = {target: target_rarities(text, rarities) for target, text in self.targets.items()}
        for unit, reach in candidates:
            self._add_features(unit, reach)
        for unit, reach in sorted(candidates, key=lambda candidate: self._admission_key(candidate[0])):
            self.places_of[unit.id] = self._admitted_once(unit, reach)
        queues = {
            target: [
                place
                for unit, _ in sorted(candidates, key=lambda candidate: self._value_key(target, candidate[0]))
                for place in self.places_of[unit.id]
            ]
            for target in self.targets
        }
        return Frontier(queues, self.shares)

    def _add_features(self, unit: Unit, reach: Reach) -> None:
        code = read_ranges(self.index, unit.path, unit.ranges)
        for target in self.targets:
            self.features[target][unit.id] = features_of(unit, code, reach.distance, self.rarities[target])

    def _admission_key(self, unit: Unit) -> tuple[float, str, str]:
        """Of units with the same code, the one worth most to any target is judged for all of them."""
        best = max(self.features[target][unit.id].value(self.policy.weights) for target in self.targets)
        return -best, unit.content_sha256, unit.id

    def _value_key(self, target: str, unit: Unit) -> tuple[float, str, str]:
        return value_key(unit, self.features[target][unit.id], self.policy.weights)

    def _drawing(self, frontier: Frontier) -> dict[str, bool]:
        """Each target still drawing, mapped to whether it draws only its hops. Under a settling policy
        a target with a unit that cleared its bar first gets that unit's hops, and settles once none of
        its hops is left to judge and its required roles are covered. Without role observations it
        keeps drawing the ordinary queue as well as hops."""
        if not self.policy.settles:
            return dict.fromkeys(self.targets, False)
        drawing = {}
        for target in self.targets:
            if target in self.settled:
                continue
            clearing = self._clearing(target)
            self._push_hops(frontier, target, sorted(clearing - self.expanded[target]))
            coverage = self.policy.role_coverage
            judged_units = {
                self.unit_of_place[answer.place.id]
                for answer in self.judged[target]
                if answer.place is not None and answer.place.id in self.unit_of_place
            }
            roles_complete = coverage is None or coverage.complete(target, judged_units)
            if not clearing or self._awaiting(target) or not roles_complete:
                drawing[target] = bool(clearing) and roles_complete
            else:
                self.settled.append(target)
        return drawing

    def _clearing(self, target: str) -> set[str]:
        """The units whose answer for ``target`` is yes by the Judge's thresholds."""
        return {
            self.unit_of_place[answer.place.id]
            for answer in self.judged[target]
            if answer.verdict is NoulVerdict.YES and answer.place.id in self.unit_of_place
        }

    def _push_hops(self, frontier: Frontier, target: str, unit_ids: Sequence[str]) -> None:
        """Gives ``target`` the units the hop sources reach from each clearing unit, worth most to it
        first; a unit a push brought in pushes nothing, so hops go one step deep."""
        for unit_id in unit_ids:
            self.expanded[target].add(unit_id)
            if unit_id in self.hopped:
                continue
            hops = sorted(self._hops_of(unit_id), key=lambda hop: self._value_key(target, self.units[hop]))
            places = [place for hop in hops for place in self._still_pending(hop)]
            frontier.push_hops(target, places)
            self.awaiting[target].update(place.id for place in places)

    def _hops_of(self, unit_id: str) -> tuple[str, ...]:
        """The units the hop sources reach from ``unit_id``, seeded with that unit alone, each admitted on
        first sight; the unit itself is never its own hop."""
        if unit_id not in self.pushed:
            seeds = Seeds(units=(self.units[unit_id],))
            reached = (
                pair
                for source in self.hops
                for pair in self._units_reached(list(source.reach(self.index, seeds)))
            )
            hops = {}
            for hop, reach in reached:
                if hop.id != unit_id:
                    hops.setdefault(hop.id, (hop, reach))
            for hop, reach in hops.values():
                if hop.id not in self.units:
                    self._add_features(hop, reach)
                    self.places_of[hop.id] = self._admitted_once(hop, reach)
                    self.hopped.add(hop.id)
            self.pushed[unit_id] = tuple(hops)
        return self.pushed[unit_id]

    def _still_pending(self, unit_id: str) -> list[Item]:
        """The places of the unit judged for ``unit_id`` (itself, or the copy it repeats) still waiting."""
        judged_as = self.repeat_of.get(unit_id, unit_id)
        return [place for place in self.places_of.get(judged_as, []) if self._waiting(place.id)]

    def _awaiting(self, target: str) -> bool:
        return any(self._waiting(place_id) for place_id in self.awaiting[target])

    def _waiting(self, place_id: str) -> bool:
        """Still to judge: neither answered nor cut (too large, delivered or refused)."""
        refused = any(refusal.place is not None and refusal.place.id == place_id for refusal in self.refusals)
        return self.not_judged.get(place_id) == NOT_REACHED and not refused

    def _candidates(self) -> list[tuple[Unit, Reach]]:
        """Every unit the start sources reached, once, with the reach that puts it nearest: the earlier
        source's on a tie."""
        nearest: dict[str, tuple[Unit, Reach]] = {}
        for reaches in self._start_reaches():
            for unit, reach in self._units_reached(reaches):
                if unit.id not in nearest or reach.distance < nearest[unit.id][1].distance:
                    nearest[unit.id] = (unit, reach)
        return list(nearest.values())

    def _admitted_once(self, unit: Unit, reach: Reach) -> list[Item]:
        """``unit``'s places to judge, or none when its code repeats a unit already admitted."""
        first = self.first_with_content.setdefault(unit.content_sha256, unit.id)
        if first == unit.id:
            return self._admitted([(unit, reach)])
        self.units[unit.id] = unit
        self.entered_by[unit.id] = reach.source
        self.repeat_of[unit.id] = first
        return []

    def _units_reached(self, reaches: Sequence[Reach]) -> list[tuple[Unit, Reach]]:
        """The units ``reaches`` name, each with the reach that named it: every listed unit of a file,
        the units an anchor names."""
        if not self.policy.expands:
            files = [reach for reach in reaches if _is_file_reach(reach)]
            anchors = [reach for reach in reaches if not _is_file_reach(reach)]
            return self._units_of_files(files) + self._units_at_anchors(anchors)
        unseen = list(
            {reach.at: reach for reach in reversed(reaches) if reach.at not in self.resolved_units}.values()
        )
        unseen.reverse()
        files = [reach for reach in unseen if _is_file_reach(reach)]
        anchors = [reach for reach in unseen if not _is_file_reach(reach)]
        resolved = self._units_of_files(files) + self._units_at_anchors(anchors)
        by_place: dict[str | Anchor, list[Unit]] = {reach.at: [] for reach in unseen}
        for unit, reach in resolved:
            by_place[reach.at].append(unit)
        self.resolved_units.update((at, tuple(units)) for at, units in by_place.items())
        result = []
        admitted = set(self.units)
        for reach in reaches:
            units = self.resolved_units[reach.at]
            self._record_name_resolution(reach, not units)
            for unit in units:
                if unit.id not in admitted:
                    admitted.add(unit.id)
                    result.append((unit, reach))
        return result

    def _units_of_files(self, reaches: Sequence[Reach]) -> list[tuple[Unit, Reach]]:
        if not reaches:
            return []
        listing = self.reader.list_files([reach.at for reach in reaches])
        self.unlisted.update(listing.unlisted)
        reach_of: dict[str, Reach] = {}
        for reach in reaches:
            reach_of.setdefault(reach.at, reach)
        listed = {unit.path for unit in listing.units}
        for reach in reaches:
            self._record_name_resolution(reach, reach.at not in listed)
        units = listing.units
        if self.policy.search_order:
            order = {file: position for position, file in enumerate(reach_of)}
            units = sorted(units, key=lambda unit: order[unit.path])
        return [(unit, reach_of[unit.path]) for unit in units]

    def _record_name_resolution(self, reach: Reach, without_unit: bool) -> None:
        for name in reach.names:
            self.reached[name].add(reach.at)
            if without_unit:
                self.without_unit[name].add(reach.at)

    def _units_at_anchors(self, reaches: Sequence[Reach]) -> list[tuple[Unit, Reach]]:
        """The units each anchor names. An anchor reached by a name that names none counts against that
        name; any other is ``unresolved``."""
        anchors = [reach.at for reach in reaches]
        resolved = ((anchor, *self.reader.resolve(anchor)) for anchor in anchors)
        units = []
        for reach, (anchor, named, problem) in zip(reaches, resolved, strict=True):
            self._record_name_resolution(reach, bool(problem))
            if problem and not reach.names:
                self.unresolved.append(UnresolvedAnchor(anchor, problem))
            units += [(unit, reach) for unit in named]
        return units

    def _admitted(self, reached: Iterable[tuple[Unit, Reach]]) -> list[Item]:
        """The places still to judge of the units not seen before, each unit registered on first sight
        with the source that reached it."""
        places = []
        for unit, reach in reached:
            if unit.id not in self.units:
                self.units[unit.id] = unit
                self.entered_by[unit.id] = reach.source
                self.unit_of_place.update((place.id, unit.id) for place in items_to_judge(unit))
                places += self._pending_places(unit)
        return places

    def _pending_places(self, unit: Unit) -> list[Item]:
        if self.policy.expands and unit.ranges:
            delivered = self.delivered.get(unit.path, frozenset())
            if all(line in delivered for first, last in unit.ranges for line in range(first, last + 1)):
                self.delivered_units.add(unit.id)
        for piece in unit.too_large_pieces:
            self.not_judged[unit.piece_id(piece)] = TOO_LARGE
        pending = []
        for place in items_to_judge(unit):
            if self._already_delivered(place):
                self.not_judged[place.id] = DELIVERED
            elif self._answered(place):
                continue
            elif not self._fits_alone(place):
                self.not_judged[place.id] = TOO_LARGE
            else:
                self.not_judged[place.id] = NOT_REACHED
                pending.append(place)
        return pending

    def _fits_alone(self, place: Item) -> bool:
        """Whether the request asking every target about ``place`` alone fits the Judge's limits as
        masked: masking can lengthen code past the room measured before it."""
        [entry] = self._entries([place])
        return self.judge.fits_alone(self.checks, entry, self.shared, ITEMS)

    def _already_delivered(self, place: Item) -> bool:
        lines = self.delivered.get(place.file, frozenset())
        return all(line in lines for first, last in place.ranges for line in range(first, last + 1))

    def _judge(self, places: Sequence[Item]) -> None:
        if not places or self.stopped():
            return
        for name, answer in self.judge.iter_check_every(
            self.checks,
            self._entries(places),
            self.shared,
            list_name=ITEMS,
            cancelled=self.cancelled,
            places=places,
            refusals=self.refusals,
            keep_order=self.policy.ranked or self.policy.search_order,
        ):
            self._record(self.target_of[name], answer)

    def _record(self, target: str, answer: CheckResult) -> None:
        if answer.place is None:
            raise ValueError("a Find All answer names its place")
        if (target, answer.place.id) in self.answered:
            return
        self.answered.add((target, answer.place.id))
        self.judged[target].append(answer)
        self.found_one |= self.stops_when_found and answer.verdict is NoulVerdict.YES
        if self._answered(answer.place):
            self.not_judged.pop(answer.place.id, None)

    def _answered(self, place: Item) -> bool:
        return all((target, place.id) in self.answered for target in self.targets)

    def result(self, stop: str, failure: Exception | None) -> FindAllResult:
        refused = {
            refusal.place.id
            for refusal in self.refusals
            if refusal.place is not None and refusal.place.id in self.not_judged
        }
        not_judged = self.not_judged | dict.fromkeys(refused, REFUSED)
        return FindAllResult(
            self.targets,
            self.room,
            self.batches_per_wave,
            tuple(sorted(self.units.values(), key=_unit_order)),
            {target: tuple(sorted(answers, key=_answer_order)) for target, answers in self.judged.items()},
            not_judged | self._repeats_left(not_judged),
            dict(self.unlisted),
            tuple(self.unresolved),
            {
                name: NameHits(len(places), len(self.reached[name]), len(self.without_unit[name]))
                for name, places in self.found.items()
            },
            self.index.observed_unparsed_files,
            stop,
            self.judge.calls,
            failure,
            tuple(self.refusals),
            sources=self.sources + self.hops,
            entered_by=dict(self.entered_by),
            policy=self.policy,
            features={target: dict(features) for target, features in self.features.items() if features},
            repeat_of=dict(self.repeat_of),
            pushed=dict(self.pushed),
            settled=tuple(self.settled),
        )

    def _repeats_left(self, not_judged: Mapping[str, str]) -> dict[str, str]:
        """Each place of a repeat whose copy's place was left unjudged, with the copy's reason."""
        left = {}
        for repeat_id, first_id in self.repeat_of.items():
            places = zip(_place_ids(self.units[repeat_id]), _place_ids(self.units[first_id]), strict=True)
            left |= {repeat: not_judged[first] for repeat, first in places if first in not_judged}
        return left


def _room(judge: Judge, index: CodeIndex, checks: Sequence[Check], shared: Mapping) -> int:
    """The room one unit's code has in a request: the client's box less what a one-item request
    carries beside the code (the shared targets, the longest path in scope, and the longest of the
    item's questions), measured as the box measures it."""
    longest_path = max(index.files, key=serialized_chars, default="")
    beside = {**shared, ITEMS: [{"file": longest_path, "code": ""}]}
    longest_question = max(serialized_chars(check.to_question(item_path(ITEMS, 0))) for check in checks)
    return judge.input_limits.box_chars - serialized_chars(beside) - longest_question + serialized_chars("")


def _is_file_reach(reach: Reach) -> bool:
    return isinstance(reach.at, str)


def _lines_by_file(regions: Sequence[RangeAnchor]) -> dict[str, frozenset[int]]:
    lines: dict[str, set[int]] = {}
    for region in regions:
        lines.setdefault(region.file, set()).update(range(region.start, region.end + 1))
    return {file: frozenset(numbers) for file, numbers in lines.items()}


def _chunks(values: Iterable[T], size: int) -> Iterator[list[T]]:
    chunk: list[T] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _unit_score(
    unit: Unit, judged_as: Unit, answers: Mapping[str, CheckResult], probabilities: Mapping[str, float]
) -> UnitScore | None:
    """``unit``'s answer, read at the unit judged in its place: itself, or the copy it repeats."""
    if not judged_as.pieces:
        answer = answers.get(judged_as.id)
        return None if answer is None else UnitScore(unit, answer, None)
    piece = best_piece(judged_as, probabilities)
    if piece is None:
        return None
    return UnitScore(unit, answers[judged_as.piece_id(piece)], unit.pieces[piece.index])


def _place_ids(unit: Unit) -> list[str]:
    return [unit.id, *(unit.piece_id(piece) for piece in unit.pieces)]


def _unit_order(unit: Unit) -> tuple:
    return unit.path, unit.start, unit.end, unit.id


def _answer_order(answer: CheckResult) -> tuple:
    """Answers in file and line order, whatever order they arrived in."""
    place = answer.place
    return place.file, place.ranges[0][0], place.ranges[-1][1], place.id
