"""agent_search: one composed search a calling agent writes, run as a composition of JVN blocks.

The agent writes hypotheses, each with evidence points and refuting points, and the composition to
search with (``agent_search_request``). Each point becomes one target whose text sits in state, bound
into the admitted J1-3 question (``find_all.match_check``); a hypothesis's mechanism is never sent.
Every request asks every point about the same units, at most one request's worth (16) per round. A
model that takes fewer items per request (its route's ``ITEMS_PER_REQUEST``) is sent each round in
smaller requests. The budget counts requests of the judge's own size: a round, an existence request or a
label request costs one however many smaller requests carry it, so a capped model gets the same rounds.

Ranking says where to look; existence says whether the result answers the point:

1. **List.** The start sources (anchors, files, files the points name, and the hits of the request's
   terms plus the code names the points spell, and any extra source the caller passes) reach places.
   Each source's places are kept to the request's scope (anchors always) and handed to ``find_all``
   through ``sources.ReachedSource``, which lists and ranks the population under ``VALUE`` with no
   call allowed: every unit is admitted with its code features before anything is judged.
2. **Rank** (each round). ``frontier.Frontier`` draws one request's worth of units across the open
   points, each point's pushed hops first, then its own queue ordered by ``VALUE`` plus the relevance
   its likely units (J1-3 at 0.5 or more) spread over the code graph (``selection.reranked``, the step
   ``active_search`` re-ranks with). ``find_all`` judges exactly those units, pinned as anchors.
3. **Existence.** Each open point whose shortlist (its ``beam_width`` best units) changed is asked once
   whether the shortlist holds it (``existence.ask_existence``), all such points in one request over
   the union of their shortlists, split only when the union exceeds a request.
4. **Band.** High: the point is established and stops. Middle: its shortlist's callers, callees and
   named files (as the request's ``follow`` says) are listed and pushed to it; once they are judged, an
   unchanged shortlist stops the point (``stable_beam``). Low: it keeps drawing its own queue, and once
   nothing is left to judge for it, it is not found in this scope: not among the places the request's
   sources and hops reached, which ``coverage.units_reached`` counts, not the whole of ``scope``.

The search ends when every point has stopped, nothing is left to judge, or the budget is spent; the
budget counts every request, ranking, existence and role labels alike. Role labels
(``judgments.role_labels``) go last, to each point's shortlisted places at 0.5 or more, refuting points
first, while the budget lasts.

The search is written once, as steps (``Steps``) that yield each block call and each piece of
repository work. ``agent_search`` makes every call in place; ``agent_search_async`` awaits each
block's async form and runs the repository work in a worker thread. Both send the same requests.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from ..index.code_index import CodeIndex
from ..index.scope import Scope, kept_by_path
from ..index.units import (
    Anchor,
    Item,
    LineAnchor,
    RangeAnchor,
    Unit,
    UnresolvedAnchor,
    best_piece,
    items_to_judge,
    read_ranges,
)
from ..judgments.judge import CallCapReachedError, CheckResult, Judge
from ..judgments.known_values import with_known_values, with_known_values_async
from ..judgments.role_labels import LabelPiece, label_roles, label_roles_async
from ..mentions import code_names_in, names_from_text
from ..selection.active import normalized_scores, reranked
from ..selection.graph import CodeGraph, GraphEdge, graph_from_index
from ..sources import (
    ANCHORS,
    CALLEES,
    CALLERS,
    FILES,
    NAMED_FILES,
    NAMES,
    Reach,
    ReachedSource,
    Seeds,
    Source,
)
from .agent_search_request import AgentSearchRequest, SearchScope, parse_agent_search_request
from .agent_search_result import (
    AgentSearchResult,
    Band,
    Conflict,
    HypothesisOutcome,
    Outcome,
    PointOutcome,
    RankedPlace,
    RequestUse,
    SearchCoverage,
    UnmatchedAnchor,
    code_within,
)
from .existence import ExistenceAnswer, ask_existence, ask_existence_async, existence_fits
from .find_all import NOT_REACHED, FindAllResult, find_all, find_all_async
from .frontier import VALUE, Features, Frontier
from .search_coverage import PointResult, Round, point_results

START_SOURCES: tuple[Source, ...] = (ANCHORS, FILES, NAMED_FILES, NAMES)
FOLLOW_SOURCES: Mapping[str, Source] = {"callers": CALLERS, "callees": CALLEES, "named_files": NAMED_FILES}
POSSIBLE_AT = 0.5
"""A place at or above this J1-3 probability is a possible match: it spreads relevance, counts as a
possible file and is role-labelled when shortlisted."""
DEFAULT_BEAM_WIDTH = 4
DEFAULT_LABEL_REQUESTS = 2
DEFAULT_MAX_CODE_CHARS = 40_000
NEXT_PLACES = 8
EVIDENCE = "evidence"
REFUTING = "refuting"
ESTABLISHED = "established"
STABLE_BEAM = "stable_beam"
FRONTIER_EXHAUSTED = "frontier_exhausted"
POINTS_SETTLED = "points_settled"
BUDGET = "budget"
CANCELLED = "cancelled"
FAILED = "failed"
NOT_RESOLVED = "not resolved again when pinned"


@dataclass(frozen=True)
class Bands:
    """Where an existence probability counts as high (``high`` or more) or low (below ``low``);
    provisional defaults, reported with every result rather than tuned in place."""

    high: float = 0.7
    low: float = 0.35

    def __post_init__(self) -> None:
        if not 0 <= self.low < self.high <= 1:
            raise ValueError(f"bands need 0 <= low < high <= 1, got low {self.low} and high {self.high}")

    def of(self, probability: float) -> Band:
        if probability >= self.high:
            return Band.HIGH
        return Band.LOW if probability < self.low else Band.MIDDLE


def agent_search(
    request: AgentSearchRequest | Mapping[str, Any] | str | bytes,
    index: CodeIndex,
    judge: Judge,
    *,
    anchors: Sequence[Anchor] = (),
    files: Sequence[str] = (),
    extra_sources: Sequence[Source] = (),
    beam_width: int = DEFAULT_BEAM_WIDTH,
    bands: Bands = Bands(),  # noqa: B008 - a frozen value, never mutated
    label_requests: int = DEFAULT_LABEL_REQUESTS,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
    cancelled: Callable[[], bool] | None = None,
) -> AgentSearchResult:
    """Run one composed search for a calling agent and return each point's outcome.

    ``request`` is an ``AgentSearchRequest`` or its JSON; an invalid one raises
    ``InvalidAgentSearchRequestError`` naming the field. ``anchors`` and ``files`` are the caller's
    defaults, such as cited lines, searched beside the request's own. ``extra_sources`` start the search
    beside the built-in sources, kept to the scope like them. ``label_requests`` of the request's budget
    are kept for role labels while ranking runs; whatever ranking leaves goes to labels too. The
    judge's own call cap, store, masker and journal apply to every request."""
    options = _Options.of(beam_width, bands, label_requests, max_code_chars, extra_sources, cancelled)
    judge = with_known_values(judge, index)
    return _driven(_run(request, index, judge, options, anchors, files).steps())


async def agent_search_async(
    request: AgentSearchRequest | Mapping[str, Any] | str | bytes,
    index: CodeIndex,
    judge: Judge,
    *,
    anchors: Sequence[Anchor] = (),
    files: Sequence[str] = (),
    extra_sources: Sequence[Source] = (),
    beam_width: int = DEFAULT_BEAM_WIDTH,
    bands: Bands = Bands(),  # noqa: B008 - a frozen value, never mutated
    label_requests: int = DEFAULT_LABEL_REQUESTS,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
    cancelled: Callable[[], bool] | None = None,
) -> AgentSearchResult:
    """``agent_search`` through the Judge's async form, for an async client: the same steps send the
    same requests, each block through its async form (``find_all_async``, ``ask_existence_async``,
    ``label_roles_async``), and reaching into the repository runs in a worker thread, so the event loop
    stays free. ``cancelled`` is read between rounds as in ``agent_search``; a cancelled task's
    ``CancelledError`` is never caught."""
    options = _Options.of(beam_width, bands, label_requests, max_code_chars, extra_sources, cancelled)
    judge = await with_known_values_async(judge, index)
    return await _driven_async(_run(request, index, judge, options, anchors, files).steps())


def _run(
    request: AgentSearchRequest | Mapping[str, Any] | str | bytes,
    index: CodeIndex,
    judge: Judge,
    options: _Options,
    anchors: Sequence[Anchor],
    files: Sequence[str],
) -> _Run:
    return _Run(parse_agent_search_request(request), index, judge, options, tuple(anchors), tuple(files))


T = TypeVar("T")


@dataclass(frozen=True)
class _Call:
    """A step the search hands to the form running it: a block that may send requests, or work that
    reaches into the repository. ``agent_search`` runs ``sync`` in place; ``agent_search_async`` awaits
    ``in_loop``, or runs ``sync`` in a worker thread when there is none."""

    sync: Callable[..., Any]
    in_loop: Callable[..., Awaitable[Any]] | None
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]

    def run(self) -> Any:
        return self.sync(*self.args, **self.kwargs)

    async def run_async(self) -> Any:
        if self.in_loop is None:
            return await asyncio.to_thread(self.sync, *self.args, **self.kwargs)
        return await self.in_loop(*self.args, **self.kwargs)


Steps = Generator[_Call, Any, T]
"""The search as one sequence of steps: each yields a ``_Call`` and is sent back what it returned, or
has the error it raised thrown in, so both forms run the same search."""


def _block(sync: Callable[..., T], in_loop: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> _Call:
    return _Call(sync, in_loop, args, kwargs)


def _work(sync: Callable[..., Any], *args: Any, **kwargs: Any) -> _Call:
    return _Call(sync, None, args, kwargs)


def _driven(steps: Steps[T]) -> T:
    """Run ``steps`` with every call made in place."""
    outcome: tuple[Any, BaseException | None] = (None, None)
    while True:
        try:
            call = _resumed(steps, *outcome)
        except StopIteration as done:
            return done.value
        try:
            outcome = (call.run(), None)
        except (KeyboardInterrupt, Exception) as error:  # noqa: BLE001 - thrown back into the step that made the call
            outcome = (None, error)


async def _driven_async(steps: Steps[T]) -> T:
    """Run ``steps`` with every call awaited; a ``CancelledError`` is never thrown back in."""
    outcome: tuple[Any, BaseException | None] = (None, None)
    while True:
        try:
            call = _resumed(steps, *outcome)
        except StopIteration as done:
            return done.value
        try:
            outcome = (await call.run_async(), None)
        except (KeyboardInterrupt, Exception) as error:  # noqa: BLE001 - thrown back into the step that made the call
            outcome = (None, error)


def _resumed(steps: Steps[Any], sent: Any, raised: BaseException | None) -> _Call:
    return steps.send(sent) if raised is None else steps.throw(raised)


@dataclass(frozen=True)
class _Options:
    beam_width: int
    bands: Bands
    label_requests: int
    max_code_chars: int
    extra_sources: tuple[Source, ...]
    cancelled: Callable[[], bool] | None

    @classmethod
    def of(
        cls,
        beam_width: int,
        bands: Bands,
        label_requests: int,
        max_code_chars: int,
        extra_sources: Sequence[Source],
        cancelled: Callable[[], bool] | None,
    ) -> _Options:
        if beam_width < 1 or label_requests < 0 or max_code_chars < 0:
            raise ValueError(
                "beam_width must be at least 1 and label_requests and max_code_chars nonnegative"
            )
        return cls(beam_width, bands, label_requests, max_code_chars, tuple(extra_sources), cancelled)


@dataclass
class _Point:
    """A point's search state: ``key`` is its target name in state, ``id`` its name in the result."""

    key: str
    id: str
    hypothesis: str
    kind: str
    text: str
    band: Band | None = None
    existence: ExistenceAnswer | None = None
    asked_over: frozenset[str] = frozenset()
    unseen: frozenset[str] = frozenset()
    expanded_from: frozenset[str] | None = None
    awaiting: frozenset[str] = frozenset()
    hops: list[str] = field(default_factory=list)
    closed_by: str | None = None
    labels: str = "not labelled: the search stopped first"
    roles: dict[str, dict[str, float]] = field(default_factory=dict)


class _Stopped(Exception):  # noqa: N818 - a control-flow signal, not an error a caller sees
    def __init__(self, reason: str, failure: Exception | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.failure = failure


class _Run:
    def __init__(
        self,
        request: AgentSearchRequest,
        index: CodeIndex,
        judge: Judge,
        options: _Options,
        anchors: tuple[Anchor, ...],
        files: tuple[str, ...],
    ) -> None:
        self.request = request
        self.index = index
        self.options = options
        self.points = _points(request)
        self.targets = {point.key: point.text for point in self.points}
        self.sent = -(-judge.items_per_request // judge.items_per_sent_request())
        """The requests one round's items travel in: one, or more for a model that takes fewer items."""
        self.charged = 0
        """Sent requests added to round each step up to whole requests of the judge's size."""
        self.total = _capped(judge, request.budget_requests * self.sent)
        self.searching = _capped(
            self.total, max(1, request.budget_requests - options.label_requests) * self.sent
        )
        self.in_scope = _scope_rule(index, request.scope)
        self.anchors = (*anchors, *(_anchor(anchor) for anchor in request.anchors))
        self.files = tuple(dict.fromkeys((*files, *request.files)))
        self.names = _names(request, self.points)
        self.follow = tuple(FOLLOW_SOURCES[kind] for kind in request.follow)
        self.units: dict[str, Unit] = {}
        self.entered: dict[str, str] = {}
        self.labels_of: dict[str, str] = {}
        self.features: dict[str, dict[str, Features]] = {key: {} for key in self.targets}
        self.copy_of: dict[str, str] = {}
        self.by_content: dict[str, str] = {}
        self.queue: list[str] = []
        self.answers: dict[str, dict[str, CheckResult]] = {key: {} for key in self.targets}
        self.cut: dict[str, str] = {}
        self.rounds: list[Round] = []
        self.hops_of: dict[str, tuple[str, ...]] = {}
        self.graph = CodeGraph(())
        self.unlisted: dict[str, str] = {}
        self.unresolved: list[UnresolvedAnchor] = []
        self.outside_scope = tuple(file for file in self.files if not self.in_scope(file))
        self.calls: Counter[str] = Counter()
        self.stopped_by = ""
        self.failure: Exception | None = None

    def steps(self) -> Steps[AgentSearchResult]:
        """The whole search, then its role labels, then its result."""
        yield from self.search()
        yield from self.label()
        return self.result()

    # The search -------------------------------------------------------------------------------

    def search(self) -> Steps[None]:
        try:
            yield from self._start()
            self.stopped_by = yield from self._rounds()
        except _Stopped as stop:
            self.stopped_by, self.failure = stop.reason, stop.failure
        except CallCapReachedError:
            self.stopped_by = BUDGET
        except KeyboardInterrupt:
            self.stopped_by = CANCELLED
        except Exception as error:  # noqa: BLE001 - the result keeps every answer and names the failure
            self.stopped_by, self.failure = FAILED, error
        self._close_exhausted(at_end=self.stopped_by == FRONTIER_EXHAUSTED)

    def _start(self) -> Steps[None]:
        seeds = Seeds(
            self.names, tuple(self.targets.values()), self.files, self.anchors, in_scope=self.in_scope
        )
        sources = (*START_SOURCES, *self.options.extra_sources)
        reached: list[ReachedSource] = yield _work(self._reached, sources, seeds)
        listing = yield from self._list(reached)
        self.queue = [unit.id for unit in listing.units if unit.id not in self.copy_of]
        self.graph = yield _work(graph_from_index, self.index, listing.units, cochange_limit=0)

    def _reached(self, sources: Sequence[Source], seeds: Seeds) -> list[ReachedSource]:
        """Each source as the places it reaches from ``seeds``, kept to the scope."""
        return [ReachedSource.of(source, self._kept(source, seeds)) for source in sources]

    def _rounds(self) -> Steps[str]:
        while True:
            if self.options.cancelled is not None and self.options.cancelled():
                return CANCELLED
            if all(point.closed_by for point in self.points):
                return POINTS_SETTLED
            if self.searching.calls_left() == 0:
                return BUDGET
            wave = self._wave()
            if not wave:
                return FRONTIER_EXHAUSTED
            yield from self._judge(wave)
            self._reopen_unasked_matches()
            self._settle_expansions()
            yield from self._ask_existence()
            self._apply_bands()
            yield from self._expand()
            self._close_exhausted(at_end=False)

    def _kept(self, source: Source, seeds: Seeds) -> list[Reach]:
        """The places ``source`` reaches, kept to the scope; an anchor the caller named is always kept."""
        reaches = source.reach(self.index, seeds)
        return list(reaches) if source is ANCHORS else [r for r in reaches if self.in_scope(_file_of(r))]

    def _list(self, sources: Sequence[Source]) -> Steps[FindAllResult]:
        """The units ``sources`` reach, listed and ranked by ``find_all`` under VALUE with no call
        allowed; any answer the store already holds for them is kept."""
        result = yield _block(
            find_all,
            find_all_async,
            self.index,
            _capped(self.searching, 0),
            self.targets,
            names=self.names,
            sources=sources,
            policy=VALUE,
            hops=(),
            batches_per_wave=1,
            cancelled=self.options.cancelled,
        )
        self._absorb(result)
        return result

    def _judge(self, wave: Sequence[Unit]) -> Steps[None]:
        """One request's worth of items asking every point about ``wave``, pinned through their first
        lines."""
        before = self.searching.calls
        result = yield _block(
            find_all,
            find_all_async,
            self.index,
            _capped(self.searching, self.sent),
            self.targets,
            anchors=[LineAnchor(unit.path, unit.ranges[0][0]) for unit in wave],
            sources=(ANCHORS,),
            hops=(),
            batches_per_wave=1,
            completed={key: tuple(answers.values()) for key, answers in self.answers.items()},
            cancelled=self.options.cancelled,
        )
        self.calls["ranking"] += self.searching.calls - before
        self._count_whole(self.searching.calls - before)
        progress = self._absorb(result)
        for unit in wave:
            if unit.id not in {judged.id for judged in result.units}:
                self.cut.update(dict.fromkeys((place.id for place in items_to_judge(unit)), NOT_RESOLVED))
                progress = True
        if result.stopped_by in (FAILED, CANCELLED):
            raise _Stopped(result.stopped_by, result.failure)
        if progress:
            return
        if self.searching.calls_left() == 0:
            raise _Stopped(BUDGET)
        raise RuntimeError(f"a ranking round judged none of {len(wave)} pinned units ({result.stopped_by})")

    def _absorb(self, result: FindAllResult) -> bool:
        """Take in a ``find_all`` round's units, answers, cuts and gaps; True when it judged or cut a
        place not judged or cut before."""
        before = len(self.cut) + sum(len(answers) for answers in self.answers.values())
        for unit in result.units:
            self._admit(unit, result.entered_by.get(unit.id, ANCHORS.name))
        for key, features in result.features.items():
            for unit_id, unit_features in features.items():
                self.features[key].setdefault(unit_id, unit_features)
        for source in result.sources:
            self.labels_of.setdefault(source.name, source.label)
        for key, answers in result.judged.items():
            for answer in answers:
                self.answers[key].setdefault(answer.place.id, answer)
        for place_id, reason in result.not_judged.items():
            if reason != NOT_REACHED:
                self.cut.setdefault(place_id, reason)
        self.unlisted.update(result.unlisted)
        self.unresolved.extend(gap for gap in result.unresolved if gap not in self.unresolved)
        self.rounds.append(Round(result))
        return len(self.cut) + sum(len(answers) for answers in self.answers.values()) > before

    def _admit(self, unit: Unit, source: str) -> None:
        if unit.id in self.units:
            return
        self.units[unit.id] = unit
        self.entered[unit.id] = source
        copy = self.by_content.setdefault(unit.content_sha256, unit.id)
        if copy != unit.id:
            self.copy_of[unit.id] = copy

    # Ranking ----------------------------------------------------------------------------------

    def _wave(self) -> list[Unit]:
        """One request's worth of units, drawn across the open points by ``Frontier``: each point's
        pushed hops first, then its own queue, both in re-ranked order."""
        open_points = [point for point in self.points if not point.closed_by]
        pending = {unit_id for unit_id in self.units if self._pending(unit_id)}
        queues = {
            point.key: self._items(point, [u for u in self.queue if u in pending]) for point in open_points
        }
        frontier = Frontier(queues, {})
        for point in open_points:
            frontier.push_hops(point.key, self._items(point, [u for u in point.hops if u in pending]))
        drawn = frontier.wave(self.searching.items_per_request, dict.fromkeys(queues, False))
        return self._within_one_request([self.units[item.id] for item in drawn])

    def _items(self, point: _Point, unit_ids: list[str]) -> list[Item]:
        """``unit_ids`` best first for ``point``: VALUE, plus the relevance its likely units spread."""
        values = {unit_id: self._value(point.key, unit_id) for unit_id in unit_ids}
        likely = {
            unit_id: answer.probability
            for unit_id, answer in self._ranked(point.key)
            if answer.probability >= POSSIBLE_AT
        }
        ordered, _ = reranked(unit_ids, normalized_scores(unit_ids, values), self.graph, likely)
        return [Item(unit_id, self.units[unit_id].path, self.units[unit_id].ranges) for unit_id in ordered]

    def _within_one_request(self, units: Sequence[Unit]) -> list[Unit]:
        """The drawn units whose open places fit one request's item count; the first always goes."""
        kept, places = [], 0
        for unit in units:
            open_places = sum(1 for place in items_to_judge(unit) if self._open(place.id))
            if kept and places + open_places > self.searching.items_per_request:
                break
            kept.append(unit)
            places += open_places
        return kept

    def _pending(self, unit_id: str) -> bool:
        return unit_id not in self.copy_of and any(
            self._open(place.id) for place in items_to_judge(self.units[unit_id])
        )

    def _open(self, place_id: str) -> bool:
        return place_id not in self.cut and not all(place_id in answers for answers in self.answers.values())

    def _value(self, key: str, unit_id: str) -> float:
        features = self.features[key].get(unit_id)
        return 0.0 if features is None else features.value(VALUE.weights)

    def _score(self, key: str, unit_id: str) -> CheckResult | None:
        """A unit's answer for one point: its own, or a cut unit's best judged piece's."""
        unit, answers = self.units[unit_id], self.answers[key]
        if not unit.pieces:
            return answers.get(unit.id)
        piece = best_piece(unit, {place_id: answer.probability for place_id, answer in answers.items()})
        return None if piece is None else answers[unit.piece_id(piece)]

    def _ranked(self, key: str) -> list[tuple[str, CheckResult]]:
        """Every judged unit's answer for ``key``, best first; a tie goes to the unit worth more by
        code, then to its content hash."""
        scored = [
            (unit_id, answer)
            for unit_id in self.units
            if unit_id not in self.copy_of and (answer := self._score(key, unit_id)) is not None
        ]
        return sorted(scored, key=lambda pair: self._rank_key(key, *pair))

    def _rank_key(self, key: str, unit_id: str, answer: CheckResult) -> tuple[float, float, str, str]:
        return -answer.probability, -self._value(key, unit_id), self.units[unit_id].content_sha256, unit_id

    def _beam(self, point: _Point) -> list[tuple[str, CheckResult]]:
        return self._ranked(point.key)[: self.options.beam_width]

    def _beam_ids(self, point: _Point) -> frozenset[str]:
        return frozenset(unit_id for unit_id, _ in self._beam(point))

    # Existence and bands ----------------------------------------------------------------------

    def _ask_existence(self) -> Steps[None]:
        """Ask each open point whose shortlist changed, in as few requests as the union allows."""
        asking = [
            point
            for point in self._open_points()
            if (beam := self._beam_ids(point)) and beam != point.asked_over
        ]
        for points, pieces in self._existence_groups(asking):
            if self.searching.calls_left() == 0:
                return
            before = self.searching.calls
            targets = {point.key: point.text for point in points}
            answers = yield _block(ask_existence, ask_existence_async, self.searching, targets, pieces)
            self.calls["existence"] += self.searching.calls - before
            self._count_whole(self.searching.calls - before)
            for point in points:
                point.existence = answers[point.key]
                point.asked_over = self._beam_ids(point)
                evicted = set(point.existence.evicted)
                point.unseen = frozenset(u for u, answer in self._beam(point) if answer.place.id in evicted)

    def _existence_groups(self, points: Sequence[_Point]) -> list[tuple[list[_Point], list[LabelPiece]]]:
        """Points packed in order into groups whose union of shortlisted pieces fits one request."""
        groups: list[tuple[list[_Point], list[LabelPiece]]] = []
        members: list[_Point] = []
        pieces: dict[str, LabelPiece] = {}
        for point in points:
            own = self._beam_pieces(point)
            merged = {**pieces, **own}
            if members and not self._fits(members + [point], merged):
                groups.append((members, self._weakest_first(members, pieces)))
                members, merged = [], own
            members.append(point)
            pieces = merged
        if members:
            groups.append((members, self._weakest_first(members, pieces)))
        return groups

    def _weakest_first(self, points: Sequence[_Point], pieces: Mapping[str, LabelPiece]) -> list[LabelPiece]:
        """The pieces with the lowest best answer first, so a box that must evict code evicts the
        places least likely to hold any of the points; ties keep place order."""

        def best(piece: LabelPiece) -> float:
            return max(
                (
                    self.answers[point.key][piece.place.id].probability
                    for point in points
                    if piece.place.id in self.answers[point.key]
                ),
                default=0.0,
            )

        return sorted(_in_place_order(pieces), key=best)

    def _fits(self, points: Sequence[_Point], pieces: Mapping[str, LabelPiece]) -> bool:
        if len(pieces) > self.searching.items_per_request:
            return False
        targets = {point.key: point.text for point in points}
        return existence_fits(self.searching, targets, _in_place_order(pieces))

    def _beam_pieces(self, point: _Point) -> dict[str, LabelPiece]:
        return {answer.place.id: self._piece(answer.place) for _, answer in self._beam(point)}

    def _piece(self, place: Item) -> LabelPiece:
        """A place's raw code for a request Jev reads: the Judge masks the request as one, so a value
        one piece reveals is hidden in every other piece too."""
        return LabelPiece(place, read_ranges(self.index, place.file, place.ranges))

    def _apply_bands(self) -> None:
        for point in self._open_points():
            if point.existence is None:
                continue
            point.band = self.options.bands.of(point.existence.probability)
            if point.band is Band.HIGH:
                point.closed_by = ESTABLISHED

    # Expansion --------------------------------------------------------------------------------

    def _expand(self) -> Steps[None]:
        """Push the hops of each middle-band point's shortlist to it, listing hops not reached yet."""
        expanding = [p for p in self._open_points() if p.band is Band.MIDDLE and p.expanded_from is None]
        if not expanding or not self.follow:
            return
        seeds = list(dict.fromkeys(u for point in expanding for u in sorted(self._beam_ids(point))))
        yield from self._reach_hops([unit_id for unit_id in seeds if unit_id not in self.hops_of])
        for point in expanding:
            beam = self._beam_ids(point)
            hops = list(dict.fromkeys(h for u in sorted(beam) for h in self.hops_of[u] if self._pending(h)))
            if not hops:
                continue
            point.hops.extend(hop for hop in hops if hop not in point.hops)
            point.expanded_from, point.awaiting = beam, frozenset(hops)

    def _reach_hops(self, unit_ids: Sequence[str]) -> Steps[None]:
        """List what the ``follow`` sources reach from each unit, kept to the scope, and remember each
        unit's hops and their links in the graph."""
        reached: list[tuple[str, Source, Reach]] = yield _work(self._hop_reaches, unit_ids)
        listing = yield from self._list(
            [ReachedSource.of(s, [r for _, of, r in reached if of is s]) for s in self.follow]
        )
        by_file: dict[str, list[Unit]] = defaultdict(list)
        for unit in listing.units:
            by_file[unit.path].append(unit)
        hops: dict[str, list[str]] = {unit_id: [] for unit_id in unit_ids}
        for unit_id, source, reach in reached:
            for hop in _units_at(reach, by_file):
                if hop.id != unit_id and self.copy_of.get(hop.id, hop.id) != unit_id:
                    hops[unit_id].append(self.copy_of.get(hop.id, hop.id))
                    self.graph.add(GraphEdge(unit_id, hop.id, source.name))
        self.hops_of.update({unit_id: tuple(dict.fromkeys(found)) for unit_id, found in hops.items()})

    def _hop_reaches(self, unit_ids: Sequence[str]) -> list[tuple[str, Source, Reach]]:
        """Each unit's places reached by each ``follow`` source, kept to the scope."""
        reached: list[tuple[str, Source, Reach]] = []
        for unit_id in unit_ids:
            seeds = self._hop_seeds(self.units[unit_id])
            for source in self.follow:
                reached += [(unit_id, source, reach) for reach in self._kept(source, seeds)]
        return reached

    def _hop_seeds(self, unit: Unit) -> Seeds:
        """A unit as find_all's own hops seed it: its code's names, its code, its file and itself."""
        code = read_ranges(self.index, unit.path, unit.ranges)
        return Seeds(
            names=tuple(code_names_in(code)),
            texts=(code,),
            files=(unit.path,),
            units=(unit,),
            in_scope=self.in_scope,
        )

    def _settle_expansions(self) -> None:
        """Once an expansion's hops are all judged, an unchanged shortlist stops its point."""
        for point in self._open_points():
            if point.expanded_from is None or any(self._pending(hop) for hop in point.awaiting):
                continue
            if self._beam_ids(point) == point.expanded_from:
                point.closed_by = STABLE_BEAM
            point.expanded_from, point.awaiting = None, frozenset()

    def _reopen_unasked_matches(self) -> None:
        """A point closed as not found opens again once a place judged after its existence answer joins
        its shortlist at 0.5 or more, so the next existence request covers that place."""
        for point in self.points:
            if point.closed_by == FRONTIER_EXHAUSTED and self._unasked_match(point):
                point.closed_by = None

    def _unasked_match(self, point: _Point) -> bool:
        """Whether the point's shortlist holds a place at 0.5 or more its existence answer did not cover."""
        return any(
            answer.probability >= POSSIBLE_AT and (unit_id not in point.asked_over or unit_id in point.unseen)
            for unit_id, answer in self._beam(point)
        )

    def _close_exhausted(self, *, at_end: bool) -> None:
        """A low-band point with nothing left to judge is not found in this scope; at the end of a
        search that ran out of units, every open point stops for that reason."""
        pending = {unit_id for unit_id in self.units if self._pending(unit_id)}
        for point in self._open_points():
            own = pending & {*self.queue, *point.hops}
            if (point.band is Band.LOW and not own) or at_end:
                point.closed_by = FRONTIER_EXHAUSTED

    def _open_points(self) -> list[_Point]:
        return [point for point in self.points if not point.closed_by]

    # Role labels ------------------------------------------------------------------------------

    def label(self) -> Steps[None]:
        """Label each point's shortlisted places at 0.5 or more for that point, refuting points first,
        while the budget lasts; every point not labelled says why, a failed request included, so a
        labelling failure never costs the search's result."""
        for point in sorted(self.points, key=lambda point: point.kind != REFUTING):
            pieces = [
                self._piece(answer.place)
                for _, answer in self._beam(point)
                if answer.probability >= POSSIBLE_AT
            ]
            if not pieces:
                point.labels = "not labelled: no shortlisted place at 0.5 or more"
            elif self.stopped_by == CANCELLED or (
                self.options.cancelled is not None and self.options.cancelled()
            ):
                point.labels = "not labelled: the search was cancelled"
            elif self.total.calls_left() == 0:
                point.labels = "not labelled: budget"
            else:
                yield from self._label(point, pieces)

    def _label(self, point: _Point, pieces: Sequence[LabelPiece]) -> Steps[None]:
        before = self.total.calls
        try:
            labelled = yield _block(
                label_roles, label_roles_async, self.total, pieces, {point.key: point.text}
            )
        except CallCapReachedError:
            point.labels = "incomplete: the budget ran out inside this point's labelling request"
            return
        except Exception as error:  # noqa: BLE001 - the search's result stands; the point names the failure
            point.labels = f"not labelled: the request failed: {type(error).__name__}: {error}"
            return
        finally:
            self.calls["labels"] += self.total.calls - before
            self._count_whole(self.total.calls - before)
        point.roles = {piece.piece.place.id: piece.probabilities[point.key] for piece in labelled.pieces}
        refused = len(labelled.refusals)
        point.labels = "labelled" if not refused else f"labelled; {refused} place(s) refused, roles unknown"

    # The result -------------------------------------------------------------------------------

    def result(self) -> AgentSearchResult:
        rounds = self.rounds or [Round(FindAllResult.not_started(self.targets, self.stopped_by))]
        coverage = {result.point: result for result in point_results(rounds, self._yes_bar())}
        outcomes = {point.key: self._outcome(point, coverage[point.key]) for point in self.points}
        hypotheses = tuple(
            HypothesisOutcome(
                hypothesis.id,
                hypothesis.mechanism,
                tuple(outcomes[point.key] for point in self.points if point.hypothesis == hypothesis.id),
            )
            for hypothesis in self.request.hypotheses
        )
        conflicts = self._conflicts()
        code = code_within(conflicts, list(outcomes.values()), self._code, self.options.max_code_chars)
        return AgentSearchResult(
            hypotheses,
            conflicts,
            code,
            self.stopped_by,
            self._requests(),
            self._coverage(),
            self._yes_bar(),
            self.failure,
        )

    def _code(self, place: Item) -> str:
        """The code Jev judged at ``place``, masked as it was sent: raw source never leaves the search."""
        for answers in self.answers.values():
            if place.id in answers:
                return answers[place.id].item["code"]
        raise KeyError(f"{place.id} was never judged, so there is no masked code to show for it")

    def _outcome(self, point: _Point, coverage: PointResult) -> PointOutcome:
        ranked = [self._ranked_place(point, unit_id, answer) for unit_id, answer in self._ranked(point.key)]
        width = self.options.beam_width
        definite, possible = self._files(point)
        return PointOutcome(
            point.id,
            point.hypothesis,
            point.kind,
            point.text,
            _outcome_of(point, self._unasked_match(point) or bool(definite)),
            point.closed_by or self.stopped_by,
            point.band,
            point.existence,
            tuple(ranked[:width]),
            tuple(ranked[width : width + NEXT_PLACES]),
            definite,
            possible,
            point.labels,
            replace(coverage, point=point.id),
        )

    def _ranked_place(self, point: _Point, unit_id: str, answer: CheckResult) -> RankedPlace:
        roles = point.roles.get(answer.place.id)
        unit = self.units[unit_id]
        return RankedPlace(
            answer.place, unit.symbol, answer.probability, answer.request_sha256, answer.from_store, roles
        )

    def _files(self, point: _Point) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Distinct files holding a definite place for the point, and other files holding a possible
        one; a unit repeating a judged unit's code counts in its own file too."""
        copies: dict[str, list[str]] = defaultdict(list)
        for repeat, copy in self.copy_of.items():
            copies[copy].append(self.units[repeat].path)
        definite: dict[str, None] = {}
        possible: dict[str, None] = {}
        for unit_id, answer in self._ranked(point.key):
            files = [self.units[unit_id].path, *copies[unit_id]]
            if answer.probability >= self._yes_bar():
                definite.update(dict.fromkeys(files))
            elif answer.probability >= POSSIBLE_AT:
                possible.update(dict.fromkeys(files))
        return tuple(sorted(definite)), tuple(sorted(set(possible) - set(definite)))

    def _conflicts(self) -> tuple[Conflict, ...]:
        conflicts = [
            Conflict(point.id, self._ranked_place(point, unit_id, answer))
            for point in self.points
            if point.kind == REFUTING
            for unit_id, answer in self._ranked(point.key)
            if answer.probability >= self._yes_bar()
        ]
        return tuple(sorted(conflicts, key=lambda conflict: -conflict.place.probability))

    def _count_whole(self, requests: int) -> None:
        """Counts a step's ``requests`` as whole requests of the judge's size, rounded up: the caps count sent
        requests, ``self.sent`` to each whole one, so they drop by what the step left of its last one."""
        rest = -(-requests // self.sent) * self.sent - requests
        self.charged += rest
        for scope in (self.searching, self.total):
            scope.max_calls = max(scope.calls, scope.max_calls - rest)

    def _requests(self) -> RequestUse:
        return RequestUse(
            self.request.budget_requests,
            (self.total.calls + self.charged) // self.sent,
            self.calls["ranking"],
            self.calls["existence"],
            self.calls["labels"],
            self.total.replayed_answers,
        )

    def _coverage(self) -> SearchCoverage:
        reached = [unit_id for unit_id in self.units if unit_id not in self.copy_of]
        judged = [u for u in reached if any(self._score(key, u) for key in self.targets)]
        not_judged = Counter(self._why_not_judged(unit_id) for unit_id in reached if unit_id not in judged)
        evicted = tuple(
            dict.fromkeys(
                place for point in self.points if point.existence for place in point.existence.evicted
            )
        )
        return SearchCoverage(
            len(reached),
            len(judged),
            dict(not_judged),
            dict(self.unlisted),
            tuple(self.unresolved),
            self.outside_scope,
            evicted,
            self.request.scope,
            tuple(
                UnmatchedAnchor(unit_id, self.units[unit_id].symbol, best)
                for unit_id in reached
                if self.entered[unit_id] == ANCHORS.name
                and ((best := self._best(unit_id)) is None or best < POSSIBLE_AT)
            ),
        )

    def _best(self, unit_id: str) -> float | None:
        """A unit's best J1-3 probability over every point, or None when no point judged it."""
        scores = [
            answer.probability for key in self.targets if (answer := self._score(key, unit_id)) is not None
        ]
        return max(scores, default=None)

    def _why_not_judged(self, unit_id: str) -> str:
        if self._pending(unit_id):
            source = self.entered[unit_id]
            return f"not reached: {self.labels_of.get(source, source)}"
        reasons = {self.cut.get(place.id, NOT_REACHED) for place in items_to_judge(self.units[unit_id])}
        return next(iter(sorted(reasons))) if reasons else "no part small enough to judge"

    def _yes_bar(self) -> float:
        return self.total.thresholds.noul_yes_at


def _points(request: AgentSearchRequest) -> list[_Point]:
    return [
        _Point(f"{hypothesis.id}_{point.id}", f"{hypothesis.id}.{point.id}", hypothesis.id, kind, point.point)
        for hypothesis in request.hypotheses
        for kind, points in ((EVIDENCE, hypothesis.evidence), (REFUTING, hypothesis.refuted_by))
        for point in points
    ]


def _names(request: AgentSearchRequest, points: Iterable[_Point]) -> tuple[str, ...]:
    """The request's terms, then the code names its points spell, each once."""
    spelled = (name for point in points for name in names_from_text(point.text).code)
    return tuple(dict.fromkeys((*request.terms, *spelled)))


def _anchor(anchor) -> Anchor:
    if anchor.end is None:
        return LineAnchor(anchor.file, anchor.line)
    return RangeAnchor(anchor.file, anchor.line, anchor.end)


def _scope_rule(index: CodeIndex, scope: SearchScope) -> Callable[[str], bool]:
    """The request's scope as the scope owner's path rules over this index's files."""
    rules = Scope(
        repo=index.root,
        include=scope.include,
        exclude=scope.exclude,
        with_tests=scope.with_tests,
        with_generated=True,
        with_vendored=True,
        with_docs=False,
        max_files=max(1, len(index.files)),
    )
    return lambda path: kept_by_path(rules, path)


def _capped(judge: Judge, calls: int) -> Judge:
    """A scope of ``judge`` that may send at most ``calls`` requests, counted toward ``judge`` too."""
    scope = judge.scope()
    scope.max_calls = calls
    return scope


def _file_of(reach: Reach) -> str:
    return reach.at if isinstance(reach.at, str) else reach.at.file


def _units_at(reach: Reach, by_file: Mapping[str, Sequence[Unit]]) -> list[Unit]:
    """The listed units a reach names: every unit of a file, or the units holding an anchor's lines."""
    if isinstance(reach.at, str):
        return list(by_file.get(reach.at, ()))
    anchor = reach.at
    first, last = (anchor.line, anchor.line) if isinstance(anchor, LineAnchor) else (anchor.start, anchor.end)
    return [
        unit
        for unit in by_file.get(anchor.file, ())
        if any(start <= last and first <= end for start, end in unit.ranges)
    ]


def _in_place_order(pieces: Mapping[str, LabelPiece]) -> list[LabelPiece]:
    return sorted(pieces.values(), key=lambda piece: (piece.place.file, piece.place.ranges, piece.place.id))


def _outcome_of(point: _Point, open_match: bool) -> Outcome:
    """Not found only while the low existence answer covers every likely place of the shortlist and no
    place was judged a definite match: such a place contradicts "not found", so the point stays
    undecided and the agent reads both answers."""
    if point.closed_by == ESTABLISHED:
        return Outcome.ESTABLISHED
    if point.closed_by == FRONTIER_EXHAUSTED and point.band is Band.LOW and not open_match:
        return Outcome.NOT_FOUND_IN_SCOPE
    return Outcome.UNDECIDED
