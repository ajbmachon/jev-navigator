"""find_code: open places best first until the code a description names is found.

Each opened place gets one request with two kinds of yes/no question: does this code contain the
target, and, per neighbour code lists, could the target be inside that neighbour. An optional pick
names the neighbour to open first. Jev sees only the target description, the opened code and the
neighbour signatures; it never sees the search history, and code alone decides what happens next.

Each round opens ``beam_width`` places at once: start places first, then the neighbours Jev picked
to open next, in the order it picked them, whatever its confidence, then the other neighbours by
falling could-contain probability. A start place is judged but never ends the search as found, since
the caller already had it. A visited set removes overlapping paths, and an in-run cache skips code
already judged. With ``beam_width=1`` this is a plain sequential best-first search.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import os
import signal
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum

from ..history import (
    DEFAULT_QUESTION_RESERVE,
    DEFAULT_STOP_SECTIONS,
    JEV_STATE_TOKEN_LIMIT,
    FetchedSpan,
    History,
    HistoryJudgment,
    HistoryOutcome,
    HistoryStep,
    judge_history,
    judge_history_async,
)
from ..index.code_index import CodeIndex
from ..index.spans import CodeSlice
from ..judgments.judge import CallCapReachedError, Judge
from ..judgments.questions import Check, Criterion, Pick, content_hash
from ..judgments.thresholds import NoulVerdict, Thresholds
from .places import MOVES, Move, Place, neighbours_and_omissions
from .shown import MAX_LINE_CHARS, MAX_SLICE_CHARS, cut_long_line, shown_slice

FOUND = Check(
    name="contains_target",
    instructions="Does `slice.code` contain the code described in `target.description`?",
    yes=Criterion(
        "A line or block in `slice.code` is itself the code the description names.",
        examples=(
            "Description 'the check that limits items per order' and the slice holds "
            "`if len(order.items) > limit: raise`.",
        ),
    ),
    no=Criterion(
        "`slice.code` only calls, mentions or sits near that code, or does something else entirely.",
        not_for="Code that merely has a similar name.",
        examples=(
            "The slice calls `check_limits(order)` but the comparison is inside `check_limits`.",
            "The slice formats an invoice.",
        ),
    ),
)
COULD_CONTAIN = Check(
    name="could_contain_target",
    instructions=(
        "Does `{item}.preview`, under the signature `{item}.signature`, suggest that this function itself"
        " implements the code described in `target.description`?"
    ),
    yes=Criterion(
        "The signature and preview suggest this function itself implements the description.",
        examples=(
            "Description 'the check that limits items per order' and the candidate is "
            "`def check_limits(order):` reading a limit setting.",
        ),
    ),
    no=Criterion(
        "The function is unrelated, or it only uses or wraps code that implements the description.",
        examples=(
            "A route handler that calls `place_order`, when the description is the item limit check.",
            "A logging helper.",
        ),
    ),
)
NO_CLEAR_FIRST = "none"
OPEN_FIRST = Pick(
    name="open_first",
    instructions=(
        "Which entry of `candidates` most likely contains the code described in `target.description`?"
    ),
    extra_options=((NO_CLEAR_FIRST, "None of the entries is likely to contain it."),),
)


@dataclass(frozen=True)
class SearchQuestions:
    """The wording find_code asks with. Replace any part; a new wording is a new question id.
    ``open_first`` gets the candidates as options named by their position; its own no-match option,
    if any, must be named ``none``."""

    found: Check = FOUND
    could_contain: Check = COULD_CONTAIN
    open_first: Pick | None = OPEN_FIRST


DEFAULT_SEARCH_QUESTIONS = SearchQuestions()


class Outcome(StrEnum):
    FOUND = "found"
    STOP_RULE = "stop_rule"
    CANCELLED = "cancelled"
    UNSURE_ONLY = "unsure_only"
    NOTHING_LEFT = "nothing_left"
    SCOPE_INCOMPLETE = "scope_incomplete"
    BUDGET = "budget"


@dataclass(frozen=True)
class SearchBudget:
    """Optional caller-selected search limits. ``None`` means the finite frontier, rather than an
    arbitrary library default, decides when the search is complete."""

    max_depth: int | None = None
    max_steps: int | None = None
    max_calls: int | None = None
    beam_width: int = 3
    neighbours_per_kind: int | None = None
    preview_lines: int = 8
    max_slice_chars: int = MAX_SLICE_CHARS
    max_line_chars: int = MAX_LINE_CHARS

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> SearchBudget:
        """Library defaults, overridden by ``JEV_NAVIGATOR_<FIELD>`` variables; read once at the edge."""
        environment = os.environ if environment is None else environment
        found = {
            name: int(environment[f"JEV_NAVIGATOR_{name.upper()}"])
            for name in (
                "max_depth",
                "max_steps",
                "max_calls",
                "beam_width",
                "neighbours_per_kind",
                "preview_lines",
                "max_slice_chars",
                "max_line_chars",
            )
            if f"JEV_NAVIGATOR_{name.upper()}" in environment
        }
        return replace(cls(), **found)


@dataclass(frozen=True)
class Visit:
    """An opened place: the code the request showed of it (cut at ``SearchBudget.max_slice_chars``
    on a line boundary, so ``code.span`` ends at the last shown line), the path from a start place,
    and the found verdict."""

    place_key: str
    code: CodeSlice
    path: tuple[str, ...]
    probability: float
    verdict: NoulVerdict


class QueueTier(IntEnum):
    """Search priority and provenance, preserved through a budget interruption."""

    START = 0
    PICK = 1
    DISCOVERED = 2
    MOVE = 3


@dataclass(frozen=True)
class NotInspected:
    """A place the search did not open. ``reason`` is ``budget`` (still worth opening when the budget
    ran out), ``cancelled`` (the caller interrupted before it was opened), ``deprioritized`` (its
    signature scored low; that only lowered its priority, it was never judged), ``capped`` (cut by
    an explicit per-kind neighbour cap) or ``depth`` (beyond an explicit depth limit).
    ``tier`` preserves starts, picked places and scored neighbours through Resume."""

    place_key: str
    signature: str
    reason: str
    priority: float
    depth: int
    path: tuple[str, ...]
    place: Place = field(compare=False, repr=False)
    tier: QueueTier = QueueTier.MOVE


@dataclass(frozen=True)
class FindResult:
    """Three explicit sets: ``found``; ``searched`` (opened and judged at or below the no bar, each
    with its probability) plus ``unsure``; and ``not_inspected``, the frontier a later call can resume
    from with ``resume=``. Neither ``searched`` nor the outcome ``nothing_left`` proves the code is
    absent: one "no" about one place can be wrong. When nothing is found, rank the opened places by
    their probability; the best one is the likeliest place. ``starts`` holds the start places with
    their verdicts: a start is never a find, because the caller already had it. ``unparsed_files``
    lists scope files the index could not parse; while it is not empty the outcome is never
    ``nothing_left``."""

    outcome: Outcome
    found: tuple[Visit, ...]
    searched: tuple[Visit, ...]
    unsure: tuple[Visit, ...]
    not_inspected: tuple[NotInspected, ...]
    steps: int
    calls: int
    visited: frozenset[str] = frozenset()
    judged_code: frozenset[str] = frozenset()
    history: History | None = None
    stop_judgment: HistoryJudgment | None = None
    unparsed_files: frozenset[str] = frozenset()
    moves: tuple[str, ...] = ()
    starts: tuple[Visit, ...] = ()
    parser_scans_completed: tuple[str, ...] = ()
    parser_scans_pending: tuple[str, ...] = ()
    unavailable_files: Mapping[str, str] = field(default_factory=dict)


@dataclass(order=True)
class _Queued:
    """Starts come first, then picks in the order they were made, then moves by falling probability."""

    tier: QueueTier
    rank: float
    order: int
    place: Place = field(compare=False)
    depth: int = field(compare=False)
    path: tuple[str, ...] = field(compare=False)
    probability: float = field(compare=False)


@dataclass
class _Search:
    target: Mapping
    thresholds: Thresholds
    budget: SearchBudget
    questions: SearchQuestions = DEFAULT_SEARCH_QUESTIONS
    stop_rule: StopRule | None = None
    history: History = field(default_factory=History)
    moves: Mapping[str, Move] = field(default_factory=lambda: MOVES)
    stop_judgment: HistoryJudgment | None = None
    queue: list[_Queued] = field(default_factory=list)
    visited: set[str] = field(default_factory=set)
    judged_code: set[str] = field(default_factory=set)
    found: list[Visit] = field(default_factory=list)
    searched: list[Visit] = field(default_factory=list)
    unsure: list[Visit] = field(default_factory=list)
    starts: list[Visit] = field(default_factory=list)
    set_aside: list[NotInspected] = field(default_factory=list)
    steps: int = 0
    cap_reached: bool = False
    counter: itertools.count = field(default_factory=itertools.count)

    def push(
        self,
        place: Place,
        probability: float,
        depth: int,
        path: tuple[str, ...],
        tier: QueueTier = QueueTier.MOVE,
    ) -> None:
        if place.key in self.visited:
            return
        if self.budget.max_depth is not None and depth > self.budget.max_depth:
            self.set_aside.append(
                NotInspected(place.key, place.signature, "depth", probability, depth, path, place, tier)
            )
            return
        rank = -probability if tier in (QueueTier.DISCOVERED, QueueTier.MOVE) else 0.0
        heapq.heappush(self.queue, _Queued(tier, rank, next(self.counter), place, depth, path, probability))

    def next_beam(self, calls_left: int | None) -> list[_Queued]:
        beam = []
        # A spent live-call budget still permits answers already in the store.
        width = self.budget.beam_width if calls_left in (None, 0) else min(self.budget.beam_width, calls_left)
        while self.queue and len(beam) < width:
            item = heapq.heappop(self.queue)
            if item.place.key not in self.visited:
                self.visited.add(item.place.key)
                beam.append(item)
        return beam

    def worth_opening(self) -> bool:
        return any(
            self.still_worth_opening(item) for item in self.queue if item.place.key not in self.visited
        )

    def still_worth_opening(self, item: _Queued) -> bool:
        """A waiting start or pick always is; a neighbour only when scored above the no bar."""
        return item.tier != QueueTier.MOVE or item.probability > self.thresholds.noul_no_at


def find_code(
    index: CodeIndex,
    judge: Judge,
    target_description: str,
    start: Sequence[Place],
    *,
    budget: SearchBudget | None = None,
    thresholds: Mapping[str, float] | None = None,
    questions: SearchQuestions = DEFAULT_SEARCH_QUESTIONS,
    resume: FindResult | None = None,
    commit: str | None = None,
    stop_rule: StopRule | None = None,
    moves: Mapping[str, Move] | None = None,
    initial_candidates: Sequence[tuple[Place, float]] = (),
) -> FindResult:
    """``commit``: the revision the caller means; the index must hold exactly it. ``resume``: continue
    a stopped search from its frontier with a fresh budget. ``stop_rule`` (off by default): after each
    round the caller's check is asked over the history; a yes ends the search with outcome
    ``stop_rule``. ``initial_candidates`` are system-discovered places: the first is opened as a
    picked place and may be found, while the rest keep their supplied probabilities in the ordinary
    move frontier. Explicit ``start`` places retain their caller-known semantics and never count as
    finds. ``moves`` chooses how neighbours are listed (default ``places.MOVES``); pass a
    subset, or add a move of your own. Each round's places are asked concurrently in threads;
    ``find_code_async`` is the same search for an async client."""
    options = _SearchOptions(
        budget, thresholds, questions, resume, commit, stop_rule, moves, initial_candidates
    )
    search, judge = _begin(index, judge, target_description, start, options)
    stop = None
    try:
        while (stop := _stop_reason(search, index)) is None:
            opened = _open_round(index, search, judge)
            if not opened:
                continue
            responses, cancelled = _ask_round(judge, search, opened)
            with _defer_keyboard_interrupts():
                _merge_round(search, opened, responses)
            if cancelled:
                stop = Outcome.CANCELLED
                break
            _apply_stop_rule(judge, search)
    except KeyboardInterrupt:
        stop = Outcome.CANCELLED
    assert stop is not None
    return _result(search, stop, judge, index)


async def find_code_async(
    index: CodeIndex,
    judge: Judge,
    target_description: str,
    start: Sequence[Place],
    *,
    budget: SearchBudget | None = None,
    thresholds: Mapping[str, float] | None = None,
    questions: SearchQuestions = DEFAULT_SEARCH_QUESTIONS,
    resume: FindResult | None = None,
    commit: str | None = None,
    stop_rule: StopRule | None = None,
    moves: Mapping[str, Move] | None = None,
    initial_candidates: Sequence[tuple[Place, float]] = (),
) -> FindResult:
    """``find_code`` with each round's places sent concurrently with ``asyncio.gather``; budgets,
    masking, the store, the journal and the history work exactly as in ``find_code``."""
    options = _SearchOptions(
        budget, thresholds, questions, resume, commit, stop_rule, moves, initial_candidates
    )
    search, judge = _begin(index, judge, target_description, start, options)
    while (stop := _stop_reason(search, index)) is None:
        opened = _open_round(index, search, judge)
        if not opened:
            continue
        responses = await asyncio.gather(
            *(_ask_within_cap_async(judge, search, opening) for opening in opened)
        )
        _merge_round(search, opened, responses)
        await _apply_stop_rule_async(judge, search)
    return _result(search, stop, judge, index)


@dataclass(frozen=True)
class _SearchOptions:
    """The keyword arguments ``find_code`` and ``find_code_async`` share."""

    budget: SearchBudget | None
    thresholds: Mapping[str, float] | None
    questions: SearchQuestions
    resume: FindResult | None
    commit: str | None
    stop_rule: StopRule | None
    moves: Mapping[str, Move] | None
    initial_candidates: Sequence[tuple[Place, float]]


def _begin(
    index: CodeIndex, judge: Judge, target_description: str, start: Sequence[Place], options: _SearchOptions
) -> tuple[_Search, Judge]:
    if options.commit is not None:
        index.require_commit(options.commit)
    target = {"description": target_description}
    rule = options.stop_rule
    history = rule.new_history(target) if rule else History(sections={SUBJECT: target})
    if judge.journal is not None and hasattr(judge.journal, "record_step"):
        history.recorder = judge.journal
    search = _Search(
        target,
        judge.effective(options.thresholds),
        options.budget or SearchBudget(),
        options.questions,
        rule,
        history,
        MOVES if options.moves is None else options.moves,
    )
    if options.resume is not None:
        _restore(search, options.resume)
    for place in start:
        search.push(place, 1.0, 0, (place.key,), QueueTier.START)
    for position, (place, probability) in enumerate(options.initial_candidates):
        tier = QueueTier.PICK if position == 0 else QueueTier.DISCOVERED
        search.push(place, probability, 0, (place.key,), tier)
    scoped_judge = judge.scope()
    scoped_judge.max_calls = search.budget.max_calls
    return search, scoped_judge


def _open_round(index: CodeIndex, search: _Search, judge: Judge) -> list[_Opening]:
    beam = []
    opened = []
    processed = 0
    try:
        with _defer_keyboard_interrupts():
            beam = search.next_beam(judge.calls_left())
        _record_choice(search, beam)
        for item in beam:
            if opening := _open(index, search, item):
                opened.append(opening)
            processed += 1
        return opened
    except BaseException:
        for opening in opened:
            _restore_opening(search, opening)
            heapq.heappush(search.queue, opening.item)
        for item in beam[processed:]:
            search.visited.discard(item.place.key)
            heapq.heappush(search.queue, item)
        raise


def _ask_round(judge: Judge, search: _Search, opened: list[_Opening]) -> tuple[list, bool]:
    """Ask one beam concurrently. A caller interrupt stops future rounds after the already-sent
    requests settle; successful responses still count and interrupted places return to the frontier."""
    with ThreadPoolExecutor(max_workers=len(opened)) as pool:
        futures = [pool.submit(_ask_within_cap, judge, search, opening) for opening in opened]
        try:
            return [future.result() for future in futures], False
        except KeyboardInterrupt:
            with _defer_keyboard_interrupts(re_raise=False):
                judge.cancel()
                for future in futures:
                    future.cancel()
                wait(futures)
                responses = []
                for future in futures:
                    try:
                        responses.append(future.result())
                    except (CancelledError, KeyboardInterrupt):
                        responses.append(_Unanswered.CANCELLED)
                    except Exception:
                        responses.append(_Unanswered.CANCELLED)
            return responses, True


def _merge_round(search: _Search, opened: list[_Opening], responses: list) -> None:
    for opening, response in zip(opened, responses, strict=True):
        if response is _Unanswered.CANCELLED:
            _set_aside_unasked(search, opening, "cancelled")
        elif response is _Unanswered.BUDGET:
            _set_aside_unasked(search, opening, "budget")
        else:
            _merge(search, opening, response)


class _Unanswered(StrEnum):
    BUDGET = "budget"
    CANCELLED = "cancelled"


@contextmanager
def _defer_keyboard_interrupts(*, re_raise: bool = True):
    """Keep the small receipt commit indivisible on the main thread.

    A first interrupt is delivered after a normal merge has preserved its completed responses. Once
    cancellation has begun, later interrupts are coalesced while the owned transport settles.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    interrupted = False

    def defer(signum, frame) -> None:
        del signum, frame
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if interrupted and re_raise:
        raise KeyboardInterrupt


SUBJECT = "subject"


@dataclass(frozen=True)
class StopRule:
    """A caller-defined yes/no check over the history, asked after each round. It reads only the
    ``sections`` it selects: by default ``fetched``, the code opened so far with its sources and none
    of the search's own verdicts; ``history`` adds each step's operation and arguments, and
    ``decisions`` holds the verdicts for a check meant to read them. The history declares ``subject``
    (the target description) and any ``context`` sections the caller adds, such as the code shown
    with a comment; ``shared`` is extra state outside the history."""

    check: Check
    shared: Mapping = field(default_factory=dict)
    budget_tokens: int = JEV_STATE_TOKEN_LIMIT - DEFAULT_QUESTION_RESERVE
    sections: tuple[str, ...] = DEFAULT_STOP_SECTIONS
    context: Mapping[str, object] = field(default_factory=dict)

    def new_history(self, subject: Mapping) -> History:
        return History(budget_tokens=self.budget_tokens, sections={SUBJECT: subject, **self.context})


def _apply_stop_rule(judge: Judge, search: _Search) -> None:
    rule = search.stop_rule
    if rule is None:
        return
    try:
        search.stop_judgment = judge_history(
            judge, search.history, rule.check, rule.shared, sections=rule.sections
        )
    except CallCapReachedError:
        search.cap_reached = True


async def _apply_stop_rule_async(judge: Judge, search: _Search) -> None:
    rule = search.stop_rule
    if rule is None:
        return
    try:
        search.stop_judgment = await judge_history_async(
            judge, search.history, rule.check, rule.shared, sections=rule.sections
        )
    except CallCapReachedError:
        search.cap_reached = True


def _restore(search: _Search, previous: FindResult) -> None:
    search.visited |= previous.visited
    search.judged_code |= previous.judged_code
    search.searched += previous.searched
    search.unsure += previous.unsure
    search.starts += previous.starts
    for entry in previous.not_inspected:
        search.push(entry.place, entry.priority, entry.depth, entry.path, entry.tier)


def _stop_reason(search: _Search, index: CodeIndex) -> Outcome | None:
    if search.found:
        return Outcome.FOUND
    if search.stop_judgment is not None and search.stop_judgment.outcome == HistoryOutcome.FOUND:
        return Outcome.STOP_RULE
    steps_used = search.budget.max_steps is not None and search.steps >= search.budget.max_steps
    if steps_used:
        return Outcome.BUDGET
    if not search.worth_opening():
        if search.cap_reached:
            return Outcome.BUDGET
        return _nothing_worth_opening(search, index)
    return None


def _nothing_worth_opening(search: _Search, index: CodeIndex) -> Outcome:
    """``nothing_left`` only when every scope file was parsed; code in an unparsed file was never
    offered, so the search cannot say it looked everywhere."""
    if search.unsure:
        return Outcome.UNSURE_ONLY
    return (
        Outcome.SCOPE_INCOMPLETE if index.unparsed_files or index.unavailable_files else Outcome.NOTHING_LEFT
    )


@dataclass(frozen=True)
class _Opening:
    """``code`` is what the request shows of the place; ``opened_key`` names all of it."""

    item: _Queued
    code: CodeSlice
    opened_key: str
    fingerprint: str
    candidates: list[Place]
    capped: tuple[NotInspected, ...] = ()


def _open(index: CodeIndex, search: _Search, item: _Queued) -> _Opening | None:
    """Opens a place in code; the same code reached by another path is not judged twice. Moves list
    neighbours from all of the code, and only lines the request does not show can hold a new place."""
    code = item.place.open()
    fingerprint = content_hash(code.text)
    if fingerprint in search.judged_code:
        return None
    shown = shown_slice(code, search.budget.max_slice_chars, search.budget.max_line_chars)
    if shown is None:
        search.visited.discard(item.place.key)
        search.cap_reached = True
        _set_aside_for_budget(search, item)
        return None
    search.judged_code.add(fingerprint)
    search.visited.add(code.key)
    search.steps += 1
    set_aside_before = len(search.set_aside)
    try:
        if search.budget.max_depth is not None and item.depth >= search.budget.max_depth:
            return _Opening(item, shown, code.key, fingerprint, [])
        candidates, omitted = neighbours_and_omissions(
            index, code, search.budget.neighbours_per_kind, search.moves, shown.span
        )
        capped = tuple(
            NotInspected(
                place.key,
                place.signature,
                "capped",
                0.5,
                item.depth + 1,
                (*item.path, place.key),
                place,
            )
            for place in omitted
            if place.key not in search.visited
        )
        search.set_aside.extend(capped)
        unseen = [place for place in candidates if place.key not in search.visited]
        available = [place for place in unseen if place.open().text.strip()]
        return _Opening(item, shown, code.key, fingerprint, available, capped)
    except BaseException:
        del search.set_aside[set_aside_before:]
        search.judged_code.discard(fingerprint)
        search.visited -= {item.place.key, code.key}
        search.steps -= 1
        raise


@dataclass(frozen=True)
class _OpeningRequest:
    state: Mapping
    questions: Mapping
    sources: Mapping


def _opening_request(search: _Search, opening: _Opening) -> _OpeningRequest:
    code, candidates = opening.code, opening.candidates
    state = {
        "target": search.target,
        "slice": {"file": code.span.file, "lines": f"{code.span.start}-{code.span.end}", "code": code.text},
        "candidates": [_candidate_state(place, search.budget) for place in candidates],
    }
    asked = search.questions
    questions = {asked.found.question_id: asked.found.to_question()}
    for slot in range(len(candidates)):
        questions[f"{asked.could_contain.question_id}#{slot}"] = asked.could_contain.to_question(
            f"candidates[{slot}]"
        )
    if asked.open_first is not None and len(candidates) > 1:
        options = {
            str(slot): cut_long_line(place.signature, search.budget.max_line_chars)
            for slot, place in enumerate(candidates)
        }
        questions[asked.open_first.question_id] = asked.open_first.to_question(options)
    sources = {asked.found.question_id: code.source()}
    sources.update(
        {
            f"{asked.could_contain.question_id}#{slot}": {"place": place.key}
            for slot, place in enumerate(candidates)
        }
    )
    return _OpeningRequest(state, questions, sources)


def _ask_within_cap(judge: Judge, search: _Search, opening: _Opening):
    """None when a judge's global cap, shared with other callers, ran out before this request."""
    request = _opening_request(search, opening)
    try:
        return judge.ask(
            request.state, request.questions, thresholds=search.thresholds, sources=request.sources
        )
    except CallCapReachedError:
        search.cap_reached = True
        return _Unanswered.BUDGET


async def _ask_within_cap_async(judge: Judge, search: _Search, opening: _Opening):
    request = _opening_request(search, opening)
    try:
        return await judge.ask_async(
            request.state, request.questions, thresholds=search.thresholds, sources=request.sources
        )
    except CallCapReachedError:
        search.cap_reached = True
        return _Unanswered.BUDGET


def _set_aside_unasked(search: _Search, opening: _Opening, reason: str) -> None:
    _restore_opening(search, opening)
    _set_aside(search, opening.item, reason)


def _restore_opening(search: _Search, opening: _Opening) -> None:
    item = opening.item
    search.visited -= {item.place.key, opening.opened_key}
    search.judged_code.discard(opening.fingerprint)
    search.steps -= 1


def _set_aside_for_budget(search: _Search, item: _Queued) -> None:
    _set_aside(search, item, "budget")


def _set_aside(search: _Search, item: _Queued, reason: str) -> None:
    search.set_aside.append(
        NotInspected(
            item.place.key,
            item.place.signature,
            reason,
            item.probability,
            item.depth,
            item.path,
            item.place,
            item.tier,
        )
    )


def _merge(search: _Search, opening: _Opening, response) -> None:
    item, code, candidates = opening.item, opening.code, opening.candidates
    found_probability = response.noul(search.questions.found.question_id).probability
    visit = Visit(
        item.place.key, code, item.path, found_probability, search.thresholds.noul_verdict(found_probability)
    )
    _file_visit(search, visit, item.tier)
    picked = _picked_slot(search, response)
    set_aside_before = len(search.set_aside)
    offered = []
    for slot, place in enumerate(candidates):
        probability = _could_contain(search, response, slot)
        verdict = search.thresholds.noul_verdict(probability)
        path = (*item.path, place.key)
        search.push(
            place, probability, item.depth + 1, path, QueueTier.PICK if slot == picked else QueueTier.MOVE
        )
        offered.append(
            {
                "place": place.key,
                "signature": cut_long_line(place.signature, search.budget.max_line_chars),
                "probability": probability,
                "verdict": verdict,
            }
        )
    not_opened = [*opening.capped, *search.set_aside[set_aside_before:]]
    search.history.append(_open_step(search, opening, visit, offered, response, not_opened))


def _file_visit(search: _Search, visit: Visit, tier: QueueTier) -> None:
    """A start keeps its verdict in ``starts``; only a place the search reached can be found."""
    if tier == QueueTier.START:
        search.starts.append(visit)
    elif visit.verdict == NoulVerdict.YES:
        search.found.append(visit)
    elif visit.verdict == NoulVerdict.UNSURE:
        search.unsure.append(visit)
    else:
        search.searched.append(visit)


_DECISIONS = {NoulVerdict.YES: "found", NoulVerdict.UNSURE: "unsure", NoulVerdict.NO: "searched"}


def _open_step(
    search: _Search,
    opening: _Opening,
    visit: Visit,
    offered: list[dict],
    response,
    not_opened: list[NotInspected],
) -> HistoryStep:
    """What was opened, what Jev answered about it and its neighbours, and what code set aside."""
    judgments: dict[str, object] = {
        "contains_target": {"probability": visit.probability, "verdict": visit.verdict},
        "could_contain": offered,
    }
    pick = search.questions.open_first
    if pick is not None and pick.question_id in response.answers:
        answer = response.choice(pick.question_id)
        judgments["open_first"] = {
            "choice": _picked_place(answer.choice, opening.candidates),
            "confidence": answer.confidence,
            "used": _picked_slot(search, response) is not None,
        }
    if not_opened:
        judgments["not_opened"] = [_frontier_entry(entry) for entry in not_opened]
    return HistoryStep(
        "open",
        {"place": visit.place_key, "depth": opening.item.depth, "path": list(visit.path)},
        (FetchedSpan(visit.code.source(), visit.code.text),),
        judgments,
        f"start judged {visit.verdict}"
        if opening.item.tier == QueueTier.START
        else _DECISIONS[visit.verdict],
    )


def _picked_place(choice: str, candidates: list[Place]) -> str:
    return choice if choice == NO_CLEAR_FIRST else candidates[int(choice)].key


def _record_choice(search: _Search, beam: list[_Queued]) -> None:
    if not beam:
        return
    chosen = [
        {
            "place": item.place.key,
            "priority": item.probability,
            "depth": item.depth,
            "reason": _choice_reason(item),
        }
        for item in beam
    ]
    search.history.append(
        HistoryStep(
            "choose_next",
            {"chosen": chosen, "still_queued": len(search.queue)},
            decision=f"open {len(chosen)} of the best-scored places",
        )
    )


def _choice_reason(item: _Queued) -> str:
    """``start`` for a caller's start place, ``open_first`` for a place Jev picked to open next, else
    ``queue_score`` (its could_contain probability)."""
    return {
        QueueTier.START: "start",
        QueueTier.PICK: "open_first",
        QueueTier.DISCOVERED: "automatic_entry_alternative",
        QueueTier.MOVE: "queue_score",
    }[item.tier]


def _frontier_entry(entry: NotInspected) -> dict:
    return {"place": entry.place_key, "reason": entry.reason, "priority": entry.priority}


def _picked_slot(search: _Search, response) -> int | None:
    """The candidate Jev picked to open next, whatever its confidence; ``none`` picks nothing."""
    pick = search.questions.open_first
    if pick is None or pick.question_id not in response.answers:
        return None
    choice = response.choice(pick.question_id).choice
    return None if choice == NO_CLEAR_FIRST else int(choice)


def _could_contain(search: _Search, response, slot: int) -> float:
    return response.noul(f"{search.questions.could_contain.question_id}#{slot}").probability


def _candidate_state(place: Place, budget: SearchBudget) -> dict:
    """The signature line plus the first lines of the candidate's code, so the judgment rests on
    more than a name; every line is cut at the request's line limit."""
    lines = place.open().text.split("\n")[: budget.preview_lines]
    return {
        "signature": cut_long_line(place.signature, budget.max_line_chars),
        "preview": "\n".join(cut_long_line(line, budget.max_line_chars) for line in lines),
    }


def _result(search: _Search, outcome: Outcome, judge: Judge, index: CodeIndex) -> FindResult:
    left = [item for item in search.queue if item.place.key not in search.visited]
    frontier = [
        NotInspected(
            item.place.key,
            item.place.signature,
            _reason(search, item, outcome),
            item.probability,
            item.depth,
            item.path,
            item.place,
            item.tier,
        )
        for item in sorted(left)
    ]
    frontier += [entry for entry in search.set_aside if entry.place_key not in search.visited]
    not_inspected = tuple({entry.place_key: entry for entry in frontier}.values())
    unparsed = index.observed_unparsed_files
    completed_scans = index.parser_scans_completed
    pending_scans = index.parser_scans_pending
    unavailable = index.unavailable_files
    search.history.append(
        _stop_step(search, outcome, not_inspected, unparsed, completed_scans, pending_scans, unavailable)
    )
    return FindResult(
        outcome,
        tuple(search.found),
        tuple(search.searched),
        tuple(search.unsure),
        not_inspected,
        search.steps,
        judge.calls,
        frozenset(search.visited),
        frozenset(search.judged_code),
        search.history,
        search.stop_judgment,
        unparsed,
        tuple(search.moves),
        tuple(search.starts),
        completed_scans,
        pending_scans,
        unavailable,
    )


def _stop_step(
    search: _Search,
    outcome: Outcome,
    not_inspected: tuple[NotInspected, ...],
    unparsed: frozenset[str],
    completed_scans: tuple[str, ...],
    pending_scans: tuple[str, ...],
    unavailable: Mapping[str, str],
) -> HistoryStep:
    judgments: dict[str, object] = {"not_inspected": [_frontier_entry(entry) for entry in not_inspected]}
    if unparsed:
        judgments["unparsed_files"] = sorted(unparsed)
    judgments["parser_scans"] = {
        "completed": list(completed_scans),
        "pending": list(pending_scans),
    }
    if unavailable:
        judgments["unavailable_files"] = dict(unavailable)
    if search.stop_judgment is not None:
        judgments["last_stop_check"] = {
            "probability": search.stop_judgment.probability,
            "outcome": search.stop_judgment.outcome,
        }
    arguments = {"outcome": outcome, "moves": list(search.moves)}
    return HistoryStep("stop", arguments, (), judgments, f"stopped: {outcome}")


def _reason(search: _Search, item: _Queued, outcome: Outcome) -> str:
    if outcome == Outcome.CANCELLED:
        return "cancelled"
    if not search.still_worth_opening(item):
        return "deprioritized"
    return "target_found" if outcome == Outcome.FOUND else outcome.value
