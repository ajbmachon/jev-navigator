"""A navigation history Jev can read: typed steps, exposed as named, bounded sections.

Each step records the operation, its arguments, the code it fetched (with the source of every span),
the judgments made with their raw probabilities, and the decision code took. Jev never sees the steps
directly. A check selects named sections and ``History.state_for(names)`` builds exactly that state:

- ``fetched`` (the default a history check reads): every code body fetched, with its source, and
  nothing else, so a check never leans on the search's own verdicts;
- ``history``: ``{"steps": [...]}``, each step's operation, arguments and fetched code, without
  judgments or decisions;
- ``decisions``: every step without code (judgments with probabilities, candidates, choices, places
  set aside), for a check that is meant to read the verdicts;
- ``previous_judgments``: the last answer of each history check, with its probability;
- any section the caller declares, for example ``subject`` or ``shown_code``.

Every section has its own limit (``SectionLimit``: newest entries kept, long text cut), applied before
the character budget. When the selected state still does not fit, a pluggable eviction policy trims it; the
default replaces the oldest code bodies with a stub that keeps the source, and every eviction is
recorded. Checks that select the same sections share one request; different selections run in
parallel.

The character budget is capped by Jev's input box for ``state`` plus the longest question: the
documented 32,000 tokens at 2.4 characters per token (``REQUEST_CHARS_PER_TOKEN``, the Engine's
value), 76,800 characters. The Engine measured 32,883 tokens accepted and about 33,200 refused on
27.09.2026. A whole request may be larger, up to the documented 64k tokens.
The docs also warn that accuracy falls as unrelated state grows,
so select only the sections a check needs, and measure with ``ceiling_curve``.

Whether the history is enough is a caller-defined yes/no question about a concrete property, for
example "Does `fetched` contain code that compares the item count with a limit?", never "is it enough".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

from .judgments.answers import JevResponse
from .judgments.client import JEV_INPUT_BOX_CHARS, QUESTION_RESERVE_CHARS
from .judgments.judge import Judge
from .judgments.questions import Check, content_hash, serialized_chars
from .judgments.thresholds import NoulVerdict

EVICTED = "[evicted]"
HISTORY = "history"
FETCHED = "fetched"
DECISIONS = "decisions"
PREVIOUS_JUDGMENTS = "previous_judgments"
BUILT_IN_SECTIONS = (HISTORY, FETCHED, DECISIONS, PREVIOUS_JUDGMENTS)
DEFAULT_STOP_SECTIONS = (FETCHED,)
_SECTIONS_WITH_CODE = frozenset({HISTORY, FETCHED})


@dataclass(frozen=True)
class FetchedSpan:
    source: Mapping
    code: str | None

    @property
    def evicted(self) -> bool:
        return self.code is None


@dataclass(frozen=True)
class HistoryStep:
    operation: str
    arguments: Mapping = field(default_factory=dict)
    fetched: tuple[FetchedSpan, ...] = ()
    judgments: Mapping[str, object] = field(default_factory=dict)
    decision: str = ""

    def to_json(self) -> dict:
        return {
            "operation": self.operation,
            "arguments": dict(self.arguments),
            "fetched": [
                {**dict(span.source), "code": span.code if span.code is not None else EVICTED}
                for span in self.fetched
            ],
            "judgments": dict(self.judgments),
            "decision": self.decision,
        }

    def without_code(self) -> dict:
        """The step with each code body replaced by its hash, for journals that must not keep code."""
        return {
            **self.to_json(),
            "fetched": [
                {**dict(span.source), "code_sha256": content_hash(span.code or "")} for span in self.fetched
            ],
        }

    def history_json(self) -> dict:
        """The step without its judgments or decision: what the ``history`` section shows."""
        full = self.to_json()
        return {"operation": full["operation"], "arguments": full["arguments"], "fetched": full["fetched"]}

    def decision_json(self) -> dict:
        """The step without its code: what the ``decisions`` section shows."""
        return {
            "operation": self.operation,
            "arguments": dict(self.arguments),
            "judgments": dict(self.judgments),
            "decision": self.decision,
        }


EvictionPolicy = Callable[
    [list[HistoryStep], Callable[[list[HistoryStep]], bool]], tuple[list[HistoryStep], list[dict]]
]


class StepRecorder(Protocol):
    def record_step(self, step: Mapping) -> None: ...


class HistoryTooLargeError(ValueError):
    """Even with every code body evicted the selected sections do not fit the budget."""


class UnknownSectionError(KeyError):
    """A check selected, or a caller set, a section the history does not declare."""


@dataclass(frozen=True)
class SectionLimit:
    """``max_entries`` keeps the newest entries of a list section; ``max_chars`` cuts each text value."""

    max_entries: int | None = None
    max_chars: int | None = None


DEFAULT_LIMITS: Mapping[str, SectionLimit] = {
    FETCHED: SectionLimit(max_chars=8_000),
    DECISIONS: SectionLimit(max_entries=40, max_chars=2_000),
}


def drop_oldest_code(
    steps: list[HistoryStep], fits: Callable[[list[HistoryStep]], bool]
) -> tuple[list[HistoryStep], list[dict]]:
    """Replaces code bodies with stubs, oldest first, until the steps fit; decisions are kept."""
    trimmed = list(steps)
    evicted: list[dict] = []
    for step_number, step in enumerate(trimmed):
        for span_number, span in enumerate(step.fetched):
            if fits(trimmed):
                return trimmed, evicted
            if span.evicted:
                continue
            fetched = list(trimmed[step_number].fetched)
            fetched[span_number] = FetchedSpan(span.source, None)
            trimmed[step_number] = replace(trimmed[step_number], fetched=tuple(fetched))
            evicted.append({"step": step_number, "span": span_number, **dict(span.source)})
    return trimmed, evicted


@dataclass
class History:
    """``sections`` declares the caller's own sections with their values; only declared names and the
    built-in ones (``history``, ``fetched``, ``decisions``, ``previous_judgments``) can be selected."""

    budget_chars: int = JEV_INPUT_BOX_CHARS - QUESTION_RESERVE_CHARS
    evict: EvictionPolicy = drop_oldest_code
    recorder: StepRecorder | None = None
    sections: dict[str, object] = field(default_factory=dict)
    limits: Mapping[str, SectionLimit] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    steps: list[HistoryStep] = field(default_factory=list)
    previous_judgments: dict[str, HistoryJudgment] = field(default_factory=dict)
    evictions: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.budget_chars = min(self.budget_chars, JEV_INPUT_BOX_CHARS)
        reserved = set(self.sections) & set(BUILT_IN_SECTIONS)
        if reserved:
            raise ValueError(f"{sorted(reserved)} are built-in section names")

    def append(self, step: HistoryStep) -> None:
        """Adds a step; the recorder gets it without code bodies (their hashes and sources only)."""
        self.steps.append(step)
        if self.recorder is not None:
            self.recorder.record_step(step.without_code())

    def set_section(self, name: str, value: object) -> None:
        if name not in self.sections:
            raise UnknownSectionError(f"{name} is not a declared section")
        self.sections[name] = value

    def state_for(self, names: Sequence[str]) -> dict:
        """Exactly the selected sections, each within its limit and all within the character budget;
        ``self.evictions`` lists what this call trimmed."""
        self._require_known(names)

        def fits(steps: list[HistoryStep]) -> bool:
            return self.size(self._build(names, steps)) <= self.budget_chars

        reads_code = bool(_SECTIONS_WITH_CODE & set(names))
        kept, self.evictions = self.evict(self.steps, fits) if reads_code else (self.steps, [])
        if not fits(kept):
            raise HistoryTooLargeError(f"the selected sections need more than {self.budget_chars} characters")
        return self._build(names, kept)

    def size(self, state: Mapping) -> int:
        return serialized_chars(state)

    def _require_known(self, names: Sequence[str]) -> None:
        unknown = set(names) - set(self.sections) - set(BUILT_IN_SECTIONS)
        if unknown:
            raise UnknownSectionError(f"unknown sections: {sorted(unknown)}")

    def _build(self, names: Sequence[str], steps: list[HistoryStep]) -> dict:
        return {name: _limited(self._section(name, steps), self.limits.get(name)) for name in names}

    def _section(self, name: str, steps: list[HistoryStep]) -> object:
        if name == HISTORY:
            return {"steps": [step.history_json() for step in steps]}
        if name == FETCHED:
            return [
                {**dict(span.source), "code": span.code if span.code is not None else EVICTED}
                for step in steps
                for span in step.fetched
            ]
        if name == DECISIONS:
            return [step.decision_json() for step in steps]
        if name == PREVIOUS_JUDGMENTS:
            return {
                check: {"probability": judged.probability, "outcome": judged.outcome}
                for check, judged in self.previous_judgments.items()
            }
        return self.sections[name]


def _limited(value: object, limit: SectionLimit | None) -> object:
    if limit is None:
        return value
    if isinstance(value, list):
        kept = value[-limit.max_entries :] if limit.max_entries is not None else value
        return [_cut(entry, limit.max_chars) for entry in kept]
    return _cut(value, limit.max_chars)


def _cut(value: object, max_chars: int | None) -> object:
    if max_chars is None:
        return value
    if isinstance(value, str):
        return (
            value
            if len(value) <= max_chars
            else f"{value[:max_chars]}[... {len(value) - max_chars} characters cut]"
        )
    if isinstance(value, Mapping):
        return {key: _cut(entry, max_chars) for key, entry in value.items()}
    if isinstance(value, list):
        return [_cut(entry, max_chars) for entry in value]
    return value


class HistoryOutcome(StrEnum):
    FOUND = "found"
    SEARCHED_NOT_FOUND = "searched_not_found"
    NOT_INSPECTED = "not_inspected"
    CONTINUE = "continue"


@dataclass(frozen=True)
class HistoryJudgment:
    outcome: HistoryOutcome
    probability: float
    chars: int
    evictions: tuple[dict, ...]
    sections: tuple[str, ...] = DEFAULT_STOP_SECTIONS


@dataclass(frozen=True)
class HistoryCheck:
    """A caller's concrete yes/no question over the history, and the sections it reads."""

    check: Check
    sections: tuple[str, ...] = DEFAULT_STOP_SECTIONS


def judge_history(
    judge: Judge,
    history: History,
    check: Check,
    shared: Mapping | None = None,
    *,
    sections: tuple[str, ...] = DEFAULT_STOP_SECTIONS,
    exhausted: bool = False,
) -> HistoryJudgment:
    """Asks the caller's check over the selected sections (state fields next to ``shared``). Yes
    means found; no means the history was searched and does not hold it; unsure means keep going,
    or, once the caller's budget is ``exhausted``, not inspected, never absent."""
    checks = {check.name: HistoryCheck(check, sections)}
    return judge_sections(judge, history, checks, shared, exhausted=exhausted)[check.name]


async def judge_history_async(
    judge: Judge,
    history: History,
    check: Check,
    shared: Mapping | None = None,
    *,
    sections: tuple[str, ...] = DEFAULT_STOP_SECTIONS,
    exhausted: bool = False,
) -> HistoryJudgment:
    checks = {check.name: HistoryCheck(check, sections)}
    judged = await judge_sections_async(judge, history, checks, shared, exhausted=exhausted)
    return judged[check.name]


def judge_sections(
    judge: Judge,
    history: History,
    checks: Mapping[str, HistoryCheck],
    shared: Mapping | None = None,
    *,
    exhausted: bool = False,
) -> dict[str, HistoryJudgment]:
    """Several history checks: those selecting the same sections share one request, and different
    selections are asked in parallel. Each answer is kept in ``history.previous_judgments``."""
    groups = _grouped(history, checks, shared or {})
    with ThreadPoolExecutor(max_workers=len(groups) or 1) as pool:
        responses = list(pool.map(lambda group: _ask_group(judge, group), groups))
    return _judged(judge, history, groups, responses, exhausted)


async def judge_sections_async(
    judge: Judge,
    history: History,
    checks: Mapping[str, HistoryCheck],
    shared: Mapping | None = None,
    *,
    exhausted: bool = False,
) -> dict[str, HistoryJudgment]:
    groups = _grouped(history, checks, shared or {})
    responses = await asyncio.gather(*(_ask_group_async(judge, group) for group in groups))
    return _judged(judge, history, groups, list(responses), exhausted)


@dataclass(frozen=True)
class _Group:
    sections: tuple[str, ...]
    state: Mapping
    checks: Mapping[str, Check]

    @property
    def questions(self) -> dict:
        return {check.question_id: check.to_question() for check in self.checks.values()}


def _grouped(history: History, checks: Mapping[str, HistoryCheck], shared: Mapping) -> list[_Group]:
    by_sections: dict[tuple[str, ...], dict[str, Check]] = {}
    for name, entry in checks.items():
        by_sections.setdefault(entry.sections, {})[name] = entry.check
    return [
        _Group(sections, _state(history, sections, shared), grouped)
        for sections, grouped in by_sections.items()
    ]


def _ask_group(judge: Judge, group: _Group) -> JevResponse:
    return judge.ask(group.state, group.questions, thresholds=judge.thresholds)


async def _ask_group_async(judge: Judge, group: _Group) -> JevResponse:
    return await judge.ask_async(group.state, group.questions, thresholds=judge.thresholds)


def _judged(
    judge: Judge,
    history: History,
    groups: list[_Group],
    responses: list[JevResponse],
    exhausted: bool,
) -> dict[str, HistoryJudgment]:
    results: dict[str, HistoryJudgment] = {}
    for group, response in zip(groups, responses, strict=True):
        chars = history.size(group.state)
        for name, check in group.checks.items():
            probability = response.noul(check.question_id).probability
            results[name] = _judgment(judge, probability, chars, history, group.sections, exhausted)
    history.previous_judgments.update(results)
    return results


def _state(history: History, sections: tuple[str, ...], shared: Mapping) -> dict:
    selected = history.state_for(sections)
    overlap = set(selected) & set(shared)
    if overlap:
        raise ValueError(f"shared state and history sections both use {sorted(overlap)}")
    return {**shared, **selected}


def _judgment(
    judge: Judge,
    probability: float,
    chars: int,
    history: History,
    sections: tuple[str, ...],
    exhausted: bool,
) -> HistoryJudgment:
    verdict = judge.thresholds.noul_verdict(probability)
    outcome = {
        NoulVerdict.YES: HistoryOutcome.FOUND,
        NoulVerdict.NO: HistoryOutcome.SEARCHED_NOT_FOUND,
    }.get(verdict, HistoryOutcome.NOT_INSPECTED if exhausted else HistoryOutcome.CONTINUE)
    return HistoryJudgment(outcome, probability, chars, tuple(history.evictions), sections)


@dataclass(frozen=True)
class CeilingPoint:
    steps: int
    chars: int
    probability: float
    evicted_spans: int


def ceiling_curve(
    judge: Judge,
    steps: Sequence[HistoryStep],
    check: Check,
    shared: Mapping | None = None,
    *,
    sections: tuple[str, ...] = DEFAULT_STOP_SECTIONS,
    budget_chars: int = JEV_INPUT_BOX_CHARS - QUESTION_RESERVE_CHARS,
) -> list[CeilingPoint]:
    """Replays a recorded search with a growing history and reports the check's probability at each
    size, to see where more characters stop helping. Use it with a replay or scripted client."""
    points = []
    for count in range(1, len(steps) + 1):
        history = History(budget_chars=budget_chars, steps=list(steps[:count]))
        judged = judge_history(judge, history, check, shared, sections=sections)
        points.append(CeilingPoint(count, judged.chars, judged.probability, len(judged.evictions)))
    return points
