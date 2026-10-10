"""Whether a set of supplied code holds what each point describes: one existence Noul per point.

A ranking (J1-3 per unit) says where to look; this says whether what was found answers the point. The
caller supplies the pieces, such as the best few units of each point, and the points. Every point's
question goes into one request whose state is the points (``targets``) and the pieces (``fetched``,
each entry only its file and code), through ``history.judge_sections``, which owns the request's
size, masking, store and call accounting; ``ask_existence_async`` asks through its async form. The
wording is ``judgments/existence_question.json``; a point's text sits in state, never in the wording.

When the pieces do not fit the client's box, the history evicts the earliest entries' code and every
eviction is named in the answer; ``existence_fits`` tells a caller beforehand, so it can split the
points over smaller sets instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..history import (
    FETCHED,
    FetchedSpan,
    History,
    HistoryCheck,
    HistoryJudgment,
    HistoryStep,
    SectionLimit,
    judge_sections,
    judge_sections_async,
)
from ..judgments.answers import AnswerSource
from ..judgments.judge import Judge
from ..judgments.profiles import EXISTS
from ..judgments.questions import Check, serialized_chars
from ..judgments.role_labels import LabelPiece

TARGETS = "targets"
_SHOWN = "shown"


@dataclass(frozen=True)
class ExistenceAnswer:
    """One point's raw P(yes) over ``shown`` (the place ids in the state, in state order), the answer
    behind it, and the place ids whose code the box forced out of the state (``evicted``)."""

    target: str
    probability: float
    answered_by: AnswerSource | None
    shown: tuple[str, ...]
    evicted: tuple[str, ...]


def existence_check(target: str) -> Check:
    """The existence question bound to one point in shared state."""
    return EXISTS.questions(target)[0]


def existence_fits(judge: Judge, targets: Mapping[str, str], pieces: Sequence[LabelPiece]) -> bool:
    """Whether every point's existence question over ``pieces`` fits the judge's box without
    evicting any piece's code, measured the way the request is built."""
    history = _history(pieces)
    longest = max(serialized_chars(existence_check(target).to_question()) for target in targets)
    history.state_for((FETCHED,), _shared(targets), longest, box_chars=judge.input_limits.box_chars)
    return not history.evictions


def ask_existence(
    judge: Judge, targets: Mapping[str, str], pieces: Sequence[LabelPiece]
) -> dict[str, ExistenceAnswer]:
    """Each point's existence answer over the same ``pieces``, all in one request."""
    history, checks = _request(targets, pieces)
    return _answers(history, pieces, judge_sections(judge, history, checks, _shared(targets)))


async def ask_existence_async(
    judge: Judge, targets: Mapping[str, str], pieces: Sequence[LabelPiece]
) -> dict[str, ExistenceAnswer]:
    """``ask_existence`` through the Judge's async form, for an async client."""
    history, checks = _request(targets, pieces)
    return _answers(history, pieces, await judge_sections_async(judge, history, checks, _shared(targets)))


def _request(
    targets: Mapping[str, str], pieces: Sequence[LabelPiece]
) -> tuple[History, dict[str, HistoryCheck]]:
    if not targets or not pieces:
        raise ValueError("an existence question needs at least one point and one piece")
    checks = {target: HistoryCheck(existence_check(target), (FETCHED,)) for target in targets}
    return _history(pieces), checks


def _answers(
    history: History, pieces: Sequence[LabelPiece], judged: Mapping[str, HistoryJudgment]
) -> dict[str, ExistenceAnswer]:
    shown = tuple(piece.place.id for piece in pieces)
    evicted = tuple(pieces[eviction["span"]].place.id for eviction in history.evictions)
    return {
        target: ExistenceAnswer(target, answer.probability, answer.answered_by, shown, evicted)
        for target, answer in judged.items()
    }


def _history(pieces: Sequence[LabelPiece]) -> History:
    """One step holding the pieces, each entry only its file and code, with no per-entry cut: a piece
    either shows whole or is evicted and named."""
    history = History(limits={FETCHED: SectionLimit()})
    spans = tuple(FetchedSpan({"file": piece.place.file}, piece.code) for piece in pieces)
    history.append(HistoryStep(_SHOWN, fetched=spans))
    return history


def _shared(targets: Mapping[str, str]) -> dict:
    return {TARGETS: dict(targets)}
