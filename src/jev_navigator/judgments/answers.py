"""Typed Jev answers, independent of any SDK, in the form the answer store keeps them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float

    @classmethod
    def from_probabilities(cls, probabilities: Mapping[str, float]) -> ChoiceAnswer:
        """An answer whose confidence is derived as the API does: (n * max - 1) / (n - 1)."""
        winner = max(probabilities, key=probabilities.__getitem__)
        return cls(winner, dict(probabilities), distribution_confidence(probabilities))

    def to_json(self) -> dict:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class NoulAnswer:
    probability: float

    def to_json(self) -> dict:
        return {"type": "noul", "noul": self.probability}


@dataclass(frozen=True)
class ScoreAnswer:
    """``score`` is the probability-weighted level; ``probabilities`` are keyed by level number as text."""

    score: float
    probabilities: Mapping[str, float]
    confidence: float

    @classmethod
    def from_probabilities(cls, probabilities: Mapping[str, float]) -> ScoreAnswer:
        expected = sum(int(level) * probability for level, probability in probabilities.items())
        return cls(expected, dict(probabilities), distribution_confidence(probabilities))

    def to_json(self) -> dict:
        return {
            "type": "score",
            "score": self.score,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer


NOT_REPORTED_TEXT = "not reported"
"""How a missing token count (``None``) reads in text for people; data keeps ``None``/``null``."""


@dataclass(frozen=True)
class AnswerSource:
    """Where one answer came from: the request's hash and the question id it was asked under, which
    a journal's request row lists, and whether the store replayed it."""

    request_sha256: str
    question_id: str
    from_store: bool

    def to_json(self) -> dict:
        return {
            "request_sha256": self.request_sha256,
            "question_id": self.question_id,
            "from_store": self.from_store,
        }


ANSWERED_BY = "answered_by"
SCORED_BY = "scored_by"
ANSWER_SOURCE_FIELDS = frozenset({ANSWERED_BY, SCORED_BY})
"""Run-file join keys, never shown to Jev: ``from_store`` differs between a run and its replay."""


def answered_by(source: AnswerSource | None) -> dict:
    """A record's ``answered_by`` field, or nothing when no request is known: the one form every run
    file uses to join a judgment to its journal answer."""
    return {ANSWERED_BY: source.to_json()} if source is not None else {}


def scored_by(source: AnswerSource | None) -> dict:
    """A queued place's ``scored_by`` field: the answer whose probability became its priority."""
    return {SCORED_BY: source.to_json()} if source is not None else {}


def without_answer_sources(value: object) -> object:
    """``value`` with every answer source field removed, at any depth."""
    if isinstance(value, Mapping):
        return {
            key: without_answer_sources(item)
            for key, item in value.items()
            if key not in ANSWER_SOURCE_FIELDS
        }
    if isinstance(value, list | tuple):
        return [without_answer_sources(item) for item in value]
    return value


@dataclass(frozen=True)
class JevResponse:
    """``input_tokens`` is what the provider reported for the request, ``None`` when it reported
    nothing; a missing count is never 0. ``from_store`` marks a replay: it sent nothing and carries
    no count. A response composed from several requests carries none either; totals count only
    requests sent. Its ``sources`` name the request and question behind each of its answers."""

    answers: Mapping[str, Answer]
    model: str
    input_tokens: int | None = None
    request_sha256: str = ""
    from_store: bool = False
    extra: Mapping[str, object] = field(default_factory=dict)
    sources: Mapping[str, AnswerSource] = field(default_factory=dict)

    def source(self, question_id: str) -> AnswerSource | None:
        """The request and question that answered ``question_id``; None when no request is known."""
        if question_id in self.sources:
            return self.sources[question_id]
        if not self.request_sha256:
            return None
        return AnswerSource(self.request_sha256, question_id, self.from_store)

    def choice(self, question_id: str) -> ChoiceAnswer:
        answer = self.answers[question_id]
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError(f"{question_id} is not a Choice answer")
        return answer

    def score(self, question_id: str) -> ScoreAnswer:
        answer = self.answers[question_id]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"{question_id} is not a Score answer")
        return answer

    def noul(self, question_id: str) -> NoulAnswer:
        answer = self.answers[question_id]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"{question_id} is not a Noul answer")
        return answer


def answer_from_json(raw: Mapping) -> Answer:
    if raw["type"] == "noul":
        return NoulAnswer(float(raw["noul"]))
    probabilities = {str(key): float(value) for key, value in raw["probabilities"].items()}
    if raw["type"] == "score":
        return ScoreAnswer(float(raw["score"]), probabilities, float(raw["confidence"]))
    confidence = raw.get("confidence")
    if confidence is None:
        return ChoiceAnswer.from_probabilities(probabilities)
    return ChoiceAnswer(str(raw["choice"]), probabilities, float(confidence))


def distribution_confidence(probabilities: Mapping[str, float]) -> float:
    option_count = len(probabilities)
    if option_count < 2:
        return 1.0
    return (option_count * max(probabilities.values()) - 1) / (option_count - 1)


def reported_input_tokens(raw: Mapping) -> int | None:
    return _reported_usage_count(raw, "input_tokens")


def reported_output_tokens(raw: Mapping) -> int | None:
    return _reported_usage_count(raw, "output_tokens")


def _reported_usage_count(raw: Mapping, name: str) -> int | None:
    """A ``usage`` count when it is a non-negative integer, else ``None``: the one place a raw
    response's token counts are read."""
    usage = raw.get("usage")
    reported = usage.get(name) if isinstance(usage, Mapping) else None
    return (
        reported if isinstance(reported, int) and not isinstance(reported, bool) and reported >= 0 else None
    )


def response_from_raw(raw: Mapping) -> JevResponse:
    """Parses a raw System One response: ``{"model", "usage": {"input_tokens"}, "answers": {...}}``."""
    answers = {question_id: answer_from_json(answer) for question_id, answer in raw["answers"].items()}
    return JevResponse(answers, str(raw["model"]), reported_input_tokens(raw))


def response_to_raw(response: JevResponse) -> dict:
    usage = {} if response.input_tokens is None else {"input_tokens": response.input_tokens}
    return {
        "model": response.model,
        "usage": usage,
        "answers": {question_id: answer.to_json() for question_id, answer in response.answers.items()},
    }


@dataclass
class TokenTotal:
    """The tokens responses reported, how many responses there were, and how many reported none."""

    reported: int = 0
    not_reported: int = 0
    responses: int = 0

    def add(self, tokens: int | None) -> None:
        self.responses += 1
        if tokens is None:
            self.not_reported += 1
        else:
            self.reported += tokens
