"""A scripted Jev client for offline tests: answers come from tables, and every request is kept."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .judgments.answers import (
    ChoiceAnswer,
    JevResponse,
    NoulAnswer,
    ScoreAnswer,
    response_from_raw,
    response_to_raw,
)
from .judgments.journal import RawResponse

NoulScript = Mapping[str, float] | Callable[[str, Mapping, Mapping], float]
DEFAULT_TOP_PROBABILITY = 0.9


@dataclass
class ScriptedJevClient:
    """Looks up each question by its full id, then without the ``#slot`` suffix, then by its name
    before ``@``. A Noul without an entry gets ``default_noul``; a Choice without an entry picks its
    first option with probability 0.9."""

    nouls: NoulScript = field(default_factory=dict)
    choices: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    scores: Mapping[str, Sequence[float]] = field(default_factory=dict)
    default_noul: float = 0.5
    model: str = "jev-scripted"
    input_tokens_per_call: int | None = 100
    requests: list[tuple[Mapping, Mapping]] = field(default_factory=list)

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        body = json.dumps(response_to_raw(self._answer_all(state, questions)), sort_keys=True).encode()
        return RawResponse(body, 200, "application/json")

    def parse(self, raw: RawResponse) -> JevResponse:
        return response_from_raw(raw.json())

    def _answer_all(self, state: Mapping, questions: Mapping) -> JevResponse:
        self.requests.append((state, questions))
        answers = {
            question_id: self._answer(question_id, question, state)
            for question_id, question in questions.items()
        }
        return JevResponse(answers, self.model, self.input_tokens_per_call)

    def _answer(self, question_id: str, question: Mapping, state: Mapping):
        if question["type"] == "noul":
            return NoulAnswer(self._noul(question_id, question, state))
        if question["type"] == "score":
            return ScoreAnswer.from_probabilities(self._score(question_id, question))
        return ChoiceAnswer.from_probabilities(self._choice(question_id, question))

    def _noul(self, question_id: str, question: Mapping, state: Mapping) -> float:
        if callable(self.nouls):
            return self.nouls(question_id, question, state)
        return next(
            (self.nouls[key] for key in _lookup_keys(question_id) if key in self.nouls), self.default_noul
        )

    def _score(self, question_id: str, question: Mapping) -> dict[str, float]:
        scripted = next((self.scores[key] for key in _lookup_keys(question_id) if key in self.scores), None)
        return _level_probabilities(len(question["criteria"]), scripted)

    def _choice(self, question_id: str, question: Mapping) -> dict[str, float]:
        options = list(question["criteria"])
        scripted = next((self.choices[key] for key in _lookup_keys(question_id) if key in self.choices), None)
        if scripted is not None:
            return {option: scripted.get(option, 0.0) for option in options}
        return _first_option_wins(options)


@dataclass
class AsyncScriptedJevClient:
    """The async form of ``ScriptedJevClient``: same scripts, an awaitable ``send``, and the same
    request list. It yields to the event loop once per request, so concurrent sends interleave."""

    script: ScriptedJevClient = field(default_factory=ScriptedJevClient)

    @property
    def model(self) -> str:
        return self.script.model

    @property
    def requests(self) -> list[tuple[Mapping, Mapping]]:
        return self.script.requests

    async def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(await self.send(state, questions))

    async def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        await asyncio.sleep(0)
        return self.script.send(state, questions)

    def parse(self, raw: RawResponse) -> JevResponse:
        return self.script.parse(raw)


def _level_probabilities(levels: int, scripted: Sequence[float] | None) -> dict[str, float]:
    if scripted is not None:
        return {str(level): probability for level, probability in enumerate(scripted)}
    return _first_option_wins([str(level) for level in range(levels)])


def _lookup_keys(question_id: str) -> list[str]:
    without_slot = question_id.split("#")[0]
    name = without_slot.split("@")[0]
    slot = question_id.split("#")[1] if "#" in question_id else None
    keys = [question_id, without_slot]
    if slot is not None:
        keys.insert(1, f"{name}#{slot}")
    return [*keys, name, name.split(".")[0]]


def _first_option_wins(options: list[str]) -> dict[str, float]:
    if len(options) == 1:
        return {options[0]: 1.0}
    rest = (1 - DEFAULT_TOP_PROBABILITY) / (len(options) - 1)
    return {
        option: DEFAULT_TOP_PROBABILITY if position == 0 else rest for position, option in enumerate(options)
    }
