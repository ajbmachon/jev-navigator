"""The one method the library needs from a Jev connection, so hosts can pass their own runtime.

It also owns the provider's size limits (the character boxes) and the typed refusal a client raises
when a request is too large."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .answers import JevResponse
from .questions import serialized_chars

LATEST_JEV = "jev-latest"

MAX_TOKENS_MARKER = "max_tokens_exceeded"
"""The provider's error_type when a request's input exceeds the model's input budget."""

REQUEST_CHARS_PER_TOKEN = 2.4
"""ASCII-escaped characters (``serialized_chars``) per input token. Token limits become character
boxes with ``chars_for_tokens``; route boxes use the same ratio. The conservative measured fit
and its provenance are retained in ``measurements/runtime-limits/REPORT.md``."""


def chars_for_tokens(tokens: int) -> int:
    return int(tokens * REQUEST_CHARS_PER_TOKEN)


JEV_STATE_TOKEN_LIMIT = 32_000
"""The input Jev accepts for the state plus the longest single question (TypeSafe Models page,
docs.typesafe.ai/models). Historical boundary measurements and their provenance are retained in
``measurements/runtime-limits/REPORT.md``."""

JEV_REQUEST_TOKEN_LIMIT = 64_000
"""The input Jev accepts for a whole request; a request of 48,951 tokens was accepted."""


Wire = Callable[[Mapping, Mapping], tuple[Mapping, Mapping]]
"""A route's wire dialect: the state and questions its server accepts, built from the judge's request.
A wire only ever lengthens a request (text for structure, an added character), never shortens it."""


@dataclass(frozen=True)
class InputLimits:
    """A model's input limits in serialized characters: ``box_chars`` bounds the state plus the
    longest single question, and ``request_chars`` the whole body when the provider documents a
    bound for it (``None`` when it does not). Packing sends within them; a provider's typed refusal
    stays authoritative.

    ``wire`` is the dialect the route sends, when it is stricter than the request as built (Drex's).
    The limits hold for what the server receives, so ``exceeded_by`` measures the request in that
    form, and a request that fits as built but not as sent is refused before it is sent, which the
    judge splits like any size refusal. It is not part of the limits' identity: equal boxes are equal
    limits."""

    box_chars: int
    request_chars: int | None = None
    wire: Wire | None = field(default=None, compare=False)

    @classmethod
    def from_tokens(cls, box_tokens: int, request_tokens: int | None = None) -> InputLimits:
        request_chars = None if request_tokens is None else chars_for_tokens(request_tokens)
        return cls(chars_for_tokens(box_tokens), request_chars)

    def exceeded_by(self, state: Mapping, questions: Mapping) -> bool:
        """Whether this request, as the route sends it, is outside the limits, measured in
        ASCII-escaped JSON (``serialized_chars``), the one measure of every size box."""
        if self.wire is not None:
            state, questions = self.wire(state, questions)
        longest_question = max((serialized_chars(question) for question in questions.values()), default=0)
        if serialized_chars(state) + longest_question > self.box_chars:
            return True
        body = serialized_chars({"state": state, "questions": questions})
        return self.request_chars is not None and body > self.request_chars

    def tightest(self, other: InputLimits) -> InputLimits:
        """The limits a request must keep to fit both. Since a wire only lengthens a request,
        measuring it in every route's wire form at the smallest box is a fit for each route."""
        bounds = [limit for limit in (self.request_chars, other.request_chars) if limit is not None]
        box_chars = min(self.box_chars, other.box_chars)
        return InputLimits(box_chars, min(bounds) if bounds else None, _both(self.wire, other.wire))


def _both(first: Wire | None, second: Wire | None) -> Wire | None:
    if first is None or second is None or first is second:
        return first or second
    return lambda state, questions: second(*first(state, questions))


JEV_INPUT_LIMITS = InputLimits.from_tokens(JEV_STATE_TOKEN_LIMIT, JEV_REQUEST_TOKEN_LIMIT)
"""Jev's limits: ``JEV_STATE_TOKEN_LIMIT`` for the state plus the longest question and
``JEV_REQUEST_TOKEN_LIMIT`` for a body, in characters at ``REQUEST_CHARS_PER_TOKEN``."""


def input_limits_of(client: object) -> InputLimits:
    """The limits a client declares as ``input_limits``; a client that declares none is taken to be Jev."""
    return getattr(client, "input_limits", JEV_INPUT_LIMITS)


class JevClient(Protocol):
    """``ask`` sends and parses. A client that can also split the two offers ``send`` (returns a
    ``RawResponse``: body bytes, HTTP status, content type) and ``parse`` (``RawResponse`` to
    ``JevResponse``); the judge then journals the raw response before parsing it."""

    model: str

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse: ...


class AsyncJevClient(Protocol):
    """The async form, for hosts with an async runtime. ``send`` may be async as well; ``parse``
    stays a plain function. Use it with the judge's ``*_async`` methods and ``find_code_async``."""

    model: str

    async def ask(self, state: Mapping, questions: Mapping) -> JevResponse: ...


class MissingAnswerError(LookupError):
    """Replay found no stored answer for a request."""


class UnansweredQuestionError(RuntimeError):
    """The provider's response left out the answer to a question the request asked."""


class InputBudgetExceededError(RuntimeError):
    """The provider refused a request whose input exceeded the model's input budget. A routed client
    names the ``model`` that refused and its ``box_chars``, so the refusal is remembered under them."""

    def __init__(self, message: str, *, model: str | None = None, box_chars: int | None = None) -> None:
        super().__init__(message)
        self.model = model
        self.box_chars = box_chars


def input_budget_error(error: BaseException) -> InputBudgetExceededError | None:
    """The typed input-budget error for a provider rejection that names an exceeded input budget,
    or None when the error is anything else.

    The official SDK reports the breach as a bad request (400) whose body carries
    ``{"detail": {"error_type": "max_tokens_exceeded"}}``. The classifier reads the status and
    body off the error without importing the optional SDK, so any client that surfaces them
    (the TypeSafe adapter, a routed System-One client, a host's own runtime) translates the
    same way, and a gateway between the library and the provider does not hide the contract.
    """
    if getattr(error, "status", None) != 400:
        return None
    body = getattr(error, "body", None)
    if isinstance(body, (str, bytes, bytearray)):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    if not isinstance(body, Mapping):
        return None
    detail = body.get("detail")
    if not isinstance(detail, Mapping) or detail.get("error_type") != MAX_TOKENS_MARKER:
        return None
    return InputBudgetExceededError(str(error))


class ReplayOnlyClient:
    """Never calls Jev: a replay run accepts stored answers from any served model, and fails loudly
    on any request the store cannot answer."""

    model = LATEST_JEV
    replays_any_model = True

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        raise MissingAnswerError("no stored answer for this request, and this client never calls Jev")
