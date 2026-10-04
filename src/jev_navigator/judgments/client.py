"""The one method the library needs from a Jev connection, so hosts can pass their own runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol

from ..errors import JvnRefusal
from .answers import JevResponse

LATEST_JEV = "jev-latest"

MAX_TOKENS_MARKER = "max_tokens_exceeded"
"""The provider's error_type when a request's input exceeds the model's input budget."""

JEV_STATE_TOKEN_LIMIT = 32_000
"""The input Jev accepts for the state plus the longest single question. The TypeSafe Models page
(docs.typesafe.ai/models) documents 32k for it and 64k tokens per whole request. The Engine
measured the binding one on 27.09.2026: 32,883 input tokens pass and about 33,200 are refused with
``max_tokens_exceeded``, while a whole request of 48,951 tokens was accepted."""

DEFAULT_QUESTION_RESERVE = 2_000

TOKENS_PER_BYTE = 0.60
"""The densest measured provider tokens per serialized body byte (0.43 to 0.60 on recorded traffic),
so an estimate with it never undercounts the content this library sends."""


def estimate_tokens(text: str) -> int:
    """A conservative token estimate from the UTF-8 size, when no tokenizer is supplied."""
    return int(len(text.encode()) * TOKENS_PER_BYTE) + 1


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


class MissingAnswerError(JvnRefusal, LookupError):
    """Replay found no stored answer for a request."""


class InputBudgetExceededError(JvnRefusal, RuntimeError):
    """The provider refused a request whose input exceeded the model's input budget."""


def input_budget_error(error: BaseException) -> InputBudgetExceededError | None:
    """The typed input-budget error for a provider rejection that names an exceeded input budget,
    or None when the error is anything else.

    The official SDK reports the breach as a bad request (400) whose body carries
    ``{"detail": {"error_type": "max_tokens_exceeded"}}``. The classifier reads the status and
    body off the error without importing the optional SDK, so any client that surfaces them —
    the TypeSafe adapter, a routed System-One client, a host's own runtime — translates the same
    way, and a gateway between the library and the provider does not hide the contract.
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
