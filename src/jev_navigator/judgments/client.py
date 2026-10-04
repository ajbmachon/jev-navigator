"""The one method the library needs from a Jev connection, so hosts can pass their own runtime.

It also owns the provider's size limits (the character boxes) and the typed refusal a client raises
when a request is too large."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol

from .answers import JevResponse

LATEST_JEV = "jev-latest"

MAX_TOKENS_MARKER = "max_tokens_exceeded"
"""The provider's error_type when a request's input exceeds the model's input budget."""

REQUEST_CHARS_PER_TOKEN = 2.4
"""ASCII-escaped characters (``serialized_chars``) per input token, the same value and meaning as the
Engine's ``REQUEST_CHARS_PER_TOKEN`` (analysis-engine ``enginepy/host/system_one.py``). A limit in
tokens becomes a box in characters with ``chars_for_tokens``, never with a second ratio: route boxes
(Drex's 8,192 tokens is 19,660 characters) use it too.

The fit and its data are in ``jvn-eval-2026-10-03/census/request-size-fit`` (``fit-table-request-size.md``).
On 3,096 real requests (03.10.2026) Jev's input is 263 tokens plus 0.22 per state character and 0.29
per question character. The 264 requests of 2,000 tokens or more cost at most 0.38 tokens per
character (2.63 characters per token), so 2.4 (0.417) keeps a 1.10 margin. The data reaches only 11,652
tokens; the limit itself rests on the Engine's measurement of the edge (32,883 pass, about 33,200
refused)."""


def chars_for_tokens(tokens: int) -> int:
    return int(tokens * REQUEST_CHARS_PER_TOKEN)


JEV_STATE_TOKEN_LIMIT = 32_000
"""The input Jev accepts for the state plus the longest single question (TypeSafe Models page,
docs.typesafe.ai/models). The Engine measured it on 27.09.2026: 32,883 input tokens pass and about
33,200 are refused with ``max_tokens_exceeded``."""

JEV_REQUEST_TOKEN_LIMIT = 64_000
"""The input Jev accepts for a whole request; a request of 48,951 tokens was accepted."""

JEV_INPUT_BOX_CHARS = chars_for_tokens(JEV_STATE_TOKEN_LIMIT)
"""The character box for the state plus the longest single question: 76,800."""

MAX_REQUEST_CHARS = chars_for_tokens(JEV_REQUEST_TOKEN_LIMIT)
"""The character box for a whole request body: 153,600."""

QUESTION_RESERVE_CHARS = chars_for_tokens(2_000)
"""What a state leaves free for the question that reads it."""


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
    """The provider refused a request whose input exceeded the model's input budget."""


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
