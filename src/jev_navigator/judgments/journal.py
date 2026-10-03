"""A durable record of every attempt, kept apart from the parsed-answer store.

The judge records the exact (masked) request before dispatch and the raw response before parsing:
the body bytes as received, the HTTP status, the content type and the input tokens the provider
reported (or ``not reported``), with the decoded form optional. A
transport error or a response that fails to parse is recorded as a failure. Hosts inject their
own journal (their runtime's, an evaluation journal); ``JsonlJournal`` is a simple local one.

A request holds code, and the library cannot know whose code it is. So by default ``JsonlJournal``
keeps only the request hash, the question ids and a hash of the state; the full request text is kept
only with ``keep_request_text=True``, which is meant for your own or open-source code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .answers import reported_input_tokens
from .questions import content_hash


@dataclass(frozen=True)
class JournalRequest:
    """``body`` is the request as handed to the client, keys in the order they were built."""

    request_sha256: str
    requested_model: str
    state: Mapping
    questions: Mapping
    body: bytes = b""


@dataclass(frozen=True)
class RawResponse:
    """A response as received. ``body`` is the exact bytes when the client captured them
    (``exact``); a client that only returns a decoded response is recorded with its JSON re-encoded,
    ``exact=False`` and no status. ``sent_body`` is the exact request body the client sent, when its
    transport captured it."""

    body: bytes
    status: int | None
    content_type: str
    decoded: Mapping | None = None
    exact: bool = True
    sent_body: bytes | None = None
    attempts: tuple[RawAttempt, ...] = ()

    @classmethod
    def from_decoded(cls, decoded: Mapping) -> RawResponse:
        body = json.dumps(decoded, sort_keys=True).encode()
        return cls(body, None, "application/json", decoded, exact=False)

    def json(self) -> Mapping:
        return self.decoded if self.decoded is not None else json.loads(self.body)


@dataclass(frozen=True)
class RawAttempt:
    """One physical HTTP request made while the SDK executes a logical request."""

    attempt_no: int
    duration_ms: float
    sent_body: bytes
    response: RawResponse | None = None
    error_type: str | None = None
    error: str | None = None
    route: str | None = None


class AttemptJournalCallbackError(RuntimeError):
    """Prevents the provider SDK from retrying after its attempt sink failed to persist."""

    def __init__(self, original_error: Exception) -> None:
        super().__init__("the HTTP attempt journal callback failed")
        self.original_error = original_error


class Journal(Protocol):
    def record_request(self, request: JournalRequest) -> str:
        """Returns the id later records refer to."""
        ...

    def record_response(self, request_id: str, response: RawResponse) -> None: ...

    def record_attempt(self, request_id: str, attempt: RawAttempt) -> None: ...

    def record_failure(self, request_id: str, error: str, response: RawResponse | None = None) -> None: ...


class JsonlJournal:
    def __init__(self, path: Path, *, keep_request_text: bool = False) -> None:
        self.path = Path(path)
        self.keep_request_text = keep_request_text
        self._write_lock = threading.Lock()

    def record_request(self, request: JournalRequest) -> str:
        request_id = uuid.uuid4().hex
        fields = _request_with_text(request) if self.keep_request_text else _request_without_text(request)
        self._append({"kind": "request", "request_id": request_id, **fields})
        return request_id

    def record_response(self, request_id: str, response: RawResponse) -> None:
        self._append({"kind": "response", "request_id": request_id, **self._response_fields(response)})

    def record_attempt(self, request_id: str, attempt: RawAttempt) -> None:
        fields = {
            "kind": "http_attempt",
            "request_id": request_id,
            "attempt_no": attempt.attempt_no,
            "duration_ms": attempt.duration_ms,
            "sent_body_sha256": hashlib.sha256(attempt.sent_body).hexdigest(),
        }
        if self.keep_request_text:
            fields["sent_body_base64"] = _base64(attempt.sent_body)
        if attempt.route is not None:
            fields["route"] = attempt.route
        if attempt.response is not None:
            fields.update(_response_fields(attempt.response))
            fields["outcome"] = "response"
        else:
            fields.update(
                {
                    "outcome": "failure",
                    "error_type": attempt.error_type,
                    "error": attempt.error,
                }
            )
        self._append(fields)

    def record_step(self, step: Mapping) -> None:
        """Lets a ``History`` record every appended step in the same file."""
        self._append({"kind": "history_step", "step": dict(step)})

    def record_failure(self, request_id: str, error: str, response: RawResponse | None = None) -> None:
        fields = self._response_fields(response) if response is not None else {}
        self._append({"kind": "failure", "request_id": request_id, "error": error, **fields})

    def _response_fields(self, response: RawResponse) -> dict:
        """The sent body holds code, so it is kept only with ``keep_request_text``."""
        fields = _response_fields(response)
        if self.keep_request_text and response.sent_body is not None:
            fields["sent_body_base64"] = _base64(response.sent_body)
        return fields

    def _append(self, line: dict) -> None:
        line["recorded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as lines:
                lines.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def _response_fields(response: RawResponse) -> dict:
    return {
        "body_base64": _base64(response.body),
        "status": response.status,
        "content_type": response.content_type,
        "exact": response.exact,
        "input_tokens": _reported_input_tokens(response.body),
    }


NOT_REPORTED = "not reported"


def _reported_input_tokens(body: bytes) -> int | str:
    """The provider's ``usage.input_tokens``, or ``NOT_REPORTED`` when the body has none."""
    try:
        raw = json.loads(body)
    except ValueError:
        return NOT_REPORTED
    reported = reported_input_tokens(raw) if isinstance(raw, Mapping) else None
    return NOT_REPORTED if reported is None else reported


def _request_with_text(request: JournalRequest) -> dict:
    """The state and questions for reading (written with sorted keys) and the body in its exact order."""
    fields = {key: value for key, value in asdict(request).items() if key != "body"}
    return {**fields, "body_base64": _base64(request.body)}


def _base64(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def _request_without_text(request: JournalRequest) -> dict:
    return {
        "request_sha256": request.request_sha256,
        "requested_model": request.requested_model,
        "question_ids": list(request.questions),
        "state_sha256": content_hash(request.state),
    }
