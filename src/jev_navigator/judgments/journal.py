"""A durable record of every attempt, kept apart from the parsed-answer store.

The judge records the exact (masked) request before dispatch and the raw response before parsing:
the body bytes as received, the HTTP status, the content type and the input tokens the provider
reported (``null`` when it reported none, never 0), with the decoded form optional. The tokens are
on the ``response`` line only, once per logical request. A transport error, a response that fails
to parse and one that leaves out an asked answer are recorded as failures. Hosts inject their
own journal (their runtime's, an evaluation journal); ``JsonlJournal`` is a simple local one.

A request holds code, and the library cannot know whose code it is. So by default ``JsonlJournal``
keeps only the request hash, the question ids and a hash of the state; the full request text is kept
only with ``keep_request_text=True``, which is meant for your own or open-source code. An error can
echo its request (a 422 validation body often does). Its message and the body of a response with an
error status are kept as they are, unless ``keep_error_text=False`` (the CLI's ``--no-error-text`` or
``JEV_NAVIGATOR_ERROR_TEXT=off``) keeps only their length and SHA-256 (``message_fields``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import CancelledError
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


ERROR_TEXT_VARIABLE = "JEV_NAVIGATOR_ERROR_TEXT"
"""``on`` (the default) keeps error messages and error bodies in run files; ``off`` keeps their digests."""


def error_text_kept(no_error_text: bool) -> bool:
    """Whether a run keeps error text: not with ``--no-error-text``, else as ``ERROR_TEXT_VARIABLE``
    says."""
    if no_error_text:
        return False
    setting = os.environ.get(ERROR_TEXT_VARIABLE, "on")
    if setting not in ("on", "off"):
        raise ValueError(f"{ERROR_TEXT_VARIABLE}={setting!r} must be on or off")
    return setting == "on"


def error_message(error: BaseException) -> str:
    """The message an error is recorded with; a bare ``CancelledError`` says what it means."""
    if isinstance(error, CancelledError) and not str(error):
        return "the request was cancelled after it was sent"
    return str(error)


def message_fields(message: str, *, keep_text: bool) -> dict:
    """An error message as a run file stores it: the text only when ``keep_text``, otherwise its
    length and SHA-256, so a reader can still match two records of one error without the text."""
    if keep_text:
        return {"message": message}
    return {"message_length": len(message), "message_sha256": hashlib.sha256(message.encode()).hexdigest()}


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

    def record_failure(
        self, request_id: str, error: BaseException, response: RawResponse | None = None
    ) -> None: ...


class JsonlJournal:
    def __init__(self, path: Path, *, keep_request_text: bool = False, keep_error_text: bool = True) -> None:
        self.path = Path(path)
        self.keep_request_text = keep_request_text
        self.keeps_error_text = keep_request_text or keep_error_text
        self._write_lock = threading.Lock()

    def record_request(self, request: JournalRequest) -> str:
        request_id = uuid.uuid4().hex
        fields = _request_with_text(request) if self.keep_request_text else _request_without_text(request)
        self._append({"kind": "request", "request_id": request_id, **fields})
        return request_id

    def record_response(self, request_id: str, response: RawResponse) -> None:
        fields = self._response_fields(response)
        tokens = {"input_tokens": _reported_input_tokens(response.body)}
        self._append({"kind": "response", "request_id": request_id, **fields, **tokens})

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
            fields.update(_response_fields(attempt.response, keep_error_body=self.keeps_error_text))
            fields["outcome"] = "response"
        else:
            fields.update({"outcome": "failure", "error_type": attempt.error_type})
            if attempt.error is not None:
                fields.update(message_fields(attempt.error, keep_text=self.keeps_error_text))
        self._append(fields)

    def record_step(self, step: Mapping) -> None:
        """Lets a ``History`` record every appended step in the same file."""
        self._append({"kind": "history_step", "step": dict(step)})

    def record_failure(
        self, request_id: str, error: BaseException, response: RawResponse | None = None
    ) -> None:
        fields = self._response_fields(response) if response is not None else {}
        message = message_fields(error_message(error), keep_text=self.keeps_error_text)
        self._append(
            {
                "kind": "failure",
                "request_id": request_id,
                "error_type": type(error).__name__,
                **message,
                **fields,
            }
        )

    def _response_fields(self, response: RawResponse) -> dict:
        """The sent body holds code, so it is kept only with ``keep_request_text``."""
        fields = _response_fields(response, keep_error_body=self.keeps_error_text)
        if self.keep_request_text and response.sent_body is not None:
            fields["sent_body_base64"] = _base64(response.sent_body)
        return fields

    def _append(self, line: dict) -> None:
        line["recorded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as lines:
                lines.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def _response_fields(response: RawResponse, *, keep_error_body: bool) -> dict:
    """An answer's body is kept to replay it; a body with an error status may echo the request, so
    without ``keep_error_body`` only its length and SHA-256 are."""
    fields = {"status": response.status, "content_type": response.content_type, "exact": response.exact}
    if keep_error_body or not _is_error_status(response.status):
        return {**fields, "body_base64": _base64(response.body)}
    return {**fields, **_body_digest(response.body)}


def error_text_digested(record: Mapping) -> dict:
    """A written journal record as it reads with error text off: an error message keeps only its
    length and SHA-256, and so does a body with an error status. A record without either is unchanged."""
    digested = dict(record)
    if "message" in digested:
        digested.update(message_fields(digested.pop("message"), keep_text=False))
    if "body_base64" in digested and _is_error_status(digested.get("status")):
        digested.update(_body_digest(base64.b64decode(digested.pop("body_base64"))))
    return digested


def _body_digest(body: bytes) -> dict:
    return {"body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest()}


def _is_error_status(status: int | None) -> bool:
    return status is not None and status >= 400


def _reported_input_tokens(body: bytes) -> int | None:
    """The provider's ``usage.input_tokens``, or ``None`` when the body has none."""
    try:
        raw = json.loads(body)
    except ValueError:
        return None
    return reported_input_tokens(raw) if isinstance(raw, Mapping) else None


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


def keeps_request_text(path: Path) -> bool:
    """Whether the run file at ``path`` holds the text of any request: a journal written with
    ``keep_request_text`` (a request's state and body, or a body as sent) or an answer store written
    with ``keep_requests`` (a record's request, or its body as sent). A line that is not a JSON
    record, such as one a crash cut off, tells nothing."""
    with path.open() as lines:
        return any(_holds_request_text(record) for record in _records(lines))


def _holds_request_text(record: Mapping) -> bool:
    if record.get("sent_body_base64") is not None or record.get("request") is not None:
        return True
    return record.get("kind") == "request" and "body_base64" in record


def _records(lines: Iterable[str]) -> Iterator[Mapping]:
    for line in lines:
        try:
            yield json.loads(line)
        except ValueError:
            continue
