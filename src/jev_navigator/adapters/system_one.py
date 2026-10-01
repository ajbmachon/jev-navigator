"""The contract every decision-model adapter follows, and the shared base HTTP adapters subclass.

An adapter is a `JevClient` for one model service. It declares these class attributes
(`CONTRACT_ATTRIBUTES`):

- ``name``: the route name that selects it (`SYSTEM_ONE_ROUTES=drex`).
- ``endpoint``: its hosted endpoint; empty when every route must configure one, or when it runs in
  this process.
- ``default_model``: the model asked for when none is given.
- ``api_key_env``: the service's own key variable. A key comes from ``api_key`` or this variable,
  never from another service's, so no key reaches a vendor that did not issue it.
- ``needs_key``: refuse to build without a key. A self-hosted server may need none.
- ``pinned``: refuse any endpoint but ``endpoint``.
- ``runs_locally``: the model runs in this process (`local.LocalModelClient`), not behind HTTP.

It is built as ``Adapter(model=None, *, api_key=None, endpoint=None, transport=None, timeout=None,
max_retries=None)``, where None means the adapter's default, and offers ``send`` (the response as
received, for the journal), ``parse``, ``ask``, ``cancel`` (abort the requests in flight; a
cancelled search waits on them) and ``close``. The model a response names is the version that
answered: the answer store reuses answers by it, so a service must name each checkpoint, never a
moving alias. `adapters/registry.py` lists the adapters shipped here; a route can name any other
by import path. `tests/test_adapters.py` holds each registered one to this contract.

`SystemOneClient` implements the contract over the System-One wire (`POST /v1/systemone`) with the
standard library only, so an adapter built on it needs no extra installed. A vendor subclass sets
the attributes and overrides a hook only where its service differs: ``wire_questions`` and
``wire_body`` for the request dialect, ``headers`` for authentication, ``parse`` for the response
dialect. Sending, retrying, cancelling and exact-byte capture stay shared, so every adapter behaves
the same under failure. Unsubclassed, it is the adapter for any endpoint a route names, such as
your own model server; it sends a key only when it has one. `TypeSafeJevClient` follows the same
contract over the official TypeSafe SDK.
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.metadata
import json
import math
import os
import random
import socket
import ssl
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import ClassVar, Protocol
from urllib.parse import urlsplit

from ..judgments.answers import JevResponse, response_from_raw
from ..judgments.client import input_budget_error
from ..judgments.journal import RawResponse

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})
BACKOFF_SECONDS = (0.5, 5.0)
MAX_RETRY_AFTER_SECONDS = 60.0
MAX_ERROR_MESSAGE = 200


CONTRACT_ATTRIBUTES = (
    "name",
    "endpoint",
    "default_model",
    "api_key_env",
    "needs_key",
    "pinned",
    "runs_locally",
    "send",
    "parse",
    "ask",
    "cancel",
    "close",
)


class Adapter(Protocol):
    name: ClassVar[str]
    endpoint: ClassVar[str]
    default_model: ClassVar[str]
    api_key_env: ClassVar[str]
    needs_key: ClassVar[bool]
    pinned: ClassVar[bool]
    runs_locally: ClassVar[bool]
    model: str

    def send(self, state: Mapping, questions: Mapping) -> RawResponse: ...

    def parse(self, raw: RawResponse) -> JevResponse: ...

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HttpRequest:
    url: str
    body: bytes
    headers: Mapping[str, str]
    timeout: float


@dataclass(frozen=True)
class HttpResponse:
    """``headers`` are keyed by lower-case name."""

    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[HttpRequest], HttpResponse]


class AdapterError(RuntimeError):
    """The service refused the request, or stayed unavailable through every retry. ``body`` is the
    response body as received, which `input_budget_error` reads to recognise a size refusal."""

    def __init__(self, adapter: str, status: int, message: str, body: bytes = b"") -> None:
        super().__init__(f"{adapter} answered HTTP {status}: {message}")
        self.status = status
        self.body = body


class SystemOneClient:
    name: ClassVar[str] = "system-one"
    endpoint: ClassVar[str] = ""
    default_model: ClassVar[str] = ""
    api_key_env: ClassVar[str] = ""
    needs_key: ClassVar[bool] = False
    pinned: ClassVar[bool] = False
    runs_locally: ClassVar[bool] = False
    path: ClassVar[str] = "/v1/systemone"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        transport: Transport | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """``transport`` sends one HTTP request; the default is `StdlibTransport`. ``timeout`` is
        per attempt, in seconds."""
        endpoint = endpoint or self.endpoint
        if not endpoint:
            raise ValueError(f"{self.name} needs an endpoint")
        endpoint = _checked_endpoint(self.name, endpoint)
        if self.pinned and endpoint != self.endpoint.rstrip("/"):
            raise ValueError(f"{self.name} is pinned to {self.endpoint}; {endpoint} is refused")
        self.model = model or self.default_model
        if not self.model:
            raise ValueError(f"{self.name} needs a model")
        self._api_key = api_key or (os.environ.get(self.api_key_env, "").strip() if self.api_key_env else "")
        if self.needs_key and not self._api_key:
            raise RuntimeError(
                f"{self.api_key_env} is unset: export it or pass api_key"
                if self.api_key_env
                else f"{self.name} needs an API key: pass api_key"
            )
        self.url = endpoint + self.path
        self._transport = transport or StdlibTransport()
        self._timeout = DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
        self._max_retries = MAX_RETRIES if max_retries is None else max_retries
        self._cancelled = threading.Event()

    # --- the dialect: override only where a service differs -------------------------------------

    def wire_questions(self, questions: Mapping) -> dict:
        return dict(questions)

    def wire_body(self, state: Mapping, questions: Mapping) -> dict:
        return {"model": self.model, "state": dict(state), "questions": self.wire_questions(questions)}

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def parse(self, raw: RawResponse) -> JevResponse:
        return response_from_raw(raw.json())

    # --- shared by every adapter ----------------------------------------------------------------

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        """The response as received. A status in `RETRY_STATUSES` or a connection failure is
        retried up to ``max_retries`` times, after the wait the service asks for or a backoff;
        any other failure raises `AdapterError` at once, except a provider input-budget refusal,
        which raises the typed ``InputBudgetExceededError`` so the batching owner can split the
        batch."""
        body = json.dumps(self.wire_body(state, questions), ensure_ascii=False).encode()
        request = HttpRequest(self.url, body, self.headers(), self._timeout)
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            if self._cancelled.is_set():
                raise CancelledError
            try:
                response = self._transport(request)
            except CancelledError:
                raise
            except (OSError, http.client.HTTPException) as error:
                if last:
                    raise ConnectionError(f"{self.name} is unreachable: {error}") from error
                self._wait(_backoff(attempt))
                continue
            if 200 <= response.status < 300:
                content_type = response.headers.get("content-type", "")
                return RawResponse(response.body, response.status, content_type, sent_body=body)
            if last or response.status not in RETRY_STATUSES:
                error = AdapterError(self.name, response.status, _error_message(response.body), response.body)
                typed = input_budget_error(error)
                if typed is not None:
                    raise typed from error
                raise error
            self._wait(_retry_after(response.headers) or _backoff(attempt))
        raise AssertionError("the retry loop always returns or raises")

    def cancel(self) -> None:
        """Abort the requests in flight and refuse every later one, as a cancelled search needs."""
        self._cancelled.set()
        cancel = getattr(self._transport, "cancel", None)
        if cancel is not None:
            cancel()

    def close(self) -> None:
        close = getattr(self._transport, "close", None)
        if close is not None:
            close()

    def _wait(self, seconds: float) -> None:
        if self._cancelled.wait(seconds):
            raise CancelledError


class StdlibTransport:
    """One `http.client` connection per request, verified against the system's certificates.
    ``cancel`` shuts every open socket, so a blocked read returns at once, and refuses every later
    request."""

    def __init__(self) -> None:
        self._context = ssl.create_default_context()
        self._lock = threading.Lock()
        self._open: set[http.client.HTTPConnection] = set()
        self._cancelled = False

    def __call__(self, request: HttpRequest) -> HttpResponse:
        url = urlsplit(request.url)
        connection = (
            http.client.HTTPSConnection(
                url.hostname, url.port, timeout=request.timeout, context=self._context
            )
            if url.scheme == "https"
            else http.client.HTTPConnection(url.hostname, url.port, timeout=request.timeout)
        )
        with self._lock:
            self._refuse_if_cancelled()
            self._open.add(connection)
        try:
            connection.connect()
            with self._lock:
                self._refuse_if_cancelled()
            target = url.path + (f"?{url.query}" if url.query else "")
            connection.request("POST", target, body=request.body, headers=dict(request.headers))
            response = connection.getresponse()
            body = response.read()
            return HttpResponse(response.status, {k.lower(): v for k, v in response.getheaders()}, body)
        except Exception:
            if self._cancelled:
                raise CancelledError from None
            raise
        finally:
            with self._lock:
                self._open.discard(connection)
            connection.close()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            open_connections = tuple(self._open)
        for connection in open_connections:
            if connection.sock is not None:
                # The plain socket's shutdown, also under TLS: it ends a read blocked in another
                # thread without tearing down the TLS object that read is using.
                with contextlib.suppress(OSError):
                    socket.socket.shutdown(connection.sock, socket.SHUT_RDWR)

    def _refuse_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError


def _checked_endpoint(adapter: str, endpoint: str) -> str:
    """``endpoint`` without its trailing slash, once it is an ``http``/``https`` URL with a host that
    the wire path can follow. Refused here rather than on the first request, where a route would
    take the failure for an outage and fail over."""
    url = urlsplit(endpoint)
    if url.username is not None or url.password is not None:
        # Not repeated in the message: what sits before the `@` is a secret.
        raise ValueError(f"{adapter} endpoint may not carry credentials; set its API key instead")
    if url.scheme not in ("http", "https"):
        problem = "needs http:// or https://"
    elif not url.hostname:
        problem = "names no host"
    elif url.query or url.fragment:
        problem = "may not carry a query or fragment, which the request path would follow"
    else:
        try:
            url.port  # noqa: B018 - parsing the port is the check
        except ValueError:
            problem = "has a port that is no number"
        else:
            return endpoint.rstrip("/")
    raise ValueError(f"{adapter} endpoint {endpoint!r} {problem}")


def _backoff(attempt: int) -> float:
    initial, maximum = BACKOFF_SECONDS
    return min(maximum, initial * 2**attempt) * (1 - 0.25 * random.random())  # noqa: S311 - jitter


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The wait the service asks for: `retry-after-ms`, else `retry-after` in seconds or as a date."""
    try:
        milliseconds = float(headers.get("retry-after-ms", ""))
        if math.isfinite(milliseconds) and milliseconds >= 0:
            return min(milliseconds / 1000, MAX_RETRY_AFTER_SECONDS)
    except ValueError:
        pass
    value = headers.get("retry-after", "").strip()
    if value.isdigit():
        return min(float(value), MAX_RETRY_AFTER_SECONDS)
    try:
        wait = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
    except (TypeError, ValueError):
        return None
    return min(max(wait, 0.0), MAX_RETRY_AFTER_SECONDS)


def _error_message(body: bytes) -> str:
    """The service's own message when it sent a JSON error, else the start of the body."""
    try:
        decoded = json.loads(body)
        error = decoded.get("error", decoded) if isinstance(decoded, dict) else decoded
        message = (error.get("message") or error.get("detail")) if isinstance(error, dict) else None
        text = str(message) if message else json.dumps(decoded, ensure_ascii=False)
    except ValueError:
        text = body.decode("utf-8", "replace")
    return text[:MAX_ERROR_MESSAGE]


def _user_agent() -> str:
    try:
        return f"jev-navigator/{importlib.metadata.version('jev-navigator')}"
    except importlib.metadata.PackageNotFoundError:
        return "jev-navigator"


USER_AGENT = _user_agent()
