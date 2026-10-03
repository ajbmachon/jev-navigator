"""JevClient over the official TypeSafe SDK (install the ``typesafe`` extra). Always the latest Jev.

Hosts with their own runtime (for example one that routes requests by region) pass an adapter over
that instead; the library only needs ``ask`` and ``model``.

When this adapter builds the SDK client itself, it wraps the HTTP transport so ``send`` returns the
exact response bytes, status and content type for a journal. A caller-supplied SDK client is not
wrapped; its responses are journaled decoded and marked inexact.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import os
import threading
from collections.abc import Coroutine, Mapping
from contextlib import contextmanager, nullcontext
from time import perf_counter_ns
from typing import Any

from ..judgments.answers import JevResponse, response_from_raw
from ..judgments.client import LATEST_JEV, input_budget_error
from ..judgments.journal import AttemptJournalCallbackError, RawAttempt, RawResponse


class _AttemptCollection:
    def __init__(self, on_attempt=None) -> None:
        self.on_attempt = on_attempt
        self.attempts: list[RawAttempt] = []

    def record(self, attempt: RawAttempt) -> None:
        self.attempts.append(attempt)
        if self.on_attempt is not None:
            try:
                self.on_attempt(attempt)
            except Exception as error:
                raise AttemptJournalCallbackError(error) from error


class CapturingTransport:
    """Passes requests through and records each physical request in the active logical call."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._active = threading.local()

    @contextmanager
    def collecting(self, on_attempt=None):
        collection = _AttemptCollection(on_attempt)
        previous = getattr(self._active, "collection", None)
        self._active.collection = collection
        try:
            yield collection
        finally:
            if previous is None:
                del self._active.collection
            else:
                self._active.collection = previous

    def handle_request(self, request):
        started = perf_counter_ns()
        try:
            response = self._inner.handle_request(request)
            response.read()
        except BaseException as error:
            self._record(
                RawAttempt(
                    self._next_attempt(),
                    _duration_ms(started),
                    request.content,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )
            raise
        content_type = response.headers.get("content-type", "")
        self._record(
            RawAttempt(
                self._next_attempt(),
                _duration_ms(started),
                request.content,
                RawResponse(response.content, response.status_code, content_type, sent_body=request.content),
            )
        )
        return response

    def _next_attempt(self) -> int:
        collection = getattr(self._active, "collection", None)
        return len(collection.attempts) + 1 if collection is not None else 1

    def _record(self, attempt: RawAttempt) -> None:
        collection = getattr(self._active, "collection", None)
        if collection is not None:
            collection.record(attempt)

    def close(self) -> None:
        self._inner.close()


class CapturingAsyncTransport:
    """Async equivalent, isolating each logical call with a context-local attempt collection."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._active: contextvars.ContextVar[_AttemptCollection | None] = contextvars.ContextVar(
            "typesafe_http_attempts", default=None
        )

    @contextmanager
    def collecting(self, on_attempt=None):
        collection = _AttemptCollection(on_attempt)
        token = self._active.set(collection)
        try:
            yield collection
        finally:
            self._active.reset(token)

    async def handle_async_request(self, request):
        started = perf_counter_ns()
        try:
            response = await self._inner.handle_async_request(request)
            await response.aread()
        except BaseException as error:
            self._record(
                RawAttempt(
                    self._next_attempt(),
                    _duration_ms(started),
                    request.content,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )
            raise
        content_type = response.headers.get("content-type", "")
        self._record(
            RawAttempt(
                self._next_attempt(),
                _duration_ms(started),
                request.content,
                RawResponse(response.content, response.status_code, content_type, sent_body=request.content),
            )
        )
        return response

    def _next_attempt(self) -> int:
        collection = self._active.get()
        return len(collection.attempts) + 1 if collection is not None else 1

    def _record(self, attempt: RawAttempt) -> None:
        collection = self._active.get()
        if collection is not None:
            collection.record(attempt)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _AsyncRunner:
    """Runs the official async SDK behind the adapter's synchronous boundary.

    The loop gives the CLI a real cancellation owner: cancelling the submitted coroutine aborts
    its in-flight HTTP operation instead of leaving a worker thread blocked in a sync socket call.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jev-typesafe", daemon=True)
        self._futures: set[concurrent.futures.Future] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._cancelled = False
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    def call(self, operation: Coroutine[Any, Any, Any]):
        return self._call(operation, allow_cancelled=False)

    def _call(self, operation: Coroutine[Any, Any, Any], *, allow_cancelled: bool):
        with self._lock:
            if self._closed:
                operation.close()
                raise RuntimeError("the TypeSafe client is closed")
            if self._cancelled and not allow_cancelled:
                operation.close()
                raise concurrent.futures.CancelledError
            future = asyncio.run_coroutine_threadsafe(operation, self._loop)
            self._futures.add(future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._futures.discard(future)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            futures = tuple(self._futures)
        for future in futures:
            future.cancel()

    def close(self, operation: Coroutine[Any, Any, Any]) -> None:
        if self._closed:
            operation.close()
            return
        self.cancel()
        try:
            self._call(operation, allow_cancelled=True)
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()


class TypeSafeJevClient:
    def __init__(self, sdk_client=None, model: str | None = LATEST_JEV, *, transport=None) -> None:
        """``transport`` is a sync test or host transport. The default uses the official async SDK
        behind the synchronous JevClient interface so ``cancel`` can abort active HTTP requests.
        ``model=None`` resolves `TYPESAFE_DEFAULT_MODEL` from the environment, falling back to
        `LATEST_JEV` — how consumers point the client at Drex or a finetuned endpoint."""
        self._capture: CapturingTransport | CapturingAsyncTransport | None = None
        self._runner: _AsyncRunner | None = None
        self._async_sdk = False
        if model is None:
            model = os.environ.get("TYPESAFE_DEFAULT_MODEL", "").strip() or LATEST_JEV
        if sdk_client is None:
            import httpx2

            if transport is None:
                from typesafe_sdk import AsyncTypeSafeClient

                self._capture = CapturingAsyncTransport(httpx2.AsyncHTTPTransport())
                sdk_client = AsyncTypeSafeClient(model=model, transport=self._capture)
                self._runner = _AsyncRunner()
                self._async_sdk = True
            else:
                from typesafe_sdk import TypeSafeClient

                self._capture = CapturingTransport(transport)
                sdk_client = TypeSafeClient(model=model, transport=self._capture)
        self._sdk = sdk_client
        self.model = model

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        """The response as received, with the SDK's decoded form attached.

        A provider refusal that names an exceeded input budget is translated to the typed
        ``InputBudgetExceededError`` so the batching owner can split the batch instead of the run
        dying on an untyped 400."""
        if self._async_sdk:
            assert self._runner is not None
            try:
                return self._runner.call(self._send_async(state, questions))
            except Exception as error:
                typed = input_budget_error(error)
                if typed is not None:
                    raise typed from error
                raise
        return self.send_with_attempts(state, questions)

    def send_with_attempts(self, state: Mapping, questions: Mapping, on_attempt=None) -> RawResponse:
        """Send one logical request and notify the owner as each physical HTTP attempt completes."""
        if self._async_sdk:
            assert self._runner is not None
            try:
                return self._runner.call(self._send_async(state, questions, on_attempt))
            except Exception as error:
                typed = input_budget_error(error)
                if typed is not None:
                    raise typed from error
                raise
        scope = self._capture.collecting(on_attempt) if self._capture else nullcontext(_AttemptCollection())
        with scope as collection:
            try:
                response = self._sdk.system_one(dict(state), dict(questions))
            except Exception as error:
                typed = input_budget_error(error)
                if typed is not None:
                    raise typed from error
                raise
            return self._raw_response(response, collection.attempts)

    async def _send_async(self, state: Mapping, questions: Mapping, on_attempt=None) -> RawResponse:
        scope = self._capture.collecting(on_attempt) if self._capture else nullcontext(_AttemptCollection())
        with scope as collection:
            try:
                response = await self._sdk.system_one(dict(state), dict(questions))
            except Exception as error:
                typed = input_budget_error(error)
                if typed is not None:
                    raise typed from error
                raise
            return self._raw_response(response, collection.attempts)

    def _raw_response(self, response, attempts: list[RawAttempt] | None = None) -> RawResponse:
        decoded = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        attempts = attempts or []
        captured = attempts[-1].response if attempts else None
        if captured is None:
            return RawResponse.from_decoded(decoded)
        return RawResponse(
            captured.body,
            captured.status,
            captured.content_type,
            decoded,
            sent_body=captured.sent_body,
            attempts=tuple(attempts),
        )

    def parse(self, raw: RawResponse) -> JevResponse:
        return response_from_raw(raw.json())

    def cancel(self) -> None:
        """Abort requests owned by this adapter. A supplied SDK client owns its own cancellation."""
        if self._runner is not None:
            self._runner.cancel()
            return
        cancel = getattr(self._sdk, "cancel", None)
        if cancel is not None:
            cancel()

    def close(self) -> None:
        """Release the official SDK and its HTTP transport."""
        if self._runner is not None:
            self._runner.close(self._sdk.aclose())
            return
        close = getattr(self._sdk, "close", None)
        if close is not None:
            close()


def provider_errors() -> tuple[type[Exception], ...]:
    """The TypeSafe SDK's error base, such as a refused key or a rate limit, when the ``typesafe``
    extra is installed; none without it, since nothing can raise one then."""
    try:
        from typesafe_sdk import TypeSafeError
    except ImportError:
        return ()
    return (TypeSafeError,)


def _duration_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000
