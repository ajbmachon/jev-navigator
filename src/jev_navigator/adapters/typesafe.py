"""JevClient over the official TypeSafe SDK (install the ``typesafe`` extra). Always the latest Jev.

It follows the adapter contract in `system_one.py` and is registered as the ``jev`` route; unlike
the other adapters it runs on the official SDK rather than `SystemOneClient`. Hosts with their own
runtime (for example one that routes requests by region) pass an adapter over that instead; the
library only needs ``ask`` and ``model``.

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
from typing import Any

from ..judgments.answers import JevResponse, response_from_raw
from ..judgments.client import LATEST_JEV, input_budget_error
from ..judgments.journal import RawResponse


class CapturingTransport:
    """Passes requests to ``inner`` and keeps each thread's last response, with the exact request
    body that produced it, as a ``RawResponse``."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._last = threading.local()

    def handle_request(self, request):
        response = self._inner.handle_request(request)
        response.read()
        content_type = response.headers.get("content-type", "")
        self._last.response = RawResponse(
            response.content, response.status_code, content_type, sent_body=request.content
        )
        return response

    def take(self) -> RawResponse | None:
        captured = getattr(self._last, "response", None)
        self._last.response = None
        return captured

    def close(self) -> None:
        self._inner.close()


class CapturingAsyncTransport:
    """Async equivalent of ``CapturingTransport``. A context variable keeps concurrent requests'
    responses attached to the task that sent each one."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._last: contextvars.ContextVar[RawResponse | None] = contextvars.ContextVar(
            "typesafe_raw_response", default=None
        )

    async def handle_async_request(self, request):
        response = await self._inner.handle_async_request(request)
        await response.aread()
        content_type = response.headers.get("content-type", "")
        self._last.set(
            RawResponse(response.content, response.status_code, content_type, sent_body=request.content)
        )
        return response

    def take(self) -> RawResponse | None:
        captured = self._last.get()
        self._last.set(None)
        return captured

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
    name = "jev"
    endpoint = "https://api.typesafe.ai"
    default_model = LATEST_JEV
    api_key_env = "TYPESAFE_API_KEY"
    needs_key = True
    pinned = False
    runs_locally = False

    def __init__(
        self,
        sdk_client=None,
        model: str | None = LATEST_JEV,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        transport=None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """``transport`` is a sync test or host transport. The default uses the official async SDK
        behind the synchronous JevClient interface so ``cancel`` can abort active HTTP requests.
        ``model=None`` resolves `TYPESAFE_DEFAULT_MODEL` from the environment, falling back to
        `LATEST_JEV`; ``api_key`` and ``endpoint`` left unset resolve `TYPESAFE_API_KEY` and
        `TYPESAFE_BASE_URL` the same way. ``timeout`` and ``max_retries`` left unset keep the SDK's
        defaults."""
        self._capture: CapturingTransport | CapturingAsyncTransport | None = None
        self._runner: _AsyncRunner | None = None
        self._async_sdk = False
        if model is None:
            model = os.environ.get("TYPESAFE_DEFAULT_MODEL", "").strip() or LATEST_JEV
        if sdk_client is None:
            import httpx2
            from typesafe_sdk import RetryPolicy

            options = {"model": model, "api_key": api_key, "base_url": endpoint, "timeout": timeout}
            if max_retries is not None:
                options["retry"] = RetryPolicy(max_retries=max_retries)

            if transport is None:
                from typesafe_sdk import AsyncTypeSafeClient

                self._capture = CapturingAsyncTransport(httpx2.AsyncHTTPTransport())
                sdk_client = AsyncTypeSafeClient(**options, transport=self._capture)
                self._runner = _AsyncRunner()
                self._async_sdk = True
            else:
                from typesafe_sdk import TypeSafeClient

                self._capture = CapturingTransport(transport)
                sdk_client = TypeSafeClient(**options, transport=self._capture)
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
        try:
            response = self._sdk.system_one(dict(state), dict(questions))
        except Exception as error:
            typed = input_budget_error(error)
            if typed is not None:
                raise typed from error
            raise
        return self._raw_response(response)

    async def _send_async(self, state: Mapping, questions: Mapping) -> RawResponse:
        try:
            response = await self._sdk.system_one(dict(state), dict(questions))
        except Exception as error:
            typed = input_budget_error(error)
            if typed is not None:
                raise typed from error
            raise
        return self._raw_response(response)

    def _raw_response(self, response) -> RawResponse:
        decoded = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        captured = self._capture.take() if self._capture else None
        if captured is None:
            return RawResponse.from_decoded(decoded)
        return RawResponse(
            captured.body, captured.status, captured.content_type, decoded, sent_body=captured.sent_body
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
