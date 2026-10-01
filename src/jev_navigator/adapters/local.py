"""The adapter contract for a model that runs in this process, such as a classifier loaded from
disk, instead of behind HTTP.

A subclass sets the contract attributes (``runs_locally`` is already True, ``endpoint`` stays
empty) and implements three hooks:

- ``load()``: load the model. It runs once, on the first question, so a slow cold start costs
  nothing until the model is asked.
- ``answer(state, questions)``: the System-One response body as a mapping, ``{"answers": ...}``
  with optional ``"usage"``. ``"model"`` defaults to ``self.model``; name the loaded checkpoint
  there, never a moving alias, because the answer store reuses answers by it. A model whose input
  is too large for it raises `InputBudgetExceededError`, which reaches the judge unchanged so it
  splits the batch instead of a route failing over.
- ``unload()``: release the model; optional.

The hooks run one call at a time on a worker thread the adapter owns, since many runtimes (one
ONNX session, one GPU) are not safe to call concurrently while the judge asks from several
threads. Each caller waits on its own call, so ``cancel`` returns every waiting caller at once with
`CancelledError` and queued calls never run. A model call already running cannot be interrupted:
it finishes on the worker and its answer is dropped. ``timeout`` bounds each caller's wait, queue
included, with no limit by default. There is no endpoint, transport or retry to configure, and
the journal records the response as the model returned it, marked inexact.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import ClassVar

from ..judgments.answers import JevResponse, response_from_raw
from ..judgments.journal import RawResponse

# How long a caller blocks at a time while it waits for the model. Between slices the thread runs
# Python again, which is where a Ctrl-C is handled: one untimed wait can block on through a signal
# that lands just as it starts, and the search would then wait for the model to finish.
WAIT_SLICE_SECONDS = 0.05


class LocalModelClient:
    name: ClassVar[str] = "local"
    endpoint: ClassVar[str] = ""
    default_model: ClassVar[str] = ""
    api_key_env: ClassVar[str] = ""
    needs_key: ClassVar[bool] = False
    pinned: ClassVar[bool] = False
    runs_locally: ClassVar[bool] = True

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        transport=None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        given = {"endpoint": endpoint, "transport": transport, "max_retries": max_retries}
        refused = [setting for setting, value in given.items() if value is not None]
        if refused:
            raise ValueError(f"{self.name} runs in this process; it takes no {', '.join(refused)}")
        self.model = model or self.default_model
        if not self.model:
            raise ValueError(f"{self.name} needs a model")
        self.api_key = api_key or (os.environ.get(self.api_key_env, "").strip() if self.api_key_env else "")
        if self.needs_key and not self.api_key:
            raise RuntimeError(
                f"{self.api_key_env} is unset: export it or pass api_key"
                if self.api_key_env
                else f"{self.name} needs an API key: pass api_key"
            )
        self._timeout = timeout
        self._calls: queue.SimpleQueue[_Call | None] = queue.SimpleQueue()
        self._waiting: set[_Call] = set()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._cancelled = threading.Event()
        self._closed = False

    # --- the model: override in a subclass ------------------------------------------------------

    def load(self) -> None:
        pass

    def answer(self, state: Mapping, questions: Mapping) -> Mapping:
        raise NotImplementedError(f"{type(self).__name__} must implement answer()")

    def unload(self) -> None:
        pass

    # --- shared by every in-process adapter -----------------------------------------------------

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        call = _Call(state, questions)
        with self._lock:
            if self._cancelled.is_set():
                raise CancelledError
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            self._waiting.add(call)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name=f"jev-{self.name}", daemon=True)
                self._worker.start()
            self._calls.put(call)
        try:
            _wait_in_slices(call.done, self._timeout)
        finally:
            with self._lock:
                self._waiting.discard(call)
        if call.answered is not None:
            return RawResponse.from_decoded({"model": self.model, **call.answered})
        if call.error is not None:
            raise call.error
        call.abandoned = True
        if self._cancelled.is_set():
            raise CancelledError
        raise TimeoutError(f"{self.name} did not answer within {self._timeout} seconds")

    def parse(self, raw: RawResponse) -> JevResponse:
        return response_from_raw(raw.json())

    def cancel(self) -> None:
        """Return every waiting caller with `CancelledError` and refuse every later call."""
        with self._lock:
            self._cancelled.set()
            waiting = tuple(self._waiting)
        for call in waiting:
            call.done.set()

    def close(self) -> None:
        """Unload the model; waits for a running call unless the adapter was cancelled."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
            if worker is not None:
                self._calls.put(None)
        if worker is not None and not self._cancelled.is_set():
            worker.join()

    def _work(self) -> None:
        loaded = False
        try:
            while (call := self._calls.get()) is not None:
                if call.abandoned or self._cancelled.is_set():
                    call.done.set()
                    continue
                try:
                    if not loaded:
                        self.load()
                        loaded = True
                    call.answered = self.answer(call.state, call.questions)
                except Exception as error:  # noqa: BLE001 - handed to the caller
                    call.error = error
                finally:
                    call.done.set()
        finally:
            if loaded:
                self.unload()


def _wait_in_slices(done: threading.Event, timeout: float | None) -> None:
    """Wait until ``done`` is set or ``timeout`` seconds pass (None: no limit), in slices of at
    most `WAIT_SLICE_SECONDS` so a Ctrl-C is handled as soon as it arrives."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        remaining = WAIT_SLICE_SECONDS if deadline is None else deadline - time.monotonic()
        if remaining <= 0 or done.wait(min(remaining, WAIT_SLICE_SECONDS)):
            return


@dataclass(eq=False)
class _Call:
    state: Mapping
    questions: Mapping
    done: threading.Event = field(default_factory=threading.Event)
    answered: Mapping | None = None
    error: Exception | None = None
    abandoned: bool = False
