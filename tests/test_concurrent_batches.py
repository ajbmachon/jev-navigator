"""Batches of one judging call travel concurrently, within the call cap and cancellation."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS, InputBudgetExceededError
from jev_navigator.judgments.journal import JsonlJournal, RawResponse
from jev_navigator.judgments.judge import CallCapReachedError, Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import ScriptedJevClient

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
SHARED = {"doc": {"sentence": "s"}}
ONE_ITEM_PER_BATCH_CHARS = JEV_INPUT_BOX_CHARS * 3 // 5
SMALL_ITEMS = [{"file": f"f{index}.py", "code": f"v = {index}"} for index in range(8)]


@dataclass
class OverlapClient:
    """Stands in for the remote provider: records every request and how many were in flight. The
    first ``first_wave`` sends wait at a barrier until all of them are in flight together, so a
    judge that sends them one after another breaks the barrier instead of passing by timing."""

    first_wave: int = 1
    when_together: Callable[[], None] | None = None
    script: ScriptedJevClient = field(default_factory=lambda: ScriptedJevClient(default_noul=0.9))
    in_flight: int = 0
    peak: int = 0
    arrivals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._together = threading.Barrier(self.first_wave, action=self.when_together, timeout=5)

    @property
    def model(self) -> str:
        return self.script.model

    @property
    def requests(self) -> list:
        return self.script.requests

    def ask(self, state: Mapping, questions: Mapping):
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            arrival, self.arrivals = self.arrivals, self.arrivals + 1
        if arrival < self.first_wave:
            self._together.wait()
        with self._lock:
            self.in_flight -= 1
        return self.script.send(state, questions)

    def parse(self, raw: RawResponse):
        return self.script.parse(raw)


def _items(count: int) -> list[dict]:
    return [
        {"file": f"f{index}.py", "code": f"# {index}\n" + "x" * ONE_ITEM_PER_BATCH_CHARS}
        for index in range(count)
    ]


def test_batches_of_one_call_are_in_flight_together_up_to_the_concurrency_limit() -> None:
    # Arrange
    client = OverlapClient(first_wave=4)
    judge = Judge(client, max_concurrency=4)

    # Act
    results = judge.check_each(DESCRIBES, _items(10), SHARED)

    # Assert
    assert len(client.requests) == 10
    assert client.peak == 4
    assert [result.item["file"] for result in results] == [f"f{index}.py" for index in range(10)]


def test_streamed_batches_overlap_and_every_item_is_answered_once() -> None:
    # Arrange
    client = OverlapClient(first_wave=6)
    judge = Judge(client, max_concurrency=16)

    # Act
    answered = [result.item["file"] for result in judge.iter_check_each(DESCRIBES, _items(6), SHARED)]

    # Assert
    assert client.peak == 6
    assert sorted(answered) == sorted(f"f{index}.py" for index in range(6))


def test_the_call_cap_stays_exact_under_concurrency_and_answered_batches_still_yield() -> None:
    # Arrange
    client = OverlapClient()
    judge = Judge(client, max_calls=3, max_concurrency=8)
    answered: list[str] = []

    # Act
    with pytest.raises(CallCapReachedError):
        for result in judge.iter_check_each(DESCRIBES, _items(8), SHARED):
            answered.append(result.item["file"])

    # Assert
    assert len(client.requests) == 3
    assert judge.calls == 3
    assert len(answered) == 3


def test_cancellation_stops_every_batch_that_has_not_started_sending() -> None:
    # Arrange
    stop = threading.Event()
    client = OverlapClient(first_wave=2, when_together=stop.set)
    judge = Judge(client, max_concurrency=2)

    # Act
    answered = list(judge.iter_check_every([DESCRIBES], _items(8), SHARED, cancelled=stop.is_set))

    # Assert
    assert len(client.requests) == 2, "only the two requests already in flight finish"
    assert len(answered) == len(client.requests)


def test_a_concurrency_of_one_sends_batches_one_after_another() -> None:
    # Arrange
    client = OverlapClient()
    judge = Judge(client, max_concurrency=1)

    # Act
    judge.check_each(DESCRIBES, _items(3), SHARED)

    # Assert
    assert client.peak == 1
    assert len(client.requests) == 3


def test_a_provider_failure_stops_batches_that_have_not_started_and_raises_once_drained() -> None:
    # Arrange
    client = OverlapClient()
    judge = Judge(client, max_concurrency=1)
    original_send = client.send

    def fail_first(state: Mapping, questions: Mapping) -> RawResponse:
        if not client.requests:
            client.requests.append((state, questions))
            raise ConnectionError("provider unavailable")
        return original_send(state, questions)

    client.send = fail_first

    # Act / Assert
    with pytest.raises(ConnectionError):
        judge.check_each(DESCRIBES, _items(5), SHARED)
    assert len(client.requests) == 1


def test_a_capped_call_always_answers_the_first_batches_in_their_stable_order() -> None:
    # Arrange
    items = SMALL_ITEMS
    answered_runs = []

    # Act
    for _ in range(20):
        client = ScriptedJevClient(default_noul=0.9)
        answered: list[str] = []
        with pytest.raises(CallCapReachedError):
            judge = Judge(client, max_calls=3, max_concurrency=8, items_per_request=1)
            for result in judge.iter_check_each(DESCRIBES, items, SHARED):
                answered.append(result.item["file"])
        answered_runs.append(sorted(answered))

    # Assert
    assert len({tuple(run) for run in answered_runs}) == 1


def _interrupt() -> None:
    os.kill(os.getpid(), signal.SIGINT)


@dataclass
class HangingClient:
    """Stands in for the provider: every send hangs until the judge cancels it, then raises
    ``on_cancel``: by default the ``CancelledError`` the TypeSafe adapter raises for an aborted send.
    The last request to get in flight sends SIGINT to this process, as a terminal does on Ctrl-C."""

    in_flight_before_interrupt: int
    on_cancel: Exception = field(default_factory=CancelledError)
    script: ScriptedJevClient = field(default_factory=ScriptedJevClient)
    released: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    timed_out: bool = False

    def __post_init__(self) -> None:
        self.arrived = threading.Barrier(self.in_flight_before_interrupt, action=_interrupt, timeout=5)

    @property
    def model(self) -> str:
        return self.script.model

    @property
    def requests(self) -> list:
        return self.script.requests

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        self.script.requests.append((state, questions))
        self.arrived.wait()
        if not self.released.wait(timeout=5):
            self.timed_out = True
        if self.cancelled:
            raise self.on_cancel
        raise ConnectionError("never cancelled")

    def parse(self, raw: RawResponse):
        return self.script.parse(raw)

    def cancel(self) -> None:
        self.cancelled = True
        self.released.set()


@pytest.mark.usefixtures("python_sigint_handler")
def test_an_interrupt_with_requests_in_flight_cancels_them_and_sends_nothing_new() -> None:
    # Arrange
    client = HangingClient(in_flight_before_interrupt=2)
    judge = Judge(client, max_concurrency=2)

    # Act
    with pytest.raises(KeyboardInterrupt):
        list(judge.iter_check_each(DESCRIBES, _items(6), SHARED))

    # Assert: the cancel released the hanging sends; none ran into its timeout
    assert client.cancelled and not client.timed_out
    assert len(client.requests) == 2


class ProviderError(RuntimeError):
    """A failure the provider reports, such as a 503, as opposed to a send the caller aborted."""


@pytest.mark.usefixtures("python_sigint_handler")
def test_a_provider_error_during_an_interrupt_comes_out_as_that_error() -> None:
    # Arrange
    cause = ConnectionResetError("connection reset by peer")
    error = ProviderError("Jev answered 503")
    error.__cause__ = cause
    client = HangingClient(in_flight_before_interrupt=2, on_cancel=error)
    judge = Judge(client, max_concurrency=2)

    # Act: the interrupt is caught too, so hiding the error fails this test instead of ending the session
    try:
        list(judge.iter_check_each(DESCRIBES, _items(6), SHARED))
    except (ProviderError, KeyboardInterrupt) as stopped:
        raised = stopped

    # Assert
    assert client.cancelled and not client.timed_out
    assert raised is error
    assert raised.__cause__ is cause


@dataclass
class CountingAsyncClient:
    """Stands in for the provider on the async path: records how many sends overlap."""

    script: ScriptedJevClient = field(default_factory=lambda: ScriptedJevClient(default_noul=0.9))
    in_flight: int = 0
    peak: int = 0

    @property
    def model(self) -> str:
        return self.script.model

    @property
    def requests(self) -> list:
        return self.script.requests

    async def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        for _ in range(3):
            await asyncio.sleep(0)
        self.in_flight -= 1
        return self.script.send(state, questions)

    def parse(self, raw: RawResponse):
        return self.script.parse(raw)


def test_the_async_path_keeps_at_most_max_concurrency_requests_in_flight() -> None:
    # Arrange
    client = CountingAsyncClient()
    judge = Judge(client, max_concurrency=2)

    # Act
    asyncio.run(judge.check_each_async(DESCRIBES, _items(6), SHARED))

    # Assert
    assert len(client.requests) == 6
    assert client.peak == 2


def test_a_capped_async_call_always_answers_the_first_batches_in_their_stable_order() -> None:
    # Arrange
    answered_runs = []

    # Act
    for _ in range(20):
        client = CountingAsyncClient()
        with pytest.raises(CallCapReachedError):
            asyncio.run(
                Judge(client, max_calls=3, max_concurrency=8, items_per_request=1).check_each_async(
                    DESCRIBES, SMALL_ITEMS, SHARED
                )
            )
        answered_runs.append(
            tuple(sorted(item["file"] for state, _ in client.requests for item in state["items"]))
        )

    # Assert
    assert len(set(answered_runs)) == 1
    assert len(answered_runs[0]) == 3


class HoldingScanner:
    """A pre-send scanner that finds no secret but holds the request carrying ``held`` until
    ``release`` returns true or ten seconds pass, so a test can decide which batch reserves a call
    first."""

    def __init__(self, held: Callable[[str], bool], release: Callable[[], bool]) -> None:
        self.held = held
        self.release = release

    def findings(self, text: str) -> list[str]:
        if self.held(text):
            deadline = threading.Event()
            for _ in range(1000):
                if self.release():
                    break
                deadline.wait(0.01)
        return []


def _answered_files(client: ScriptedJevClient) -> list[str]:
    return sorted(item["file"] for state, _ in client.requests for item in state["items"])


def test_a_capped_call_answers_its_first_batches_even_when_the_first_is_held_back() -> None:
    # Arrange: three calls for eight one-item batches; f0 waits until two other requests were sent
    client = ScriptedJevClient(default_noul=0.9)
    scanner = HoldingScanner(lambda text: text == "f0.py", lambda: len(client.requests) >= 2)
    judge = Judge(client, scanner=scanner, max_calls=3, max_concurrency=8, items_per_request=1)

    # Act
    with pytest.raises(CallCapReachedError):
        judge.check_each(DESCRIBES, SMALL_ITEMS, SHARED)

    # Assert: the waves keep later batches from taking the calls of the first three
    assert _answered_files(client) == ["f0.py", "f1.py", "f2.py"]


class RefusesThePairWithF0(ScriptedJevClient):
    """Refuses, for its input size, the one request that carries f0 together with another item."""

    refused: threading.Event

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        files = [item["file"] for item in state["items"]]
        if "f0.py" in files and len(files) > 1:
            self.refused.set()
            raise InputBudgetExceededError("max_tokens_exceeded")
        return super().send(state, questions)


def test_a_batch_split_in_a_capped_wave_never_takes_a_call_from_its_wave() -> None:
    # Arrange: three calls for three two-item batches; the f0 pair is refused for its size, and the
    # other two batches reserve their calls only after that refusal
    client = RefusesThePairWithF0(default_noul=0.9)
    client.refused = threading.Event()
    scanner = HoldingScanner(lambda text: text in ("f2.py", "f4.py"), client.refused.is_set)
    judge = Judge(client, scanner=scanner, max_calls=3, max_concurrency=8, items_per_request=2)

    # Act
    with pytest.raises(CallCapReachedError):
        judge.check_each(DESCRIBES, SMALL_ITEMS[:6], SHARED)

    # Assert: the halves of the refused pair wait for the next wave, which has no call left
    assert _answered_files(client) == ["f2.py", "f3.py", "f4.py", "f5.py"]


@pytest.mark.parametrize("setting", ["max_concurrency", "items_per_request"])
def test_a_judge_refuses_a_batch_setting_below_one(setting: str) -> None:
    # Act and Assert: zero would hang the async path and make no batch at all
    with pytest.raises(ValueError, match=f"{setting} must be at least 1"):
        Judge(ScriptedJevClient(), **{setting: 0})


class FailsF2Late(ScriptedJevClient):
    """A provider whose request carrying f2 fails, the way a 503 does after its retries, only once
    ``after`` is set and the other requests have had a moment to settle."""

    after: threading.Event

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        if any(item.get("file") == "f2.py" for item in state.get("items", [])):
            self.after.wait(10)
            threading.Event().wait(0.2)
            raise ConnectionError("503 Service Unavailable")
        return super().send(state, questions)


def test_a_provider_failure_is_raised_over_a_call_cap_reached_in_the_same_wave() -> None:
    # Arrange: three calls for three batches; another caller of the same judge takes one call while
    # f0 is held, so f0, first in the wave, meets the cap before f2's provider fails
    client = FailsF2Late(default_noul=0.9)
    other_caller_done = threading.Event()
    client.after = other_caller_done
    scanner = HoldingScanner(lambda text: text == "f0.py", other_caller_done.is_set)
    parent = Judge(client, scanner=scanner, max_calls=3, items_per_request=1)

    def other_caller() -> None:
        parent.scope().ask(
            {"other": "state"}, {"q": {"type": "noul", "instructions": "y?"}}, thresholds=parent.thresholds
        )
        other_caller_done.set()

    threading.Timer(0.05, other_caller).start()

    # Act
    with pytest.raises(ConnectionError) as raised:
        parent.scope().check_each(DESCRIBES, SMALL_ITEMS[:3], SHARED)

    # Assert: the real failure comes out, and the cap it shadowed is named beside it
    assert "503" in str(raised.value)
    assert any("CallCapReachedError" in note for note in getattr(raised.value, "__notes__", []))


@dataclass
class OneFailingAsyncClient:
    """An async provider whose request for ``f0.py`` fails once the other requests of its wave have
    started, while they are still in flight; it records each request that starts and each that
    finishes."""

    script: ScriptedJevClient = field(default_factory=lambda: ScriptedJevClient(default_noul=0.9))
    started: list[str] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.script.model

    async def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        file = state["items"][0]["file"]
        self.started.append(file)
        if file == "f0.py":
            await asyncio.sleep(0)
            raise ConnectionError("the provider is down")
        for _ in range(3):
            await asyncio.sleep(0)
        self.finished.append(file)
        return self.script.send(state, questions)

    def parse(self, raw: RawResponse):
        return self.script.parse(raw)


def test_an_async_batch_failure_is_raised_only_after_the_rest_of_its_wave_settles(tmp_path: Path) -> None:
    # Arrange
    client = OneFailingAsyncClient()
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    journal = JsonlJournal(tmp_path / "journal.jsonl")
    judge = Judge(
        client,
        store=store,
        journal=journal,
        served_model=client.model,
        max_concurrency=3,
        items_per_request=1,
    )
    finished_when_raised: list[str] = []

    async def judged_then_given_time() -> None:
        try:
            await judge.check_each_async(DESCRIBES, SMALL_ITEMS[:3], SHARED)
        except ConnectionError:
            finished_when_raised.extend(client.finished)
            raise
        finally:
            for _ in range(10):
                await asyncio.sleep(0)

    # Act
    with pytest.raises(ConnectionError, match="the provider is down"):
        asyncio.run(judged_then_given_time())

    # Assert: no request of the failed call was still running when the failure came out, and every
    # answer that came back after the failure is journaled, stored for a resume and counted
    assert sorted(finished_when_raised) == sorted(client.finished) == ["f1.py", "f2.py"]
    assert _answers_kept(journal, store, client.model) == (2, 2, 1)
    assert judge.input_total.reported == 2 * 100


def _answers_kept(journal: JsonlJournal, store: JsonlAnswerStore, model: str) -> tuple[int, int, int]:
    """How many requests the journal shows answered, how many of those the store replays, and how
    many failed."""
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    hashes = {row["request_id"]: row["request_sha256"] for row in rows if row["kind"] == "request"}
    answered = [hashes[row["request_id"]] for row in rows if row["kind"] == "response"]
    stored = [digest for digest in answered if store.by_request(digest, model) is not None]
    return len(answered), len(stored), sum(row["kind"] == "failure" for row in rows)


def test_an_async_batch_failure_stops_the_batches_still_waiting_for_a_slot() -> None:
    # Arrange: one slot, so f1 and f2 wait while f0 fails
    client = OneFailingAsyncClient()
    judge = Judge(client, max_concurrency=1, items_per_request=1)

    # Act
    with pytest.raises(ConnectionError, match="the provider is down"):
        asyncio.run(judge.check_each_async(DESCRIBES, SMALL_ITEMS[:3], SHARED))

    # Assert
    assert client.started == ["f0.py"]
