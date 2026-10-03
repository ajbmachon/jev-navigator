"""Batches of one judging call travel concurrently, within the call cap and cancellation."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pytest

from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS
from jev_navigator.judgments.journal import RawResponse
from jev_navigator.judgments.judge import CallCapReachedError, Judge
from jev_navigator.judgments.questions import Check, Criterion
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
    """Stands in for the provider: every send hangs until the judge cancels it. The last request to
    get in flight sends SIGINT to this process, as a terminal does on Ctrl-C."""

    in_flight_before_interrupt: int
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
        raise ConnectionError("cancelled by the judge" if self.cancelled else "never cancelled")

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
