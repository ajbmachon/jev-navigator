"""The in-process adapter base: a model loaded once and asked one call at a time on a worker the
adapter owns, whose waiting callers a cancelled search releases at once."""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor

import pytest

from jev_navigator.adapters.local import LocalModelClient

QUESTIONS = {"adds_one": {"type": "noul", "instructions": "Does `code` add one?"}}


class ToyModel(LocalModelClient):
    """Answers 0.9 to every question; ``gate`` holds each answer until it is set."""

    name = "toy"
    default_model = "toy@2026-09-30"

    def __init__(self, **settings) -> None:
        super().__init__(**settings)
        self.gate = threading.Event()
        self.gate.set()
        self.events: list[str] = []
        self.running = 0
        self.most_running = 0

    def load(self) -> None:
        self.events.append("load")

    def answer(self, state, questions):
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        self.events.append(f"answer {state['code']}")
        self.gate.wait()
        time.sleep(0.01)
        self.running -= 1
        if state["code"] == "raise":
            raise ValueError("the model failed")
        return {"answers": {question_id: {"type": "noul", "noul": 0.9} for question_id in questions}}

    def unload(self) -> None:
        self.events.append("unload")


def test_a_local_model_loads_once_answers_one_call_at_a_time_and_unloads_on_close():
    model = ToyModel()

    with ThreadPoolExecutor(max_workers=6) as pool:
        answers = list(pool.map(lambda code: model.ask({"code": code}, QUESTIONS), range(6)))
    model.close()

    assert [answer.model for answer in answers] == ["toy@2026-09-30"] * 6
    assert {answer.noul("adds_one").probability for answer in answers} == {0.9}
    assert model.most_running == 1
    assert (model.events[0], model.events.count("load"), model.events[-1]) == ("load", 1, "unload")


def test_a_model_error_reaches_its_caller_and_the_next_call_still_runs():
    model = ToyModel()

    with pytest.raises(ValueError, match="the model failed"):
        model.ask({"code": "raise"}, QUESTIONS)
    answer = model.ask({"code": "x + 1"}, QUESTIONS)
    model.close()

    assert answer.noul("adds_one").probability == 0.9


def test_cancel_releases_every_waiting_caller_at_once_and_queued_calls_never_run():
    # Arrange: one call running on the held model, one queued behind it.
    model = ToyModel()
    model.gate.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            running = pool.submit(model.ask, {"code": "running"}, QUESTIONS)
            _wait_for(lambda: "answer running" in model.events)
            queued = pool.submit(model.ask, {"code": "queued"}, QUESTIONS)
            time.sleep(0.05)

            # Act
            model.cancel()

            # Assert
            with pytest.raises(CancelledError):
                running.result(timeout=1)
            with pytest.raises(CancelledError):
                queued.result(timeout=1)
        finally:
            model.gate.set()
    with pytest.raises(CancelledError):
        model.ask({"code": "later"}, QUESTIONS)
    model.close()
    _wait_for(lambda: "unload" in model.events)
    assert model.events == ["load", "answer running", "unload"]


def test_a_caller_stops_waiting_after_its_timeout():
    model = ToyModel(timeout=0.05)
    model.gate.clear()
    release = threading.Timer(2, model.gate.set)
    release.start()

    try:
        with pytest.raises(TimeoutError, match="toy did not answer within 0.05 seconds"):
            model.ask({"code": "slow"}, QUESTIONS)
    finally:
        release.cancel()
        model.gate.set()
        model.close()


@pytest.mark.parametrize("setting", [{"endpoint": "http://gpu-box:8000"}, {"max_retries": 2}])
def test_a_local_model_takes_no_endpoint_or_retries(setting):
    with pytest.raises(ValueError, match=f"toy runs in this process; it takes no {next(iter(setting))}"):
        ToyModel(**setting)


def _wait_for(condition, seconds: float = 2) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.001)
