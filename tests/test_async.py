from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pytest

from jev_navigator.directives.find_code import Outcome, SearchBudget, StopRule, find_code, find_code_async
from jev_navigator.directives.places import place_for_line
from jev_navigator.history import History, judge_history_async
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import CallCapReachedError, CallOffer, Judge
from jev_navigator.judgments.questions import Check, Criterion, Pick, Rate
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
NAMES_VALUE = Check(
    name="names_value",
    instructions="Does `doc.sentence` name a value?",
    yes=Criterion("The sentence names a value."),
    no=Criterion("The sentence names no value."),
)
HOLDS_LIMIT = Check(
    "holds_limit_check",
    "Does `fetched` contain code that compares the number of items with a limit?",
    Criterion("A code body in `fetched` compares an item count with a limit."),
    Criterion("No code body in `fetched` makes that comparison."),
)
TARGET = "the check that limits how many items an order may have"
SHARED = {"doc": {"sentence": "s"}}


def run(coroutine):
    return asyncio.run(coroutine)


BIG_ITEM = "1" * (JEV_INPUT_BOX_CHARS * 3 // 5)
"""Too big for two to share a request, small enough to send alone."""
THREE_BATCH_ITEMS = [{"code": f"x{index} = {BIG_ITEM}"} for index in range(3)]


class OverlappingAsyncClient:
    """Wraps an ``AsyncScriptedJevClient`` and records how many sends were in flight at once."""

    def __init__(self, script: ScriptedJevClient) -> None:
        self.wrapped = AsyncScriptedJevClient(script)
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def model(self) -> str:
        return self.wrapped.model

    async def send(self, state: dict, questions: dict):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            return await self.wrapped.send(state, questions)
        finally:
            self.in_flight -= 1

    def parse(self, raw):
        return self.wrapped.parse(raw)


def limit_check_answers(question_id: str, question: dict, state: dict) -> float:
    if "slice" in state and question_id.startswith("contains_target"):
        return 0.95 if "len(order.items) <= limit" in state["slice"]["code"] else 0.05
    return 0.9 if "check_limits" in str(state.get("candidates", "")) else 0.4


def test_the_async_path_masks_stores_and_journals_exactly_like_the_sync_path(tmp_path: Path) -> None:
    # Arrange
    items = [{"code": f'TOKEN = "ghp_{"a1" * 18}"'}, {"code": "y = 2"}]
    sync_client = ScriptedJevClient(nouls={"describes": 0.9})
    async_client = AsyncScriptedJevClient(ScriptedJevClient(nouls={"describes": 0.9}))
    sync_judge = Judge(sync_client, store=JsonlAnswerStore(tmp_path / "sync.jsonl"))
    async_judge = Judge(
        async_client,
        store=JsonlAnswerStore(tmp_path / "async.jsonl"),
        journal=JsonlJournal(tmp_path / "journal.jsonl"),
    )

    # Act
    expected = sync_judge.check_each(DESCRIBES, items, SHARED)
    results = run(async_judge.check_each_async(DESCRIBES, items, SHARED))
    replayed = run(async_judge.check_each_async(DESCRIBES, items, SHARED))

    # Assert
    assert [(r.probability, r.verdict, r.request_sha256) for r in results] == [
        (r.probability, r.verdict, r.request_sha256) for r in expected
    ]
    assert async_client.requests == sync_client.requests
    assert "ghp_" not in str(async_client.requests)
    assert all(result.from_store for result in replayed) and async_judge.calls == 1
    journaled = (tmp_path / "journal.jsonl").read_text().splitlines()
    assert ['"kind": "request"' in line for line in journaled] == [True, False]


def test_ask_all_pick_and_choose_call_have_async_forms() -> None:
    # Arrange
    client = AsyncScriptedJevClient(
        ScriptedJevClient(choices={"kind": {"why": 0.8, "what": 0.2}}, scores={"useful": [0.1, 0.9]})
    )
    judge = Judge(client)
    kind = Pick("kind", "What kind of sentence is `doc.sentence`?")
    usefulness = Rate("useful", "How much does `doc.sentence` add?", ("Nothing.", "Something."))
    offers = [CallOffer("search_text", "lines with a key", Pick("key", "Which key?"), {"k": "1 hit"})]

    # Act
    everything = run(
        judge.ask_all_async(
            SHARED, checks=[NAMES_VALUE], picks=[(kind, {"why": "", "what": ""})], scores=[usefulness]
        )
    )
    picked = run(judge.pick_async(kind, {"why": "", "what": ""}, SHARED))
    decision = run(judge.choose_call_async(Pick("route", "Which lookup?"), offers, SHARED))

    # Assert
    assert everything.picks["kind"].choice == "why" and everything.scores["useful"].score == pytest.approx(
        0.9
    )
    assert picked.choice == "why"
    assert (decision.operation, decision.argument) == ("search_text", "k")
    assert len(client.requests) == 3


def test_a_sync_call_with_an_async_client_is_refused() -> None:
    judge = Judge(AsyncScriptedJevClient())
    with pytest.raises(TypeError, match="async"):
        judge.check_each(DESCRIBES, [{"code": "x = 1"}], SHARED)


def test_the_async_path_counts_calls_against_the_same_caps() -> None:
    # Arrange
    judge = Judge(AsyncScriptedJevClient(ScriptedJevClient(nouls={"describes": 0.9})), max_calls=1)
    run(judge.check_each_async(DESCRIBES, [{"code": "x = 1"}], SHARED))

    # Act and Assert
    with pytest.raises(CallCapReachedError):
        run(judge.check_each_async(DESCRIBES, [{"code": "y = 2"}], SHARED))


def test_find_code_async_searches_like_find_code(sample_index: CodeIndex) -> None:
    # Arrange
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]
    budget = SearchBudget(beam_width=2)
    sync_client = ScriptedJevClient(nouls=limit_check_answers)
    async_client = AsyncScriptedJevClient(ScriptedJevClient(nouls=limit_check_answers))

    # Act
    expected = find_code(sample_index, Judge(sync_client), TARGET, start, budget=budget)
    result = run(find_code_async(sample_index, Judge(async_client), TARGET, start, budget=budget))

    # Assert
    assert result.outcome == expected.outcome == Outcome.FOUND
    assert result.found[0].code.span.name == "check_limits"
    assert (result.steps, result.calls) == (expected.steps, expected.calls)
    assert sorted(map(str, async_client.requests)) == sorted(map(str, sync_client.requests))


def test_find_code_async_keeps_the_event_loop_running_while_a_place_is_opened(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    start = place_for_line(sample_index, "app/orders.py", 6, "start")
    opening: list[float] = []

    def slow_open():
        opening.append(time.monotonic())
        time.sleep(0.2)
        opening.append(time.monotonic())
        return start.open()

    judge = Judge(AsyncScriptedJevClient(ScriptedJevClient(nouls=limit_check_answers)))
    ticks: list[float] = []

    async def search_beside_a_ticker():
        async def tick():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        try:
            return await find_code_async(
                sample_index, judge, TARGET, [replace(start, open=slow_open)], moves={}
            )
        finally:
            ticker.cancel()

    # Act
    result = run(search_beside_a_ticker())

    # Assert
    began, ended = opening[0], opening[1]
    assert result.steps == 1
    assert len([tick for tick in ticks if began < tick < ended]) >= 2


def test_find_code_async_applies_the_stop_rule_through_the_async_history_check(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    def answers(question_id: str, question: dict, state: dict) -> float:
        if "fetched" in state:
            bodies = [span["code"] for span in state["fetched"]]
            return 0.9 if any("<= limit" in body for body in bodies) else 0.1
        return 0.3

    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]
    judge = Judge(AsyncScriptedJevClient(ScriptedJevClient(nouls=answers)))

    # Act
    result = run(
        find_code_async(
            sample_index,
            judge,
            TARGET,
            start,
            budget=SearchBudget(beam_width=1),
            stop_rule=StopRule(HOLDS_LIMIT),
        )
    )

    # Assert
    assert result.outcome == Outcome.STOP_RULE


def test_the_first_async_check_each_replays_the_remaining_batches_from_the_store(
    tmp_path: Path,
) -> None:
    # Arrange: a prior run fills the store; the served model is unknown until a live answer arrives.
    path = tmp_path / "answers.jsonl"
    warm_client = ScriptedJevClient(nouls={"describes": 0.9})
    Judge(warm_client, store=JsonlAnswerStore(path)).check_each(DESCRIBES, THREE_BATCH_ITEMS, SHARED)
    assert len(JsonlAnswerStore(path).records()) == 3
    client = OverlappingAsyncClient(ScriptedJevClient(nouls={"describes": 0.9}))
    judge = Judge(client, store=JsonlAnswerStore(path))

    # Act
    results = run(judge.check_each_async(DESCRIBES, THREE_BATCH_ITEMS, SHARED))

    # Assert: the first batch pins the served model, the other batches replay for free.
    assert judge.calls == 1
    assert sorted(result.from_store for result in results) == [False, True, True]
    assert all(result.probability == 0.9 for result in results)
    # Replayed batches never send, so only the live first batch can be in flight.
    assert client.max_in_flight == 1
    # The one live request is byte-for-byte the request the sequential path sent for that batch.
    assert client.wrapped.requests == warm_client.requests[:1]


def test_the_first_async_check_each_stays_all_parallel_when_the_served_model_is_known(
    tmp_path: Path,
) -> None:
    # Arrange
    client = OverlappingAsyncClient(ScriptedJevClient(nouls={"describes": 0.9}))
    judge = Judge(client, store=JsonlAnswerStore(tmp_path / "answers.jsonl"), served_model="jev-scripted")

    # Act
    run(judge.check_each_async(DESCRIBES, THREE_BATCH_ITEMS, SHARED))

    # Assert
    assert judge.calls == 3 and client.max_in_flight == 3


def test_async_check_each_without_an_answer_store_keeps_all_batches_parallel() -> None:
    client = OverlappingAsyncClient(ScriptedJevClient(nouls={"describes": 0.9}))
    judge = Judge(client)

    results = run(judge.check_each_async(DESCRIBES, THREE_BATCH_ITEMS, SHARED))

    assert judge.calls == 3 and client.max_in_flight == 3
    assert [result.probability for result in results] == [0.9, 0.9, 0.9]


def test_the_first_async_check_each_rejects_store_answers_from_a_changed_served_model(
    tmp_path: Path,
) -> None:
    # Arrange: the store holds answers from an older model version.
    path = tmp_path / "answers.jsonl"
    Judge(
        ScriptedJevClient(nouls={"describes": 0.9}, model="jev-1.12.0"), store=JsonlAnswerStore(path)
    ).check_each(DESCRIBES, THREE_BATCH_ITEMS, SHARED)
    client = OverlappingAsyncClient(ScriptedJevClient(nouls={"describes": 0.1}, model="jev-1.13.0"))
    judge = Judge(client, store=JsonlAnswerStore(path))

    # Act
    results = run(judge.check_each_async(DESCRIBES, THREE_BATCH_ITEMS, SHARED))

    # Assert: once the first live answer pins the new served model, the older records are refused.
    assert judge.calls == 3
    assert [result.probability for result in results] == [0.1, 0.1, 0.1]
    assert all(result.from_store is False for result in results)


def test_concurrent_first_async_calls_do_not_trust_the_store_across_models(tmp_path: Path) -> None:
    # Arrange: the store holds answers from an older model for two unrelated item lists; a new
    # model serves two concurrent first calls over the same judge and store.
    path = tmp_path / "answers.jsonl"
    other_items = [{"code": "y = 2"}, {"code": "z = 3"}]
    warm = Judge(
        ScriptedJevClient(nouls={"describes": 0.9, "names_value": 0.9}, model="jev-1.12.0"),
        store=JsonlAnswerStore(path),
    )
    warm.check_each(DESCRIBES, THREE_BATCH_ITEMS, SHARED)
    warm.check_each(NAMES_VALUE, other_items, SHARED)
    assert len(JsonlAnswerStore(path).records()) == 4
    client = OverlappingAsyncClient(ScriptedJevClient(nouls={"describes": 0.1, "names_value": 0.2}))
    judge = Judge(client, store=JsonlAnswerStore(path))

    async def both():
        return await asyncio.gather(
            judge.check_each_async(DESCRIBES, THREE_BATCH_ITEMS, SHARED),
            judge.check_each_async(NAMES_VALUE, other_items, SHARED),
        )

    # Act
    first, second = run(both())

    # Assert: every batch goes live under the new model, also on the call whose _prepare ran after
    # the other call had already pinned the served model.
    assert judge.calls == 4
    assert all(result.from_store is False for result in [*first, *second])
    assert [result.probability for result in first] == [0.1, 0.1, 0.1]
    assert [result.probability for result in second] == [0.2, 0.2]


def test_the_async_history_check_matches_the_sync_one() -> None:
    # Arrange
    history = History()
    judge = Judge(AsyncScriptedJevClient(ScriptedJevClient(default_noul=0.9)))

    # Act
    judged = run(judge_history_async(judge, history, HOLDS_LIMIT))

    # Assert
    assert judged.probability == 0.9
    assert history.previous_judgments["holds_limit_check"] == judged


TWO_ITEMS = [{"code": "x = 1"}, {"code": "y = 2"}]


def test_the_first_async_batch_of_check_every_asks_each_item_once_per_check(
    tmp_path: Path,
) -> None:
    """Two checks over two items are two item slots and four questions. The model-pinning first
    batch once sent four copies of the two items and eight questions, duplicating the answers too."""
    # Arrange: a warm store whose answers stay unusable until a live answer pins the served model.
    path = tmp_path / "answers.jsonl"
    warm = Judge(
        ScriptedJevClient(nouls={"describes": 0.9, "names_value": 0.9}), store=JsonlAnswerStore(path)
    )
    warm.check_every([DESCRIBES, NAMES_VALUE], TWO_ITEMS, SHARED)
    client = OverlappingAsyncClient(ScriptedJevClient(nouls={"describes": 0.9, "names_value": 0.9}))
    judge = Judge(client, store=JsonlAnswerStore(path))

    # Act
    results = run(judge.check_every_async([DESCRIBES, NAMES_VALUE], TWO_ITEMS, SHARED))

    # Assert: the first batch deduplicates its item positions exactly like the sibling senders.
    state, questions = client.wrapped.requests[0]
    assert len(state["items"]) == 2
    assert len(questions) == 4
    assert judge.calls == 1
    for name in ("describes", "names_value"):
        assert [result.probability for result in results[name]] == [0.9, 0.9]
