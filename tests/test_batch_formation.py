"""Batches form by count over a stable item order, and an answer is reused only with its batch mates."""

from __future__ import annotations

from pathlib import Path

from conftest import BudgetedClient

from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS, MAX_REQUEST_CHARS
from jev_navigator.judgments.judge import Judge, request_exceeds_input_budget
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import ScriptedJevClient

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
CHANGES = Check(
    name="changes",
    instructions="Does `{item}.code` change a value?",
    yes=Criterion("It changes a value."),
    no=Criterion("It does not change a value."),
)
SHARED = {"doc": {"sentence": "s"}}


def _items(count: int, code_chars: int = 10) -> list[dict]:
    return [{"file": f"f{index}.py", "code": f"v{index} = '{'x' * code_chars}'"} for index in range(count)]


def _sent_members(client: ScriptedJevClient) -> list[list[str]]:
    return [[item["file"] for item in state["items"]] for state, _ in client.requests]


def test_sixteen_small_items_travel_in_one_request() -> None:
    # Arrange
    client = ScriptedJevClient()

    # Act
    Judge(client).check_each(DESCRIBES, _items(16), SHARED)

    # Assert
    assert len(client.requests) == 1
    assert len(client.requests[0][1]) == 16


def test_the_seventeenth_item_opens_a_second_request() -> None:
    # Arrange
    client = ScriptedJevClient()

    # Act
    Judge(client).check_each(DESCRIBES, _items(17), SHARED)

    # Assert
    assert sorted(len(members) for members in _sent_members(client)) == [1, 16]


def test_the_items_per_request_are_configurable() -> None:
    # Arrange
    client = ScriptedJevClient()

    # Act
    Judge(client, items_per_request=4).check_each(DESCRIBES, _items(10), SHARED)

    # Assert
    assert sorted(len(members) for members in _sent_members(client)) == [2, 4, 4]


def test_the_same_population_forms_the_same_batches_in_any_input_order() -> None:
    # Arrange
    items = _items(40)
    forward, backward = ScriptedJevClient(), ScriptedJevClient()

    # Act
    Judge(forward).check_each(DESCRIBES, items, SHARED)
    Judge(backward).check_each(DESCRIBES, items[::-1], SHARED)

    # Assert
    assert sorted(_sent_members(forward)) == sorted(_sent_members(backward))
    assert sorted(forward.requests, key=str) == sorted(backward.requests, key=str)


def test_items_too_large_to_share_a_request_close_the_batch_early_at_the_character_box() -> None:
    # Arrange: two items fit the box beside each other, a third does not
    client = BudgetedClient(MAX_REQUEST_CHARS, input_box=JEV_INPUT_BOX_CHARS)

    # Act
    Judge(client).check_each(DESCRIBES, _items(3, code_chars=JEV_INPUT_BOX_CHARS * 2 // 5), SHARED)

    # Assert: the first two in stable order share a request; halving would have sent f0 alone
    assert sorted([item["file"] for item in state["items"]] for state, _ in client.requests) == [
        ["f0.py", "f1.py"],
        ["f2.py"],
    ]
    assert client.refusals == 0
    assert all(not request_exceeds_input_budget(state, questions) for state, questions in client.requests)


def test_long_question_wording_closes_the_batch_at_the_whole_request_box() -> None:
    # Arrange: tiny items, but each item's question is long enough that 16 of them overflow the body box
    long_check = Check(
        name="long",
        instructions="Is `{item}.code` what `doc.sentence` describes? " + "Read every detail. " * 550,
        yes=Criterion("Yes."),
        no=Criterion("No."),
    )
    client = BudgetedClient(MAX_REQUEST_CHARS, input_box=JEV_INPUT_BOX_CHARS)

    # Act
    Judge(client).check_each(long_check, _items(16), SHARED)

    # Assert: the first request is filled as far as the body box allows, not halved
    sizes = sorted((len(state["items"]) for state, _ in client.requests), reverse=True)
    assert len(sizes) == 2 and sizes[0] > 8
    assert client.refusals == 0
    assert all(not request_exceeds_input_budget(state, questions) for state, questions in client.requests)


def test_an_answer_is_reused_only_inside_the_same_batch(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    pair = [{"code": "x = 1"}, {"code": "y = 2"}]
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(path)).check_each(DESCRIBES, pair, SHARED)
    alone, again = ScriptedJevClient(), ScriptedJevClient()

    # Act
    alone_results = Judge(alone, store=JsonlAnswerStore(path), served_model="jev-scripted").check_each(
        DESCRIBES, pair[:1], SHARED
    )
    again_results = Judge(again, store=JsonlAnswerStore(path), served_model="jev-scripted").check_each(
        DESCRIBES, pair[::-1], SHARED
    )

    # Assert
    assert alone_results[0].from_store is False and len(alone.requests) == 1
    assert all(result.from_store for result in again_results) and again.requests == []


def test_a_request_carries_every_batch_mate_and_asks_only_the_open_questions(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    pair = [{"code": "x = 1"}, {"code": "y = 2"}]
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(path)).check_each(DESCRIBES, pair, SHARED)
    client = ScriptedJevClient()

    # Act
    results = Judge(client, store=JsonlAnswerStore(path), served_model="jev-scripted").check_every(
        [DESCRIBES, CHANGES], pair, SHARED
    )

    # Assert
    assert len(client.requests) == 1
    state, questions = client.requests[0]
    assert sorted(item["code"] for item in state["items"]) == ["x = 1", "y = 2"]
    assert {question_id.split("#")[0] for question_id in questions} == {CHANGES.question_id}
    assert all(result.from_store for result in results["describes"])


def test_a_new_commit_does_not_reorder_the_batches() -> None:
    # Arrange
    def units(commit: str) -> list[dict]:
        return [
            {
                "file": f"f{index % 3}.py",
                "lines": [index * 10, index * 10 + 5],
                "commit": commit,
                "code": f"v = {index}",
            }
            for index in range(20)
        ]

    before, after = ScriptedJevClient(), ScriptedJevClient()

    # Act
    Judge(before, items_per_request=4).check_each(DESCRIBES, units("aaaa"), SHARED)
    Judge(after, items_per_request=4).check_each(DESCRIBES, units("bbbb"), SHARED)

    # Assert
    def members(client: ScriptedJevClient) -> list[list[tuple]]:
        return sorted(
            [[(item["file"], item["lines"][0]) for item in state["items"]] for state, _ in client.requests]
        )

    assert members(before) == members(after)
