"""Every stored item answer is also found by its relaxed key: the unmasked item, the shared state its
question names, and the question with its wording, in any batch company. Production lookups stay on
the strict key, which includes the batch mates."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from jev_navigator.directives.find_all import TARGETS, match_check
from jev_navigator.judgments.item_keys import relaxed_item_key
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import AnswerStore, JsonlAnswerStore, SqliteAnswerStore
from jev_navigator.testing import ScriptedJevClient

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
SHARED = {"doc": {"sentence": "s"}}
PAIR = [{"code": "x = 1"}, {"code": "y = 2"}]
MODEL = "jev-scripted"

STORES: dict[str, Callable[[Path], AnswerStore]] = {
    "jsonl": lambda folder: JsonlAnswerStore(folder / "answers.jsonl"),
    "sqlite": lambda folder: SqliteAnswerStore(folder / "answers.sqlite"),
}


@pytest.fixture(params=sorted(STORES))
def judged_pair(request: pytest.FixtureRequest, tmp_path: Path) -> AnswerStore:
    """A store holding the answers for ``PAIR`` judged together in one batch."""
    store = STORES[request.param](tmp_path)
    Judge(ScriptedJevClient(), store=store).check_each(DESCRIBES, PAIR, SHARED)
    return store


def test_an_item_judged_in_a_pair_is_found_by_its_relaxed_key_alone(judged_pair: AnswerStore) -> None:
    # Act
    found = judged_pair.by_item(relaxed_item_key(DESCRIBES, PAIR[0], SHARED), MODEL)

    # Assert
    assert found is not None


@pytest.mark.parametrize(
    ("check", "item", "shared"),
    [
        pytest.param(DESCRIBES, {"code": "x = 3"}, SHARED, id="changed unit"),
        pytest.param(
            Check(DESCRIBES.name, DESCRIBES.instructions + " Read it closely.", DESCRIBES.yes, DESCRIBES.no),
            PAIR[0],
            SHARED,
            id="changed wording",
        ),
        pytest.param(DESCRIBES, PAIR[0], {"doc": {"sentence": "t"}}, id="changed point"),
    ],
)
def test_a_changed_unit_wording_or_point_misses(
    judged_pair: AnswerStore, check: Check, item: dict, shared: dict
) -> None:
    # Act
    found = judged_pair.by_item(relaxed_item_key(check, item, shared), MODEL)

    # Assert
    assert found is None


def test_a_shared_field_the_question_does_not_name_does_not_change_the_key(judged_pair: AnswerStore) -> None:
    # Act
    found = judged_pair.by_item(relaxed_item_key(DESCRIBES, PAIR[0], {**SHARED, "other": {"x": 1}}), MODEL)

    # Assert
    assert found is not None


def test_another_model_misses(judged_pair: AnswerStore) -> None:
    # Act
    found = judged_pair.by_item(relaxed_item_key(DESCRIBES, PAIR[0], SHARED), "jev-other")

    # Assert
    assert found is None


def test_the_relaxed_key_hashes_the_item_before_masking(tmp_path: Path) -> None:
    # Arrange: the masker hides the value in the item as it is sent
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    secret_item = {"code": 'password = "hunter2go"'}
    client = ScriptedJevClient()

    # Act
    Judge(client, store=store).check_each(DESCRIBES, [secret_item], SHARED)

    # Assert
    assert "hunter2go" not in str(client.requests)
    assert store.by_item(relaxed_item_key(DESCRIBES, secret_item, SHARED), MODEL) is not None


def test_the_relaxed_key_hashes_the_shared_state_before_masking(tmp_path: Path) -> None:
    # Arrange: the masker hides the value in the shared state as it is sent
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    secret_shared = {"doc": {"sentence": 'the check that rejects password = "hunter2go"'}}
    client = ScriptedJevClient()

    # Act
    Judge(client, store=store).check_each(DESCRIBES, PAIR[:1], secret_shared)

    # Assert
    assert "hunter2go" not in str(client.requests)
    assert store.by_item(relaxed_item_key(DESCRIBES, PAIR[0], secret_shared), MODEL) is not None


def test_production_lookups_stay_on_the_strict_key(judged_pair: AnswerStore) -> None:
    # Arrange
    alone = ScriptedJevClient()

    # Act
    results = Judge(alone, store=judged_pair, served_model=MODEL).check_each(DESCRIBES, PAIR[:1], SHARED)

    # Assert
    assert results[0].from_store is False and len(alone.requests) == 1


def test_a_find_all_answer_keeps_its_key_when_another_point_changes() -> None:
    # Arrange
    check = match_check("p0")
    before = {TARGETS: {"p0": "the admin check", "p1": "the retry loop"}}
    after = {TARGETS: {"p0": "the admin check", "p1": "the cache"}}

    # Act
    keys = {relaxed_item_key(check, PAIR[0], shared) for shared in (before, after)}

    # Assert
    assert len(keys) == 1
    assert relaxed_item_key(check, PAIR[0], {TARGETS: {"p0": "the login check"}}) not in keys
