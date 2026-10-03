"""One answer store shared across runs: answers replay in a later run and land in that run's pack."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from conftest import BudgetedClient

from jev_navigator.judgments import judge as judge_module
from jev_navigator.judgments.client import ReplayOnlyClient
from jev_navigator.judgments.judge import BATCHING_RULE, Judge
from jev_navigator.judgments.questions import Check, Criterion, Pick
from jev_navigator.judgments.store import (
    SHARED_STORE_VERSION,
    AnswerRecord,
    JsonlAnswerStore,
    LayeredAnswerStore,
    SqliteAnswerStore,
    UnsupportedAnswerStoreError,
)
from jev_navigator.judgments.thresholds import Thresholds
from jev_navigator.testing import ScriptedJevClient

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
SHARED = {"doc": {"sentence": "s"}}
ITEMS = [{"code": "x = 1"}, {"code": "y = 2"}, {"code": "z = 3"}]


def _run_store(run: Path, shared: Path) -> LayeredAnswerStore:
    return LayeredAnswerStore(JsonlAnswerStore(run / "answers.jsonl"), SqliteAnswerStore(shared))


def test_a_later_run_replays_every_answer_from_the_shared_store(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    Judge(ScriptedJevClient(default_noul=0.9), store=_run_store(tmp_path / "run1", shared)).check_each(
        DESCRIBES, ITEMS, SHARED
    )
    client = ScriptedJevClient(default_noul=0.1)

    # Act
    results = Judge(
        client, store=_run_store(tmp_path / "run2", shared), served_model="jev-scripted"
    ).check_each(DESCRIBES, ITEMS, SHARED)

    # Assert
    assert client.requests == []
    assert [(result.probability, result.from_store) for result in results] == [(0.9, True)] * 3


def test_answers_replayed_from_the_shared_store_land_in_the_runs_own_pack(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    Judge(ScriptedJevClient(default_noul=0.9), store=_run_store(tmp_path / "run1", shared)).check_each(
        DESCRIBES, ITEMS, SHARED
    )
    Judge(
        ScriptedJevClient(), store=_run_store(tmp_path / "run2", shared), served_model="jev-scripted"
    ).check_each(DESCRIBES, ITEMS, SHARED)

    # Act: the second run's pack alone, without the shared store, answers offline
    pack_only = Judge(ReplayOnlyClient(), store=JsonlAnswerStore(tmp_path / "run2" / "answers.jsonl"))
    results = pack_only.check_each(DESCRIBES, ITEMS, SHARED)

    # Assert
    assert [result.probability for result in results] == [0.9] * 3


def test_answers_from_another_served_model_are_kept_and_not_reused(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    old = ScriptedJevClient(default_noul=0.9, model="jev-1.12.0")
    Judge(old, store=SqliteAnswerStore(shared)).check_each(DESCRIBES, ITEMS, SHARED)
    upgraded = ScriptedJevClient(default_noul=0.2, model="jev-1.13.0")

    # Act
    Judge(upgraded, store=SqliteAnswerStore(shared), served_model="jev-1.13.0").check_each(
        DESCRIBES, ITEMS, SHARED
    )
    replay_old = Judge(
        ScriptedJevClient(), store=SqliteAnswerStore(shared), served_model="jev-1.12.0"
    ).check_each(DESCRIBES, ITEMS, SHARED)

    # Assert
    assert len(upgraded.requests) == 1
    assert [result.probability for result in replay_old] == [0.9] * 3


def _every_stored_text(path: Path) -> str:
    database = sqlite3.connect(path)
    tables = [name for (name,) in database.execute("select name from sqlite_master where type = 'table'")]
    return "\n".join(str(row) for table in tables for row in database.execute(f"select * from {table}"))


def test_the_shared_store_never_holds_code_state_or_question_text(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    items = [
        {
            "file": "hub.py",
            "lines": [3, 9],
            "signature": "def code_marker_signature(order):",
            "links": ["call send: hub.py -> mail.py | at hub.py:5 code_marker_link_line(order)"],
            "code": "def code_marker_body(): ...",
        }
    ]
    shared_state = {"doc": {"sentence": "state_marker_sentence"}}

    # Act
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared)).check_each(DESCRIBES, items, shared_state)

    # Assert
    stored = _every_stored_text(shared)
    assert "hub.py" in stored
    for marker in ("code_marker", "state_marker", "doc.sentence"):
        assert marker not in stored


def test_every_shared_row_records_its_batch_members_and_composition(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"

    # Act
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared), items_per_request=2).check_each(
        DESCRIBES, ITEMS, SHARED
    )

    # Assert
    batches = [record.batch for record in SqliteAnswerStore(shared).records()]
    assert sorted(len(batch["members"]) for batch in batches) == [1, 2]
    assert {batch["items_per_request"] for batch in batches} == {2}
    assert {batch["batching_rule"] for batch in batches} == {BATCHING_RULE}


def test_replayed_answers_are_counted_apart_from_live_requests(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared)).check_each(DESCRIBES, ITEMS, SHARED)
    judge = Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared), served_model="jev-scripted")
    scoped = judge.scope()

    # Act
    scoped.check_each(DESCRIBES, ITEMS, SHARED)
    scoped.check_each(DESCRIBES, [{"code": "w = 0"}], SHARED)

    # Assert
    assert (scoped.calls, scoped.replayed_answers) == (1, 3)
    assert (judge.calls, judge.replayed_answers) == (1, 3)


def test_two_writers_on_one_shared_file_lose_no_record(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    writers = [SqliteAnswerStore(shared), SqliteAnswerStore(shared)]

    def write(store: SqliteAnswerStore, prefix: str) -> None:
        for index in range(50):
            store.put(AnswerRecord(f"{prefix}{index}", ("q",), {"q": {"type": "noul", "p": 0.5}}, "m", 1, {}))

    threads = [
        threading.Thread(target=write, args=(store, prefix))
        for store, prefix in zip(writers, "ab", strict=True)
    ]

    # Act
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    rows = sqlite3.connect(shared).execute("select count(*) from answers").fetchone()[0]
    assert rows == 100


def test_a_size_refusal_is_remembered_across_runs(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    items = [{"file": f"p{index}.py", "code": "y" * 12_000} for index in range(4)]
    Judge(BudgetedClient(34_000), store=SqliteAnswerStore(shared), served_model="jev-scripted").check_each(
        DESCRIBES, items, SHARED
    )
    replay = BudgetedClient(34_000)

    # Act
    Judge(replay, store=SqliteAnswerStore(shared), served_model="jev-scripted").check_each(
        DESCRIBES, items, SHARED
    )

    # Assert
    assert replay.requests == [] and replay.refusals == 0


def test_a_whole_request_replayed_from_the_store_counts_its_answers(tmp_path: Path) -> None:
    # Arrange
    judge = Judge(ScriptedJevClient(), store=SqliteAnswerStore(tmp_path / "answers.sqlite"))
    pick = Pick("first", "Which entry of `options` comes first?")
    judge.pick(pick, {"0": "a", "1": "b"}, SHARED)

    # Act
    judge.pick(pick, {"0": "a", "1": "b"}, SHARED)

    # Assert
    assert (judge.calls, judge.replayed_answers) == (1, 1)


@pytest.mark.parametrize("store_kind", [JsonlAnswerStore, SqliteAnswerStore])
def test_an_answer_replayed_from_a_store_reports_no_token_count(tmp_path: Path, store_kind) -> None:
    # Arrange: the first ask is sent and its provider reports 100 input tokens
    judge = Judge(ScriptedJevClient(), store=store_kind(tmp_path / "answers"))
    state = {"slice": {"code": "x = 1"}}
    questions = {"adds_one": {"type": "noul", "instructions": "Does `slice.code` add one?"}}
    sent = judge.ask(state, questions, thresholds=Thresholds())

    # Act
    replayed = judge.ask(state, questions, thresholds=Thresholds())

    # Assert: nothing was sent for the replay, so no count was reported, which is not a measured 0
    assert (sent.input_tokens, sent.from_store) == (100, False)
    assert (replayed.input_tokens, replayed.from_store) == (None, True)


def test_a_size_refusal_under_another_input_box_is_not_honoured(tmp_path: Path, monkeypatch) -> None:
    # Arrange: a refusal recorded while the route's box was different
    shared = tmp_path / "answers.sqlite"
    items = [{"file": f"p{index}.py", "code": "y" * 12_000} for index in range(4)]
    Judge(BudgetedClient(34_000), store=SqliteAnswerStore(shared), served_model="jev-scripted").check_each(
        DESCRIBES, items, SHARED
    )
    monkeypatch.setattr(judge_module, "JEV_INPUT_BOX_CHARS", judge_module.JEV_INPUT_BOX_CHARS + 1)
    larger = BudgetedClient(200_000)

    # Act
    Judge(larger, store=SqliteAnswerStore(shared), served_model="jev-scripted").check_each(
        DESCRIBES, items, SHARED
    )

    # Assert: the whole batch is tried again under the new box, and the larger route accepts it
    assert [len(state["items"]) for state, _ in larger.requests] == [4]


def test_a_store_from_an_older_layout_is_refused_with_what_to_delete(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    old = sqlite3.connect(shared)
    old.execute("create table answers (request_sha256 text, model text, record text)")
    old.commit()
    old.close()

    # Act / Assert
    with pytest.raises(UnsupportedAnswerStoreError, match=f"delete {shared}"):
        SqliteAnswerStore(shared)


def test_a_store_with_an_unknown_version_is_refused(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    SqliteAnswerStore(shared)
    newer = sqlite3.connect(shared)
    newer.execute("pragma user_version = 999")
    newer.commit()
    newer.close()

    # Act / Assert
    with pytest.raises(UnsupportedAnswerStoreError, match="version 999"):
        SqliteAnswerStore(shared)


def test_a_new_store_is_created_with_the_current_version_and_reopens(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"

    # Act
    SqliteAnswerStore(shared)
    reopened = SqliteAnswerStore(shared)

    # Assert
    assert reopened.records() == ()
    assert sqlite3.connect(shared).execute("pragma user_version").fetchone()[0] == SHARED_STORE_VERSION
