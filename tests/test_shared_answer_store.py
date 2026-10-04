"""One answer store shared across runs: answers replay in a later run and land in that run's pack."""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
from pathlib import Path

import pytest
from conftest import BudgetedClient

from jev_navigator.confirmation import today
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
    default_shared_store,
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


def test_a_store_from_an_older_layout_is_refused_pointing_at_a_new_file_never_at_deleting_it(
    tmp_path: Path,
) -> None:
    # Arrange: other runs may be writing this file, so the refusal must not advise deleting it
    shared = tmp_path / "answers.sqlite"
    old = sqlite3.connect(shared)
    old.execute("create table answers (request_sha256 text, model text, record text)")
    old.commit()
    old.close()

    # Act
    with pytest.raises(UnsupportedAnswerStoreError) as refused:
        SqliteAnswerStore(shared)

    # Assert
    assert "point --answer-store or JEV_NAVIGATOR_ANSWER_STORE at a new file" in str(refused.value)
    assert "delete" not in str(refused.value)


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


def test_a_whole_request_replays_the_served_models_answer_when_another_model_answered_later(
    tmp_path: Path,
) -> None:
    # Arrange: model A answers a pick, then model B answers the same pick
    shared = tmp_path / "answers.sqlite"
    pick = Pick("first", "Which entry of `options` comes first?")
    for model in ("jev-a", "jev-b"):
        Judge(ScriptedJevClient(model=model), store=_run_store(tmp_path / model, shared)).pick(
            pick, {"0": "a", "1": "b"}, SHARED
        )
    client = ScriptedJevClient(model="jev-a")
    pack = tmp_path / "third" / "answers.jsonl"

    # Act
    Judge(client, store=_run_store(tmp_path / "third", shared), served_model="jev-a").pick(
        pick, {"0": "a", "1": "b"}, SHARED
    )

    # Assert: model A's answer replays, and model B's never reaches this run's pack
    assert client.requests == []
    assert {record.model for record in JsonlAnswerStore(pack).records()} == {"jev-a"}


def test_a_whole_request_replayed_from_the_shared_store_lands_in_the_runs_own_pack(tmp_path: Path) -> None:
    # Arrange: run 1 asks a whole request; run 2 replays it from the shared store
    shared = tmp_path / "answers.sqlite"
    pick = Pick("first", "Which entry of `options` comes first?")
    options = {"0": "a", "1": "b"}
    first = Judge(
        ScriptedJevClient(choices={"first": {"0": 0.1, "1": 0.9}}),
        store=_run_store(tmp_path / "run1", shared),
    )
    first.pick(pick, options, SHARED)
    second = Judge(
        ScriptedJevClient(), store=_run_store(tmp_path / "run2", shared), served_model="jev-scripted"
    )
    second.pick(pick, options, SHARED)

    # Act: the second run's pack alone, without the shared store, answers offline
    pack_only = Judge(ReplayOnlyClient(), store=JsonlAnswerStore(tmp_path / "run2" / "answers.jsonl"))
    replayed = pack_only.pick(pick, options, SHARED)

    # Assert
    assert replayed is not None and replayed.choice == "1"


def _open_and_write_after(barrier, path: str, request_sha256: str) -> None:
    barrier.wait()
    store = SqliteAnswerStore(Path(path))
    store.put(AnswerRecord(request_sha256, ("q",), {"q": {"type": "noul", "p": 0.5}}, "m", 1, {}))


def test_processes_creating_a_new_store_at_once_all_open_it(tmp_path: Path) -> None:
    # Arrange: five rounds of four processes, each round on a store file that does not exist yet
    context = multiprocessing.get_context("spawn")
    rounds = []

    # Act
    for round_number in range(5):
        shared, barrier = tmp_path / f"round-{round_number}" / "answers.sqlite", context.Barrier(4)
        openers = [
            context.Process(target=_open_and_write_after, args=(barrier, str(shared), f"r{n}"))
            for n in range(4)
        ]
        for opener in openers:
            opener.start()
        for opener in openers:
            opener.join(timeout=60)
        rounds.append((shared, [opener.exitcode for opener in openers]))

    # Assert
    for shared, exit_codes in rounds:
        assert exit_codes == [0, 0, 0, 0]
        assert len(SqliteAnswerStore(shared).records()) == 4


def _confirmed(shared: Path) -> dict[str, int]:
    with sqlite3.connect(shared) as database:
        return dict(database.execute("select request_sha256, confirmed from confirmations"))


def _age_every_confirmation(shared: Path, days: int) -> None:
    with sqlite3.connect(shared) as database:
        database.execute("update confirmations set confirmed = ?", (today() - days,))


def _rows_for(shared: Path, request_sha256: str) -> dict[str, int]:
    with sqlite3.connect(shared) as database:
        return {
            table: database.execute(
                f"select count(*) from {table} where request_sha256 = ?", (request_sha256,)
            ).fetchone()[0]
            for table in ("answers", "item_answers", "refusals", "confirmations")
        }


def test_a_reused_answer_stays_and_an_unused_one_goes_with_its_item_answers_and_refusals(
    tmp_path: Path,
) -> None:
    # Arrange: two requests stored 40 days ago, each with a refusal; only the first is reused today
    shared = tmp_path / "answers.sqlite"
    reused_items, unused_items = ITEMS[:2], [{"code": "w = 0"}]
    Judge(ScriptedJevClient(default_noul=0.9), store=SqliteAnswerStore(shared)).check_each(
        DESCRIBES, reused_items, SHARED
    )
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared)).check_each(DESCRIBES, unused_items, SHARED)
    reused, unused = (record.request_sha256 for record in SqliteAnswerStore(shared).records())
    for request in (reused, unused):
        SqliteAnswerStore(shared).put_refusal(request, "route", 1)
    _age_every_confirmation(shared, 40)
    replay = ScriptedJevClient(default_noul=0.1)
    Judge(replay, store=SqliteAnswerStore(shared), served_model="jev-scripted").check_each(
        DESCRIBES, reused_items, SHARED
    )

    # Act
    forgotten = SqliteAnswerStore(shared).forget_unconfirmed(before=today() - 30, limit=2_000)

    # Assert
    assert replay.requests == []
    assert forgotten == 1
    assert _rows_for(shared, unused) == {"answers": 0, "item_answers": 0, "refusals": 0, "confirmations": 0}
    assert _rows_for(shared, reused) == {"answers": 1, "item_answers": 2, "refusals": 1, "confirmations": 1}
    again = Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared), served_model="jev-scripted")
    assert [result.probability for result in again.check_each(DESCRIBES, reused_items, SHARED)] == [0.9] * 2


@pytest.mark.parametrize("hit", ["by_request", "by_request_for_a_model", "by_item", "refused"])
def test_every_kind_of_reuse_confirms_its_request(tmp_path: Path, hit: str) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    store = SqliteAnswerStore(shared)
    store.put(
        AnswerRecord("r1", ("q",), {"q": {"type": "noul", "noul": 0.5}}, "m", 1, {}, item_keys={"k1": "q"})
    )
    store.put_refusal("r1", "route", 1)
    _age_every_confirmation(shared, 40)
    lookups = {
        "by_request": lambda: store.by_request("r1", None),
        "by_request_for_a_model": lambda: store.by_request("r1", "m"),
        "by_item": lambda: store.by_item("k1", "m"),
        "refused": lambda: store.refused("r1", "route", 1),
    }

    # Act
    found = lookups[hit]()

    # Assert
    assert found
    assert _confirmed(shared) == {"r1": today()}


def test_a_same_day_replay_takes_no_write_lock(tmp_path: Path) -> None:
    # Arrange: another process holds the store's write lock for the whole replay
    shared = tmp_path / "answers.sqlite"
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared)).check_each(DESCRIBES, ITEMS, SHARED)
    writer = sqlite3.connect(shared, isolation_level=None)
    writer.execute("begin immediate")
    outcome: list[object] = []

    def replay() -> None:
        store = SqliteAnswerStore(shared)
        outcome.append(
            Judge(ScriptedJevClient(), store=store, served_model="jev-scripted").check_each(
                DESCRIBES, ITEMS, SHARED
            )
        )

    # Act
    run = threading.Thread(target=replay, daemon=True)
    run.start()
    run.join(timeout=10)
    writer.execute("rollback")

    # Assert
    assert not run.is_alive(), "the same-day replay waited for the store's write lock"
    assert len(outcome) == 1


def test_forgetting_takes_the_least_recently_used_requests_first_and_stops_at_the_limit(
    tmp_path: Path,
) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    store = SqliteAnswerStore(shared)
    for request in ("r1", "r2", "r3"):
        store.put(AnswerRecord(request, ("q",), {"q": {"type": "noul", "p": 0.5}}, "m", 1, {}))
    with sqlite3.connect(shared) as database:
        for age, request in enumerate(("r3", "r1", "r2")):
            database.execute(
                "update confirmations set confirmed = ? where request_sha256 = ?", (today() - age, request)
            )

    # Act
    forgotten = store.forget_unconfirmed(before=today() + 1, limit=2)

    # Assert
    assert forgotten == 2
    assert set(_confirmed(shared)) == {"r3"}


def test_the_default_store_file_names_its_layout(private_cache_root: Path) -> None:
    # Act
    path = default_shared_store()

    # Assert
    assert path == private_cache_root / f"answers-v{SHARED_STORE_VERSION}.sqlite"
    assert SHARED_STORE_VERSION == 2


def test_a_reuse_whose_stamp_cannot_be_written_still_answers_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange: another process holds the write lock past the store's (shortened) busy wait
    shared = tmp_path / "answers.sqlite"
    store = SqliteAnswerStore(shared)
    store.put(AnswerRecord("r1", ("q",), {"q": {"type": "noul", "noul": 0.5}}, "m", 1, {}))
    _age_every_confirmation(shared, 40)
    store._db.execute("pragma busy_timeout = 50")
    writer = sqlite3.connect(shared, isolation_level=None)
    writer.execute("begin immediate")

    # Act
    try:
        found = store.by_request("r1", None)
    finally:
        writer.execute("rollback")

    # Assert
    assert found is not None and found.request_sha256 == "r1"
    assert "not stamped" in caplog.text
    assert _confirmed(shared) == {"r1": today() - 40}
