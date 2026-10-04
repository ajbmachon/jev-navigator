"""Housekeeping keeps JVN's disk use bounded: caches stay while they represent real files, run folders
for a while, and nothing outside JVN's own folders is ever touched."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator import housekeeping
from jev_navigator.confirmation import today
from jev_navigator.data_root import default_run_folder, runs_root
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.fact_cache import FactCache, user_fact_cache
from jev_navigator.index.name_table import NameTable
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import SHARED_STORE_VARIABLE, SqliteAnswerStore, default_shared_store
from jev_navigator.testing import ScriptedJevClient

REPOSITORY = {
    "app/rules.py": "LIMIT = 3\n\n\ndef check(order):\n    return len(order) <= LIMIT\n",
    "app/orders.py": "from app.rules import check\n\n\ndef place(order):\n    return check(order)\n",
}
OTHER_REPOSITORY = {"lib/other.py": "def elsewhere(x):\n    return x\n"}
DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
SHARED = {"doc": {"sentence": "s"}}


def days_ago(path: Path, days: float) -> None:
    stamp = time.time() - days * 86_400
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def retired_folder(name: str, files: int, days: float, size: int = 10) -> Path:
    """Another JVN version's identity folder for Python facts, last used ``days`` ago."""
    folder = user_fact_cache() / "python" / name
    folder.mkdir(parents=True)
    for number in range(files):
        (folder / f"{number:064x}.json").write_text("x" * size)
    days_ago(folder, days)
    return folder


def indexed(root: Path, files: dict[str, str]) -> CodeIndex:
    """An index over a new repository at ``root`` that has covered its scope, filling the fact cache
    and the name table."""
    commit_files(root, files)
    index = CodeIndex.from_git(root)
    index.find_callers("check")
    for file in files:
        index.functions_in(file)
    return index


def run_folder(age_days: float, *, resumable: bool, size: int = 1_000) -> Path:
    folder = default_run_folder(Path("shop"), datetime.now(UTC) - timedelta(days=age_days))
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text("m" * size)
    if resumable:
        (folder / "resume.json").write_text("{}")
    return folder


def answered(store: Path, items: list[dict]) -> str:
    """Stores the answers for ``items`` and returns their request hash."""
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(store)).check_each(DESCRIBES, items, SHARED)
    return SqliteAnswerStore(store).records()[-1].request_sha256


def age_answers(store: Path, days: int) -> None:
    with sqlite3.connect(store) as database:
        database.execute("update confirmations set confirmed = ?", (today() - days,))


def age_name_rows(days: int) -> None:
    with sqlite3.connect(NameTable().path) as database:
        database.execute("update files set confirmed = ?", (today() - days,))


def disk_use(*roots: Path) -> int:
    return sum(path.lstat().st_blocks * 512 for root in roots if root.exists() for path in root.rglob("*"))


def test_another_versions_facts_go_when_no_jvn_used_them_for_three_days(tmp_path: Path) -> None:
    # Arrange
    indexed(tmp_path / "repo", REPOSITORY)
    [current] = FactCache().current_folders()
    unused = retired_folder("a" * 64, files=3, days=4)
    recent = retired_folder("b" * 64, files=3, days=1)

    # Act
    housekeeping.prune()

    # Assert
    assert not unused.exists()
    assert recent.exists() and len(list(recent.iterdir())) == 3
    assert len(list(current.iterdir())) == len(REPOSITORY)


def test_an_entry_from_the_layout_before_identity_folders_goes_after_three_days(tmp_path: Path) -> None:
    # Arrange
    flat = user_fact_cache() / "python"
    flat.mkdir(parents=True)
    old, young = flat / f"{'1' * 64}.json", flat / f"{'2' * 64}.json"
    for entry, age in ((old, 4), (young, 1)):
        entry.write_text("{}")
        days_ago(entry, age)

    # Act
    housekeeping.prune()

    # Assert
    assert not old.exists()
    assert young.exists()


def test_a_fact_entry_no_run_confirmed_for_thirty_days_goes(tmp_path: Path) -> None:
    # Arrange
    indexed(tmp_path / "repo", REPOSITORY)
    [current] = FactCache().current_folders()
    stale, kept = sorted(current.iterdir())
    days_ago(stale, 31)
    days_ago(kept, 29)

    # Act
    housekeeping.prune()

    # Assert
    assert not stale.exists()
    assert kept.exists()


def test_a_run_deletes_at_most_two_thousand_files_and_the_next_run_goes_on(tmp_path: Path) -> None:
    # Arrange
    retired_folder("a" * 64, files=2_500, days=4)

    # Act
    first = housekeeping.tidy()
    left_after_first = disk_use(user_fact_cache().parent / ".trash")
    second = housekeeping.tidy()

    # Assert
    assert (first.deleted_files, first.finished) == (2_000, False)
    assert left_after_first > 0
    assert (second.deleted_files, second.finished) == (500, True)
    assert not (user_fact_cache() / "python" / ("a" * 64)).exists()


def test_a_name_table_no_jvn_used_for_three_days_goes(tmp_path: Path, private_cache_root: Path) -> None:
    # Arrange
    indexed(tmp_path / "repo", REPOSITORY)
    names = private_cache_root / "names"
    unused = [names / f"{'a' * 64}.sqlite", names / f"{'a' * 64}.sqlite-wal"]
    recent = names / f"{'b' * 64}.sqlite"
    for path, age in ((unused[0], 4), (unused[1], 4), (recent, 1)):
        path.write_bytes(b"table")
        days_ago(path, age)

    # Act
    housekeeping.prune()

    # Assert
    assert not any(path.exists() for path in unused)
    assert recent.exists()
    assert NameTable().path.exists()


def test_name_rows_go_once_their_files_went_unconfirmed_for_thirty_days(tmp_path: Path) -> None:
    # Arrange: two repositories share the table; only the first is searched again
    indexed(tmp_path / "kept", REPOSITORY)
    indexed(tmp_path / "gone", OTHER_REPOSITORY)
    age_name_rows(40)
    indexed_again = CodeIndex.from_git(tmp_path / "kept")
    indexed_again.find_callers("check")

    # Act
    sweep = housekeeping.prune()

    # Assert
    assert sweep.forgotten["names"] == len(OTHER_REPOSITORY)
    assert NameTable().confirmations(before=today()).held == len(REPOSITORY)


def test_default_store_answers_no_run_reused_for_thirty_days_go(tmp_path: Path) -> None:
    # Arrange
    store = default_shared_store()
    reused_items, unused_items = [{"code": "x = 1"}], [{"code": "y = 2"}]
    reused, unused = answered(store, reused_items), answered(store, unused_items)
    age_answers(store, 40)
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(store), served_model="jev-scripted").check_each(
        DESCRIBES, reused_items, SHARED
    )

    # Act
    sweep = housekeeping.prune()

    # Assert
    assert sweep.forgotten["answers"] == 1
    assert [record.request_sha256 for record in SqliteAnswerStore(store).records()] == [reused]
    assert unused != reused


def test_a_store_the_user_names_is_never_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    named = tmp_path / "eval-arm.sqlite"
    answered(named, [{"code": "x = 1"}])
    age_answers(named, 40)
    monkeypatch.setenv(SHARED_STORE_VARIABLE, str(named))
    days_ago(named, 40)
    before = (named.read_bytes(), named.stat().st_mtime)

    # Act
    housekeeping.prune()

    # Assert
    assert (named.read_bytes(), named.stat().st_mtime) == before


def test_an_older_default_store_layout_follows_the_thirty_day_answer_rule(private_cache_root: Path) -> None:
    # Arrange: two older layouts' default files, and a file someone put in the cache folder by hand
    unused = [private_cache_root / "answers.sqlite", private_cache_root / "answers.sqlite-shm"]
    recent = private_cache_root / "answers-v1.sqlite"
    put_there_by_hand = private_cache_root / "answers-of-my-eval.sqlite"
    private_cache_root.mkdir(parents=True)
    for path, age in ((unused[0], 31), (unused[1], 31), (recent, 29), (put_there_by_hand, 31)):
        path.write_bytes(b"old layout")
        days_ago(path, age)

    # Act
    housekeeping.prune()

    # Assert
    assert not any(path.exists() for path in unused)
    assert recent.exists()
    assert put_there_by_hand.exists()


def test_finished_runs_go_after_fourteen_days_and_resumable_runs_after_thirty() -> None:
    # Arrange
    finished_old, finished_young = run_folder(15, resumable=False), run_folder(13, resumable=False)
    resumable_young, resumable_old = run_folder(15, resumable=True), run_folder(31, resumable=True)

    # Act
    housekeeping.prune()

    # Assert
    assert [folder.exists() for folder in (finished_old, finished_young, resumable_young, resumable_old)] == [
        False,
        True,
        True,
        False,
    ]


def test_a_folder_jvn_did_not_name_and_a_link_are_never_touched(tmp_path: Path) -> None:
    # Arrange: an old --out folder elsewhere, a folder the user put among the runs, and a link to one
    elsewhere = tmp_path / "shop-20200101T000000000000Z"
    elsewhere.mkdir()
    (elsewhere / "manifest.json").write_text("{}")
    named = runs_root() / "order-evidence"
    named.mkdir(parents=True)
    days_ago(named, 400)
    link = runs_root() / "shop-20200101T000000000000Z"
    link.symlink_to(elsewhere)

    # Act
    housekeeping.prune()

    # Assert
    assert named.exists()
    assert link.is_symlink()
    assert (elsewhere / "manifest.json").exists()


def test_a_link_in_the_cache_is_removed_as_a_link_and_what_it_points_to_stays(tmp_path: Path) -> None:
    # Arrange: an unused "identity folder" that is a link to a folder outside JVN's roots
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.json").write_text("{}")
    link = user_fact_cache() / "python" / ("c" * 64)
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    days_ago(link, 4)

    # Act
    housekeeping.prune()

    # Assert
    assert not link.is_symlink()
    assert (outside / "keep.json").exists()


def test_over_budget_run_folders_go_before_any_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a young run folder pushes JVN over its budget
    indexed(tmp_path / "repo", REPOSITORY)
    run = run_folder(1, resumable=True, size=400_000)
    caches = disk_use(user_fact_cache().parent)
    monkeypatch.setenv(housekeeping.DISK_BUDGET_VARIABLE, str(caches + 100_000))

    # Act
    housekeeping.prune()

    # Assert
    assert not run.exists()
    assert disk_use(user_fact_cache().parent) >= caches - 50_000
    assert len(list(FactCache().current_folders()[0].iterdir())) == len(REPOSITORY)


def test_over_budget_answers_go_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a run, another version's facts, current facts and names, the default store's answers,
    # and an older layout of the default store, last used before the other version's facts
    indexed(tmp_path / "repo", REPOSITORY)
    run_folder(1, resumable=False, size=50_000)
    retired_folder("a" * 64, files=5, days=1, size=20_000)
    store = default_shared_store()
    request = answered(store, [{"code": "x = 1"}])
    with sqlite3.connect(store) as database:
        database.execute("pragma wal_checkpoint(truncate)")
    older_layout = store.parent / "answers.sqlite"
    older_layout.write_bytes(b"o" * 30_000)
    days_ago(older_layout, 2)
    answers = sum(path.stat().st_blocks * 512 for path in store.parent.glob("answers*.sqlite*"))
    monkeypatch.setenv(housekeeping.DISK_BUDGET_VARIABLE, str(answers + 120_000))

    # Act
    housekeeping.prune()

    # Assert
    assert not any(runs_root().iterdir())
    assert FactCache().retired() == []
    assert older_layout.exists()
    assert [record.request_sha256 for record in SqliteAnswerStore(store).records()] == [request]


def test_over_budget_an_older_answer_layout_goes_before_the_current_stores_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: half the older layout's size over the budget, so removing either one would do
    store = default_shared_store()
    request = answered(store, [{"code": "x = 1"}])
    older_layout = store.parent / "answers.sqlite"
    older_layout.write_bytes(b"o" * 400_000)
    over_by = older_layout.stat().st_blocks * 512 // 2
    monkeypatch.setenv(housekeeping.DISK_BUDGET_VARIABLE, str(housekeeping.status().used - over_by))

    # Act
    housekeeping.prune()

    # Assert
    assert not older_layout.exists()
    assert [record.request_sha256 for record in SqliteAnswerStore(store).records()] == [request]


@pytest.mark.parametrize(
    ("text", "expected"), [("5GB", 5_000_000_000), ("750 MB", 750_000_000), ("1024", 1024)]
)
def test_the_budget_reads_bytes_or_decimal_units(
    monkeypatch: pytest.MonkeyPatch, text: str, expected: int
) -> None:
    # Arrange
    monkeypatch.setenv(housekeeping.DISK_BUDGET_VARIABLE, text)

    # Act
    budget = housekeeping.disk_budget()

    # Assert
    assert budget == expected


def test_without_a_setting_the_budget_is_five_gigabytes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv(housekeeping.DISK_BUDGET_VARIABLE, raising=False)

    # Act
    budget = housekeeping.disk_budget()

    # Assert
    assert budget == 5_000_000_000


def test_a_budget_jvn_cannot_read_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv(housekeeping.DISK_BUDGET_VARIABLE, "lots")

    # Act / Assert
    with pytest.raises(ValueError, match=housekeeping.DISK_BUDGET_VARIABLE):
        housekeeping.disk_budget()


def test_a_second_run_the_same_day_does_not_sweep_again(private_cache_root: Path) -> None:
    # Arrange
    housekeeping.tidy()
    unused = retired_folder("a" * 64, files=3, days=4)

    # Act
    same_day = housekeeping.tidy()
    kept_on_the_same_day = unused.exists()
    days_ago(private_cache_root / ".swept", 2)
    next_day = housekeeping.tidy()

    # Assert
    assert same_day is None and kept_on_the_same_day
    assert next_day is not None and not unused.exists()


def test_status_counts_what_each_rule_would_remove(tmp_path: Path) -> None:
    # Arrange
    indexed(tmp_path / "repo", REPOSITORY)
    retired_folder("a" * 64, files=4, days=4)
    retired_folder("b" * 64, files=2, days=1)
    [current] = FactCache().current_folders()
    days_ago(next(current.iterdir()), 31)
    run_folder(15, resumable=False)
    run_folder(1, resumable=False)
    for layout, age in (("answers.sqlite", 4), ("answers-v1.sqlite", 31)):
        (default_shared_store().parent / layout).write_bytes(b"old layout")
        days_ago(default_shared_store().parent / layout, age)

    # Act
    status = housekeeping.status()

    # Assert
    assert (status.facts.held, status.facts.unconfirmed) == (len(REPOSITORY), 1)
    assert (status.facts.retired, status.facts.retired_unused) == (2, 1)
    assert (status.names.held, status.names.unconfirmed) == (len(REPOSITORY), 0)
    assert status.answers.held == 0
    assert (status.answers.retired, status.answers.retired_unused) == (2, 1)
    assert (status.runs.count, status.runs.expired) == (2, 1)
    assert status.budget == 5_000_000_000
    assert status.used > 0
    assert any(runs_root().iterdir()), "status removes nothing"
