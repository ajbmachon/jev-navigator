"""``jvn cache`` shows and prunes JVN's folders, and every search or stats run tidies them as it ends,
without a housekeeping failure ever failing the run."""

from __future__ import annotations

import os
import signal
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator import housekeeping
from jev_navigator.cli import main
from jev_navigator.data_root import default_run_folder, runs_root
from jev_navigator.index.fact_cache import user_fact_cache


def days_ago(path: Path, days: float) -> None:
    stamp = time.time() - days * 86_400
    os.utime(path, (stamp, stamp))


def expired_run() -> Path:
    folder = default_run_folder(Path("shop"), datetime.now(UTC) - timedelta(days=20))
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text("{}")
    return folder


def unused_identity() -> Path:
    folder = user_fact_cache() / "python" / ("a" * 64)
    folder.mkdir(parents=True)
    (folder / f"{'0' * 64}.json").write_text("{}")
    days_ago(folder, 4)
    return folder


def test_cache_status_reports_each_store_and_removes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    run, identity = expired_run(), unused_identity()

    # Act
    exit_code = main(["cache", "status"])

    # Assert
    output = capsys.readouterr().out
    assert exit_code == 0
    [facts, answers] = (line for line in output.splitlines() if line.startswith(("facts:", "answers:")))
    assert facts.startswith("facts: 0 entries") and "1 from other JVN versions (1 unused for 3 days)" in facts
    assert "0 from other JVN versions (0 unused for 30 days)" in answers
    assert "runs: 1 run" in output and "1 past retention" in output
    assert "budget 5.0 GB" in output
    assert run.exists() and identity.exists()


def test_cache_prune_removes_what_the_rules_name_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    run, identity = expired_run(), unused_identity()

    # Act
    exit_code = main(["cache", "prune"])

    # Assert
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "removed 2 files" in output
    assert not run.exists() and not identity.exists()


def test_a_stats_run_tidies_jvns_folders_as_it_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    commit_files(tmp_path / "repo", {"app.py": "def f():\n    return 1\n"})
    monkeypatch.chdir(tmp_path / "repo")
    run = expired_run()

    # Act
    exit_code = main(["stats"])

    # Assert
    assert exit_code == 0
    assert not run.exists()
    assert len(list(runs_root().iterdir())) == 1


def test_a_housekeeping_failure_never_fails_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    private_cache_root: Path,
) -> None:
    # Arrange: the trash is a file, so the unused identity cannot be moved into it
    commit_files(tmp_path / "repo", {"app.py": "def f():\n    return 1\n"})
    monkeypatch.chdir(tmp_path / "repo")
    unused_identity()
    (private_cache_root / ".trash").write_text("not a folder")

    # Act
    exit_code = main(["stats"])

    # Assert
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "statistics pack:" in captured.out
    assert "jvn: housekeeping skipped:" in captured.err and ".trash" in captured.err


@pytest.fixture
def ctrl_c_as_the_sweep_starts(monkeypatch: pytest.MonkeyPatch, python_sigint_handler: None) -> None:
    """The terminal sends a real Ctrl-C as the end-of-run sweep starts."""
    real_sweep = housekeeping._sweep

    def sweep_interrupted(allowance):
        os.kill(os.getpid(), signal.SIGINT)
        return real_sweep(allowance)

    monkeypatch.setattr(housekeeping, "_sweep", sweep_interrupted)


@pytest.mark.usefixtures("ctrl_c_as_the_sweep_starts")
def test_ctrl_c_during_the_cleanup_ends_it_with_one_notice_after_the_result_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    commit_files(tmp_path / "repo", {"app.py": "def f():\n    return 1\n"})
    monkeypatch.chdir(tmp_path / "repo")

    # Act
    exit_code = main(["stats"])

    # Assert
    captured = capsys.readouterr()
    [pack] = runs_root().iterdir()
    assert exit_code == 130
    assert "statistics pack:" in captured.out and (pack / "statistics.json").is_file()
    assert captured.err.count("jvn: housekeeping interrupted") == 1
    assert "Traceback" not in captured.err


@pytest.mark.usefixtures("ctrl_c_as_the_sweep_starts")
def test_ctrl_c_during_the_cleanup_after_a_failed_run_also_ends_it_with_130_and_claims_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange: a size range no symbol fits, so the run fails before it saves anything
    commit_files(tmp_path / "repo", {"app.py": "def f():\n    return 1\n"})
    monkeypatch.chdir(tmp_path / "repo")

    # Act
    exit_code = main(["stats", "--min-lines", "10", "--max-lines", "2"])

    # Assert
    captured = capsys.readouterr()
    assert exit_code == 130
    assert not runs_root().exists()
    notice = "jvn: housekeeping interrupted after the run ended; a later run finishes it"
    assert captured.err.count(notice) == 1
    assert "Traceback" not in captured.err
