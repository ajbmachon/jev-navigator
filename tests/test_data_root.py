"""Run folders live in JVN's own data folder, which honours ``$XDG_DATA_HOME``, never in the user's
directory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_navigator.data_root import default_run_folder, run_folder_started, runs_root


def test_runs_live_in_the_xdg_data_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    # Act
    root = runs_root()

    # Assert
    assert root == tmp_path / "jev-navigator" / "runs"


def test_without_xdg_data_home_runs_live_under_the_home_data_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    # Act
    root = runs_root()

    # Assert
    assert root == Path.home() / ".local" / "share" / "jev-navigator" / "runs"


def test_a_relative_xdg_data_home_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: a relative path would put run folders inside whatever directory JVN runs in
    monkeypatch.setenv("XDG_DATA_HOME", "relative/data")

    # Act
    root = runs_root()

    # Assert
    assert root == Path.home() / ".local" / "share" / "jev-navigator" / "runs"


def test_a_default_run_folder_names_the_repository_and_when_it_started(
    tmp_path: Path, private_data_root: Path
) -> None:
    # Arrange
    started = datetime(2026, 10, 4, 12, 30, 5, 123456, tzinfo=UTC)

    # Act
    folder = default_run_folder(tmp_path / "shop", started)

    # Assert
    assert folder == runs_root() / "shop-20261004T123005123456Z"
    assert run_folder_started(folder.name) == started


@pytest.mark.parametrize("name", ["order-evidence", "shop-2026", "shop-20261004T123005Z", ".trash"])
def test_a_folder_not_named_by_jvn_has_no_start(name: str) -> None:
    # Act
    started = run_folder_started(name)

    # Assert
    assert started is None
