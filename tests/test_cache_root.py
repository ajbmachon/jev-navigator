"""The answer store and the fact cache live in one cache folder that honours ``$XDG_CACHE_HOME``."""

from __future__ import annotations

from pathlib import Path

import pytest

from jev_navigator.index.fact_cache import FactCache
from jev_navigator.judgments.store import SHARED_STORE_VARIABLE, shared_store_path


def test_the_answer_store_and_the_fact_cache_share_the_xdg_cache_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv(SHARED_STORE_VARIABLE, raising=False)

    # Act
    answers, facts = shared_store_path(), FactCache().root

    # Assert
    assert answers == tmp_path / "jev-navigator" / "answers.sqlite"
    assert facts == tmp_path / "jev-navigator" / "facts"


def test_without_xdg_cache_home_both_live_under_the_home_cache_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv(SHARED_STORE_VARIABLE, raising=False)

    # Act
    answers, facts = shared_store_path(), FactCache().root

    # Assert
    assert answers == Path.home() / ".cache" / "jev-navigator" / "answers.sqlite"
    assert facts == Path.home() / ".cache" / "jev-navigator" / "facts"
