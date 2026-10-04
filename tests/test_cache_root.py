"""The answer store and the fact cache live in one cache folder that honours ``$XDG_CACHE_HOME``."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from jev_navigator.cache_root import cache_root
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.fact_cache import FactCache
from jev_navigator.index.scope_scan import Unparsed, scan_facts
from jev_navigator.judgments.store import (
    SHARED_STORE_VARIABLE,
    SHARED_STORE_VERSION,
    StoreInCacheFolderError,
    shared_store_path,
)


def test_the_answer_store_and_the_fact_cache_share_the_xdg_cache_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv(SHARED_STORE_VARIABLE, raising=False)

    # Act
    answers, facts = shared_store_path(), FactCache().root

    # Assert
    assert answers == tmp_path / "jev-navigator" / f"answers-v{SHARED_STORE_VERSION}.sqlite"
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
    assert answers == Path.home() / ".cache" / "jev-navigator" / f"answers-v{SHARED_STORE_VERSION}.sqlite"
    assert facts == Path.home() / ".cache" / "jev-navigator" / "facts"


@pytest.mark.parametrize("relative", ["cache", "~/cache", "./cache"])
def test_a_relative_xdg_cache_home_is_ignored_so_no_cache_lands_in_the_analysed_repository(
    relative: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("XDG_CACHE_HOME", relative)

    # Act
    root = cache_root()

    # Assert
    assert root == Path.home() / ".cache" / "jev-navigator"


def test_a_store_named_in_a_folder_beside_the_cache_folder_is_accepted(private_cache_root: Path) -> None:
    # Arrange: a folder whose name only starts with the cache folder's name
    store = Path(f"{private_cache_root}-evals") / "arm.sqlite"

    # Act
    resolved = shared_store_path(str(store))

    # Assert
    assert resolved == store


@pytest.mark.parametrize("through_link", ["cache folder", "store"])
def test_a_store_inside_the_cache_folder_is_refused_when_either_path_runs_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, through_link: str
) -> None:
    # Arrange: one of the two paths reaches the same folder through a link, as macOS's /tmp and /var do
    real, link = tmp_path / "real", tmp_path / "link"
    real.mkdir()
    link.symlink_to(real)
    cache_home, store_home = (link, real) if through_link == "cache folder" else (real, link)
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    # Act / Assert
    with pytest.raises(StoreInCacheFolderError, match="JVN prunes that folder"):
        shared_store_path(str(store_home / "jev-navigator" / "answers.sqlite"))


def test_facts_planted_in_the_suites_own_cache_never_reach_a_tests_index(
    tmp_path: Path, outer_cache_root: Path
) -> None:
    # Arrange: wrong facts for this exact content, written where an unisolated index would look
    content = f"def real():\n    return {uuid.uuid4().int}\n"
    (tmp_path / "module.py").write_text(content)
    (tmp_path / "planted.py").write_text("def planted():\n    return 1\n")
    wrong = scan_facts(["planted.py"], tmp_path, Unparsed())["planted.py"]
    planted = FactCache(outer_cache_root / "facts")
    before = set(planted.root.rglob("*.json"))
    planted.save("module.py", content.encode(), wrong)
    try:
        # Act
        names = [span.name for span in CodeIndex(tmp_path, ["module.py"]).functions_in("module.py")]
    finally:
        for entry in set(planted.root.rglob("*.json")) - before:
            entry.unlink()

    # Assert
    assert names == ["real"]
