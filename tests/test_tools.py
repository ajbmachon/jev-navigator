"""Exporting git blobs to disk: each blob at its path, refused escapes and missing objects."""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import git

from jev_navigator.index import tools

MISSING_OBJECT = "0" * 40


def stored_blob(repository: Path, text: str) -> str:
    """Writes ``text`` into the repository's object store only and returns its object id."""
    git(repository, "init", "-q")
    return git(repository, "hash-object", "-w", "--stdin", stdin=text).strip()


def test_export_writes_each_blob_at_its_path(tmp_path: Path) -> None:
    # Arrange
    repository, destination = tmp_path / "repo", tmp_path / "export"
    repository.mkdir()
    object_id = stored_blob(repository, "def kept():\n    return 1\n")

    # Act
    tools.export_blobs(repository, {"app/kept.py": object_id, "copy.py": object_id}, destination)

    # Assert
    assert (destination / "app/kept.py").read_text() == "def kept():\n    return 1\n"
    assert (destination / "copy.py").read_text() == "def kept():\n    return 1\n"


def test_export_refuses_a_path_that_leaves_the_destination(tmp_path: Path) -> None:
    # Arrange
    repository, destination = tmp_path / "repo", tmp_path / "export"
    repository.mkdir()
    object_id = stored_blob(repository, "escaped = True\n")

    # Act
    with pytest.raises(tools.ToolFailedError, match="outside"):
        tools.export_blobs(repository, {"../escaped.py": object_id}, destination)

    # Assert
    assert not (tmp_path / "escaped.py").exists()


def test_export_fails_loudly_when_git_lacks_an_object(tmp_path: Path) -> None:
    # Arrange
    repository = tmp_path / "repo"
    repository.mkdir()
    stored_blob(repository, "present = True\n")

    # Act and assert
    with pytest.raises(tools.ToolFailedError, match=MISSING_OBJECT):
        tools.export_blobs(repository, {"gone.py": MISSING_OBJECT}, tmp_path / "export")
