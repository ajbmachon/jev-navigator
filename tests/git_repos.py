"""Small git repositories for tests: write files, run git with a fixed author, commit."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest


def git(root: Path, *arguments: str, stdin: str | None = None) -> str:
    """Runs git in ``root`` with a fixed author, so commits need no user configuration, and returns
    its output."""
    completed = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *arguments],
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def write_files(root: Path, files: Mapping[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)


def commit_all(root: Path) -> None:
    """Commits everything under ``root`` as the first commit of a new repository."""
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "c")


def commit_files(root: Path, files: Mapping[str, str]) -> None:
    write_files(root, files)
    commit_all(root)


def read_files(root: Path, files: Iterable[str]) -> dict[str, bytes]:
    """Each file's bytes as an index first reads them, for calling the fact scan directly."""
    return {file: (root / file).read_bytes() for file in files}


def assume_another_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes git believe another user owns every repository and read no system or user configuration,
    so no ``safe.directory`` trusts it and git refuses it with "dubious ownership" on every machine.
    CI runners trust every directory in their git configuration."""
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
