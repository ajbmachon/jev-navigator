"""What the programs JVN starts inherit from their host, proved with real processes.

A host may hold model keys, bearer tokens and proxy credentials while JVN reads untrusted code, so
the tools it runs over that code (ast-grep, ripgrep, git) get only ``CHILD_ENVIRONMENT_NAMES`` and the
locale. A model step's command-line connector is the one child that keeps the whole environment: the
model CLI needs its credentials. Every value below is a synthetic marker.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from jev_navigator.connectors import CommandConnector
from jev_navigator.index import tools

SECRET = "synthetic-model-key"


@pytest.fixture
def credentialed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHETIC_MODEL_API_KEY", SECRET)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:synthetic-proxy-password@127.0.0.1:9")
    monkeypatch.setenv("LC_CTYPE", "C.UTF-8")


def _environment(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def test_a_tool_jvn_starts_gets_only_the_allowlisted_environment(tmp_path: Path, credentialed_host) -> None:
    seen = _environment(tools.command_output(["env"], tmp_path).decode())

    assert "SYNTHETIC_MODEL_API_KEY" not in seen
    assert "HTTPS_PROXY" not in seen
    assert seen["PATH"] == os.environ["PATH"]
    assert seen["LC_CTYPE"] == "C.UTF-8"


def test_git_runs_in_the_c_locale_without_the_hosts_credentials(
    tmp_path: Path, credentialed_host, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "git-environment"
    stand_in = tmp_path / "bin" / "git"
    stand_in.parent.mkdir()
    stand_in.write_text(f"#!/bin/sh\nenv > '{record}'\necho true\n")
    stand_in.chmod(stand_in.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{stand_in.parent}{os.pathsep}{os.environ['PATH']}")

    assert tools.inside_git_worktree(tmp_path) is True
    seen = _environment(record.read_text())
    assert "SYNTHETIC_MODEL_API_KEY" not in seen
    assert seen["LC_ALL"] == "C"


def test_a_model_step_connector_keeps_the_hosts_environment(credentialed_host) -> None:
    reply = CommandConnector(["env"], name="env").complete("unused prompt")

    assert _environment(reply)["SYNTHETIC_MODEL_API_KEY"] == SECRET
