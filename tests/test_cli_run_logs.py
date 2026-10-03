"""A CLI run folder keeps code locations and hashes by default; full code only with keep_requests."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.cli import create_evidence_pack
from jev_navigator.cli_trace import create_trace_evidence_pack
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.judgments.questions import request_sha256
from jev_navigator.testing import ScriptedJevClient

MARKER = "ZEBRA_MARKER_7731"
TARGET = "the check that limits the number of items"


def marked_repository(root: Path) -> Path:
    """Every function holds the marker below its first line, so locations never quote it."""
    commit_files(
        root,
        {
            "app/entry.py": (
                "from .policy import admit\n\n"
                f"def handle(item):\n    # {MARKER} entry\n    return admit(item)\n"
            ),
            "app/policy.py": f"def admit(item):\n    # {MARKER} policy\n    return len(item) <= 3\n",
        },
    )
    return root


def limit_client() -> ScriptedJevClient:
    return ScriptedJevClient(
        nouls=lambda question_id, question, state: 0.96 if "len(item) <= 3" in str(state) else 0.04,
        choices={"open_first": {"0": 1.0}},
    )


def files_holding_code(folder: Path) -> list[str]:
    encoded = [base64.b64encode(f"{MARKER}{tail}".encode()) for tail in (" entry", " policy")]
    return sorted(
        path.name
        for path in folder.iterdir()
        if MARKER.encode() in path.read_bytes() or any(value[:20] in path.read_bytes() for value in encoded)
    )


def find_pack(repository: Path, output: Path, workflow: str, max_calls: int, **options) -> dict:
    return create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        ("app/entry.py:5",),
        output,
        SearchBudget(max_calls=max_calls, beam_width=1),
        limit_client(),
        fact_cache_dir=output.parent / "fact-cache",
        workflow=workflow,
        **options,
    )


@pytest.mark.parametrize(("workflow", "max_calls"), [("find", 5), ("find", 1), ("findall", 5)])
def test_a_default_run_folder_holds_no_code_text(tmp_path: Path, workflow: str, max_calls: int) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    manifest = find_pack(repository, output, workflow, max_calls)

    # Assert
    assert manifest["search"]["calls"] > 0
    assert files_holding_code(output) == []


def test_a_default_trace_folder_holds_no_code_text(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "trace"

    # Act
    create_trace_evidence_pack(
        repository,
        TARGET,
        ("app/entry.py:5",),
        output,
        limit_client(),
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Assert
    assert files_holding_code(output) == []


def test_keep_requests_keeps_the_exact_request_body(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    manifest = find_pack(repository, output, "find", 5, keep_requests=True)

    # Assert
    records = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
    requests = [record for record in records if record["kind"] == "request"]
    bodies = [json.loads(base64.b64decode(request["body_base64"])) for request in requests]
    assert requests
    assert [request_sha256(body["state"], body["questions"]) for body in bodies] == [
        request["request_sha256"] for request in requests
    ]
    assert all(MARKER in json.dumps(body["state"]) for body in bodies)
    assert MARKER in manifest["search"]["found"][0]["code"]
    assert MARKER in (output / "report.md").read_text()


def test_a_resumed_default_run_replays_its_stored_answers_and_finds_the_code(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    first, second = tmp_path / "first", tmp_path / "second"
    budget = SearchBudget(max_calls=1, beam_width=1)
    cap = create_evidence_pack(repository, ("app/",), TARGET, (), first, budget, limit_client())
    client = limit_client()

    # Act
    resumed = create_evidence_pack(
        repository, ("app/",), TARGET, (), second, budget, client, resume_from=first
    )

    # Assert
    assert cap["search"]["outcome"] == "budget"
    assert len(client.requests) == 1
    assert resumed["search"]["calls"] == 2
    assert files_holding_code(first) == files_holding_code(second) == []


@pytest.mark.parametrize("keep_requests", [False, True])
def test_the_json_request_field_keep_requests_reaches_the_run_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_requests: bool
) -> None:
    # Arrange
    from jev_navigator import cli

    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    def client() -> ScriptedJevClient:
        instance = limit_client()
        instance.close = lambda: None
        return instance

    monkeypatch.setattr(cli, "_load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "TypeSafeJevClient", client)
    request = {
        "target": TARGET,
        "repo": str(repository),
        "start": ["app/entry.py:5"],
        "out": str(output),
        "keep_requests": keep_requests,
    }

    # Act
    exit_code = cli.main(["--json", json.dumps(request)])

    # Assert
    assert exit_code == 0
    assert (files_holding_code(output) != []) is keep_requests
