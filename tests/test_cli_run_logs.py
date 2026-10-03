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
from jev_navigator.directives.places import MOVES
from jev_navigator.judgments.questions import request_sha256
from jev_navigator.run_files import KEY_MENTION_MOVE
from jev_navigator.testing import ScriptedJevClient

MARKER = "zebra_marker_7731"
TARGET = "the check that limits the number of items"


def marked_repository(root: Path) -> Path:
    """The marker sits on each function's first line, which a neighbour's signature quotes, and in
    its body; symbol names stay free of it."""
    commit_files(
        root,
        {
            "app/entry.py": (
                "from .policy import admit\n\n"
                f"def handle(item, {MARKER}=None):\n    # {MARKER} entry\n    return admit(item)\n"
            ),
            "app/policy.py": (
                f"def admit(item, {MARKER}=None):\n    # {MARKER} policy\n    return len(item) <= 3\n"
            ),
        },
    )
    return root


def limit_client() -> ScriptedJevClient:
    return ScriptedJevClient(
        nouls=lambda question_id, question, state: 0.96 if "len(item) <= 3" in str(state) else 0.04,
        choices={"open_first": {"0": 1.0}},
    )


def decoded_base64_fields(value: object) -> list[bytes]:
    if isinstance(value, dict):
        return [
            decoded
            for key, item in value.items()
            for decoded in (
                [base64.b64decode(item or "")] if key.endswith("_base64") else decoded_base64_fields(item)
            )
        ]
    if isinstance(value, list):
        return [decoded for item in value for decoded in decoded_base64_fields(item)]
    return []


def holds_code(path: Path) -> bool:
    content = path.read_bytes()
    if MARKER.encode() in content:
        return True
    if path.suffix != ".jsonl":
        return False
    records = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    return any(MARKER.encode() in decoded for decoded in decoded_base64_fields(records))


def files_holding_code(folder: Path) -> list[str]:
    """Run files holding the marker as text or inside a base64 field."""
    return sorted(path.name for path in folder.iterdir() if holds_code(path))


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


def test_a_capped_run_and_its_resume_send_exactly_the_requests_of_an_uninterrupted_run(
    tmp_path: Path,
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    whole_client, first_client, resumed_client = limit_client(), limit_client(), limit_client()
    options = {"fact_cache_dir": tmp_path / "fact-cache"}
    start = ("app/entry.py:5",)
    whole_budget = SearchBudget(beam_width=1, max_calls=5)
    capped_budget = SearchBudget(beam_width=1, max_calls=1)
    create_evidence_pack(
        repository, ("app/",), TARGET, start, tmp_path / "whole", whole_budget, whole_client, **options
    )
    create_evidence_pack(
        repository, ("app/",), TARGET, start, tmp_path / "first", capped_budget, first_client, **options
    )

    # Act
    resumed = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        start,
        tmp_path / "second",
        whole_budget,
        resumed_client,
        resume_from=tmp_path / "first",
        **options,
    )

    # Assert
    def hashes(client: ScriptedJevClient) -> list[str]:
        return [request_sha256(state, questions) for state, questions in client.requests]

    assert resumed["search"]["outcome"] == "found"
    assert hashes(first_client) + hashes(resumed_client) == hashes(whole_client)
    assert any(MARKER in json.dumps(state) for state, _ in resumed_client.requests)
    assert files_holding_code(tmp_path / "first") == []


def offered_signatures(manifest: dict) -> list[str]:
    return [
        offered["signature"]
        for step in manifest["search"]["history"]
        for offered in step["judgments"].get("could_contain", [])
    ]


def test_a_default_run_names_a_neighbour_by_location_and_symbol(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")

    # Act
    manifest = find_pack(repository, tmp_path / "pack", "find", 5)

    # Assert
    assert "app/policy.py:1 admit" in offered_signatures(manifest)


def test_keep_requests_keeps_each_neighbour_signature_whole(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")

    # Act
    manifest = find_pack(repository, tmp_path / "pack", "find", 5, keep_requests=True)

    # Assert
    assert f"app/policy.py:1 `def admit(item, {MARKER}=None):` (called by handle)" in offered_signatures(
        manifest
    )


def template_literal_repository(root: Path) -> Path:
    """TypeScript functions whose first lines hold a template literal and a sql-tagged literal, each
    with backticks and the marker, plus a caller that makes them neighbours."""
    commit_files(
        root,
        {
            "web/routes.ts": (
                "export function load(router: any, id: string) "
                f"{{ return router.get(`/{MARKER}/runs/${{id}}`); }}\n"
                "\n"
                f"export function query(db: any) {{ return db.run(sql`SELECT {MARKER} FROM users`); }}\n"
                "\n"
                "export function handle(router: any, db: any) {\n"
                "  load(router, 'one');\n"
                "  return query(db);\n"
                "}\n"
            ),
        },
    )
    return root


@pytest.mark.parametrize("max_calls", [5, 1])
def test_a_default_run_folder_holds_no_code_from_template_literals(tmp_path: Path, max_calls: int) -> None:
    # Arrange
    repository = template_literal_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    create_evidence_pack(
        repository,
        ("web/",),
        TARGET,
        ("web/routes.ts:6",),
        output,
        SearchBudget(max_calls=max_calls, beam_width=1),
        ScriptedJevClient(nouls=lambda question_id, question, state: 0.04),
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Assert
    assert files_holding_code(output) == []


def key_mention_repository(root: Path) -> Path:
    """A dictionary key holding the marker, read in one file and mentioned again in another."""
    commit_files(
        root,
        {
            "app/settings.py": f"def limit(config):\n    return config['{MARKER}']\n",
            "app/other.py": f"def other(config):\n    value = config.get('{MARKER}')\n    return value\n",
        },
    )
    return root


@pytest.mark.parametrize("max_calls", [5, 1])
def test_a_default_run_folder_never_stores_a_mentioned_key(tmp_path: Path, max_calls: int) -> None:
    # Arrange
    repository = key_mention_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    manifest = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        ("app/settings.py:2",),
        output,
        SearchBudget(max_calls=max_calls, beam_width=1),
        ScriptedJevClient(nouls=lambda question_id, question, state: 0.5),
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Assert
    relations = [
        offered["relationship"]["relation"]
        for step in manifest["search"]["history"]
        for offered in step["judgments"].get("could_contain", [])
        if (offered.get("relationship") or {}).get("move") == "keys_mentioned"
    ]
    assert relations and set(relations) == {"mentions a key (app/other.py:1)"}
    assert [name for name in files_holding_code(output) if name != "answers.jsonl"] == []


def test_the_key_mention_move_name_is_the_one_places_lists() -> None:
    # Act and assert
    assert KEY_MENTION_MOVE in MOVES


def test_keep_requests_keeps_a_key_mention_relation_verbatim(tmp_path: Path) -> None:
    # Arrange
    repository = key_mention_repository(tmp_path / "repository")

    # Act
    manifest = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        ("app/settings.py:2",),
        tmp_path / "pack",
        SearchBudget(max_calls=5, beam_width=1),
        ScriptedJevClient(nouls=lambda question_id, question, state: 0.5),
        fact_cache_dir=tmp_path / "fact-cache",
        keep_requests=True,
    )

    # Assert
    relations = {
        offered["relationship"]["relation"]
        for step in manifest["search"]["history"]
        for offered in step["judgments"].get("could_contain", [])
        if (offered.get("relationship") or {}).get("move") == KEY_MENTION_MOVE
    }
    assert relations == {f"mentions `{MARKER}`"}
