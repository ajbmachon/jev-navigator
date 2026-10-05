"""A CLI run folder keeps code locations and hashes by default; full code only with keep_requests."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from git_repos import commit_files
from isolated_jvn import JVN
from stored_messages import digested

from jev_navigator.cli import FIND_ALL_QUESTION, create_evidence_pack
from jev_navigator.cli_resume import load_resume
from jev_navigator.cli_trace import create_trace_evidence_pack
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.journal import ERROR_TEXT_VARIABLE, error_text_kept
from jev_navigator.judgments.questions import request_sha256
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


def find_pack(
    repository: Path, output: Path, workflow: str, max_calls: int, client: object | None = None, **options
) -> dict:
    return create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        ("app/entry.py:5",),
        output,
        SearchBudget(max_calls=max_calls, beam_width=1),
        client or limit_client(),
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


def test_a_pack_that_kept_its_request_text_is_not_resumed_without_keep_requests(tmp_path: Path) -> None:
    # Arrange: a budget-stopped pack whose journal holds the text of its requests
    repository = marked_repository(tmp_path / "repository")
    first, second = tmp_path / "first", tmp_path / "second"
    find_pack(repository, first, "find", 1, keep_requests=True)

    # Act
    with pytest.raises(ValueError) as refusal:
        find_pack(repository, second, "find", 1, resume_from=first)

    # Assert: the run writes nothing, and says which flag the pack needs
    assert str(first) in str(refusal.value)
    assert "--keep-requests" in str(refusal.value)
    assert not second.exists()


def test_a_pack_that_kept_its_request_text_resumes_with_keep_requests(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    first, second = tmp_path / "first", tmp_path / "second"
    find_pack(repository, first, "find", 1, keep_requests=True)

    # Act
    resumed = find_pack(repository, second, "find", 1, keep_requests=True, resume_from=first)

    # Assert
    assert resumed["search"]["calls"] == 2


def with_code_signatures(value: object, code_signatures: Mapping[str, str]) -> object:
    """``value`` with the ``signature`` of every place named in ``code_signatures`` set to its code."""
    if isinstance(value, list):
        return [with_code_signatures(item, code_signatures) for item in value]
    if not isinstance(value, dict):
        return value
    shown = {key: with_code_signatures(item, code_signatures) for key, item in value.items()}
    place = value.get("place", value.get("place_key"))
    if "signature" in value and place in code_signatures:
        shown["signature"] = code_signatures[place]
    return shown


def save_as_before_labels(pack: Path, code_signatures: Mapping[str, str]) -> None:
    """Rewrite ``pack`` as JVN wrote it from 29.09 to 03.10: the manifest, the journal and resume.json
    gave each neighbour its code signature, which quotes the neighbour's first line."""
    for name in ("manifest.json", "resume.json"):
        record = json.loads((pack / name).read_text())
        (pack / name).write_text(json.dumps(with_code_signatures(record, code_signatures)))
    lines = (pack / "journal.jsonl").read_text().splitlines()
    records = [with_code_signatures(json.loads(line), code_signatures) for line in lines]
    (pack / "journal.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))


def test_a_run_resumed_from_a_pack_that_saved_neighbour_code_holds_no_code_text(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    first, second = tmp_path / "first", tmp_path / "second"
    find_pack(repository, first, "find", 1)
    index = CodeIndex.from_directory(repository, ("app/",), fact_cache_dir=tmp_path / "fact-cache")
    frontier = load_resume(
        first / "resume.json", index, find_all_question=FIND_ALL_QUESTION.question_id
    ).result.not_inspected
    save_as_before_labels(first, {entry.place_key: entry.signature for entry in frontier})
    assert files_holding_code(first) == ["journal.jsonl", "manifest.json", "resume.json"]

    # Act
    find_pack(repository, second, "find", 1, resume_from=first)

    # Assert
    assert files_holding_code(second) == []


@pytest.mark.parametrize("keep_requests", [False, True])
def test_a_resumed_run_continues_a_journal_written_in_its_own_mode_unchanged(
    tmp_path: Path, keep_requests: bool
) -> None:
    # Arrange: a default journal names each neighbour by its label; with keep_requests it quotes the
    # neighbour's code.
    repository = marked_repository(tmp_path / "repository")
    first, second = tmp_path / "first", tmp_path / "second"
    find_pack(repository, first, "find", 1, keep_requests=keep_requests)
    earlier = (first / "journal.jsonl").read_text()

    # Act
    find_pack(repository, second, "find", 1, keep_requests=keep_requests, resume_from=first)

    # Assert
    assert (MARKER in earlier) is keep_requests
    assert (second / "journal.jsonl").read_text().startswith(earlier)


def test_a_search_stopped_in_a_folder_whose_name_holds_a_tilde_resumes(tmp_path: Path) -> None:
    # Arrange: the stopped search leaves unopened the import line before `handle`, a line range whose
    # key, app/v~2/entry.py:1-2, holds a tilde before its line numbers.
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/v~2/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/v~2/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )

    def search(output: Path, **options) -> dict:
        start, budget = ("app/v~2/entry.py:4",), SearchBudget(max_calls=1, beam_width=1)
        options["fact_cache_dir"] = tmp_path / "fact-cache"
        return create_evidence_pack(
            repository, ("app/",), TARGET, start, output, budget, limit_client(), **options
        )

    stopped = search(tmp_path / "first")

    # Act
    resumed = search(tmp_path / "second", resume_from=tmp_path / "first")

    # Assert
    assert "app/v~2/entry.py:1-2" in [entry["place"] for entry in stopped["search"]["not_inspected"]]
    assert resumed["search"]["outcome"] == "found"
    assert [visit["place"] for visit in resumed["search"]["found"]] == ["app/v~2/policy.py:1-2"]


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

    monkeypatch.setattr(cli, "load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "system_one_client", lambda environment: client())
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
    interrupted = {**options, "answer_store": tmp_path / "interrupted-answers.sqlite"}
    start = ("app/entry.py:5",)
    whole_budget = SearchBudget(beam_width=1, max_calls=5)
    capped_budget = SearchBudget(beam_width=1, max_calls=1)
    create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        start,
        tmp_path / "whole",
        whole_budget,
        whole_client,
        answer_store=tmp_path / "whole-answers.sqlite",
        **options,
    )
    create_evidence_pack(
        repository, ("app/",), TARGET, start, tmp_path / "first", capped_budget, first_client, **interrupted
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
        **interrupted,
    )

    # Assert
    def hashes(client: ScriptedJevClient) -> list[str]:
        return [request_sha256(state, questions) for state, questions in client.requests]

    assert resumed["search"]["outcome"] == "found"
    assert hashes(first_client) + hashes(resumed_client) == hashes(whole_client)
    assert any(MARKER in json.dumps(state) for state, _ in resumed_client.requests)
    assert files_holding_code(tmp_path / "first") == []


class InterruptedOnce(ThreadPoolExecutor):
    """The round's pool, with one Ctrl-C in one window between opening the round and merging it:
    ``"creating"``, before any request is sent, or ``"shutting_down"``, after the answers landed."""

    window = "creating"
    interrupts_left = 1

    def __init__(self, *args, **kwargs) -> None:
        self._interrupt_in("creating")
        super().__init__(*args, **kwargs)

    def shutdown(self, *args, **kwargs) -> None:
        super().shutdown(*args, **kwargs)
        self._interrupt_in("shutting_down")

    def _interrupt_in(self, window: str) -> None:
        if type(self).window == window and type(self).interrupts_left:
            type(self).interrupts_left -= 1
            raise KeyboardInterrupt


@pytest.mark.parametrize("window", ["creating", "shutting_down"])
def test_ctrl_c_before_a_round_is_merged_and_its_resume_send_exactly_the_requests_of_an_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str
) -> None:
    """Ctrl-C lands after a round's places were opened but before its answers are merged: while the
    round's thread pool is created, or after the answers arrived while it shuts down. The opened
    place must stay on the saved frontier, so Resume asks it or replays its stored answer and goes
    on, instead of ending with nothing left to open."""
    # Arrange
    from jev_navigator.directives import find_code as find_code_module

    class Interrupted(InterruptedOnce):
        pass

    Interrupted.window = window

    repository = marked_repository(tmp_path / "repository")
    whole_client, first_client, resumed_client = limit_client(), limit_client(), limit_client()
    options = {"fact_cache_dir": tmp_path / "fact-cache"}
    interrupted = {**options, "answer_store": tmp_path / "interrupted-answers.sqlite"}
    start = ("app/entry.py:5",)
    budget = SearchBudget(beam_width=1, max_calls=5)
    create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        start,
        tmp_path / "whole",
        budget,
        whole_client,
        answer_store=tmp_path / "whole-answers.sqlite",
        **options,
    )
    monkeypatch.setattr(find_code_module, "ThreadPoolExecutor", Interrupted)
    cancelled = create_evidence_pack(
        repository, ("app/",), TARGET, start, tmp_path / "first", budget, first_client, **interrupted
    )

    # Act
    resumed = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        start,
        tmp_path / "second",
        budget,
        resumed_client,
        resume_from=tmp_path / "first",
        **interrupted,
    )

    # Assert
    def hashes(client: ScriptedJevClient) -> list[str]:
        return [request_sha256(state, questions) for state, questions in client.requests]

    assert cancelled["search"]["outcome"] == "cancelled"
    assert resumed["search"]["outcome"] == "found"
    assert hashes(first_client) + hashes(resumed_client) == hashes(whole_client)


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


KEY_MENTION_SCOPES = {
    "dictionary key": (
        {
            "app/settings.py": f"def limit(config):\n    return config['{MARKER}']\n",
            "app/other.py": f"def other(config):\n    value = config.get('{MARKER}')\n    return value\n",
        },
        "app/settings.py:2",
        "app/other.py:1",
    ),
    "backtick-quoted key": (
        {
            "app/settings.ts": (
                f"export function limit(api: any) {{\n  return api.get(`v1/{MARKER}/runs`);\n}}\n"
            ),
            "app/other.ts": (
                f"export function other(api: any) {{\n  const runs = api.get(`v1/{MARKER}/runs`);\n"
                "  return runs;\n}\n"
            ),
        },
        "app/settings.ts:2",
        "app/other.ts:1",
    ),
}


def key_mention_pack(tmp_path: Path, scope: str, max_calls: int, **options) -> dict:
    files, start, _ = KEY_MENTION_SCOPES[scope]
    repository = tmp_path / "repository"
    commit_files(repository, files)
    return create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        (start,),
        tmp_path / "pack",
        SearchBudget(max_calls=max_calls, beam_width=1),
        ScriptedJevClient(nouls=lambda question_id, question, state: 0.5),
        fact_cache_dir=tmp_path / "fact-cache",
        **options,
    )


def neighbour_relations(manifest: dict) -> set[str]:
    return {
        offered["relationship"]["relation"]
        for step in manifest["search"]["history"]
        for offered in step["judgments"].get("could_contain", [])
        if "relation" in (offered.get("relationship") or {})
    }


@pytest.mark.parametrize("scope", sorted(KEY_MENTION_SCOPES))
@pytest.mark.parametrize("max_calls", [5, 1])
def test_a_default_run_folder_never_stores_a_mentioned_key(
    tmp_path: Path, scope: str, max_calls: int
) -> None:
    # Act
    manifest = key_mention_pack(tmp_path, scope, max_calls)

    # Assert
    mention = KEY_MENTION_SCOPES[scope][2]
    assert f"mentions a key ({mention})" in neighbour_relations(manifest)
    assert files_holding_code(tmp_path / "pack") == []


@pytest.mark.parametrize("scope", sorted(KEY_MENTION_SCOPES))
def test_keep_requests_keeps_a_key_mention_relation_verbatim(tmp_path: Path, scope: str) -> None:
    # Act
    manifest = key_mention_pack(tmp_path, scope, 5, keep_requests=True)

    # Assert
    assert any(MARKER in relation for relation in neighbour_relations(manifest))


def test_a_budget_stop_after_opening_a_key_mention_keeps_the_key_out_of_resume_state(
    tmp_path: Path,
) -> None:
    # Arrange
    files, start, _ = KEY_MENTION_SCOPES["dictionary key"]
    mentions = {
        f"app/mention_{number}.py": f"def mention_{number}(config):\n    return config['{MARKER}']\n"
        for number in range(5)
    }
    repository = tmp_path / "repository"
    commit_files(repository, {**files, **mentions})
    output = tmp_path / "pack"

    # Act
    manifest = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        (start,),
        output,
        SearchBudget(max_calls=3, beam_width=1),
        ScriptedJevClient(nouls=lambda question_id, question, state: 0.5),
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Assert
    opened_by_mention = [
        visit
        for visit in manifest["search"]["searched"] + manifest["search"]["unsure"]
        if visit["source"]["reached_by"].startswith("mentions a key")
    ]
    assert manifest["search"]["outcome"] == "budget"
    assert opened_by_mention
    assert files_holding_code(output) == []


def strings_starting_with(value: object, prefix: str) -> set[str]:
    if isinstance(value, dict):
        return {text for item in value.values() for text in strings_starting_with(item, prefix)}
    if isinstance(value, list):
        return {text for item in value for text in strings_starting_with(item, prefix)}
    return {value} if isinstance(value, str) and value.startswith(prefix) else set()


def test_a_key_mention_outside_any_function_is_shown_at_its_mention_line_everywhere(
    tmp_path: Path,
) -> None:
    # Arrange
    preamble = "".join(f"STEP_{number} = {number}\n" for number in range(1, 15))
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/settings.py": f"def limit(config):\n    return config['{MARKER}']\n",
            "app/other.py": f"{preamble}print(CONFIG['{MARKER}'])\n",
        },
    )

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
    )

    # Assert
    opened = [visit["place"] for visit in manifest["search"]["searched"] + manifest["search"]["unsure"]]
    assert any(place.startswith("app/other.py:15~") for place in opened)
    assert strings_starting_with(manifest, "mentions a key") == {"mentions a key (app/other.py:15)"}


def echoing_server() -> ThreadingHTTPServer:
    """A provider that refuses every request with 422 and echoes the request it got, as many
    validation errors do: the request, and so the code, comes back in the error. ``served`` counts
    the requests it received."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.server.served += 1
            sent = json.loads(self.rfile.read(int(self.headers["content-length"])))
            served = json.dumps({"detail": [{"msg": "unprocessable request", "input": sent}]}).encode()
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(served)))
            self.end_headers()
            self.wfile.write(served)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.served = 0
    threading.Thread(target=server.serve_forever, name="echoing-jev", daemon=True).start()
    return server


ERROR_TEXT_SETTINGS = {
    "default": ([], {}),
    "flag": (["--no-error-text"], {}),
    "variable": ([], {"JEV_NAVIGATOR_ERROR_TEXT": "off"}),
}


def echoed_run(
    tmp_path: Path, command: str, setting: str, *, output_name: str = "pack", extra: tuple[str, ...] = ()
) -> tuple[subprocess.CompletedProcess, Path, int]:
    """``jvn COMMAND`` through the real TypeSafe client against a provider that echoes every request
    in a 422, with the error-text setting named by ``setting``; also how many requests the provider
    received. A second run in the same ``tmp_path`` reuses its repository."""
    repository = tmp_path / "repository"
    if not repository.exists():
        marked_repository(repository)
    output = tmp_path / output_name
    options, variables = ERROR_TEXT_SETTINGS[setting]
    arguments = [
        command,
        TARGET,
        "--repo",
        str(repository),
        "--start",
        "app/entry.py:5",
        "--out",
        str(output),
        *extra,
    ]
    server = echoing_server()
    environment = {
        **os.environ,
        **variables,
        "TYPESAFE_API_KEY": "local-test-key",
        "TYPESAFE_BASE_URL": f"http://127.0.0.1:{server.server_port}",
    }
    try:
        finished = subprocess.run(
            [*JVN, *arguments, *options], capture_output=True, text=True, env=environment
        )
    finally:
        server.shutdown()
        server.server_close()
    return finished, output, server.served


@pytest.mark.parametrize("command", ["find", "findall", "trace"])
@pytest.mark.parametrize("setting", ["default", "flag", "variable"])
def test_an_echoed_error_body_stays_in_the_run_folder_unless_error_text_is_off(
    tmp_path: Path, command: str, setting: str
) -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")

    # Act
    finished, output, served = echoed_run(tmp_path, command, setting)

    # Assert
    assert finished.returncode == 1, finished.stderr
    records = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
    assert any(record["kind"] == "failure" for record in records)
    assert served == sum(record["kind"] == "http_attempt" for record in records) > 0
    assert files_holding_code(output) == (["journal.jsonl"] if setting == "default" else [])
    if command != "trace":
        assert json.loads((output / "manifest.json").read_text())["search"]["failure"]["status"] == 422


def test_a_resume_with_error_text_off_digests_the_echoed_error_bodies_it_carries(tmp_path: Path) -> None:
    # Arrange: the first run keeps its error text, so its journal holds the echoed 422 bodies
    pytest.importorskip("typesafe_sdk")
    first, first_output, _ = echoed_run(tmp_path, "find", "default")
    assert first.returncode == 1 and files_holding_code(first_output) == ["journal.jsonl"]

    # Act
    resumed, resumed_output, _ = echoed_run(
        tmp_path, "find", "flag", output_name="resumed", extra=("--resume", str(first_output))
    )

    # Assert
    assert resumed.returncode == 1, resumed.stderr
    assert files_holding_code(resumed_output) == []


class EchoesTheRequest:
    """Answers its first request, then fails with an error whose message quotes the request."""

    def __init__(self) -> None:
        self.script = limit_client()
        self.model = self.script.model
        self.asked = 0

    def ask(self, state, questions):
        self.asked += 1
        if self.asked == 2:
            raise RuntimeError(f"422 Unprocessable Entity: {json.dumps(state)}")
        return self.script.ask(state, questions)

    def close(self) -> None:
        pass


class CausedByTheRequest(EchoesTheRequest):
    """Fails its second request with a plain message, raised from a cause that quotes the request."""

    def ask(self, state, questions):
        try:
            return super().ask(state, questions)
        except RuntimeError as quoting:
            raise RuntimeError("the provider rejected the request") from quoting


@pytest.mark.parametrize("keep_error_text", [True, False])
def test_a_cause_quoting_its_request_stays_out_of_the_run_files_when_error_text_is_off(
    tmp_path: Path, keep_error_text: bool
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        find_pack(repository, output, "find", 5, CausedByTheRequest(), keep_error_text=keep_error_text)

    # Assert
    [cause] = json.loads((output / "manifest.json").read_text())["search"]["failure"]["causes"]
    assert (MARKER in cause.get("message", "")) is keep_error_text
    assert files_holding_code(output) == (["manifest.json"] if keep_error_text else [])


@pytest.mark.parametrize("workflow", ["find", "findall"])
@pytest.mark.parametrize("keep_error_text", [True, False])
def test_an_error_quoting_its_request_reaches_stderr_and_the_run_files_unless_error_text_is_off(
    tmp_path: Path, workflow: str, keep_error_text: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    with pytest.raises(RuntimeError, match=MARKER):
        find_pack(repository, output, workflow, 5, EchoesTheRequest(), keep_error_text=keep_error_text)

    # Assert
    assert MARKER in capsys.readouterr().err
    assert json.loads((output / "manifest.json").read_text())["search"]["outcome"] == "failed"
    expected = ["journal.jsonl", "manifest.json", "report.md"] if keep_error_text else []
    assert files_holding_code(output) == expected


def failed_pack_that_kept_its_error_text(tmp_path: Path) -> tuple[Path, Path]:
    """A pack written with error text on whose search failed on an error quoting its request."""
    repository = marked_repository(tmp_path / "repository")
    first = tmp_path / "first"
    with pytest.raises(RuntimeError, match=MARKER):
        find_pack(repository, first, "find", 5, EchoesTheRequest(), keep_error_text=True)
    return repository, first


@pytest.mark.parametrize("keep_error_text", [True, False])
def test_a_resume_rewrites_the_carried_error_text_as_its_own_error_text_setting_says(
    tmp_path: Path, keep_error_text: bool
) -> None:
    # Arrange
    repository, first = failed_pack_that_kept_its_error_text(tmp_path)
    second = tmp_path / "second"

    # Act
    find_pack(repository, second, "find", 5, keep_error_text=keep_error_text, resume_from=first)

    # Assert
    assert files_holding_code(second) == (["journal.jsonl", "manifest.json"] if keep_error_text else [])
    records = [json.loads(line) for line in (second / "journal.jsonl").read_text().splitlines()]
    carried = [record for record in records if record["kind"] == "failure"]
    assert carried and all(("message" in record) is keep_error_text for record in carried)


def test_keep_requests_keeps_the_whole_error_message_even_with_error_text_off(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    with pytest.raises(RuntimeError) as raised:
        find_pack(
            repository, output, "find", 5, EchoesTheRequest(), keep_requests=True, keep_error_text=False
        )

    # Assert
    message = str(raised.value)
    records = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
    assert [record["message"] for record in records if record["kind"] == "failure"] == [message]
    assert json.loads((output / "manifest.json").read_text())["search"]["failure"]["message"] == message


def test_error_text_off_keeps_each_message_as_its_length_and_sha256(tmp_path: Path) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    output = tmp_path / "pack"

    # Act
    with pytest.raises(RuntimeError) as raised:
        find_pack(repository, output, "find", 5, EchoesTheRequest(), keep_error_text=False)

    # Assert
    stored = digested(str(raised.value))
    records = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
    failure_rows = [record for record in records if record["kind"] == "failure"]
    assert [row.items() >= stored.items() and "message" not in row for row in failure_rows] == [True]
    manifest_failure = json.loads((output / "manifest.json").read_text())["search"]["failure"]
    assert manifest_failure.items() >= stored.items() and "message" not in manifest_failure


@pytest.mark.parametrize("value", ["", "no", "OFF"])
def test_an_error_text_setting_other_than_on_or_off_is_refused(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ERROR_TEXT_VARIABLE, value)

    with pytest.raises(ValueError, match=ERROR_TEXT_VARIABLE):
        error_text_kept(no_error_text=False)
