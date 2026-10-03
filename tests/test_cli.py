from __future__ import annotations

import inspect
import io
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from git_repos import commit_files

from jev_navigator.cli import (
    SCHEMA_VERSION,
    _load_typesafe_environment,
    _scope_warning,
    create_evidence_pack,
    main,
)
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.judgments.store import SHARED_STORE_VARIABLE
from jev_navigator.testing import ScriptedJevClient


def test_an_empty_find_reports_how_much_of_the_scope_it_examined(tmp_path: Path) -> None:
    # Arrange
    repository = tmp_path / "repository"
    commit_files(repository, {"app/entry.py": "def handle(item):\n    return item\n"})
    commit_files(
        repository,
        {
            "app/mail.py": "def send(letter):\n    return letter\n",
            "app/billing.py": "def bill(account):\n    return account\n",
        },
    )
    client = ScriptedJevClient(nouls=lambda question_id, question, state: 0.04)
    output = tmp_path / "evidence"

    # Act
    manifest = create_evidence_pack(
        repository,
        ("app/",),
        "the check that limits the number of items",
        ("app/entry.py:2",),
        output,
        SearchBudget(),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Assert
    assert manifest["search"]["outcome"] == "scope_incomplete"
    assert (manifest["search"]["files_examined"], manifest["search"]["code_files"]) == (1, 3)
    assert (
        "Outcome: **scope_incomplete, not found: 1 of 3 code files examined, 2 never reached**"
        in (output / "report.md").read_text()
    )


def test_evidence_pack_runs_the_real_index_and_search_boundary(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )
    output = tmp_path / "evidence"
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: (
            0.96 if "len(item) <= 3" in state["slice"]["code"] else 0.04
        ),
        choices={"open_first": {"0": 1.0}},
    )

    manifest = create_evidence_pack(
        repository,
        ("app/",),
        "the check that limits the number of items",
        ("app/entry.py:4",),
        output,
        SearchBudget(max_depth=2, max_steps=3, max_calls=3, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    written = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == written["schema_version"] == SCHEMA_VERSION
    assert written["source"]["revision"] == repository_commit(repository)
    assert written["navigator"]["package_version"] == "0.1.0"
    assert len(written["navigator"]["source_tree_sha256"]) == 64
    assert written["provider"] == {
        "input_tokens": 200,
        "responses_without_usage": 0,
        "requested_model": "jev-scripted",
        "served_model": "jev-scripted",
    }
    assert written["search"]["outcome"] == "found"
    assert written["search"]["found"][0]["source"]["file"] == "app/policy.py"
    assert written["search"]["found"][0]["probability"] == 0.96
    assert written["search"]["history"][-1]["operation"] == "stop"
    assert "Candidates not independently opened" in (output / "report.md").read_text()
    assert (output / "journal.jsonl").read_text()


def test_evidence_pack_counts_responses_without_usage_instead_of_adding_zero_tokens(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )
    output = tmp_path / "evidence"
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: (
            0.96 if "len(item) <= 3" in state["slice"]["code"] else 0.04
        ),
        choices={"open_first": {"0": 1.0}},
        input_tokens_per_call=None,
    )

    manifest = create_evidence_pack(
        repository,
        ("app/",),
        "the check that limits the number of items",
        ("app/entry.py:4",),
        output,
        SearchBudget(max_depth=2, max_steps=3, max_calls=3, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    provider = json.loads((output / "manifest.json").read_text())["provider"]
    assert provider["input_tokens"] == 0
    assert provider["responses_without_usage"] == manifest["search"]["calls"] > 0


def test_budget_pack_reopens_its_saved_frontier_in_a_second_cli_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jev_navigator import cli

    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )
    clients: list[ScriptedJevClient] = []

    def client() -> ScriptedJevClient:
        instance = ScriptedJevClient(
            nouls=lambda question_id, question, state: (
                0.96 if "len(item) <= 3" in state["slice"]["code"] else 0.04
            ),
            choices={"open_first": {"0": 1.0}},
        )
        instance.close = lambda: None
        clients.append(instance)
        return instance

    monkeypatch.setattr(cli, "_load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "TypeSafeJevClient", client)
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = ["find", "the item limit", "--repo", str(repository), "--start", "app/entry.py:4"]

    assert main([*common, "--max-calls", "1", "--beam-width", "1", "--out", str(first)]) == 0
    cap = json.loads((first / "manifest.json").read_text())
    assert cap["search"]["outcome"] == "budget"
    assert cap["search"]["calls"] == len(clients[0].requests) == 1
    assert cap["search"]["not_inspected"]
    assert (first / "resume.json").is_file()
    cap_bytes = (first / "manifest.json").read_bytes()
    capsys.readouterr()

    assert (
        main([*common, "--max-calls", "1", "--beam-width", "1", "--resume", str(first), "--out", str(second)])
        == 0
    )
    resumed = json.loads((second / "manifest.json").read_text())
    assert resumed["search"]["outcome"] == "found"
    assert resumed["search"]["found"][0]["source"]["file"] == "app/policy.py"
    assert resumed["search"]["starts"] == cap["search"]["starts"]
    assert resumed["search"]["calls"] == 2
    assert resumed["search"]["steps"] == 2
    assert len(resumed["search"]["history"]) > len(cap["search"]["history"])
    assert len(clients[1].requests) == 1
    assert (first / "manifest.json").read_bytes() == cap_bytes
    assert (second / "answers.jsonl").read_text().count("\n") == 2


def test_entry_selection_replays_cached_calls_after_its_cap(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/one.py": "def one():\n    return 1\n",
            "app/two.py": "def two():\n    return 2\n",
            "tests/test_one.py": "def test_one():\n    assert True\n",
        },
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    budget = SearchBudget(max_calls=1, beam_width=1)
    client1 = ScriptedJevClient()
    cap = create_evidence_pack(repository, (), "find one", (), first, budget, client1)

    assert cap["search"]["outcome"] == "budget"
    assert cap["entry_selection"] is None
    assert json.loads((first / "resume.json").read_text())["stage"] == "entry"
    assert len(client1.requests) == 1

    client2 = ScriptedJevClient()
    resumed = create_evidence_pack(repository, (), "find one", (), second, budget, client2, resume_from=first)
    assert resumed["entry_selection"] is not None
    assert resumed["search"]["calls"] == 2
    assert resumed["search"]["entry_calls"] == 2
    assert len(client2.requests) == 1  # the first entry decision came from answers.jsonl
    assert json.loads((second / "resume.json").read_text())["stage"] == "navigation"


def _capped_pack(tmp_path: Path, name: str) -> tuple[Path, Path, SearchBudget]:
    repository = tmp_path / "repository"
    if not repository.exists():
        commit_files(
            repository,
            {
                "app/one.py": "def one():\n    return 1\n",
                "app/two.py": "def two():\n    return 2\n",
                "tests/test_one.py": "def test_one():\n    assert True\n",
            },
        )
    budget = SearchBudget(max_calls=1, beam_width=1)
    create_evidence_pack(repository, (), "find one", (), tmp_path / name, budget, ScriptedJevClient())
    return repository, tmp_path / name, budget


def test_resuming_an_earlier_receipt_without_the_unreported_count_keeps_that_count_unknown(
    tmp_path: Path,
) -> None:
    repository, first, budget = _capped_pack(tmp_path, "first")
    manifest = json.loads((first / "manifest.json").read_text())
    del manifest["provider"]["responses_without_usage"]
    (first / "manifest.json").write_text(json.dumps(manifest))

    resumed = create_evidence_pack(
        repository,
        (),
        "find one",
        (),
        tmp_path / "second",
        budget,
        ScriptedJevClient(input_tokens_per_call=None),
        resume_from=first,
    )

    assert resumed["provider"]["responses_without_usage"] is None
    assert (
        "Responses without usage: not known (earlier receipt)"
        in (tmp_path / "second" / "report.md").read_text()
    )


def test_resuming_a_receipt_that_knows_its_unreported_count_keeps_adding_to_it(tmp_path: Path) -> None:
    repository, first, budget = _capped_pack(tmp_path, "first")

    resumed = create_evidence_pack(
        repository,
        (),
        "find one",
        (),
        tmp_path / "second",
        budget,
        ScriptedJevClient(input_tokens_per_call=None),
        resume_from=first,
    )

    assert resumed["provider"]["responses_without_usage"] == resumed["search"]["calls_this_invocation"] > 0


def test_resume_rejects_changed_source_before_reusing_the_frontier(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(repository, {"policy.py": "def policy():\n    return 1\n"})
    first = tmp_path / "first"
    create_evidence_pack(
        repository, (), "policy", ("policy.py:1",), first, SearchBudget(max_calls=0), ScriptedJevClient()
    )
    (repository / "policy.py").write_text("def policy():\n    return 2\n")
    next_client = ScriptedJevClient()

    with pytest.raises(ValueError, match="source or scope changed"):
        create_evidence_pack(
            repository,
            (),
            "policy",
            ("policy.py:1",),
            tmp_path / "second",
            SearchBudget(max_calls=1),
            next_client,
            resume_from=first,
        )
    assert next_client.requests == []


def test_budget_pack_survives_an_unreadable_unopened_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/live.py": "def live():\n    return 1\n",
            "app/unreadable.py": "def unavailable():\n    return 2\n",
        },
    )
    unreadable = repository / "app/unreadable.py"
    unreadable.chmod(0)
    try:
        output = tmp_path / "evidence"
        client = ScriptedJevClient()
        manifest = create_evidence_pack(
            repository,
            (),
            "live behavior",
            ("app/live.py:1",),
            output,
            SearchBudget(max_calls=0, max_depth=0),
            client,
        )
        assert manifest["search"]["outcome"] == "budget"
        assert manifest["search"]["unavailable_files"]["app/unreadable.py"]
        assert (output / "resume.json").is_file()
        assert (
            "PermissionError"
            in json.loads((output / "resume.json").read_text())["unavailable_files"]["app/unreadable.py"]
        )
        assert client.requests == []
    finally:
        unreadable.chmod(0o600)


def test_findall_pack_composes_seed_search_with_disconnected_enumeration(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "entry.py": "from policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "policy.py": "def admit(item):\n    return len(item) <= 3\n",
            "independent.py": "def fits(values):\n    return not len(values) > 3\n",
        },
    )

    def answer(question_id, question, state):
        code = (
            state["items"][int(question_id.rsplit("#", 1)[1])]["code"]
            if "items" in state
            else state["slice"]["code"]
        )
        return 0.96 if "len(" in code else 0.04

    output = tmp_path / "all-evidence"
    client = ScriptedJevClient(nouls=answer)
    manifest = create_evidence_pack(
        repository,
        (),
        "the item count limit check",
        ("entry.py:4",),
        output,
        SearchBudget(beam_width=1),
        client,
        workflow="findall",
        fact_cache_dir=tmp_path / "facts",
    )

    saved = json.loads((output / "manifest.json").read_text())
    assert manifest["search"]["outcome"] == "scope_examined"
    assert saved["workflow"] == "findall"
    assert saved["search"]["coverage"] == "functions_examined"
    assert {item["name"] for item in saved["search"]["found"]} == {"admit", "fits"}
    assert saved["seed_search"]["outcome"] == "found"
    assert saved["search"]["calls"] == len(client.requests)
    assert saved["seed_search"]["calls"] + saved["search"]["enumeration_calls"] == len(client.requests)
    assert all(item["request_sha256"] for item in saved["search"]["found"])
    assert "independent.py" in (output / "report.md").read_text()
    assert (output / "journal.jsonl").read_text()


def test_findall_schema_and_json_use_the_shared_cli_contract(capsys):
    from jev_navigator.cli import _parse_args

    assert main(["schema", "findall"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["properties"]["command"]["const"] == "findall"
    assert schema["properties"]["max_calls"]["default"] == 48
    assert "resume" in schema["properties"]
    args = _parse_args(["--json", '{"command":"findall","target":"the item limit check"}'])
    assert args.command == "findall"
    assert args.target == "the item limit check"
    assert args.max_calls == 48


def test_evidence_pack_chooses_a_real_entry_when_no_start_is_supplied(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: (
            0.96 if "len(item) <= 3" in state["slice"]["code"] else 0.04
        ),
        choices={"automatic_entry_path": {"1": 1.0}},
    )

    manifest = create_evidence_pack(
        repository,
        (),
        "the check that limits the number of items",
        (),
        tmp_path / "evidence-auto",
        SearchBudget(max_depth=2, max_steps=3, max_calls=3, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    assert manifest["search"]["outcome"] == "found"
    assert manifest["search"]["calls"] == 2
    assert manifest["search"]["entry_calls"] == 1
    assert manifest["entry_selection"]["selected_file"] == "app/policy.py"
    assert manifest["search"]["found"][0]["source"]["file"] == "app/policy.py"


def test_zero_choice_probability_remains_zero_in_the_uninspected_frontier(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "policy.py": (
                "def first():\n    return 'wrong'\n\n"
                "def target():\n    return 'wanted'\n\n"
                "def third():\n    return 'also wrong'\n"
            )
        },
    )
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: 0.04,
        choices={"automatic_entry_span": {"0": 0.8, "1": 0.0, "2": 0.2}},
    )

    manifest = create_evidence_pack(
        repository,
        (),
        "wanted",
        (),
        tmp_path / "zero-probability",
        SearchBudget(max_steps=1, max_calls=2, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    target = next(item for item in manifest["search"]["not_inspected"] if "target" in item["signature"])
    assert target["priority"] == 0.0


def test_cancelled_navigation_writes_a_resumable_evidence_pack(tmp_path: Path) -> None:
    # Arrange
    repository = tmp_path / "repository"
    commit_files(repository, {"policy.py": "def policy(item):\n    return item\n"})

    class InterruptingClient:
        model = "interrupting"

        def ask(self, state, questions):
            del state, questions
            raise KeyboardInterrupt

    output = tmp_path / "cancelled-evidence"

    # Act
    manifest = create_evidence_pack(
        repository,
        (),
        "the policy",
        ("policy.py:1",),
        output,
        SearchBudget(beam_width=1),
        InterruptingClient(),
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Assert
    assert manifest["search"]["outcome"] == "cancelled"
    assert manifest["search"]["steps"] == 0
    assert manifest["search"]["not_inspected"][0]["reason"] == "cancelled"
    assert "Outcome: **cancelled**" in (output / "report.md").read_text()
    records = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
    assert any(record["kind"] == "history_step" for record in records)
    assert records[-1]["kind"] == "terminal"
    assert records[-1]["outcome"] == "cancelled"


@pytest.fixture
def offline_main(monkeypatch: pytest.MonkeyPatch) -> dict:
    """``main`` without credentials or a search: each pack request is recorded by parameter name and
    answered with ``outcome``."""
    from jev_navigator import cli

    class Client:
        model = "test"

        def close(self) -> None:
            pass

    signature = inspect.signature(cli.create_evidence_pack)
    calls: dict = {"outcome": "found", "packs": []}

    def create_evidence_pack(*args, **kwargs):
        calls["packs"].append(signature.bind(*args, **kwargs).arguments)
        return {"search": {"outcome": calls["outcome"], "calls": 1}, "provider": {"requested_model": "test"}}

    monkeypatch.setattr(cli, "_load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "TypeSafeJevClient", Client)
    monkeypatch.setattr(cli, "create_evidence_pack", create_evidence_pack)
    return calls


def test_main_maps_a_cancelled_pack_to_the_shell_interrupt_status(tmp_path: Path, offline_main: dict) -> None:
    # Arrange
    offline_main["outcome"] = "cancelled"

    # Act
    status = main(["find", "the policy", "--repo", str(tmp_path), "--out", str(tmp_path / "out")])

    # Assert
    assert status == 130


@pytest.mark.parametrize(
    ("request_options", "max_calls"),
    [
        ({}, 24),
        ({"--max-calls": "5"}, 5),
        ({"--max-calls": "none"}, None),
    ],
    ids=["capped-by-default", "explicit-cap", "explicitly-uncapped"],
)
@pytest.mark.parametrize("input_mode", ["flags", "json"])
def test_live_calls_are_capped_unless_the_caller_lifts_the_cap(
    tmp_path: Path, offline_main: dict, input_mode: str, request_options: dict, max_calls: int | None
) -> None:
    # Arrange
    if input_mode == "flags":
        arguments = ["find", "the policy", "--repo", str(tmp_path), "--out", str(tmp_path / "out")]
        arguments += [part for option in request_options.items() for part in option]
    else:
        request = {"target": "the policy", "repo": str(tmp_path), "out": str(tmp_path / "out")}
        if request_options:
            value = request_options["--max-calls"]
            request["max_calls"] = None if value == "none" else int(value)
        arguments = ["--json", json.dumps(request)]

    # Act
    status = main(arguments)

    # Assert
    assert status == 0
    assert offline_main["packs"][0]["budget"].max_calls == max_calls


def repository_commit(repository: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_each_existing_typesafe_environment_value_wins_independently(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    path = tmp_path / "env"
    path.write_text("TYPESAFE_API_KEY=file-key\nTYPESAFE_BASE_URL=http://file.example/gateway\n")
    environment = {"TYPESAFE_API_KEY": "process-key"}

    _load_typesafe_environment(environment, path)

    assert environment == {
        "TYPESAFE_API_KEY": "process-key",
        "TYPESAFE_BASE_URL": "http://file.example/gateway",
    }


def test_user_dotenv_key_is_loaded_without_shell_evaluation(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    path = tmp_path / "env"
    path.write_text(
        "TYPESAFE_API_KEY='file-value'\n"
        "TYPESAFE_BASE_URL='http://127.0.0.1:4777/jvn'\n"
        "UNRELATED=$(touch should-not-run)\n"
    )
    environment: dict[str, str] = {}

    _load_typesafe_environment(environment, path)

    assert environment == {
        "TYPESAFE_API_KEY": "file-value",
        "TYPESAFE_BASE_URL": "http://127.0.0.1:4777/jvn",
        "UNRELATED": "$(touch should-not-run)",  # loaded literally, never shell-evaluated
    }
    assert not (tmp_path / "should-not-run").exists()


def test_dotenv_base_url_reaches_the_real_sdk_system_one_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("typesafe_sdk")
    pytest.importorskip("dotenv")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    received: list[tuple[str, bytes]] = []
    response = (
        b'{"model":"jev-1.13.0","usage":{"input_tokens":1,"output_tokens":1},'
        b'"answers":{"match":{"type":"noul","noul":0.9}}}'
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            body = self.rfile.read(int(self.headers["content-length"]))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
        path = tmp_path / "env"
        path.write_text(
            "TYPESAFE_API_KEY=local-viewer-key\n"
            f"TYPESAFE_BASE_URL=http://127.0.0.1:{server.server_port}/jvn\n"
        )
        _load_typesafe_environment(os.environ, path)

        answer = TypeSafeJevClient().ask(
            {"code": "return wanted"},
            {"match": {"type": "noul", "instructions": "Does code return wanted?"}},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert received and received[0][0] == "/jvn/v1/systemone"
    assert b'"code":"return wanted"' in received[0][1]
    assert answer.noul("match").probability == 0.9


def test_large_scope_warning_starts_above_twenty_thousand_files() -> None:
    assert _scope_warning(20_000) is None
    assert "20,001" in _scope_warning(20_001)


@pytest.mark.parametrize("search_here", [False, True])
@pytest.mark.parametrize("input_mode", ["flags", "json-file", "json-stdin", "json-inline"])
def test_find_defaults_to_unique_results_under_invocation_directory(
    tmp_path, monkeypatch, capsys, search_here, input_mode
):
    from jev_navigator import cli

    repository = tmp_path if search_here else tmp_path / "repo"
    repository.mkdir(exist_ok=True)
    (repository / "policy.py").write_text("def admit(item):\n    return len(item) <= 3\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_load_typesafe_environment", lambda environment: None)

    class Client(ScriptedJevClient):
        def close(self):
            pass

    monkeypatch.setattr(
        cli,
        "TypeSafeJevClient",
        lambda: Client(
            nouls=lambda question_id, question, state: 1.0,
            choices={"open_first": {"0": 1.0}},
        ),
    )
    for _ in range(2):
        request = {"target": "the policy"}
        if not search_here:
            request["repo"] = str(repository)
        if input_mode == "flags":
            arguments = ["find", "the policy", "--repo", str(repository)]
        elif input_mode == "json-file":
            request_file = tmp_path / "jvn-results" / "request.json"
            request_file.parent.mkdir(exist_ok=True)
            request_file.write_text(json.dumps(request))
            arguments = ["--json", str(request_file)]
        elif input_mode == "json-inline":
            arguments = ["--json", json.dumps(request)]
        else:
            monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
            arguments = ["--json", "-"]
        assert main(arguments) == 0
        captured = capsys.readouterr()
        if input_mode != "flags":
            response = json.loads(captured.out)
            assert response["search"]["outcome"] == "found"
            pack = Path(response["output_directory"])
            assert response["manifest"] == str(pack / "manifest.json")
            assert response["report"] == str(pack / "report.md")
            assert "jvn" in captured.err
    packs = [path for path in (tmp_path / "jvn-results").iterdir() if path.is_dir()]
    assert len(packs) == 2
    for pack in packs:
        manifest = json.loads((pack / "manifest.json").read_text())
        assert manifest["search"]["outcome"] == "found"
        assert manifest["source"]["tracked_files"] == 1
        assert (pack / "report.md").is_file()
        assert (pack / "journal.jsonl").is_file()
    if not search_here:
        assert not (repository / "jvn-results").exists()


def test_report_distinguishes_included_lines_from_an_unopened_candidate(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "policy.py").write_text(
        "class Policy:\n    def classify(self, probability):\n        return probability >= 0.8\n"
    )
    manifest = create_evidence_pack(
        repository,
        (),
        "the complete policy",
        (),
        tmp_path / "pack",
        SearchBudget(beam_width=1),
        ScriptedJevClient(nouls=lambda *_: 0.95),
        fact_cache_dir=tmp_path / "cache",
    )
    assert manifest["search"]["outcome"] == "found"
    assert manifest["search"]["not_inspected"]
    covered = [entry for entry in manifest["search"]["not_inspected"] if entry["included_in_opened_span"]]
    assert covered
    assert all(entry["included_in_opened_span"] == "policy.py:1-3" for entry in covered)
    report = (tmp_path / "pack" / "report.md").read_text()
    assert "Lines included in opened span" in report
    assert "Candidates not independently opened" in report
    assert "Search stopped after finding a match" in report


@pytest.mark.parametrize(
    "request_text, message",
    [
        ("[]", "object"),
        ("{", "JSON"),
        ('{"target": "policy", "typo": 2}', "typo"),
        ('{"target": "policy", "max_steps": "two"}', "max_steps"),
        ('{"target": "policy", "max_steps": true}', "max_steps"),
        ('{"target": "policy", "max_steps": -1}', "max_steps"),
        ('{"target": "policy", "max_calls": "none"}', "max_calls"),
        ('{"target": "policy", "beam_width": 0}', "beam_width"),
        ('{"target": "policy", "prefix": "app/"}', "prefix"),
        ('{"target": "policy", "verbose": "yes"}', "verbose"),
        ('{"target": "policy", "repo": null}', "repo"),
        ('{"target": 42}', "target"),
        ('{"command": "missing", "target": "policy"}', "command"),
        ("{}", "target"),
    ],
)
def test_json_request_errors_fail_before_search(monkeypatch, capsys, request_text, message):
    from jev_navigator import cli

    def unexpected_client():
        pytest.fail("invalid input reached the model client")

    monkeypatch.setattr(cli, "TypeSafeJevClient", unexpected_client)
    monkeypatch.setattr("sys.stdin", io.StringIO(request_text))
    with pytest.raises(SystemExit) as error:
        main(["--json", "-"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert message in captured.err


def test_json_pipeline_reaches_sdk_and_preserves_explicit_options(tmp_path):
    pytest.importorskip("typesafe_sdk")
    repository = tmp_path / "repo"
    repository.mkdir()
    for name in ("first.py", "second.py"):
        (repository / name).write_text(
            "def entry(items):\n    return admit(items)\n\ndef admit(items):\n    return len(items) <= 3\n"
        )
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            received.append(payload)
            answers = {}
            for name, question in payload["questions"].items():
                if question["type"] == "noul":
                    answers[name] = {"type": "noul", "noul": 0.95}
                else:
                    labels = list(question["criteria"])
                    answers[name] = {
                        "type": "choice",
                        "choice": labels[0],
                        "confidence": 1.0,
                        "probabilities": {label: float(label == labels[0]) for label in labels},
                    }
            body = json.dumps(
                {"model": "jev-test", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 10}}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    payload = {
        "command": "find",
        "target": "--the item-count check",
        "repo": str(repository),
        "prefix": ["first.py", "second.py"],
        "start": ["first.py:1", "second.py:1"],
        "max_calls": None,
        "beam_width": 2,
        "verbose": True,
        "out": str(tmp_path / "pack"),
    }
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from jev_navigator.cli import main; raise SystemExit(main())",
                "--json",
                "-",
            ],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "TYPESAFE_API_KEY": "local-test-key",
                "TYPESAFE_BASE_URL": f"http://127.0.0.1:{server.server_port}",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    manifest = json.loads(Path(response["manifest"]).read_text())
    assert received
    assert manifest["target"] == "--the item-count check"
    assert manifest["source"]["prefixes"] == ["first.py", "second.py"]
    assert manifest["requested_starts"] == ["first.py:1", "second.py:1"]
    assert manifest["budget"]["max_calls"] is None
    assert manifest["budget"]["beam_width"] == 2
    assert response["search"]["outcome"] == "found"
    assert "request" in result.stderr


@pytest.mark.parametrize("arguments", [["--help"], ["help"], ["find", "--help"], ["help", "find"]])
def test_help_teaches_the_minimal_and_structured_invocations(arguments, capsys):
    try:
        assert main(arguments) == 0
    except SystemExit as exit_status:
        assert exit_status.code == 0
    output = capsys.readouterr()
    assert not output.err
    assert "Examples:" in output.out
    assert "jvn schema find" in output.out
    assert "jvn find" in output.out


def test_schema_discovery_needs_no_credentials_or_model(monkeypatch, capsys):
    from jev_navigator import cli

    def unexpected_client():
        pytest.fail("schema discovery reached the model client")

    monkeypatch.setattr(cli, "TypeSafeJevClient", unexpected_client)
    assert main(["schema", "find"]) == 0
    output = capsys.readouterr()
    assert not output.err
    schema = json.loads(output.out)
    assert schema["required"] == ["target"]
    assert schema["additionalProperties"] is False
    fields = schema["properties"]
    assert fields["prefix"]["type"] == "array"
    assert fields["start"]["items"] == {"type": "string"}
    assert fields["beam_width"]["default"] == 3
    assert fields["beam_width"]["minimum"] == 1
    assert fields["max_calls"]["default"] == 24
    assert fields["max_calls"]["type"] == ["integer", "null"]
    assert fields["repo"]["default"] == "."
    assert fields["verbose"]["type"] == "boolean"


@pytest.mark.parametrize("cap", [0, 2, 3])
def test_findall_budget_stop_writes_partial_pack_with_completed_results(tmp_path, cap):
    repo = tmp_path / "repo"
    commit_files(
        repo,
        {
            "policy.py": "def admit(item):\n    return len(item) <= 3\n",
            "other.py": "def fits(item):\n    return len(item) <= 3\n",
        },
    )
    provider = ScriptedJevClient(default_noul=0.96)
    out = tmp_path / "pack"
    result = create_evidence_pack(
        repo,
        (),
        "item limit",
        ("policy.py:1",),
        out,
        SearchBudget(max_calls=cap),
        provider,
        workflow="findall",
    )
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["search"]["outcome"] == result["search"]["outcome"]
    assert saved["search"]["calls"] <= cap
    if cap < 3:
        assert saved["search"]["outcome"] == "budget"
        assert saved["search"]["coverage"] == "partial"
        assert saved["search"]["remaining_files"]
    else:
        assert saved["search"]["found"]
    assert (out / "report.md").is_file()


def test_stats_cli_measures_methods_and_filters_line_ranges_without_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jev_navigator import cli

    root = tmp_path / "source"
    root.mkdir()
    (root / "example.py").write_text(
        "class Box:\n"
        "    def larger(self):\n"
        "        value = 1\n"
        "        value += 2\n"
        "        return value\n\n"
        "def small():\n"
        "    return 1\n"
    )
    monkeypatch.chdir(tmp_path)

    def forbidden_provider():
        raise AssertionError("structural analysis must not construct a model client")

    monkeypatch.setattr(cli, "TypeSafeJevClient", forbidden_provider)
    assert (
        main(
            [
                "--json",
                json.dumps(
                    {
                        "command": "stats",
                        "repo": str(root),
                        "kind": ["function"],
                        "limit": 1,
                        "min_lines": 4,
                        "max_lines": 4,
                    }
                ),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    output = Path(result["output_directory"])
    assert output.parent == tmp_path / "jvn-results"
    pack = json.loads((output / "statistics.json").read_text())
    assert pack["counts"]["totals"] == {"function": 2}
    assert pack["largest"]["ranking"][0]["name"] == "larger"
    assert pack["largest"]["ranking"][0]["size"] == 4
    assert pack["ranges"]["measured"] == 2
    assert pack["ranges"]["listed"] == 1
    assert pack["ranges"]["per_file"]["example.py"][0]["lines"] == [2, 5]
    assert "example.py:2-5" in (output / "statistics.md").read_text()
    assert "whole files" not in (output / "statistics.md").read_text()
    assert "0 requests" in captured.err


def test_stats_schema_and_validation_share_the_cli_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["schema", "stats"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "target" not in schema["properties"]
    assert schema["properties"]["operation"]["items"]["enum"] == ["count", "largest", "range"]
    with pytest.raises(SystemExit) as exc:
        main(["--json", '{"command":"stats","operation":["guess"]}'])
    assert exc.value.code == 2
    assert not (tmp_path / "jvn-results").exists()
    assert main(["stats", "--min-lines", "10", "--max-lines", "2"]) == 1
    assert not (tmp_path / "jvn-results").exists()


def test_each_run_names_its_answer_store_and_a_fresh_store_isolates_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jev_navigator import cli

    # Arrange
    repository = tmp_path / "repository"
    functions = "".join(
        f"def admit_{index}(item):\n    return len(item) <= {index}\n\n\n" for index in range(40)
    )
    commit_files(repository, {"app/policy.py": functions})
    clients: list[ScriptedJevClient] = []

    def client() -> ScriptedJevClient:
        instance = ScriptedJevClient(default_noul=0.96)
        instance.close = lambda: None
        clients.append(instance)
        return instance

    monkeypatch.setattr(cli, "_load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "TypeSafeJevClient", client)
    common = ["findall", "the item limit", "--repo", str(repository)]
    arm_a, arm_b = tmp_path / "arm-a.sqlite", tmp_path / "arm-b.sqlite"
    default_store = os.environ.get(SHARED_STORE_VARIABLE)

    # Act
    main([*common, "--answer-store", str(arm_a), "--out", str(tmp_path / "a1")])
    first_err = capsys.readouterr().err
    main([*common, "--answer-store", str(arm_a), "--out", str(tmp_path / "a2")])
    main([*common, "--answer-store", str(arm_b), "--out", str(tmp_path / "b1")])
    later_err = capsys.readouterr().err

    # Assert: the repeat on the same store sends only the one request that learns the served model
    assert f"answer store: {arm_a}" in first_err
    assert f"answer store: {arm_b}" in later_err
    first, repeat, isolated = (len(instance.requests) for instance in clients)
    assert first > 1 and repeat == 1 and isolated == first
    assert os.environ.get(SHARED_STORE_VARIABLE) == default_store, (
        "the flag never travels through the environment"
    )
