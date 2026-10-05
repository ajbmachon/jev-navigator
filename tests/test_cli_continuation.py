"""The terminal may grant another find allowance; saved evidence survives declining it."""

from __future__ import annotations

import json
import os
import pty
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from git_repos import commit_files
from isolated_jvn import JVN

from jev_navigator import cli
from jev_navigator.cli import FIND_ALL_QUESTION
from jev_navigator.cli_resume import load_resume, save_resume
from jev_navigator.directives.find_code import SearchBudget, find_code
from jev_navigator.directives.places import MOVES, function_place
from jev_navigator.index import tools
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.run_files import PlaceLabels
from jev_navigator.testing import ScriptedJevClient


def test_saved_find_frontier_restores_relationship_binding(tmp_path: Path) -> None:
    # Arrange
    files = {
        "app/target.py": "def check():\n    return True\n",
        "app/entry.py": "from app.target import check\n\n\ndef handle():\n    return check()\n",
    }
    repository = tmp_path / "repository"
    commit_files(repository, files)
    index = CodeIndex(repository, list(files))
    start = function_place(index, index.find_definition("handle")[0])
    client = ScriptedJevClient(nouls=lambda _question_id, _question, _state: 0.1)
    result = find_code(
        index,
        Judge(client),
        "the check function",
        [start],
        moves={"callees": MOVES["callees"]},
        budget=SearchBudget(max_steps=1, beam_width=1),
    )
    resume_file = tmp_path / "resume.json"

    # Act
    save_resume(resume_file, index, result, labels=PlaceLabels(index), entry_pending=False)
    saved = json.loads(resume_file.read_text())
    restored = load_resume(resume_file, index, find_all_question=FIND_ALL_QUESTION.question_id)

    # Assert
    frontier_record = saved["result"]["not_inspected"][0]
    assert frontier_record["relationship"]["move"] == "callees"
    assert frontier_record["relationship"]["binding"] == {
        "status": "resolved",
        "reason": "imported from app/target.py",
        "proven": True,
        "target": {"file": "app/target.py", "start": 1, "end": 2, "name": "check"},
    }
    frontier = restored.result.not_inspected[0]
    assert frontier.place.move == "callees"
    assert frontier.place.binding.status == "resolved"
    assert frontier.place.binding.target.file == "app/target.py"


def _frontier_labels(resume_file: Path) -> list[tuple[str, str]]:
    records = json.loads(resume_file.read_text())["result"]["not_inspected"]
    return [(record["place_key"], record["signature"]) for record in records]


def test_a_resumed_search_that_stops_again_keeps_its_frontier_names_while_the_parser_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: a stopped search leaves the caller of `check` unopened. A resumed run, whose fact cache
    # does not hold that file, stops again before opening it, and every ast-grep it starts is killed.
    files = {
        "app/target.py": "def check():\n    return True\n",
        "app/entry.py": "from app.target import check\n\n\ndef handle():\n    return check()\n",
    }
    repository = tmp_path / "repository"
    commit_files(repository, files)
    first = CodeIndex(repository, list(files), fact_cache_dir=tmp_path / "first-facts")
    start = function_place(first, first.find_definition("check")[0])
    client = ScriptedJevClient(nouls=lambda _question_id, _question, _state: 0.1)
    stopped = find_code(
        first,
        Judge(client),
        "the check function",
        [start],
        moves={"callers": MOVES["callers"]},
        budget=SearchBudget(max_steps=1, beam_width=1),
    )
    save_resume(tmp_path / "first.json", first, stopped, labels=PlaceLabels(first), entry_pending=False)
    resumed = CodeIndex(repository, list(files), fact_cache_dir=tmp_path / "resumed-facts")
    checkpoint = load_resume(
        tmp_path / "first.json", resumed, find_all_question=FIND_ALL_QUESTION.question_id
    )
    killed = tmp_path / "killed-bin"
    killed.mkdir()
    (killed / tools.AST_GREP).write_text("#!/bin/sh\nkill -9 $$\n")
    (killed / tools.AST_GREP).chmod(0o755)
    monkeypatch.setenv("PATH", f"{killed}{os.pathsep}{os.environ['PATH']}")

    # Act
    save_resume(
        tmp_path / "second.json",
        resumed,
        checkpoint.result,
        labels=PlaceLabels(resumed, checkpoint.frontier_labels),
        entry_pending=False,
    )

    # Assert: the frontier is saved again with the name the first save wrote, though the resumed index
    # never parsed the caller's file. A place no save labelled shows only its location, never parsing.
    assert _frontier_labels(tmp_path / "second.json") == _frontier_labels(tmp_path / "first.json")
    assert _frontier_labels(tmp_path / "first.json") == [("app/entry.py:4-5", "app/entry.py:4 handle")]
    assert PlaceLabels(resumed)("app/entry.py:4-5") == "app/entry.py:4"
    assert "app/entry.py" not in resumed.parsed_files


def test_a_restored_place_set_aside_again_carries_its_code_signature(tmp_path: Path) -> None:
    # Arrange: a stopped search leaves the caller of `check` unopened; its resumption opens that
    # caller and is interrupted at its first request, which sets the caller aside again.
    files = {
        "app/target.py": "def check():\n    return True\n",
        "app/entry.py": "from app.target import check\n\n\ndef handle():\n    return check()\n",
    }
    repository = tmp_path / "repository"
    commit_files(repository, files)
    index = CodeIndex(repository, list(files))
    start = function_place(index, index.find_definition("check")[0])
    moves = {"callers": MOVES["callers"]}
    unsure = ScriptedJevClient(nouls=lambda _question_id, _question, _state: 0.5)
    budget = SearchBudget(max_steps=1, beam_width=1)
    stopped = find_code(index, Judge(unsure), "the check function", [start], moves=moves, budget=budget)
    save_resume(tmp_path / "resume.json", index, stopped, labels=PlaceLabels(index), entry_pending=False)
    restored = load_resume(
        tmp_path / "resume.json", index, find_all_question=FIND_ALL_QUESTION.question_id
    ).result

    class InterruptingClient:
        model = "interrupting"

        def ask(self, state, questions):
            del state, questions
            raise KeyboardInterrupt

    # Act
    cancelled = find_code(
        index, Judge(InterruptingClient()), "the check function", [], moves=moves, resume=restored
    )

    # Assert: the caller keeps the code signature the first search gave it, not the saved label.
    (caller,) = stopped.not_inspected
    (set_aside,) = cancelled.not_inspected
    assert (set_aside.place_key, set_aside.reason) == ("app/entry.py:4-5", "cancelled")
    assert set_aside.signature == set_aside.place.signature == caller.signature
    assert caller.signature.startswith("app/entry.py:4 `def handle():`")


def _callers_repository(root: Path) -> Path:
    """Six callers of `admit`, which fill the frontier of a search started at it."""
    calls_admit = "from .policy import admit\n\n\ndef call{}(item):\n    return admit(item)\n"
    callers = {f"app/caller{number}.py": calls_admit.format(number) for number in range(6)}
    commit_files(root, {"app/policy.py": "def admit(item):\n    return len(item) <= 3\n", **callers})
    return root


def _stop_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *resumed_options: str) -> tuple[Path, Path]:
    """Two `jvn find` runs at `admit`, the second resuming the first. No answer is sure enough to stop
    either, so each opens one place and stops on its budget."""
    repository = _callers_repository(tmp_path / "repository")
    unsure = ScriptedJevClient(nouls=lambda _question_id, _question, _state: 0.5)
    unsure.close = lambda: None
    clients = iter([unsure, unsure])
    monkeypatch.setattr(cli, "load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "system_one_client", lambda environment: next(clients))
    first, second = tmp_path / "first", tmp_path / "second"

    def jvn_find(output: Path, *options: str) -> list[str]:
        command = ["find", "the item limit", "--repo", str(repository), "--start", "app/policy.py:2"]
        return [*command, "--max-calls", "1", "--out", str(output), *options]

    assert cli.main(jvn_find(first)) == 0
    assert cli.main(jvn_find(second, "--resume", str(first), *resumed_options)) == 0
    return first, second


def _manifest_frontier(pack: Path) -> dict[str, str]:
    manifest = json.loads((pack / "manifest.json").read_text())
    return {entry["place"]: entry["signature"] for entry in manifest["search"]["not_inspected"]}


def test_a_resumed_jvn_find_that_stops_again_keeps_every_carried_over_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Act
    first, second = _stop_twice(tmp_path, monkeypatch)

    # Assert: every place carried over and still unopened keeps the label its first save wrote, in the
    # resume state and in the manifest, and each caller's label names its function.
    first_labels = dict(_frontier_labels(first / "resume.json"))
    resumed_labels = dict(_frontier_labels(second / "resume.json"))
    shown = _manifest_frontier(second)
    carried = first_labels.keys() & resumed_labels.keys()
    functions = carried & {f"app/caller{number}.py:4-5" for number in range(6)}
    assert len(functions) >= 4
    assert all(" call" in first_labels[key] for key in functions)
    assert {key: resumed_labels[key] for key in carried} == {key: first_labels[key] for key in carried}
    assert {key: shown[key] for key in carried} == {key: first_labels[key] for key in carried}


def test_with_keep_requests_a_carried_over_place_shows_its_code_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Act
    first, second = _stop_twice(tmp_path, monkeypatch, "--keep-requests")

    # Assert: --keep-requests keeps each frontier place's signature, the code line Jev would read,
    # for a place the earlier run found as for one this run found.
    shown = _manifest_frontier(second)
    code_lines = {f"app/caller{number}.py:4-5": f"`def call{number}(item):`" for number in range(6)}
    functions = dict(_frontier_labels(first / "resume.json")).keys() & shown.keys() & code_lines.keys()
    assert len(functions) >= 4
    assert all(shown[key].startswith(f"{key.removesuffix('-5')} {code_lines[key]}") for key in functions)


@pytest.mark.parametrize("workflow", ["find", "findall"])
@pytest.mark.parametrize("answer", ["yes", "no", "eof", "json", "pipe", "zero"])
def test_search_continues_only_with_terminal_consent(tmp_path: Path, answer: str, workflow: str) -> None:
    pytest.importorskip("typesafe_sdk")
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )
    client = ScriptedJevClient(
        nouls=lambda _id, _question, state: (
            0.96
            if "len(item) <= 3"
            in (
                state["items"][int(_id.rsplit("#", 1)[1])]["code"]
                if "items" in state
                else state["slice"]["code"]
            )
            else 0.04
        ),
        choices={"open_first": {"0": 1.0}},
    )

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - HTTP server boundary
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            response = client.send(request["state"], request["questions"])
            body = json.dumps(response.json()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    output = tmp_path / "packs" / "first"
    request = {
        "command": workflow,
        "target": "the item limit",
        "repo": str(repository),
        "start": ["app/entry.py:4"],
        "out": str(output),
        "max_calls": 0 if answer == "zero" else 2 if workflow == "findall" else 1,
        "beam_width": 1,
    }
    arguments = (
        ["--json", json.dumps(request)]
        if answer == "json"
        else [
            workflow,
            request["target"],
            "--repo",
            str(repository),
            "--start",
            "app/entry.py:4",
            "--out",
            str(output),
            "--max-calls",
            str(request["max_calls"]),
            "--beam-width",
            "1",
        ]
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    serving = Thread(target=server.serve_forever)
    serving.start()
    command = [*JVN, *arguments]
    env = {
        **os.environ,
        "TYPESAFE_API_KEY": "local-test-key",
        "TYPESAFE_BASE_URL": f"http://127.0.0.1:{server.server_port}",
        "TYPESAFE_DEFAULT_MODEL": "jev-scripted",
    }
    try:
        if answer == "pipe":
            done = subprocess.run(command, env=env, input="yes\n", text=True, capture_output=True, timeout=60)
            code, transcript = done.returncode, done.stdout + done.stderr
        else:
            code, transcript = _terminal(
                command,
                env,
                b"\x04" if answer == "eof" else b"yes\n" if answer in ("yes", "json", "zero") else b"no\n",
            )
    finally:
        server.shutdown()
        server.server_close()
        serving.join()
    assert code == 0, transcript
    first = json.loads((output / "manifest.json").read_text())
    assert first["search"]["outcome"] == "budget"
    assert (output / "resume.json").is_file()
    assert first["search"]["calls_this_invocation"] == request["max_calls"]
    packs = list(output.parent.glob("*/manifest.json"))
    if answer == "yes":
        assert len(client.requests) == (3 if workflow == "findall" else 2)
        assert len(packs) == 2
        resumed = json.loads(next(p for p in packs if p.parent != output).read_text())
        assert resumed["resume_from"] == str(output)
        assert resumed["search"]["outcome"] == ("scope_examined" if workflow == "findall" else "found")
        assert resumed["search"]["calls"] == len(client.requests)
        assert resumed["search"]["calls_this_invocation"] == 1
        assert resumed["search"]["found"][0]["source"]["file"] == "app/policy.py"
        if workflow == "find":
            assert resumed["search"]["starts"] == first["search"]["starts"]
        else:
            assert resumed["seed_search"] == first["seed_search"]
    else:
        assert len(client.requests) == request["max_calls"]
        assert len(packs) == 1
    prompted = "Continue the saved search" in transcript
    assert prompted == (answer in ("yes", "no", "eof"))


def _terminal(command: list[str], env: dict, answer: bytes) -> tuple[int, str]:
    master, slave = pty.openpty()
    received: list[bytes] = []

    def read() -> None:
        try:
            while chunk := os.read(master, 8192):
                received.append(chunk)
        except OSError:  # PTYs return EIO after the child closes its last slave descriptor.
            pass

    child = subprocess.Popen(command, env=env, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    reader = Thread(target=read)
    reader.start()
    try:
        os.write(master, answer)
        status = child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        reader.join()
        os.close(master)
    return status, b"".join(received).decode()
