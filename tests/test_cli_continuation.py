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

from jev_navigator.testing import ScriptedJevClient


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
