"""`SYSTEM_ONE_ROUTES` through the real `jvn find`: each route is a local stand-in server, and the
run never reads the machine's own key or settings."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from subprocess import PIPE

import pytest
from system_one_stand_in import stand_in

DEAD_ENDPOINT = "http://127.0.0.1:9"
"""Where a request goes when no stand-in is named: a closed local port, never the hosted service."""


def run_find(tmp_path: Path, settings: Mapping[str, str]) -> subprocess.CompletedProcess:
    """`jvn find` on a two-function repository, with only ``settings`` as its System One setup and a
    fake key, so no request can reach a hosted provider."""
    command, environment = find_command(tmp_path, settings)
    return subprocess.run(command, env=environment, text=True, capture_output=True, timeout=120)


def find_command(tmp_path: Path, settings: Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
    repository = tmp_path / "repo"
    repository.mkdir(exist_ok=True)
    (repository / "admit.py").write_text(
        "def entry(items):\n    return admit(items)\n\ndef admit(items):\n    return len(items) <= 3\n"
    )
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith(("SYSTEM_ONE_", "TYPESAFE_"))
    }
    environment |= {"HOME": str(home), "TYPESAFE_API_KEY": "local-test-key"}
    environment |= {"SYSTEM_ONE_DREX_API_KEY": "local-drex-key"}
    environment |= {"SYSTEM_ONE_DECIDER_API_KEY": "local-decider-key"}
    environment |= {"TYPESAFE_BASE_URL": DEAD_ENDPOINT, **settings}
    program = (
        "import signal; signal.signal(signal.SIGINT, signal.default_int_handler); "
        "from jev_navigator.cli import main; raise SystemExit(main())"
    )
    arguments = ["find", "the item-count check", "--repo", str(repository), "--start", "admit.py:1"]
    arguments += ["--max-calls", "4", "--out", str(tmp_path / "pack")]
    return [sys.executable, "-c", program, *arguments], environment


def journal_attempts(tmp_path: Path) -> list[dict]:
    lines = (tmp_path / "pack" / "journal.jsonl").read_text().splitlines()
    return [record for record in map(json.loads, lines) if record["kind"] == "http_attempt"]


def test_the_first_named_route_answers_the_first_request(tmp_path: Path) -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")
    with stand_in("drex-test") as drex, stand_in("jev-test") as jev:
        settings = {
            "SYSTEM_ONE_ROUTES": "drex,jev",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "SYSTEM_ONE_DREX_MODEL": "drex-test",
            "SYSTEM_ONE_JEV_ENDPOINT": jev.url,
            "SYSTEM_ONE_JEV_MODEL": "jev-test",
            "TYPESAFE_BASE_URL": jev.url,
        }

        # Act
        result = run_find(tmp_path, settings)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert drex.received and not jev.received
    manifest = json.loads((tmp_path / "pack" / "manifest.json").read_text())
    assert manifest["provider"]["served_model"] == "drex-test"


def test_a_failing_route_falls_back_to_the_next_and_both_attempts_are_journaled(tmp_path: Path) -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")
    with stand_in("drex-test", status=503) as drex, stand_in("jev-test") as jev:
        settings = {
            "SYSTEM_ONE_ROUTES": "drex,jev",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "SYSTEM_ONE_DREX_MODEL": "drex-test",
            "SYSTEM_ONE_JEV_ENDPOINT": jev.url,
            "SYSTEM_ONE_JEV_MODEL": "jev-test",
        }

        # Act
        result = run_find(tmp_path, settings)

    # Assert
    assert result.returncode == 0, result.stderr
    assert drex.received and jev.received
    first_request = [(attempt["route"], attempt["status"]) for attempt in journal_attempts(tmp_path)[:4]]
    assert first_request == [("drex", 503)] * 3 + [("jev", 200)]


def test_a_configured_local_decider_answers_and_the_hosted_jev_receives_nothing(tmp_path: Path) -> None:
    # Arrange: the default client's address is a stand-in for the hosted Jev
    pytest.importorskip("typesafe_sdk")
    with stand_in("decider-test") as decider, stand_in("jev-hosted") as hosted:
        settings = {
            "SYSTEM_ONE_ROUTES": "decider",
            "SYSTEM_ONE_DECIDER_ENDPOINT": decider.url,
            "SYSTEM_ONE_DECIDER_MODEL": "decider-test",
            "SYSTEM_ONE_DECIDER_INPUT_TOKENS": "8192",
            "SYSTEM_ONE_DECIDER_CONCURRENCY": "4",
            "TYPESAFE_BASE_URL": hosted.url,
        }

        # Act
        result = run_find(tmp_path, settings)

    # Assert
    assert result.returncode == 0, result.stderr
    assert decider.received
    assert not hosted.received


def test_without_routes_the_default_jev_client_answers(tmp_path: Path) -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")
    with stand_in("jev-test") as jev:
        # Act
        result = run_find(tmp_path, {"TYPESAFE_BASE_URL": jev.url})

    # Assert
    assert result.returncode == 0, result.stderr
    assert jev.received
    assert all("route" not in attempt for attempt in journal_attempts(tmp_path))


def test_an_incomplete_route_is_refused_by_name_before_any_request(tmp_path: Path) -> None:
    # Arrange
    with stand_in("jev-test") as jev:
        # Act
        result = run_find(tmp_path, {"SYSTEM_ONE_ROUTES": "decider,jev", "TYPESAFE_BASE_URL": jev.url})

    # Assert
    assert result.returncode == 1
    assert "route 'decider' is incomplete" in result.stderr
    assert not jev.received


def test_an_interrupt_under_routes_finishes_the_request_in_flight_and_leaves_a_resumable_pack(
    tmp_path: Path,
) -> None:
    # Arrange: Drex holds its first answer until the interrupt has been sent
    pytest.importorskip("typesafe_sdk")
    with stand_in("drex-test", held=True) as drex, stand_in("jev-test") as jev:
        settings = {
            "SYSTEM_ONE_ROUTES": "drex,jev",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "SYSTEM_ONE_DREX_MODEL": "drex-test",
            "SYSTEM_ONE_JEV_ENDPOINT": jev.url,
            "SYSTEM_ONE_JEV_MODEL": "jev-test",
        }
        command, environment = find_command(tmp_path, settings)
        running = subprocess.Popen(command, env=environment, text=True, stdout=PIPE, stderr=PIPE)
        try:
            assert drex.arrived.wait(timeout=60), "the first request never reached the Drex stand-in"

            # Act
            running.send_signal(signal.SIGINT)
            drex.release.set()
            _, stderr = running.communicate(timeout=60)
        finally:
            if running.poll() is None:
                running.kill()
                running.wait()

    # Assert
    assert running.returncode == 130, stderr
    assert len(drex.received) == 1 and not jev.received
    attempts = [(attempt["route"], attempt["status"]) for attempt in journal_attempts(tmp_path)]
    assert attempts == [("drex", 200)]
    assert (tmp_path / "pack" / "answers.jsonl").read_text().strip()
    assert (tmp_path / "pack" / "resume.json").is_file()


def test_drex_never_sees_more_requests_in_flight_than_it_admits(tmp_path: Path) -> None:
    # Arrange: a hundred functions make several batches at once; this Drex answers 429 to a third in flight
    pytest.importorskip("typesafe_sdk")
    repository = tmp_path / "repo"
    repository.mkdir()
    functions = "".join(
        f"def check_{index}(items):\n    return len(items) <= {index}\n\n" for index in range(100)
    )
    (repository / "checks.py").write_text(functions)
    with stand_in("drex-test", max_in_flight=2) as drex, stand_in("jev-test") as jev:
        settings = {
            "SYSTEM_ONE_ROUTES": "drex,jev",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "SYSTEM_ONE_DREX_MODEL": "drex-test",
            "SYSTEM_ONE_JEV_ENDPOINT": jev.url,
            "SYSTEM_ONE_JEV_MODEL": "jev-test",
        }
        command, environment = find_command(tmp_path, settings)
        command = [*command[:3], "findall", "the item-count check", "--repo", str(repository)]
        command += ["--max-calls", "none", "--out", str(tmp_path / "pack")]

        # Act
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=120)

    # Assert
    assert result.returncode == 0, result.stderr
    assert len(drex.received) > 2
    assert drex.throttled == 0
    assert not jev.received
