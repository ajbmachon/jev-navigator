"""The route table: named decision-model routes with automatic fallback, and the generic
System-One client every route runs."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from system_one_stand_in import stand_in

from jev_navigator.adapters.routes import (
    JEV_CONCURRENCY,
    Route,
    RoutedJevClient,
    SystemOneClient,
    routes_from_env,
)
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.thresholds import Thresholds


def test_no_routes_variable_resolves_to_no_routes():
    assert routes_from_env({}) == ()


def test_unknown_route_needs_endpoint_and_model():
    with pytest.raises(ValueError, match="route 'decider' is incomplete"):
        routes_from_env({"SYSTEM_ONE_ROUTES": "decider"})


def test_known_route_shorthand_needs_only_the_flag():
    _requires_typesafe()
    routes = routes_from_env(
        {"SYSTEM_ONE_ROUTES": "drex", "SYSTEM_ONE_DREX": "1", "SYSTEM_ONE_DREX_API_KEY": "test-key"}
    )
    assert routes[0].name == "drex"
    assert routes[0].client.model == "drex-latest"


def test_per_route_settings_beat_the_shorthand():
    _requires_typesafe()
    routes = routes_from_env(
        {
            "SYSTEM_ONE_ROUTES": "drex",
            "SYSTEM_ONE_DREX": "1",
            "SYSTEM_ONE_DREX_MODEL": "drex-v1.1",
            "SYSTEM_ONE_DREX_API_KEY": "test-key",
        }
    )
    assert routes[0].client.model == "drex-v1.1"


TYPESAFE_KEY = "tsk-FAKE-typesafe-key"
CUSTOM_DECIDER = {
    "SYSTEM_ONE_DECIDER_ENDPOINT": "http://127.0.0.1:9",
    "SYSTEM_ONE_DECIDER_MODEL": "decider-4b",
    "SYSTEM_ONE_DECIDER_INPUT_TOKENS": "4096",
    "SYSTEM_ONE_DECIDER_CONCURRENCY": "2",
}


@pytest.mark.parametrize(
    ("name", "settings"), [("drex", {"SYSTEM_ONE_DREX": "1"}), ("decider", CUSTOM_DECIDER)]
)
def test_a_route_other_than_jev_without_its_own_key_is_refused_naming_the_setting(name, settings):
    # Arrange
    environment = {"SYSTEM_ONE_ROUTES": name, "TYPESAFE_API_KEY": TYPESAFE_KEY, **settings}

    # Act and Assert
    with pytest.raises(ValueError, match=f"route '{name}' has no key: set SYSTEM_ONE_{name.upper()}_API_KEY"):
        routes_from_env(environment)


def test_a_drex_route_without_its_own_key_sends_drex_nothing_on_the_wire(monkeypatch):
    # Arrange: the TypeSafe key is in the settings and in the process environment the SDK reads
    _requires_typesafe()
    monkeypatch.setenv("TYPESAFE_API_KEY", TYPESAFE_KEY)
    with stand_in("drex-test", status=500) as drex:
        environment = {
            "SYSTEM_ONE_ROUTES": "drex",
            "SYSTEM_ONE_DREX": "1",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "TYPESAFE_API_KEY": TYPESAFE_KEY,
        }

        # Act and Assert
        with pytest.raises(ValueError, match="set SYSTEM_ONE_DREX_API_KEY"):
            RoutedJevClient(routes_from_env(environment)).ask(
                {"case": "x"}, {"q": {"type": "noul", "instructions": "y?"}}
            )

    # Assert
    assert drex.authorizations == []


def test_the_typesafe_key_reaches_only_the_jev_route_on_the_wire(monkeypatch):
    # Arrange: Drex fails, so the request also goes to Jev; the process environment holds the key too
    _requires_typesafe()
    monkeypatch.setenv("TYPESAFE_API_KEY", TYPESAFE_KEY)
    with stand_in("drex-test", status=500) as drex, stand_in("jev-test") as jev:
        environment = {
            "SYSTEM_ONE_ROUTES": "drex,jev",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "SYSTEM_ONE_DREX_MODEL": "drex-test",
            "SYSTEM_ONE_DREX_API_KEY": "nace-FAKE-drex-key",
            "SYSTEM_ONE_JEV_ENDPOINT": jev.url,
            "SYSTEM_ONE_JEV_MODEL": "jev-test",
            "TYPESAFE_API_KEY": TYPESAFE_KEY,
        }
        routed = RoutedJevClient(routes_from_env(environment))

        # Act
        try:
            routed.ask({"case": "x"}, {"q": {"type": "noul", "instructions": "y?"}})
        finally:
            routed.close()

    # Assert
    assert drex.authorizations and set(drex.authorizations) == {"Bearer nace-FAKE-drex-key"}
    assert set(jev.authorizations) == {f"Bearer {TYPESAFE_KEY}"}


def test_the_first_route_answers_and_the_second_never_runs():
    exchanges: list[tuple[bytes, bytes]] = []
    primary = _server(exchanges)
    backup = _server([])
    routed = RoutedJevClient((Route("primary", primary), Route("backup", backup)))

    answer = routed.ask({"case": "x"}, {"adds_one": {"type": "noul", "instructions": "y?"}})

    assert answer.answers["adds_one"].probability == 0.9
    assert len(exchanges) == 1


def test_failover_asks_the_next_route_after_a_failure():
    dead = _dead_server()
    exchanges: list[tuple[bytes, bytes]] = []
    backup = _server(exchanges)
    routed = RoutedJevClient((Route("dead", dead), Route("backup", backup)))

    answer = routed.ask({"case": "x"}, {"adds_one": {"type": "noul", "instructions": "y?"}})

    assert answer.answers["adds_one"].probability == 0.9
    assert len(exchanges) == 1


def test_all_routes_failing_names_every_route():
    routed = RoutedJevClient((Route("a", _dead_server()), Route("b", _dead_server())))

    with pytest.raises(ConnectionError, match="a.*b"):
        routed.ask({}, {})


def test_routed_client_without_routes_is_refused():
    with pytest.raises(ValueError, match="at least one route"):
        RoutedJevClient(())


def test_a_route_client_hits_its_own_endpoint_not_the_default():
    exchanges: list[tuple[bytes, bytes]] = []
    client = _server(exchanges, model="finetuned-4b")

    answer = client.ask({"case": "x"}, {"adds_one": {"type": "noul", "instructions": "y?"}})

    assert answer.model == "finetuned-4b"
    assert len(exchanges) == 1


def test_direct_system_one_journal_keeps_each_retry_at_the_sdk_boundary(tmp_path):
    _requires_typesafe()
    exchanges: list[tuple[bytes, int, bytes]] = []
    server = _retry_server(exchanges)
    client = SystemOneClient(
        model="test",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    journal_path = tmp_path / "route-retries.jsonl"
    judge = Judge(client, journal=JsonlJournal(journal_path, keep_request_text=True))

    try:
        answer = judge.ask(
            {"marker": "route"},
            {"adds_one": {"type": "noul", "instructions": "Does it add one?"}},
            thresholds=Thresholds(),
        )
    finally:
        client.close()
        server.shutdown()
        server.server_close()

    assert answer.model == "route-served-model"
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    request = next(record for record in records if record["kind"] == "request")
    attempts = [record for record in records if record["kind"] == "http_attempt"]
    assert [record["attempt_no"] for record in attempts] == [1, 2]
    assert [record["status"] for record in attempts] == [503, 200]
    assert all(record["request_id"] == request["request_id"] for record in attempts)
    assert [
        json.loads(base64.b64decode(record["sent_body_base64"]))["state"]["marker"] for record in attempts
    ] == ["route", "route"]
    assert [record["duration_ms"] >= 0 for record in attempts] == [True, True]
    assert len(exchanges) == 2


def test_direct_system_one_journal_keeps_attempts_before_terminal_sdk_failure(tmp_path):
    _requires_typesafe()
    from typesafe_sdk import TypeSafeInternalServerError

    exchanges: list[tuple[bytes, int, bytes]] = []
    server = _retry_server(exchanges, always_fail=True)
    client = SystemOneClient(
        model="test",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    journal_path = tmp_path / "route-terminal.jsonl"
    judge = Judge(client, journal=JsonlJournal(journal_path, keep_request_text=True))

    try:
        with pytest.raises(TypeSafeInternalServerError):
            judge.ask(
                {"marker": "route-terminal"},
                {"adds_one": {"type": "noul", "instructions": "Does it add one?"}},
                thresholds=Thresholds(),
            )
    finally:
        client.close()
        server.shutdown()
        server.server_close()

    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    request = next(record for record in records if record["kind"] == "request")
    attempts = [record for record in records if record["kind"] == "http_attempt"]
    failure = next(record for record in records if record["kind"] == "failure")
    assert [record["attempt_no"] for record in attempts] == [1, 2, 3]
    assert all(record["status"] == 503 for record in attempts)
    assert all(record["request_id"] == request["request_id"] for record in attempts)
    assert failure["request_id"] == request["request_id"]
    assert failure["error_type"] == "TypeSafeInternalServerError"


def test_routed_journal_keeps_failed_primary_and_successful_backup_under_one_request(tmp_path):
    _requires_typesafe()
    primary_exchanges: list[tuple[bytes, int, bytes]] = []
    backup_exchanges: list[tuple[bytes, int, bytes]] = []
    primary_server = _retry_server(primary_exchanges, always_fail=True)
    backup_server = _retry_server(backup_exchanges)
    primary = SystemOneClient(
        model="primary-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{primary_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    backup = SystemOneClient(
        model="backup-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{backup_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    routed = RoutedJevClient((Route("primary", primary), Route("backup", backup)))
    journal_path = tmp_path / "routed-retries.jsonl"

    try:
        answer = Judge(routed, journal=JsonlJournal(journal_path, keep_request_text=True)).ask(
            {"marker": "routed"},
            {"adds_one": {"type": "noul", "instructions": "Does it add one?"}},
            thresholds=Thresholds(),
        )
    finally:
        primary.close()
        backup.close()
        primary_server.shutdown()
        backup_server.shutdown()
        primary_server.server_close()
        backup_server.server_close()

    assert answer.model == "route-served-model"
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    request = next(record for record in records if record["kind"] == "request")
    attempts = [record for record in records if record["kind"] == "http_attempt"]
    assert [record["route"] for record in attempts] == ["primary"] * 3 + ["backup"] * 2
    assert [record["attempt_no"] for record in attempts] == [1, 2, 3, 4, 5]
    assert [record["status"] for record in attempts] == [503, 503, 503, 503, 200]
    assert all(record["request_id"] == request["request_id"] for record in attempts)
    assert [base64.b64decode(record["sent_body_base64"]) for record in attempts] == [
        sent for sent, _, _ in [*primary_exchanges, *backup_exchanges]
    ]


def test_routed_journal_records_all_routes_before_terminal_failure(tmp_path):
    _requires_typesafe()

    first_exchanges: list[tuple[bytes, int, bytes]] = []
    second_exchanges: list[tuple[bytes, int, bytes]] = []
    first_server = _retry_server(first_exchanges, always_fail=True)
    second_server = _retry_server(second_exchanges, always_fail=True)
    first = SystemOneClient(
        model="first-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{first_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    second = SystemOneClient(
        model="second-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{second_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    routed = RoutedJevClient((Route("first", first), Route("second", second)))
    journal_path = tmp_path / "routed-terminal.jsonl"

    try:
        with pytest.raises(ConnectionError, match="first.*second"):
            Judge(routed, journal=JsonlJournal(journal_path, keep_request_text=True)).ask(
                {"marker": "terminal-routed"},
                {"adds_one": {"type": "noul", "instructions": "Does it add one?"}},
                thresholds=Thresholds(),
            )
    finally:
        first.close()
        second.close()
        first_server.shutdown()
        second_server.shutdown()
        first_server.server_close()
        second_server.server_close()

    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    request = next(record for record in records if record["kind"] == "request")
    attempts = [record for record in records if record["kind"] == "http_attempt"]
    failure = next(record for record in records if record["kind"] == "failure")
    assert [record["route"] for record in attempts] == ["first"] * 3 + ["second"] * 3
    assert [record["attempt_no"] for record in attempts] == [1, 2, 3, 4, 5, 6]
    assert all(record["status"] == 503 for record in attempts)
    assert all(record["request_id"] == request["request_id"] for record in attempts)
    assert failure["request_id"] == request["request_id"]
    assert failure["error_type"] == "ConnectionError"


def test_routed_parse_failure_keeps_fallback_behavior_and_both_attempts(tmp_path):
    _requires_typesafe()
    primary_exchanges: list[tuple[bytes, int, bytes]] = []
    backup_exchanges: list[tuple[bytes, int, bytes]] = []
    primary_server = _retry_server(primary_exchanges, malformed_first=True)
    backup_server = _retry_server(backup_exchanges)
    primary = SystemOneClient(
        model="primary-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{primary_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    backup = SystemOneClient(
        model="backup-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{backup_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    journal_path = tmp_path / "routed-parse-fallback.jsonl"

    try:
        answer = Judge(
            RoutedJevClient((Route("primary", primary), Route("backup", backup))),
            journal=JsonlJournal(journal_path, keep_request_text=True),
        ).ask(
            {"marker": "parse-fallback"},
            {"adds_one": {"type": "noul", "instructions": "Does it add one?"}},
            thresholds=Thresholds(),
        )
    finally:
        primary.close()
        backup.close()
        primary_server.shutdown()
        backup_server.shutdown()
        primary_server.server_close()
        backup_server.server_close()

    assert answer.model == "route-served-model"
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    attempts = [record for record in records if record["kind"] == "http_attempt"]
    assert [record["route"] for record in attempts] == ["primary", "backup", "backup"]
    assert [record["status"] for record in attempts] == [200, 503, 200]


def test_route_attempt_journal_failure_propagates_without_using_backup(tmp_path):
    _requires_typesafe()

    class RejectFirstAttemptWrite(JsonlJournal):
        def __init__(self, path):
            super().__init__(path, keep_request_text=True)
            self.rejected = False

        def record_attempt(self, request_id, attempt):
            if not self.rejected:
                self.rejected = True
                raise OSError("journal attempt write failed")
            super().record_attempt(request_id, attempt)

    primary_exchanges: list[tuple[bytes, int, bytes]] = []
    backup_exchanges: list[tuple[bytes, int, bytes]] = []
    primary_server = _retry_server(primary_exchanges, always_fail=True)
    backup_server = _retry_server(backup_exchanges)
    primary = SystemOneClient(
        model="primary-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{primary_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    backup = SystemOneClient(
        model="backup-model",
        api_key="local-test-key",
        base_url=f"http://127.0.0.1:{backup_server.server_port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    journal = RejectFirstAttemptWrite(tmp_path / "failed-attempt-write.jsonl")
    judge = Judge(RoutedJevClient((Route("primary", primary), Route("backup", backup))), journal=journal)

    try:
        with pytest.raises(OSError, match="journal attempt write failed"):
            judge.ask(
                {"marker": "journal-failure"},
                {"adds_one": {"type": "noul", "instructions": "Does it add one?"}},
                thresholds=Thresholds(),
            )
    finally:
        primary.close()
        backup.close()
        primary_server.shutdown()
        backup_server.shutdown()
        primary_server.server_close()
        backup_server.server_close()

    assert journal.rejected
    assert len(primary_exchanges) == 1
    assert backup_exchanges == []


# --- local System-One endpoints -------------------------------------------------------------


def _requires_typesafe() -> None:
    pytest.importorskip("httpx2")
    pytest.importorskip("typesafe_sdk")


def _server(exchanges: list[tuple[bytes, bytes]], model: str = "jev-1.13.0") -> SystemOneClient:
    _requires_typesafe()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            sent = self.rfile.read(int(self.headers["content-length"]))
            answers = {id_: {"type": "noul", "noul": 0.9} for id_ in json.loads(sent)["questions"]}
            served = json.dumps(
                {
                    "model": model,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                    "answers": answers,
                }
            ).encode()
            exchanges.append((sent, served))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(served)))
            self.end_headers()
            self.wfile.write(served)

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading_daemon = __import__("threading").Thread(target=httpd.serve_forever, daemon=True)
    threading_daemon.start()
    return SystemOneClient(
        model="test",
        api_key="test-key",
        base_url=f"http://127.0.0.1:{port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )


def _dead_server() -> SystemOneClient:
    _requires_typesafe()
    client = SystemOneClient(
        model="test",
        api_key="test-key",
        base_url="http://127.0.0.1:1",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    return client


def _retry_server(
    exchanges: list[tuple[bytes, int, bytes]], *, always_fail: bool = False, malformed_first: bool = False
):
    counts = 0
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal counts
            sent = self.rfile.read(int(self.headers["content-length"]))
            with lock:
                counts += 1
                attempt = counts
            if malformed_first and attempt == 1:
                status = 200
                body = b'{"model":"route-served-model"}'
            elif attempt == 1 or always_fail:
                status = 503
                body = b'{"detail":{"error_type":"temporary_unavailable"}}'
            else:
                status = 200
                body = json.dumps(
                    {
                        "model": "route-served-model",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                        "answers": {
                            key: {"type": "noul", "noul": 0.9} for key in json.loads(sent)["questions"]
                        },
                    }
                ).encode()
            exchanges.append((sent, status, body))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_routed_budget_refusal_reaches_the_judge_and_splits_without_failover():
    """The real route transport preserves a size refusal for the batching owner to split."""
    _requires_typesafe()
    refused: list[bytes] = []
    accepted: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            sent = self.rfile.read(int(self.headers["content-length"]))
            if len(sent) > 40_000:
                refused.append(sent)
                body = b'{"detail":{"error_type":"max_tokens_exceeded"}}'
                self.send_response(400)
            else:
                accepted.append(sent)
                body = json.dumps(
                    {
                        "model": "drex-latest",
                        "usage": {"input_tokens": 12, "output_tokens": 1},
                        "answers": {
                            key: {"type": "noul", "noul": 0.9} for key in json.loads(sent)["questions"]
                        },
                    }
                ).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    __import__("threading").Thread(target=httpd.serve_forever, daemon=True).start()
    client = SystemOneClient(
        model="drex-latest",
        api_key="test-key",
        base_url=f"http://127.0.0.1:{port}",
        input_limits=JEV_INPUT_LIMITS,
        max_concurrency=JEV_CONCURRENCY,
    )
    backup_exchanges: list[tuple[bytes, bytes]] = []
    routed = RoutedJevClient((Route("drex", client), Route("backup", _server(backup_exchanges))))
    check = Check(
        "has_code",
        "Does `{item}.code` contain code?",
        yes=Criterion("Code is present."),
        no=Criterion("Code is absent."),
    )
    results = Judge(routed).check_every(
        [check], [{"code": "x" * 24_000}, {"code": "y" * 24_000}], list_name="items"
    )

    assert len(results["has_code"]) == 2
    assert len(refused) == 1
    assert len(accepted) == 2
    assert backup_exchanges == []
