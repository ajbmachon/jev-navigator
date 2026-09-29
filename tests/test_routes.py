"""The route table: named decision-model routes with automatic fallback, and the generic
System-One client every route runs."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jev_navigator.adapters.routes import (
    Route,
    RoutedJevClient,
    SystemOneClient,
    routes_from_env,
)


def test_no_routes_variable_resolves_to_no_routes():
    assert routes_from_env({}) == ()


def test_unknown_route_needs_endpoint_and_model():
    with pytest.raises(ValueError, match="route 'decider' is incomplete"):
        routes_from_env({"SYSTEM_ONE_ROUTES": "decider"})


def test_known_route_shorthand_needs_only_the_flag():
    _requires_typesafe()
    routes = routes_from_env(
        {"SYSTEM_ONE_ROUTES": "drex", "SYSTEM_ONE_DREX": "1", "TYPESAFE_API_KEY": "test-key"}
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
            "TYPESAFE_API_KEY": "test-key",
        }
    )
    assert routes[0].client.model == "drex-v1.1"


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
    return SystemOneClient(model="test", api_key="test-key", base_url=f"http://127.0.0.1:{port}")


def _dead_server() -> SystemOneClient:
    _requires_typesafe()
    client = SystemOneClient(model="test", api_key="test-key", base_url="http://127.0.0.1:1")
    return client
