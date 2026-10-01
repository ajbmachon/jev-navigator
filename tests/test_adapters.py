"""The adapter contract (`adapters/system_one.py`), held against every registered adapter: a new
entry in `adapters/registry.py` runs these without being named here. `test_local.py` holds the
in-process base to the parts an HTTP exchange cannot show."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from jev_navigator.adapters.registry import ADAPTERS
from jev_navigator.adapters.routes import routes_from_env
from jev_navigator.judgments.questions import Check, Criterion, Pick, Rate

REGISTERED = pytest.mark.parametrize("adapter", ADAPTERS.values(), ids=ADAPTERS.keys())
KEYED = [adapter for adapter in ADAPTERS.values() if adapter.needs_key]
PINNED = [adapter for adapter in ADAPTERS.values() if adapter.pinned]
OVER_HTTP_UNPINNED = [
    adapter for adapter in ADAPTERS.values() if not (adapter.pinned or adapter.runs_locally)
]


@REGISTERED
def test_an_adapter_reports_the_model_it_was_built_for(adapter):
    client = _built(adapter, model="conformance-model", api_key="conformance-key")
    try:
        assert client.model == "conformance-model"
    finally:
        client.close()


@REGISTERED
def test_a_registered_adapter_runs_as_a_route_from_its_name_and_key_alone(adapter):
    environment = {"SYSTEM_ONE_ROUTES": adapter.name}
    if adapter.needs_key:
        environment[adapter.api_key_env] = "conformance-key"
    try:
        (route,) = routes_from_env(environment)
    except ImportError as missing:
        pytest.skip(f"{adapter.name} needs an optional dependency: {missing}")
    try:
        assert type(route.client) is adapter
    finally:
        route.client.close()


@pytest.mark.parametrize("adapter", KEYED, ids=[adapter.name for adapter in KEYED])
def test_an_adapter_never_borrows_another_services_key(adapter, monkeypatch):
    monkeypatch.delenv(adapter.api_key_env, raising=False)
    for other in ADAPTERS.values():
        if other.api_key_env and other.api_key_env != adapter.api_key_env:
            monkeypatch.setenv(other.api_key_env, f"a key issued for {other.name}")

    with pytest.raises(Exception, match=adapter.api_key_env):
        _built(adapter)


@pytest.mark.parametrize("adapter", PINNED, ids=[adapter.name for adapter in PINNED])
def test_a_pinned_adapter_refuses_another_endpoint(adapter):
    with pytest.raises(ValueError, match="pinned"):
        _built(adapter, api_key="conformance-key", endpoint="https://proxy.example")


@pytest.mark.parametrize("adapter", OVER_HTTP_UNPINNED, ids=[adapter.name for adapter in OVER_HTTP_UNPINNED])
def test_an_unpinned_adapter_asks_its_configured_endpoint_and_parses_every_answer_type(adapter):
    # Arrange: a local System-One endpoint answering a check, a pick and a rate.
    check = Check("adds_one", "Does `code` add one?", Criterion("It adds one."), Criterion("It does not."))
    pick = Pick("kind", "What does `code` do?")
    rate = Rate("fit", "How well does `code` add one?", ("not at all", "exactly"))
    questions = {
        check.question_id: check.to_question(),
        pick.question_id: pick.to_question({"add": "adds", "sub": "subtracts"}),
        rate.question_id: rate.to_question(),
    }
    received: list[tuple[str, str]] = []
    served = {
        "model": "conformance-served",
        "usage": {"input_tokens": 7, "output_tokens": 3},
        "answers": {
            check.question_id: {"type": "noul", "noul": 0.9},
            pick.question_id: {
                "type": "choice",
                "choice": "add",
                "probabilities": {"add": 0.8, "sub": 0.2},
                "confidence": 0.6,
            },
            rate.question_id: {
                "type": "score",
                "score": 0.9,
                "legend": {"0": "not at all", "1": "exactly"},
                "probabilities": {"0": 0.1, "1": 0.9},
                "confidence": 0.8,
            },
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.rfile.read(int(self.headers["content-length"]))
            received.append((self.path, self.headers["authorization"]))
            body = json.dumps(served).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        client = _built(
            adapter,
            model="conformance-model",
            api_key="conformance-key",
            endpoint=f"http://127.0.0.1:{server.server_port}/gateway",
        )

        # Act
        try:
            answer = client.ask({"code": "x + 1"}, questions)
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    # Assert
    assert received == [("/gateway/v1/systemone", "Bearer conformance-key")]
    assert answer.model == "conformance-served"
    assert answer.noul(check.question_id).probability == 0.9
    assert answer.choice(pick.question_id).choice == "add"
    assert answer.score(rate.question_id).score == 0.9


def _built(adapter, **settings):
    """The adapter built from settings, or a skip when it needs an extra that is not installed."""
    try:
        return adapter(**settings)
    except ImportError as missing:
        pytest.skip(f"{adapter.name} needs an optional dependency: {missing}")
