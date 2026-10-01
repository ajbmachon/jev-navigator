"""The shared adapter base: exact-byte capture, one retry policy, and cancellation, which every
adapter built on `SystemOneClient` inherits."""

from __future__ import annotations

import json
import threading
from concurrent.futures import CancelledError
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jev_navigator.adapters.system_one import AdapterError, HttpResponse, SystemOneClient
from jev_navigator.judgments.client import InputBudgetExceededError

QUESTIONS = {"adds_one": {"type": "noul", "instructions": "Does `code` add one?"}}
ANSWER = b'{"model":"decider-1","answers":{"adds_one":{"type":"noul","noul":0.9}}}'


def test_the_response_and_the_request_are_kept_as_the_exact_wire_bytes():
    # Arrange
    received: list[bytes] = []
    served = b'{ "model": "decider-1",\n  "answers": {"adds_one": {"type": "noul", "noul": 0.9}} }'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            received.append(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(served)))
            self.end_headers()
            self.wfile.write(served)

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    with _serving(Handler) as endpoint:
        client = SystemOneClient("decider-1", api_key="local-key", endpoint=endpoint)

        # Act
        raw = client.send({"code": "x + 1"}, QUESTIONS)

    # Assert
    assert (raw.body, raw.status, raw.content_type) == (served, 200, "application/json; charset=utf-8")
    assert raw.sent_body == received[0]
    assert json.loads(raw.sent_body)["model"] == "decider-1"
    assert client.parse(raw).noul("adds_one").probability == 0.9


@pytest.mark.parametrize("status", [429, 529])
def test_a_busy_service_is_asked_again_after_the_wait_it_names(status):
    replies = [HttpResponse(status, {"retry-after-ms": "1"}, b"{}"), HttpResponse(200, {}, ANSWER)]
    sent = []

    def transport(request):
        sent.append(request)
        return replies[len(sent) - 1]

    client = SystemOneClient(
        "decider-1", api_key="local-key", endpoint="https://decider.example", transport=transport
    )

    answer = client.ask({}, QUESTIONS)

    assert answer.noul("adds_one").probability == 0.9
    assert len(sent) == 2


@pytest.mark.parametrize("status", [401, 422])
def test_a_refused_request_fails_at_once_with_the_services_message(status):
    sent = []
    refusal = b'{"error":{"type":"invalid_request_error","message":"questions: must be a string"}}'

    def transport(request):
        sent.append(request)
        return HttpResponse(status, {"content-type": "application/json"}, refusal)

    client = SystemOneClient(
        "decider-1", api_key="local-key", endpoint="https://decider.example", transport=transport
    )

    with pytest.raises(AdapterError, match=f"HTTP {status}: questions: must be a string") as refused:
        client.ask({}, QUESTIONS)

    assert refused.value.status == status
    assert "local-key" not in str(refused.value)
    assert len(sent) == 1


@pytest.mark.parametrize(
    ("refusal", "raised"),
    [
        (b'{"detail":{"error_type":"max_tokens_exceeded"}}', InputBudgetExceededError),
        (b'{"detail":{"error_type":"invalid_questions"}}', AdapterError),
    ],
)
def test_only_an_input_budget_refusal_is_typed_for_the_batching_owner_to_split(refusal, raised):
    # Arrange
    sent = []

    def transport(request):
        sent.append(request)
        return HttpResponse(400, {"content-type": "application/json"}, refusal)

    client = SystemOneClient(
        "decider-1", api_key="local-key", endpoint="https://decider.example", transport=transport
    )

    # Act
    with pytest.raises((InputBudgetExceededError, AdapterError)) as refused:
        client.ask({}, QUESTIONS)

    # Assert: the size refusal is sent once and typed; any other refusal keeps the body it came with
    assert type(refused.value) is raised
    assert len(sent) == 1
    if raised is AdapterError:
        assert refused.value.body == refusal


@pytest.mark.parametrize(
    ("endpoint", "problem"),
    [
        ("ftp://decider.example", "needs http:// or https://"),
        ("decider.example:8900", "needs http:// or https://"),
        ("https://", "names no host"),
        ("https://decider.example/?tenant=a", "query or fragment"),
        ("https://decider.example/#v1", "query or fragment"),
        ("https://decider.example:port", "port"),
    ],
)
def test_an_endpoint_no_request_can_reach_is_refused_when_the_client_is_built(endpoint, problem):
    # Each of these used to build, then fail on the first request: as plain HTTP with the key for
    # ftp://, deep in http.client without a scheme, or with the request path after the query.
    with pytest.raises(ValueError, match=problem):
        SystemOneClient("decider-1", api_key="local-key", endpoint=endpoint)


def test_an_endpoint_carrying_credentials_is_refused_without_repeating_them():
    with pytest.raises(ValueError, match="credentials") as refused:
        SystemOneClient("decider-1", endpoint="https://user:s3cret@decider.example")

    assert "s3cret" not in str(refused.value)


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:8900", "http://[::1]:8900", "http://gpu-box:8000", "https://decider.example/gateway/"],
)
def test_an_http_endpoint_with_a_host_is_sent_to_at_the_wire_path(endpoint):
    client = SystemOneClient("decider-1", endpoint=endpoint)

    assert client.url == endpoint.rstrip("/") + "/v1/systemone"


def test_cancel_aborts_a_request_in_flight_and_refuses_later_ones():
    # Arrange: a service that never answers.
    entered, release = threading.Event(), threading.Event()
    requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal requests
            self.rfile.read(int(self.headers["content-length"]))
            requests += 1
            entered.set()
            release.wait()

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    failure: list[BaseException] = []
    finished = threading.Event()
    with _serving(Handler) as endpoint:
        client = SystemOneClient("decider-1", api_key="local-key", endpoint=endpoint)

        def ask() -> None:
            try:
                client.ask({}, QUESTIONS)
            except BaseException as error:
                failure.append(error)
            finally:
                finished.set()

        asking = threading.Thread(target=ask)
        asking.start()
        try:
            assert entered.wait(2), "the request did not reach the local service"

            # Act
            client.cancel()

            # Assert
            assert finished.wait(2), "cancel left the request blocked"
            assert isinstance(failure[0], CancelledError)
            with pytest.raises(CancelledError):
                client.ask({}, QUESTIONS)
            assert requests == 1
        finally:
            release.set()
            asking.join()


@contextmanager
def _serving(handler):
    """A local HTTP service for the duration of a ``with`` block; yields its endpoint."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
