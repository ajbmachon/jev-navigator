from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jev_navigator.directives.find_code import Outcome, SearchBudget, find_code, find_code_async
from jev_navigator.directives.places import MOVES, function_place, range_place
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.thresholds import NoulVerdict, Thresholds

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)

STATE = {"slice": {"file": "counter.py", "code": "def add_one(x):\n    return x + 1"}}
QUESTIONS = {"adds_one": {"type": "noul", "instructions": "Does `slice.code` add one?"}}


def _jev_server(
    exchanges: list[tuple[bytes, bytes]],
    *,
    input_limit: int | None = None,
    priority_failure_status: int | None = None,
    refusal_body: bytes | None = None,
) -> ThreadingHTTPServer:
    """A local Jev endpoint recording the exact bytes of every request it answers, and its answer.

    Those pairs are ground truth for what crossed the wire, so a journal or store entry is checked
    against them rather than against another copy of what the library thinks it sent.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            sent = self.rfile.read(int(self.headers["content-length"]))
            questions = json.loads(sent)["questions"]
            status = 200
            if refusal_body is not None:
                status, served = 400, refusal_body
            elif priority_failure_status is not None and all(
                q["type"] == "choice" for q in questions.values()
            ):
                status = priority_failure_status
                detail = "max_tokens_exceeded" if status == 400 else "invalid_api_key"
                served = json.dumps({"detail": {"error_type": detail}}).encode()
            elif input_limit is not None and len(sent) > input_limit:
                served = b'{"detail":{"error_type":"max_tokens_exceeded"}}'
                status = 400
            else:
                answers = {}
                for id_, question in questions.items():
                    if question["type"] == "choice":
                        options = question["criteria"]
                        picked = next(iter(options))
                        answers[id_] = {
                            "type": "choice",
                            "choice": picked,
                            "probabilities": {key: float(key == picked) for key in options},
                            "confidence": 1.0,
                        }
                    else:
                        answers[id_] = {"type": "noul", "noul": 0.9}
                served = json.dumps(
                    {
                        "model": "jev-1.13.0",
                        "usage": {"input_tokens": 12, "output_tokens": 1},
                        "answers": answers,
                    }
                ).encode()
            exchanges.append((sent, served))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(served)))
            self.end_headers()
            self.wfile.write(served)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, name="jev-local-server", daemon=True).start()
    return server


def _stop(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


def _seed_index(root: Path, neighbours: int) -> tuple[CodeIndex, list[str]]:
    blocks = ["def entry():\n    return 0\n"]
    for number in range(neighbours):
        lines = [f"def candidate_{number}():"]
        lines.extend(f"    field_{line} = '{'x' * 130}'" for line in range(6))
        lines.append("    return field_0")
        blocks.append("\n".join(lines))
    (root / "workflow.py").write_text("\n\n".join(blocks))
    return CodeIndex.from_directory(root), blocks[1:]


@pytest.mark.parametrize("async_search", [False, True])
@pytest.mark.parametrize("neighbours,input_limit", [(159, 40_000), (2, 3_000)])
def test_seed_search_batches_every_neighbour_at_the_real_input_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, async_search: bool, neighbours: int, input_limit: int
) -> None:
    """The original Find All seed sent 159 previews in one 236KB request and died on HTTP 400."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    index, previews = _seed_index(tmp_path, neighbours)
    start = function_place(index, index.find_definition("entry")[0])
    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges, input_limit=input_limit)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    judge = Judge(client, journal=JsonlJournal(tmp_path / "journal.jsonl", keep_request_text=True))
    search = find_code_async if async_search else find_code

    try:
        pending = search(
            index,
            judge,
            "the function returning field_0",
            [start],
            budget=SearchBudget(max_steps=1, beam_width=1),
            moves={"same_file": MOVES["same_file"]},
        )
        result = asyncio.run(pending) if async_search else pending
    finally:
        client.close()
        _stop(server)

    opened = next(step for step in result.history.steps if step.operation == "open")
    assessed = opened.judgments["could_contain"]
    assert len(assessed) == neighbours
    assert len({candidate["place"] for candidate in assessed}) == neighbours
    assert len(result.not_inspected) == neighbours
    accepted = [
        (json.loads(sent), json.loads(body)) for sent, body in exchanges if "answers" in json.loads(body)
    ]
    assert sum(len(body["answers"]) for _, body in accepted) >= neighbours + 1
    assert all(len(sent) <= input_limit for sent, body in exchanges if "answers" in json.loads(body))
    delivered = [
        candidate["code"]
        for request, _ in accepted
        if any(key.startswith("could_contain_target@") for key in request["questions"])
        for candidate in request["state"]["candidates"]
    ]
    assert sorted(delivered) == sorted(previews)
    assert judge.calls == result.calls == len(exchanges)
    if neighbours > 2:
        assert opened.judgments["open_first"]["used"] is False
        assert "request-size packing estimate" in opened.judgments["open_first"]["unavailable"]
    else:
        assert opened.judgments["open_first"]["used"] is True


def test_split_seed_resumes_completed_neighbours_without_another_paid_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    index, previews = _seed_index(tmp_path, 159)
    start = function_place(index, index.find_definition("entry")[0])
    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges, input_limit=40_000)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    store_path = tmp_path / "answers.jsonl"
    first_judge = Judge(client, store=JsonlAnswerStore(store_path))
    try:
        first = find_code(
            index,
            first_judge,
            "the function returning field_0",
            [start],
            budget=SearchBudget(max_calls=3, max_steps=1, beam_width=1),
            moves={"same_file": MOVES["same_file"]},
        )
        assert first.outcome == Outcome.BUDGET
        assert first.calls == 3
        assert JsonlAnswerStore(store_path).records()
        resumed_judge = Judge(
            client,
            store=JsonlAnswerStore(store_path),
            served_model=first_judge.served_model,
        )
        resumed = find_code(
            index,
            resumed_judge,
            "the function returning field_0",
            [],
            budget=SearchBudget(max_steps=1, beam_width=1),
            moves={"same_file": MOVES["same_file"]},
            resume=first,
        )
    finally:
        client.close()
        _stop(server)

    delivered = [
        candidate["code"]
        for sent, body in exchanges
        if "answers" in json.loads(body)
        and any(key.startswith("could_contain_target@") for key in json.loads(sent)["questions"])
        for candidate in json.loads(sent)["state"]["candidates"]
    ]
    assert sorted(delivered) == sorted(previews)
    assert len(resumed.not_inspected) == len(previews)
    assert len(resumed.starts) == 1
    assert first.calls + resumed_judge.calls == len(exchanges)
    stored_text = store_path.read_text()
    assert not any(json.dumps(preview) in stored_text for preview in previews)


@pytest.mark.parametrize("status", [400, 401])
def test_optional_priority_keeps_size_failure_but_propagates_auth_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import TypeSafeAuthenticationError

    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    index, _ = _seed_index(tmp_path, 2)
    start = function_place(index, index.find_definition("entry")[0])
    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges, input_limit=3_000, priority_failure_status=status)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    journal_path = tmp_path / "journal.jsonl"
    judge = Judge(client, journal=JsonlJournal(journal_path))
    try:

        def search():
            return find_code(
                index,
                judge,
                "the function returning field_0",
                [start],
                budget=SearchBudget(max_steps=1, beam_width=1),
                moves={"same_file": MOVES["same_file"]},
            )

        if status == 401:
            with pytest.raises(TypeSafeAuthenticationError, match="invalid_api_key"):
                search()
        else:
            result = search()
            opened = next(step for step in result.history.steps if step.operation == "open")
            assert len(opened.judgments["could_contain"]) == 2
            assert opened.judgments["open_first"]["used"] is False
            assert "max_tokens_exceeded" in opened.judgments["open_first"]["unavailable"]
    finally:
        client.close()
        _stop(server)

    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    cause = "max_tokens_exceeded" if status == 400 else "invalid_api_key"
    assert any(cause in record.get("error", "") for record in records if record["kind"] == "failure")


@pytest.mark.parametrize("async_checks", [False, True])
@pytest.mark.parametrize(
    "error_type,message",
    [
        ("invalid_question", "unknown question id max_tokens_exceeded"),
        ("max_tokens_exceeded", None),
        ("max_tokens_exceeded", "The model's input is too long."),
    ],
)
def test_structured_error_type_controls_retry_at_the_sdk_boundary(
    monkeypatch: pytest.MonkeyPatch, async_checks: bool, error_type: str, message: str | None
) -> None:
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import TypeSafeBadRequestError

    from jev_navigator.adapters.typesafe import TypeSafeJevClient
    from jev_navigator.judgments.client import InputBudgetExceededError

    detail = {"error_type": error_type}
    if message is not None:
        detail["message"] = message
    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges, refusal_body=json.dumps({"detail": detail}).encode())
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    judge = Judge(client)
    expected = InputBudgetExceededError if error_type == "max_tokens_exceeded" else TypeSafeBadRequestError
    try:
        with pytest.raises(expected):
            items = [{"code": "def one(): return 1"}, {"code": "def two(): return 2"}]
            shared = {"doc": {"sentence": "returns a number"}}
            if async_checks:
                asyncio.run(judge.check_every_async([DESCRIBES], items, shared))
            else:
                judge.check_every([DESCRIBES], items, shared)
    finally:
        client.close()
        _stop(server)

    assert judge.calls == len(exchanges)
    assert len(exchanges) == (2 if error_type == "max_tokens_exceeded" else 1)


def test_input_batches_keep_values_masked_across_request_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    planted = "order-hook-4f7a1c"
    items = [
        {"code": f'WEBHOOK_TOKEN = "{planted}"'},
        {"code": f'post("{planted}", order)'},
    ]
    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    try:
        answers = Judge(client).check_every(
            [DESCRIBES], items, {"doc": {"sentence": "posts an order"}}, batch_budget=1
        )
    finally:
        client.close()
        _stop(server)

    assert len(answers["describes"]) == 2
    assert len(exchanges) == 2
    assert all(planted.encode() not in sent for sent, _ in exchanges)


def test_cancel_aborts_an_active_official_sdk_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production adapter owns cancellation through the SDK's async HTTP boundary."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    entered = threading.Event()
    release = threading.Event()
    requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal requests
            self.rfile.read(int(self.headers["content-length"]))
            requests += 1
            entered.set()
            release.wait()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    finished = threading.Event()
    failure: list[BaseException] = []

    def ask() -> None:
        try:
            client.ask(
                {"code": "return wanted"},
                {"match": {"type": "noul", "instructions": "Does code return wanted?"}},
            )
        except BaseException as error:
            failure.append(error)
        finally:
            finished.set()

    request_thread = threading.Thread(target=ask)
    request_thread.start()
    try:
        assert entered.wait(2), "the SDK request did not reach the local HTTP server"

        client.cancel()

        assert finished.wait(2), "cancellation left the SDK request blocked"
        assert isinstance(failure[0], concurrent.futures.CancelledError)
        with pytest.raises(concurrent.futures.CancelledError):
            client.ask(
                {"code": "a late beam request"},
                {"match": {"type": "noul", "instructions": "Does code match?"}},
            )
        assert requests == 1
    finally:
        release.set()
        request_thread.join()
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_sigint_returns_the_active_http_place_as_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed sync search and official SDK share one prompt cancellation boundary."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    all_entered = threading.Event()
    release = threading.Event()
    entered = 0
    entered_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal entered
            self.rfile.read(int(self.headers["content-length"]))
            with entered_lock:
                entered += 1
                if entered == 2:
                    all_entered.set()
            release.wait()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    (tmp_path / "policy.py").write_text(
        "def first(item):\n    return item\n\ndef second(item):\n    return item\n"
    )
    index = CodeIndex(tmp_path, ["policy.py"])
    places = [
        range_place(index, "policy.py", 1, 2, "candidate"),
        range_place(index, "policy.py", 4, 5, "candidate"),
    ]

    def interrupt_when_sent() -> None:
        all_entered.wait()
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_when_sent)
    interrupter.start()
    try:
        result = find_code(
            index,
            Judge(client),
            "the policy",
            [],
            budget=SearchBudget(beam_width=2),
            moves={},
            initial_candidates=[(place, 1.0) for place in places],
        )

        assert result.outcome == Outcome.CANCELLED
        assert {entry.place_key for entry in result.not_inspected} == {place.key for place in places}
        assert {entry.reason for entry in result.not_inspected} == {"cancelled"}
    finally:
        interrupter.join()
        release.set()
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_the_production_transport_journals_the_exact_bytes_it_sent_and_received(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`exact` in the journal means "these are the wire bytes", so prove it over a real socket.

    The journal's `exact` flag and the answer store's `sent_exact` flag describe different things: the
    first is the response as received, the second is a kept copy of the request. This pins the first.
    """
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    journal_path = tmp_path / "journal.jsonl"
    judge = Judge(client, journal=JsonlJournal(journal_path, keep_request_text=True))

    try:
        answer = judge.ask(STATE, QUESTIONS, thresholds=Thresholds())
    finally:
        client.close()
        _stop(server)

    # Assert
    assert answer.noul("adds_one").probability == 0.9
    sent, served = exchanges[0]
    journaled = json.loads(journal_path.read_text().splitlines()[1])
    assert (journaled["status"], journaled["content_type"], journaled["exact"]) == (
        200,
        "application/json",
        True,
    )
    assert base64.b64decode(journaled["body_base64"]) == served
    captured = base64.b64decode(journaled["sent_body_base64"])
    assert captured == sent
    assert b'"model"' in captured  # the wire body names the model; the library's handover does not


def test_the_answer_store_keeps_no_request_bytes_by_default_even_after_a_real_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `sent_exact: false` answer record is the privacy default, not evidence of a lost capture."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    silent_journal = JsonlJournal(tmp_path / "silent-journal.jsonl", keep_request_text=True)
    keeping_journal = JsonlJournal(tmp_path / "keeping-journal.jsonl", keep_request_text=True)
    silent = JsonlAnswerStore(tmp_path / "silent-answers.jsonl")
    keeping = JsonlAnswerStore(tmp_path / "keeping-answers.jsonl", keep_requests=True)
    client = TypeSafeJevClient()

    try:
        Judge(client, store=silent, journal=silent_journal).ask(STATE, QUESTIONS, thresholds=Thresholds())
        Judge(client, store=keeping, journal=keeping_journal).ask(STATE, QUESTIONS, thresholds=Thresholds())
    finally:
        client.close()
        _stop(server)

    # Assert: the journal kept the wire bytes of the request and the answer of the response either
    # way, so the answer store's own flag below cannot be read as "the transport captured nothing".
    for journal in (silent_journal, keeping_journal):
        line = json.loads(journal.path.read_text().splitlines()[1])
        assert (line["exact"], line["status"]) == (True, 200)
        assert base64.b64decode(line["sent_body_base64"]) == exchanges[0][0]

    dropped = silent.records()[0]
    assert (dropped.sent_exact, dropped.sent_body_base64, dropped.request) == (False, None, None)
    with pytest.raises(ValueError, match="keep_requests"):
        dropped.sent_request()

    kept = keeping.records()[0]
    assert kept.sent_exact is True
    assert base64.b64decode(kept.sent_body_base64) == exchanges[1][0]
    assert kept.sent_request() == (STATE, QUESTIONS)


def test_a_jev_route_asks_the_sdk_with_the_routes_own_timeout_and_retries() -> None:
    pytest.importorskip("typesafe_sdk")
    import httpx2
    from typesafe_sdk import TypeSafeRateLimitError

    from jev_navigator.adapters.routes import routes_from_env

    timeouts: list[dict] = []

    def busy(request: httpx2.Request) -> httpx2.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx2.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "busy"}})

    (route,) = routes_from_env(
        {
            "SYSTEM_ONE_ROUTES": "jev",
            "TYPESAFE_API_KEY": "local-test-key",
            "SYSTEM_ONE_JEV_TIMEOUT": "2.5",
            "SYSTEM_ONE_JEV_RETRIES": "0",
        },
        transport=httpx2.MockTransport(busy),
    )

    try:
        with pytest.raises(TypeSafeRateLimitError):
            route.client.ask(STATE, QUESTIONS)
    finally:
        route.client.close()
    assert [timeout["read"] for timeout in timeouts] == [2.5]


def test_a_max_tokens_exceeded_response_is_typed_and_the_batch_splits_at_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved trace run's request 5 died on an untyped 400 max_tokens_exceeded. The transport
    owns the provider contract: it translates the refusal, and the batching owner splits the batch
    so the same questions travel in requests the provider accepts. Sanitized fixture, real socket."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    refusals: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            sent = self.rfile.read(int(self.headers["content-length"]))
            if len(sent) > 40_000:
                refusals.append(sent)
                body = b'{"detail":{"error_type":"max_tokens_exceeded"}}'
                self.send_response(400)
            else:
                answers = {id_: {"type": "noul", "noul": 0.9} for id_ in json.loads(sent)["questions"]}
                body = json.dumps(
                    {
                        "model": "jev-1.13.0",
                        "usage": {"input_tokens": 12, "output_tokens": 1},
                        "answers": answers,
                    }
                ).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, name="jev-local-server", daemon=True).start()
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    journal_path = tmp_path / "journal.jsonl"
    judge = Judge(client, journal=JsonlJournal(journal_path, keep_request_text=True))
    items = [
        {"file": f"part{index}.py", "lines": [1, 2], "code": "x = " + "y" * 24_000} for index in range(2)
    ]

    try:
        results = judge.check_every([DESCRIBES], items, {}, list_name="parts")
    finally:
        client.close()
        _stop(server)

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES, NoulVerdict.YES]
    assert len(refusals) == 1, "the provider's own refusal stays the evidence of the oversized batch"
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    failures = [record for record in records if record["kind"] == "failure"]
    responses = [record for record in records if record["kind"] == "response"]
    assert len(failures) == 1 and "max_tokens_exceeded" in failures[0]["error"]
    assert len(responses) == 2 and all(record["status"] == 200 for record in responses)
    for record in responses:
        assert len(base64.b64decode(record["sent_body_base64"])) <= 40_000
