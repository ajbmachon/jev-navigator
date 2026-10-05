from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from jev_navigator.judgments.answers import response_from_raw
from jev_navigator.judgments.client import UnansweredQuestionError
from jev_navigator.judgments.journal import (
    JournalRequest,
    JsonlJournal,
    RawAttempt,
    RawResponse,
    keeps_request_text,
)
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import AnswerRecord, JsonlAnswerStore
from jev_navigator.judgments.thresholds import Thresholds
from jev_navigator.testing import ScriptedJevClient

STATE = {"slice": {"code": "def increment(x): return x + 1"}}
QUESTIONS = {"adds_one": {"type": "noul", "instructions": "Does `slice.code` add one?"}}


class RecordingJournal:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def record_request(self, request: JournalRequest) -> str:
        self.events.append(("request", request.request_sha256, dict(request.state)))
        return f"attempt-{len(self.events)}"

    def record_response(self, request_id: str, response: RawResponse) -> None:
        self.events.append(("response", request_id, response))

    def record_attempt(self, request_id: str, attempt: RawAttempt) -> None:
        self.events.append(("attempt", request_id, attempt))

    def record_failure(
        self, request_id: str, error: BaseException, response: RawResponse | None = None
    ) -> None:
        self.events.append(("failure", request_id, error, response))


MALFORMED_BODY = b'{"model": "jev-1.13.0", "answers": {"adds_one": {"type": "noul"}}}'


class MalformedClient(ScriptedJevClient):
    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        self.requests.append((state, questions))
        return RawResponse(MALFORMED_BODY, 200, "application/json")


def test_the_request_is_journaled_before_dispatch_and_the_raw_response_before_parsing() -> None:
    # Arrange
    journal = RecordingJournal()
    judge = Judge(ScriptedJevClient(default_noul=0.9, model="jev-1.13.0"), journal=journal)

    # Act
    answer = judge.ask(STATE, QUESTIONS, thresholds=Thresholds())

    # Assert
    assert [event[0] for event in journal.events] == ["request", "response"]
    assert journal.events[0][1] == answer.request_sha256
    response = journal.events[1][2]
    assert (response.status, response.content_type) == (200, "application/json")
    decoded = response.json()
    assert decoded["model"] == "jev-1.13.0" and decoded["answers"]["adds_one"]["noul"] == 0.9


def test_a_malformed_response_is_journaled_raw_then_the_error_raises() -> None:
    # Arrange
    journal = RecordingJournal()
    judge = Judge(MalformedClient(), journal=journal)

    # Act and Assert
    with pytest.raises(KeyError):
        judge.ask(STATE, QUESTIONS, thresholds=Thresholds())
    assert [event[0] for event in journal.events] == ["request", "response", "failure"]
    assert journal.events[1][2].body == MALFORMED_BODY


def test_a_transport_failure_is_journaled() -> None:
    # Arrange
    class DownClient(ScriptedJevClient):
        def send(self, state: Mapping, questions: Mapping) -> RawResponse:
            raise ConnectionError("provider down")

    journal = RecordingJournal()

    # Act and Assert
    with pytest.raises(ConnectionError):
        Judge(DownClient(), journal=journal).ask(STATE, QUESTIONS, thresholds=Thresholds())
    kind, request_id, error = journal.events[-1][:3]
    assert (kind, request_id, type(error), str(error)) == (
        "failure",
        "attempt-1",
        ConnectionError,
        "provider down",
    )


def test_the_journal_sees_the_masked_request_and_the_request_hash_is_model_free(tmp_path: Path) -> None:
    # Arrange
    journal = RecordingJournal()
    token = f"ghp_{'f6' * 18}"
    state = {"slice": {"code": f'TOKEN = "{token}"'}}
    first = Judge(ScriptedJevClient(model="model-a"), journal=journal).ask(
        state, QUESTIONS, thresholds=Thresholds()
    )
    second = Judge(ScriptedJevClient(model="model-b")).ask(state, QUESTIONS, thresholds=Thresholds())

    # Assert
    assert token not in str(journal.events[0][2])
    assert first.request_sha256 == second.request_sha256


def test_jsonl_journal_appends_one_line_per_event(tmp_path: Path) -> None:
    # Arrange
    journal = JsonlJournal(tmp_path / "journal.jsonl")
    judge = Judge(ScriptedJevClient(), journal=journal)
    check = Check(
        "adds_one", "Does `{item}.code` add one?", Criterion("It adds one."), Criterion("It does not.")
    )

    # Act
    judge.check_each(check, [{"code": "x + 1"}], {})

    # Assert
    kinds = [
        line.split('"kind": "')[1].split('"')[0]
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
    ]
    assert kinds == ["request", "response"]


def test_jsonl_journal_keeps_the_exact_response_bytes_status_and_content_type(tmp_path: Path) -> None:
    # Arrange
    body = b'{"answers": {"adds_one": {"type": "noul", "noul": 0.9}},  "model": "jev-1.13.0"}'

    class WireClient(ScriptedJevClient):
        def send(self, state: Mapping, questions: Mapping) -> RawResponse:
            return RawResponse(body, 200, "application/json; charset=utf-8")

    journal = JsonlJournal(tmp_path / "journal.jsonl")

    # Act
    Judge(WireClient(), journal=journal).ask(STATE, QUESTIONS, thresholds=Thresholds())

    # Assert
    lines = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    response = lines[1]
    assert base64.b64decode(response["body_base64"]) == body
    assert (response["status"], response["content_type"], response["exact"]) == (
        200,
        "application/json; charset=utf-8",
        True,
    )


def test_a_client_that_only_parses_is_journaled_as_decoded_and_marked_inexact() -> None:
    # Arrange
    class ParseOnlyClient:
        model = "jev-latest"

        def ask(self, state: Mapping, questions: Mapping):
            return ScriptedJevClient(default_noul=0.9).ask(state, questions)

    journal = RecordingJournal()

    # Act
    Judge(ParseOnlyClient(), journal=journal).ask(STATE, QUESTIONS, thresholds=Thresholds())

    # Assert
    response = journal.events[1][2]
    assert response.exact is False and response.status is None
    assert response.json()["answers"]["adds_one"]["noul"] == 0.9


def test_the_typesafe_adapter_journals_the_exact_bytes_it_received(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    httpx2 = pytest.importorskip("httpx2")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-for-a-mock-transport")
    body = (
        b'{"model": "jev-1.13.0", "usage": {"input_tokens": 12, "output_tokens": 1},'
        b' "answers": {"adds_one": {"type": "noul", "noul": 0.9}}}'
    )
    mock = httpx2.MockTransport(
        lambda request: httpx2.Response(200, content=body, headers={"content-type": "application/json"})
    )
    journal = JsonlJournal(tmp_path / "journal.jsonl")

    # Act
    answer = Judge(TypeSafeJevClient(transport=mock), journal=journal).ask(
        STATE, QUESTIONS, thresholds=Thresholds()
    )

    # Assert
    response = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[1])
    assert base64.b64decode(response["body_base64"]) == body and response["exact"] is True
    assert answer.noul("adds_one").probability == 0.9


ANSWERS_ONE_OF_TWO = (
    b'{"model": "jev-1.13.0", "usage": {"input_tokens": 12},'
    b' "answers": {"adds_one": {"type": "noul", "noul": 0.9}}}'
)


class _SendsOneOfTwo(ScriptedJevClient):
    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        return RawResponse(ANSWERS_ONE_OF_TWO, 200, "application/json")


class _AsksOneOfTwo:
    """A client that only parses, like the Engine's, answering one of two asked questions."""

    model = "jev-1.13.0"

    def ask(self, state: Mapping, questions: Mapping):
        return response_from_raw(json.loads(ANSWERS_ONE_OF_TWO))


def _ask_sync(judge: Judge, questions: Mapping) -> None:
    judge.ask(STATE, questions, thresholds=Thresholds())


def _ask_async(judge: Judge, questions: Mapping) -> None:
    asyncio.run(judge.ask_async(STATE, questions, thresholds=Thresholds()))


@pytest.mark.parametrize(
    ("client", "ask"),
    [(_SendsOneOfTwo, _ask_sync), (_SendsOneOfTwo, _ask_async), (_AsksOneOfTwo, _ask_sync)],
    ids=["send", "send-async", "ask-only"],
)
def test_a_response_missing_an_asked_answer_leaves_exactly_one_failure_row(
    tmp_path: Path, client, ask
) -> None:
    # Arrange
    judge = Judge(client(), journal=JsonlJournal(tmp_path / "journal.jsonl"))
    questions = {**QUESTIONS, "doubles": {"type": "noul", "instructions": "Does `slice.code` double?"}}

    # Act
    with pytest.raises(UnansweredQuestionError):
        ask(judge, questions)

    # Assert: the failure names the request and the error, and the tokens the response cost are on
    # that request's one response line, so they are counted once
    lines = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    request_id = next(line["request_id"] for line in lines if line["kind"] == "request")
    failures = [line for line in lines if line["kind"] == "failure"]
    responses = [line for line in lines if line["kind"] == "response"]
    assert [failure["request_id"] for failure in failures] == [request_id]
    assert failures[0]["error_type"] == "UnansweredQuestionError"
    assert failures[0]["message"] == "jev-1.13.0 returned no answer for doubles"
    assert [(response["request_id"], response["input_tokens"]) for response in responses] == [
        (request_id, 12)
    ]
    assert judge.input_total.reported == 12


def _recorded_response(tmp_path: Path, body: bytes) -> dict:
    journal = JsonlJournal(tmp_path / "usage.jsonl")
    journal.record_response("r1", RawResponse(body, 200, "application/json"))
    return json.loads((tmp_path / "usage.jsonl").read_text().splitlines()[0])


def test_a_response_records_the_input_tokens_the_provider_reported(tmp_path: Path) -> None:
    body = b'{"model": "m", "usage": {"input_tokens": 12, "output_tokens": 1}, "answers": {}}'

    assert _recorded_response(tmp_path, body)["input_tokens"] == 12


def test_a_reported_zero_is_recorded_as_zero(tmp_path: Path) -> None:
    body = b'{"model": "m", "usage": {"input_tokens": 0}, "answers": {}}'

    assert _recorded_response(tmp_path, body)["input_tokens"] == 0


@pytest.mark.parametrize(
    "body",
    [b'{"model": "m", "answers": {}}', b'{"usage": {}}', b'{"usage": null}', b"not json", b'["list"]'],
)
def test_a_response_without_reported_input_tokens_records_null(tmp_path: Path, body: bytes) -> None:
    recorded = _recorded_response(tmp_path, body)

    assert "input_tokens" in recorded
    assert recorded["input_tokens"] is None


def test_the_tokens_are_on_the_response_line_only_not_on_its_attempt_or_failure_lines(
    tmp_path: Path,
) -> None:
    body = b'{"model": "m", "usage": {"input_tokens": 12}, "answers": {}}'
    raw = RawResponse(body, 200, "application/json")
    journal = JsonlJournal(tmp_path / "usage.jsonl")

    journal.record_attempt("r1", RawAttempt(1, 5.0, b"{}", response=raw))
    journal.record_failure("r1", ValueError("bad"), raw)
    journal.record_response("r1", raw)

    lines = {
        line["kind"]: line for line in map(json.loads, (tmp_path / "usage.jsonl").read_text().splitlines())
    }
    assert lines["response"]["input_tokens"] == 12
    assert "input_tokens" not in lines["http_attempt"]
    assert "input_tokens" not in lines["failure"]


def test_jsonl_journal_keeps_no_request_text_unless_asked(tmp_path: Path) -> None:
    # Arrange
    default = JsonlJournal(tmp_path / "default.jsonl")
    keeping = JsonlJournal(tmp_path / "keeping.jsonl", keep_request_text=True)

    # Act
    Judge(ScriptedJevClient(), journal=default).ask(STATE, QUESTIONS, thresholds=Thresholds())
    Judge(ScriptedJevClient(), journal=keeping).ask(STATE, QUESTIONS, thresholds=Thresholds())

    # Assert
    request = json.loads((tmp_path / "default.jsonl").read_text().splitlines()[0])
    assert "return x + 1" not in (tmp_path / "default.jsonl").read_text()
    assert request["question_ids"] == ["adds_one"] and "state_sha256" in request
    assert "return x + 1" in (tmp_path / "keeping.jsonl").read_text()


ORDERED_STATE = {
    "target": {"description": "the check that limits items per order"},
    "slice": {"file": "orders.py", "lines": "1-2", "code": "if len(items) > limit:\n    raise"},
    "candidates": [],
}
JEV_BODY = (
    b'{"model": "jev-1.13.0", "usage": {"input_tokens": 12, "output_tokens": 1},'
    b' "answers": {"adds_one": {"type": "noul", "noul": 0.9}}}'
)


def test_the_store_keeps_the_request_in_the_order_it_was_sent(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    judge = Judge(ScriptedJevClient(), store=JsonlAnswerStore(path, keep_requests=True))
    judge.ask(ORDERED_STATE, QUESTIONS, thresholds=Thresholds())

    # Act
    record = JsonlAnswerStore(path).records()[0]
    sent_state, sent_questions = record.sent_request()

    # Assert
    assert list(sent_state) == ["target", "slice", "candidates"]
    assert list(record.request["state"]) == ["candidates", "slice", "target"]
    assert sent_questions == QUESTIONS and record.sent_exact is False


def test_a_request_asked_again_from_the_store_sends_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    httpx2 = pytest.importorskip("httpx2")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-for-a-mock-transport")
    sent: list[bytes] = []

    def respond(request):
        sent.append(request.content)
        return httpx2.Response(200, content=JEV_BODY, headers={"content-type": "application/json"})

    path = tmp_path / "answers.jsonl"
    first = Judge(
        TypeSafeJevClient(transport=httpx2.MockTransport(respond)),
        store=JsonlAnswerStore(path, keep_requests=True),
    )
    first.ask(ORDERED_STATE, QUESTIONS, thresholds=Thresholds())
    record = JsonlAnswerStore(path).records()[0]

    # Act
    again = Judge(TypeSafeJevClient(transport=httpx2.MockTransport(respond)))
    again.ask(*record.sent_request(), thresholds=Thresholds())

    # Assert
    assert record.sent_exact is True
    assert base64.b64decode(record.sent_body_base64) == sent[0] == sent[1]


def test_the_journal_keeps_the_request_as_handed_to_the_client_in_its_order(tmp_path: Path) -> None:
    # Arrange
    journal = JsonlJournal(tmp_path / "journal.jsonl", keep_request_text=True)

    # Act
    Judge(ScriptedJevClient(), journal=journal).ask(ORDERED_STATE, QUESTIONS, thresholds=Thresholds())

    # Assert
    request = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[0])
    handed = json.loads(base64.b64decode(request["body_base64"]))
    assert list(handed["state"]) == ["target", "slice", "candidates"]


def test_a_request_cancelled_after_it_was_sent_is_journaled_as_cancelled_after_it_was_sent(
    tmp_path: Path,
) -> None:
    import concurrent.futures

    class CancelledClient:
        model = "cancelled"

        def ask(self, state, questions):
            raise concurrent.futures.CancelledError

    journal = JsonlJournal(tmp_path / "journal.jsonl")

    with pytest.raises(concurrent.futures.CancelledError):
        Judge(CancelledClient(), journal=journal).ask(STATE, QUESTIONS, thresholds=Thresholds())

    failure = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[1])
    assert failure["kind"] == "failure"
    assert failure["error_type"] == "CancelledError"
    assert failure["message"] == "the request was cancelled after it was sent"


SENT = json.dumps({"state": STATE, "questions": QUESTIONS}).encode()
ANSWER = RawResponse(b'{"answers": {}}', 200, "application/json", sent_body=SENT)
RECORDS = {
    "request": lambda journal: journal.record_request(JournalRequest("h", "jev", STATE, QUESTIONS, SENT)),
    "attempt": lambda journal: journal.record_attempt("r1", RawAttempt(1, 5.0, SENT, response=ANSWER)),
    "response": lambda journal: journal.record_response("r1", ANSWER),
    "failure": lambda journal: journal.record_failure("r1", RuntimeError("refused"), ANSWER),
}


@pytest.mark.parametrize("record", sorted(RECORDS))
@pytest.mark.parametrize("keep_request_text", [False, True])
def test_a_journal_says_whether_any_record_kept_a_requests_text(
    tmp_path: Path, record: str, keep_request_text: bool
) -> None:
    # Arrange
    path = tmp_path / "journal.jsonl"
    RECORDS[record](JsonlJournal(path, keep_request_text=keep_request_text))

    # Act
    kept = keeps_request_text(path)

    # Assert
    assert kept is keep_request_text


@pytest.mark.parametrize(
    "record",
    [
        {"request": {"state": STATE, "questions": QUESTIONS}},
        {"sent_body_base64": base64.b64encode(SENT).decode()},
    ],
    ids=["request", "sent body"],
)
@pytest.mark.parametrize("keep_requests", [False, True])
def test_an_answer_store_says_whether_any_record_kept_a_requests_text(
    tmp_path: Path, record: dict, keep_requests: bool
) -> None:
    # Arrange: the store keeps a record's request text only with keep_requests
    path = tmp_path / "answers.jsonl"
    answer = AnswerRecord(
        "h", ("adds_one",), {"adds_one": {"type": "noul", "p": 0.9}}, "jev", 10, {}, **record
    )
    JsonlAnswerStore(path, keep_requests=keep_requests).put(answer)

    # Act
    kept = keeps_request_text(path)

    # Assert
    assert kept is keep_requests


def test_a_line_a_crash_cut_off_tells_nothing_about_request_text(tmp_path: Path) -> None:
    # Arrange: a journal without request text whose last line was cut off mid-write
    path = tmp_path / "journal.jsonl"
    RECORDS["response"](JsonlJournal(path))
    path.write_text(path.read_text() + '{"kind": "request", "body_base64": "ZGVm')

    # Act
    kept = keeps_request_text(path)

    # Assert
    assert kept is False
