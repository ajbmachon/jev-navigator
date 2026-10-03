"""The optional LLM step: when it runs, retries of unparsable replies, durability and connector failures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from jev_navigator.connectors import CommandConnector, ConnectorError, OpenAICompatibleConnector
from jev_navigator.judgments.judge import Judge, PickResult
from jev_navigator.judgments.questions import Pick
from jev_navigator.judgments.secrets import SecretInRequestError
from jev_navigator.llm_step import JsonContract, LlmGuard, LlmStep, PickFromOptions
from jev_navigator.testing import ScriptedJevClient

FAKE_LLM = Path(__file__).parent / "fake_llm" / "reply.sh"
PHRASE = Pick("phrase", "Which phrase in `comment.text` names what `slice.code` does?")
PHRASES = {"p0": "retries twice", "p1": "logs the error"}
STATE = {
    "comment": {"text": "retries twice and logs the error"},
    "slice": {"code": "for attempt in range(2): ..."},
}


def fake_connector(tmp_path: Path, *replies: str, prompt_log: Path | None = None) -> CommandConnector:
    replies_file = tmp_path / "replies.txt"
    replies_file.write_text("\n".join(replies) + "\n")
    if prompt_log is None:
        return CommandConnector(["sh", str(FAKE_LLM), str(replies_file)], name="fake", model="fake-1")
    return CommandConnector(
        [
            "sh",
            "-c",
            'cat >> "$1"; head -n 1 "$2"; tail -n +2 "$2" > "$2.rest" && mv "$2.rest" "$2"',
            "x",
            str(prompt_log),
            str(replies_file),
        ],
        name="fake",
        model="fake-1",
    )


def phrase_step(connector, guard: LlmGuard | None = None) -> LlmStep[PickResult, str]:
    """The README example: ask an LLM when the phrase Choice is below 0.70."""
    return LlmStep(
        name="phrase_fallback",
        when=lambda result: result.confidence < 0.70,
        context=lambda result: {"comment": STATE["comment"], "slice": STATE["slice"], "options": PHRASES},
        answer=PickFromOptions(
            "Which phrase of the comment names what the code does?", answer_field="phrase"
        ),
        connector=connector,
        guard=guard or LlmGuard(),
    )


def unsure_phrase_pick() -> PickResult:
    client = ScriptedJevClient(choices={"phrase": {"p0": 0.55, "p1": 0.45}})
    return Judge(client).pick(PHRASE, PHRASES, STATE)


def records_in(store: Path) -> list[dict]:
    return [json.loads(line) for line in store.read_text().splitlines()]


def test_the_step_runs_only_when_the_user_predicate_says_so(tmp_path: Path) -> None:
    # Arrange
    confident = Judge(ScriptedJevClient()).pick(PHRASE, PHRASES, STATE)
    step = phrase_step(fake_connector(tmp_path, '{"phrase": "p1"}'))

    # Act
    skipped = step.run(confident)
    called = step.run(unsure_phrase_pick())

    # Assert
    assert skipped.status == "not_requested" and skipped.answer is None and skipped.attempts == 0
    assert called.status == "answered"
    assert called.answer == "p1" and called.connector == "fake"


def test_an_unparsable_reply_is_retried_once_then_reported(tmp_path: Path) -> None:
    # Arrange
    step = phrase_step(fake_connector(tmp_path, "not json", '{"phrase": "made-up"}'))

    # Act
    call = step.run(unsure_phrase_pick())

    # Assert
    assert call.status == "parse_failed"
    assert call.answer is None
    assert "made-up" in call.parse_error
    assert call.attempts == 2


def test_a_retry_that_parses_is_used(tmp_path: Path) -> None:
    # Act
    call = phrase_step(fake_connector(tmp_path, "I pick p0", '{"phrase": "p0"}')).run(unsure_phrase_pick())

    # Assert
    assert call.status == "answered" and call.answer == "p0"


def test_a_failed_first_reply_survives_a_successful_retry_with_distinct_attempts(tmp_path: Path) -> None:
    # Arrange
    store = tmp_path / "llm.jsonl"
    prompt_log = tmp_path / "prompts.txt"
    step = phrase_step(
        fake_connector(tmp_path, "I pick p0", '{"phrase": "p0"}', prompt_log=prompt_log),
        LlmGuard(store_path=store),
    )

    # Act
    call = step.run(unsure_phrase_pick())

    # Assert
    records = records_in(store)
    assert "I pick p0\n" in [record.get("reply") for record in records]
    assert call.status == "answered" and call.answer == "p0" and call.attempts == 2
    attempts = [r for r in records if r.get("event") == "attempt"]
    replies = [r for r in records if r.get("event") == "reply"]
    parses = [r for r in records if r.get("event") == "parse"]
    assert len(attempts) == len(replies) == len(parses) == 2
    assert attempts[0]["attempt"] != attempts[1]["attempt"]
    assert [r["connector"] for r in attempts] == ["fake", "fake"]
    assert [r["model"] for r in attempts] == ["fake-1", "fake-1"]
    assert attempts[0]["prompt_sha256"] != attempts[1]["prompt_sha256"]
    assert [r["attempt"] for r in replies] == [attempts[0]["attempt"], attempts[1]["attempt"]]
    # The exact replies, newline included, exactly as the connector returned them.
    assert [r["reply"] for r in replies] == ["I pick p0\n", '{"phrase": "p0"}\n']
    assert parses[0]["parse_error"] and parses[0]["attempt"] == attempts[0]["attempt"]
    assert not parses[1]["parse_error"] and parses[1]["attempt"] == attempts[1]["attempt"]
    assert call.prompt_sha256 == attempts[1]["prompt_sha256"]
    # The retry prompt is the one that carries the parse error.
    assert prompt_log.read_text().count("could not be used") == 1


def test_the_reply_is_durable_in_the_store_by_the_time_the_parser_runs(tmp_path: Path) -> None:
    # Arrange
    store = tmp_path / "llm.jsonl"
    observed: list[str] = []

    class Parser:
        def render(self, context: Mapping, parse_error: str = "") -> str:
            return "Reply with one line of JSON."

        def parse(self, reply: str, context: Mapping) -> str:
            observed.extend(r["reply"] for r in records_in(store) if r.get("event") == "reply")
            return reply.upper()

    step = LlmStep(
        "durable",
        lambda result: True,
        lambda result: {},
        Parser(),
        fake_connector(tmp_path, "raw reply"),
        LlmGuard(store_path=store),
    )

    # Act
    result = step.run(None)

    # Assert
    assert result.status == "answered" and result.answer == "RAW REPLY\n"
    assert observed == ["raw reply\n"]


def test_a_failing_connector_attempt_is_recorded_and_its_exception_still_propagates(tmp_path: Path) -> None:
    # Arrange
    store = tmp_path / "llm.jsonl"
    failing = CommandConnector(["sh", "-c", "echo provider down >&2; exit 3"], name="failing", model="f-1")
    step = phrase_step(failing, LlmGuard(store_path=store))

    # Act and Assert
    with pytest.raises(ConnectorError):
        step.run(unsure_phrase_pick())

    attempts = [r for r in records_in(store) if r.get("event") == "attempt"]
    failures = [r for r in records_in(store) if r.get("event") == "failure"]
    assert len(attempts) == 1 and len(failures) == 1
    assert failures[0]["attempt"] == attempts[0]["attempt"]
    assert attempts[0]["connector"] == "failing" and attempts[0]["model"] == "f-1"
    assert attempts[0]["prompt_sha256"]
    assert not [r for r in records_in(store) if r.get("event") == "reply"]
    assert "exited 3" in failures[0]["error"]


def test_not_requested_and_budget_exhausted_are_distinct_and_never_call_the_connector(tmp_path: Path) -> None:
    # Arrange
    prompts: list[str] = []

    class Counting:
        name, model = "counting", "c"

        def complete(self, prompt: str) -> str:
            prompts.append(prompt)
            return '{"phrase": "p0"}'

    # Act
    not_requested = phrase_step(Counting()).run(Judge(ScriptedJevClient()).pick(PHRASE, PHRASES, STATE))
    budgeted = phrase_step(Counting(), LlmGuard(max_calls=1))
    first = budgeted.run(unsure_phrase_pick())
    exhausted = budgeted.run(unsure_phrase_pick())

    # Assert
    assert not_requested.status == "not_requested"
    assert first.status == "answered"
    assert exhausted.status == "budget_exhausted"
    assert exhausted.answer is None and exhausted.parse_error == "" and exhausted.attempts == 0
    assert len(prompts) == 1


def test_the_budget_stops_further_calls(tmp_path: Path) -> None:
    # Arrange
    step = phrase_step(
        fake_connector(tmp_path, '{"phrase": "p0"}', '{"phrase": "p1"}'), LlmGuard(max_calls=1)
    )
    step.run(unsure_phrase_pick())

    # Act
    second = step.run(unsure_phrase_pick())

    # Assert
    assert second.status == "budget_exhausted"


def test_context_is_masked_and_a_prompt_with_a_secret_is_refused(tmp_path: Path) -> None:
    # Arrange
    prompts: list[str] = []

    class Recording:
        name, model = "recording", "r"

        def complete(self, prompt: str) -> str:
            prompts.append(prompt)
            return '{"phrase": "p0"}'

    token = f"ghp_{'e5' * 18}"
    leaky = LlmStep(
        "leaky",
        lambda result: True,
        lambda result: {"code": f'TOKEN = "{token}"', "options": PHRASES},
        PickFromOptions("Pick", answer_field="phrase"),
        Recording(),
    )
    unmasked = LlmStep(
        "unmasked",
        lambda result: True,
        lambda result: {"code": f'TOKEN = "{token}"', "options": PHRASES},
        PickFromOptions("Pick", answer_field="phrase"),
        Recording(),
        LlmGuard(masker=None),
    )

    # Act
    answered = leaky.run(None)

    # Assert
    assert answered.status == "answered" and answered.answer == "p0"
    assert token not in prompts[0]
    with pytest.raises(SecretInRequestError):
        unmasked.run(None)
    assert len(prompts) == 1


def test_every_call_is_stored_with_prompt_hash_reply_and_answer(tmp_path: Path) -> None:
    # Arrange
    store = tmp_path / "llm.jsonl"
    step = phrase_step(fake_connector(tmp_path, '{"phrase": "p1"}'), LlmGuard(store_path=store))

    # Act
    step.run(unsure_phrase_pick())

    # Assert
    records = records_in(store)
    assert all(r["kind"] == "llm_step" for r in records)
    summary = [r for r in records if r.get("event") == "call"][0]
    assert (summary["step"], summary["answer"]) == ("phrase_fallback", "p1")
    assert summary["prompt_sha256"] and '"phrase": "p1"' in summary["reply"]
    attempt = [r for r in records if r.get("event") == "attempt"][0]
    assert (attempt["connector"], attempt["model"]) == ("fake", "fake-1")


def test_json_contract_checks_required_fields() -> None:
    # Arrange
    contract = JsonContract("Say whether the comment is accurate.", {"accurate": bool, "line": int})

    # Act and Assert
    assert contract.parse('{"accurate": true, "line": 4}', {}) == {"accurate": True, "line": 4}
    with pytest.raises(ValueError, match="line"):
        contract.parse('{"accurate": true}', {})


def test_openai_compatible_connector_posts_the_prompt() -> None:
    # Arrange
    posted: list[tuple[str, dict]] = []

    def post(url: str, body: dict, headers: dict, timeout: float) -> dict:
        posted.append((url, body))
        return {"choices": [{"message": {"content": "hello"}}]}

    # Act
    reply = OpenAICompatibleConnector("http://localhost:8000/v1", "glm", post=post).complete("prompt")

    # Assert
    assert reply == "hello"
    assert posted[0][0] == "http://localhost:8000/v1/chat/completions" and posted[0][1]["model"] == "glm"
