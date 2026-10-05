"""Every Jev judgment a history step records names the answer it came from: the request's hash and the
question id, so a run pack joins each decision to its journal request and response without guessing
from journal order."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest
from test_cli_run_logs import TARGET, limit_client, marked_repository
from test_history import FETCHED_HOLDS_LIMIT
from test_typesafe import _jev_server, _seed_index, _stop

from jev_navigator.cli import create_evidence_pack
from jev_navigator.directives.find_code import SearchBudget, StopRule, find_code, find_code_async
from jev_navigator.directives.places import MOVES, function_place, place_for_line
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.store import JsonlAnswerStore, SqliteAnswerStore
from jev_navigator.testing import ScriptedJevClient


def run_pack(tmp_path: Path) -> Path:
    output = tmp_path / "pack"
    create_evidence_pack(
        marked_repository(tmp_path / "repository"),
        ("app/",),
        TARGET,
        ("app/entry.py:5",),
        output,
        SearchBudget(max_calls=5, beam_width=1),
        limit_client(),
        fact_cache_dir=tmp_path / "fact-cache",
    )
    return output


def journal(output: Path) -> list[dict]:
    return [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]


def answered_probability(records: list[dict], source: dict) -> float:
    """The probability the journal's response holds for ``source``'s question, reached through the
    request row with ``source``'s hash, then that request's response row."""
    requests = [
        record
        for record in records
        if record["kind"] == "request" and record["request_sha256"] == source["request_sha256"]
    ]
    assert [source["question_id"] in request["question_ids"] for request in requests] == [True]
    return journal_response(records, source)["answers"][source["question_id"]]["noul"]


def journal_response(records: list[dict], source: dict) -> dict:
    """The decoded response body of the one request row with ``source``'s hash."""
    (request,) = [
        record
        for record in records
        if record["kind"] == "request" and record["request_sha256"] == source["request_sha256"]
    ]
    assert source["question_id"] in request["question_ids"]
    response = next(
        record
        for record in records
        if record["kind"] == "response" and record["request_id"] == request["request_id"]
    )
    return json.loads(base64.b64decode(response["body_base64"]))


def test_each_judgment_of_an_open_step_joins_to_the_answer_that_decided_it(tmp_path: Path) -> None:
    # Arrange
    output = run_pack(tmp_path)

    # Act
    records = journal(output)

    # Assert
    steps = [record["step"] for record in records if record["kind"] == "history_step"]
    opens = [step for step in steps if step["operation"] == "open"]
    picks = [
        step["judgments"]["open_first"]
        for step in opens
        if "choice" in step["judgments"].get("open_first", {})
    ]
    assert opens and picks
    for pick in picks:
        response = journal_response(records, pick["answered_by"])
        assert response["answers"][pick["answered_by"]["question_id"]]["confidence"] == pick["confidence"]
    for step in opens:
        contains = step["judgments"]["contains_target"]
        assert answered_probability(records, contains["answered_by"]) == contains["probability"]
        for offered in step["judgments"]["could_contain"]:
            assert answered_probability(records, offered["answered_by"]) == offered["probability"]


def test_a_place_a_choose_next_step_opens_names_the_answer_that_scored_it(tmp_path: Path) -> None:
    # Arrange
    output = run_pack(tmp_path)

    # Act
    records = journal(output)

    # Assert
    steps = [record["step"] for record in records if record["kind"] == "history_step"]
    scored = [
        chosen
        for step in steps
        if step["operation"] == "choose_next"
        for chosen in step["arguments"]["chosen"]
        if chosen["reason"] in ("queue_score", "open_first")
    ]
    assert scored
    for chosen in scored:
        assert answered_probability(records, chosen["scored_by"]) == chosen["priority"]


def opened_split(tmp_path: Path, judge, async_search: bool):
    """One opening of ``entry`` beside 159 neighbours, too large for one request, through the real
    TypeSafe client against a local Jev that refuses requests over 40,000 characters."""
    tmp_path.mkdir()
    index, _ = _seed_index(tmp_path, 159)
    start = function_place(index, index.find_definition("entry")[0])
    search = find_code_async if async_search else find_code
    pending = search(
        index,
        judge,
        "the function returning field_0",
        [start],
        budget=SearchBudget(max_steps=1, beam_width=1),
        moves={"same_file": MOVES["same_file"]},
    )
    result = asyncio.run(pending) if async_search else pending
    return next(step for step in result.history.steps if step.operation == "open")


@pytest.mark.parametrize("store_kind", [JsonlAnswerStore, SqliteAnswerStore])
@pytest.mark.parametrize("async_search", [False, True])
def test_a_split_opening_joins_each_judgment_to_its_own_sub_request_also_when_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, async_search: bool, store_kind: type
) -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    server = _jev_server([], input_limit=40_000)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    store = tmp_path / "answers.store"
    live = Judge(client, store=store_kind(store), journal=JsonlJournal(tmp_path / "journal.jsonl"))

    # Act
    try:
        first = opened_split(tmp_path / "first", live, async_search)
        replaying = Judge(client, store=store_kind(store), served_model=live.served_model)
        replayed = opened_split(tmp_path / "second", replaying, async_search)
    finally:
        client.close()
        _stop(server)

    # Assert
    records = journal(tmp_path)
    judged = [first.judgments["contains_target"], *first.judgments["could_contain"]]
    assert len({entry["answered_by"]["request_sha256"] for entry in judged}) > 2
    for entry in judged:
        assert answered_probability(records, entry["answered_by"]) == entry["probability"]
    assert replaying.calls == 0
    again = [replayed.judgments["contains_target"], *replayed.judgments["could_contain"]]
    assert [entry["answered_by"] for entry in again] == [
        {**entry["answered_by"], "from_store": True} for entry in judged
    ]


def test_the_stop_step_names_the_answer_of_its_last_stop_check(sample_index, tmp_path: Path) -> None:
    # Arrange
    def answer(question_id: str, question: dict, state: dict) -> float:
        if "fetched" in state:
            bodies = [span["code"] for span in state["fetched"]]
            return 0.9 if any("<= limit" in body for body in bodies) else 0.1
        return 0.3

    journal_path = tmp_path / "journal.jsonl"
    judge = Judge(ScriptedJevClient(nouls=answer), journal=JsonlJournal(journal_path))
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]

    # Act
    result = find_code(
        sample_index,
        judge,
        "the item limit check",
        start,
        budget=SearchBudget(beam_width=1),
        stop_rule=StopRule(FETCHED_HOLDS_LIMIT),
    )

    # Assert
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    stop_check = result.history.steps[-1].judgments["last_stop_check"]
    assert answered_probability(records, stop_check["answered_by"]) == stop_check["probability"]


def test_each_entry_selection_decision_joins_to_the_answer_that_chose_it(tmp_path: Path) -> None:
    # Arrange: no start, so the search chooses its own entry point
    output = tmp_path / "pack"
    manifest = create_evidence_pack(
        marked_repository(tmp_path / "repository"),
        ("app/",),
        TARGET,
        (),
        output,
        SearchBudget(max_calls=6, beam_width=1),
        limit_client(),
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Act
    records = journal(output)

    # Assert
    asked = [
        decision
        for decision in manifest["entry_selection"]["decisions"]
        if decision["confidence"] is not None
    ]
    assert asked
    for decision in asked:
        assert answered_choice(records, decision["answered_by"]) == decision["probabilities"]


def answered_choice(records: list[dict], source: dict) -> dict:
    """The option probabilities the journal's response holds for ``source``'s Choice question."""
    response = journal_response(records, source)
    return response["answers"][source["question_id"]]["probabilities"]


def test_each_find_all_verdict_joins_to_the_answer_that_judged_it(tmp_path: Path) -> None:
    # Arrange
    output = tmp_path / "pack"
    manifest = create_evidence_pack(
        marked_repository(tmp_path / "repository"),
        ("app/",),
        TARGET,
        ("app/entry.py:5",),
        output,
        SearchBudget(max_calls=8, beam_width=1),
        limit_client(),
        fact_cache_dir=tmp_path / "fact-cache",
        workflow="findall",
    )

    # Act
    records = journal(output)

    # Assert
    search = manifest["search"]
    verdicts = [*search["found"], *search["unsure"], *search["searched"]]
    assert verdicts
    for verdict in verdicts:
        assert answered_probability(records, verdict["answered_by"]) == verdict["probability"]


@pytest.mark.parametrize("section", ["decisions", "history"])
def test_a_stop_rule_reading_the_history_replays_without_a_live_request(
    sample_index, tmp_path: Path, section: str
) -> None:
    # Arrange: the answer sources differ between a live run and its replay (from_store), so Jev
    # must never see them, or the replayed stop check becomes a new request
    states: list[dict] = []

    def answer(question_id: str, question: dict, state: dict) -> float:
        states.append(state)
        return 0.2 if section in state else 0.3

    rule = StopRule(FETCHED_HOLDS_LIMIT, sections=(section,))
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]
    store = tmp_path / "answers.jsonl"
    client = ScriptedJevClient(nouls=answer)
    live = Judge(client, store=JsonlAnswerStore(store))
    find_code(
        sample_index,
        live,
        "the item limit check",
        start,
        budget=SearchBudget(beam_width=1, max_steps=3),
        stop_rule=rule,
    )

    # Act
    replaying = Judge(client, store=JsonlAnswerStore(store), served_model=live.served_model)
    find_code(
        sample_index,
        replaying,
        "the item limit check",
        start,
        budget=SearchBudget(beam_width=1, max_steps=3),
        stop_rule=rule,
    )

    # Assert
    assert live.calls > 0
    assert replaying.calls == 0
    assert any(section in state for state in states)
    assert not keys_at_any_depth(states) & {"answered_by", "scored_by"}


def keys_at_any_depth(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in keys_at_any_depth(item)}
    if isinstance(value, list):
        return {key for item in value for key in keys_at_any_depth(item)}
    return set()
