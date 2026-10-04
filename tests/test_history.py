from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jev_navigator.directives.find_code import (
    Outcome,
    SearchBudget,
    StopRule,
    StopRuleTooLargeError,
    find_code,
)
from jev_navigator.directives.places import place_for_line
from jev_navigator.history import (
    FetchedSpan,
    History,
    HistoryCheck,
    HistoryOutcome,
    HistoryStep,
    HistoryTooLargeError,
    SectionLimit,
    UnknownSectionError,
    ceiling_curve,
    judge_history,
    judge_sections,
)
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS, InputBudgetExceededError
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import Judge, request_exceeds_input_budget
from jev_navigator.judgments.questions import Check, Criterion, serialized_chars
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import ScriptedJevClient

HOLDS_LIMIT = Check(
    "holds_limit_check",
    "Does `history` contain code that compares the number of items with a limit?",
    Criterion("A fetched code body compares an item count with a limit."),
    Criterion("No fetched code body makes that comparison."),
)
FETCHED_HOLDS_LIMIT = Check(
    "fetched_holds_limit_check",
    "Does `fetched` contain code that compares the number of items with a limit?",
    Criterion("A code body in `fetched` compares an item count with a limit."),
    Criterion("No code body in `fetched` makes that comparison."),
)


def step(number: int, code: str = "x = 1", decision: str = "opened") -> HistoryStep:
    source = {"file": f"f{number}.py", "lines": [1, 1], "commit": "abc"}
    return HistoryStep("open", {"n": number}, (FetchedSpan(source, code),), {"found": 0.1}, decision)


def test_fetched_holds_only_code_with_sources_and_decisions_hold_the_rest() -> None:
    # Arrange
    history = History()
    history.append(step(1, "def f(): pass"))

    # Act
    state = history.state_for(["fetched", "decisions"])

    # Assert
    assert state["fetched"] == [{"file": "f1.py", "lines": [1, 1], "commit": "abc", "code": "def f(): pass"}]
    assert state["decisions"] == [
        {"operation": "open", "arguments": {"n": 1}, "judgments": {"found": 0.1}, "decision": "opened"}
    ]


def test_the_history_section_holds_each_step_with_its_code_but_not_its_judgments() -> None:
    # Arrange
    history = History()
    history.append(step(1, "def f(): pass"))

    # Act
    state = history.state_for(["history"])

    # Assert
    assert state["history"] == {
        "steps": [
            {
                "operation": "open",
                "arguments": {"n": 1},
                "fetched": [{"file": "f1.py", "lines": [1, 1], "commit": "abc", "code": "def f(): pass"}],
            }
        ]
    }


def test_a_check_gets_exactly_the_sections_it_selects() -> None:
    # Arrange
    history = History(sections={"subject": {"description": "the limit check"}, "shown_code": ""})
    history.append(step(1))
    history.set_section("shown_code", "if x: pass")

    # Act
    state = history.state_for(["subject", "shown_code"])

    # Assert
    assert state == {"subject": {"description": "the limit check"}, "shown_code": "if x: pass"}
    with pytest.raises(UnknownSectionError):
        history.state_for(["notes"])
    with pytest.raises(UnknownSectionError):
        history.set_section("notes", "x")


def test_section_limits_apply_before_any_code_is_evicted() -> None:
    # Arrange
    limits = {
        "fetched": SectionLimit(max_chars=5),
        "history": SectionLimit(max_chars=5),
        "decisions": SectionLimit(max_entries=2, max_chars=5),
    }
    history = History(limits=limits)
    for number in range(3):
        history.append(replace(step(number, "y" * 20), judgments={"candidates": [{"preview": "z" * 20}]}))

    # Act
    state = history.state_for(["fetched", "history", "decisions"])

    # Assert
    assert [entry["arguments"]["n"] for entry in state["decisions"]] == [1, 2]
    assert state["fetched"][0]["code"] == "yyyyy[... 15 characters cut]"
    assert state["history"]["steps"][0]["fetched"][0]["code"] == "yyyyy[... 15 characters cut]"
    assert state["decisions"][0]["judgments"]["candidates"][0]["preview"] == "zzzzz[... 15 characters cut]"
    assert history.steps[0].fetched[0].code == "y" * 20
    assert history.steps[0].judgments["candidates"][0]["preview"] == "z" * 20
    assert history.evictions == []


def test_the_oldest_code_is_evicted_first_and_every_eviction_is_recorded() -> None:
    # Arrange
    history = History(budget_chars=1_000)
    for number in range(3):
        history.append(step(number, "y" * 300))

    # Act
    state = history.state_for(["fetched", "decisions"])

    # Assert
    codes = [entry["code"] for entry in state["fetched"]]
    assert codes[0] == "[evicted]" and codes[-1] == "y" * 300
    assert [entry["decision"] for entry in state["decisions"]] == ["opened"] * 3
    assert history.evictions[0]["file"] == "f0.py"
    assert history.steps[0].fetched[0].code == "y" * 300


def test_a_check_that_reads_no_code_never_evicts_code() -> None:
    # Arrange
    history = History(budget_chars=1_000)
    for number in range(3):
        history.append(step(number, "y" * 300))

    # Act
    history.state_for(["decisions"])

    # Assert
    assert history.evictions == []


def test_the_budget_is_capped_at_jevs_state_limit() -> None:
    assert History(budget_chars=10**6).budget_chars == JEV_INPUT_BOX_CHARS


def test_sections_that_do_not_fit_even_without_code_raise_and_evict_nothing() -> None:
    # Arrange: the decisions alone need more than the budget, and no code is read to evict.
    history = History(budget_chars=100)
    for number in range(3):
        history.append(step(number, "y" * 300, decision="opened " + "z" * 100))

    # Act and Assert
    with pytest.raises(HistoryTooLargeError):
        history.state_for(["decisions"])
    assert history.evictions == []


def test_code_is_evicted_before_the_history_is_declared_too_large() -> None:
    # Arrange: the stubs fit the budget, the code does not.
    history = History(budget_chars=900)
    for number in range(3):
        history.append(step(number, "y" * 600))

    # Act
    state = history.state_for(["fetched"])

    # Assert
    assert [entry["code"] for entry in state["fetched"]] == ["[evicted]"] * 2 + ["y" * 600]
    assert len(history.evictions) == 2


def test_the_shared_state_counts_against_the_history_budget() -> None:
    history = History(budget_chars=2_500)
    history.append(step(1, "y" * 1_500))
    client = ScriptedJevClient()

    judge_history(Judge(client), history, FETCHED_HOLDS_LIMIT, {"pad": "x" * 1_200})

    state, _ = client.requests[0]
    assert serialized_chars(state) <= 2_500
    assert state["fetched"][0]["code"] == "[evicted]"
    assert [eviction["file"] for eviction in history.evictions] == ["f1.py"]


def test_the_history_leaves_room_for_the_question_that_reads_it() -> None:
    history = History(budget_chars=10**6)
    for number in range(100):
        history.append(step(number, "y" * 1_000))
    long_question = Check(
        "long_question",
        "Does `fetched` contain code that compares the number of items with a limit? " + "z" * 3_000,
        Criterion("A code body in `fetched` compares an item count with a limit."),
        Criterion("No code body in `fetched` makes that comparison."),
    )
    client = ScriptedJevClient()

    judge_history(Judge(client), history, long_question)

    state, questions = client.requests[0]
    assert not request_exceeds_input_budget(state, questions)
    assert history.evictions


def test_every_appended_step_is_journaled(tmp_path: Path) -> None:
    # Arrange
    history = History(recorder=JsonlJournal(tmp_path / "journal.jsonl"))

    # Act
    history.append(step(1))
    history.append(step(2))

    # Assert
    text = (tmp_path / "journal.jsonl").read_text()
    lines = [json.loads(line) for line in text.splitlines()]
    assert [line["kind"] for line in lines] == ["history_step", "history_step"]
    assert "x = 1" not in text
    assert (
        lines[0]["step"]["fetched"][0]["file"] == "f1.py" and "code_sha256" in lines[0]["step"]["fetched"][0]
    )


def test_checks_with_one_selection_share_a_request_and_other_selections_ask_apart() -> None:
    # Arrange
    names_limit = Check(
        "names_limit",
        "Does `fetched` contain a line that reads a setting named like a limit?",
        Criterion("A line reads such a setting."),
        Criterion("No line reads such a setting."),
    )
    chose_twice = Check(
        "chose_twice",
        "Does `decisions` list the same place under two choose_next steps?",
        Criterion("One place appears in two choose_next steps."),
        Criterion("Every place appears once."),
    )
    client = ScriptedJevClient(default_noul=0.9)
    history = History()
    history.append(step(1))

    # Act
    judged = judge_sections(
        Judge(client),
        history,
        {
            "holds": HistoryCheck(FETCHED_HOLDS_LIMIT, ("fetched",)),
            "names": HistoryCheck(names_limit, ("fetched",)),
            "twice": HistoryCheck(chose_twice, ("decisions",)),
        },
    )

    # Assert
    assert sorted(len(questions) for _, questions in client.requests) == [1, 2]
    assert sorted(tuple(state) for state, _ in client.requests) == [("decisions",), ("fetched",)]
    assert set(judged) == {"holds", "names", "twice"}
    assert history.state_for(["previous_judgments"])["previous_judgments"]["holds"]["probability"] == 0.9


@pytest.mark.parametrize(
    ("probability", "exhausted", "outcome"),
    [
        (0.9, False, HistoryOutcome.FOUND),
        (0.1, False, HistoryOutcome.SEARCHED_NOT_FOUND),
        (0.5, False, HistoryOutcome.CONTINUE),
        (0.5, True, HistoryOutcome.NOT_INSPECTED),
    ],
)
def test_the_callers_check_maps_to_three_outcomes(probability: float, exhausted: bool, outcome) -> None:
    # Arrange
    history = History()
    history.append(step(1))

    # Act
    judged = judge_history(
        Judge(ScriptedJevClient(default_noul=probability)), history, HOLDS_LIMIT, exhausted=exhausted
    )

    # Assert
    assert judged.outcome == outcome


def test_the_cache_key_covers_the_history_actually_sent(tmp_path: Path) -> None:
    # Arrange
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    first = History()
    first.append(step(1, "a = 1"))
    second = History()
    second.append(step(1, "a = 2"))
    client = ScriptedJevClient()
    judge = Judge(client, store=store)

    # Act
    judge_history(judge, first, HOLDS_LIMIT)
    judge_history(judge, first, HOLDS_LIMIT)
    judge_history(judge, second, HOLDS_LIMIT)

    # Assert
    assert len(client.requests) == 2


def test_find_code_can_stop_on_the_callers_history_check(sample_index: CodeIndex) -> None:
    # Arrange
    def answer(question_id: str, question: dict, state: dict) -> float:
        if "fetched" in state:
            bodies = [span["code"] for span in state["fetched"]]
            return 0.9 if any("<= limit" in body for body in bodies) else 0.1
        return 0.3

    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]

    # Act
    result = find_code(
        sample_index,
        Judge(ScriptedJevClient(nouls=answer)),
        "the item limit check",
        start,
        budget=SearchBudget(beam_width=1),
        stop_rule=StopRule(FETCHED_HOLDS_LIMIT),
    )

    # Assert
    assert result.outcome == Outcome.STOP_RULE
    assert result.stop_judgment.outcome == HistoryOutcome.FOUND
    opened_files = [
        entry.fetched[0].source["file"] for entry in result.history.steps if entry.operation == "open"
    ]
    assert opened_files[0] == "app/orders.py" and "app/validation.py" in opened_files


def test_a_stop_rule_whose_history_cannot_fit_stops_the_search_with_a_named_error(
    sample_index: CodeIndex,
) -> None:
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]
    rule = StopRule(HOLDS_LIMIT, sections=("decisions",), budget_chars=5)

    with pytest.raises(StopRuleTooLargeError, match="holds_limit_check") as raised:
        find_code(
            sample_index,
            Judge(ScriptedJevClient()),
            "the item limit check",
            start,
            budget=SearchBudget(beam_width=1),
            stop_rule=rule,
        )

    assert isinstance(raised.value.__cause__, HistoryTooLargeError)


def test_a_stop_rule_the_provider_refuses_for_size_stops_the_search_with_the_same_named_error(
    sample_index: CodeIndex,
) -> None:
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]
    rule = StopRule(FETCHED_HOLDS_LIMIT)

    class RefusesTheStopRequest(ScriptedJevClient):
        def send(self, state, questions):
            if "fetched" in state:
                raise InputBudgetExceededError("max_tokens_exceeded")
            return super().send(state, questions)

    with pytest.raises(StopRuleTooLargeError, match="fetched_holds_limit_check") as raised:
        find_code(
            sample_index,
            Judge(RefusesTheStopRequest()),
            "the item limit check",
            start,
            budget=SearchBudget(beam_width=1),
            stop_rule=rule,
        )

    assert isinstance(raised.value.__cause__, InputBudgetExceededError)


def test_the_ceiling_curve_reports_probability_against_history_size() -> None:
    # Arrange
    def grows_with_evidence(question_id: str, question: dict, state: dict) -> float:
        return min(0.95, 0.2 * len(state["fetched"]))

    steps = [step(number) for number in range(5)]

    # Act
    points = ceiling_curve(Judge(ScriptedJevClient(nouls=grows_with_evidence)), steps, FETCHED_HOLDS_LIMIT)

    # Assert
    assert [point.steps for point in points] == [1, 2, 3, 4, 5]
    assert [round(point.probability, 2) for point in points] == [0.2, 0.4, 0.6, 0.8, 0.95]
    assert points[0].chars < points[-1].chars


def test_each_search_step_records_judgments_candidates_and_why_the_next_place_was_chosen(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: 0.1 if question_id.startswith("contains_target") else 0.5
    )
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        "the item limit check",
        start,
        budget=SearchBudget(max_steps=2, beam_width=1),
    )

    # Assert
    steps = [entry.to_json() for entry in result.history.steps]
    assert [entry["operation"] for entry in steps] == ["choose_next", "open", "choose_next", "open", "stop"]
    first_choice, first_open, second_choice = steps[0], steps[1], steps[2]
    assert first_choice["arguments"]["chosen"][0]["reason"] == "start"
    assert first_open["judgments"]["contains_target"] == {"probability": 0.1, "verdict": "no"}
    offered = first_open["judgments"]["could_contain"]
    assert offered and {"place", "signature", "probability", "verdict"} <= set(offered[0])
    assert first_open["decision"] == "start judged no"
    assert second_choice["arguments"]["chosen"][0]["reason"] == "open_first"
    assert second_choice["arguments"]["chosen"][0]["place"] in {entry["place"] for entry in offered}


def test_a_budget_stop_ends_the_history_with_the_not_inspected_frontier(sample_index: CodeIndex) -> None:
    # Arrange
    def answer(question_id: str, question: dict, state: dict) -> float:
        return 0.5 if "fetched" in state or question_id.startswith("could_contain") else 0.1

    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]

    # Act
    result = find_code(
        sample_index,
        Judge(ScriptedJevClient(nouls=answer)),
        "the item limit check",
        start,
        budget=SearchBudget(max_steps=1, beam_width=1),
        stop_rule=StopRule(FETCHED_HOLDS_LIMIT),
    )

    # Assert
    assert result.outcome == Outcome.BUDGET
    assert result.stop_judgment.outcome == HistoryOutcome.CONTINUE
    last = result.history.steps[-1].to_json()
    assert last["decision"] == "stopped: budget"
    assert [(entry["place"], entry["reason"]) for entry in last["judgments"]["not_inspected"]] == [
        (entry.place_key, entry.reason) for entry in result.not_inspected
    ]


def test_without_a_stop_rule_the_history_is_still_recorded_and_costs_no_calls(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=lambda question_id, question, state: 0.1)
    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]

    # Act
    result = find_code(
        sample_index, Judge(client), "the item limit check", start, budget=SearchBudget(max_steps=1)
    )

    # Assert
    assert result.history is not None and result.stop_judgment is None
    assert result.calls == len(client.requests) == 1


def test_the_default_stop_view_is_the_fetched_code_and_verdicts_need_the_decisions_section(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    seen_states: list[dict] = []

    def answer(question_id: str, question: dict, state: dict) -> float:
        if question_id.startswith(("holds_limit_check", "fetched_holds_limit_check")):
            seen_states.append(state)
        return 0.1

    start = [place_for_line(sample_index, "app/orders.py", 6, "start")]
    budget = SearchBudget(max_steps=1, beam_width=1)
    judge = Judge(ScriptedJevClient(nouls=answer))

    # Act
    for rule in (
        StopRule(FETCHED_HOLDS_LIMIT),
        StopRule(HOLDS_LIMIT, sections=("history",)),
        StopRule(HOLDS_LIMIT, sections=("decisions",)),
    ):
        find_code(sample_index, judge, "the item limit check", start, budget=budget, stop_rule=rule)

    # Assert
    code_only, steps_with_code, verdicts = seen_states
    assert set(code_only) == {"fetched"} and code_only["fetched"][0]["code"]
    assert "probability" not in json.dumps(code_only) and "verdict" not in json.dumps(code_only)
    assert {tuple(entry) for entry in steps_with_code["history"]["steps"]} == {
        ("operation", "arguments", "fetched")
    }
    assert "probability" not in json.dumps(steps_with_code)
    assert verdicts["decisions"][-1]["judgments"]["contains_target"]["verdict"] == "no"
