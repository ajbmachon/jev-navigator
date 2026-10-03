from __future__ import annotations

import os
import re
import signal
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.find_code import OPEN_FIRST, Outcome, SearchBudget, find_code
from jev_navigator.directives.places import (
    MOVES,
    Place,
    function_place,
    neighbours,
    place_for_line,
    range_place,
)
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import CodeSlice, Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import ScriptedJevClient

TARGET = "the check that limits how many items an order may have"
_SLOT = re.compile(r"candidates\[(\d+)\]")


def scripted(
    found: Callable[[str], float], could_contain: Callable[[str], float]
) -> Callable[[str, Mapping, Mapping], float]:
    """Answers the found question from the opened code and each neighbour question from its signature."""

    def answer(question_id: str, question: Mapping, state: Mapping) -> float:
        slot = _SLOT.search(question["instructions"])
        if slot is None:
            return found(state["slice"]["code"])
        return could_contain(state["candidates"][int(slot.group(1))]["signature"])

    return answer


def start_at_place(index: CodeIndex) -> list[Place]:
    return [place_for_line(index, "app/orders.py", 6, "start")]


def test_search_follows_likely_neighbours_until_the_target_is_found(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "len(order.items) <= limit" in code else 0.05,
            could_contain=lambda signature: (
                0.9 if "check_limits" in signature or "validate_order" in signature else 0.1
            ),
        )
    )

    # Act
    result = find_code(
        sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=SearchBudget(beam_width=1)
    )

    # Assert
    assert result.outcome == Outcome.FOUND
    assert result.found[0].code.span.name == "check_limits"
    assert [key.split(":")[0] for key in result.found[0].path] == [
        "app/orders.py",
        "app/validation.py",
        "app/validation.py",
    ]
    assert result.not_inspected
    assert result.steps == 3


def test_a_low_neighbour_score_keeps_the_neighbour_as_not_inspected(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"none": 1.0}},
    )

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert [visit.code.span.name for visit in result.starts] == ["place"]
    assert result.not_inspected and {entry.reason for entry in result.not_inspected} == {"deprioritized"}
    assert result.found == () and result.unsure == () and result.searched == ()


def test_a_system_discovered_initial_candidate_can_be_found(sample_index: CodeIndex) -> None:
    target = sample_index.find_definition("check_limits")[0]
    candidate = function_place(sample_index, target, "automatic entry selection")
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.95, could_contain=lambda _: 0.1))

    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        initial_candidates=[(candidate, 0.9)],
    )

    assert result.outcome == Outcome.FOUND
    assert result.found[0].code.span.name == "check_limits"


def test_found_search_records_why_a_viable_alternative_was_not_opened(sample_index):
    target = function_place(sample_index, sample_index.find_definition("check_limits")[0])
    other = function_place(sample_index, sample_index.find_definition("place")[0])
    result = find_code(
        sample_index,
        Judge(ScriptedJevClient(nouls=lambda *_: 0.95)),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        initial_candidates=[(target, 0.9), (other, 0.8)],
        moves={},
    )
    assert result.outcome == Outcome.FOUND
    assert len(result.not_inspected) == 1
    assert result.not_inspected[0].place_key == other.key
    assert result.not_inspected[0].reason == "target_found"


def test_a_low_choice_probability_does_not_discard_an_unjudged_entry_alternative(
    sample_index: CodeIndex,
) -> None:
    wrong = function_place(sample_index, sample_index.find_definition("place")[0])
    target = function_place(sample_index, sample_index.find_definition("check_limits")[0])
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "len(order.items) <= limit" in code else 0.05,
            could_contain=lambda _: 0.05,
        )
    )

    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        [],
        budget=SearchBudget(max_steps=2, beam_width=1),
        initial_candidates=[(wrong, 0.99), (target, 0.01)],
        moves={},
    )

    assert result.outcome == Outcome.FOUND
    assert result.found[0].code.span.name == "check_limits"


def test_an_unopened_entry_alternative_remains_visible_at_the_step_budget(
    sample_index: CodeIndex,
) -> None:
    wrong = function_place(sample_index, sample_index.find_definition("place")[0])
    target = function_place(sample_index, sample_index.find_definition("check_limits")[0])
    client = ScriptedJevClient(nouls=scripted(found=lambda _: 0.05, could_contain=lambda _: 0.05))

    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        [],
        budget=SearchBudget(max_steps=1, beam_width=1),
        initial_candidates=[(wrong, 0.99), (target, 0.01)],
        moves={},
    )

    assert result.outcome == Outcome.BUDGET
    remaining = next(item for item in result.not_inspected if item.place_key == target.key)
    assert remaining.reason == "budget"
    assert remaining.tier.name == "DISCOVERED"


def test_search_reports_unsure_only_when_only_unsure_places_remain(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.5, could_contain=lambda signature: 0.1))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_depth=1, beam_width=1),
    )

    # Assert
    assert result.outcome == Outcome.UNSURE_ONLY
    assert len(result.unsure) == 1
    assert [visit.verdict for visit in result.starts] == ["unsure"]


def test_search_stops_at_the_step_budget_and_lists_unopened_candidates(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.5))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_steps=2, beam_width=1),
    )

    # Assert
    assert result.outcome == Outcome.BUDGET
    assert result.steps == 2
    assert result.not_inspected
    assert "budget" in {entry.reason for entry in result.not_inspected}


def test_budget_receipt_does_not_start_unused_scope_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The structure scan is needed to expose real parser recovery. Calls and references are not
    # needed when this bounded search has no moves, and must remain explicitly pending.
    (tmp_path / "broken.js").write_text(
        "// @flow\n"
        "export opaque type Query = Object;\n"
        "export function find(query: Query): Query { return query; }\n"
    )
    (tmp_path / "unused.py").write_text("def unused():\n    return 1\n")
    index = CodeIndex(tmp_path, ["broken.js", "unused.py"], fact_cache_dir=tmp_path / "cache")
    from jev_navigator.index import code_index

    actual_scan = code_index.scan_facts
    scanned: list[tuple[str, ...]] = []

    def observe_scan(files, root, unparsed):
        scanned.append(tuple(files))
        return actual_scan(files, root, unparsed)

    monkeypatch.setattr(code_index, "scan_facts", observe_scan)
    index.functions_in("broken.js")
    start = range_place(index, "broken.js", 1, 3, "test start")
    client = ScriptedJevClient(nouls=scripted(found=lambda _: 0.05, could_contain=lambda _: 0.05))

    result = find_code(
        index,
        Judge(client),
        TARGET,
        [start],
        budget=SearchBudget(max_steps=1, beam_width=1),
        moves={},
    )

    assert result.outcome == Outcome.BUDGET
    assert result.unparsed_files == {"broken.js"}
    assert result.parser_scans_completed == ()
    assert result.parser_scans_pending == ("facts",)
    assert scanned == [("broken.js",)]


def test_a_beam_opens_several_places_per_round_as_separate_requests(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_steps=4, beam_width=3),
    )

    # Assert
    assert result.steps == 4
    assert len(client.requests) == 4
    opened_keys = [f"{state['slice']['file']}:{state['slice']['lines']}" for state, _ in client.requests]
    assert len(set(opened_keys)) == 4


def test_the_same_place_reached_twice_is_opened_once(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"none": 1.0}},
    )
    place = sample_index.enclosing_symbol("app/orders.py", 6)
    starts = [function_place(sample_index, place), place_for_line(sample_index, "app/orders.py", 7, "start")]

    # Act
    result = find_code(sample_index, Judge(client), TARGET, starts)

    # Assert
    assert result.steps == 1


def test_identical_code_under_two_places_is_judged_once(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"none": 1.0}},
    )
    same_code = CodeSlice(Span("app/orders.py", 5, 7, "place"), "def place(self, order): ...")
    starts = [
        Place("first", "function", "first", lambda: same_code),
        Place("second", "function", "second", lambda: same_code),
    ]

    # Act
    result = find_code(sample_index, Judge(client), TARGET, starts)

    # Assert
    assert result.steps == 1
    assert len(client.requests) == 1


def test_jev_sees_only_the_target_the_opened_code_and_neighbour_signatures(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))

    # Act
    find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    state, questions = client.requests[0]
    assert set(state) == {"target", "slice", "candidates"}
    assert all("goal" not in question["instructions"] for question in questions.values())


def test_find_packets_and_history_keep_parsed_relationship_bindings(tmp_path: Path) -> None:
    # Arrange
    repository_files = {
        "app/target.py": "def check():\n    return 'target'\n",
        "app/duplicate.py": "def check():\n    return 'duplicate'\n",
        "app/imported.py": "from app.target import check\n\n\ndef imported():\n    return check()\n",
        "app/ambiguous.py": "def ambiguous():\n    return check()\n",
    }
    commit_files(tmp_path, repository_files)
    index = CodeIndex(tmp_path, list(repository_files))
    starts = [function_place(index, index.find_definition(name)[0]) for name in ("imported", "ambiguous")]
    client = ScriptedJevClient(
        nouls=lambda _question_id, _question, _state: 0.1,
        choices={"open_first": {"none": 1.0}},
    )

    # Act
    result = find_code(
        index,
        Judge(client),
        "the check function",
        starts,
        moves={"callees": MOVES["callees"]},
        budget=SearchBudget(max_steps=2, beam_width=2),
    )

    # Assert
    packet_candidates = [
        candidate for state, _questions in client.requests for candidate in state["candidates"]
    ]
    packet_relationships = [candidate["relationship"] for candidate in packet_candidates]
    assert {item["binding"]["status"] for item in packet_relationships} == {"resolved", "candidate"}
    resolved = next(item for item in packet_relationships if item["binding"]["status"] == "resolved")
    ambiguous = next(item for item in packet_relationships if item["binding"]["status"] == "candidate")
    assert resolved["move"] == ambiguous["move"] == "callees"
    assert resolved["binding"]["target"]["file"] == "app/target.py"
    assert "target" not in ambiguous["binding"]

    history_candidates = [
        candidate
        for step in result.history.steps
        if step.operation == "open"
        for candidate in step.to_json()["judgments"]["could_contain"]
    ]
    assert {candidate["relationship"]["binding"]["status"] for candidate in history_candidates} == {
        "resolved",
        "candidate",
    }
    assert {candidate["probability"] for candidate in history_candidates} == {0.1}
    assert {candidate["verdict"] for candidate in history_candidates} == {"no"}

    imported_check = next(span for span in index.find_definition("check") if span.file == "app/target.py")
    caller_client = ScriptedJevClient(nouls=lambda _question_id, _question, _state: 0.1)
    caller_result = find_code(
        index,
        Judge(caller_client),
        "the check function",
        [function_place(index, imported_check)],
        moves={"callers": MOVES["callers"]},
        budget=SearchBudget(max_steps=1),
    )
    caller_relationships = [
        candidate["relationship"]
        for state, _questions in caller_client.requests
        for candidate in state["candidates"]
    ]
    assert {item["binding"]["status"] for item in caller_relationships} == {"resolved", "candidate"}
    assert {item["move"] for item in caller_relationships} == {"callers"}
    caller_open_step = next(step for step in caller_result.history.steps if step.operation == "open")
    assert {
        candidate["relationship"]["binding"]["status"]
        for candidate in caller_open_step.to_json()["judgments"]["could_contain"]
    } == {"resolved", "candidate"}


def test_beam_width_and_budgets_can_come_from_the_environment() -> None:
    # Act
    budget = SearchBudget.from_env({"JEV_NAVIGATOR_BEAM_WIDTH": "1", "JEV_NAVIGATOR_MAX_STEPS": "5"})

    # Assert
    assert budget == SearchBudget(beam_width=1, max_steps=5)


def test_empty_neighbours_are_never_offered(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))

    # Act
    find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    offered = [candidate["signature"] for candidate in client.requests[0][0]["candidates"]]
    assert offered and not any(signature.startswith("app/__init__.py") for signature in offered)


def test_found_code_carries_its_source_and_the_path_that_reached_it(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "len(order.items) <= limit" in code else 0.05,
            could_contain=lambda signature: (
                0.9 if "validate_order" in signature or "check_limits" in signature else 0.1
            ),
        )
    )

    # Act
    result = find_code(
        sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=SearchBudget(beam_width=1)
    )

    # Assert
    source = result.found[0].code.source()
    assert source["file"] == "app/validation.py" and source["commit"] == sample_index.commit
    assert source["reached_by"] == "called by validate_order"
    assert len(result.found[0].path) == 3


def test_a_stopped_search_resumes_from_its_frontier(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "len(order.items) <= limit" in code else 0.05,
            could_contain=lambda signature: (
                0.9 if "check_limits" in signature or "validate_order" in signature else 0.1
            ),
        )
    )
    judge = Judge(client)
    first = find_code(
        sample_index,
        judge,
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_steps=1, beam_width=1),
    )

    # Act
    resumed = find_code(
        sample_index, judge, TARGET, [], budget=SearchBudget(max_steps=5, beam_width=1), resume=first
    )

    # Assert
    assert first.outcome == Outcome.BUDGET
    assert resumed.outcome == Outcome.FOUND
    assert resumed.found[0].code.span.name == "check_limits"
    assert [visit.code.span.name for visit in resumed.starts].count("place") == 1


def test_capped_neighbours_are_reported_as_not_inspected(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(neighbours_per_kind=0),
    )

    # Assert
    assert any(entry.reason == "capped" for entry in result.not_inspected)


def test_a_requested_revision_that_the_index_does_not_hold_is_an_error(sample_index: CodeIndex) -> None:
    # Arrange
    import pytest

    from jev_navigator.index.code_index import RevisionMismatchError

    # Act and Assert
    with pytest.raises(RevisionMismatchError):
        find_code(
            sample_index, Judge(ScriptedJevClient()), TARGET, start_at_place(sample_index), commit="0" * 40
        )
    find_code(
        sample_index,
        Judge(ScriptedJevClient()),
        TARGET,
        start_at_place(sample_index),
        commit=sample_index.commit,
        budget=SearchBudget(max_steps=1),
    )


def test_the_search_wording_can_be_replaced(sample_index: CodeIndex) -> None:
    # Arrange
    from jev_navigator.directives.find_code import FOUND, SearchQuestions
    from jev_navigator.judgments.questions import Check, Criterion

    own = Check(
        "holds_rule",
        "Does `slice.code` enforce what `target.description` states?",
        Criterion("It enforces it."),
        Criterion("It does not."),
    )
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))

    # Act
    find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        questions=SearchQuestions(found=own, open_first=None),
    )

    # Assert
    asked = client.requests[0][1]
    assert own.question_id in asked and FOUND.question_id not in asked
    assert not any(question_id.startswith("open_first") for question_id in asked)


def test_a_search_counts_only_its_own_calls_when_another_search_shares_the_judge(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    nested_results = []
    answer = scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9)

    def answer_and_start_a_second_search(question_id: str, question: Mapping, state: Mapping) -> float:
        if not nested_results:
            nested_results.append(None)
            nested_results[0] = find_code(
                sample_index, judge, TARGET, start_at_place(sample_index), budget=SearchBudget(max_calls=3)
            )
        return answer(question_id, question, state)

    judge = Judge(ScriptedJevClient(nouls=answer_and_start_a_second_search))

    # Act
    outer = find_code(
        sample_index,
        judge,
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_calls=2, beam_width=1),
    )

    # Assert
    assert nested_results[0].calls == 3
    assert outer.calls == 2
    assert outer.steps == 2
    assert judge.calls == 5


def test_a_global_call_cap_on_the_judge_ends_a_search_as_budget(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))
    judge = Judge(client, max_calls=2)

    # Act
    result = find_code(
        sample_index, judge, TARGET, start_at_place(sample_index), budget=SearchBudget(beam_width=1)
    )

    # Assert
    assert result.outcome == Outcome.BUDGET
    assert result.calls == 2
    assert len(client.requests) == 2


def test_cached_search_answer_is_free_at_zero_live_call_budget(
    sample_index: CodeIndex, tmp_path: Path
) -> None:
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    first_client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1)
    )
    first = find_code(
        sample_index, Judge(first_client, store=store), TARGET, start_at_place(sample_index), moves={}
    )
    assert len(first_client.requests) == 1

    second_client = ScriptedJevClient()
    replayed = find_code(
        sample_index,
        Judge(
            second_client,
            store=JsonlAnswerStore(store.path),
            served_model=first_client.model,
            max_calls=0,
        ),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_calls=0),
        moves={},
    )

    assert replayed.outcome == first.outcome
    assert replayed.starts == first.starts
    assert replayed.calls == 0
    assert second_client.requests == []

    uncached_client = ScriptedJevClient()
    uncached = find_code(
        sample_index,
        Judge(uncached_client, max_calls=0),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_calls=0),
        moves={},
    )
    assert uncached.outcome == Outcome.BUDGET
    assert uncached.not_inspected[0].reason == "budget"
    assert uncached_client.requests == []


def test_find_code_with_no_moves_opens_only_its_start(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index), moves={})

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert result.steps == 1 and len(client.requests) == 1


def test_default_search_limits_are_unbounded_and_a_finite_frontier_terminates(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    budget = SearchBudget(beam_width=1)
    client = ScriptedJevClient(nouls=scripted(found=lambda _: 0.05, could_contain=lambda _: 0.9))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=budget,
        moves={},
    )

    # Assert
    assert (budget.max_depth, budget.max_steps, budget.max_calls, budget.neighbours_per_kind) == (
        None,
        None,
        None,
        None,
    )
    assert result.outcome == Outcome.NOTHING_LEFT
    assert result.not_inspected == ()


def test_an_unbounded_search_can_reach_beyond_the_old_default_depth(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "chain.txt").write_text("start\none\ntwo\nthree\ntarget\n")
    index = CodeIndex(tmp_path, ["chain.txt"])
    chain = [range_place(index, "chain.txt", line, line, "chain") for line in range(1, 6)]
    successor = {current.key: following for current, following in zip(chain[:-1], chain[1:], strict=True)}

    def next_in_chain(index: CodeIndex, opened: CodeSlice) -> list[Place]:
        del index
        following = successor.get(opened.key)
        return [following] if following is not None else []

    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if code == "target" else 0.05,
            could_contain=lambda _: 0.95,
        )
    )

    # Act
    result = find_code(
        index,
        Judge(client),
        "the target line",
        [chain[0]],
        budget=SearchBudget(beam_width=1),
        moves={"chain": next_in_chain},
    )

    # Assert
    assert result.outcome == Outcome.FOUND
    assert result.found[0].code.span.start == 5
    assert result.found[0].path == tuple(place.key for place in chain)


def test_caller_interrupt_returns_a_resumable_frontier(sample_index: CodeIndex) -> None:
    # Arrange
    class InterruptingClient:
        model = "interrupting"

        def ask(self, state, questions):
            del state, questions
            raise KeyboardInterrupt

    target = function_place(sample_index, sample_index.find_definition("check_limits")[0])

    # Act
    cancelled = find_code(
        sample_index,
        Judge(InterruptingClient()),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        moves={},
        initial_candidates=[(target, 1.0)],
    )
    resumed = find_code(
        sample_index,
        Judge(ScriptedJevClient(nouls={"contains_target": 0.95})),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        moves={},
        resume=cancelled,
    )

    # Assert
    assert cancelled.outcome == Outcome.CANCELLED
    assert cancelled.steps == 0
    assert [(entry.place_key, entry.reason) for entry in cancelled.not_inspected] == [
        (target.key, "cancelled")
    ]
    assert resumed.outcome == Outcome.FOUND
    assert resumed.found[0].code.span == target.open().span


def test_interrupt_while_opening_a_beam_restores_every_popped_place(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "places.txt").write_text("first\nsecond\n")
    index = CodeIndex(tmp_path, ["places.txt"])
    places = [range_place(index, "places.txt", line, line, "candidate") for line in (1, 2)]
    opened = 0

    def interrupt_second_open(index: CodeIndex, code: CodeSlice) -> list[Place]:
        del index, code
        nonlocal opened
        opened += 1
        if opened == 2:
            raise KeyboardInterrupt
        return []

    # Act
    cancelled = find_code(
        index,
        Judge(ScriptedJevClient()),
        TARGET,
        [],
        budget=SearchBudget(beam_width=2),
        moves={"interrupt": interrupt_second_open},
        initial_candidates=[(place, 1.0) for place in places],
    )

    # Assert
    assert cancelled.outcome == Outcome.CANCELLED
    assert cancelled.steps == 0
    assert {entry.place_key for entry in cancelled.not_inspected} == {place.key for place in places}
    assert {entry.reason for entry in cancelled.not_inspected} == {"cancelled"}


def test_interrupt_while_filtering_a_candidate_resumes_and_processes_it(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "places.txt").write_text("start\ntarget\n")
    index = CodeIndex(tmp_path, ["places.txt"])
    start = range_place(index, "places.txt", 1, 1, "candidate")
    real_candidate = range_place(index, "places.txt", 2, 2, "move")
    candidate_opens = 0

    def open_candidate() -> CodeSlice:
        nonlocal candidate_opens
        candidate_opens += 1
        if candidate_opens == 3:
            raise KeyboardInterrupt
        return real_candidate.open()

    candidate = Place(
        real_candidate.key,
        real_candidate.kind,
        real_candidate.signature,
        open_candidate,
    )

    def offer_candidate(index: CodeIndex, code: CodeSlice) -> list[Place]:
        del index, code
        return [candidate]

    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if code == "target" else 0.05,
            could_contain=lambda signature: 0.95,
        )
    )

    # Act
    cancelled = find_code(
        index,
        Judge(client),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        moves={"candidate": offer_candidate},
        initial_candidates=[(start, 1.0)],
    )
    resumed = find_code(
        index,
        Judge(client),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        moves={"candidate": offer_candidate},
        resume=cancelled,
    )

    # Assert
    assert cancelled.outcome == Outcome.CANCELLED
    assert [(entry.place_key, entry.reason) for entry in cancelled.not_inspected] == [
        (start.key, "cancelled")
    ]
    assert resumed.outcome == Outcome.FOUND
    assert resumed.found[0].place_key == candidate.key


def test_interrupt_while_popping_a_beam_restores_it_for_resume(
    sample_index: CodeIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    from jev_navigator.directives import find_code as find_code_module

    original = find_code_module._Search.next_beam
    interrupted = False

    def interrupt_after_pop(search, calls_left):
        nonlocal interrupted
        beam = original(search, calls_left)
        if not interrupted:
            interrupted = True
            os.kill(os.getpid(), signal.SIGINT)
        return beam

    monkeypatch.setattr(find_code_module._Search, "next_beam", interrupt_after_pop)
    target = function_place(sample_index, sample_index.find_definition("check_limits")[0])

    # Act
    cancelled = find_code(
        sample_index,
        Judge(ScriptedJevClient()),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        moves={},
        initial_candidates=[(target, 1.0)],
    )
    resumed = find_code(
        sample_index,
        Judge(ScriptedJevClient(nouls={"contains_target": 0.95})),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        moves={},
        resume=cancelled,
    )

    # Assert
    assert cancelled.outcome == Outcome.CANCELLED
    assert [(entry.place_key, entry.reason) for entry in cancelled.not_inspected] == [
        (target.key, "cancelled")
    ]
    assert resumed.outcome == Outcome.FOUND
    assert resumed.found[0].place_key == target.key


def test_interrupt_while_submitting_a_round_cancels_the_requests_already_sent(
    sample_index: CodeIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    from concurrent.futures import ThreadPoolExecutor

    from jev_navigator.directives import find_code as find_code_module

    class InterruptedOnSecondSubmit(ThreadPoolExecutor):
        submitted = 0

        def submit(self, *args, **kwargs):
            type(self).submitted += 1
            if type(self).submitted == 2:
                raise KeyboardInterrupt
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(find_code_module, "ThreadPoolExecutor", InterruptedOnSecondSubmit)
    places = [
        function_place(sample_index, sample_index.find_definition(name)[0])
        for name in ("check_limits", "validate_order")
    ]

    # Act
    cancelled = find_code(
        sample_index,
        Judge(ScriptedJevClient()),
        TARGET,
        [],
        budget=SearchBudget(beam_width=2),
        moves={},
        initial_candidates=[(place, 1.0) for place in places],
    )

    # Assert
    assert cancelled.outcome == Outcome.CANCELLED
    accounted = {entry.place_key for entry in cancelled.not_inspected}
    accounted |= {visit.place_key for visit in (*cancelled.found, *cancelled.searched, *cancelled.unsure)}
    assert accounted == {place.key for place in places}


def test_interrupt_while_recording_a_round_choice_restores_the_popped_place(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    class InterruptingRecorder:
        def __init__(self) -> None:
            self.interrupted = False

        def record_step(self, step) -> None:
            del step
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt

    target = function_place(sample_index, sample_index.find_definition("check_limits")[0])

    # Act
    cancelled = find_code(
        sample_index,
        Judge(ScriptedJevClient(), journal=InterruptingRecorder()),
        TARGET,
        [],
        budget=SearchBudget(beam_width=1),
        moves={},
        initial_candidates=[(target, 1.0)],
    )

    # Assert
    assert cancelled.outcome == Outcome.CANCELLED
    assert [(entry.place_key, entry.reason) for entry in cancelled.not_inspected] == [
        (target.key, "cancelled")
    ]


def test_cancellation_keeps_a_successful_response_from_the_same_beam(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "places.txt").write_text("first\nsecond\n")
    index = CodeIndex(tmp_path, ["places.txt"])
    places = [range_place(index, "places.txt", line, line, "candidate") for line in (1, 2)]

    class PartlyInterruptingClient:
        model = "partly-interrupting"

        def ask(self, state, questions):
            if state["slice"]["code"] == "second":
                raise KeyboardInterrupt
            return ScriptedJevClient(default_noul=0.05).ask(state, questions)

    # Act
    cancelled = find_code(
        index,
        Judge(PartlyInterruptingClient()),
        TARGET,
        [],
        budget=SearchBudget(beam_width=2),
        moves={},
        initial_candidates=[(place, 1.0) for place in places],
    )

    # Assert
    assert cancelled.outcome == Outcome.CANCELLED
    assert [visit.place_key for visit in cancelled.searched] == [places[0].key]
    assert [(entry.place_key, entry.reason) for entry in cancelled.not_inspected] == [
        (places[1].key, "cancelled")
    ]


def test_the_result_and_the_stop_step_name_the_moves_the_search_used(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))
    chosen = {"callers": MOVES["callers"], "same_file": MOVES["same_file"]}

    # Act
    chosen_result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index), moves=chosen)
    default_result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    assert chosen_result.moves == ("callers", "same_file")
    assert chosen_result.history.steps[-1].arguments["moves"] == ["callers", "same_file"]
    assert default_result.moves == tuple(MOVES)


def open_steps(result) -> list[dict]:
    return [step.to_json() for step in result.history.steps if step.operation == "open"]


def test_open_first_offers_a_real_none_option_whose_wording_is_part_of_the_question_id(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))
    tie_wording = replace(OPEN_FIRST, extra_options=(("none", "No entry is more likely than the others."),))

    # Act
    budget = SearchBudget(max_steps=1)
    find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=budget)

    # Assert
    questions = client.requests[0][1]
    pick = next(question for question_id, question in questions.items() if "open_first" in question_id)
    assert pick["criteria"]["none"] == "None of the entries is likely to contain it."
    assert tie_wording.question_id != OPEN_FIRST.question_id


@pytest.mark.parametrize("pick_score", [0.15, 0.35])
def test_a_pick_is_used_whatever_its_own_score(sample_index: CodeIndex, pick_score: float) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: pick_score),
        choices={"open_first": {"0": 0.3}},
    )

    # Act
    result = find_code(
        sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=SearchBudget(max_steps=2)
    )

    # Assert
    first_open = open_steps(result)[0]
    assert first_open["judgments"]["open_first"]["used"] is True
    reasons = [
        chosen["reason"]
        for step in result.history.steps
        if step.operation == "choose_next"
        for chosen in step.arguments["chosen"]
    ]
    assert "open_first" in reasons


def slot_of(index: CodeIndex, start: Place, name: str) -> int:
    candidates = [place for place in neighbours(index, start.open()) if place.open().text.strip()]
    return next(slot for slot, place in enumerate(candidates) if name in place.signature)


def opened_first_lines(client: ScriptedJevClient) -> list[str]:
    return [state["slice"]["code"].split("\n")[0].strip() for state, _ in client.requests]


def test_the_top_pick_opens_next_even_at_low_confidence_and_a_low_score(sample_index: CodeIndex) -> None:
    # Arrange
    start = start_at_place(sample_index)
    cancel = slot_of(sample_index, start[0], "cancel")
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.05, could_contain=lambda signature: 0.1 if "cancel" in signature else 0.9
        ),
        choices={"open_first": {str(cancel): 0.3}},
    )

    # Act
    find_code(sample_index, Judge(client), TARGET, start, budget=SearchBudget(max_steps=2, beam_width=1))

    # Assert
    assert opened_first_lines(client)[1] == "def cancel(order_id):"


def test_starts_open_before_any_pick_and_picks_open_in_the_order_made(sample_index: CodeIndex) -> None:
    # Arrange
    starts = [
        place_for_line(sample_index, "app/orders.py", 6, "start"),
        place_for_line(sample_index, "app/validation.py", 11, "start"),
    ]
    first_pick = slot_of(sample_index, starts[0], "cancel")
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {str(first_pick): 0.3}},
    )

    # Act
    result = find_code(
        sample_index, Judge(client), TARGET, starts, budget=SearchBudget(max_steps=3, beam_width=1)
    )

    # Assert
    opened = opened_first_lines(client)
    assert opened[:3] == ["def place(self, order):", "def check_limits(order):", "def cancel(order_id):"]
    reasons = [
        chosen["reason"]
        for step in result.history.steps
        if step.operation == "choose_next"
        for chosen in step.arguments["chosen"]
    ]
    assert reasons == ["start", "start", "open_first"]


def test_a_none_pick_queues_nothing(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"none": 1.0}},
    )

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert result.steps == 1


def test_a_waiting_pick_keeps_the_search_going_when_no_move_scores_above_the_no_bar(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"0": 0.3}},
    )

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_depth=1, beam_width=1),
    )

    # Assert
    assert result.steps == 2
    assert result.outcome == Outcome.NOTHING_LEFT


def test_a_start_judged_to_hold_the_target_is_recorded_but_never_ends_the_search(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "def place(self, order)" in code else 0.05,
            could_contain=lambda signature: 0.1,
        ),
        choices={"open_first": {"none": 1.0}},
    )

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert result.found == ()
    assert [(visit.code.span.name, visit.verdict) for visit in result.starts] == [("place", "yes")]


def test_a_resumed_search_opens_its_waiting_pick_first(sample_index: CodeIndex) -> None:
    # Arrange
    start = start_at_place(sample_index)
    cancel = slot_of(sample_index, start[0], "cancel")
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.05, could_contain=lambda signature: 0.1 if "cancel" in signature else 0.9
        ),
        choices={"open_first": {str(cancel): 0.3}},
    )
    judge = Judge(client)
    first = find_code(sample_index, judge, TARGET, start, budget=SearchBudget(max_steps=1, beam_width=1))

    # Act
    find_code(sample_index, judge, TARGET, [], budget=SearchBudget(max_steps=1, beam_width=1), resume=first)

    # Assert
    assert opened_first_lines(client)[1] == "def cancel(order_id):"


def long_function_index(root: Path) -> CodeIndex:
    """``handle`` is about 16,000 characters: a call at its top, filler lines, then a nested helper."""
    filler = "".join(
        f"    step_{number} = 'a line of filler text that pads the function out to length'\n"
        for number in range(200)
    )
    parameters = ", ".join(f"option_{number}=None" for number in range(30))
    handle = (
        f"def handle(event):\n    helper(event)\n{filler}"
        "    def helper(event):\n        return event\n\n    return event\n"
    )
    commit_files(root, {"handler.py": handle + f"\n\ndef configure({parameters}):\n    return None\n"})
    return CodeIndex(root, ["handler.py"])


def test_a_slice_longer_than_the_limit_is_cut_on_a_line_boundary(tmp_path: Path) -> None:
    # Arrange
    index = long_function_index(tmp_path)
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"none": 1.0}},
    )

    # Act
    result = find_code(index, Judge(client), TARGET, [place_for_line(index, "handler.py", 2, "start")])

    # Assert
    shown = client.requests[0][0]["slice"]
    last_line = int(shown["lines"].split("-")[1])
    assert len(shown["code"]) <= 12_000 + 80
    assert shown["code"].endswith("lines at 12000 characters]")
    assert last_line < index.find_definition("handle")[0].end
    assert result.starts[0].code.span.end == last_line


def test_a_place_inside_the_cut_off_tail_is_still_offered(tmp_path: Path) -> None:
    # Arrange
    index = long_function_index(tmp_path)
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"none": 1.0}},
    )

    # Act
    find_code(index, Judge(client), TARGET, [place_for_line(index, "handler.py", 2, "start")])

    # Assert
    offered = [candidate["signature"] for candidate in client.requests[0][0]["candidates"]]
    assert any("`def helper(event):`" in signature for signature in offered)


def test_a_long_signature_is_cut_the_same_way_in_the_candidate_and_the_pick_options(tmp_path: Path) -> None:
    # Arrange
    index = long_function_index(tmp_path)
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1),
        choices={"open_first": {"none": 1.0}},
    )

    # Act
    find_code(index, Judge(client), TARGET, [place_for_line(index, "handler.py", 2, "start")])

    # Assert
    state, questions = client.requests[0]
    signatures = [candidate["signature"] for candidate in state["candidates"]]
    pick = next(question for question_id, question in questions.items() if "open_first" in question_id)
    configure = next(signature for signature in signatures if "def configure(" in signature)
    assert len(configure) == 240 + len(" [line cut]") and configure.endswith(" [line cut]")
    assert [pick["criteria"][str(slot)] for slot in range(len(signatures))] == signatures
