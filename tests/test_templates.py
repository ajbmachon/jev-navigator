"""Find v2 question templates, rendered through the real judge over the fixture repository."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check
from jev_navigator.judgments.rebuild import rebuild_request
from jev_navigator.judgments.review import CapturingJevClient
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.templates import (
    BEHAVIOR_ROLE_QUESTIONS,
    MATCH,
    TARGET,
    TEMPLATES,
    target_state,
    unit_entry,
)
from jev_navigator.testing import ScriptedJevClient

REQUEST = {
    "command": "find",
    "scope": {"repo": "/withheld/scope-root", "include": ["app/"]},
    "anchors": [{"symbol": "withheld_anchor_symbol"}],
    "behavior": "the check that limits how many items an order may hold",
    "want": ["consumes"],
    "budget": {"max_functions": 4321, "max_rounds": 2},
    "include_code": False,
    "keep_requests": False,
}
CONDITIONS = {"deployment_zone": "zone-7f3a"}
BEHAVIOR_ROLES = {"performs", "hands_off", "selects_or_configures", "consumes"}
BACKTICKED_PATH = re.compile(r"`([A-Za-z_]\w*(?:\[\d+\])?(?:\.[A-Za-z_]\w*(?:\[\d+\])?)*)`")
PATH_STEP = re.compile(r"([A-Za-z_]\w*)|\[(\d+)\]")


def fixture_spans(index: CodeIndex) -> list[Span]:
    """Every function of the fixture, sorted by unit id as round 0 cuts its batches."""
    return sorted(index.functions_in_files(index.files), key=lambda span: span.key)


def rendered(
    index: CodeIndex, checks: Sequence[Check], request: Mapping, units: int | None = None
) -> list[tuple[Mapping, Mapping]]:
    capture = CapturingJevClient()
    items = [unit_entry(index, span) for span in fixture_spans(index)[:units]]
    Judge(capture).check_every(list(checks), items, target_state(request))
    return capture.requests


def named_paths(question: Mapping) -> list[str]:
    return BACKTICKED_PATH.findall(json.dumps(question))


def resolve(state: Mapping, path: str) -> object:
    value: object = state
    for name, position in PATH_STEP.findall(path):
        value = value[name] if name else value[int(position)]
    return value


def test_each_template_id_changes_when_its_wording_changes() -> None:
    # Arrange
    ids = [template.question_id for template in TEMPLATES]

    # Act
    edited = {
        template.question_id: [
            replace(template, instructions=template.instructions + " "),
            replace(template, yes=replace(template.yes, what=template.yes.what + " ")),
            replace(template, no=replace(template.no, what=template.no.what + " ")),
        ]
        for template in TEMPLATES
    }

    # Assert
    assert len(set(ids)) == len(ids) == 1 + len(BEHAVIOR_ROLE_QUESTIONS)
    assert all(
        variant.question_id != original for original, variants in edited.items() for variant in variants
    )


def test_only_the_behaviour_its_conditions_and_each_units_location_and_code_reach_jev(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    request = {**REQUEST, "conditions": CONDITIONS}

    # Act
    requests = rendered(sample_index, [MATCH, *BEHAVIOR_ROLE_QUESTIONS.values()], request, units=3)

    # Assert
    for state, questions in requests:
        assert {key for key in state if key != "items"} == {TARGET}
        assert state[TARGET] == {"behavior": request["behavior"], "conditions": CONDITIONS}
        assert all(set(item) == {"file", "lines", "code"} for item in state["items"])
        sent = json.dumps({"state": state, "questions": questions})
        assert not any(
            withheld in sent for withheld in ("/withheld/scope-root", "withheld_anchor_symbol", "4321")
        )


def test_conditions_reach_the_state_as_data_not_as_wording(sample_index: CodeIndex) -> None:
    # Arrange
    request = {**REQUEST, "conditions": CONDITIONS}

    # Act
    requests = rendered(sample_index, [MATCH, *BEHAVIOR_ROLE_QUESTIONS.values()], request, units=3)

    # Assert
    for state, questions in requests:
        assert state[TARGET]["conditions"] == CONDITIONS
        wording = json.dumps(questions)
        assert "deployment_zone" not in wording and "zone-7f3a" not in wording


@pytest.mark.parametrize("conditions", [None, {}])
def test_a_request_without_conditions_sends_no_conditions_field(
    sample_index: CodeIndex, conditions: dict | None
) -> None:
    # Arrange
    request = REQUEST if conditions is None else {**REQUEST, "conditions": conditions}

    # Act
    requests = rendered(sample_index, [MATCH], request, units=2)

    # Assert
    assert [state[TARGET] for state, _ in requests] == [{"behavior": REQUEST["behavior"]}]


def test_role_questions_are_four_independent_nouls_per_unit_in_one_request(sample_index: CodeIndex) -> None:
    # Act
    requests = rendered(sample_index, list(BEHAVIOR_ROLE_QUESTIONS.values()), REQUEST, units=3)

    # Assert
    assert len(requests) == 1
    _, questions = requests[0]
    asked = [(question_id.split("@")[0], question_id.rsplit("#", 1)[1]) for question_id in questions]
    roles_per_unit = {slot: {role for role, asked_slot in asked if asked_slot == slot} for _, slot in asked}
    assert {slot: len(roles) for slot, roles in roles_per_unit.items()} == {"0": 4, "1": 4, "2": 4}
    assert len(asked) == 12
    assert set(BEHAVIOR_ROLE_QUESTIONS) == BEHAVIOR_ROLES
    assert all(roles == BEHAVIOR_ROLES for roles in roles_per_unit.values())
    assert {question["type"] for question in questions.values()} == {"noul"}


def test_each_per_unit_question_names_its_own_unit_and_no_other(sample_index: CodeIndex) -> None:
    # Act
    requests = rendered(sample_index, [MATCH, *BEHAVIOR_ROLE_QUESTIONS.values()], REQUEST)

    # Assert
    for _, questions in requests:
        for question_id, question in questions.items():
            slot = question_id.rsplit("#", 1)[1]
            units_named = {path.split(".")[0] for path in named_paths(question) if path.startswith("items[")}
            assert units_named == {f"items[{slot}]"}


@pytest.mark.parametrize("conditions", [None, CONDITIONS])
def test_every_state_path_a_rendered_question_names_resolves_in_its_state(
    sample_index: CodeIndex, conditions: dict | None
) -> None:
    # Arrange
    request = REQUEST if conditions is None else {**REQUEST, "conditions": conditions}

    # Act
    requests = rendered(sample_index, [MATCH, *BEHAVIOR_ROLE_QUESTIONS.values()], request)

    # Assert
    paths = [
        (state, path) for state, questions in requests for q in questions.values() for path in named_paths(q)
    ]
    assert paths
    for state, path in paths:
        resolve(state, path)


def test_a_stored_match_answer_rebuilds_exactly_from_the_repository(
    sample_index: CodeIndex, tmp_path: Path
) -> None:
    # Arrange
    store_path = tmp_path / "answers.jsonl"
    items = [unit_entry(sample_index, span) for span in fixture_spans(sample_index)[:3]]
    shared = target_state({**REQUEST, "conditions": CONDITIONS})
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(store_path)).check_every([MATCH], items, shared)
    record = JsonlAnswerStore(store_path).records()[0]

    # Act
    rebuilt = rebuild_request(record, sample_index, shared)

    # Assert
    assert rebuilt.matches, rebuilt.differences


def test_a_request_without_a_behaviour_cannot_render_a_match_question() -> None:
    # Arrange
    trace_by_relation = {"command": "trace", "anchors": [{"symbol": "handle"}], "relation": "calls"}

    # Act and assert
    with pytest.raises(ValueError, match="behavior"):
        target_state(trace_by_relation)
