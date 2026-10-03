"""The small directives: comment context, similar code, and exporting a masked request for review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_navigator.directives.comment_context import context_for_comment
from jev_navigator.directives.similar import find_similar_code
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.review import CapturingJevClient, export_for_review
from jev_navigator.testing import ScriptedJevClient


def test_context_for_comment_describes_the_target_with_the_comment_text(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=lambda question_id, question, state: 0.1)

    # Act
    context_for_comment(sample_index, Judge(client), "app/orders.py", 11)

    # Assert
    assert "Cancels an order that has not shipped." in client.requests[0][0]["target"]["description"]


def test_context_for_comment_searches_with_the_moves_the_caller_chose(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=lambda question_id, question, state: 0.1)

    # Act
    result = context_for_comment(sample_index, Judge(client), "app/orders.py", 11, moves={})

    # Assert
    assert result.moves == ()
    assert all(state["candidates"] == [] for state, _ in client.requests)


def test_find_similar_code_judges_each_candidate_against_the_subject(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls={"same_behaviour": 0.5})

    # Act
    similar = find_similar_code(sample_index, Judge(client), "handleOrder")

    # Assert
    assert similar.subject.name == "handleOrder"
    assert similar.same == () and len(similar.unsure) >= 1
    assert "subject" in client.requests[0][0]


def test_export_for_review_writes_the_masked_request_with_intended_uses(
    sample_index: CodeIndex, tmp_path: Path
) -> None:
    # Arrange
    capture = CapturingJevClient()
    context_for_comment(sample_index, Judge(capture), "app/comments.py", 3)
    state, questions = capture.requests[0]
    uses = {question_id: "stop when yes; open neighbours above the no cut" for question_id in questions}

    # Act
    path = export_for_review(state, questions, uses, tmp_path / "candidate.json", case_id="c1", group_id="g")

    # Assert
    candidate = json.loads(path.read_text())
    assert candidate["request"]["model"] == "jev-latest"
    assert "ghp_" not in path.read_text()
    assert set(candidate["intended_uses"]) == set(questions)


def test_export_for_review_refuses_a_question_without_an_intended_use(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="intended use"):
        export_for_review({}, {"q": {"type": "noul"}}, {}, tmp_path / "c.json", case_id="c", group_id="g")


def test_the_capturing_client_answers_each_primitive_in_its_own_shape() -> None:
    # Arrange
    from jev_navigator.judgments.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer

    questions = {
        "yes_no": {"type": "noul", "instructions": "Is it?"},
        "which": {"type": "choice", "instructions": "Which?", "criteria": {"a": "", "b": ""}},
        "how_much": {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]},
    }

    # Act
    response = CapturingJevClient().ask({}, questions)

    # Assert
    assert isinstance(response.answers["yes_no"], NoulAnswer)
    assert isinstance(response.answers["which"], ChoiceAnswer)
    assert isinstance(response.answers["how_much"], ScoreAnswer)
    assert response.score("how_much").probabilities == pytest.approx({"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})


def test_export_for_review_keeps_the_request_in_the_order_it_is_sent(tmp_path: Path) -> None:
    # Arrange
    state = {"target": {"description": "the limit check"}, "slice": {"code": "x = 1"}, "candidates": []}
    questions = {"b_question": {"type": "noul"}, "a_question": {"type": "noul"}}
    uses = {"b_question": "open when yes", "a_question": "stop when yes"}

    # Act
    path = export_for_review(state, questions, uses, tmp_path / "candidate.json", case_id="c1", group_id="g")

    # Assert
    request = json.loads(path.read_text())["request"]
    assert list(request["state"]) == ["target", "slice", "candidates"]
    assert list(request["questions"]) == ["b_question", "a_question"]
