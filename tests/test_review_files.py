"""The committed review files are the frozen contract of the question wording. The current templates,
rendered over each committed file's own state through the real judge, must ask exactly what was
reviewed. The code sampled into that state may change; the wording, the intended uses and the
target may not."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from jev_navigator.directives.find_code import COULD_CONTAIN, FOUND, OPEN_FIRST
from jev_navigator.judgments.templates import target_state

BUILDER_PATH = Path(__file__).parents[1] / "question-templates" / "build_candidates.py"


def load_builder() -> ModuleType:
    """The review-file builder script, imported as the module its dataclasses need it to be."""
    spec = importlib.util.spec_from_file_location("build_candidates", BUILDER_PATH)
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    return builder


BUILDER = load_builder()
FIND_REVIEW_SETS = [review_set for review_set in BUILDER.CHECK_REVIEW_SETS if review_set.request is not None]


def review_set_id(review_set) -> str:
    return f"{review_set.name}/{review_set.revision}"


def reviewed(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.mark.parametrize("review_set", BUILDER.CHECK_REVIEW_SETS, ids=review_set_id)
def test_the_templates_ask_each_reviewed_file_word_for_word_over_its_own_state(review_set) -> None:
    # Arrange
    candidate = reviewed(review_set.path)
    state = candidate["request"]["state"]
    shared = {name: value for name, value in state.items() if name != "items"}

    # Act
    rendered_state, questions, uses = review_set.render(state["items"], shared)

    # Assert
    assert questions == candidate["request"]["questions"]
    assert uses == candidate["intended_uses"]
    assert rendered_state == state


@pytest.mark.parametrize("review_set", FIND_REVIEW_SETS, ids=review_set_id)
def test_each_reviewed_find_file_holds_the_target_its_request_builds(review_set) -> None:
    # Act
    shared = target_state(review_set.request)

    # Assert
    assert reviewed(review_set.path)["request"]["state"]["target"] == shared["target"]


def test_the_reviewed_old_find_file_asks_the_current_search_wording() -> None:
    # Arrange
    candidate = reviewed(BUILDER.HERE / "find_code" / BUILDER.SEARCH_REVISION / "candidate.json")
    questions = candidate["request"]["questions"]

    # Act
    templates = {question_id.split("#")[0] for question_id in questions}

    # Assert
    assert templates == {FOUND.question_id, COULD_CONTAIN.question_id, OPEN_FIRST.question_id}
    assert candidate["intended_uses"] == {
        question_id: BUILDER.SEARCH_USES[question_id.split("@")[0]] for question_id in questions
    }
