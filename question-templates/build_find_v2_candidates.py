"""Writes the Meta Builder review files for the Find v2 templates, using this repository's own code.

Round 0 sends the match question for a batch of 16 units cut from units sorted by unit id; the best few
then get the four role questions in one request. Both requests are rendered by the real judge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check
from jev_navigator.judgments.review import CapturingJevClient, export_for_review
from jev_navigator.judgments.templates import BEHAVIOR_ROLE_QUESTIONS, MATCH, target_state, unit_entry

HERE = Path(__file__).parent
REVISION = "v1"
SCOPE = "src/jev_navigator/judgments/"
UNITS_PER_REQUEST = 16
REQUEST = {
    "command": "find",
    "scope": {"repo": ".", "include": [SCOPE]},
    "behavior": "the check that refuses to send a request that still contains a secret",
}
CONDITIONS = {"secret_masking": "off"}
TARGET_SYMBOL = "refuse_if_secret"
BEST_FEW_SYMBOLS = ("refuse_if_secret", "_prepare", "mask_request")
ROLE_USE = "Labels the unit's role in the returned group. Role answers never change which units are expanded."
USES = {
    "match": (
        "The probability ranks this unit against every unit judged in the search. The best few (3 to"
        " start) get the role questions and the literal pick, and their neighbours are judged in the next"
        " round. A best unit at or above the high-score bar (0.80 to start) is Confirmed; one clearly"
        " ahead of the next (by 0.15 to start) but under the bar gets a second look. Both values are"
        " evaluation settings."
    ),
    "performs": (
        f"{ROLE_USE} A group needs at least one unit at or above the yes threshold here; a second place"
        " is reported only when it holds such a unit."
    ),
    "hands_off": f"{ROLE_USE} Each hand-off is either followed in the group or named as a gap.",
    "selects_or_configures": ROLE_USE,
    "consumes": ROLE_USE,
}


def main() -> None:
    index = CodeIndex.from_git(HERE.parent, prefixes=(SCOPE,))
    batch = round_zero_batch(index)
    write_candidate(index, [MATCH], batch, REQUEST, "find_v2_match")
    write_candidate(index, [MATCH], batch, {**REQUEST, "conditions": CONDITIONS}, "find_v2_match_conditions")
    write_candidate(index, list(BEHAVIOR_ROLE_QUESTIONS.values()), best_few(index), REQUEST, "find_v2_roles")


def round_zero_batch(index: CodeIndex) -> list[Span]:
    """The batch of 16 that holds the target, cut as round 0 cuts the scope's units."""
    units = sorted(index.functions_in_files(index.files), key=lambda span: span.key)
    batches = [units[start : start + UNITS_PER_REQUEST] for start in range(0, len(units), UNITS_PER_REQUEST)]
    return next(batch for batch in batches if any(span.name == TARGET_SYMBOL for span in batch))


def best_few(index: CodeIndex) -> list[Span]:
    return sorted(
        (index.find_definition(symbol)[0] for symbol in BEST_FEW_SYMBOLS), key=lambda span: span.key
    )


def write_candidate(
    index: CodeIndex, checks: Sequence[Check], spans: Sequence[Span], request: Mapping, name: str
) -> None:
    capture = CapturingJevClient()
    Judge(capture).check_every(
        list(checks), [unit_entry(index, span) for span in spans], target_state(request)
    )
    [(state, questions)] = capture.requests
    export_for_review(
        state,
        questions,
        {question_id: USES[question_id.split("@")[0]] for question_id in questions},
        HERE / name / REVISION / "candidate.json",
        case_id=f"{name}:secret-refusal",
        group_id="jev-navigator-find-v2",
        revision_id=REVISION,
    )


if __name__ == "__main__":
    main()
