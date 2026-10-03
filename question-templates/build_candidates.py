"""Writes one Meta Builder review file per question set, using this repository's own code.

Every anchor is a symbol, so an edit elsewhere in a file never moves it. ``export_for_review``
refuses to replace a revision's file with a different request: a changed request is written as a
new revision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from jev_navigator.directives.find_code import SearchBudget, find_code
from jev_navigator.directives.places import function_place
from jev_navigator.directives.similar import SAME_BEHAVIOUR
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check
from jev_navigator.judgments.review import CapturingJevClient, export_for_review
from jev_navigator.judgments.templates import (
    BEHAVIOR_ROLE_QUESTIONS,
    BEST_FEW_QUESTIONS,
    MATCH,
    target_state,
    unit_entry,
)

HERE = Path(__file__).parent
SECRET_REFUSAL = "the check that refuses to send a request that still contains a secret"

SEARCH_REVISION = "v4"
SEARCH_START = "mask_request"
SEARCH_USES = {
    "contains_target": (
        "Yes (0.80 or more) ends the search and reports this code as found; unsure keeps it as an"
        " unsure place; no records it as searched without finding the target."
    ),
    "could_contain_target": "The probability orders this neighbour in the best-first queue. A low value only"
    " lowers its priority; the neighbour is never discarded and stays in the result as not inspected.",
    "open_first": (
        "When confidence is 0.70 or more and the answer is not 'none', that neighbour is opened first."
    ),
}

SAME_BEHAVIOUR_SUBJECT = "mask_by_content"
SAME_BEHAVIOUR_CANDIDATES = ("mask_request", "mask_everywhere", "safe_options")
SAME_BEHAVIOUR_USE = (
    "Yes lists the candidate as duplicated behaviour; unsure is listed separately; no is dropped from"
    " the duplication candidates."
)

FIND_V2_SCOPE = "src/jev_navigator/judgments/"
FIND_V2_UNITS_PER_REQUEST = 16
FIND_V2_REQUEST = {
    "command": "find",
    "scope": {"repo": ".", "include": [FIND_V2_SCOPE]},
    "behavior": SECRET_REFUSAL,
}
FIND_V2_CONDITIONS = {"secret_masking": "off"}
FIND_V2_TARGET = "refuse_if_secret"
FIND_V2_BEST_FEW = ("refuse_if_secret", "_prepare", "mask_request")
ROLE_USE = "Labels the unit's role in the returned group. Role answers never change which units are expanded."
FIND_V2_USES = {
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
BEST_FEW_USES = {
    **FIND_V2_USES,
    "match": (
        "Asked again of each of the best few in their shared request, beside the role questions. Code keeps"
        " this score next to the unit's score from its 16-unit pass; the evaluation decides which of the two"
        " the Confirmed rule reads."
    ),
}


@dataclass(frozen=True)
class ReviewSet:
    """A review file built from Check templates: the questions it asks of each unit, what code does
    with each answer, and where the file lives. ``request`` is the Find request whose ``target`` the
    file's shared state holds, when it has one."""

    name: str
    revision: str
    checks: tuple[Check, ...]
    uses_by_template: Mapping[str, str]
    case: str
    group: str
    request: Mapping | None = None

    @property
    def path(self) -> Path:
        return HERE / self.name / self.revision / "candidate.json"

    def render(self, items: Sequence[Mapping], shared: Mapping) -> tuple[Mapping, Mapping, dict[str, str]]:
        """The state and questions the real judge sends for ``items`` and ``shared``, and each
        question's intended use."""
        capture = CapturingJevClient()
        Judge(capture).check_every(list(self.checks), list(items), shared)
        [(state, questions)] = capture.requests
        uses = {question_id: self.uses_by_template[question_id.split("@")[0]] for question_id in questions}
        return state, questions, uses

    def write(self, items: Sequence[Mapping], shared: Mapping) -> None:
        state, questions, uses = self.render(items, shared)
        write_candidate(state, questions, uses, self.name, self.revision, case=self.case, group=self.group)


SAME_BEHAVIOUR_SET = ReviewSet(
    "same_behaviour",
    "v4",
    (SAME_BEHAVIOUR,),
    {SAME_BEHAVIOUR.name: SAME_BEHAVIOUR_USE},
    case="secrets-module",
    group="same_behaviour",
)
FIND_V2_MATCH_SET = ReviewSet(
    "find_v2_match",
    "v3",
    (MATCH,),
    FIND_V2_USES,
    case="secret-refusal",
    group="find-v2",
    request=FIND_V2_REQUEST,
)
FIND_V2_MATCH_CONDITIONS_SET = ReviewSet(
    "find_v2_match_conditions",
    "v3",
    (MATCH,),
    FIND_V2_USES,
    case="secret-refusal",
    group="find-v2",
    request={**FIND_V2_REQUEST, "conditions": FIND_V2_CONDITIONS},
)
FIND_V2_ROLES_SET = ReviewSet(
    "find_v2_roles",
    "v3",
    tuple(BEHAVIOR_ROLE_QUESTIONS.values()),
    FIND_V2_USES,
    case="secret-refusal",
    group="find-v2",
    request=FIND_V2_REQUEST,
)
FIND_V2_BEST_FEW_SET = ReviewSet(
    "find_v2_best_few",
    "v1",
    BEST_FEW_QUESTIONS,
    BEST_FEW_USES,
    case="secret-refusal",
    group="find-v2",
    request=FIND_V2_REQUEST,
)
CHECK_REVIEW_SETS = (
    SAME_BEHAVIOUR_SET,
    FIND_V2_MATCH_SET,
    FIND_V2_MATCH_CONDITIONS_SET,
    FIND_V2_ROLES_SET,
    FIND_V2_BEST_FEW_SET,
)


def main() -> None:
    index = CodeIndex.from_git(HERE.parent, prefixes=("src/",))
    write_search_candidate(index)
    write_same_behaviour_candidate(index)
    write_find_v2_candidates(index)


def definition(index: CodeIndex, symbol: str) -> Span:
    """The one definition of ``symbol``; an anchor that names none or several stops the build."""
    spans = index.find_definition(symbol)
    if len(spans) != 1:
        raise ValueError(f"the anchor {symbol!r} has {len(spans)} definitions; it must name exactly one")
    return spans[0]


def write_search_candidate(index: CodeIndex) -> None:
    capture = CapturingJevClient()
    start = [function_place(index, definition(index, SEARCH_START), "start")]
    find_code(index, Judge(capture), SECRET_REFUSAL, start, budget=SearchBudget(max_steps=1, beam_width=1))
    state, questions = capture.requests[0]
    uses = {question_id: SEARCH_USES[question_id.split("@")[0]] for question_id in questions}
    write_candidate(
        state, questions, uses, "find_code", SEARCH_REVISION, case="secret-refusal", group="find-code"
    )


def write_same_behaviour_candidate(index: CodeIndex) -> None:
    subject = {"subject": {"code": index.read_slice(definition(index, SAME_BEHAVIOUR_SUBJECT)).text}}
    items = [unit_entry(index, definition(index, symbol)) for symbol in SAME_BEHAVIOUR_CANDIDATES]
    SAME_BEHAVIOUR_SET.write(items, subject)


def write_find_v2_candidates(index: CodeIndex) -> None:
    """Round 0's match request for the batch of 16 that holds the target, with and without a
    condition, the role request for three best-few units, and their shared request that asks the
    match question again beside the roles."""
    batch = round_zero_batch(index)
    best_few = sorted((definition(index, symbol) for symbol in FIND_V2_BEST_FEW), key=lambda span: span.key)
    sets_and_units = (
        (FIND_V2_MATCH_SET, batch),
        (FIND_V2_MATCH_CONDITIONS_SET, batch),
        (FIND_V2_ROLES_SET, best_few),
        (FIND_V2_BEST_FEW_SET, best_few),
    )
    for review_set, spans in sets_and_units:
        review_set.write([unit_entry(index, span) for span in spans], target_state(review_set.request))


def round_zero_batch(index: CodeIndex) -> list[Span]:
    """The batch of 16 that holds the target, cut as round 0 cuts the scope's units."""
    in_scope = [file for file in index.files if file.startswith(FIND_V2_SCOPE)]
    units = sorted(index.functions_in_files(in_scope), key=lambda span: span.key)
    size = FIND_V2_UNITS_PER_REQUEST
    batches = [units[start : start + size] for start in range(0, len(units), size)]
    return next(batch for batch in batches if any(span.name == FIND_V2_TARGET for span in batch))


def write_candidate(
    state: Mapping, questions: Mapping, uses: Mapping, name: str, revision: str, *, case: str, group: str
) -> None:
    export_for_review(
        state,
        questions,
        uses,
        HERE / name / revision / "candidate.json",
        case_id=f"{name}:{case}",
        group_id=f"jev-navigator-{group}",
        revision_id=revision,
    )


if __name__ == "__main__":
    main()
