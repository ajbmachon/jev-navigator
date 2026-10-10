"""Code that does the same thing as a given function: code lists candidates, Jev judges each one."""

from __future__ import annotations

from dataclasses import dataclass

from .. import operations
from ..index.code_index import CodeIndex
from ..index.spans import Span
from ..judgments.judge import CheckResult, Judge
from ..judgments.known_values import with_known_values
from ..judgments.questions import Check, Criterion
from ..judgments.thresholds import NoulVerdict

SAME_BEHAVIOUR = Check(
    name="same_behaviour",
    instructions="Does `{item}.code` do the same thing as `subject.code`?",
    yes=Criterion(
        "Both compute the same result from the same kind of input, even if written differently.",
        examples=(
            "Two functions that each sum item prices and apply the same 10 percent discount above 100.",
        ),
    ),
    no=Criterion(
        "They share names or helpers but do different work.",
        examples=("`parse_order` and `parse_refund` both call `json.loads` but build different records.",),
    ),
)


@dataclass(frozen=True)
class SimilarCode:
    subject: Span
    same: tuple[CheckResult, ...]
    unsure: tuple[CheckResult, ...]


def find_similar_code(
    index: CodeIndex, judge: Judge, symbol: str, *, check: Check = SAME_BEHAVIOUR
) -> SimilarCode | None:
    """Code lists the candidates; Jev judges each one against the subject separately."""
    judge = with_known_values(judge, index)
    definitions = index.find_definition(symbol)
    if not definitions:
        return None
    subject = definitions[0]
    candidates = operations.similar_functions(index, symbol)
    items = [{"file": span.file, "code": index.read_slice(span).text} for span in candidates]
    shared = {"subject": {"code": index.read_slice(subject).text}}
    checks = judge.check_each(check, items, shared) if items else []
    return SimilarCode(
        subject,
        tuple(check for check in checks if check.verdict == NoulVerdict.YES),
        tuple(check for check in checks if check.verdict == NoulVerdict.UNSURE),
    )
