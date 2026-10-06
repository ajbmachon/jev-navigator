"""The code a comment talks about, found with find_code starting under the comment."""

from __future__ import annotations

from collections.abc import Mapping

from .. import operations
from ..index.code_index import CodeIndex
from ..judgments.judge import Judge
from ..sources import Source
from .find_code import FindResult, SearchBudget, find_code
from .places import function_place


def context_for_comment(
    index: CodeIndex,
    judge: Judge,
    file: str,
    line: int,
    *,
    budget: SearchBudget | None = None,
    moves: Mapping[str, Source] | None = None,
) -> FindResult:
    """The code a comment talks about: starts from the whole symbol after the comment and searches
    outward only if that code does not contain what the comment describes. ``moves`` chooses how
    neighbours are listed, as in ``find_code``."""
    comment = index.read_window(file, line, radius=0).text.strip()
    described = operations.code_described_by_comment(index, file, line)
    start = function_place(index, described.span, "the code under the comment")
    description = f"the code this comment describes: {comment}"
    return find_code(index, judge, description, [start], budget=budget, moves=moves)
