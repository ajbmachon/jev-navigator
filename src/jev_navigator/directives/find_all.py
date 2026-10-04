"""Seed-first semantic enumeration: expand relationships, judge, then search outside them.

The host obtains seeds through find_code or symbol lookup. This composition examines function
bodies. Static reachability prioritizes candidates; it never proves their semantic relevance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from .. import operations
from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.spans import Span
from ..judgments.judge import CallCapReachedError, CheckResult, Judge, unit_place
from ..judgments.questions import Check
from ..judgments.thresholds import NoulVerdict
from .find_code import FOUND

CONTAINS_IMPLEMENTATION = replace(
    FOUND,
    instructions=FOUND.instructions.replace("slice.code", "{item}.code"),
    yes=replace(FOUND.yes, what=FOUND.yes.what.replace("slice.code", "{item}.code")),
    no=replace(FOUND.no, what=FOUND.no.what.replace("slice.code", "{item}.code")),
)


@dataclass(frozen=True)
class FindAllResult:
    target: str
    graph: operations.TraceGraph
    judged: tuple[CheckResult, ...]
    remaining_files: tuple[str, ...]
    unparsed_files: frozenset[str]
    unsupported_files: tuple[str, ...]
    unavailable_files: Mapping[str, str]
    stopped_by: str
    calls: int

    @property
    def matched(self) -> tuple[CheckResult, ...]:
        return tuple(answer for answer in self.judged if answer.verdict == NoulVerdict.YES)

    @property
    def uncertain(self) -> tuple[CheckResult, ...]:
        return tuple(answer for answer in self.judged if answer.verdict == NoulVerdict.UNSURE)

    @property
    def negative(self) -> tuple[CheckResult, ...]:
        return tuple(answer for answer in self.judged if answer.verdict == NoulVerdict.NO)

    @property
    def coverage(self) -> str:
        """Examination coverage, never proof that the semantic answers are correct."""
        if self.stopped_by != "scope_examined" or self.remaining_files:
            return "partial"
        if self.unparsed_files or self.unsupported_files or self.unavailable_files:
            return "scope_incomplete"
        return "functions_examined"


def find_all(
    index: CodeIndex,
    judge: Judge,
    target: str,
    seeds: Sequence[Span],
    *,
    include_disconnected: bool = True,
    completed: Sequence[CheckResult] = (),
    check: Check = CONTAINS_IMPLEMENTATION,
    cancelled: Callable[[], bool] | None = None,
) -> FindAllResult:
    """Expand from concrete seeds, batch a property check, then examine remaining functions.

    The judged functions are listed in file and line order, however their answers arrive, so the
    same run gives the same result. Seeds are candidate locations, not assumed matches. Candidate
    and unresolved graph bindings remain unchanged. The fallback includes disconnected and
    differently named functions. An empty seed list runs just that fallback;
    include_disconnected=False reports partial coverage.

    ``completed`` retains answers from an interrupted enumeration of the same source, target,
    check and thresholds. The caller owns that identity check (the CLI validates its saved scope).
    Completed functions are not re-judged, including when remaining batches regroup on resume.

    Judge owns packing, caching, secrets and caller-selected live-call budgets. This composition
    has no file/function/call cap and does not truncate bodies. Provider errors propagate with
    their cause; the Judge journal retains the requests and responses actually made.
    """
    judge = judge.scope()
    graph = operations.trace_graph(index, seeds, cancelled=cancelled)
    judged = list(completed)
    seen = {answer.item["span_key"] for answer in completed}
    remaining_files = list(index.files)
    stop = "connected_component"
    inventoried: dict[str, set[str]] = {}

    def stopped() -> bool:
        return cancelled is not None and cancelled()

    def assess(spans: Sequence[Span]) -> None:
        fresh = {span.key: span for span in spans if span.key not in seen}
        items = []
        for span in fresh.values():
            code = index.read_slice(span, origin="findall")
            items.append({**code.source(), "span_key": span.key, "name": span.name, "code": code.text})
        for answer in judge.iter_check_each(check, items, {"target": {"description": target}}):
            judged.append(answer)
            seen.add(answer.item["span_key"])

    def inventory(files: Sequence[str]) -> tuple[Span, ...]:
        spans = tuple(index.functions_in_files(files))
        for file in files:
            inventoried[file] = set()
        for span in spans:
            inventoried[span.file].add(span.key)
        return spans

    if graph.stop == "cancelled" or stopped():
        stop = "cancelled"
    else:
        # Graphs may reach classes/declarations too. They remain in the graph; the answer unit is
        # a function body, so only function candidates are judged here.
        try:
            functions = set(inventory(tuple(dict.fromkeys(span.file for span in graph.functions))))
            assess([span for span in graph.functions if span in functions])
            if include_disconnected and not stopped():
                pending = inventory(index.available_files)
                if not stopped():
                    assess(pending)
                    stop = "scope_examined"
        except CallCapReachedError:
            stop = "budget"
        if stopped():
            stop = "cancelled"

    # Only facts already collected may establish that a file is finished. No final parser sweep.
    remaining_files = [
        file for file in remaining_files if file not in inventoried or not inventoried[file] <= seen
    ]

    return FindAllResult(
        target,
        graph,
        tuple(sorted(judged, key=_place_order)),
        tuple(remaining_files),
        index.observed_unparsed_files,
        tuple(file for file in index.files if not language_of(file)),
        index.unavailable_files,
        stop,
        judge.calls,
    )


def _place_order(result: CheckResult) -> tuple:
    """Judged functions in file and line order, whatever order their answers arrived in."""
    return (*unit_place(result.item), result.item["span_key"])
