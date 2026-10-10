"""Explain a connected workflow with static links and narrow Jev evidence judgments.

The index owns connectivity. Jev only judges whether concrete, source-identified code contributes
evidence for the typed obligations, in batches owned by ``Judge``. A positive judgment never changes
a binding's static status.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .. import operations
from ..index.bindings import BindingStatus
from ..index.code_index import CodeIndex
from ..index.spans import Span
from ..judgments.judge import CallCapReachedError, CheckResult, Judge, Refusal
from ..judgments.known_values import with_known_values
from ..judgments.questions import Check, Criterion
from ..judgments.secrets import WORKFLOW
from ..judgments.thresholds import NoulVerdict


class EvidenceStatus(StrEnum):
    EVIDENCE_BACKED = "evidence_backed"
    GAP_TO_INVESTIGATE = "gap_to_investigate"
    UNRESOLVED = "unresolved"


INPUT_ORIGIN = Check(
    name="trace_input_origin",
    instructions=(
        "Does `{item}` contain concrete evidence of where an input, event, or request relevant to "
        "`workflow.question` enters the workflow?"
    ),
    yes=Criterion(
        "The supplied source reads, receives, decodes, or extracts the relevant input and shows its "
        "source. A linked registration line can establish the entry boundary."
    ),
    no=Criterion(
        "The supplied source neither receives the relevant input nor identifies its entry boundary."
    ),
)
TRANSFORMATION = Check(
    name="trace_transformation",
    instructions=(
        "Does `{item}` contain concrete evidence of a transformation relevant to `workflow.question`?"
    ),
    yes=Criterion(
        "The supplied source validates, normalizes, parses, computes, or otherwise changes the "
        "relevant input into a representation used by a later stage."
    ),
    no=Criterion(
        "The supplied source only receives, forwards, registers, or observes the value; it does not "
        "show the relevant transformation."
    ),
)
HANDOFF = Check(
    name="trace_handoff",
    instructions=(
        "Does `{item}` contain concrete evidence of the handoff that connects stages of "
        "`workflow.question`, including callback or handler registration when that is the handoff?"
    ),
    yes=Criterion(
        "The supplied source passes the relevant value or control to another stage, or explicitly "
        "registers the callback or handler that receives it."
    ),
    no=Criterion(
        "The supplied source shows an isolated stage but no relevant transfer, callback, or "
        "registration connecting it to another stage."
    ),
)
OBSERVABLE_OUTCOME = Check(
    name="trace_observable_outcome",
    instructions=(
        "Does `{item}` contain concrete evidence of the externally observable outcome requested by "
        "`workflow.question`?"
    ),
    yes=Criterion(
        "The supplied source returns, writes, emits, publishes, or otherwise produces the requested "
        "observable result."
    ),
    no=Criterion(
        "The supplied source only performs an intermediate step and does not show the requested "
        "observable result."
    ),
)
RELEVANT_BRANCH = Check(
    name="trace_relevant_branch",
    instructions=(
        "Does `{item}` contain concrete evidence of a condition or alternative path that materially "
        "changes `workflow.question`?"
    ),
    yes=Criterion(
        "The supplied source has a condition, match, exception path, or alternative callback whose "
        "choice changes the relevant transformation, handoff, or outcome."
    ),
    no=Criterion(
        "The supplied source has no branch relevant to the requested workflow; incidental control "
        "flow does not count."
    ),
)

TRACE_EVIDENCE_CHECKS = (
    INPUT_ORIGIN,
    TRANSFORMATION,
    HANDOFF,
    OBSERVABLE_OUTCOME,
    RELEVANT_BRANCH,
)


@dataclass(frozen=True)
class TraceObligation:
    """One obligation and its source-identified Jev evidence.

    ``evidence_backed`` means Jev supported at least one supplied source item. It is semantic model
    evidence, never a promotion of candidate or unresolved links to static proof. ``examined`` says
    whether every span of the walked component was judged for this obligation; when a budget stop or
    a refused request left spans unjudged, the obligation stays ``unresolved`` instead of becoming a gap.
    """

    name: str
    status: EvidenceStatus
    evidence: tuple[CheckResult, ...]
    unresolved: tuple[CheckResult, ...]
    checked: tuple[CheckResult, ...]
    examined: bool


@dataclass(frozen=True)
class TraceResult:
    """``refusals`` are the spans whose request was refused: they stay unjudged, so every obligation
    stays unexamined, and the trace goes on past them."""

    question: str
    graph: operations.TraceGraph
    included: tuple[Span, ...]
    excluded: tuple[Span, ...]
    obligations: tuple[TraceObligation, ...]
    unresolved_links: tuple[operations.TraceLink, ...]
    budget_stopped: bool
    cancelled: bool
    refusals: tuple[Refusal, ...] = ()


def trace_workflow(
    index: CodeIndex,
    judge: Judge,
    question: str,
    starts: Sequence[Span],
    *,
    depth: int | None = None,
    cancelled: Callable[[], bool] | None = None,
    checks: Sequence[Check] = TRACE_EVIDENCE_CHECKS,
) -> TraceResult:
    """Trace the static component around ``starts`` and assess five independent evidence duties.

    ``starts`` are concrete spans selected by the caller's existing entry or find owner. The static
    walk stops at a fixed point unless the caller supplies a depth or cancellation boundary. Every
    typed obligation is asked about every span in the batches owned by ``Judge``, so independent
    questions about the same code share a request; a stopped or cancelled walk makes no request at
    all. When the judge's call cap stops a later batch, the walk keeps every answered batch, marks
    the obligations with unexamined spans ``unresolved`` — never a gap — and reports
    ``budget_stopped``. The result retains every excluded span and every unresolved static link.
    """
    if not starts:
        raise ValueError("trace_workflow needs at least one concrete start span")
    judge = with_known_values(judge, index)
    graph = operations.trace_graph(index, starts, depth=depth, cancelled=cancelled)
    stopped = graph.stop == "cancelled" or (cancelled is not None and cancelled())
    answers: dict[str, list[CheckResult]] = {check.name: [] for check in checks}
    refusals: list[Refusal] = []
    budget_stopped = False

    def cancellation_requested() -> bool:
        nonlocal stopped
        stopped = stopped or (cancelled is not None and cancelled())
        return stopped

    if not stopped:
        links_by_function: dict[str, list[operations.TraceLink]] = {}
        function_keys = {
            span.key for file in {span.file for span in graph.functions} for span in index.functions_in(file)
        }
        for link in graph.links:
            endpoints = {span for span in (link.source, link.target) if span is not None}
            for span in endpoints:
                # A class's method sites are judged with their methods. Repeating every incoming
                # method/caller link on the class makes one class item exceed the request budget.
                if span == link.target and link.source is not None and span.key not in function_keys:
                    continue
                links_by_function.setdefault(span.key, []).append(link)
        items = tuple(
            _trace_item(index, span, links_by_function.get(span.key, ())) for span in graph.functions
        )
        try:
            for name, result in judge.iter_check_every(
                checks,
                items,
                {WORKFLOW: {"question": question}},
                list_name="trace",
                cancelled=cancellation_requested,
                refusals=refusals,
            ):
                answers[name].append(result)
        except CallCapReachedError:
            budget_stopped = True
    obligations = []
    evidence_keys: set[str] = set()
    item_order = {span.key: position for position, span in enumerate(graph.functions)}
    for check in checks:
        checked = tuple(sorted(answers[check.name], key=lambda result: item_order[result.item["span_key"]]))
        examined = len(checked) == len(graph.functions)
        evidence = tuple(result for result in checked if result.verdict == NoulVerdict.YES)
        unresolved = tuple(result for result in checked if result.verdict == NoulVerdict.UNSURE)
        if stopped and not checked:
            status = EvidenceStatus.UNRESOLVED
        elif evidence:
            status = EvidenceStatus.EVIDENCE_BACKED
        elif unresolved or not examined:
            # Unexamined spans may still hold evidence, so their silence stays unresolved.
            status = EvidenceStatus.UNRESOLVED
        else:
            status = EvidenceStatus.GAP_TO_INVESTIGATE
        obligations.append(TraceObligation(check.name, status, evidence, unresolved, checked, examined))
        evidence_keys.update(_item_span_key(result.item) for result in evidence)
    included_keys = _connected_backbone(graph, evidence_keys)
    included = tuple(span for span in graph.functions if span.key in included_keys)
    excluded = tuple(span for span in graph.functions if span.key not in included_keys)
    unresolved_links = tuple(
        link
        for link in graph.links
        if link.source is None
        or link.target is None
        or link.binding is None
        or link.binding.status != BindingStatus.RESOLVED
    )
    return TraceResult(
        question,
        graph,
        included,
        excluded,
        tuple(obligations),
        unresolved_links,
        budget_stopped,
        stopped,
        tuple(refusals),
    )


def _trace_item(index: CodeIndex, span: Span, links: Sequence[operations.TraceLink]) -> dict[str, object]:
    source = index.read_slice(span, origin="trace workflow")
    return {
        **source.source(),
        "span_key": span.key,
        "code": source.text,
        "links": [_link_item(index, link, span.key) for link in links],
    }


def _link_item(index: CodeIndex, link: operations.TraceLink, own_key: str) -> str:
    """One link as one dense line, carrying every fact: both endpoints with their names, the site's
    file, line and code, and the binding with its reason. A hub span's links otherwise repeat the
    same identity fields - key, file, lines, commit, content hash, reached-by - once per link, past
    any input budget; a 227-character span with 165 links measured 138,002 characters that way and
    49,000 this way, with no fact dropped."""
    site = index.read_slice(Span(link.file, link.line, link.line), origin=f"trace {link.relation}")
    parts = [
        f"{link.relation} {link.name}: {_link_ref(link.source, own_key)} ->"
        f" {_link_ref(link.target, own_key)}",
        f"at {link.file}:{link.line} {site.text}",
    ]
    binding = link.binding
    if binding is None:
        parts.append("binding none")
    else:
        differing = ""
        if binding.target is not None and (link.target is None or binding.target.key != link.target.key):
            differing = f" -> {_link_ref(binding.target, own_key)}"
        parts.append(f"binding {binding.status}: {binding.reason}{differing}")
    return " | ".join(parts)


def _link_ref(span: Span | None, own_key: str) -> str:
    """`self` for the judged span itself, else the endpoint's key with its symbol name; a missing
    endpoint stays `unknown`, never silently dropped."""
    if span is None:
        return "unknown"
    key = "self" if span.key == own_key else span.key
    return f"{key}#{span.name}" if span.name else key


def _item_span_key(item: Mapping) -> str:
    return str(item["span_key"])


def _connected_backbone(graph: operations.TraceGraph, evidence_keys: set[str]) -> set[str]:
    roots = {span.key for span in graph.roots}
    included = set(roots)
    adjacency: dict[str, set[str]] = {span.key: set() for span in graph.functions}
    for link in graph.links:
        if link.source is None or link.target is None:
            continue
        adjacency[link.source.key].add(link.target.key)
        adjacency[link.target.key].add(link.source.key)
    for target in evidence_keys - roots:
        path = _path_from_roots(roots, target, adjacency)
        included.update(path)
    return included


def _path_from_roots(roots: set[str], target: str, adjacency: Mapping[str, set[str]]) -> tuple[str, ...]:
    pending = deque(sorted(roots))
    previous: dict[str, str | None] = {root: None for root in roots}
    while pending:
        current = pending.popleft()
        if current == target:
            path = []
            cursor: str | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = previous[cursor]
            return tuple(reversed(path))
        for neighbour in sorted(adjacency.get(current, ())):
            if neighbour not in previous:
                previous[neighbour] = current
                pending.append(neighbour)
    return ()
