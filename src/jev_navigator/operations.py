"""Code-owned, multi-step operations over a CodeIndex, each callable as one unit and needing no model.

These are the main API. A directive offers them to Jev as moves, but they are just as useful on
their own: in a script, a test, or a pipeline that never calls a model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .index.bindings import Binding, BindingStatus, binding_can_target
from .index.code_index import CodeIndex
from .index.languages import language_of
from .index.spans import CallSite, CodeSlice, Span

MAX_FUNCTION_LINES = 120
_DOC_IDENTIFIER = re.compile(
    r"`([A-Za-z_][\w.]*)(?:\(\))?`"
    r"|\b([a-z]+(?:_[a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]*)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"
)
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "*/")
_TOKEN_SPLIT = re.compile(r"_|(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class TraceStep:
    """A function a trace reached, how many calls away, the function it was reached from, and the
    binding of the call that links them. A ``candidate`` or ``unresolved`` binding is a name match,
    not proof that the call reaches this function."""

    hop: int
    function: Span
    reached_from: str
    binding: Binding | None = None


@dataclass(frozen=True)
class TraceLink:
    """One static call or reference. Missing endpoints and non-resolved bindings remain evidence
    gaps; walking the graph never turns a name match into a proven edge."""

    hop: int
    source: Span | None
    target: Span | None
    relation: str
    name: str
    file: str
    line: int
    binding: Binding | None = None


@dataclass(frozen=True)
class TraceGraph:
    """The finite static component reached from ``roots`` and why traversal stopped."""

    roots: tuple[Span, ...]
    functions: tuple[Span, ...]
    links: tuple[TraceLink, ...]
    stop: str


def slice_around(index: CodeIndex, file: str, line: int, radius: int = 10) -> CodeSlice:
    """The whole enclosing function when it is short enough, otherwise a window around the line."""
    enclosing = index.enclosing_symbol(file, line)
    if enclosing is not None and enclosing.size() <= MAX_FUNCTION_LINES:
        return index.read_slice(enclosing, origin="slice_around")
    return index.read_window(file, line, radius, origin="slice_around")


def code_described_by_comment(index: CodeIndex, file: str, line: int) -> CodeSlice:
    """The code a comment is about, whole: the next function or class (decorators included) after
    a comment line, past blank lines; or, for a comment trailing code, that statement, with its
    block when the statement opens one."""
    lines = index.lines(file)
    trailing = _is_trailing_comment(lines[line - 1])
    target = line if trailing else _next_code_line(lines, line)
    if target is None:
        return index.read_window(file, line, origin="code_described_by_comment")
    symbol = _symbol_starting_at(index, file, lines, target)
    if symbol is not None:
        return index.read_slice(
            Span(file, target, symbol.end, symbol.name), origin="code_described_by_comment"
        )
    end = _block_end(lines, target) if (not trailing or _opens_block(lines[target - 1])) else target
    return index.read_slice(Span(file, target, end), origin="code_described_by_comment")


def callers_of_file(index: CodeIndex, path: str) -> tuple[CallSite, ...]:
    """Every call from another file to a named function defined in ``path``, deduplicated."""
    names = {span.name for span in index.functions_in(path) if not span.name.startswith("<")}
    sites = {site for name in sorted(names) for site in index.find_callers(name) if site.file != path}
    return tuple(sorted(sites, key=lambda site: (site.file, site.line)))


def trace_callers(index: CodeIndex, symbol: str, depth: int | None = None) -> tuple[TraceStep, ...]:
    """Functions calling ``symbol``, then their callers, to a fixed point or explicit depth."""
    return _trace(index, symbol, depth, _caller_functions)


def trace_callees(index: CodeIndex, symbol: str, depth: int | None = None) -> tuple[TraceStep, ...]:
    """In-scope callees of ``symbol``, then theirs, to a fixed point or explicit depth."""
    return _trace(index, symbol, depth, _callee_functions)


def trace_graph(
    index: CodeIndex,
    roots: Sequence[Span],
    *,
    depth: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> TraceGraph:
    """Walk calls and non-call references from concrete roots.

    With no ``depth`` this stops only when the finite static frontier is empty. An explicit depth is
    caller policy. ``cancelled`` is checked between functions so an owning host can interrupt a
    large component without a hidden time or size limit.
    """
    _require_depth(depth)
    unique_roots = tuple({span.key: span for span in roots}.values())
    if unique_roots and depth != 0 and not (cancelled is not None and cancelled()):
        # A bidirectional trace searches the whole scope for incoming relationships. Build that
        # fact inventory once, rather than grep every distinct name across still-unparsed files.
        index.functions_in_files(index.files)
    functions = {span.key: span for span in unique_roots}
    frontier = list(unique_roots)
    links: dict[tuple, TraceLink] = {}
    hop = 0
    while frontier and (depth is None or hop < depth):
        hop += 1
        next_frontier: list[Span] = []
        for function in frontier:
            if cancelled is not None and cancelled():
                return TraceGraph(unique_roots, tuple(functions.values()), tuple(links.values()), "cancelled")
            for link in _links_at(index, function, hop):
                links.setdefault(_link_key(link), link)
                neighbour = _other_end(function, link)
                if neighbour is None or neighbour.key in functions:
                    continue
                functions[neighbour.key] = neighbour
                next_frontier.append(neighbour)
        frontier = next_frontier
    stop = "depth" if frontier else "fixed_point"
    return TraceGraph(unique_roots, tuple(functions.values()), tuple(links.values()), stop)


def similar_functions(index: CodeIndex, symbol: str, limit: int = 10) -> tuple[Span, ...]:
    """Candidates for duplication: functions sharing called names or name words with ``symbol``."""
    subjects = index.find_definition(symbol)
    if not subjects:
        return ()
    subject = subjects[0]
    subject_calls = set(index.find_callees(subject))
    ranked = [
        (_similarity(subject, subject_calls, candidate, set(index.find_callees(candidate))), candidate)
        for candidate in _named_functions(index)
        if candidate != subject
    ]
    return tuple(candidate for score, candidate in sorted(ranked, key=_by_score) if score > 0)[:limit]


def code_named_in_doc(index: CodeIndex, doc_text: str) -> tuple[Span, ...]:
    """Function definitions in scope whose names the text mentions, in order of first mention."""
    mentioned = dict.fromkeys(_doc_identifiers(doc_text))
    return tuple(span for name in mentioned for span in index.find_definition(name))


def _is_comment_line(text: str) -> bool:
    return text.strip().startswith(_COMMENT_PREFIXES)


def _is_trailing_comment(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _is_comment_line(text):
        return False
    for marker in (" #", " //"):
        code, found, _ = stripped.partition(marker)
        if found and code.count('"') % 2 == 0 and code.count("'") % 2 == 0:
            return True
    return False


def _next_code_line(lines: tuple[str, ...], comment_line: int) -> int | None:
    for number in range(comment_line + 1, len(lines) + 1):
        text = lines[number - 1]
        if text.strip() and not _is_comment_line(text):
            return number
    return None


def _symbol_starting_at(index: CodeIndex, file: str, lines: tuple[str, ...], first_line: int) -> Span | None:
    declaration = first_line
    while declaration <= len(lines) and lines[declaration - 1].strip().startswith("@"):
        declaration += 1
    starting = [span for span in index.symbols_in(file) if span.start == declaration]
    return max(starting, key=Span.size, default=None)


def _opens_block(text: str) -> bool:
    return _strip_trailing_comment(text).rstrip().endswith((":", "{"))


def _strip_trailing_comment(text: str) -> str:
    for marker in (" #", " //"):
        text = text.partition(marker)[0]
    return text


def _block_end(lines: tuple[str, ...], start: int) -> int:
    base_indent = _indent(lines[start - 1])
    end = start
    for number in range(start + 1, len(lines) + 1):
        text = lines[number - 1]
        if not text.strip() or _indent(text) < base_indent:
            break
        end = number
    return end


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


def _trace(index: CodeIndex, symbol: str, depth: int | None, neighbours) -> tuple[TraceStep, ...]:
    _require_depth(depth)
    frontier = list(index.find_definition(symbol))
    seen = {span.key for span in frontier}
    steps: list[TraceStep] = []
    hop = 0
    while frontier and (depth is None or hop < depth):
        hop += 1
        reached = _next_hop(index, frontier, hop, seen, neighbours)
        steps += reached
        frontier = [step.function for step in reached]
    return tuple(steps)


def _next_hop(
    index: CodeIndex, frontier: list[Span], hop: int, seen: set[str], neighbours
) -> list[TraceStep]:
    reached = []
    for function in frontier:
        for neighbour, binding in neighbours(index, function):
            if neighbour.key in seen:
                continue
            seen.add(neighbour.key)
            reached.append(TraceStep(hop, neighbour, function.name, binding))
    return reached


def _require_depth(depth: int | None) -> None:
    if depth is not None and depth < 0:
        raise ValueError("trace depth must be non-negative")


def _links_at(index: CodeIndex, function: Span, hop: int) -> tuple[TraceLink, ...]:
    links: list[TraceLink] = []
    members = (
        tuple(
            member
            for member in index.functions_in(function.file)
            if function.start < member.start and member.end <= function.end
        )
        if function not in index.functions_in(function.file)
        else ()
    )
    for member in members:
        links.append(
            TraceLink(
                hop,
                function,
                member,
                "contains",
                member.name,
                member.file,
                member.start,
                Binding(BindingStatus.RESOLVED, "lexical containment, not invocation", member),
            )
        )

    def inside_member(line: int) -> bool:
        return any(member.contains(line) for member in members)

    for edge in index.callee_edges(function):
        if inside_member(edge.line):
            continue
        targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
        if not targets:
            links.append(
                TraceLink(hop, function, None, "call", edge.name, function.file, edge.line, edge.binding)
            )
        else:
            links.extend(
                TraceLink(hop, function, target, "call", edge.name, function.file, edge.line, edge.binding)
                for target in targets
            )
    for site in index.find_callers(function.name):
        if not binding_can_target(site.binding, function):
            continue
        links.append(
            TraceLink(
                hop,
                site.caller,
                function,
                "call",
                function.name,
                site.file,
                site.line,
                site.binding,
            )
        )
    for reference in index.references_in(function):
        if inside_member(reference.line):
            continue
        targets = (
            [reference.binding.target]
            if reference.binding is not None and reference.binding.target is not None
            else index.find_definition(reference.name)
        )
        if not targets:
            links.append(
                TraceLink(
                    hop,
                    function,
                    None,
                    reference.role,
                    reference.name,
                    reference.file,
                    reference.line,
                    reference.binding,
                )
            )
        else:
            links.extend(
                TraceLink(
                    hop,
                    function,
                    target,
                    reference.role,
                    reference.name,
                    reference.file,
                    reference.line,
                    reference.binding,
                )
                for target in targets
            )
    for reference in index.find_references(function.name):
        if not binding_can_target(reference.binding, function):
            continue
        links.append(
            TraceLink(
                hop,
                reference.holder,
                function,
                reference.role,
                function.name,
                reference.file,
                reference.line,
                reference.binding,
            )
        )
    return tuple(links)


def _other_end(function: Span, link: TraceLink) -> Span | None:
    if link.source is not None and link.source.key == function.key:
        return link.target
    if link.target is not None and link.target.key == function.key:
        return link.source
    return None


def _link_key(link: TraceLink) -> tuple:
    return (
        link.source.key if link.source is not None else "",
        link.target.key if link.target is not None else "",
        link.relation,
        link.name,
        link.file,
        link.line,
        link.binding.status if link.binding is not None else "",
        link.binding.reason if link.binding is not None else "",
    )


def _caller_functions(index: CodeIndex, function: Span) -> list[tuple[Span, Binding | None]]:
    return [
        (site.caller, site.binding)
        for site in index.find_callers(function.name)
        if site.caller is not None and binding_can_target(site.binding, function)
    ]


def _callee_functions(index: CodeIndex, function: Span) -> list[tuple[Span, Binding | None]]:
    """Each callee definition, with the call's binding; when the binding names its target, only that one."""
    linked = []
    for edge in index.callee_edges(function):
        targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
        linked += [(target, edge.binding) for target in targets]
    return linked


def _named_functions(index: CodeIndex) -> Iterable[Span]:
    for file in index.files:
        if language_of(file) is None:
            continue
        yield from (span for span in index.functions_in(file) if not span.name.startswith("<"))


def _similarity(subject: Span, subject_calls: set[str], candidate: Span, candidate_calls: set[str]) -> float:
    return _overlap(subject_calls, candidate_calls) + _overlap(_name_words(subject), _name_words(candidate))


def _overlap(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _name_words(span: Span) -> set[str]:
    return {word.lower() for word in _TOKEN_SPLIT.split(span.name) if word}


def _by_score(item: tuple[float, Span]) -> tuple[float, str]:
    score, span = item
    return -score, span.key


def _doc_identifiers(doc_text: str) -> Iterable[str]:
    for backticked, bare in _DOC_IDENTIFIER.findall(doc_text):
        yield (backticked or bare).split(".")[-1]
