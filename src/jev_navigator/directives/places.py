"""Places the search can open, and the neighbours code lists for each from the parser and git.

A place is concrete: a function, a window of lines, the start of a file. Its ``signature`` is the
short text Jev reads when deciding whether the target could be inside it. Jev never invents a place.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from ..index.bindings import Binding
from ..index.code_index import CodeIndex
from ..index.spans import CallEdge, CodeSlice, Span, TextHit
from ..judgments.relations import key_mention

MAX_DEFINITION_LINES = 120
REST_OF_FILE_LINES = 40
CO_CHANGE_HEAD_LINES = 40
IMPORTED_HEAD_LINES = 40
_ENVIRONMENT_READ = re.compile(
    r"""(?:environ(?:\.get)?\(?\[?|getenv\(|process\.env\.)\s*["']?([A-Z][A-Z0-9_]{2,})"""
)
_QUOTED_KEY = re.compile(r"""["'`]([A-Za-z_][\w.:/\-]{5,79})["'`]""")
_KEY_SHAPE = re.compile(r"[._:/-]")
MAX_KEY_HITS = 30
_PASSED_ON_ROLES = frozenset(
    {"argument", "decorator", "collection", "assignment", "export", "return", "receiver", "type"}
)
_TEST_DIRECTORIES = frozenset({"test", "tests", "__tests__", "spec"})
_TEST_FILE_NAME = re.compile(r"^test_|_test\.|\.test\.|\.spec\.|^conftest\.py$")


@dataclass(frozen=True)
class Place:
    key: str
    kind: str
    signature: str
    open: Callable[[], CodeSlice]
    relation: str | None = None
    binding: Binding | None = None
    move: str | None = None


def function_place(
    index: CodeIndex, span: Span, relation: str = "", *, binding: Binding | None = None
) -> Place:
    """A function, class or declaration, opened whole."""
    first_line = index.read_slice(Span(span.file, span.start, span.start)).text.strip()
    shown_relation = _with_binding(relation, binding)
    note = f" ({shown_relation})" if shown_relation else ""
    signature = f"{span.file}:{span.start} `{first_line}`{note}"
    return Place(
        span.key,
        "function",
        signature,
        lambda: index.read_slice(span, origin=shown_relation or "function"),
        relation or None,
        binding,
    )


def window_place(
    index: CodeIndex,
    file: str,
    line: int,
    relation: str,
    radius: int = 10,
    name: str = "",
    *,
    binding: Binding | None = None,
) -> Place:
    """The lines around ``line``, the line that made this place a neighbour (a call, a reference, a
    mentioned key); the signature quotes that line. ``name`` names the class or declaration the
    window lies in, so the moves can follow that name from the window."""

    shown_relation = _with_binding(relation, binding)

    def open_window() -> CodeSlice:
        window = index.read_window(file, line, radius).span
        return index.read_slice(replace(window, name=name), origin=shown_relation)

    span = open_window().span
    text_line = index.read_slice(Span(file, line, line)).text.strip()
    signature = f"{file}:{span.start}-{span.end} line {line} `{text_line}` ({shown_relation})"
    return Place(f"{file}:{line}~{radius}", "window", signature, open_window, relation or None, binding)


def range_place(index: CodeIndex, file: str, start: int, end: int, relation: str) -> Place:
    """Lines chosen by their position (before or after a place, the start of a file); no single line
    made them a neighbour, so the signature quotes their first line of code and carries no binding."""
    span = Span(file, start, end)
    lines = index.read_slice(span).text.split("\n")
    code_line = first_code_line(lines, file)
    quoted = (
        next((line.strip() for line in lines if line.strip()), "") if code_line is None else lines[code_line]
    )
    signature = f"{span.key} `{quoted.strip()}` ({relation})"
    return Place(
        span.key, "window", signature, lambda: index.read_slice(span, origin=relation), relation or None
    )


def first_code_line(lines: Sequence[str], file: str) -> int | None:
    """The index of the first line of code, past blank lines, comments, a shebang, a Python
    docstring and a "use strict" directive; None when there is none."""
    python = file.endswith(".py")
    openers = ('"""', "'''") if python else ("/*",)
    closing = None
    for number, line in enumerate(lines):
        text = line.strip()
        if closing is not None:
            closing = None if closing in text else closing
            continue
        if not text or text.startswith(("#!", "//")) or (python and text.startswith("#")):
            continue
        if text.strip("'\"; ") == "use strict":
            continue
        opener = next((mark for mark in openers if text.startswith(mark)), None)
        if opener is None:
            return number
        closing = "*/" if opener == "/*" else opener
        closing = None if closing in text[len(opener) :] else closing
    return None


def place_for_line(
    index: CodeIndex, file: str, line: int, relation: str, *, binding: Binding | None = None
) -> Place:
    """The function holding ``line``. Outside every function, the class or module-level declaration
    holding it, whole when it has at most ``MAX_DEFINITION_LINES`` lines and otherwise as the window
    around the line under its name, so the moves can follow that name; outside every definition,
    the window around the line."""
    function = index.enclosing_symbol(file, line)
    if function is not None:
        return function_place(index, function, relation, binding=binding)
    definition = _enclosing_definition(index, file, line)
    if definition is None:
        return window_place(index, file, line, relation, binding=binding)
    if definition.size() <= MAX_DEFINITION_LINES:
        return function_place(index, definition, relation, binding=binding)
    return window_place(index, file, line, relation, name=definition.name, binding=binding)


def place_relationship(place: Place) -> dict | None:
    """Machine-readable origin for a place, without manufacturing relationship endpoints."""
    if place.relation is None and place.binding is None and place.move is None:
        return None
    relationship = {}
    if place.move is not None:
        relationship["move"] = place.move
    if place.relation is not None:
        relationship["relation"] = place.relation
    if place.binding is not None:
        binding = {
            "status": place.binding.status,
            "reason": place.binding.reason,
            "proven": place.binding.proven,
        }
        if place.binding.target is not None:
            binding["target"] = asdict(place.binding.target)
        relationship["binding"] = binding
    return relationship


def _enclosing_definition(index: CodeIndex, file: str, line: int) -> Span | None:
    definitions = (*index.symbols_in(file), *index.declarations_in(file))
    return min((span for span in definitions if span.contains(line)), key=Span.size, default=None)


Move = Callable[[CodeIndex, CodeSlice], list[Place]]


def neighbours(
    index: CodeIndex,
    opened: CodeSlice,
    per_kind: int | None = None,
    moves: Mapping[str, Move] | None = None,
) -> list[Place]:
    return neighbours_and_omissions(index, opened, per_kind, moves)[0]


def neighbours_and_omissions(
    index: CodeIndex,
    opened: CodeSlice,
    per_kind: int | None = None,
    moves: Mapping[str, Move] | None = None,
    shown: Span | None = None,
) -> tuple[list[Place], list[Place]]:
    """The places each move lists for ``opened``, at most ``per_kind`` per move, in the order of
    ``moves`` (default ``MOVES``: callers, callees, code that refers to it without calling it, code
    it passes on without calling, the other functions of its file, lines anywhere in scope that
    mention its quoted keys or environment variables, files usually committed with it, and the lines
    before and after it). A move is any function of the index and the opened code that returns
    places, so callers can drop moves or add their own. The places cut by the cap come back
    separately, so a caller can report them as not inspected.

    Places are one when they open the same lines of the same file, whatever their keys, and a place
    wholly inside the lines a request shows of ``opened`` (``shown``, by default all of them) is left
    out. The first relation found is kept: calls and references come
    before file position, so the strongest reason a place is a neighbour is the one shown. A move's
    cap counts only places no earlier move kept."""
    on_screen = opened.span if shown is None else shown
    kept_lines: set[str] = set()
    kept: list[Place] = []
    beyond_cap: list[Place] = []
    for move, build in (MOVES if moves is None else moves).items():
        related = [replace(place, move=move) for place in build(index, opened)]
        new = _new_places(related, on_screen, kept_lines)
        selected = new if per_kind is None else new[:per_kind]
        kept += selected
        kept_lines |= {place.open().span.key for place in selected}
        if per_kind is not None:
            beyond_cap += new[per_kind:]
    return kept, _new_places(beyond_cap, on_screen, kept_lines)


def _new_places(places: list[Place], on_screen: Span, kept_lines: set[str]) -> list[Place]:
    """One place per stretch of lines, leaving out stretches already kept and those on screen."""
    seen = set(kept_lines)
    new = []
    for place in places:
        span = place.open().span
        if span.key not in seen and not _within(span, on_screen):
            seen.add(span.key)
            new.append(place)
    return new


def _within(inner: Span, outer: Span) -> bool:
    return inner.file == outer.file and outer.start <= inner.start and inner.end <= outer.end


def _callers(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    sites = sorted(
        (site for site in index.find_callers(opened.span.name) if _may_reach(site.binding, opened.span)),
        key=lambda site: _is_test_file(site.file),
    )
    return [
        place_for_line(index, site.file, site.line, f"calls {opened.span.name}", binding=site.binding)
        for site in sites
    ]


def _callees(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """What the opened code calls, with proven production targets before name-only candidates."""
    places = []
    edges = sorted(index.callee_edges(opened.span), key=lambda edge: _callee_rank(index, edge))
    source = _span_label(opened.span)
    for edge in edges:
        targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
        relation = f"called by {source}"
        places += [function_place(index, span, relation, binding=edge.binding) for span in targets]
    return places


def _callee_rank(index: CodeIndex, edge: CallEdge) -> tuple[bool, bool, int]:
    """A callee with no definition yields no place, so its call sites are never counted."""
    targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
    only_tests = bool(targets) and all(_is_test_file(target.file) for target in targets)
    return not edge.binding.proven, only_tests, index.call_site_count(edge.name) if targets else 0


def _referenced_by(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    name = opened.span.name
    return [
        place_for_line(index, ref.file, ref.line, f"refers to {name} as {ref.role}", binding=ref.binding)
        for ref in index.find_references(name)
        if _may_reach(ref.binding, opened.span)
    ]


def _passed_on(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    places = []
    source = _span_label(opened.span)
    for ref in (ref for ref in index.references_in(opened.span) if ref.role in _PASSED_ON_ROLES):
        targets = (
            [ref.binding.target] if ref.binding and ref.binding.target else index.find_definition(ref.name)
        )
        relation = f"passed on by {source} as {ref.role}"
        places += [function_place(index, span, relation, binding=ref.binding) for span in targets]
    return places


def _imported(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """What the opened code imports, re-exports or requires from files in scope. Code outside every
    function and class stands for its module, so all of its file's import statements count; a
    function or class counts only its own lines, as callees already follow the calls it makes. A name
    taken by name opens its definition in the module the import resolves to; a whole module, or a
    name that module only passes on from elsewhere, opens the start of that module."""
    span = opened.span
    module_level = not any(symbol.contains(span.start) for symbol in index.symbols_in(span.file))
    text = "\n".join(index.lines(span.file)) if module_level else opened.text
    source = span.file if module_level else _span_label(span)
    places = []
    for fact, names in index.imports_in(span.file, text):
        relation = (
            f"imported by {source}" if fact.proven else f"imported by {source}, candidate: {fact.reason}"
        )
        definitions = sorted(
            (
                found
                for name in names or ()
                for found in index.find_definition(name)
                if found.file == fact.path
            ),
            key=lambda found: found.start,
        )
        places += [function_place(index, definition, relation) for definition in definitions]
        if names is None or not names <= {definition.name for definition in definitions}:
            end = min(len(index.lines(fact.path)), IMPORTED_HEAD_LINES)
            places.append(range_place(index, fact.path, 1, end, f"start of a module {relation}"))
    return places


def _may_reach(binding: Binding | None, span: Span) -> bool:
    """False when ``binding`` names a definition other than ``span``. ``span`` may be a window inside
    the class or declaration it is named after, so a target overlapping it still counts."""
    target = None if binding is None else binding.target
    return target is None or (target.file == span.file and _overlaps(target, span))


def _with_binding(relation: str, binding: Binding | None) -> str:
    """A name-match link is marked, so neither Jev nor the result treats it as a proven call."""
    if binding is None or binding.proven:
        return relation
    return f"{relation}, {binding.status}: {binding.reason}"


def _is_test_file(path: str) -> bool:
    directories, _, name = path.rpartition("/")
    in_test_directory = not _TEST_DIRECTORIES.isdisjoint(directories.split("/"))
    return in_test_directory or bool(_TEST_FILE_NAME.search(name))


def _same_file(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """The other functions of the file, nearest to the opened code first; a function nested in
    another is part of that function. An anonymous function first offers its nearest named
    container, else its nearest container: a callback in a test's callback offers that test."""
    relation = f"in the same file as {_span_label(opened.span)}"
    functions = index.functions_in(opened.span.file)
    container = None
    if not _is_named(opened.span):
        containers = [
            span
            for span in index.symbols_in(opened.span.file)
            if span != opened.span and span.contains(opened.span.start)
        ]
        named = [span for span in containers if _is_named(span)]
        container = min(named or containers, key=Span.size, default=None)
    outermost = [span for span in functions if not any(_encloses(other, span) for other in functions)]
    others = [span for span in outermost if not _overlaps(span, opened.span)]
    nearest_first = sorted(others, key=lambda span: (_distance(span, opened.span), span.start))
    ordered = ([container] if container is not None else []) + nearest_first
    return [function_place(index, span, relation) for span in ordered]


def _overlaps(first: Span, second: Span) -> bool:
    return first.start <= second.end and second.start <= first.end


def _encloses(outer: Span, inner: Span) -> bool:
    same_lines = (outer.start, outer.end) == (inner.start, inner.end)
    return not same_lines and outer.start <= inner.start and inner.end <= outer.end


def _distance(span: Span, opened: Span) -> int:
    return opened.start - span.end if span.end < opened.start else span.start - opened.end


def _keys_mentioned(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """Lines elsewhere that mention an environment variable it reads or a key it quotes, the rarest
    key first. A quoted key has at least six characters and a key's shape: a dot, underscore, colon,
    slash or dash. A key found on more than ``MAX_KEY_HITS`` lines is too common to point anywhere
    and is skipped."""
    hits_by_key = {key: _lines_mentioning(index, opened, key) for key in _keys_in(opened.text)}
    usable = [(key, hits) for key, hits in hits_by_key.items() if 0 < len(hits) <= MAX_KEY_HITS]
    rarest_first = sorted(usable, key=lambda item: len(item[1]))
    return [
        place_for_line(index, hit.file, hit.line, key_mention(key))
        for key, hits in rarest_first
        for hit in hits
    ]


def _keys_in(code: str) -> list[str]:
    quoted = [key for key in _QUOTED_KEY.findall(code) if _KEY_SHAPE.search(key)]
    return list(dict.fromkeys(_ENVIRONMENT_READ.findall(code) + quoted))


def _lines_mentioning(index: CodeIndex, opened: CodeSlice, key: str) -> list[TextHit]:
    whole_key = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")
    return [
        hit
        for hit in index.search_text(key, MAX_KEY_HITS + 1)
        if whole_key.search(hit.text)
        and not (hit.file == opened.span.file and opened.span.contains(hit.line))
    ]


def _co_changed(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    places = []
    for other, commits in index.co_changed_files(opened.span.file, limit=2):
        relation = f"start of a file committed with {opened.span.file} {commits} times"
        end = min(len(index.lines(other)), CO_CHANGE_HEAD_LINES)
        places.append(range_place(index, other, 1, end, relation))
    return places


def _lines_before(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    span = opened.span
    if span.start <= 1:
        return []
    start = max(1, span.start - REST_OF_FILE_LINES)
    return [range_place(index, span.file, start, span.start - 1, f"the lines before {span.key}")]


def _rest_of_file(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    span = opened.span
    line_count = len(index.lines(span.file))
    if span.end >= line_count:
        return []
    end = min(line_count, span.end + REST_OF_FILE_LINES)
    return [range_place(index, span.file, span.end + 1, end, f"the lines after {span.key}")]


def _is_named(span: Span) -> bool:
    return bool(span.name) and not span.name.startswith("<")


def _span_label(span: Span) -> str:
    return span.name if _is_named(span) else span.key


def starting_places(index: CodeIndex, locations: Sequence[tuple[str, int]]) -> list[Place]:
    return [place_for_line(index, file, line, "start") for file, line in locations]


MOVES: Mapping[str, Move] = MappingProxyType(
    {
        "callers": _callers,
        "callees": _callees,
        "referenced_by": _referenced_by,
        "passed_on": _passed_on,
        "imported": _imported,
        "same_file": _same_file,
        "keys_mentioned": _keys_mentioned,
        "co_changed": _co_changed,
        "lines_before": _lines_before,
        "rest_of_file": _rest_of_file,
    }
)
