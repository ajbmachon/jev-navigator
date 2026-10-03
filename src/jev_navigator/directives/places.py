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
from ..index.spans import CallEdge, CodeSlice, Reference, Span, TextHit

MAX_DEFINITION_LINES = 120
REST_OF_FILE_LINES = 40
CO_CHANGE_HEAD_LINES = 40
_ENVIRONMENT_READ = re.compile(
    r"""(?:environ(?:\.get)?\(?\[?|getenv\(|process\.env\.)\s*["']?([A-Z][A-Z0-9_]{2,})"""
)
_QUOTED_KEY = re.compile(r"""["'`]([A-Za-z_][\w.:/\-]{5,79})["'`]""")
_KEY_SHAPE = re.compile(r"[._:/-]")
MAX_KEY_HITS = 30
_PASSED_ON_ROLES = frozenset(
    {"argument", "decorator", "collection", "assignment", "export", "return", "receiver", "type", "base"}
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


def range_place(
    index: CodeIndex,
    file: str,
    start: int,
    end: int,
    relation: str,
    *,
    binding: Binding | None = None,
) -> Place:
    """Lines chosen by their position (before or after a place, the start of a file); no single line
    made them a neighbour, so the signature quotes their first line of code."""
    span = Span(file, start, end)
    first_code_line = next(
        (line.strip() for line in index.read_slice(span).text.split("\n") if line.strip()), ""
    )
    shown_relation = _with_binding(relation, binding)
    signature = f"{span.key} `{first_code_line}` ({shown_relation})"
    return Place(
        span.key,
        "window",
        signature,
        lambda: index.read_slice(span, origin=shown_relation),
        relation or None,
        binding,
    )


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
    it passes on without calling, a class's own and inherited methods, the other functions of its
    file, lines anywhere in scope that mention its quoted keys or environment variables, files
    usually committed with it, and the lines before and after it). A move is any function of the
    index and the opened code that returns
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
    sites = sorted(index.find_callers(opened.span.name), key=lambda site: _is_test_file(site.file))
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
    targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
    only_tests = bool(targets) and all(_is_test_file(target.file) for target in targets)
    return not edge.binding.proven, only_tests, index.call_site_count(edge.name)


def _referenced_by(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    name = opened.span.name
    return [
        place_for_line(index, ref.file, ref.line, f"refers to {name} as {ref.role}", binding=ref.binding)
        for ref in index.find_references(name)
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


def _members(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """The methods of the opened class, or of the class a window opened inside: its own, then those
    it inherits from base classes in scope, nearest base first (depth first, left to right). A name
    comes from the nearest class that defines it, so an override hides the base's method."""
    owner = _opened_class(index, opened.span)
    if owner is None:
        return []
    places = []
    named: set[str] = set()
    for holder, binding in _class_and_bases(index, owner):
        relation = (
            f"method of {owner.name}" if holder == owner else f"inherited by {owner.name} from {holder.name}"
        )
        for method in _methods_of(index, holder):
            if method.name not in named:
                named.add(method.name)
                places.append(function_place(index, method, relation, binding=binding))
    return places


def _opened_class(index: CodeIndex, opened: Span) -> Span | None:
    named = [span for span in _classes_in(index, opened.file) if span.name == opened.name]
    return min((span for span in named if span.contains(opened.start)), key=Span.size, default=None)


def _class_and_bases(index: CodeIndex, owner: Span) -> list[tuple[Span, Binding | None]]:
    """The class, then every base class in scope it reaches, once each, with the weakest binding on
    the way to it (none for the class itself)."""
    reached: list[tuple[Span, Binding | None]] = []
    seen: set[Span] = set()
    pending: list[tuple[Span, Binding | None]] = [(owner, None)]
    while pending:
        current, binding = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        reached.append((current, binding))
        pending += reversed([(base, _weaker(binding, link)) for base, link in _bases_of(index, current)])
    return reached


def _bases_of(index: CodeIndex, owner: Span) -> list[tuple[Span, Binding | None]]:
    """The classes ``owner`` names as its bases, in the order its head lists them."""
    references = [ref for ref in index.references_in(_class_head(index, owner)) if ref.role == "base"]
    bases = []
    for ref in sorted(references, key=lambda ref: (ref.line, _column_of(index, ref))):
        targets = (
            [ref.binding.target] if ref.binding and ref.binding.target else index.find_definition(ref.name)
        )
        bases += [(target, ref.binding) for target in targets if _is_class(index, target)]
    return bases


def _class_head(index: CodeIndex, owner: Span) -> Span:
    """The lines from ``class`` to the line that opens its body, so only the head's names are bound."""
    lines = index.lines(owner.file)
    for number in range(owner.start, owner.end + 1):
        code = lines[number - 1].split("#", 1)[0].rstrip()
        if code.endswith(":") or "{" in code:
            return replace(owner, end=number)
    return owner


def _column_of(index: CodeIndex, ref: Reference) -> int:
    match = re.search(rf"\b{re.escape(ref.name)}\b", index.lines(ref.file)[ref.line - 1])
    return match.start() if match else 0


def _weaker(earlier: Binding | None, link: Binding | None) -> Binding | None:
    """A chain of bases is only as proven as its least proven link."""
    return earlier if earlier is not None and not earlier.proven else link


def _methods_of(index: CodeIndex, owner: Span) -> list[Span]:
    """The named functions whose innermost holder is ``owner``: its methods, not functions nested in
    them and not the methods of a nested class."""
    symbols = index.symbols_in(owner.file)
    return [
        function
        for function in index.functions_in(owner.file)
        if _is_named(function) and _holder(symbols, function) == owner
    ]


def _holder(symbols: Sequence[Span], span: Span) -> Span | None:
    holding = (
        other for other in symbols if other != span and other.start <= span.start <= span.end <= other.end
    )
    return min(holding, key=Span.size, default=None)


def _is_class(index: CodeIndex, span: Span) -> bool:
    return span in _classes_in(index, span.file)


def _classes_in(index: CodeIndex, file: str) -> list[Span]:
    functions = set(index.functions_in(file))
    return [span for span in index.symbols_in(file) if span not in functions]


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
    another is part of that function."""
    relation = f"in the same file as {_span_label(opened.span)}"
    functions = index.functions_in(opened.span.file)
    container = None
    if not _is_named(opened.span):
        container = min(
            (
                span
                for span in index.symbols_in(opened.span.file)
                if span != opened.span and _is_named(span) and span.contains(opened.span.start)
            ),
            key=Span.size,
            default=None,
        )
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
        place_for_line(index, hit.file, hit.line, f"mentions `{key}`")
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
        "members": _members,
        "same_file": _same_file,
        "keys_mentioned": _keys_mentioned,
        "co_changed": _co_changed,
        "lines_before": _lines_before,
        "rest_of_file": _rest_of_file,
    }
)
