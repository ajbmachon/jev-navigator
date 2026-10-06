"""Places the search can open, and the neighbours its moves list for each.

A place is concrete: a function, a window of lines, the start of a file. Its ``signature`` is the
short text Jev reads when deciding whether the target could be inside it. Jev never invents a place.
A move is a source (``jev_navigator.sources``) seeded with the opened code; each place it reaches
becomes a place here by ``reached_place``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from .. import operations, sources
from ..index.bindings import Binding
from ..index.code_index import CodeIndex
from ..index.spans import CodeSlice, Span
from ..index.units import LineAnchor, RangeAnchor
from ..sources import Reach, Seeds, Source
from .shown import LINE_CUT_MARK

MAX_DEFINITION_LINES = 120
_WINDOW_KEY_LINES = re.compile(r"(\d+)~\d+")
_PLACE_LINES = re.compile(r":\d+(?:-\d+)? ")
_WINDOW_LINE = re.compile(r"line \d+ ")


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
    shown_relation = _with_binding(relation, binding)
    return Place(
        span.key,
        "function",
        _function_signature(index, span, shown_relation),
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

    signature = _window_signature(index, open_window().span, line, shown_relation)
    return Place(f"{file}:{line}~{radius}", "window", signature, open_window, relation or None, binding)


def range_place(index: CodeIndex, file: str, start: int, end: int, relation: str) -> Place:
    """Lines chosen by their position (before or after a place, the start of a file); no single line
    made them a neighbour, so the signature quotes their first line of code and carries no binding."""
    span = Span(file, start, end)
    signature = _range_signature(index, span, relation)
    return Place(
        span.key, "window", signature, lambda: index.read_slice(span, origin=relation), relation or None
    )


def restored_signature(
    index: CodeIndex, key: str, kind: str, span: Span, relation: str | None, binding: Binding | None
) -> str:
    """The signature the builder of a place with this ``key`` and ``kind`` gives it, from the code of
    its opened ``span``: a saved place keeps no code, so a restored one rebuilds what the builder
    showed. Only lines are read, nothing is parsed."""
    if kind == "function":
        return _function_signature(index, span, _with_binding(relation or "", binding))
    line = _window_line(key)
    if line is not None:
        return _window_signature(index, span, line, _with_binding(relation or "", binding))
    return _range_signature(index, span, relation or "")


def _window_line(key: str) -> int | None:
    """The line a window key, ``path:line~radius``, names; None for any other key."""
    lines = _WINDOW_KEY_LINES.fullmatch(key.rpartition(":")[2])
    return int(lines[1]) if lines else None


def _function_signature(index: CodeIndex, span: Span, shown_relation: str) -> str:
    first_line = index.read_slice(Span(span.file, span.start, span.start)).text.strip()
    note = f" ({shown_relation})" if shown_relation else ""
    return f"{span.file}:{span.start} `{first_line}`{note}"


def _window_signature(index: CodeIndex, window: Span, line: int, shown_relation: str) -> str:
    """The window's lines and the line that made it a neighbour, quoted."""
    text_line = index.read_slice(Span(window.file, line, line)).text.strip()
    return f"{window.file}:{window.start}-{window.end} line {line} `{text_line}` ({shown_relation})"


def _range_signature(index: CodeIndex, span: Span, relation: str) -> str:
    """The range's first line of code, quoted; no single line made it a neighbour."""
    lines = index.read_slice(span).text.split("\n")
    code_line = first_code_line(lines, span.file)
    quoted = (
        next((line.strip() for line in lines if line.strip()), "") if code_line is None else lines[code_line]
    )
    return f"{span.key} `{quoted.strip()}` ({relation})"


def located_file(signature: str) -> str | None:
    """The file a place's signature names, parsed by the grammar the signature builders write: the
    file, ``:lines`` and a space at each separator, then the place's text (see ``_is_place_text``).
    None when no split fits, and when more than one does (a path or a quoted code line that holds a
    separator itself), so a caller that needs the file reads such a signature as config."""
    files: list[str] = []
    for separator in _PLACE_LINES.finditer(signature):
        if separator.start() and _is_place_text(signature[separator.end() :]):
            files.append(signature[: separator.start()])
            if len(files) > 1:
                return None
    return files[0] if files else None


def _is_place_text(text: str) -> bool:
    """A place's text: an optional ``line N `` then quoted code, ending with the closing quote or a
    parenthesised relation, or anywhere when ``cut_long_line`` cut it."""
    body = text.removesuffix(LINE_CUT_MARK)
    window_line = _WINDOW_LINE.match(body)
    quoted = body[window_line.end() :] if window_line else body
    if not quoted.startswith("`"):
        return False
    return body != text or quoted.endswith("`") or ("` (" in quoted and quoted.endswith(")"))


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
    definitions = (
        *index.symbols_in(file),
        *index.declarations_in(file),
        *operations.schema_block_spans(index, file),
    )
    return min((span for span in definitions if span.contains(line)), key=Span.size, default=None)


def reached_place(index: CodeIndex, reach: Reach) -> Place:
    """The place find opens for what a source reached: a definition whole, the code holding a line
    (``place_for_line``), lines chosen by position, or the start of a file."""
    at = reach.at
    if isinstance(at, Span):
        return function_place(index, at, reach.relation, binding=reach.binding)
    if isinstance(at, LineAnchor):
        return place_for_line(index, at.file, at.line, reach.relation, binding=reach.binding)
    lines = at if isinstance(at, RangeAnchor) else sources.file_head(index, at)
    return range_place(index, lines.file, lines.start, lines.end, reach.relation)


def neighbours(
    index: CodeIndex,
    opened: CodeSlice,
    per_kind: int | None = None,
    moves: Mapping[str, Source] | None = None,
) -> list[Place]:
    return neighbours_and_omissions(index, opened, per_kind, moves)[0]


def neighbours_and_omissions(
    index: CodeIndex,
    opened: CodeSlice,
    per_kind: int | None = None,
    moves: Mapping[str, Source] | None = None,
    shown: Span | None = None,
) -> tuple[list[Place], list[Place]]:
    """The places each move lists for ``opened``, at most ``per_kind`` per move, in the order of
    ``moves`` (default ``MOVES``: callers, callees, code that refers to it without calling it, code
    it passes on without calling, the other functions of its file, lines anywhere in scope that
    mention its quoted keys or environment variables, files usually committed with it, and the lines
    before and after it). A move is a source seeded with the opened span, named by its key in
    ``moves``, so callers can drop moves or add sources of their own. The places cut by the cap come
    back separately, so a caller can report them as not inspected.

    Places are one when they open the same lines of the same file, whatever their keys, and a place
    wholly inside the lines a request shows of ``opened`` (``shown``, by default all of them) is left
    out. The first relation found is kept: calls and references come
    before file position, so the strongest reason a place is a neighbour is the one shown. A move's
    cap counts only places no earlier move kept."""
    on_screen = opened.span if shown is None else shown
    seeds = Seeds(spans=(opened.span,))
    kept_lines: set[str] = set()
    kept: list[Place] = []
    beyond_cap: list[Place] = []
    for move, source in (MOVES if moves is None else moves).items():
        related = [replace(reached_place(index, reach), move=move) for reach in source.reach(index, seeds)]
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


def _with_binding(relation: str, binding: Binding | None) -> str:
    """A name-match link is marked, so neither Jev nor the result treats it as a proven call."""
    if binding is None or binding.proven:
        return relation
    return f"{relation}, {binding.status}: {binding.reason}"


def starting_places(index: CodeIndex, locations: Sequence[tuple[str, int]]) -> list[Place]:
    return [place_for_line(index, file, line, "start") for file, line in locations]


MOVES: Mapping[str, Source] = MappingProxyType(
    {
        "callers": sources.CALL_SITES,
        "client_calls": sources.CLIENT_CALLS,
        "callees": sources.CALLEES,
        "queried_models": sources.MODELS,
        "referenced_by": sources.REFERRERS,
        "passed_on": sources.PASSED_ON,
        "imported": sources.IMPORTED_CODE,
        "same_file": sources.SAME_FILE,
        "keys_mentioned": sources.KEY_MENTIONS,
        "co_changed": sources.CO_CHANGED,
        "lines_before": sources.LINES_BEFORE,
        "rest_of_file": sources.LINES_AFTER,
    }
)
