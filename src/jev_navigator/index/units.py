"""Units: what a search judges and what a result names, and the one resolver of lines to units.

A unit is one function, one method, one block of a Prisma schema (a model, view, enum or composite
type, header to closing brace), or one file's top-level code: its lines outside every function,
method and block, class bodies included, kept as runs of lines in order. That is a code reading
(``Reading.CODE``), what find and find_all judge. A text reading (``Reading.TEXT``), what a text search
judges, reads only the files JVN does not parse, as plain text: one unit of kind ``text`` per block its
format gives (``text_blocks``), unless ``scope.text_files_left_out`` leaves the file out. Neither
reading ever lists the other's units. A function's or method's
unit starts at its first decorator, so a route travels with its handler, while its id stays the
index's span key. A stub, a function whose body only declares a shape (``CodeIndex.stubs_in``), is
no unit: its lines are top-level code. A record carries the unit's identity, kind, qualified symbol
and the hash of its own text. Only a unit larger than its room in a request (``box_chars``) is cut, into
pieces of up to 60 lines with no overlap; a piece never spans two runs. A YAML or JSON text block is cut
at its value's keys instead (``text_blocks.child_blocks``), so a key that fits stays whole in one piece:
neighbouring keys are packed together while they fit 60 lines and the room, and a key over the room is
cut by lines. The unit stays one unit, scored by its best piece. A piece still over that room is too
large to judge: it is named with its range and size and never judged.

Spans are lines, so functions on the same lines have the same text and are one unit, named by the
first named of them. A function nested in another is a unit of its own (``nested_in`` names the
function holding it), but a listing leaves it out: its text is already inside its holder's.

Records hold locations and hashes, never code. ``items_to_judge`` gives what a request judges (the
unit, or its pieces that fit the box) and ``read_ranges`` reads their code through the index.
``resolve_anchors`` names the units that hold a caller's lines and line ranges.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from ..judgments.questions import serialized_chars
from .code_index import CodeIndex
from .imports import import_lines, without_comments
from .languages import TEXT_LANGUAGE, is_schema_file, language_of, language_read
from .scope import is_test_file
from .spans import Span, holder_of
from .text_blocks import TextBlock, child_blocks

PIECE_LINES = 60
TOP_LEVEL_SYMBOL = "<top level>"
UNSUPPORTED_LANGUAGE = "language not supported"
CODE_FILE = "code, which a text search leaves to find_all"
OUTSIDE_SCOPE = "not in the index scope"
_UNLISTED_TOP_LEVEL = (
    "top-level code of only imports, comments, directives and brackets, which a listing leaves out"
)
_DIRECTIVE = re.compile(r"""^\s*["']use (?:client|server|strict)["']\s*;?\s*$""")
_CLOSING_BRACKETS = re.compile(r"^[\s)\]};,]*$")

LineRange = tuple[int, int]


class UnitKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    SCHEMA_BLOCK = "schema_block"
    TOP_LEVEL = "top_level"
    TEXT = "text"


class Reading(StrEnum):
    """Which files a listing reads, and how: code through its parser, or the rest as plain text."""

    CODE = "code"
    TEXT = "text"
    MIXED = "mixed"


@dataclass(frozen=True)
class Piece:
    """Lines ``start`` to ``end`` of a unit too large for the box, numbered from 0 in line order.
    ``chars`` is its size as a request spells it; a piece over the box is too large to judge."""

    index: int
    start: int
    end: int
    content_sha256: str
    chars: int
    too_large_to_judge: bool


@dataclass(frozen=True)
class Unit:
    """``ranges`` holds one range for a function and every run of lines for top-level code.
    ``nested_in`` names the function unit whose text already holds this one's, if any."""

    id: str
    path: str
    ranges: tuple[LineRange, ...]
    kind: UnitKind
    symbol: str
    language: str
    test: bool
    revision: str
    content_sha256: str
    nested_in: str | None = None
    pieces: tuple[Piece, ...] = ()

    @property
    def start(self) -> int:
        return self.ranges[0][0]

    @property
    def end(self) -> int:
        return self.ranges[-1][1]

    @property
    def judged_pieces(self) -> tuple[Piece, ...]:
        return tuple(piece for piece in self.pieces if not piece.too_large_to_judge)

    @property
    def too_large_pieces(self) -> tuple[Piece, ...]:
        return tuple(piece for piece in self.pieces if piece.too_large_to_judge)

    def piece_id(self, piece: Piece) -> str:
        return f"{self.id}#p{piece.index}"


@dataclass(frozen=True)
class UnitListing:
    """The units of a set of files, and the files that gave none, each with the reason."""

    units: tuple[Unit, ...]
    unlisted: Mapping[str, str]


def list_units(
    index: CodeIndex, files: Sequence[str], *, box_chars: int, reading: Reading = Reading.CODE
) -> UnitListing:
    """The functions and methods of ``files`` that no other function holds, the blocks of each Prisma
    schema, and each file's top-level code, in file order and then by position, parsing every source
    file in one batched scan; with ``reading`` TEXT, the text units of the files JVN does not parse
    instead. Every line of code is in a listed unit. A file whose top-level code is only imports,
    comments, directives (``"use client"``), lines of closing brackets and blank lines lists no
    top-level unit. A file the reading leaves out (see ``_left_out``), a file gone since the inventory,
    and a file outside the index's scope are named in ``unlisted``, the last with the index's own
    reason where it has one. ``box_chars`` is the room one unit's text has in a request, as
    ``serialized_chars`` counts it: the client's box (``InputLimits.box_chars``) less what the request
    carries beside the unit."""
    return UnitReader(index, box_chars, listed_only=True, reading=reading).list_files(files)


@dataclass(frozen=True)
class Item:
    """One thing a request judges: a whole unit, or one piece of a unit larger than the box."""

    id: str
    file: str
    ranges: tuple[LineRange, ...]


def items_to_judge(unit: Unit) -> tuple[Item, ...]:
    """The whole unit, or for a cut unit its pieces that fit the box; a piece too large to judge is
    never an item."""
    if not unit.pieces:
        return (Item(unit.id, unit.path, unit.ranges),)
    return tuple(
        Item(unit.piece_id(piece), unit.path, ((piece.start, piece.end),)) for piece in unit.judged_pieces
    )


def read_ranges(index: CodeIndex, file: str, ranges: Iterable[Sequence[int]]) -> str:
    """The code of ``ranges`` of ``file``, joined in order with a newline: a unit's or a piece's own
    text, the one way it is read."""
    return "\n".join(index.read_slice(Span(file, start, end)).text for start, end in ranges)


def unit_score(unit: Unit, scores: Mapping[str, float]) -> float | None:
    """A unit's score for one question from ``scores`` keyed by unit or piece id: its own, or for a
    cut unit its best judged piece's; None when nothing of it was scored."""
    if not unit.pieces:
        return scores.get(unit.id)
    best = best_piece(unit, scores)
    return None if best is None else scores[unit.piece_id(best)]


def best_piece(unit: Unit, scores: Mapping[str, float]) -> Piece | None:
    """The highest-scored piece of a cut unit, the earliest on a tie: the place to read."""
    scored = [piece for piece in unit.pieces if unit.piece_id(piece) in scores]
    return max(scored, key=lambda piece: scores[unit.piece_id(piece)], default=None)


@dataclass(frozen=True)
class LineAnchor:
    file: str
    line: int


@dataclass(frozen=True)
class RangeAnchor:
    file: str
    start: int
    end: int


Anchor = LineAnchor | RangeAnchor


@dataclass(frozen=True)
class UnresolvedAnchor:
    anchor: Anchor
    problem: str


@dataclass(frozen=True)
class AnchorResolution:
    """Each unit the anchors name, once, in the order first named, and every anchor that named none."""

    units: tuple[Unit, ...]
    unresolved: tuple[UnresolvedAnchor, ...]


def resolve_anchors(
    index: CodeIndex,
    anchors: Iterable[Anchor],
    *,
    box_chars: int,
    listed_only: bool = False,
    reading: Reading = Reading.CODE,
) -> AnchorResolution:
    """The units ``anchors`` name. A line names the innermost unit holding it: a function, decorators
    included, or the file's top-level code outside every function, stubs included, even top-level
    code the listing leaves out. A range names each unit its non-blank lines touch, leaving out units
    nested in another it names. With ``listed_only`` every unit named is one ``list_units`` lists: a
    nested function gives way to the outermost function holding it, and top-level code the listing
    leaves out names nothing. Nothing is guessed: a file outside the scope or one ``reading`` leaves
    out, a line outside its file, a reversed range, a blank line in a file with no top-level code, and
    with ``listed_only`` lines of only unlisted top-level code are reported, and a file is parsed only
    after its anchor is known to point inside it. ``box_chars`` and ``reading`` are ``list_units``'."""
    found: dict[str, Unit] = {}
    unresolved = []
    for anchor, units, problem in resolve_each(
        index, anchors, box_chars=box_chars, listed_only=listed_only, reading=reading
    ):
        if problem:
            unresolved.append(UnresolvedAnchor(anchor, problem))
        for unit in units:
            found.setdefault(unit.id, unit)
    return AnchorResolution(tuple(found.values()), tuple(unresolved))


def resolve_each(
    index: CodeIndex,
    anchors: Iterable[Anchor],
    *,
    box_chars: int,
    listed_only: bool = False,
    reading: Reading = Reading.CODE,
) -> Iterator[tuple[Anchor, tuple[Unit, ...], str]]:
    """Each anchor with the units it names and its problem (empty when it named one), in the anchors'
    order, as ``resolve_anchors`` names them; each file is read once for all of them."""
    resolver = UnitReader(index, box_chars, listed_only, reading)
    for anchor in anchors:
        units, problem = resolver.resolve(anchor)
        yield anchor, units, problem


def _left_out(index: CodeIndex, files: Sequence[str], reading: Reading) -> dict[str, str]:
    """The ``files`` in scope that ``reading`` leaves out, each with the reason: a code reading leaves
    out every file JVN does not parse, and a text reading every file it does, and the text files
    ``CodeIndex.text_files_left_out`` names."""
    if reading is Reading.CODE:
        return {file: UNSUPPORTED_LANGUAGE for file in files if not language_read(file)}
    if reading is Reading.MIXED:
        return index.text_files_left_out([file for file in files if not language_read(file)])
    code = {file: CODE_FILE for file in files if language_read(file)}
    return code | index.text_files_left_out([file for file in files if file not in code])


def _file_units(index: CodeIndex, file: str, box_chars: int) -> _FileUnits:
    if is_schema_file(file):
        return _SchemaFile(index, file, box_chars)
    if language_of(file):
        return _SourceFile(index, file, box_chars)
    return _TextFile(index, file, box_chars)


class _FileUnits:
    """The units of one file: its inner units, built by each kind of file on first use, and its
    top-level code, the lines outside every inner unit."""

    def __init__(self, index: CodeIndex, file: str, box_chars: int) -> None:
        self._index = index
        self._file = file
        self._box_chars = box_chars
        self._lines = index.lines(file)

    @cached_property
    def inner(self) -> tuple[Unit, ...]:
        return self._inner_units()

    def _inner_units(self) -> tuple[Unit, ...]:
        raise NotImplementedError

    @cached_property
    def listed(self) -> tuple[Unit, ...]:
        outermost = tuple(unit for unit in self.inner if unit.nested_in is None)
        top_level = () if self.top_level is None or self._holds_no_code() else (self.top_level,)
        return (*outermost, *top_level)

    def unit_at(self, line: int) -> Unit | None:
        holding = [unit for unit in self.inner if unit.start <= line <= unit.end]
        return min(holding, key=lambda unit: unit.end - unit.start, default=self.top_level)

    def listed_holder(self, unit: Unit) -> Unit | None:
        """``unit`` itself when a listing lists it, else the outermost function holding it; None for
        top-level code the listing leaves out."""
        while unit.nested_in is not None:
            unit = self._inner_by_id[unit.nested_in]
        return unit if unit in self.listed else None

    @cached_property
    def _inner_by_id(self) -> dict[str, Unit]:
        return {unit.id: unit for unit in self.inner}

    @cached_property
    def top_level(self) -> Unit | None:
        inside = _lines_of(self.inner)
        outside = (line for line in range(1, len(self._lines) + 1) if line not in inside)
        ranges = tuple(trimmed for run in _runs(outside) if (trimmed := self._without_blank_edges(run)))
        if not ranges:
            return None
        return self._unit(f"{self._file}:top", ranges, UnitKind.TOP_LEVEL, TOP_LEVEL_SYMBOL)

    def _holds_no_code(self) -> bool:
        non_code = _non_code_lines("\n".join(self._lines), self._file)
        return all(line in non_code for start, end in self.top_level.ranges for line in range(start, end + 1))

    def _without_blank_edges(self, run: LineRange) -> LineRange | None:
        start, end = run
        while start <= end and not self._lines[start - 1].strip():
            start += 1
        while end >= start and not self._lines[end - 1].strip():
            end -= 1
        return (start, end) if start <= end else None

    def _unit(
        self,
        unit_id: str,
        ranges: tuple[LineRange, ...],
        kind: UnitKind,
        symbol: str,
        nested_in: str | None = None,
    ) -> Unit:
        text = read_ranges(self._index, self._file, ranges)
        pieces = self._pieces(ranges) if serialized_chars(text) > self._box_chars else ()
        return Unit(
            unit_id,
            self._file,
            ranges,
            kind,
            symbol,
            language_read(self._file) or TEXT_LANGUAGE,
            is_test_file(self._file),
            self._index.read_slice(Span(self._file, *ranges[0])).commit,
            _sha256(text),
            nested_in,
            pieces,
        )

    def _pieces(self, ranges: Sequence[LineRange]) -> tuple[Piece, ...]:
        cuts = self._piece_cuts(ranges)
        return tuple(self._piece(number, start, end) for number, (start, end) in enumerate(cuts))

    def _piece_cuts(self, ranges: Sequence[LineRange]) -> list[LineRange]:
        return [cut for start, end in ranges for cut in _line_cuts(start, end)]

    def _over_box(self, start: int, end: int) -> bool:
        return serialized_chars(read_ranges(self._index, self._file, ((start, end),))) > self._box_chars

    def _piece(self, number: int, start: int, end: int) -> Piece:
        text = read_ranges(self._index, self._file, ((start, end),))
        chars = serialized_chars(text)
        return Piece(number, start, end, _sha256(text), chars, chars > self._box_chars)


class _SourceFile(_FileUnits):
    """A source file's units: its functions and methods, from the index's spans."""

    def __init__(self, index: CodeIndex, file: str, box_chars: int) -> None:
        super().__init__(index, file, box_chars)
        self._symbols = index.symbols_in(file)
        self._constant_names = index.constant_function_names(file)
        self._all_functions = frozenset(index.functions_in(file))
        self._decorator_starts = index.decorator_starts_in(file)

    def _inner_units(self) -> tuple[Unit, ...]:
        stubs = frozenset(self._index.stubs_in(self._file))
        functions = _one_per_range(span for span in self._index.functions_in(self._file) if span not in stubs)
        return tuple(self._function_unit(span, functions) for span in functions)

    def qualified(self, span: Span) -> str:
        """``span``'s name after every holder's: ``OrderService.place``, ``registerRoutes.<anonymous:4>``,
        ``run.<anonymous:2>`` inside a function a module-level constant's call builds."""
        names = []
        current: Span | None = span
        while current is not None:
            names.append(self._own_name(current))
            current = holder_of(self._symbols, current)
        return ".".join(reversed(names))

    def _own_name(self, span: Span) -> str:
        """The index's name for a function a module-level constant's call builds, ``userRouter.list``
        (``CodeIndex.constant_function_names``), else the syntax's, else the line it starts on."""
        if span in self._constant_names:
            return self._constant_names[span]
        return span.name if span.is_named else f"<anonymous:{span.start}>"

    def _function_unit(self, span: Span, functions: Sequence[Span]) -> Unit:
        holder = holder_of(self._symbols, span)
        is_method = holder is not None and holder not in self._all_functions
        outer = holder_of(functions, span)
        kind = UnitKind.METHOD if is_method else UnitKind.FUNCTION
        nested_in = None if outer is None else outer.key
        start = self._decorator_starts.get(span, span.start)
        return self._unit(span.key, ((start, span.end),), kind, self.qualified(span), nested_in)


class _SchemaFile(_FileUnits):
    """A Prisma schema's units: one per model, view, enum and composite type block, named by its
    keyword and name (``model Website``)."""

    def _inner_units(self) -> tuple[Unit, ...]:
        return tuple(
            self._unit(
                Span(self._file, block.start, block.end).key,
                ((block.start, block.end),),
                UnitKind.SCHEMA_BLOCK,
                f"{block.keyword} {block.name}",
            )
            for block in self._index.schema_blocks_in(self._file)
        )


class _TextFile(_FileUnits):
    """A file JVN does not parse, read as plain text: one unit per block its format gives
    (``CodeIndex.text_blocks_in``), named by the block's heading or key path, without the blank lines at
    its edges. Its blocks cover every line, so it has no top-level code."""

    @cached_property
    def _blocks(self) -> tuple[TextBlock, ...]:
        return self._index.text_blocks_in(self._file)

    def _inner_units(self) -> tuple[Unit, ...]:
        runs = ((block, self._without_blank_edges((block.start, block.end))) for block in self._blocks)
        return tuple(
            self._unit(Span(self._file, *run).key, (run,), UnitKind.TEXT, block.name or TOP_LEVEL_SYMBOL)
            for block, run in runs
            if run is not None
        )

    def _piece_cuts(self, ranges: Sequence[LineRange]) -> list[LineRange]:
        """A YAML or JSON block cuts at its value's keys (``child_blocks``), any other in line cuts."""
        [(start, end)] = ranges
        block = next(block for block in self._blocks if block.start <= start <= block.end)
        children = [
            (max(child.start, start), min(child.end, end))
            for child in child_blocks(self._file, self._lines, block)
        ]
        return self._packed(children) if children else super()._piece_cuts(ranges)

    def _packed(self, children: Sequence[LineRange]) -> list[LineRange]:
        """Cuts at the children's edges: neighbours packed together while they fit ``PIECE_LINES``
        lines and the box, a longer child whole, and a child over the box in line cuts."""
        cuts: list[LineRange] = []
        for start, end in children:
            if self._over_box(start, end):
                cuts += _line_cuts(start, end)
            elif cuts and end - cuts[-1][0] < PIECE_LINES and not self._over_box(cuts[-1][0], end):
                cuts[-1] = (cuts[-1][0], end)
            else:
                cuts.append((start, end))
        return cuts


class UnitReader:
    """A search-lifetime code/text unit reader. Each file is built once for one request room."""

    def __init__(self, index: CodeIndex, box_chars: int, listed_only: bool, reading: Reading) -> None:
        self._index = index
        self._box_chars = box_chars
        self._listed_only = listed_only
        self._reading = reading
        self._sources: dict[str, _FileUnits] = {}

    def list_files(self, files: Sequence[str]) -> UnitListing:
        """List through the same file units used to resolve later anchors in this search."""
        files = tuple(dict.fromkeys(files))
        in_scope = frozenset(self._index.files)
        not_indexed = self._index.not_indexed_files
        unlisted = {file: not_indexed.get(file, OUTSIDE_SCOPE) for file in files if file not in in_scope}
        unlisted |= _left_out(self._index, [file for file in files if file in in_scope], self._reading)
        read_files = tuple(file for file in files if file in in_scope and file not in unlisted)
        self._index.functions_in_files(tuple(file for file in read_files if language_of(file)))
        units = tuple(unit for file in read_files for unit in self._source(file).listed)
        unlisted |= {file: reason for file, reason in self._index.unavailable_files.items() if file in files}
        return UnitListing(units, unlisted)

    def resolve(self, anchor: Anchor) -> tuple[tuple[Unit, ...], str]:
        start, end = (
            (anchor.line, anchor.line) if isinstance(anchor, LineAnchor) else (anchor.start, anchor.end)
        )
        problem = self._lines_problem(anchor.file, start, end)
        if problem:
            return (), problem
        units = self._units_touching(anchor.file, start, end)
        if not units:
            return (), f"{_lines_named(anchor.file, start, end)}: blank and outside every function"
        if not self._listed_only:
            return units, ""
        holders = self._listed_holders(anchor.file, units)
        if not holders:
            return (), f"{_lines_named(anchor.file, start, end)}: {_UNLISTED_TOP_LEVEL}"
        return holders, ""

    def _listed_holders(self, file: str, units: Iterable[Unit]) -> tuple[Unit, ...]:
        source = self._source(file)
        return tuple(holder for unit in units if (holder := source.listed_holder(unit)))

    def _units_touching(self, file: str, start: int, end: int) -> tuple[Unit, ...]:
        source = self._source(file)
        lines = self._index.lines(file)
        touched_lines = [line for line in range(start, end + 1) if lines[line - 1].strip()] or [start]
        touched = dict.fromkeys(unit for line in touched_lines if (unit := source.unit_at(line)) is not None)
        return tuple(unit for unit in touched if not any(_nests(unit, other) for other in touched))

    def _lines_problem(self, file: str, start: int, end: int) -> str:
        if problem := self._file_problem(file):
            return problem
        if start > end:
            return f"the range {start}-{end} ends before it starts"
        line_count = len(self._index.lines(file))
        outside = next((line for line in (start, end) if not 1 <= line <= line_count), None)
        if outside is not None:
            return f"line {outside} is outside {file}, which has {line_count} lines"
        return ""

    def _file_problem(self, file: str) -> str:
        if file not in self._index.files:
            return f"{file} is not in scope"
        return _left_out(self._index, [file], self._reading).get(file, "")

    def _source(self, file: str) -> _FileUnits:
        if file not in self._sources:
            self._sources[file] = _file_units(self._index, file, self._box_chars)
        return self._sources[file]


def _non_code_lines(source: str, file: str) -> frozenset[int]:
    """Lines holding nothing but imports, comments, a directive, closing brackets or whitespace."""
    code = without_comments(source, file).split("\n")
    trivial = {
        number
        for number, line in enumerate(code, 1)
        if _CLOSING_BRACKETS.match(line) or _DIRECTIVE.match(line)
    }
    return frozenset(trivial) | import_lines(source, file)


def _one_per_range(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Functions on the same lines have the same text, so they are one unit, named by a named one."""
    by_range: dict[LineRange, list[Span]] = {}
    for span in spans:
        by_range.setdefault((span.start, span.end), []).append(span)
    chosen = (min(group, key=lambda span: (not span.is_named, span.name)) for group in by_range.values())
    return tuple(sorted(chosen, key=lambda span: (span.start, -span.end)))


def _lines_of(units: Iterable[Unit]) -> frozenset[int]:
    return frozenset(line for unit in units for start, end in unit.ranges for line in range(start, end + 1))


def _line_cuts(start: int, end: int) -> list[LineRange]:
    return [(first, min(first + PIECE_LINES - 1, end)) for first in range(start, end + 1, PIECE_LINES)]


def _runs(lines: Iterable[int]) -> list[LineRange]:
    runs: list[list[int]] = []
    for line in lines:
        if runs and runs[-1][1] == line - 1:
            runs[-1][1] = line
        else:
            runs.append([line, line])
    return [(start, end) for start, end in runs]


def _lines_named(file: str, start: int, end: int) -> str:
    return f"line {start} of {file}" if start == end else f"lines {start} to {end} of {file}"


def _nests(inner: Unit, outer: Unit) -> bool:
    """Whether ``inner`` is a function inside ``outer``'s function lines."""
    if inner == outer or UnitKind.TOP_LEVEL in (inner.kind, outer.kind):
        return False
    return outer.start <= inner.start and inner.end <= outer.end


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
