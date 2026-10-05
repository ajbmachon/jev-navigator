"""Structural measurements of a scope, taken from the index the caller already has: no model, no guessing.

The index's parser reports *symbols*: functions and classes. Everything it calls a function is
counted as one — a module-level ``def``, a method, a function nested inside another function, an
arrow function or a function expression — because that is how ``CodeIndex.functions_in`` defines
functions, and classes are measured apart from them, never as functions. A symbol whose lines sit
inside another symbol's lines is called *held* here: a method inside its class, or a function
nested in another function. Line ranges are the physical lines a symbol occupies, inclusive at
both ends (``Span.size``), so a held symbol's lines are also its holder's lines, and the lines a
scope's symbols cover are counted once however they are nested.

Two limits come from measuring with the index rather than by reading source, and are named here
rather than hidden: a held symbol is found by line containment, so a symbol nested on the same
single line as its holder — a method on a one-line class, a function and its inner arrow sharing a
line — has no holder to report, and a class is counted as a class only where the index lists a
class there, which the parser does apart from any method written on the same line.

A file the parser could only recover partially (``FileFacts.incomplete``) keeps the symbols it did
recover and still says so: what the grammar swallowed is unknown, not absent, so a count taken
from a partial parse is a floor and never a claim of completeness. Nothing here parses source
again, greps source text, or calls a model: every fact comes from the index's own fact scan,
asked for once per scope rather than once per symbol or once per file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.scope_scan import FileFacts, FileStructure
from ..index.spans import CodeSlice, Span, holder_of

SymbolKind = str

SYMBOL_KINDS: tuple[SymbolKind, ...] = ("function", "class")


@dataclass(frozen=True)
class Symbol:
    """One function or class as the index found it, beside the symbol that holds it, if any.

    ``holder`` is the smallest other symbol whose lines contain this one's: a method's class, or a
    nested function's outer function. ``kind`` is ``"function"`` or ``"class"``.
    """

    span: Span
    kind: SymbolKind
    holder: Span | None = None

    @property
    def size(self) -> int:
        """The physical lines it fills, counted at both ends."""
        return self.span.size()

    @property
    def held(self) -> bool:
        """True when another symbol's lines contain it: a method, or a nested function."""
        return self.holder is not None

    def __str__(self) -> str:
        holder = "" if self.holder is None else f" in {self.holder.name}"
        return f"{self.span.file}:{self.span.start}-{self.span.end} {self.kind} {self.span.name}{holder}"


@dataclass(frozen=True)
class Coverage:
    """What the parser saw in a scope, beside the counts or ranking taken from it.

    ``measured`` and ``skipped`` split ``scope``: files a fact scan can bring structure back from,
    and files with no grammar to parse (a document, data, a manifest), where there is no structure
    to measure and no gap to report. ``unmeasured`` names the files that were asked about yet
    never scanned — outside the index scope, gone from the disk, or refused by the parser as too
    large to parse — and ``unparsed`` the measured
    files whose parse reported grammar errors. An empty scope measures nothing and says so.
    """

    scope: tuple[str, ...] = ()
    measured: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    unmeasured: tuple[str, ...] = ()
    unparsed: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True when a scope was given, something in it was measurable, every file of it was
        scanned, and none of them partly parsed."""
        return bool(self.scope) and bool(self.measured) and not (self.unmeasured or self.unparsed)

    @property
    def statement(self) -> str:
        """What the scan could not cover, in words, or nothing when it covered the whole scope."""
        if not self.scope:
            return "no scope was given, so no file was measured"
        if self.unmeasured and self.unparsed:
            return (
                f"{len(self.unmeasured)} file(s) in scope were never scanned and "
                f"{len(self.unparsed)} parsed only partially: what those files hold is unknown, not "
                "absent, so this measurement covers the rest of the scope only"
            )
        if self.unmeasured:
            return (
                f"{len(self.unmeasured)} file(s) in scope were never scanned (unreadable, or refused by "
                "the parser), so what they hold is unknown, not absent"
            )
        if self.unparsed:
            return (
                f"{len(self.unparsed)} file(s) parsed only partially, so the symbols counted there are "
                "the ones the parser recovered, not all the file's"
            )
        if not self.measured:
            return (
                f"none of the {len(self.scope)} file(s) in scope is code the parser can read, so "
                "nothing here could be measured"
            )
        return ""


@dataclass(frozen=True)
class FileSymbols:
    """The functions and classes the index found in one file, and what sits inside what."""

    file: str = ""
    functions: tuple[Symbol, ...] = ()
    classes: tuple[Symbol, ...] = ()
    parsed: bool = True
    kinds: tuple[SymbolKind, ...] = SYMBOL_KINDS

    @property
    def counts(self) -> dict[str, int]:
        """This file's function and class counts — a floor, and empty for a file that did not parse."""
        counts = {"function": len(self.functions), "class": len(self.classes)}
        return {kind: counts[kind] for kind in self.kinds}

    @property
    def held(self) -> tuple[Symbol, ...]:
        """The symbols another one holds: methods, and functions nested inside another function."""
        return tuple(symbol for symbol in (*self.functions, *self.classes) if symbol.held)

    @property
    def symbol_lines(self) -> int:
        """Physical lines covered by at least one symbol, a nested pair counted once."""
        return len(
            {
                line
                for symbol in (*self.functions, *self.classes)
                for line in range(symbol.span.start, symbol.span.end + 1)
            }
        )


@dataclass(frozen=True)
class Largest:
    """The symbols of a scope ranked by the lines they fill, widest first, with the scan that
    measured them.

    ``measured`` is every symbol the scan could measure and ``largest`` the part of that ranking
    shown to the caller: a ``limit`` shortens the ranking shown, never the measurement. ``biggest``
    names the tied-widest symbols, so a tie is never cut down to one name.
    """

    measured: tuple[Symbol, ...] = ()
    largest: tuple[Symbol, ...] = ()
    coverage: Coverage = Coverage()
    limit: int | None = None

    @property
    def size(self) -> int:
        """The inclusive line count of the widest symbol, or 0 when the scope held none."""
        return 0 if not self.measured else self.measured[0].size

    @property
    def biggest(self) -> tuple[Symbol, ...]:
        """The widest symbols measured, every one of them: a tie is never cut down to one name."""
        if not self.measured:
            return ()
        widest = self.measured[0].size
        return tuple(symbol for symbol in self.measured if symbol.size == widest)

    @property
    def truncated(self) -> bool:
        """True when a ``limit`` dropped symbols that were measured, so nothing here is a ceiling."""
        return len(self.largest) < len(self.measured)

    @property
    def complete(self) -> bool:
        """True when the ranking rests on a fully scanned, fully parsed scope shown in full."""
        return self.coverage.complete and not self.truncated

    @property
    def caveat(self) -> str:
        """Why this ranking is not a complete picture of the scope, or nothing when it is."""
        if not self.coverage.complete:
            return self.coverage.statement
        if self.truncated:
            return (
                f"only {len(self.largest)} of {len(self.measured)} measured symbols are shown, so what "
                "is left out may hold a narrower symbol of its own"
            )
        return ""


@dataclass(frozen=True)
class SymbolCount:
    """How many functions and classes a scope holds, and how much of it the scan could see."""

    per_file: dict[str, FileSymbols] = field(default_factory=dict)
    largest: Largest = Largest()
    kinds: tuple[SymbolKind, ...] = SYMBOL_KINDS

    @property
    def total(self) -> dict[str, int]:
        """The scope's function and class counts, added up from the files that were measured."""
        totals = dict.fromkeys(self.kinds, 0)
        for measured in self.per_file.values():
            for kind, count in measured.counts.items():
                totals[kind] = totals.get(kind, 0) + count
        return totals

    @property
    def held(self) -> tuple[Symbol, ...]:
        """Every symbol in scope that another symbol holds, across all measured files."""
        return tuple(symbol for measured in self.per_file.values() for symbol in measured.held)

    @property
    def symbol_lines(self) -> int:
        """Physical lines the measured scope covers, counted once wherever symbols are nested."""
        return sum(measured.symbol_lines for measured in self.per_file.values())

    @property
    def coverage(self) -> Coverage:
        return self.largest.coverage

    @property
    def complete(self) -> bool:
        """True when every file in scope was scanned and none of them partly parsed."""
        return self.coverage.complete

    @property
    def caveat(self) -> str:
        """Why these counts are not the whole scope, or nothing when they are."""
        return self.coverage.statement

    def count_of(self, file: str) -> FileSymbols | None:
        """The measured file, or None when this result did not measure that file."""
        return self.per_file.get(file)


def scope_of(index: CodeIndex, scope: Sequence[str] | None = None) -> tuple[str, ...]:
    """The files a measurement covers: every file in the index scope when the caller names none."""
    return tuple(dict.fromkeys(index.files if scope is None else tuple(scope)))


def facts_of(index: CodeIndex, scope: Sequence[str] | None = None) -> dict[str, FileFacts]:
    """The index's own facts for ``scope``, asked for in one fact scan instead of one per file.

    A scope path the index never inventoried is named as a gap by :func:`scan_coverage` and left
    out here: the index keeps its own scope checks, and an unreadable file is never hunted down.
    """
    wanted = tuple(
        path for path in scope_of(index, scope) if path in index.files and language_of(path) is not None
    )
    if not wanted:
        return {}
    return index.facts_in_files(wanted)


def file_structure(index: CodeIndex, file: str) -> FileStructure | None:
    """The structure the index holds for one file, or None where it holds none to read."""
    facts = facts_of(index, (file,)).get(file)
    return None if facts is None else facts.structure


def scan_coverage(index: CodeIndex, scope: Sequence[str] | None = None) -> Coverage:
    """How much of ``scope`` the parser could see: scanned, skipped, unreadable, partly parsed.

    The whole scope goes to the index's fact scan at once, so what a file's grammar could not
    recover is known before any count is taken from it. A file outside the index's inventory, a file
    that has gone from the disk, and a file the parser refused are all ``unmeasured``: none was
    scanned, so none may be read as an empty result.
    """
    wanted = scope_of(index, scope)
    unreadable = tuple(path for path in wanted if path not in index.available_files)
    readable = tuple(path for path in wanted if path in index.available_files)
    scannable = tuple(path for path in readable if language_of(path) is not None)
    facts = facts_of(index, scannable)
    unscanned = tuple(path for path in scannable if path not in facts)
    return Coverage(
        scope=wanted,
        measured=tuple(path for path in scannable if path in facts),
        skipped=tuple(path for path in readable if language_of(path) is None),
        unmeasured=unreadable + unscanned,
        unparsed=tuple(path for path in scannable if path in facts and facts[path].incomplete),
    )


def symbol_spans(
    index: CodeIndex, file: str, kinds: Sequence[SymbolKind] = SYMBOL_KINDS
) -> tuple[Symbol, ...]:
    """The symbols the index finds in one file, each beside the symbol that holds it, if any.

    Functions come from ``functions_in`` and classes from the symbols that are not functions, so a
    class is never counted as a function and a method inside one is still counted as a function.
    """
    structure = file_structure(index, file)
    if structure is None:
        return ()
    functions = tuple(structure.functions)
    function_set = set(functions)
    classes = tuple(span for span in structure.symbols if span not in function_set)
    found: list[Symbol] = []
    for kind, spans in (("function", functions), ("class", classes)):
        if kind not in kinds:
            continue
        for span in spans:
            found.append(Symbol(span, kind, holder_of(structure.symbols, span)))
    return tuple(found)


def read_source(index: CodeIndex, symbol: Symbol) -> CodeSlice:
    """The source behind a measured symbol, read through the index so evidence quotes itself."""
    return index.read_slice(symbol.span, origin="statistics")


def largest_functions(
    index: CodeIndex,
    scope: Sequence[str] | None = None,
    *,
    kinds: Sequence[SymbolKind] = ("function",),
    limit: int | None = None,
    held: bool = True,
) -> Largest:
    """Rank the symbols of ``scope`` by the lines they fill, widest first, keeping every tie.

    ``limit`` keeps that many of the widest; the measurement in ``measured`` keeps all of them
    either way, so a limit never hides a count. Methods and nested functions are functions to the
    index, so they join the ranking by default; ``held=False`` ranks only what nothing else holds.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit says how many of the widest symbols to keep, so it needs a positive number")
    coverage = scan_coverage(index, scope)
    measured = [
        symbol
        for path in coverage.measured
        for symbol in symbol_spans(index, path, kinds)
        if held or not symbol.held
    ]
    measured.sort(key=lambda symbol: (-symbol.size, symbol.span.file, symbol.span.start, symbol.span.end))
    return Largest(
        measured=tuple(measured),
        largest=tuple(measured if limit is None else measured[:limit]),
        coverage=coverage,
        limit=limit,
    )


def count_symbols(
    index: CodeIndex,
    scope: Sequence[str] | None = None,
    *,
    kinds: Sequence[SymbolKind] = SYMBOL_KINDS,
    largest_of: int | None = None,
    held: bool = True,
) -> SymbolCount:
    """Count the functions and classes of ``scope``, per file and in total, with its scan coverage.

    A function is counted where the index finds it, including inside a class or another function,
    and a file that only partly parsed reports what was recovered with a coverage statement that
    refuses to call that a complete count.
    """
    largest = largest_functions(index, scope, kinds=kinds, limit=largest_of, held=held)
    per_file = {}
    for path in largest.coverage.measured:
        found = symbol_spans(index, path, kinds)
        per_file[path] = FileSymbols(
            file=path,
            functions=tuple(symbol for symbol in found if symbol.kind == "function"),
            classes=tuple(symbol for symbol in found if symbol.kind == "class"),
            parsed=path not in largest.coverage.unparsed,
            kinds=tuple(kinds),
        )
    return SymbolCount(per_file=per_file, largest=largest, kinds=tuple(kinds))
