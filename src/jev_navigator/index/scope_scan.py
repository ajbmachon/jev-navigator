"""Syntax facts extracted together in one ast-grep pass over each requested file set.

Each requested file set is handed to ast-grep scans of a few hundred files each, which schedule
parsing across ast-grep's own worker pool; every match becomes its fact as it is printed, decoded
into only the fields its fact is built from (``ParserMatch``). The structure rules also match the
grammar's ERROR nodes: a file the parser could only recover partially is reported as unparsed too.
A JavaScript file the JavaScript grammar only partly reads is read once more as flow, which reads
Flow types written without the ``@flow`` pragma. Its matched symbols and calls still count —
recovery keeps what it could — but whatever the ERROR nodes swallowed is unknown, not absent.
``FileFacts.unparsed_lines`` keeps the lines those nodes span, so a lookup can tell which names they
may hide.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple, NotRequired, Protocol, TypedDict, TypeVar

import msgspec

from . import tools
from .imports import _exported, _local
from .languages import (
    CLASS_KINDS,
    COMMONJS_EXPORT_PAIR,
    COMMONJS_EXPORT_TARGET,
    COMMONJS_EXPORTS_OBJECT,
    DECLARED_NAME_RULES,
    DECORATED_KINDS,
    EXPRESSION_KINDS,
    FLOW_LANGUAGE,
    FUNCTION_KINDS,
    LOCAL_NAME_RULES,
    MODULE_ALIAS_RULES,
    NAME_HOLDERS,
    NAME_WRAPPERS,
    NAMESPACE_KINDS,
    PROPERTY_TARGET,
    STUB_RULES,
    TYPE_AND_VALUE_DECLARATIONS,
    TYPE_DECLARATIONS,
    VALUE_DECLARATIONS,
    VALUE_KINDS,
    export_rules,
    grammar_of,
    language_of,
    parse_language,
    reference_rules,
    sgconfig_of,
)
from .spans import Span

# The language whose files the JavaScript grammar only partly reads are read once more as flow.
READ_AGAIN_AS_FLOW = "javascript"


@dataclass
class Unparsed:
    """Files with grammar ERROR nodes the parser only recovered partially.

    Lookups keep recovered matches while treating anything the parser omitted as unknown.
    """

    files_by_scan: dict[str, set[str]] = field(default_factory=dict)

    def add(self, scan: str, files: Sequence[str]) -> None:
        self.files_by_scan.setdefault(scan, set()).update(files)

    @property
    def files(self) -> frozenset[str]:
        return frozenset(file for files in self.files_by_scan.values() for file in files)


class LocalName(NamedTuple):
    """``name`` is bound by the function on lines ``first`` to ``last`` for its own body."""

    first: int
    last: int
    name: str


class ModuleAlias(NamedTuple):
    """``name`` holds the whole script module ``specifier`` names, bound by module-level code."""

    name: str
    specifier: str


class NamespaceMember(NamedTuple):
    """``span`` is a member of the TypeScript namespace on lines ``first`` to ``last``: a function,
    class or declaration directly in its body."""

    first: int
    last: int
    span: Span


@dataclass(frozen=True)
class FileStructure:
    """``module_symbols`` are the symbols their module names: one of their syntax nodes no function,
    class, namespace or object literal holds, and no property assignment names. Symbols sharing a
    line each hold the other's first line, so lines alone cannot tell. ``declarations`` are
    module-level and namespace-level; ``namespace_members`` says which symbols and declarations a
    namespace holds. ``commonjs_exports`` are the
    functions and classes assigned to CommonJS exports (``exports.run = function () {}``), which
    another module imports by name but their own module never names.
    ``type_declarations`` and ``value_declarations`` are the declarations a type use and a value
    use may name, decided by each declaration's own syntax node. ``local_names`` are the names each
    function binds for its own body (see ``LOCAL_NAME_RULES``); a name a module-level block binds
    is not among them."""

    functions: tuple[Span, ...] = ()
    symbols: tuple[Span, ...] = ()
    declarations: tuple[Span, ...] = ()
    module_symbols: tuple[Span, ...] = ()
    commonjs_exports: tuple[Span, ...] = ()
    type_declarations: tuple[Span, ...] = ()
    value_declarations: tuple[Span, ...] = ()
    local_names: tuple[LocalName, ...] = ()
    namespace_members: tuple[NamespaceMember, ...] = ()
    # (start, end, first decorator line) of each function whose decorators sit before its first line.
    decorated: tuple[tuple[int, int, int], ...] = ()
    # (start, end) of each function whose body only declares a shape (``languages.STUB_RULES``).
    stubs: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class CallMatch:
    file: str
    line: int
    name: str
    receiver: str | None


@dataclass(frozen=True, order=True)
class ReferenceMatch:
    """One per file, line, role and name. ``receiver`` is what a qualified argument such as
    ``self.handler`` is read from, None for a bare name."""

    file: str
    line: int
    role: str
    name: str
    receiver: str | None = None


@dataclass(frozen=True)
class FileFacts:
    structure: FileStructure
    calls: tuple[CallMatch, ...]
    references: tuple[ReferenceMatch, ...]
    incomplete: bool = False
    export_names: tuple[str, ...] = ()
    # The first and last line of each stretch the grammar's ERROR nodes span, in file order.
    unparsed_lines: tuple[tuple[int, int], ...] = ()
    module_aliases: tuple[ModuleAlias, ...] = ()
    # The names a script module exports as values: its CommonJS exports (see ``EXPORTED_VALUES``).
    # Like ``export_names``, each is its definition's own name unless ``renamed_exports`` says
    # otherwise.
    exported_values: tuple[str, ...] = ()
    # Each name a script module exports a definition of another name under, with that name:
    # ("outer", "inner") for `export { inner as outer }`, ("parse", "urlParse") for `exports.parse =
    # urlParse` or `module.exports = { parse: urlParse }`, and ("default", "build") for its default
    # export, `export default build` or `module.exports = build` (see ``DEFAULT_EXPORTS``).
    renamed_exports: tuple[tuple[str, str], ...] = ()
    # Why the guard kept the file from the parser; such facts are empty and are never cached.
    refusal: str | None = None
    # The language the facts were read as: ``flow`` for JavaScript read with the tsx grammar. None
    # when no grammar read the file.
    language: str | None = None


class _Text(TypedDict):
    text: str


class _Start(TypedDict):
    line: int


class _CapturedRange(TypedDict):
    start: _Start


class _CapturedNode(TypedDict):
    range: _CapturedRange


class _Captured(TypedDict, total=False):
    CALLEE: _Text
    NAME: _Text
    OWN: _Text
    SPEC: _Text
    DECORATOR: _CapturedNode


class _MetaVariables(TypedDict, total=False):
    single: _Captured


class _End(TypedDict):
    line: int


class _ByteOffset(TypedDict):
    start: int
    end: int


class _Range(TypedDict):
    start: _Start
    end: _End
    byteOffset: _ByteOffset


class ParserMatch(TypedDict):
    """The fields of one printed ast-grep match that ``_FileFound`` reads. Every other field (the
    labels of related nodes, the other metavariables) is skipped while decoding instead of becoming
    Python objects, which makes decoding a large scan's output several times faster."""

    ruleId: str
    file: str
    text: str
    lines: str
    range: _Range
    metaVariables: NotRequired[_MetaVariables]


decode_match = msgspec.json.Decoder(ParserMatch).decode


def scan_facts(contents: Mapping[str, bytes], root: Path, unparsed: Unparsed) -> dict[str, FileFacts]:
    """Parse supported source files once; return empty facts for unsupported paths. ``contents`` holds
    each file's first-read bytes, which decide the language it is read as. Each match is turned
    into its fact as the parser prints it, so memory holds facts, never the parser's output."""
    found = {file: _FileFound(file) for file in contents}
    refused: dict[str, str] = {}
    plain, flow = _split_by_pragma(contents)
    found.update(_scanned(plain, root, refused))
    found.update(_scanned(flow, root, refused, as_flow=True))
    found.update(_read_as_flow_where_fewer_lines_fail(found, root, refused))
    unparsed.add("facts", [file for file, facts in found.items() if facts.error_lines])
    incomplete = unparsed.files
    return {
        file: replace(facts.finished(file in incomplete), refusal=refused[file])
        if file in refused
        else facts.finished(file in incomplete)
        for file, facts in found.items()
    }


def _scanned(
    files: Sequence[str], root: Path, refused: dict[str, str], *, as_flow: bool = False
) -> dict[str, _FileFound]:
    """The facts of ``files`` from one ast-grep scan, each file read as its own language, or all of
    them as flow."""
    if not files:
        return {}
    found = {file: _FileFound(file, FLOW_LANGUAGE if as_flow else language_of(file)) for file in files}
    languages = [FLOW_LANGUAGE] if as_flow else sorted({found[file].language for file in files})
    config = sgconfig_of(FLOW_LANGUAGE) if as_flow else None
    matches = tools.ast_grep_rules(
        fact_rules(languages), files, root, config=config, refused=refused, decode=decode_match
    )
    for match in matches:
        found[match["file"]].add(match)
    return found


def _read_as_flow_where_fewer_lines_fail(
    found: dict[str, _FileFound], root: Path, refused: dict[str, str]
) -> dict[str, _FileFound]:
    """Babel strips Flow types from every file it builds, so Flow-typed JavaScript often carries no
    ``@flow`` pragma. Each JavaScript file the JavaScript grammar only partly read is read again as
    flow, and that reading replaces the first one when its ERROR nodes span fewer lines."""
    partly_read = [
        file
        for file, facts in found.items()
        if facts.error_lines and language_of(file) == READ_AGAIN_AS_FLOW and file not in refused
    ]
    if not partly_read:
        return {}
    as_flow = _scanned(partly_read, root, refused, as_flow=True)
    return {
        file: facts
        for file, facts in as_flow.items()
        if file not in refused and facts.unread_line_count() < found[file].unread_line_count()
    }


def fact_rules(languages: Sequence[str]) -> str:
    """The ast-grep rules one scan of files in ``languages`` runs."""
    return "\n---\n".join(
        part
        for part in (
            _structure_rules(languages),
            _call_rules(languages),
            reference_rules(languages),
            export_rules(languages),
            _module_alias_rules(languages),
        )
        if part
    )


@dataclass
class _FileFound:
    """One file's facts, collected match by match, read as ``language``."""

    file: str
    language: str | None = None
    functions: set[Span] = field(default_factory=set)
    classes: set[Span] = field(default_factory=set)
    # Each function's and class's byte range, end exclusive, with its span.
    ranges: list[tuple[int, int, Span]] = field(default_factory=list)
    marks: dict[str, set[tuple[int, int]]] = field(
        default_factory=lambda: {rule: set() for rule in _MARK_RULES}
    )
    namespaces: list[_Namespace] = field(default_factory=list)
    declaration_nodes: list[_Declaration] = field(default_factory=list)
    declared_names: list[tuple[int, str]] = field(default_factory=list)
    bound_names: list[tuple[int, str]] = field(default_factory=list)
    decorated: set[tuple[int, int, int]] = field(default_factory=set)
    stubs: set[tuple[int, int]] = field(default_factory=set)
    calls: list[tuple[tuple[str, int, int], CallMatch]] = field(default_factory=list)
    receivers: dict[tuple[str, int, str, str], set[str | None]] = field(default_factory=dict)
    export_names: set[str] = field(default_factory=set)
    exported_values: set[str] = field(default_factory=set)
    renamed_exports: set[tuple[str, str]] = field(default_factory=set)
    module_aliases: list[tuple[int, ModuleAlias]] = field(default_factory=list)
    error_lines: list[tuple[int, int]] = field(default_factory=list)

    def add(self, match: dict) -> None:
        rule = match["ruleId"]
        if rule == _ERROR_RULE:
            # The grammar reports ERROR nodes here: whatever recovery swallowed is unknown, while the
            # symbols it did keep are still matched.
            self.error_lines.append(_lines_of(match))
        elif rule in _STRUCTURE_RULE_IDS:
            self._add_structure(match)
        elif rule == _DECORATED_RULE:
            self.decorated.add((*_lines_of(match), _captured_line(match, "DECORATOR")))
        elif rule == _STUB_RULE:
            self.stubs.add(_lines_of(match))
        elif rule == "call":
            self._add_call(match)
        elif rule in _EXPORT_RULE_IDS:
            self._add_export(match)
        elif rule == _EXPORTED_VALUE_RULE:
            self._add_exported_value(match)
        elif rule == _DEFAULT_EXPORT_RULE:
            self._add_renamed_export("default", _captured_name(match))
        elif rule == _MODULE_ALIAS_RULE:
            offset = match["range"]["byteOffset"]["start"]
            self.module_aliases += [(offset, alias) for alias in _module_aliases_of(match)]
        else:
            self._add_reference(match)

    def unread_line_count(self) -> int:
        return sum(end - start + 1 for start, end in _merged_stretches(self.error_lines))

    def finished(self, incomplete: bool) -> FileFacts:
        return FileFacts(
            self._structure(),
            tuple(call for _, call in sorted(self.calls, key=lambda entry: entry[0])),
            self._references(),
            incomplete,
            tuple(sorted(self.export_names)),
            _merged_stretches(self.error_lines),
            tuple(alias for _, alias in sorted(self.module_aliases)),
            tuple(sorted(self.exported_values)),
            tuple(sorted(self.renamed_exports)),
            language=self.language,
        )

    def _structure(self) -> FileStructure:
        functions = self.functions - _same_lines_as_a_named_symbol(self.functions | self.classes)
        symbols = functions | self.classes
        positions = _source_positions(self.ranges)
        declared = _declarations(self.file, self.declaration_nodes, self.declared_names)
        # A function held by a value or assigned to a property is the value's or the property's, and
        # one named only by itself names nothing outside itself.
        owned = self.marks[_HELD_RULE] | self.marks[_PROPERTY_VALUE_RULE] | self.marks[_SELF_NAMED_RULE]
        nodes = _Nodes(self.ranges, owned, self.namespaces)
        return FileStructure(
            _ordered(functions, positions),
            _ordered(symbols, positions),
            _sorted(span for span, _ in declared),
            _ordered(symbols & nodes.module_level(), positions),
            _ordered(_marked(self.ranges, self.marks[_MODULE_EXPORT_RULE]), positions),
            _sorted(span for span, declaration in declared if declaration.kind.named_by_types),
            _sorted(span for span, declaration in declared if declaration.kind.named_by_values),
            _local_names(self.ranges, self.classes, self.bound_names),
            nodes.namespace_members(declared),
            tuple(sorted(self.decorated)),
            tuple(sorted(self.stubs)),
        )

    def _references(self) -> tuple[ReferenceMatch, ...]:
        """When one line passes both ``x.name`` and ``name`` in the same role, the plain name stands
        for that line, as a plain call does for callers."""
        return tuple(
            ReferenceMatch(*key, None if None in found else min(found))
            for key, found in sorted(self.receivers.items())
        )

    def _add_structure(self, match: dict) -> None:
        rule, (start, end) = match["ruleId"], _lines_of(match)
        offsets = match["range"]["byteOffset"]
        if rule in _MARK_RULES:
            self.marks[rule].add((offsets["start"], offsets["end"]))
        elif rule == _NAMESPACE_RULE:
            self.namespaces.append(_Namespace(offsets["start"], offsets["end"], start, end))
        elif rule == _DECLARED_NAME_RULE:
            self.declared_names.append((offsets["start"], match["text"]))
        elif rule == _LOCAL_NAME_RULE:
            self.bound_names.append((offsets["start"], match["text"]))
        elif rule in _DECLARATION_BY_RULE:
            kind = _DECLARATION_BY_RULE[rule]
            self.declaration_nodes.append(_Declaration(offsets["start"], offsets["end"], start, end, kind))
        else:
            target = self.functions if rule == "function" else self.classes
            # The syntax tree names the symbol, never a physical line: a method on a one-line class
            # shares the line `class Box` opens, and naming it from that line would collapse it into
            # the class's span.
            span = Span(self.file, start, end, symbol_name(_captured_name(match)))
            target.add(span)
            self.ranges.append((offsets["start"], offsets["end"], span))

    def _add_call(self, match: dict) -> None:
        expression = match["metaVariables"]["single"]["CALLEE"]["text"]
        name = last_identifier(expression)
        if name:
            call = CallMatch(match["file"], _line_of(match), name, receiver_of(expression))
            self.calls.append((_outer_first(match), call))

    def _add_reference(self, match: dict) -> None:
        role, text = match["ruleId"], match["text"]
        key = (match["file"], _line_of(match), role, _reference_name(role, text))
        self.receivers.setdefault(key, set()).add(_reference_receiver(role, text))

    def _add_export(self, match: dict) -> None:
        """The names the module exports from its own definitions: exported declarations' name
        nodes, and the entries of its own ``{ ... }`` lists, an entry under another name with the
        definition it exports (``outer`` and ``inner`` for ``inner as outer``)."""
        text = match["text"]
        if match["ruleId"] == _EXPORT_STATEMENT_RULE:
            self.export_names.add(text)
            return
        self.export_names.add(_local(text))
        self._add_renamed_export(_local(text), _exported(text))

    def _add_exported_value(self, match: dict) -> None:
        name = _captured_name(match)
        self.exported_values.add(name)
        self._add_renamed_export(name, _captured_own_name(match))

    def _add_renamed_export(self, exported: str, own: str) -> None:
        if exported != own:
            self.renamed_exports.add((exported, own))


def _local_names(
    ranges: list[tuple[int, int, Span]], classes: set[Span], names: list[tuple[int, str]]
) -> tuple[LocalName, ...]:
    """Each bound name with the lines of the innermost function holding it. A name a class body binds
    (a Python class attribute) reaches none of its methods, so it is no function's local name."""
    ordered = sorted((_Node(start, end, span) for start, end, span in ranges), key=lambda node: node.start)
    starts = [node.start for node in ordered]
    found = {
        LocalName(holder.span.start, holder.span.end, name)
        for offset, name in names
        if (holder := _innermost(ordered, starts, offset)) is not None and holder.span not in classes
    }
    return tuple(sorted(found))


@dataclass(frozen=True)
class _Node:
    """A function's or class's byte range, end exclusive."""

    start: int
    end: int
    span: Span


@dataclass(frozen=True)
class _DeclarationKind:
    """What may name the declarations one rule matches: a type use, a value use, or both."""

    rule_id: str
    named_by_types: bool
    named_by_values: bool


_DECLARATION_RULES = {
    _DeclarationKind("type_declaration", True, False): TYPE_DECLARATIONS,
    _DeclarationKind("value_declaration", False, True): VALUE_DECLARATIONS,
    _DeclarationKind("declaration", True, True): TYPE_AND_VALUE_DECLARATIONS,
}
_DECLARATION_BY_RULE = {kind.rule_id: kind for kind in _DECLARATION_RULES}


@dataclass(frozen=True)
class _Declaration:
    """A declaration's byte range, end exclusive, its first and last line, and its kind."""

    start: int
    end: int
    first_line: int
    last_line: int
    kind: _DeclarationKind


def _declarations(
    file: str, nodes: list[_Declaration], names: list[tuple[int, str]]
) -> set[tuple[Span, _Declaration]]:
    """One span per name a declaration binds, over the lines of the innermost declaration holding
    the name, with that declaration."""
    ordered = sorted(nodes, key=lambda node: node.start)
    starts = [node.start for node in ordered]
    return {
        (Span(file, holder.first_line, holder.last_line, name), holder)
        for offset, name in names
        if (holder := _innermost(ordered, starts, offset)) is not None
    }


def _sorted(spans: Iterable[Span]) -> tuple[Span, ...]:
    return tuple(sorted(set(spans)))


class _ByteRange(Protocol):
    @property
    def end(self) -> int: ...


_Nested = TypeVar("_Nested", bound=_ByteRange)


def _innermost(ordered: Sequence[_Nested], starts: list[int], offset: int) -> _Nested | None:
    """Nodes nest or are disjoint, so the latest-starting node that reaches past ``offset`` holds it
    most closely."""
    for node in reversed(ordered[: bisect_right(starts, offset)]):
        if offset < node.end:
            return node
    return None


def _source_positions(ranges: Iterable[tuple[int, int, Span]]) -> dict[Span, int]:
    positions: dict[Span, int] = {}
    for start, _, span in ranges:
        positions[span] = min(positions.get(span, start), start)
    return positions


def _marked(ranges: list[tuple[int, int, Span]], marked: set[tuple[int, int]]) -> set[Span]:
    return {span for start, end, span in ranges if (start, end) in marked}


@dataclass(frozen=True)
class _Namespace:
    """A namespace node's byte range, end exclusive, and its first and last line."""

    start: int
    end: int
    first_line: int
    last_line: int


@dataclass(frozen=True)
class _Nodes:
    """A file's function and class nodes (``ranges``), the ones a value or a property holds or
    that only their own name names (``owned``), and its namespace nodes. Nodes nest or are
    disjoint, so a sweep in source order keeps the nodes still open on a stack, the innermost on
    top."""

    ranges: list[tuple[int, int, Span]]
    owned: set[tuple[int, int]]
    namespaces: list[_Namespace]

    def module_level(self) -> set[Span]:
        """The spans with a node nothing owns and no other function, class or namespace holds,
        counting the callbacks ``_same_lines_as_a_named_symbol`` drops: a function inside a one-line
        callback is the callback's. A span is lines and a name, so one such node is enough:
        `function handler() {} const table = { handler() {} };` on one line is one span holding a
        module-level function."""
        return {
            span
            for start, end, span, holder in self._sweep(())
            if holder is None and isinstance(span, Span) and (start, end) not in self.owned
        }

    def namespace_members(self, declared: Iterable[tuple[Span, _Declaration]]) -> tuple[NamespaceMember, ...]:
        """Each function, class and declaration whose innermost holder is a namespace, with that
        namespace's lines; a declaration enters the sweep as the point it starts at."""
        if not self.namespaces:
            return ()
        points = [(declaration.start, declaration.start, span) for span, declaration in declared]
        members = {
            NamespaceMember(holder.first_line, holder.last_line, span)
            for start, end, span, holder in self._sweep(points)
            if isinstance(holder, _Namespace) and isinstance(span, Span) and (start, end) not in self.owned
        }
        return tuple(sorted(members))

    def _sweep(self, points: Iterable[tuple[int, int, Span]]):
        """Each node and point with the innermost node holding it, in source order."""
        entries = [*self.ranges, *((ns.start, ns.end, ns) for ns in self.namespaces), *points]
        open_nodes: list[tuple[int, Span | _Namespace]] = []
        for start, end, item in sorted(entries, key=lambda entry: (entry[0], -entry[1])):
            while open_nodes and open_nodes[-1][0] <= start:
                open_nodes.pop()
            yield start, end, item, open_nodes[-1][1] if open_nodes else None
            if end > start:
                open_nodes.append((end, item))


def _same_lines_as_a_named_symbol(symbols: set[Span]) -> set[Span]:
    """Anonymous functions spanning exactly a named symbol's lines: ``xs.map((x) => x.id)`` on the
    one line of ``ids``. A place is lines, so such a callback is that symbol; kept apart, it would
    be a second place on the same lines."""
    named = {(span.start, span.end) for span in symbols if span.name != "<anonymous>"}
    return {span for span in symbols if span.name == "<anonymous>" and (span.start, span.end) in named}


def _merged_stretches(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """ERROR node lines with nested and overlapping nodes merged into one stretch."""
    stretches: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if stretches and start <= stretches[-1][1]:
            stretches[-1] = (stretches[-1][0], max(end, stretches[-1][1]))
        else:
            stretches.append((start, end))
    return tuple(stretches)


def _outer_first(match: dict) -> tuple[str, int, int]:
    """A match's place in its file, ordering the outer of two matches that start together first."""
    offsets = match["range"]["byteOffset"]
    return match["file"], offsets["start"], -offsets["end"]


# Roles whose node may be a qualified name, `x.name` or `pkg.mod.Name`: named by its last part, with
# the rest as its receiver.
_QUALIFIED_ROLES = frozenset({"argument", "base"})


def _reference_name(role: str, text: str) -> str:
    return last_identifier(text) if role in _QUALIFIED_ROLES else text


def _reference_receiver(role: str, text: str) -> str | None:
    return receiver_of(text) if role in _QUALIFIED_ROLES else None


OPAQUE_RECEIVER = "<expression>"


def receiver_of(expression: str) -> str | None:
    """What ``expression`` reads its last name from: a plain chain of names (``this.store``), the
    placeholder ``OPAQUE_RECEIVER`` for anything else (a call, a subscript, a literal, a template),
    or None when it reads it from nothing. Any other receiver text could quote a string literal, so
    it is never kept."""
    head, dot, _ = expression.replace("?.", ".").rpartition(".")
    if not dot:
        return None
    return head if _is_name_chain(head) else OPAQUE_RECEIVER


def _is_name_chain(expression: str) -> bool:
    """Names joined by dots, such as ``this.store``, ``self.items`` or ``super``, which quote no code."""
    return all(part.removeprefix("#").replace("$", "_").isidentifier() for part in expression.split("."))


def last_identifier(expression: str) -> str:
    tail = expression.replace("?.", ".").split(".")[-1]
    return tail if tail.isidentifier() else ""


def first_identifier(expression: str) -> str:
    """The name ``expression`` starts from: ``db`` in ``db.pool``; empty when it starts from a call
    or an item (``getDb().pool``)."""
    head = expression.replace("?.", ".").split(".")[0].strip()
    return head if head.isidentifier() else ""


def symbol_name(captured: str) -> str:
    """The name a captured name node gives a symbol: ``save`` for ``save``, ``this.save``,
    ``exports.save``, ``#save`` and the key ``"save"``. A computed key, a string key that is no
    identifier (``"risk.triage"``) and a missing name leave it anonymous."""
    quoted = captured[:1] in ("'", '"')
    name = captured[1:-1] if quoted else last_identifier(captured.replace("#", ""))
    return name if name.isidentifier() else "<anonymous>"


def _captured_name(match: dict) -> str:
    return match.get("metaVariables", {}).get("single", {}).get("NAME", {}).get("text", "")


def _captured_own_name(match: dict) -> str:
    """The definition an exported value names, `$OWN`; a shorthand entry `{ log }` captures only
    `$NAME`, its own."""
    captured = match.get("metaVariables", {}).get("single", {})
    return captured.get("OWN", captured.get("NAME", {})).get("text", "")


def _captured_line(match: dict, variable: str) -> int:
    return match["metaVariables"]["single"][variable]["range"]["start"]["line"] + 1


_ERROR_RULE = "parse_error"
_HELD_RULE = "held"
_NAMESPACE_RULE = "namespace"
_DECLARED_NAME_RULE = "declared_name"
_LOCAL_NAME_RULE = "local_name"
_MODULE_ALIAS_RULE = "module_alias"
_PROPERTY_VALUE_RULE = "property_value"
_SELF_NAMED_RULE = "self_named"
_MODULE_EXPORT_RULE = "module_export"
_MARK_RULES = (_HELD_RULE, _PROPERTY_VALUE_RULE, _MODULE_EXPORT_RULE, _SELF_NAMED_RULE)
_STRUCTURE_RULE_IDS = frozenset(
    {
        "function",
        "class",
        *_DECLARATION_BY_RULE,
        _DECLARED_NAME_RULE,
        _LOCAL_NAME_RULE,
        *_MARK_RULES,
        _NAMESPACE_RULE,
        _ERROR_RULE,
    }
)
_EXPORT_STATEMENT_RULE = "export_surface"
_EXPORT_SPECIFIER_RULE = "export_specifier"
_EXPORT_RULE_IDS = (_EXPORT_STATEMENT_RULE, _EXPORT_SPECIFIER_RULE)
_EXPORTED_VALUE_RULE = "exported_value"
_DEFAULT_EXPORT_RULE = "default_export"


def _module_aliases_of(match: dict) -> list[ModuleAlias]:
    """A script module is the captured string without its quotes. A Python import without ``as``
    makes each dotted prefix of its module reach the module of that name: ``app`` and ``app.jobs``
    for ``import app.jobs``."""
    captured = match["metaVariables"]["single"]
    name = captured["NAME"]["text"]
    if "SPEC" not in captured:
        parts = name.split(".")
        return [
            ModuleAlias(prefix, prefix)
            for prefix in (".".join(parts[:end]) for end in range(1, len(parts) + 1))
        ]
    module = captured["SPEC"]["text"]
    return [ModuleAlias(name, module[1:-1] if module[:1] in ("'", '"') else module)]


def _module_alias_rules(languages: Sequence[str]) -> str:
    return "\n---\n".join(
        _rule_document(_MODULE_ALIAS_RULE, language, rule)
        for language in languages
        for rule in MODULE_ALIAS_RULES[language]
    )


_DECORATED_RULE = "decorated"
_STUB_RULE = "stub"
# A walk back from a function past its decorators stops at the first sibling that is neither.
_END_OF_DECORATORS = "{not: {any: [{kind: decorator}, {kind: comment}]}}"


def _structure_rules(languages: Sequence[str]) -> str:
    documents = []
    for language in languages:
        documents.append(_kind_rule("function", language, FUNCTION_KINDS[language]))
        documents.append(_kind_rule("class", language, CLASS_KINDS[language]))
        documents += [
            _rule_document(kind.rule_id, language, rules[language])
            for kind, rules in _DECLARATION_RULES.items()
            if language in rules
        ]
        documents.append(_rule_document(_DECLARED_NAME_RULE, language, DECLARED_NAME_RULES[language]))
        documents.append(_rule_document(_LOCAL_NAME_RULE, language, LOCAL_NAME_RULES[language]))
        if DECORATED_KINDS[language]:
            documents.append(_decorated_rule(language))
        if language in STUB_RULES:
            documents.append(_rule_document(_STUB_RULE, language, STUB_RULES[language]))
        documents.append(_rule_document(_ERROR_RULE, language, "  kind: ERROR"))
        if VALUE_KINDS[language]:
            documents.append(_held_rule(language))
            documents += _property_rules(language)
            documents.append(_self_named_rule(language))
        if NAMESPACE_KINDS[language]:
            documents.append(
                _rule_document(_NAMESPACE_RULE, language, f"  any: {_kinds(NAMESPACE_KINDS[language])}")
            )
    return "\n---\n".join(documents)


def _rule_document(rule_id: str, language: str, rule: str) -> str:
    return f"id: {rule_id}\nlanguage: {grammar_of(language)}\nrule:\n{rule}"


def _property_rules(language: str) -> list[str]:
    """Every function and class assigned to a property, and every one that is a CommonJS export
    (see ``PROPERTY_TARGET``)."""
    symbol_kinds = (*FUNCTION_KINDS[language], *CLASS_KINDS[language])
    symbols = _kinds(symbol_kinds)
    expressions = _kinds([kind for kind in symbol_kinds if kind in EXPRESSION_KINDS])
    export = (
        "  any:\n"
        f"    - {{any: {symbols}, {_held_past_wrappers(language, COMMONJS_EXPORT_TARGET)}}}\n"
        f"    - {{kind: method_definition, not: {{not: {{inside: {COMMONJS_EXPORTS_OBJECT}}}}}}}\n"
        f"    - {{any: {expressions}, {_held_past_wrappers(language, COMMONJS_EXPORT_PAIR)}}}"
    )
    return [
        _rule_document(
            _PROPERTY_VALUE_RULE,
            language,
            f"  any: {symbols}\n  {_held_past_wrappers(language, PROPERTY_TARGET)}",
        ),
        _rule_document(_MODULE_EXPORT_RULE, language, export),
    ]


def _held_past_wrappers(language: str, holder: str) -> str:
    """A condition, printed by no match (see ``languages``): the first ancestor no wrapper (see
    ``NAME_WRAPPERS``) is ``holder``."""
    return f"not: {{not: {{inside: {{stopBy: {_past_wrappers(language)}, any: [{holder}]}}}}}}"


def _past_wrappers(language: str) -> str:
    """A ``stopBy`` that ends an ancestor search at the first node no wrapper (see ``NAME_WRAPPERS``)."""
    return f"{{not: {{any: {_kinds(NAME_WRAPPERS[grammar_of(language)])}}}}}"


def _held_rule(language: str) -> str:
    """Every function and class anywhere inside a value node (see ``VALUE_KINDS``)."""
    symbols = _kinds((*FUNCTION_KINDS[language], *CLASS_KINDS[language]))
    holders = _kinds(VALUE_KINDS[language])
    return _rule_document(
        _HELD_RULE, language, f"  any: {symbols}\n  not: {{not: {{inside: {{stopBy: end, any: {holders}}}}}}}"
    )


def _decorated_rule(language: str) -> str:
    """Each function whose decorators sit before it, its first decorator captured as ``$DECORATOR``:
    among the decorators and comments just before the function, the decorator no other one precedes."""
    first_decorator = (
        f"{{stopBy: {_END_OF_DECORATORS}, kind: decorator, pattern: $DECORATOR, "
        f"not: {{follows: {{stopBy: {_END_OF_DECORATORS}, kind: decorator}}}}}}"
    )
    return _rule_document(
        _DECORATED_RULE, language, f"  any: {_kinds(DECORATED_KINDS[language])}\n  follows: {first_decorator}"
    )


# A component rendered as `<Name ...>` or `<ns.Name ...>` is called by the code that renders it;
# lower-case names are the platform's own elements (`<div>`), defined nowhere in scope.
_JSX_CALL_RULE = """rule:
  any:
    - kind: jsx_opening_element
    - kind: jsx_self_closing_element
  has:
    field: name
    regex: "^[A-Z]|[.][A-Z][^.]*$"
    pattern: $CALLEE"""


def _call_rules(languages: Sequence[str]) -> str:
    documents = []
    for language in languages:
        grammar = grammar_of(language)
        documents.append(f"id: call\nlanguage: {grammar}\nrule:\n  pattern: $CALLEE($$$)")
        if grammar != "python":
            documents.append(f"id: call\nlanguage: {grammar}\nrule:\n  pattern: new $CALLEE($$$)")
        if grammar == "tsx":
            documents.append(f"id: call\nlanguage: {grammar}\n{_JSX_CALL_RULE}")
    return "\n---\n".join(documents)


def _kind_rule(rule_id: str, language: str, kinds: Sequence[str]) -> str:
    """Every node of ``kinds``, with the node that names it captured as ``$NAME``: a declaration's
    own name; an expression's holder (see ``EXPRESSION_KINDS``), else its own name. ``any`` takes
    the first alternative that matches, and a node with neither name still matches, unnamed."""
    grammar = grammar_of(language)
    declarations = [kind for kind in kinds if kind not in EXPRESSION_KINDS]
    expressions = [kind for kind in kinds if kind in EXPRESSION_KINDS]
    alternatives = [f"{{any: {_kinds(declarations)}, has: {_named_by('name')}}}"] if declarations else []
    if expressions:
        alternatives += [
            f"{{any: {_kinds(expressions)}, {_inside_a_holder(language)}}}",
            f"{{any: {_kinds(expressions)}, has: {_named_by('name')}}}",
        ]
    alternatives.append(f"{{any: {_kinds(kinds)}}}")
    listed = "".join(f"\n    - {alternative}" for alternative in alternatives)
    return f"id: {rule_id}\nlanguage: {grammar}\nrule:\n  any:{listed}"


def _inside_a_holder(language: str) -> str:
    """The relation of an expression to the holder that names it (see ``NAME_HOLDERS``): the search
    stops at the first ancestor that is no wrapper, and that one must be the holder."""
    holders = ", ".join(
        f"{{kind: {holder}, has: {_named_by(field)}}}" for holder, field in NAME_HOLDERS[grammar_of(language)]
    )
    return f"inside: {{stopBy: {_past_wrappers(language)}, any: [{holders}]}}"


def _self_named_rule(language: str) -> str:
    """Every function or class expression no holder names, which takes its own name:
    `run(function handler() {})`. That name is bound only inside the expression, so it names
    nothing in its module or namespace."""
    expressions = [
        kind for kind in (*FUNCTION_KINDS[language], *CLASS_KINDS[language]) if kind in EXPRESSION_KINDS
    ]
    rule = (
        f"  any: {_kinds(expressions)}\n  has: {_named_by('name')}\n  not: {{{_inside_a_holder(language)}}}"
    )
    return _rule_document(_SELF_NAMED_RULE, language, rule)


def _kinds(kinds: Sequence[str]) -> str:
    return "[" + ", ".join(f"{{kind: {kind}}}" for kind in kinds) + "]"


def _named_by(field: str) -> str:
    return f"{{field: {field}, pattern: $NAME}}"


def _split_by_pragma(contents: Mapping[str, bytes]) -> tuple[list[str], list[str]]:
    """The supported files read as their own language, and the ``@flow`` files, judged by their
    first-read bytes. The flow files take their own scan, since its config's ``languageGlobs`` would
    parse plain JavaScript with the tsx grammar too."""
    plain: list[str] = []
    flow: list[str] = []
    for file, content in contents.items():
        if language_of(file) is not None:
            (flow if parse_language(file, content) == FLOW_LANGUAGE else plain).append(file)
    return plain, flow


def _ordered(spans: set[Span], positions: dict[Span, int]) -> tuple[Span, ...]:
    """By position, outer first; symbols on the same lines in source order. A line's calls belong to
    the smallest function holding it, and among functions on the same lines the first one wins: in
    `function retry(again = () => 1) { return attempt(); }` that is `retry`, not its default."""
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end, positions[span])))


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1


def _lines_of(match: dict) -> tuple[int, int]:
    return _line_of(match), match["range"]["end"]["line"] + 1
