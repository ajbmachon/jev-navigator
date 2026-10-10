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

import zlib
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple, NotRequired, Protocol, TypedDict, TypeVar

import msgspec

from . import tools
from .imports import COMMENT_RANGE_RECORD, _exported, _local, python_submodule
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
    LOCAL_MEMBER_RULES,
    LOCAL_MODULE_BLOCK_RULES,
    LOCAL_MODULE_RULES,
    LOCAL_NAME_RULES,
    MODULE_ALIAS_RULES,
    MODULE_BINDING_RULES,
    MODULE_VARIABLE,
    NAME_HOLDERS,
    NAME_WRAPPERS,
    NAMESPACE_KINDS,
    PROPERTY_TARGET,
    PYTHON_FROM_IMPORT,
    PYTHON_FROM_IMPORT_NAMES,
    STUB_RULES,
    TYPE_AND_VALUE_DECLARATIONS,
    TYPE_DECLARATIONS,
    VALUE_DECLARATIONS,
    VALUE_KINDS,
    comment_rule,
    export_rules,
    grammar_of,
    language_of,
    parse_language,
    reference_rules,
    sgconfig_of,
)
from .spans import Span, merged_ranges

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
    """``name`` is bound on ``line`` by the function on lines ``first`` to ``last`` for its own body:
    one fact per binding, so a function that binds a name twice has two. ``module`` is the module the
    binding holds as a whole, when it is a ``const`` require (see ``LOCAL_MODULE_RULES``); ``member``
    is the member it holds, as ``object.member``, when it is a ``const`` that unpacks or reads a member
    of a plain name (see ``LOCAL_MEMBER_RULES``)."""

    first: int
    last: int
    name: str
    line: int
    module: str = ""
    # The last line of the block holding a binding that holds ``module`` or ``member``: the binding
    # holds it from ``line`` to here, since a `const` is block-scoped.
    block_end: int = 0
    member: str = ""


class ModuleAlias(NamedTuple):
    """``name`` holds the whole module ``specifier`` names, bound by module-level code. A Python
    ``from pkg import mod`` (``from_import``) binds the package's own ``mod`` instead when the
    package's ``__init__`` binds one; ``from pkg import *`` binds the name ``*`` to ``pkg``."""

    name: str
    specifier: str
    from_import: bool = False


class NamespaceMember(NamedTuple):
    """``span`` is a member of the TypeScript namespace on lines ``first`` to ``last``: a function,
    class or declaration directly in its body."""

    first: int
    last: int
    span: Span


class ObjectMember(NamedTuple):
    """``span`` is a function or class of the object literal the module-level variable ``owner``
    holds: a method, or a property's value (``const api = { list() {}, get: (id) => id }``)."""

    owner: str
    span: Span


class ConstantFunction(NamedTuple):
    """``span`` is a function or class that no function, class or namespace holds and no holder
    names, inside the call or `new` that is the value of the module-level variable ``constant``;
    ``keys`` are the keys of the pairs around it there, outer first: `export const run =
    Effect.fn("run")(function* () {})` holds one with no keys, `createWebRouter({ list:
    procedure.query(() => []) })` one with the key `list`. A callback that builds data holds none:
    `items.map((item) => item.id)`, or one inside `new Map`, `new Set` or `Array.from`."""

    constant: str
    keys: tuple[str, ...]
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
    is not among them. ``object_members`` are the functions and classes of the objects module-level
    variables hold (see ``ObjectMember``), and ``argument_members`` those of the objects a
    module-level call or `new` is passed: `errorFormatter` in `const t = create({ errorFormatter() {}
    })`. ``constant_functions`` are the functions module-level constants built by a call hold (see
    ``ConstantFunction``)."""

    functions: tuple[Span, ...] = ()
    symbols: tuple[Span, ...] = ()
    declarations: tuple[Span, ...] = ()
    module_symbols: tuple[Span, ...] = ()
    commonjs_exports: tuple[Span, ...] = ()
    type_declarations: tuple[Span, ...] = ()
    value_declarations: tuple[Span, ...] = ()
    local_names: tuple[LocalName, ...] = ()
    namespace_members: tuple[NamespaceMember, ...] = ()
    object_members: tuple[ObjectMember, ...] = ()
    argument_members: tuple[Span, ...] = ()
    constant_functions: tuple[ConstantFunction, ...] = ()
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
    # Each name module-level code binds otherwise than by an import or a module symbol, such as
    # `mod = make()` or `for mod in mods` (see ``MODULE_BINDING_RULES``).
    module_bindings: tuple[str, ...] = ()
    # Why the guard kept the file from the parser; such facts are empty and are never cached.
    refusal: str | None = None
    # The language the facts were read as: ``flow`` for JavaScript read with the tsx grammar. None
    # when no grammar read the file.
    language: str | None = None
    # Each member of a script module's default export object, with the definition it holds:
    # ("insert", "insert") and ("utc", "toUtc") for `export default { insert, utc: toUtc }` (see
    # ``DEFAULT_MEMBERS``). A default import reaches them as members.
    default_members: tuple[tuple[str, str], ...] = ()
    # Parser comment-node byte ranges in exact cached bytes, compressed start-delta/length records.
    comment_ranges: bytes = b""


class _Text(TypedDict):
    text: str


class _Start(TypedDict):
    line: int


class _Captured(TypedDict, total=False):
    CALLEE: _Text
    FROM: _Text
    KEY: _Text
    MEMBER: _Text
    NAME: _CapturedNode
    OBJ: _Text
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


class _CapturedNode(TypedDict):
    text: str
    range: _Range


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
    previous = None
    for match in matches:
        current = found[match["file"]]
        if previous is not None and current is not previous:
            previous.compact_comments()
        current.add(match)
        previous = current
    if previous is not None:
        previous.compact_comments()
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
            "\n---\n".join(comment_rule(language) for language in languages),
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
    # The byte offset and the name of each module-level variable whose value is an object.
    object_owners: list[tuple[int, str]] = field(default_factory=list)
    # Each module-level constant built by a call, by its name's offset, and each key of a pair that
    # holds a function in such a value, by the pair's byte range (see ``ConstantFunction``).
    constant_owners: list[tuple[int, str]] = field(default_factory=list)
    constant_keys: list[tuple[int, int, str]] = field(default_factory=list)
    declaration_nodes: list[_Declaration] = field(default_factory=list)
    declared_names: list[tuple[int, str]] = field(default_factory=list)
    bound_names: list[tuple[int, int, str]] = field(default_factory=list)
    # The module each `const` require binds, and the member each member `const` binds, as
    # ``object.member``, by the byte offset of the name it binds.
    local_modules: dict[int, str] = field(default_factory=dict)
    local_members: dict[int, str] = field(default_factory=dict)
    # The byte range, end exclusive, and the last line of each block a `const` require may end in.
    local_module_blocks: list[_Block] = field(default_factory=list)
    decorated: set[tuple[int, int, int]] = field(default_factory=set)
    stubs: set[tuple[int, int]] = field(default_factory=set)
    calls: list[tuple[tuple[str, int, int], CallMatch]] = field(default_factory=list)
    receivers: dict[tuple[str, int, str, str], set[str | None]] = field(default_factory=dict)
    export_names: set[str] = field(default_factory=set)
    exported_values: set[str] = field(default_factory=set)
    default_members: set[tuple[str, str]] = field(default_factory=set)
    renamed_exports: set[tuple[str, str]] = field(default_factory=set)
    module_aliases: list[tuple[int, ModuleAlias]] = field(default_factory=list)
    from_imports: list[_FromImport] = field(default_factory=list)
    # Each name a from-import takes: its byte offset, the name it binds, and the name it imports.
    from_names: list[tuple[int, str, str]] = field(default_factory=list)
    module_bindings: set[str] = field(default_factory=set)
    error_lines: list[tuple[int, int]] = field(default_factory=list)
    comment_ranges: bytearray | bytes = field(default_factory=bytearray)
    comment_start: int = 0

    def add(self, match: dict) -> None:
        rule = match["ruleId"]
        if rule == "comments":
            if isinstance(self.comment_ranges, bytes):
                self.comment_ranges = (
                    bytearray(zlib.decompress(self.comment_ranges)) if self.comment_ranges else bytearray()
                )
            offsets = match["range"]["byteOffset"]
            self.comment_ranges.extend(
                COMMENT_RANGE_RECORD.pack(
                    offsets["start"] - self.comment_start, offsets["end"] - offsets["start"]
                )
            )
            self.comment_start = offsets["start"]
        elif rule == _ERROR_RULE:
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
        elif rule == _DEFAULT_MEMBER_RULE:
            self.default_members.add((_captured_name(match), _captured_own_name(match)))
        elif rule == _MODULE_ALIAS_RULE:
            offset = match["range"]["byteOffset"]["start"]
            self.module_aliases += [(offset, alias) for alias in _module_aliases_of(match)]
        elif rule == _FROM_IMPORT_RULE:
            offsets = match["range"]["byteOffset"]
            self.from_imports.append(_FromImport(offsets["start"], offsets["end"], _captured(match, "FROM")))
        elif rule == _FROM_NAME_RULE:
            name = _captured_name(match)
            imported = _captured(match, "SPEC") or name
            self.from_names.append((match["range"]["byteOffset"]["start"], name, imported))
        elif rule == _MODULE_BINDING_RULE:
            self.module_bindings.add(match["text"])
        else:
            self._add_reference(match)

    def compact_comments(self) -> None:
        """Keep only the current parser file's ranges unpacked; ast-grep groups matches by file."""
        if isinstance(self.comment_ranges, bytearray):
            if self.comment_ranges:
                # Small compression workspace keeps compaction below the fact-scan memory bound.
                compressor = zlib.compressobj(level=1, wbits=9, memLevel=1)
                self.comment_ranges = compressor.compress(self.comment_ranges) + compressor.flush()
            else:
                self.comment_ranges = b""

    def unread_line_count(self) -> int:
        return sum(end - start + 1 for start, end in merged_ranges(self.error_lines))

    def finished(self, incomplete: bool) -> FileFacts:
        self.compact_comments()
        return FileFacts(
            self._structure(),
            tuple(call for _, call in sorted(self.calls, key=lambda entry: entry[0])),
            self._references(),
            incomplete,
            tuple(sorted(self.export_names)),
            tuple(merged_ranges(self.error_lines)),
            tuple(alias for _, alias in sorted([*self.module_aliases, *self._from_import_aliases()])),
            tuple(sorted(self.exported_values)),
            tuple(sorted(self.renamed_exports)),
            tuple(sorted(self.module_bindings)),
            language=self.language,
            default_members=tuple(sorted(self.default_members)),
            comment_ranges=self.comment_ranges,
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
            _local_names(
                self.ranges,
                self.classes,
                self.bound_names,
                _Holdings(self.local_modules, self.local_members),
                self.local_module_blocks,
            ),
            nodes.namespace_members(declared),
            _object_members(self.ranges, self.marks[_OBJECT_MEMBER_RULE], self.object_owners),
            _ordered(symbols & _marked(self.ranges, self.marks[_ARGUMENT_MEMBER_RULE]), positions),
            _constant_functions(
                self.ranges,
                _held_by_no_name(
                    symbols & nodes.outermost(), _marked(self.ranges, self.marks[_SELF_NAMED_RULE])
                )
                - _marked(self.ranges, self.marks[_DATA_CALLBACK_RULE]),
                self.declaration_nodes,
                self.constant_owners,
                self.constant_keys,
            ),
            tuple(sorted(self.decorated)),
            tuple(sorted(self.stubs)),
        )

    def _from_import_aliases(self) -> list[tuple[int, ModuleAlias]]:
        """Each name a module-level from-import binds, with the module of that name in its package
        (see ``PYTHON_FROM_IMPORT``); a name in a from-import inside a function or class has no
        statement holding it here."""
        ordered = sorted(self.from_imports, key=lambda statement: statement.start)
        starts = [statement.start for statement in ordered]
        return [
            (offset, _from_import_alias(statement.package, name, imported))
            for offset, name, imported in self.from_names
            if (statement := _innermost(ordered, starts, offset)) is not None
        ]

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
        elif rule == _OBJECT_OWNER_RULE:
            self.object_owners.append((offsets["start"], match["text"]))
        elif rule == _CONSTANT_OWNER_RULE:
            self.constant_owners.append((offsets["start"], match["text"]))
        elif rule == _CONSTANT_KEY_RULE:
            self.constant_keys.append((offsets["start"], offsets["end"], _key_text(_captured(match, "KEY"))))
        elif rule == _DECLARED_NAME_RULE:
            self.declared_names.append((offsets["start"], match["text"]))
        elif rule == _LOCAL_NAME_RULE:
            self.bound_names.append((offsets["start"], start, match["text"]))
        elif rule == _LOCAL_MODULE_RULE:
            name = match["metaVariables"]["single"]["NAME"]["range"]["byteOffset"]["start"]
            self.local_modules[name] = _unquoted(_captured(match, "SPEC"))
        elif rule == _LOCAL_MEMBER_RULE:
            self.local_members[offsets["start"]] = f"{_captured(match, 'OBJ')}.{_captured(match, 'MEMBER')}"
        elif rule == _LOCAL_MODULE_BLOCK_RULE:
            self.local_module_blocks.append(_Block(offsets["start"], offsets["end"], end))
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


@dataclass(frozen=True)
class _Holdings:
    """What a function's own bindings hold, by the byte offset of the name each binds."""

    modules: dict[int, str]
    members: dict[int, str]


def _local_names(
    ranges: list[tuple[int, int, Span]],
    classes: set[Span],
    names: list[tuple[int, int, str]],
    holdings: _Holdings,
    blocks: list[_Block],
) -> tuple[LocalName, ...]:
    """Each bound name, with its line, the lines of the innermost function holding it, and the module
    or member it holds when ``holdings`` names one at its position, until the end of its block. A name
    a class body binds (a Python class attribute) reaches none of its methods, so it is no function's
    local name."""
    ordered = sorted((_Node(start, end, span) for start, end, span in ranges), key=lambda node: node.start)
    starts = [node.start for node in ordered]
    ordered_blocks = sorted(blocks)
    block_starts = [block.start for block in ordered_blocks]
    found = set()
    for offset, line, name in names:
        holder = _innermost(ordered, starts, offset)
        if holder is None or holder.span in classes:
            continue
        module, member = holdings.modules.get(offset, ""), holdings.members.get(offset, "")
        held = module or member
        block_end = _block_end(holder, ordered_blocks, block_starts, offset) if held else 0
        found.add(LocalName(holder.span.start, holder.span.end, name, line, module, block_end, member))
    return tuple(sorted(found))


def _block_end(holder: _Node, blocks: list[_Block], starts: list[int], offset: int) -> int:
    """The last line of the block a declaration at ``offset`` in ``holder``'s own code sits in: its
    statement block, or the function's body, which ends with the function."""
    block = _innermost(blocks, starts, offset)
    return holder.span.end if block is None else min(block.last_line, holder.span.end)


class _Block(NamedTuple):
    """A block's byte range, end exclusive, and its last line."""

    start: int
    end: int
    last_line: int


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


def _held_by_no_name(spans: set[Span], self_named: set[Span]) -> set[Span]:
    """The spans no holder names: unnamed ones, and those only their own expression names."""
    return {span for span in spans if not span.is_named or span in self_named}


def _constant_functions(
    ranges: list[tuple[int, int, Span]],
    candidates: set[Span],
    declarations: Sequence[_Declaration],
    owners: list[tuple[int, str]],
    keys: list[tuple[int, int, str]],
) -> tuple[ConstantFunction, ...]:
    """Each candidate with the constant built by a call whose module-level declaration holds it,
    and the keys of the pairs around it, outer first, in file order: functions on one line share a
    span, and each is recorded."""
    ordered_owners = sorted(owners)
    found: dict[ConstantFunction, None] = {}
    for start, end, span in sorted(ranges, key=lambda entry: entry[:2]):
        constant = _constant_holding(start, declarations, ordered_owners) if span in candidates else None
        if constant is not None:
            found[ConstantFunction(constant, _keys_around(start, end, keys), span)] = None
    return tuple(found)


def _constant_holding(
    offset: int, declarations: Sequence[_Declaration], ordered_owners: list[tuple[int, str]]
) -> str | None:
    """The constant built by a call named last before ``offset`` in the declaration holding it: one
    statement may declare several, and a declarator's value follows its name."""
    declaration = next((node for node in declarations if node.start <= offset < node.end), None)
    index = bisect_right(ordered_owners, offset, key=lambda owner: owner[0]) - 1
    if declaration is None or index < 0 or ordered_owners[index][0] < declaration.start:
        return None
    return ordered_owners[index][1]


def _keys_around(start: int, end: int, keys: list[tuple[int, int, str]]) -> tuple[str, ...]:
    """The keys of the pairs holding bytes ``start`` to ``end``, outer first."""
    return tuple(key for key_start, key_end, key in sorted(keys) if key_start <= start and end <= key_end)


def _key_text(key: str) -> str:
    """A pair's key without the quotes a string key carries."""
    return key[1:-1] if key[:1] in ("'", '"', "`") else key


def _source_positions(ranges: Iterable[tuple[int, int, Span]]) -> dict[Span, int]:
    positions: dict[Span, int] = {}
    for start, _, span in ranges:
        positions[span] = min(positions.get(span, start), start)
    return positions


def _marked(ranges: list[tuple[int, int, Span]], marked: set[tuple[int, int]]) -> set[Span]:
    return {span for start, end, span in ranges if (start, end) in marked}


def _object_members(
    ranges: list[tuple[int, int, Span]], marked: set[tuple[int, int]], owners: list[tuple[int, str]]
) -> tuple[ObjectMember, ...]:
    """Each marked member with the module-level variable whose object holds it: the last one named
    before it, since module-level variables never nest."""
    ordered = sorted(owners)
    starts = [start for start, _ in ordered]
    members = {
        ObjectMember(ordered[index - 1][1], span)
        for start, end, span in ranges
        if (start, end) in marked and (index := bisect_right(starts, start)) > 0
    }
    return tuple(sorted(members, key=lambda member: member.span))


@dataclass(frozen=True)
class _FromImport:
    """A module-level ``from package import ...`` statement's byte range, end exclusive, and its
    package as written (``app.jobs``, ``.``, ``..lib``)."""

    start: int
    end: int
    package: str


def _from_import_alias(package: str, name: str, imported: str) -> ModuleAlias:
    if imported == "*":
        return ModuleAlias("*", package, from_import=True)
    return ModuleAlias(name, python_submodule(package, imported), from_import=True)


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

    def outermost(self) -> set[Span]:
        """The spans no other function, class or namespace holds, whatever value or property holds
        them: the generator in `const run = Effect.fn("run")(function* () {})`."""
        return {span for _, _, span, holder in self._sweep(()) if holder is None and isinstance(span, Span)}

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
    named = {(span.start, span.end) for span in symbols if span.is_named}
    return {span for span in symbols if not span.is_named and (span.start, span.end) in named}


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
    return _captured(match, "NAME")


def _captured(match: dict, variable: str) -> str:
    return match.get("metaVariables", {}).get("single", {}).get(variable, {}).get("text", "")


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
_LOCAL_MODULE_RULE = "local_module"
_LOCAL_MODULE_BLOCK_RULE = "local_module_block"
_LOCAL_MEMBER_RULE = "local_member"
_MODULE_ALIAS_RULE = "module_alias"
_FROM_IMPORT_RULE = "from_import"
_FROM_NAME_RULE = "from_name"
_MODULE_BINDING_RULE = "module_binding"
_PROPERTY_VALUE_RULE = "property_value"
_SELF_NAMED_RULE = "self_named"
_MODULE_EXPORT_RULE = "module_export"
_OBJECT_MEMBER_RULE = "object_member"
_ARGUMENT_MEMBER_RULE = "argument_member"
_OBJECT_OWNER_RULE = "object_owner"
_CONSTANT_OWNER_RULE = "constant_owner"
_CONSTANT_KEY_RULE = "constant_key"
_DATA_CALLBACK_RULE = "data_callback"
_MARK_RULES = (
    _HELD_RULE,
    _PROPERTY_VALUE_RULE,
    _MODULE_EXPORT_RULE,
    _SELF_NAMED_RULE,
    _OBJECT_MEMBER_RULE,
    _ARGUMENT_MEMBER_RULE,
    _DATA_CALLBACK_RULE,
)
_STRUCTURE_RULE_IDS = frozenset(
    {
        "function",
        "class",
        *_DECLARATION_BY_RULE,
        _DECLARED_NAME_RULE,
        _LOCAL_NAME_RULE,
        _LOCAL_MODULE_RULE,
        _LOCAL_MODULE_BLOCK_RULE,
        _LOCAL_MEMBER_RULE,
        *_MARK_RULES,
        _NAMESPACE_RULE,
        _OBJECT_OWNER_RULE,
        _CONSTANT_OWNER_RULE,
        _CONSTANT_KEY_RULE,
        _ERROR_RULE,
    }
)
_EXPORT_STATEMENT_RULE = "export_surface"
_EXPORT_SPECIFIER_RULE = "export_specifier"
_EXPORT_RULE_IDS = (_EXPORT_STATEMENT_RULE, _EXPORT_SPECIFIER_RULE)
_EXPORTED_VALUE_RULE = "exported_value"
_DEFAULT_EXPORT_RULE = "default_export"
_DEFAULT_MEMBER_RULE = "default_member"


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
    return [ModuleAlias(name, _unquoted(captured["SPEC"]["text"]))]


def _unquoted(module: str) -> str:
    """A script module is the captured string without its quotes."""
    return module[1:-1] if module[:1] in ("'", '"') else module


def _module_alias_rules(languages: Sequence[str]) -> str:
    documents = [
        _rule_document(_MODULE_ALIAS_RULE, language, rule)
        for language in languages
        for rule in MODULE_ALIAS_RULES[language]
    ]
    if "python" in languages:
        documents.append(_rule_document(_FROM_IMPORT_RULE, "python", PYTHON_FROM_IMPORT))
        documents += [_rule_document(_FROM_NAME_RULE, "python", rule) for rule in PYTHON_FROM_IMPORT_NAMES]
    documents += [
        _rule_document(_MODULE_BINDING_RULE, language, MODULE_BINDING_RULES[language])
        for language in languages
        if language in MODULE_BINDING_RULES
    ]
    return "\n---\n".join(documents)


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
        if language in LOCAL_MODULE_RULES:
            documents.append(_rule_document(_LOCAL_MODULE_RULE, language, LOCAL_MODULE_RULES[language]))
            documents.append(
                _rule_document(_LOCAL_MODULE_BLOCK_RULE, language, LOCAL_MODULE_BLOCK_RULES[language])
            )
            documents.append(_rule_document(_LOCAL_MEMBER_RULE, language, LOCAL_MEMBER_RULES[language]))
        if DECORATED_KINDS[language]:
            documents.append(_decorated_rule(language))
        if language in STUB_RULES:
            documents.append(_rule_document(_STUB_RULE, language, STUB_RULES[language]))
        documents.append(_rule_document(_ERROR_RULE, language, "  kind: ERROR"))
        if VALUE_KINDS[language]:
            documents.append(_held_rule(language))
            documents += _property_rules(language)
            documents += _object_member_rules(language)
            documents += _constant_rules(language)
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


def _object_member_rules(language: str) -> list[str]:
    """The members of the object literals a module-level variable holds (see ``ObjectMember``) and of
    those a module-level call or `new` is passed (see ``FileStructure.argument_members``), and the
    name of every module-level variable whose value is an object. Module level is outside every
    function, class and namespace."""
    scopes = _kinds((*FUNCTION_KINDS[language], *CLASS_KINDS[language], *NAMESPACE_KINDS[language]))
    passed = f"{{kind: arguments, not: {{inside: {{stopBy: end, any: {scopes}}}}}}}"
    objects = _past_wrappers_of(language, ("object",))
    variable = f"{{field: name, all: [{MODULE_VARIABLE}, {{has: {{field: value, any: {objects}}}}}]}}"
    owner = f"  kind: identifier\n  not: {{not: {{inside: {variable}}}}}"
    return [
        _rule_document(_OBJECT_MEMBER_RULE, language, _members_of(language, MODULE_VARIABLE)),
        _rule_document(_ARGUMENT_MEMBER_RULE, language, _members_of(language, passed)),
        _rule_document(_OBJECT_OWNER_RULE, language, owner),
    ]


def _constant_rules(language: str) -> list[str]:
    """The name of every module-level variable whose value is a call or `new`, also under wrappers
    such as `as`, and the key of every pair outside every function, class and namespace in such a
    variable's value that holds a function (see ``ConstantFunction``). A key that is computed names
    nothing. The pair rule prints the pair; every relation sits under a double negation."""
    calls = _past_wrappers_of(language, ("call_expression", "new_expression"))
    variable = f"{{field: name, all: [{MODULE_VARIABLE}, {{has: {{field: value, any: {calls}}}}}]}}"
    owner = f"  kind: identifier\n  not: {{not: {{inside: {variable}}}}}"
    symbols = _kinds((*FUNCTION_KINDS[language], *CLASS_KINDS[language]))
    scopes = _kinds((*FUNCTION_KINDS[language], *CLASS_KINDS[language], *NAMESPACE_KINDS[language]))
    key = (
        "  kind: pair\n"
        "  has: {field: key, any: [{kind: property_identifier}, {kind: string}], pattern: $KEY}\n"
        "  all:\n"
        f"    - not: {{not: {{has: {{stopBy: end, any: {symbols}}}}}}}\n"
        f"    - not: {{not: {{inside: {{stopBy: end, all: [{MODULE_VARIABLE}]}}}}}}\n"
        f"    - not: {{inside: {{stopBy: end, any: {scopes}}}}}"
    )
    return [
        _rule_document(_CONSTANT_OWNER_RULE, language, owner),
        _rule_document(_CONSTANT_KEY_RULE, language, key),
        _rule_document(_DATA_CALLBACK_RULE, language, _data_callback(language)),
    ]


# The steps a collection takes a callback for. Called on a PascalCase name the step belongs to a
# module, such as `Effect.map(effect, (value) => ...)`, which builds an Effect rather than data.
_COLLECTION_STEPS = (
    "^(map|flatMap|filter|reduce|reduceRight|find|findIndex|findLast|findLastIndex|some|every|forEach"
    "|sort|toSorted)$"
)
_COLLECTION_STEP_CALL = (
    "{kind: call_expression, has: {field: function, kind: member_expression, all: ["
    f"{{has: {{field: property, regex: '{_COLLECTION_STEPS}'}}}}, "
    "{not: {has: {field: object, kind: identifier, regex: '^[A-Z][a-z]'}}}]}}"
)
_COLLECTION_BUILDERS = (
    "[{kind: new_expression, has: {field: constructor, regex: '^(Map|Set)$'}}, "
    "{kind: call_expression, has: {field: function, regex: '^Array\\.from$'}}]"
)


def _data_callback(language: str) -> str:
    """Every function that builds data rather than being one: a callback a collection step takes,
    `items.map((item) => item.id)`, and one inside `new Map`, `new Set` or `Array.from`. A module-level
    constant built through one names no function (see ``ConstantFunction``)."""
    return (
        f"  any: {_kinds(FUNCTION_KINDS[language])}\n"
        "  not:\n    not:\n      any:\n"
        f"        - inside: {{kind: arguments, inside: {_COLLECTION_STEP_CALL}}}\n"
        f"        - inside: {{stopBy: end, any: {_COLLECTION_BUILDERS}}}"
    )


def _past_wrappers_of(language: str, kinds: Sequence[str]) -> str:
    """A node of ``kinds``, itself or as the first node no wrapper (see ``NAME_WRAPPERS``) holds: a
    call is `wrap(...)` and `wrap(...) as Runner`, never `[...] as const`."""
    wanted = _kinds(kinds)
    wrapped = f"{{stopBy: {_past_wrappers(language)}, any: {wanted}}}"
    return f"[{{any: {wanted}}}, {{any: {_kinds(NAME_WRAPPERS[grammar_of(language)])}, has: {wrapped}}}]"


def _members_of(language: str, holder: str) -> str:
    """Every function and class that is a method or a property value of an object literal whose first
    ancestor past wrappers such as `satisfies` is ``holder``, a condition printed by no match."""
    symbol_kinds = (*FUNCTION_KINDS[language], *CLASS_KINDS[language])
    expressions = _kinds([kind for kind in symbol_kinds if kind in EXPRESSION_KINDS])
    held_object = f"{{kind: object, inside: {{stopBy: {_past_wrappers(language)}, any: [{holder}]}}}}"
    held_pair = f"{{kind: pair, inside: {held_object}}}"
    return (
        "  any:\n"
        f"    - {{kind: method_definition, not: {{not: {{inside: {held_object}}}}}}}\n"
        f"    - {{any: {expressions}, {_held_past_wrappers(language, held_pair)}}}"
    )


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
