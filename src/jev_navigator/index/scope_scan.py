"""Syntax facts extracted together in one ast-grep pass over each requested file set.

Each requested file set is handed to one ast-grep scan, which schedules parsing across its own worker
pool without reparsing arbitrary fixed-size batches. The structure rules also match the grammar's
ERROR nodes: a file the parser could only recover
partially (Flow types in a JavaScript file, say) is reported as unparsed too. Its matched symbols and
calls still count — recovery keeps what it could — but whatever the ERROR nodes swallowed is unknown,
not absent. ``FileFacts.unparsed_lines`` keeps the lines those nodes span, so a lookup can tell which
names they may hide.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Protocol, TypeVar

from . import tools
from .imports import _local
from .languages import (
    CLASS_KINDS,
    COMMONJS_EXPORT_PAIR,
    COMMONJS_EXPORT_TARGET,
    COMMONJS_EXPORTS_OBJECT,
    DECLARED_NAME_RULES,
    EXPRESSION_KINDS,
    FLOW_LANGUAGE,
    FLOW_SGCONFIG,
    FUNCTION_KINDS,
    LOCAL_NAME_RULES,
    MODULE_ALIAS_RULES,
    NAME_HOLDERS,
    NAME_WRAPPERS,
    NAMESPACE_KINDS,
    PROPERTY_TARGET,
    TYPE_AND_VALUE_DECLARATIONS,
    TYPE_DECLARATIONS,
    VALUE_DECLARATIONS,
    VALUE_KINDS,
    export_rules,
    grammar_of,
    language_for,
    language_of,
    reference_rules,
)
from .spans import Span

LinesOf = Callable[[str], Sequence[str]]


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


@dataclass(frozen=True)
class FileStructure:
    """``module_symbols`` are the symbols their module names: no function, class or object literal
    holds them in the syntax tree, and no property assignment names them. Symbols sharing a line
    each hold the other's first line, so lines alone cannot tell. ``commonjs_exports`` are the
    functions and classes assigned to CommonJS exports (``exports.run = function () {}``), which
    another module imports by name but their own module never names.
    ``type_declarations`` and ``value_declarations`` are the declarations a type use and a value
    use may name, decided by each declaration's own syntax node. ``local_names`` are the names each
    function binds for its own body (see ``LOCAL_NAME_RULES``); a name a module-level block binds
    is not among them."""

    functions: tuple[Span, ...]
    symbols: tuple[Span, ...]
    declarations: tuple[Span, ...]
    module_symbols: tuple[Span, ...]
    commonjs_exports: tuple[Span, ...]
    type_declarations: tuple[Span, ...]
    value_declarations: tuple[Span, ...]
    local_names: tuple[LocalName, ...] = ()


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
    # The names a script module exports as values: its default export and its CommonJS exports of a
    # definition under its own name (see ``EXPORTED_VALUES``).
    exported_values: tuple[str, ...] = ()


def scan_facts(
    files: Sequence[str], root: Path, lines_of: LinesOf, unparsed: Unparsed
) -> dict[str, FileFacts]:
    """Parse supported source files once; return empty facts for unsupported paths."""
    matches: list[dict] = []
    supported_files = tuple(file for file in files if language_of(file) is not None)
    for config, group, languages in _scan_groups(supported_files, lines_of):
        rules = "\n---\n".join(
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
        if config is None:
            matches.extend(tools.ast_grep_rules(rules, group, root))
        else:
            matches.extend(tools.ast_grep_rules(rules, group, root, config=config))
    structure = _structure_from_matches(
        files,
        unparsed,
        (match for match in matches if match["ruleId"] in _STRUCTURE_RULE_IDS),
    )
    calls = _calls_from_matches(match for match in matches if match["ruleId"] == "call")
    references = _references_from_matches(
        match
        for match in matches
        if match["ruleId"]
        not in {*_STRUCTURE_RULE_IDS, "call", *_EXPORT_RULE_IDS, _EXPORTED_VALUE_RULE, _MODULE_ALIAS_RULE}
    )
    aliases = _module_aliases_from_matches(
        match for match in matches if match["ruleId"] == _MODULE_ALIAS_RULE
    )
    surface = _export_names_from_matches(match for match in matches if match["ruleId"] in _EXPORT_RULE_IDS)
    values = _captured_names_by_file(match for match in matches if match["ruleId"] == _EXPORTED_VALUE_RULE)
    unread = _unparsed_lines_from_matches(match for match in matches if match["ruleId"] == _ERROR_RULE)
    return {
        file: FileFacts(
            structure[file],
            tuple(call for call in calls if call.file == file),
            tuple(reference for reference in references if reference.file == file),
            file in unparsed.files,
            surface.get(file, ()),
            unread.get(file, ()),
            aliases.get(file, ()),
            values.get(file, ()),
        )
        for file in files
    }


def _unparsed_lines_from_matches(matches) -> dict[str, tuple[tuple[int, int], ...]]:
    """Each file's ERROR node lines, with nested and overlapping nodes merged into one stretch."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    for match in matches:
        ranges.setdefault(match["file"], []).append((_line_of(match), match["range"]["end"]["line"] + 1))
    merged: dict[str, tuple[tuple[int, int], ...]] = {}
    for file, found in ranges.items():
        stretches: list[tuple[int, int]] = []
        for start, end in sorted(found):
            if stretches and start <= stretches[-1][1]:
                stretches[-1] = (stretches[-1][0], max(end, stretches[-1][1]))
            else:
                stretches.append((start, end))
        merged[file] = tuple(stretches)
    return merged


def _structure_from_matches(files, unparsed, matches):
    functions: dict[str, set[Span]] = {file: set() for file in files}
    classes: dict[str, set[Span]] = {file: set() for file in files}
    declaration_nodes: dict[str, list[_Declaration]] = {file: [] for file in files}
    declared_names: dict[str, list[tuple[int, str]]] = {file: [] for file in files}
    bound_names: dict[str, list[tuple[int, str]]] = {file: [] for file in files}
    ranges: dict[str, list[tuple[int, int, Span]]] = {file: [] for file in files}
    marks: dict[str, dict[str, set[tuple[int, int]]]] = {
        file: {rule: set() for rule in _MARK_RULES} for file in files
    }
    for match in matches:
        file, start, end = match["file"], _line_of(match), match["range"]["end"]["line"] + 1
        if match["ruleId"] == _ERROR_RULE:
            # The grammar reports ERROR nodes here: whatever recovery swallowed is unknown, while the
            # symbols it did keep are still matched below.
            unparsed.add("facts", [file])
            continue
        offsets = match["range"]["byteOffset"]
        if match["ruleId"] in _MARK_RULES:
            marks[file][match["ruleId"]].add((offsets["start"], offsets["end"]))
        elif match["ruleId"] == _DECLARED_NAME_RULE:
            declared_names[file].append((offsets["start"], match["text"]))
        elif match["ruleId"] == _LOCAL_NAME_RULE:
            bound_names[file].append((offsets["start"], match["text"]))
        elif match["ruleId"] in _DECLARATION_BY_RULE:
            kind = _DECLARATION_BY_RULE[match["ruleId"]]
            declaration_nodes[file].append(_Declaration(offsets["start"], offsets["end"], start, end, kind))
        else:
            target = functions if match["ruleId"] == "function" else classes
            # The syntax tree names the symbol, never a physical line: a method on a one-line class
            # shares the line `class Box` opens, and naming it from that line would collapse it into
            # the class's span.
            span = Span(file, start, end, symbol_name(_captured_name(match)))
            target[file].add(span)
            ranges[file].append((offsets["start"], offsets["end"], span))
    positions = _source_positions(range_ for found in ranges.values() for range_ in found)
    for file in files:
        functions[file] -= _same_lines_as_a_named_symbol(functions[file] | classes[file])
    structures = {}
    for file in files:
        symbols = functions[file] | classes[file]
        declared = _declarations(file, declaration_nodes[file], declared_names[file])
        held = _held(ranges[file], marks[file][_HELD_RULE])
        module_symbols = symbols - held - _marked(ranges[file], marks[file][_PROPERTY_VALUE_RULE])
        structures[file] = FileStructure(
            _ordered(functions[file], positions),
            _ordered(symbols, positions),
            _sorted(span for span, _ in declared),
            _ordered(module_symbols, positions),
            _ordered(_marked(ranges[file], marks[file][_MODULE_EXPORT_RULE]), positions),
            _sorted(span for span, kind in declared if kind.named_by_types),
            _sorted(span for span, kind in declared if kind.named_by_values),
            _local_names(ranges[file], classes[file], bound_names[file]),
        )
    return structures


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
) -> set[tuple[Span, _DeclarationKind]]:
    """One span per name a declaration binds, over the lines of the innermost declaration holding
    the name, with that declaration's kind."""
    ordered = sorted(nodes, key=lambda node: node.start)
    starts = [node.start for node in ordered]
    return {
        (Span(file, holder.first_line, holder.last_line, name), holder.kind)
        for offset, name in names
        if (holder := _innermost(ordered, starts, offset)) is not None
    }


def _sorted(spans: Iterable[Span]) -> tuple[Span, ...]:
    return tuple(sorted(set(spans)))


class _ByteRange(Protocol):
    @property
    def end(self) -> int: ...


_Range = TypeVar("_Range", bound=_ByteRange)


def _innermost(ordered: Sequence[_Range], starts: list[int], offset: int) -> _Range | None:
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


def _held(ranges: list[tuple[int, int, Span]], held_nodes: set[tuple[int, int]]) -> set[Span]:
    """The spans whose syntax node is a value or a namespace member (see ``_held_rule``) or lies inside
    another function or class node, counting the callbacks ``_same_lines_as_a_named_symbol`` drops: a
    function inside a one-line callback is the callback's. Nodes nest or are disjoint, so a node is
    inside another exactly when one starting no later reaches at least as far."""
    held = _marked(ranges, held_nodes)
    furthest = -1
    for _start, end, span in sorted(ranges, key=lambda range_: (range_[0], -range_[1])):
        if end <= furthest:
            held.add(span)
        furthest = max(furthest, end)
    return held


def _same_lines_as_a_named_symbol(symbols: set[Span]) -> set[Span]:
    """Anonymous functions spanning exactly a named symbol's lines: ``xs.map((x) => x.id)`` on the
    one line of ``ids``. A place is lines, so such a callback is that symbol; kept apart, it would
    be a second place on the same lines."""
    named = {(span.start, span.end) for span in symbols if span.name != "<anonymous>"}
    return {span for span in symbols if span.name == "<anonymous>" and (span.start, span.end) in named}


def _calls_from_matches(matches) -> tuple[CallMatch, ...]:
    found = []
    for match in matches:
        expression = match["metaVariables"]["single"]["CALLEE"]["text"]
        name = last_identifier(expression)
        if name:
            found.append(CallMatch(match["file"], _line_of(match), name, receiver_of(expression)))
    return tuple(sorted(found, key=lambda call: (call.file, call.line)))


def _references_from_matches(matches) -> tuple[ReferenceMatch, ...]:
    """When one line passes both ``x.name`` and ``name`` in the same role, the plain name stands for
    that line, as a plain call does for callers."""
    receivers: dict[tuple[str, int, str, str], set[str | None]] = {}
    for match in matches:
        role, text = match["ruleId"], match["text"]
        key = (match["file"], _line_of(match), role, _reference_name(role, text))
        receivers.setdefault(key, set()).add(_reference_receiver(role, text))
    return tuple(
        ReferenceMatch(*key, None if None in found else min(found))
        for key, found in sorted(receivers.items())
    )


# Roles whose node may be a qualified name, `x.name` or `pkg.mod.Name`: named by its last part, with
# the rest as its receiver.
_QUALIFIED_ROLES = frozenset({"argument", "base"})


def _reference_name(role: str, text: str) -> str:
    return last_identifier(text) if role in _QUALIFIED_ROLES else text


def _reference_receiver(role: str, text: str) -> str | None:
    return receiver_of(text) if role in _QUALIFIED_ROLES else None


def receiver_of(expression: str) -> str | None:
    head, dot, _ = expression.replace("?.", ".").rpartition(".")
    return head if dot else None


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


_ERROR_RULE = "parse_error"
_HELD_RULE = "held"
_DECLARED_NAME_RULE = "declared_name"
_LOCAL_NAME_RULE = "local_name"
_MODULE_ALIAS_RULE = "module_alias"
_PROPERTY_VALUE_RULE = "property_value"
_MODULE_EXPORT_RULE = "module_export"
_MARK_RULES = (_HELD_RULE, _PROPERTY_VALUE_RULE, _MODULE_EXPORT_RULE)
_STRUCTURE_RULE_IDS = frozenset(
    {
        "function",
        "class",
        *_DECLARATION_BY_RULE,
        _DECLARED_NAME_RULE,
        _LOCAL_NAME_RULE,
        *_MARK_RULES,
        _ERROR_RULE,
    }
)
_EXPORT_STATEMENT_RULE = "export_surface"
_EXPORT_SPECIFIER_RULE = "export_specifier"
_EXPORT_RULE_IDS = (_EXPORT_STATEMENT_RULE, _EXPORT_SPECIFIER_RULE)
_EXPORTED_VALUE_RULE = "exported_value"


def _export_names_from_matches(matches) -> dict[str, tuple[str, ...]]:
    """The names each file's parser says it exports: exported declarations' name nodes, and the
    specifier nodes of ``{ ... }`` lists."""
    names: dict[str, set[str]] = {}
    for match in matches:
        found = names.setdefault(match["file"], set())
        found.add(_local(match["text"]) if match["ruleId"] == _EXPORT_SPECIFIER_RULE else match["text"])
    return {file: tuple(sorted(found)) for file, found in names.items()}


def _captured_names_by_file(matches) -> dict[str, tuple[str, ...]]:
    names: dict[str, set[str]] = {}
    for match in matches:
        names.setdefault(match["file"], set()).add(_captured_name(match))
    return {file: tuple(sorted(found)) for file, found in names.items()}


def _module_aliases_from_matches(matches) -> dict[str, tuple[ModuleAlias, ...]]:
    """Each file's module aliases in source order (see ``MODULE_ALIAS_RULES``)."""
    aliases: dict[str, list[tuple[int, ModuleAlias]]] = {}
    for match in matches:
        found = aliases.setdefault(match["file"], [])
        found += [(match["range"]["byteOffset"]["start"], alias) for alias in _module_aliases_of(match)]
    return {file: tuple(alias for _, alias in sorted(found)) for file, found in aliases.items()}


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
        documents.append(_rule_document(_ERROR_RULE, language, "  kind: ERROR"))
        if VALUE_KINDS[language] or NAMESPACE_KINDS[language]:
            documents.append(_held_rule(language))
        if VALUE_KINDS[language]:
            documents += _property_rules(language)
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
    """Every function and class anywhere inside a value or a namespace node (see ``VALUE_KINDS`` and
    ``NAMESPACE_KINDS``)."""
    symbols = _kinds((*FUNCTION_KINDS[language], *CLASS_KINDS[language]))
    holders = _kinds((*VALUE_KINDS[language], *NAMESPACE_KINDS[language]))
    return _rule_document(
        _HELD_RULE, language, f"  any: {symbols}\n  not: {{not: {{inside: {{stopBy: end, any: {holders}}}}}}}"
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
        # The search stops at the first ancestor that is no wrapper, and that one must be the holder.
        past_wrappers = _past_wrappers(language)
        holders = ", ".join(
            f"{{kind: {holder}, has: {_named_by(field)}}}" for holder, field in NAME_HOLDERS[grammar]
        )
        alternatives += [
            f"{{any: {_kinds(expressions)}, inside: {{stopBy: {past_wrappers}, any: [{holders}]}}}}",
            f"{{any: {_kinds(expressions)}, has: {_named_by('name')}}}",
        ]
    alternatives.append(f"{{any: {_kinds(kinds)}}}")
    listed = "".join(f"\n    - {alternative}" for alternative in alternatives)
    return f"id: {rule_id}\nlanguage: {grammar}\nrule:\n  any:{listed}"


def _kinds(kinds: Sequence[str]) -> str:
    return "[" + ", ".join(f"{{kind: {kind}}}" for kind in kinds) + "]"


def _named_by(field: str) -> str:
    return f"{{field: {field}, pattern: $NAME}}"


def _scan_groups(files: Sequence[str], lines_of: LinesOf) -> list[tuple[str | None, list[str], list[str]]]:
    """(sgconfig, files, languages) per invocation: every non-flow file scanned together exactly as
    before, and the ``@flow`` files in their own invocation, where the config's ``languageGlobs``
    parses the JavaScript suffixes with the tsx grammar. The globs are global per invocation, so
    mixing the two would re-parse plain JavaScript files too."""
    plain: list[str] = []
    flow: list[str] = []
    for file in files:
        if language_for(file, lines_of(file)) == FLOW_LANGUAGE:
            flow.append(file)
        else:
            plain.append(file)
    groups: list[tuple[str | None, list[str], list[str]]] = []
    if plain:
        groups.append((None, plain, _languages(plain)))
    if flow:
        groups.append((FLOW_SGCONFIG, flow, [FLOW_LANGUAGE]))
    return groups


def _languages(files: Sequence[str]) -> list[str]:
    return sorted({language for file in files if (language := language_of(file))})


def _ordered(spans: set[Span], positions: dict[Span, int]) -> tuple[Span, ...]:
    """By position, outer first; symbols on the same lines in source order. A line's calls belong to
    the smallest function holding it, and among functions on the same lines the first one wins: in
    `function retry(again = () => 1) { return attempt(); }` that is `retry`, not its default."""
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end, positions[span])))


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1
