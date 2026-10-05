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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NotRequired, TypedDict

import msgspec

from . import tools
from .imports import _local
from .languages import (
    CLASS_KINDS,
    DECLARATION_RULES,
    DECORATED_KINDS,
    EXPRESSION_KINDS,
    FLOW_LANGUAGE,
    FUNCTION_KINDS,
    NAME_HOLDERS,
    NAME_WRAPPERS,
    STUB_RULES,
    declared_name,
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


@dataclass(frozen=True)
class FileStructure:
    functions: tuple[Span, ...]
    symbols: tuple[Span, ...]
    declarations: tuple[Span, ...]
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
    # Why the guard kept the file from the parser; such facts are empty and are never cached.
    refusal: str | None = None
    # The language the facts were read as: ``flow`` for JavaScript read with the tsx grammar. None
    # when no grammar read the file.
    language: str | None = None


class _Text(TypedDict):
    text: str


class _Start(TypedDict):
    line: int


class _NodeRange(TypedDict):
    start: _Start


class _Node(TypedDict):
    range: _NodeRange


class _Captured(TypedDict, total=False):
    CALLEE: _Text
    NAME: _Text
    DECORATOR: _Node


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
    found = {file: _FileFound() for file in contents}
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
    found = {file: _FileFound(FLOW_LANGUAGE if as_flow else language_of(file)) for file in files}
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
        )
        if part
    )


@dataclass
class _FileFound:
    """One file's facts, collected match by match, read as ``language``."""

    language: str | None = None
    functions: set[Span] = field(default_factory=set)
    classes: set[Span] = field(default_factory=set)
    declarations: set[Span] = field(default_factory=set)
    decorated: set[tuple[int, int, int]] = field(default_factory=set)
    stubs: set[tuple[int, int]] = field(default_factory=set)
    calls: list[tuple[tuple[str, int, int], CallMatch]] = field(default_factory=list)
    receivers: dict[tuple[str, int, str, str], set[str | None]] = field(default_factory=dict)
    export_names: set[str] = field(default_factory=set)
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
        else:
            self._add_reference(match)

    def unread_line_count(self) -> int:
        return sum(end - start + 1 for start, end in _merged_stretches(self.error_lines))

    def finished(self, incomplete: bool) -> FileFacts:
        functions = self.functions - _same_lines_as_a_named_symbol(self.functions | self.classes)
        structure = FileStructure(
            _ordered(functions),
            _ordered(functions | self.classes),
            tuple(sorted(self.declarations)),
            tuple(sorted(self.decorated)),
            tuple(sorted(self.stubs)),
        )
        calls = tuple(call for _, call in sorted(self.calls, key=lambda entry: entry[0]))
        return FileFacts(
            structure,
            calls,
            self._references(),
            incomplete,
            tuple(sorted(self.export_names)),
            _merged_stretches(self.error_lines),
            language=self.language,
        )

    def _references(self) -> tuple[ReferenceMatch, ...]:
        """When one line passes both ``x.name`` and ``name`` in the same role, the plain name stands
        for that line, as a plain call does for callers."""
        return tuple(
            ReferenceMatch(*key, None if None in found else min(found))
            for key, found in sorted(self.receivers.items())
        )

    def _add_structure(self, match: dict) -> None:
        file, (start, end) = match["file"], _lines_of(match)
        if match["ruleId"] == "declaration":
            self.declarations.add(Span(file, start, end, declared_name(_first_line(match))))
            return
        target = self.functions if match["ruleId"] == "function" else self.classes
        # The syntax tree names the symbol, never a physical line: a method on a one-line class
        # shares the line `class Box` opens, and naming it from that line would collapse it into
        # the class's span.
        target.add(Span(file, start, end, symbol_name(_captured_name(match))))

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
        """Exported declarations' name nodes, and the specifier nodes of ``{ ... }`` lists."""
        text = match["text"]
        self.export_names.add(_local(text) if match["ruleId"] == _EXPORT_SPECIFIER_RULE else text)


def _same_lines_as_a_named_symbol(symbols: set[Span]) -> set[Span]:
    """Anonymous functions spanning exactly a named symbol's lines: ``xs.map((x) => x.id)`` on the
    one line of ``ids``. A place is lines, so such a callback is that symbol; kept apart, it would
    contain the symbol's first line and stop it being top level."""
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


def _reference_name(role: str, text: str) -> str:
    return last_identifier(text) if role == "argument" else text


def _reference_receiver(role: str, text: str) -> str | None:
    return receiver_of(text) if role == "argument" else None


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


def symbol_name(captured: str) -> str:
    """The name a captured name node gives a symbol: ``save`` for ``save``, ``this.save``,
    ``exports.save``, ``#save`` and the key ``"save"``. A computed key, a string key that is no
    identifier (``"risk.triage"``) and a missing name leave it anonymous."""
    quoted = captured[:1] in ("'", '"')
    name = captured[1:-1] if quoted else last_identifier(captured.replace("#", ""))
    return name if name.isidentifier() else "<anonymous>"


def _captured_name(match: dict) -> str:
    return match.get("metaVariables", {}).get("single", {}).get("NAME", {}).get("text", "")


def _captured_line(match: dict, variable: str) -> int:
    return match["metaVariables"]["single"][variable]["range"]["start"]["line"] + 1


_ERROR_RULE = "parse_error"
_EXPORT_STATEMENT_RULE = "export_surface"
_EXPORT_SPECIFIER_RULE = "export_specifier"
_EXPORT_RULE_IDS = (_EXPORT_STATEMENT_RULE, _EXPORT_SPECIFIER_RULE)
_STRUCTURE_RULE_IDS = ("function", "class", "declaration")
_DECORATED_RULE = "decorated"
_STUB_RULE = "stub"
# A walk back from a function past its decorators stops at the first sibling that is neither.
_END_OF_DECORATORS = "{not: {any: [{kind: decorator}, {kind: comment}]}}"


def _structure_rules(languages: Sequence[str]) -> str:
    documents = []
    for language in languages:
        documents.append(_kind_rule("function", language, FUNCTION_KINDS[language]))
        documents.append(_kind_rule("class", language, CLASS_KINDS[language]))
        documents.append(
            f"id: declaration\nlanguage: {grammar_of(language)}\nrule:\n{DECLARATION_RULES[language]}"
        )
        if DECORATED_KINDS[language]:
            documents.append(_decorated_rule(language))
        if language in STUB_RULES:
            documents.append(
                f"id: {_STUB_RULE}\nlanguage: {grammar_of(language)}\nrule:\n{STUB_RULES[language]}"
            )
        documents.append(f"id: {_ERROR_RULE}\nlanguage: {grammar_of(language)}\nrule:\n  kind: ERROR")
    return "\n---\n".join(documents)


def _decorated_rule(language: str) -> str:
    """Each function whose decorators sit before it, its first decorator captured as ``$DECORATOR``:
    among the decorators and comments just before the function, the decorator no other one precedes."""
    first_decorator = (
        f"{{stopBy: {_END_OF_DECORATORS}, kind: decorator, pattern: $DECORATOR, "
        f"not: {{follows: {{stopBy: {_END_OF_DECORATORS}, kind: decorator}}}}}}"
    )
    return (
        f"id: {_DECORATED_RULE}\nlanguage: {grammar_of(language)}\nrule:\n"
        f"  any: {_kinds(DECORATED_KINDS[language])}\n  follows: {first_decorator}"
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
        past_wrappers = f"{{not: {{any: {_kinds(NAME_WRAPPERS[grammar])}}}}}"
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


def _ordered(spans: set[Span]) -> tuple[Span, ...]:
    """By position, outer first; symbols on the same lines by name, so the order never depends on
    the process's string hashing."""
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end, span.name)))


def _first_line(match: dict) -> str:
    """The first source line of the match as ast-grep read it, so names always agree with the text
    that was parsed, even when the file changes on disk meanwhile."""
    return match["lines"].split("\n", 1)[0].replace("\r", "").removeprefix("\ufeff")


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1


def _lines_of(match: dict) -> tuple[int, int]:
    return _line_of(match), match["range"]["end"]["line"] + 1
