"""Syntax facts extracted together in one ast-grep pass over each requested file set.

Each requested file set is handed to one ast-grep scan, which schedules parsing across its own worker
pool without reparsing arbitrary fixed-size batches. The structure rules also match the grammar's
ERROR nodes: a file the parser could only recover
partially (Flow types in a JavaScript file, say) is reported as unparsed too. Its matched symbols and
calls still count — recovery keeps what it could — but whatever the ERROR nodes swallowed is unknown,
not absent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import tools
from .imports import _local
from .languages import (
    CLASS_KINDS,
    DECLARATION_RULES,
    FLOW_LANGUAGE,
    FLOW_SGCONFIG,
    FUNCTION_KINDS,
    declared_name,
    export_rules,
    function_name,
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


@dataclass(frozen=True)
class FileStructure:
    functions: tuple[Span, ...]
    symbols: tuple[Span, ...]
    declarations: tuple[Span, ...]


@dataclass(frozen=True)
class CallMatch:
    file: str
    line: int
    name: str
    receiver: str | None


@dataclass(frozen=True, order=True)
class ReferenceMatch:
    file: str
    line: int
    role: str
    name: str


@dataclass(frozen=True)
class FileFacts:
    structure: FileStructure
    calls: tuple[CallMatch, ...]
    references: tuple[ReferenceMatch, ...]
    incomplete: bool = False
    export_names: tuple[str, ...] = ()


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
            )
            if part
        )
        if config is None:
            matches.extend(tools.ast_grep_rules(rules, group, root))
        else:
            matches.extend(tools.ast_grep_rules(rules, group, root, config=config))
    structure = _structure_from_matches(
        files,
        lines_of,
        unparsed,
        (match for match in matches if match["ruleId"] in {"function", "class", "declaration", _ERROR_RULE}),
    )
    calls = _calls_from_matches(match for match in matches if match["ruleId"] == "call")
    references = _references_from_matches(
        match
        for match in matches
        if match["ruleId"] not in {"function", "class", "declaration", _ERROR_RULE, "call", *_EXPORT_RULE_IDS}
    )
    surface = _export_names_from_matches(match for match in matches if match["ruleId"] in _EXPORT_RULE_IDS)
    return {
        file: FileFacts(
            structure[file],
            tuple(call for call in calls if call.file == file),
            tuple(reference for reference in references if reference.file == file),
            file in unparsed.files,
            surface.get(file, ()),
        )
        for file in files
    }


def _structure_from_matches(files, lines_of, unparsed, matches):
    functions: dict[str, set[Span]] = {file: set() for file in files}
    classes: dict[str, set[Span]] = {file: set() for file in files}
    declarations: dict[str, set[Span]] = {file: set() for file in files}
    for match in matches:
        file, start, end = match["file"], _line_of(match), match["range"]["end"]["line"] + 1
        if match["ruleId"] == _ERROR_RULE:
            # The grammar reports ERROR nodes here: whatever recovery swallowed is unknown, while the
            # symbols it did keep are still matched below.
            unparsed.add("facts", [file])
            continue
        lines = lines_of(file)
        if match["ruleId"] == "declaration":
            declarations[file].add(Span(file, start, end, declared_name(lines[start - 1])))
        else:
            target = functions if match["ruleId"] == "function" else classes
            target[file].add(Span(file, start, end, _symbol_name(match, lines, start)))
    return {
        file: FileStructure(
            _ordered(functions[file]),
            _ordered(functions[file] | classes[file]),
            tuple(sorted(declarations[file])),
        )
        for file in files
    }


def _symbol_name(match: dict, lines: Sequence[str], start: int) -> str:
    """Only code before the node may name it: on `class Box { v() {} }` the method keeps its own name
    instead of collapsing into the class's span."""
    before_node = lines[start - 1][: match["range"]["start"]["column"]]
    line_before = lines[start - 2] if start > 1 else ""
    return function_name(match["text"], before_node, line_before)


def _calls_from_matches(matches) -> tuple[CallMatch, ...]:
    found = []
    for match in matches:
        expression = match["metaVariables"]["single"]["CALLEE"]["text"]
        name = last_identifier(expression)
        if name:
            found.append(CallMatch(match["file"], _line_of(match), name, receiver_of(expression)))
    return tuple(sorted(found, key=lambda call: (call.file, call.line)))


def _references_from_matches(matches) -> tuple[ReferenceMatch, ...]:
    return tuple(
        sorted(
            {
                ReferenceMatch(
                    match["file"],
                    _line_of(match),
                    match["ruleId"],
                    _reference_name(match["ruleId"], match["text"]),
                )
                for match in matches
            }
        )
    )


def _reference_name(role: str, text: str) -> str:
    return last_identifier(text) if role == "argument" else text


def receiver_of(expression: str) -> str | None:
    head, dot, _ = expression.replace("?.", ".").rpartition(".")
    return head if dot else None


def last_identifier(expression: str) -> str:
    tail = expression.replace("?.", ".").split(".")[-1]
    return tail if tail.isidentifier() else ""


_ERROR_RULE = "parse_error"
_EXPORT_STATEMENT_RULE = "export_surface"
_EXPORT_SPECIFIER_RULE = "export_specifier"
_EXPORT_RULE_IDS = (_EXPORT_STATEMENT_RULE, _EXPORT_SPECIFIER_RULE)


def _export_names_from_matches(matches) -> dict[str, tuple[str, ...]]:
    """The names each file's parser says it exports, from real statement and specifier nodes."""
    names: dict[str, set[str]] = {}
    for match in matches:
        found = names.setdefault(match["file"], set())
        if match["ruleId"] == _EXPORT_SPECIFIER_RULE:
            found.add(_local(match["text"]))
        elif name := _export_statement_name(match["text"]):
            found.add(name)
    return {file: tuple(sorted(found)) for file, found in names.items()}


def _export_statement_name(text: str) -> str:
    """The name an ``export`` statement declares. Default, wildcard, namespace and ``{ ... }``
    list statements contribute nothing here (lists name themselves through specifier nodes);
    declarations are named by the same helpers the spans use."""
    statement = " ".join(text.split())
    body = statement.removeprefix("export").lstrip()
    if body.startswith(("default", "*", "as ", "{", "=")):
        return ""
    name = function_name(statement)
    if name == "<anonymous>":
        name = declared_name(statement)
    return name if name.isidentifier() else ""


def _structure_rules(languages: Sequence[str]) -> str:
    documents = []
    for language in languages:
        documents.append(_kind_rule("function", language, FUNCTION_KINDS[language]))
        documents.append(_kind_rule("class", language, CLASS_KINDS[language]))
        documents.append(
            f"id: declaration\nlanguage: {grammar_of(language)}\nrule:\n{DECLARATION_RULES[language]}"
        )
        documents.append(f"id: {_ERROR_RULE}\nlanguage: {grammar_of(language)}\nrule:\n  kind: ERROR")
    return "\n---\n".join(documents)


def _call_rules(languages: Sequence[str]) -> str:
    documents = []
    for language in languages:
        grammar = grammar_of(language)
        documents.append(f"id: call\nlanguage: {grammar}\nrule:\n  pattern: $CALLEE($$$)")
        if grammar != "python":
            documents.append(f"id: call\nlanguage: {grammar}\nrule:\n  pattern: new $CALLEE($$$)")
    return "\n---\n".join(documents)


def _kind_rule(rule_id: str, language: str, kinds: Sequence[str]) -> str:
    listed = "".join(f"\n    - kind: {kind}" for kind in kinds)
    return f"id: {rule_id}\nlanguage: {grammar_of(language)}\nrule:\n  any:{listed}"


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


def _ordered(spans: set[Span]) -> tuple[Span, ...]:
    """Outer spans before the spans they hold; the name orders spans on the same lines."""
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end, span.name)))


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1
