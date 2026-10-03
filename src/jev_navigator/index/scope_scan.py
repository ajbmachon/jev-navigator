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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import tools
from .imports import _local
from .languages import (
    CLASS_KINDS,
    DECLARATION_RULES,
    EXPRESSION_KINDS,
    FLOW_LANGUAGE,
    FLOW_SGCONFIG,
    FUNCTION_KINDS,
    NAME_HOLDERS,
    NAME_WRAPPERS,
    declared_name,
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
    unread = _unparsed_lines_from_matches(match for match in matches if match["ruleId"] == _ERROR_RULE)
    return {
        file: FileFacts(
            structure[file],
            tuple(call for call in calls if call.file == file),
            tuple(reference for reference in references if reference.file == file),
            file in unparsed.files,
            surface.get(file, ()),
            unread.get(file, ()),
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
    declarations: dict[str, set[Span]] = {file: set() for file in files}
    for match in matches:
        file, start, end = match["file"], _line_of(match), match["range"]["end"]["line"] + 1
        if match["ruleId"] == _ERROR_RULE:
            # The grammar reports ERROR nodes here: whatever recovery swallowed is unknown, while the
            # symbols it did keep are still matched below.
            unparsed.add("facts", [file])
            continue
        if match["ruleId"] == "declaration":
            # Named from its own text: a declaration may start after other code on its line.
            declarations[file].add(Span(file, start, end, declared_name(match["text"])))
        else:
            target = functions if match["ruleId"] == "function" else classes
            # The syntax tree names the symbol, never a physical line: a method on a one-line class
            # shares the line `class Box` opens, and naming it from that line would collapse it into
            # the class's span.
            target[file].add(Span(file, start, end, symbol_name(_captured_name(match))))
    for file in files:
        functions[file] -= _same_lines_as_a_named_symbol(functions[file] | classes[file])
    return {
        file: FileStructure(
            _ordered(functions[file]),
            _ordered(functions[file] | classes[file]),
            tuple(sorted(declarations[file])),
        )
        for file in files
    }


def _same_lines_as_a_named_symbol(symbols: set[Span]) -> set[Span]:
    """Anonymous functions spanning exactly a named symbol's lines: ``xs.map((x) => x.id)`` on the
    one line of ``ids``. A place is lines, so such a callback is that symbol; kept apart, it would
    contain the symbol's first line and stop it being top level."""
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


def _reference_name(role: str, text: str) -> str:
    return last_identifier(text) if role == "argument" else text


def _reference_receiver(role: str, text: str) -> str | None:
    return receiver_of(text) if role == "argument" else None


def receiver_of(expression: str) -> str | None:
    head, dot, _ = expression.replace("?.", ".").rpartition(".")
    return head if dot else None


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


_ERROR_RULE = "parse_error"
_EXPORT_STATEMENT_RULE = "export_surface"
_EXPORT_SPECIFIER_RULE = "export_specifier"
_EXPORT_RULE_IDS = (_EXPORT_STATEMENT_RULE, _EXPORT_SPECIFIER_RULE)


def _export_names_from_matches(matches) -> dict[str, tuple[str, ...]]:
    """The names each file's parser says it exports: exported declarations' name nodes, and the
    specifier nodes of ``{ ... }`` lists."""
    names: dict[str, set[str]] = {}
    for match in matches:
        found = names.setdefault(match["file"], set())
        found.add(_local(match["text"]) if match["ruleId"] == _EXPORT_SPECIFIER_RULE else match["text"])
    return {file: tuple(sorted(found)) for file, found in names.items()}


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
    """By position, outer first; symbols on the same lines by name, so the order never depends on
    the process's string hashing."""
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end, span.name)))


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1
