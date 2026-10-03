"""Syntax facts extracted together in one ast-grep pass over each requested file set.

Each requested file set is handed to ast-grep scans of a few hundred files each, which schedule
parsing across ast-grep's own worker pool; every match becomes its fact as it is printed. The
structure rules also match the grammar's ERROR nodes: a file the parser could only recover
partially (Flow types in a JavaScript file, say) is reported as unparsed too. Its matched symbols and
calls still count — recovery keeps what it could — but whatever the ERROR nodes swallowed is unknown,
not absent. ``FileFacts.unparsed_lines`` keeps the lines those nodes span, so a lookup can tell which
names they may hide.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    language_of,
    parse_language,
    reference_rules,
)
from .spans import Span


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


def scan_facts(files: Sequence[str], root: Path, unparsed: Unparsed) -> dict[str, FileFacts]:
    """Parse supported source files once; return empty facts for unsupported paths. Each match is
    turned into its fact as the parser prints it, so memory holds facts, never the parser's output."""
    found = {file: _FileFound() for file in files}
    supported_files = tuple(file for file in files if language_of(file) is not None)
    for config, group, languages in _scan_groups(supported_files, root):
        rules = fact_rules(languages)
        if config is None:
            matches = tools.ast_grep_rules(rules, group, root)
        else:
            matches = tools.ast_grep_rules(rules, group, root, config=config)
        for match in matches:
            found[match["file"]].add(match)
    unparsed.add("facts", [file for file, facts in found.items() if facts.error_lines])
    incomplete = unparsed.files
    return {file: facts.finished(file in incomplete) for file, facts in found.items()}


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
    """One file's facts, collected match by match."""

    functions: set[Span] = field(default_factory=set)
    classes: set[Span] = field(default_factory=set)
    declarations: set[Span] = field(default_factory=set)
    calls: list[tuple[int, int, CallMatch]] = field(default_factory=list)
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
        elif rule == "call":
            self._add_call(match)
        elif rule in _EXPORT_RULE_IDS:
            self._add_export(match)
        else:
            self._add_reference(match)

    def finished(self, incomplete: bool) -> FileFacts:
        functions = self.functions - _same_lines_as_a_named_symbol(self.functions | self.classes)
        structure = FileStructure(
            _ordered(functions),
            _ordered(functions | self.classes),
            tuple(sorted(self.declarations)),
        )
        calls = tuple(call for *_, call in sorted(self.calls, key=_source_order))
        return FileFacts(
            structure,
            calls,
            self._references(),
            incomplete,
            tuple(sorted(self.export_names)),
            _merged_stretches(self.error_lines),
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
            self.calls.append((call.line, match["range"]["start"]["column"], call))

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


def _source_order(positioned: tuple[int, int, CallMatch]) -> tuple[int, int, str, str]:
    """Calls in the order they start in the source; two starting together (``a().b()``) by name."""
    line, column, call = positioned
    return line, column, call.name, call.receiver or ""


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
_STRUCTURE_RULE_IDS = ("function", "class", "declaration")


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


def _scan_groups(files: Sequence[str], root: Path) -> list[tuple[str | None, list[str], list[str]]]:
    """(sgconfig, files, languages) per invocation: every non-flow file scanned together exactly as
    before, and the ``@flow`` files in their own invocation, where the config's ``languageGlobs``
    parses the JavaScript suffixes with the tsx grammar. The globs are global per invocation, so
    mixing the two would re-parse plain JavaScript files too."""
    plain: list[str] = []
    flow: list[str] = []
    for file in files:
        (flow if _is_flow(root, file) else plain).append(file)
    groups: list[tuple[str | None, list[str], list[str]]] = []
    if plain:
        groups.append((None, plain, _languages(plain)))
    if flow:
        groups.append((FLOW_SGCONFIG, flow, [FLOW_LANGUAGE]))
    return groups


def _is_flow(root: Path, file: str) -> bool:
    if language_of(file) != "javascript":
        return False
    try:
        content = (root / file).read_bytes()
    except FileNotFoundError:
        return False
    return parse_language(file, content) == FLOW_LANGUAGE


def _languages(files: Sequence[str]) -> list[str]:
    return sorted({language for file in files if (language := language_of(file))})


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
