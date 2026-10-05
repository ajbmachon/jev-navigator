"""Comments found mechanically: every comment block in a scope, its kind, the code it is attached to,
cheap facts about it, and the comments a diff touched. No model is involved.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

from . import operations
from .facts import DEFAULT_COMMENT_RULES, Fact, FactRule, find_facts
from .index import tools
from .index.code_index import CodeIndex
from .index.languages import grammar_of, language_of, sgconfig_of
from .index.spans import CodeSlice, Span


class CommentKind(StrEnum):
    DOCSTRING = "docstring"
    JSDOC = "jsdoc"
    HEADER = "header"
    TOOL_DIRECTIVE = "tool_directive"
    DECLARATION = "declaration"
    INLINE = "inline"
    BLOCK = "block"


@dataclass(frozen=True)
class CommentBlock:
    """``attached`` is the code the comment sits on: the declaration or block below it, or the
    statement a trailing comment ends. ``facts`` are mechanical observations such as
    ``todo_without_owner``."""

    span: Span
    text: str
    kind: CommentKind
    attached: Span | None
    facts: tuple[Fact, ...]

    def fact_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(fact.name for fact in self.facts))


_TOOL_DIRECTIVE = re.compile(
    r"^\W*(noqa|type:\s*ignore|pylint:|mypy:|eslint-disable|eslint-enable|@ts-ignore|@ts-expect-error|@ts-nocheck"
    r"|prettier-ignore|istanbul ignore|c8 ignore|nosec|fmt:\s*(on|off)|pragma|isort:|ruff:)",
    re.I,
)
_DIVIDER = re.compile(r"^[\s#/*=\-_~+.]*$")
_LICENSE = re.compile(r"\b(license|copyright|spdx-license-identifier)\b", re.I)
_LINE_COMMENT_MARKERS = ("//", "#")
_COMMENT_MARKERS = re.compile(r"^\s*(?:#+|//+|/\*+|\*+/?|\*/)\s?")
_DOCSTRING_QUOTES = ('"""', "'''")


class CodeAboveReason(StrEnum):
    FOUND = "found"
    BLANK_LINE = "blank_line"
    START_OF_FILE = "start_of_file"


@dataclass(frozen=True)
class CodeAbove:
    code: CodeSlice | None
    reason: CodeAboveReason


DropRule = Callable[[CommentBlock], str | None]


@dataclass(frozen=True)
class DroppedComment:
    block: CommentBlock
    reason: str


@dataclass(frozen=True)
class FoundComments:
    """``kept`` and ``dropped`` together are every comment found, so callers keep the full count."""

    kept: tuple[CommentBlock, ...]
    dropped: tuple[DroppedComment, ...] = ()
    refused_files: Mapping[str, str] = field(default_factory=dict)


def find_comments(
    index: CodeIndex,
    files: Sequence[str] | None = None,
    *,
    drop: DropRule | None = None,
    fact_rules: Sequence[FactRule] = DEFAULT_COMMENT_RULES,
) -> FoundComments:
    """Every comment block in ``files`` (default: every code file in the index), adjacent line
    comments of the same syntax merged. ``drop`` returns a reason to set a block aside, or None to
    keep it; ``noise_reason`` (dividers, licence headers, bare tool directives) is one ready-made
    choice. Facts in each block come from ``fact_rules``."""
    chosen = files if files is not None else [file for file in index.files if language_of(file)]
    blocks: list[CommentBlock] = []
    refused: dict[str, str] = {}
    for file in chosen:
        found, reason = _comments_in_file(index, file, fact_rules)
        blocks.extend(found)
        if reason is not None:
            refused[file] = reason
    return replace(_split(blocks, drop), refused_files=refused)


def comments_in_diff(
    index: CodeIndex,
    base: str,
    head: str = "HEAD",
    *,
    drop: DropRule | None = None,
    fact_rules: Sequence[FactRule] = DEFAULT_COMMENT_RULES,
) -> FoundComments:
    """Comments added or changed between two commits, plus comments attached to code that changed:
    a comment left alone above changed code is the classic stale comment."""
    changed = _changed_lines(index, base, head)
    found = find_comments(index, sorted(changed), fact_rules=fact_rules)
    touched = [block for block in found.kept if _touches(block, changed[block.span.file])]
    return replace(_split(touched, drop), refused_files=found.refused_files)


def _split(blocks: Sequence[CommentBlock], drop: DropRule | None) -> FoundComments:
    kept: list[CommentBlock] = []
    dropped: list[DroppedComment] = []
    for block in blocks:
        reason = drop(block) if drop else None
        if reason is None:
            kept.append(block)
        else:
            dropped.append(DroppedComment(block, reason))
    return FoundComments(tuple(kept), tuple(dropped))


def code_above_comment(index: CodeIndex, file: str, line: int) -> CodeAbove:
    """The code just above a comment, for example the ``except`` of an empty handler: it stops at a
    blank line, or at the line that opens the enclosing block, which it includes. When no code sits
    directly above, ``code`` is None and ``reason`` says why."""
    if line == 1:
        return CodeAbove(None, CodeAboveReason.START_OF_FILE)
    lines = index.lines(file)
    if not lines[line - 2].strip():
        return CodeAbove(None, CodeAboveReason.BLANK_LINE)
    start = _start_of_code_above(lines, line)
    code = index.read_slice(Span(file, start, line - 1), origin="code_above_comment")
    return CodeAbove(code, CodeAboveReason.FOUND)


def _start_of_code_above(lines: Sequence[str], line: int) -> int:
    comment_indent = _indent(lines[line - 1])
    start = line - 1
    for number in range(line - 1, 0, -1):
        text = lines[number - 1]
        if not text.strip():
            break
        start = number
        if _indent(text) < comment_indent:
            break
    return start


def comment_facts(text: str, rules: Sequence[FactRule] = DEFAULT_COMMENT_RULES) -> tuple[Fact, ...]:
    return find_facts(text, rules)


def _comments_in_file(
    index: CodeIndex, file: str, rules: Sequence[FactRule]
) -> tuple[list[CommentBlock], str | None]:
    """The file's comment blocks, read with the grammar its facts were read with, or no blocks and the
    reason when the index could not read the file."""
    language = index.read_language(file)
    if language is None:
        return [], index.unavailable_files.get(file)
    refused: dict[str, str] = {}
    raw = _comment_ranges(_comment_nodes(index, file, language, refused))
    if file in refused:
        return [], refused[file]
    lines = index.lines(file)
    blocks = [_block(index, file, lines, start, end, rules) for start, end in _merge_adjacent(raw, lines)]
    return sorted(blocks + _docstrings(index, file, lines, rules), key=lambda block: block.span.start), None


def _comment_nodes(index: CodeIndex, file: str, language: str, refused: dict[str, str]) -> list[dict]:
    rule = f"id: comment\nlanguage: {grammar_of(language)}\nrule:\n  kind: comment"
    config = sgconfig_of(language)
    return list(tools.ast_grep_rules(rule, [file], index.root, config=config, refused=refused))


def _comment_ranges(matches: Sequence[dict]) -> list[tuple[int, int]]:
    return sorted(
        (match["range"]["start"]["line"] + 1, match["range"]["end"]["line"] + 1) for match in matches
    )


def _merge_adjacent(nodes: list[tuple[int, int]], lines: Sequence[str]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in nodes:
        if merged and _continues_line_comment(merged[-1], (start, end), lines):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _continues_line_comment(block: tuple[int, int], node: tuple[int, int], lines: Sequence[str]) -> bool:
    marker = _line_comment_marker(lines[node[0] - 1])
    return (
        node[0] == node[1] == block[1] + 1
        and marker is not None
        and marker == _line_comment_marker(lines[block[0] - 1])
    )


def _line_comment_marker(text: str) -> str | None:
    stripped = text.lstrip()
    return next((marker for marker in _LINE_COMMENT_MARKERS if stripped.startswith(marker)), None)


def _block(
    index: CodeIndex, file: str, lines: Sequence[str], start: int, end: int, rules: Sequence[FactRule]
) -> CommentBlock:
    text = "\n".join(lines[start - 1 : end])
    kind, attached = _kind_and_attachment(index, file, lines, start, end, text)
    return CommentBlock(Span(file, start, end), text, kind, attached, comment_facts(text, rules))


def _kind_and_attachment(
    index: CodeIndex, file: str, lines: Sequence[str], start: int, end: int, text: str
) -> tuple[CommentKind, Span | None]:
    if not _is_whole_line(lines[start - 1]):
        kind = CommentKind.TOOL_DIRECTIVE if _is_tool_directive(_trailing_part(text)) else CommentKind.INLINE
        return kind, Span(file, start, start)
    if _is_tool_directive(text):
        return CommentKind.TOOL_DIRECTIVE, None
    attached = operations.code_described_by_comment(index, file, end).span
    if text.lstrip().startswith("/**"):
        return CommentKind.JSDOC, attached
    if end + 1 == attached.start and attached.name:
        return CommentKind.DECLARATION, attached
    if _before_any_code(lines, start):
        return CommentKind.HEADER, None
    return CommentKind.BLOCK, attached


def _docstrings(
    index: CodeIndex, file: str, lines: Sequence[str], rules: Sequence[FactRule]
) -> list[CommentBlock]:
    if not file.endswith(".py"):
        return []
    owners = [(symbol, symbol.start + 1) for symbol in index.symbols_in(file)]
    owners.append((None, _first_code_line(lines)))
    blocks = []
    for owner, first_body_line in owners:
        end = _docstring_end(lines, first_body_line)
        if end is not None:
            text = "\n".join(lines[first_body_line - 1 : end])
            span = Span(file, first_body_line, end)
            blocks.append(CommentBlock(span, text, CommentKind.DOCSTRING, owner, comment_facts(text, rules)))
    return blocks


def _docstring_end(lines: Sequence[str], line: int | None) -> int | None:
    if line is None or line > len(lines):
        return None
    stripped = lines[line - 1].strip()
    quote = next((quote for quote in _DOCSTRING_QUOTES if stripped.startswith(quote)), None)
    if quote is None:
        return None
    if stripped.count(quote) >= 2:
        return line
    return next((number for number in range(line + 1, len(lines) + 1) if quote in lines[number - 1]), None)


def _first_code_line(lines: Sequence[str]) -> int | None:
    return next(
        (number for number, text in enumerate(lines, 1) if text.strip() and not text.startswith("#")), None
    )


def _before_any_code(lines: Sequence[str], start: int) -> bool:
    return all(not text.strip() or _is_whole_line(text) and _is_comment(text) for text in lines[: start - 1])


def noise_reason(block: CommentBlock) -> str | None:
    """A ready-made ``drop`` rule: ``tool_directive``, ``divider`` or ``licence_header``, else None."""
    if block.kind == CommentKind.TOOL_DIRECTIVE:
        return "tool_directive"
    if _DIVIDER.match(block.text.replace("\n", " ")):
        return "divider"
    body = " ".join(_comment_body_lines(block.text))
    if block.kind == CommentKind.HEADER and _LICENSE.search(body):
        return "licence_header"
    return None


def _is_tool_directive(text: str) -> bool:
    return all(_TOOL_DIRECTIVE.match(line) for line in _comment_body_lines(text) if line.strip())


def _trailing_part(line: str) -> str:
    for marker in (" #", " //"):
        if marker in line:
            return line.split(marker, 1)[1]
    return line


def _comment_body_lines(text: str) -> list[str]:
    return [_COMMENT_MARKERS.sub("", line) for line in text.splitlines()]


def _is_whole_line(text: str) -> bool:
    return _is_comment(text)


def _is_comment(text: str) -> bool:
    return text.strip().startswith(("#", "//", "/*", "*"))


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip())


def _changed_lines(index: CodeIndex, base: str, head: str) -> dict[str, set[int]]:
    """Changed line numbers per file, read from hunk headers. Paths stay unquoted so names with
    non-ASCII characters match the scope, and lines split at newlines only."""
    diff = tools.git(
        ["-c", "core.quotePath=false", "diff", "--no-color", "-U0", base, head, "--", *index.files],
        index.git_root,
    )
    changed: dict[str, set[int]] = {}
    current = None
    for line in diff.split("\n"):
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@") and current is not None:
            start, count = _hunk_range(line)
            changed.setdefault(current, set()).update(range(start, start + max(count, 1)))
    return changed


def _hunk_range(header: str) -> tuple[int, int]:
    added = header.split("+")[1].split(" ")[0]
    start, _, count = added.partition(",")
    return int(start), int(count) if count else 1


def _touches(block: CommentBlock, changed: set[int]) -> bool:
    spans = [block.span] + ([block.attached] if block.attached else [])
    return any(line in changed for span in spans for line in range(span.start, span.end + 1))
