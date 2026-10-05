"""Secret values that span structure: nested flow values and YAML blocks under a secret key.

A nested value under a secret key is hidden whole when it holds a literal (a quoted string, or a
scalar that is not a reference, under a key that does not name something), so a Kubernetes
``secret:`` volume or ``{ type: String, required: true }`` stays code. Each region is scanned once:
a start inside a region already scanned is skipped, and a flow value that never closes is hidden to
the end of the text when it holds a literal, since a clipped window may cut it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .secret_values import CALL_OR_INDEX, CODE_REFERENCE, KEY, SEPARATOR, is_plain_words, key_kind

Span = tuple[int, int]

_FLOW_VALUE = re.compile(rf"{KEY}{SEPARATOR}(?P<value>(?=[\[{{]))")
_YAML_HEADER = re.compile(
    rf"^(?P<leader>[ \t]*(?:-[ \t]+)?)[\"']?{KEY}[\"']?[ \t]*:(?![:=])[ \t]*(?P<value>[^\r\n]*+)$",
    re.M,
)
_YAML_PROPERTIES = re.compile(r"(?:(?:&[^\s|>]+|!(?:<[^>\r\n]+>|[^\s|>]*))[ \t]+)+")
_YAML_BLOCK_SCALAR = re.compile(r"(?:(?:&[^\s|>]+|!(?:<[^>\r\n]+>|[^\s|>]*))[ \t]+)*[|>][-+0-9]*")
_NESTED_COMMENT = re.compile(r"(?://|(?:^|(?<=\s))#)[^\n]*+")
_NESTED_QUOTED = re.compile(r"(?P<quote>[\"'`])(?:\\.|(?!(?P=quote)).)*+(?:(?P=quote)|\Z)", re.S)
_NESTED_LEAF = re.compile(r"(?<![\w$.-])(?P<key>[\w$.-]++)[\"']?[ \t]*:(?![:=])[ \t]*(?P<value>[^,;}\]\n]*+)")
_KEY_BEFORE = re.compile(r"(?P<key>[\w$.-]+)[\"']?[ \t]*:[ \t]*$")
_KEY_BEFORE_WINDOW = 200
# A nested key that names or describes something holds metadata, not a secret value.
_NAMING_KEY = re.compile(
    r"(?i)(?:^|[_.-]|(?<=[a-z0-9])(?=[A-Z]))(?:name|path|dir|directory|file|header|ref|url|uri|type|kind"
    r"|field|label|annotation|mount|env|key|items|mode|description|in|enabled|required|optional)$"
)


def flow_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    scanned_to = 0
    for match in _FLOW_VALUE.finditer(text):
        start = match.start("value")
        if start < scanned_to or key_kind(match["key"]) != "secret":
            continue
        end = _balanced_flow_value_end(text, start)
        if end is None:
            continue
        scanned_to = end
        if holds_literal_leaf(text[start + 1 : end]):
            spans.append((start, end))
    return spans


def holds_literal_leaf(content: str) -> bool:
    """Whether nested content under a secret key holds a literal, as the module docstring says."""
    content = _NESTED_COMMENT.sub("", content)
    quoted = any(
        not _under_naming_key(content, match.start()) and not _is_subscript(content, match)
        for match in _NESTED_QUOTED.finditer(content)
    )
    unquoted = _NESTED_QUOTED.sub("", content)
    return quoted or any(_is_literal_leaf(leaf) for leaf in _NESTED_LEAF.finditer(unquoted))


def _is_subscript(content: str, match: re.Match[str]) -> bool:
    """A quoted index (``c['gate_pass']``) names a field; it is not a literal value."""
    return content[match.start() - 1 : match.start()] == "[" and content[match.end() : match.end() + 1] == "]"


def _under_naming_key(content: str, position: int) -> bool:
    key = _KEY_BEFORE.search(content[max(0, position - _KEY_BEFORE_WINDOW) : position])
    return key is not None and bool(_NAMING_KEY.search(key["key"]))


def _is_literal_leaf(leaf: re.Match[str]) -> bool:
    value = leaf["value"].strip()
    return (
        bool(value)
        and not _NAMING_KEY.search(leaf["key"])
        and not CODE_REFERENCE.fullmatch(value)
        and not value.startswith(("(", "[", "{", "=", "&", "!", "|", "<"))
        and "=>" not in value
        and not CALL_OR_INDEX.match(value)
    )


def yaml_block_spans(text: str) -> list[Span]:
    """A secret key's YAML block scalar, empty value with indented lines, or plain value continued
    on deeper lines: the value and its continuation lines, never the sibling keys. Under a suffixed key
    (``DB_PASSWORD_PROD``) only a value the key itself holds counts; a nested table's inner keys are
    judged on their own."""
    spans: list[Span] = []
    scanned_to = 0
    for header in _YAML_HEADER.finditer(text):
        if header.start() < scanned_to or not _holds_yaml_value(header):
            continue
        continuation_end = _continuation_end(text, header)
        if continuation_end is None:
            continue
        scanned_to = continuation_end
        if _yaml_value_continues(text, header, continuation_end):
            end = continuation_end - 1 if text[continuation_end - 1] == "," else continuation_end
            spans.append((_yaml_value_start(text, header), end))
    return spans


def _holds_yaml_value(header: re.Match[str]) -> bool:
    kind = key_kind(header["key"])
    return kind == "secret" or (kind == "suffixed" and bool(header["value"].strip()))


def _continuation_end(text: str, header: re.Match[str]) -> int | None:
    """The end of the last line indented deeper than the key, or None when no such line follows."""
    key_column = len(header["leader"].expandtabs(8))
    end = None
    for line_start, line_end in _following_lines(text, header.end()):
        content = text[line_start:line_end]
        if not content.strip():
            continue
        if len(content[: len(content) - len(content.lstrip(" \t"))].expandtabs(8)) <= key_column:
            break
        end = line_end
    return end


def _following_lines(text: str, position: int) -> Iterator[Span]:
    """Each line after ``position`` (which ends a line), without its line ending."""
    while position < len(text):
        line_start = position + 1
        line_end = text.find("\n", line_start)
        line_end = len(text) if line_end == -1 else line_end
        yield line_start, line_end - 1 if text[line_end - 1 : line_end] == "\r" else line_end
        position = line_end


def _yaml_value_continues(text: str, header: re.Match[str], continuation_end: int) -> bool:
    """A block scalar, YAML properties or plain words continue as a value; an empty value continues
    as nested content, which is a value only when it holds a literal."""
    value = header["value"].strip()
    if not value:
        return holds_literal_leaf(text[header.end() : continuation_end])
    return bool(_YAML_BLOCK_SCALAR.fullmatch(value) or _YAML_PROPERTIES.match(value + " ")) or is_plain_words(
        value
    )


def _yaml_value_start(text: str, header: re.Match[str]) -> int:
    """Where the hidden value begins: at the inline value, or at the first indented line when it is empty."""
    if not header["value"].strip():
        return _first_content(text, header.end())
    return header.start("value")


def _first_content(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _balanced_flow_value_end(text: str, start: int) -> int | None:
    """The end of a quote-aware balanced object or array value (the engine's flow grammar), the end of
    the text when the value never closes, or None when a closer does not match."""
    expected_closers = {"{": "}", "[": "]"}
    stack = [expected_closers[text[start]]]
    scalar_started = [False]
    last_scalar_was_quoted = [False]
    quote: str | None = None
    index = start + 1
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == "\\" and quote == '"':
                index += 2
                continue
            if character == quote:
                if quote == "'" and index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                quote = None
                scalar_started[-1] = True
                last_scalar_was_quoted[-1] = True
            index += 1
            continue

        if not scalar_started[-1] and character in {"&", "!"}:
            if character == "!" and index + 1 < len(text) and text[index + 1] == "<":
                tag_end = text.find(">", index + 2)
                if tag_end < 0:
                    return len(text)
                index = tag_end + 1
                continue
            index += 1
            while index < len(text) and (not text[index].isspace() and text[index] not in "[]{},"):
                index += 1
            continue

        if (
            not scalar_started[-1]
            and character == "?"
            and index + 1 < len(text)
            and text[index + 1].isspace()
        ):
            index += 1
            continue

        if character == "#" and (not scalar_started[-1] or text[index - 1].isspace()):
            line_end = text.find("\n", index + 1)
            if line_end < 0:
                return len(text)
            index = line_end + 1
            continue

        if character in {'"', "'"} and not scalar_started[-1]:
            quote = character
        elif character in expected_closers:
            stack.append(expected_closers[character])
            scalar_started.append(False)
            last_scalar_was_quoted.append(False)
        elif character in {"}", "]"}:
            if character != stack[-1]:
                return None
            stack.pop()
            scalar_started.pop()
            last_scalar_was_quoted.pop()
            if not stack:
                return index + 1
            scalar_started[-1] = True
            last_scalar_was_quoted[-1] = False
        elif character == ",":
            scalar_started[-1] = False
            last_scalar_was_quoted[-1] = False
        elif character == ":":
            next_character = text[index + 1] if index + 1 < len(text) else ""
            is_separator = (
                last_scalar_was_quoted[-1]
                or not next_character
                or next_character.isspace()
                or next_character in "[]{},"
            )
            scalar_started[-1] = not is_separator
            last_scalar_was_quoted[-1] = False
        elif not character.isspace():
            scalar_started[-1] = True
            last_scalar_was_quoted[-1] = False
        index += 1
    return len(text)
