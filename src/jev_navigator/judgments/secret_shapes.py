"""What a secret value looks like in code and config, and what stays code.

``hide_secrets`` applies every rule in order and returns the masked text with the values it
hid: token and hash shapes, credentials in URLs, and values under secret-named keys (shell and
env-file words, quoted, plain, bare and fallback values, structured values in
``secret_structures``), plus literal arguments to secret-named calls and high-entropy quoted
values. Which keys are secret and which values are code is ``secret_values``. The structural
rules recognize secret literals across code and configuration; the shared corpus tests their behavior.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter
from collections.abc import Callable

from .secret_structures import flow_spans, yaml_block_spans
from .secret_values import (
    CALL_OR_INDEX,
    CODE_REFERENCE,
    DOTTED_PATH,
    KEY,
    MASK,
    NAME_LITERAL,
    SEPARATOR,
    hides_under,
    is_literal,
    is_plain_words,
    key_kind,
    looks_generated,
)

BY_CONTENT_MIN_CHARS = 8
HIGH_ENTROPY_BITS_PER_CHAR = 4.0
HIGH_ENTROPY_MIN_CHARS = 20
TOKEN_CHARACTER_CLASS = r"[A-Za-z0-9+/=_\-]"

Span = tuple[int, int]

_KEY_BEGIN = r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----"
_KEY_END = r"-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----"
_KEY_MARKER = re.compile(rf"(?P<begin>{_KEY_BEGIN})|{_KEY_END}")
# Key material on a line in any layout (bare, quoted, appended, commented, numbered or diffed): a base64
# run of 16 or more characters. A key's last line may be shorter; it is padded or a multiple of four long.
_KEY_BODY_RUN = re.compile(r"[A-Za-z0-9+/]{16,}")
_KEY_TAIL_RUN = re.compile(
    r"(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{2,}={1,2}|(?:[A-Za-z0-9+/]{4})+)(?![A-Za-z0-9+/=])"
)
_KEY_HEADER = re.compile(r"[ \t\"'`#*/>+-]*(?:Proc-Type|DEK-Info|Version|Comment|Hash|Charset|MessageID):")
# A full line of key material (PEM, OpenSSH and armor wrap at 64 to 76 characters), and how many lines of
# armor (headers and the blank line before the body, in any layout) may stand between it and the BEGIN line.
_KEY_LINE_RUN = re.compile(r"[A-Za-z0-9+/]{40,}")
_ARMOR_MAX_LINES = 6
_KEY_MARKER_LINE = re.compile(r"^.*-----(?:BEGIN|END) [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*$", re.M)
_TOKEN_SHAPES = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\$2[aby]?\$\d{2}\$[./A-Za-z0-9]{53}"),
    re.compile(r"\$argon2(?:id|i|d)\$v=\d+\$m=\d+,t=\d+,p=\d+\$[A-Za-z0-9+/]+\$[A-Za-z0-9+/]+"),
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+(?P<value>[A-Za-z0-9._~+/=-]{16,})")
_URL_PASSWORD = re.compile(
    r"(?i)(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*+://[^/\s:@'\"`]++:(?P<value>[^/\s@'\"`]++)@"
)
_QUERY_VALUE = re.compile(r"[?&](?P<key>[A-Za-z_][\w.-]*+)=(?P<value>[^&\s#'\"`]++)")
_SHELL_WORD = (
    r"""(?:"(?:\\[\s\S]|\\\Z|[^"\\])*+(?:"|\Z)|'(?:\\[\s\S]|\\\Z|[^'\\])*+(?:'|\Z)|\\[\s\S]|[^\s"'\\])++"""
)
_SHELL_ASSIGNMENT = re.compile(
    rf"^[ \t]*(?:export[ \t]+)?{KEY}=(?P<value>(?![{{\[(]){_SHELL_WORD})"
    r"(?=[ \t]*(?:$|#|;|&&|\|\||[A-Za-z_]\w*=))",
    re.M,
)
_INLINE_ENV_ASSIGNMENT = re.compile(
    r"(?:^|(?<=[ \t;&|\"'`])|(?<=\\[nrt]))(?:export[ \t]+)?(?P<key>[A-Z_][A-Z0-9_]*+)="
    r"(?P<value>(?![{\[(])[^\s\"'`;&|\\]++)(?=[ \t;&|\"'`\\]|$)",
    re.M,
)
_CLI_SECRET_FLAG = re.compile(
    r"(?:^|(?<=\s))--?(?P<key>[A-Za-z][\w-]*+)(?:=|[ \t]++)(?P<value>[^\s\"'`=<>|&;$(-][^\s\"'`]*+)"
)
_USAGE_PLACEHOLDER = re.compile(r"\.\.\.|…|<[^<>]*>|\*+|x+", re.I)
_QUOTED_VALUE = re.compile(
    rf"{KEY}{SEPARATOR}(?P<paren>\([ \t\r\n]*+)?(?:[bBrRuUfF]{{1,2}}(?=[\"']))?"
    rf"(?P<quote>\"\"\"|'''|[\"'`])(?P<value>(?:\\[\s\S]|\\\Z|(?!(?P=quote))[^\\])*+)"
    r"(?:(?P=quote)(?P<tail>\w[^\s,;})\]\"'`]*+)?|\Z)"
)
_PLAIN_VALUE = re.compile(
    rf"^[ \t]*(?:-[ \t]+)?[\"']?{KEY}{SEPARATOR}"
    r"(?P<value>[^\s\"'`{\[(#](?:[^\n#/ \t]|[ \t]++(?=[^\s#])|/(?!/))*+)",
    re.M,
)
_BARE_VALUE = re.compile(
    rf"{KEY}{SEPARATOR}(?P<value>[\w.$@%+/~-][\w.$@%+/~=-]*+)(?=[ \t]*(?:$|[,;}})\]&]|#|//))", re.M
)
_CONFIG_SCALAR = re.compile(
    rf"^[ \t]*+(?:-[ \t]+)?(?:(?:export|ENV|ARG)[ \t]+)?[\"']?{KEY}[\"']?[ \t]*+[:=][ \t]*+"
    r"(?P<value>[^\s#\"'`{\[|>&*!](?:[^\n#]*[^\s#])?)[ \t]*+(?:#[^\n]*)?$",
    re.M,
)
_CONFIG_SUFFIXES = (".yml", ".yaml", ".env", ".ini", ".cfg", ".conf", ".properties", ".toml", ".dockerfile")
_CONFIG_NON_VALUES = frozenset({"null", "~", "true", "false", "yes", "no", "on", "off"})
_CI_EXPRESSION = re.compile(r"\$\{\{[^{}]*\}\}")
_LITERAL_FALLBACK = re.compile(
    rf"{KEY}{SEPARATOR}[^\n,;:=]*?(?:\|\||\?\?|\bor\b)\s*(?P<quote>[\"'`])(?P<value>[^\"'`\n]+)(?P=quote)"
)
_CALL = re.compile(r"(?<![\w.$])(?P<name>[\w.$]++)\((?P<arguments>[^(){}\[\]\n]*+)\)")
_SECRET_CALL_WORD = re.compile(r"(?i)secret|token|password|passwd|credential|api_?key|hmac")
_QUOTED_LITERAL = re.compile(r"(?P<quote>[\"'`])(?P<value>[^\"'`\n]+)(?P=quote)")
_ALGORITHM_NAME = re.compile(r"(?i)(?:sha|md|blake2[bs]?|hs|rs|es|ps)-?\d+")
_QUOTED_ASSIGNMENT = re.compile(
    rf"""[:=]\s*["'](?P<value>{TOKEN_CHARACTER_CLASS}{{{HIGH_ENTROPY_MIN_CHARS},}}+)["']"""
)
_IDENTIFIER_WORDS = re.compile(r"[A-Za-z]+(?:_[A-Za-z]+)*")
_PLACEHOLDER = re.compile(r"(?i)pass(?:word|wd)?|pwd|secret|token|x+|\*+|<[^>]*>|\.\.\.|…")


def hide_secrets(text: str, path: str | None = None) -> tuple[str, list[str]]:
    """The text with every secret value masked, and the values that were masked, in rule order. Text
    from a config file, or from no file, also has its unquoted values under secret keys masked."""
    hidden: list[str] = []
    for rule in _CONFIG_RULES if is_config_shaped(path) else _RULES:
        spans = rule(text)
        hidden += [text[start:end] for start, end in spans]
        text = _masked_spans(text, spans)
    return text, [value for value in hidden if value and value != MASK]


def is_config_shaped(path: str | None) -> bool:
    """Text from no file, or from a config file, a dotenv file or a Dockerfile."""
    if path is None:
        return True
    name = path.rsplit("/", 1)[-1].lower()
    return name.endswith(_CONFIG_SUFFIXES) or name.startswith((".env", "dockerfile"))


def _masked_spans(text: str, spans: list[Span]) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + MASK + text[end:]
    return text


def _matches(pattern: re.Pattern[str], hides: Callable[[re.Match[str]], bool] = bool):
    """Spans of the pattern's ``value`` group (the whole match when it has none) that ``hides`` accepts."""
    group = "value" if "value" in pattern.groupindex else 0

    def spans(text: str) -> list[Span]:
        return [match.span(group) for match in pattern.finditer(text) if hides(match)]

    return spans


_CONCATENATED = re.compile(
    r"[ \t]*+(?:\+|\.\.?|\\\r?\n)?[ \t]*+(?P<quote>[\"'])(?P<value>(?:\\.|(?!(?P=quote))[^\\\n])*+)(?P=quote)"
)
_CONCATENATED_IN_PARENS = re.compile(
    r"[ \t\r\n]*+(?:\+[ \t\r\n]*+)?(?P<quote>[\"'])(?P<value>(?:\\.|(?!(?P=quote))[^\\\n])*+)(?P=quote)"
)
MAX_NESTED_STRINGS = 3


def _quoted_spans(text: str, depth: int = 0) -> list[Span]:
    """A quoted value runs to its closing quote across escapes and lines, or to the end of the text
    when it never closes; letters glued to the closing quote belong to it. A quote that closes a string
    the key sat in (``'GitLab token: ' GITLAB_TOKEN``) opens no value. A string that is not hidden is
    read again for the secret assignments inside it (``HELP = 'set password = "..."'``)."""
    quotes = _QuotePositions(text)
    spans: list[Span] = []
    for match in _QUOTED_VALUE.finditer(text):
        if _hides_quoted(match, quotes):
            end = match.end("tail") if match["tail"] else match.end("value")
            clipped = quotes.enclosing_close(match.start("key"), match["quote"], match.start("value"), end)
            spans.append((match.start("value"), clipped))
            if clipped == end and not match["tail"] and match.end("value") < len(text):
                spans += _concatenated_spans(text, match.end(), bool(match["paren"]))
        elif depth < MAX_NESTED_STRINGS:
            offset = match.start("value")
            spans += [
                (offset + start, offset + end) for start, end in _quoted_spans(match["value"], depth + 1)
            ]
    return spans


def _hides_quoted(match: re.Match[str], quotes: _QuotePositions) -> bool:
    key_start = match.start("key")
    quote, key_end = match["quote"], match.end("key")
    own_quote = match.string[key_start - 1 : key_start] == quote == match.string[key_end : key_end + 1]
    if quotes.closes_an_enclosing_string(key_start - own_quote, quote):
        return False
    return bool(match["value"]) and _keyed_value(match, is_literal(match["value"], match["quote"]))


class _QuotePositions:
    """Where each unescaped single-character quote and each line starts, to tell in logarithmic time
    whether a quote after a key closes a string the key sat in: an odd count of that quote on the key's
    line before the key (or before the key's own opening quote, for a quoted key)."""

    def __init__(self, text: str) -> None:
        self._lines = [match.end() for match in re.finditer(r"\n", text)]
        self._quotes = {
            quote: [match.start() for match in re.finditer(rf"(?<!\\){re.escape(quote)}", text)]
            for quote in ('"', "'", "`")
        }

    def closes_an_enclosing_string(self, key_start: int, quote: str) -> bool:
        positions = self._quotes.get(quote)
        if positions is None:
            return False
        return self._count_on_line_before(positions, key_start) % 2 == 1

    def enclosing_close(self, key_start: int, quote: str, value_start: int, value_end: int) -> int:
        """Where the value ends at the latest: the closing quote of another quote's string the key sits in
        (``'DB_PASSWORD="unterminated'``), or ``value_end`` when no such string encloses it."""
        for other, positions in self._quotes.items():
            if other == quote or self._count_on_line_before(positions, key_start) % 2 == 0:
                continue
            closing = bisect.bisect_left(positions, value_start)
            if closing < len(positions) and positions[closing] < value_end:
                return positions[closing]
        return value_end

    def _count_on_line_before(self, positions: list[int], position: int) -> int:
        line_index = bisect.bisect_right(self._lines, position) - 1
        line_start = self._lines[line_index] if line_index >= 0 else 0
        return bisect.bisect_left(positions, position) - bisect.bisect_left(positions, line_start)


def _concatenated_spans(text: str, position: int, in_parens: bool) -> list[Span]:
    """The literals concatenated onto a hidden value (``"abc" + "def"``, ``"abc" . "def"``, adjacent
    literals across a line continuation, or across lines inside parentheses): they are the same value."""
    joiner = _CONCATENATED_IN_PARENS if in_parens else _CONCATENATED
    spans = []
    while part := joiner.match(text, position):
        spans.append((part.start("value"), part.end("value")))
        position = part.end()
    return spans


def _keyed(holds: Callable[[re.Match[str]], bool], scalar: bool = True) -> Callable[[re.Match[str]], bool]:
    """A rule's test for a value under a key: ``holds`` must call it a literal, and the key must be
    secret; a scalar rule also takes the other key kinds that ``_keyed_value`` names."""

    def hides(match: re.Match[str]) -> bool:
        return _keyed_value(match, holds(match), scalar)

    return hides


def _keyed_value(match: re.Match[str], literal: bool, scalar: bool = True) -> bool:
    """Whether a literal under the match's key is hidden, as ``hides_under`` decides. Unquoted plain words
    (``scalar`` off) are read only under keys a secret word ends: ``secret-scan: run nightly`` is prose,
    while every string literal under a suffixed key is still hidden."""
    kind = key_kind(match["key"])
    if not literal or kind is None or (kind != "secret" and not scalar):
        return False
    return hides_under(kind, match["key"], match["value"])


def _call_literal_spans(text: str) -> list[Span]:
    """Literal arguments to a secret-named call (``getSecret("...")``, ``sign(payload, "...")``) that look
    like key material: not a name, not a hash algorithm, and holding a digit or at least eight characters."""
    return [
        (call.start("arguments") + literal.start("value"), call.start("arguments") + literal.end("value"))
        for call in _CALL.finditer(text)
        if _is_secret_call(call["name"])
        for literal in _QUOTED_LITERAL.finditer(call.group("arguments"))
        if _is_key_material(literal["value"], literal["quote"])
    ]


def _is_secret_call(name: str) -> bool:
    function = name.rsplit(".", 1)[-1]
    return function == "sign" or bool(_SECRET_CALL_WORD.search(function))


def _is_key_material(value: str, quote: str) -> bool:
    return (
        is_literal(value, quote)
        and not NAME_LITERAL.fullmatch(value)
        and not _ALGORITHM_NAME.fullmatch(value)
        and (len(value) >= BY_CONTENT_MIN_CHARS or any(character.isdigit() for character in value))
    )


def _inline_env_literal(match: re.Match[str]) -> bool:
    """An upper-case assignment anywhere on a shell, Makefile or CI line (``run: API_TOKEN=... npm test``),
    unless it is a usage placeholder (``KEY=...``, ``KEY=<credential>``)."""
    return _shell_literal(match) and not _USAGE_PLACEHOLDER.fullmatch(match["value"])


def _shell_literal(match: re.Match[str]) -> bool:
    """A shell word, unless it is code: a reference, call or index, an interpolation, or a keyword
    argument ending in a comma or bracket."""
    value = match["value"]
    quote = value[0] if value[:1] in ("'", '"') else '"'
    return (
        not value.endswith((",", ")", ";"))
        and is_literal(value.strip("\"'"), quote)
        and not DOTTED_PATH.fullmatch(value)
        and not CALL_OR_INDEX.match(value)
    )


def _plain_literal(match: re.Match[str]) -> bool:
    return is_plain_words(match["value"])


def _bare_literal(match: re.Match[str]) -> bool:
    value = match["value"]
    return any(c.isalnum() for c in value) and (not CODE_REFERENCE.fullmatch(value) or looks_generated(value))


def _quoted_literal(match: re.Match[str]) -> bool:
    return is_literal(match["value"], match["quote"])


def _url_password(match: re.Match[str]) -> bool:
    """A password in a URL, unless it is a placeholder (``user:pass@host``) or a reference."""
    return is_literal(match["value"]) and not _PLACEHOLDER.fullmatch(match["value"])


def _query_secret(match: re.Match[str]) -> bool:
    value = match["value"]
    return (
        key_kind(match["key"]) in ("secret", "suffixed")
        and is_literal(value)
        and any(c.isalnum() for c in value)
        and not _PLACEHOLDER.fullmatch(value)
    )


def _config_literal(match: re.Match[str]) -> bool:
    """An unquoted config value, unless it is empty, a boolean, or a whole reference: ``${VAR}``, ``$VAR``
    or a CI expression (``${{ secrets.TOKEN }}``)."""
    value = match["value"]
    return (
        value.lower() not in _CONFIG_NON_VALUES and not _CI_EXPRESSION.fullmatch(value) and is_literal(value)
    )


def _bearer_value(match: re.Match[str]) -> bool:
    return any(character.isdigit() for character in match["value"])


def _high_entropy_value(match: re.Match[str]) -> bool:
    """Identifier words (``RunAttemptConflictError``, ``max_items``) are code, whatever their entropy."""
    value = match["value"]
    return not _IDENTIFIER_WORDS.fullmatch(value) and is_high_entropy(value)


def is_high_entropy(value: str) -> bool:
    """Whether a value's characters are spread like a random token's: at least
    ``HIGH_ENTROPY_BITS_PER_CHAR`` bits of Shannon entropy per character."""
    counts = Counter(value)
    bits = -sum(count / len(value) * math.log2(count / len(value)) for count in counts.values())
    return bits >= HIGH_ENTROPY_BITS_PER_CHAR


def _private_key_spans(text: str) -> list[Span]:
    """Each private key block: from its BEGIN marker to the END marker that closes it, the nearest one
    no later BEGIN claims first. A BEGIN no END closes is the BEGIN line and the key body lines after
    it (``_unterminated_key_end``), never the rest of the text: code that only mentions the marker
    keeps its code. One pass over the markers, so the work grows with the text."""
    open_begins: list[int] = []
    spans: list[Span] = []
    floor = 0
    for marker in _KEY_MARKER.finditer(text):
        if marker["begin"]:
            open_begins.append(marker.start())
        elif open_begins:
            spans.append((open_begins.pop(), marker.end()))
        else:
            spans.append((_unopened_key_start(text, marker.start(), floor), marker.end()))
        floor = marker.end()
    covered = 0
    for begin in open_begins:
        if begin >= covered:
            covered = _unterminated_key_end(text, begin)
            spans.append((begin, covered))
    return merged_spans(spans)


def _unterminated_key_end(text: str, begin: int) -> int:
    """Where a key without an END marker ends: after its BEGIN line, its armor (``_armor_end``) or any
    PEM or armor headers, the following lines that hold key material, and one shorter last line: padded,
    a multiple of four characters long, or cut by the end of the text. It never reaches a line holding
    another marker, which starts or ends a key of its own."""
    end = _armor_end(text, _line_end(text, begin))
    in_body = False
    while end < len(text):
        line_end = _line_end(text, end + 1)
        line = text[end + 1 : line_end]
        if _KEY_MARKER.search(line):
            return end
        if _KEY_BODY_RUN.search(line):
            in_body = True
        elif in_body and line.strip() and (line_end == len(text) or _KEY_TAIL_RUN.search(line)):
            return line_end
        elif line.strip() and (in_body or not _KEY_HEADER.match(line)):
            return end
        end = line_end
    return end


def _armor_end(text: str, end: int) -> int:
    """Where a key's armor ends, from the end of its BEGIN line: before the first full line of key material
    within ``_ARMOR_MAX_LINES`` lines, whatever the lines between hold; the BEGIN line's end when none
    follows before another marker."""
    position = end
    for _ in range(_ARMOR_MAX_LINES + 1):
        if position >= len(text):
            break
        line_end = _line_end(text, position + 1)
        line = text[position + 1 : line_end]
        if _KEY_MARKER.search(line):
            break
        if _KEY_LINE_RUN.search(line):
            return position
        position = line_end
    return end


def _unopened_key_start(text: str, end_marker: int, floor: int) -> int:
    """Where a key whose BEGIN marker lies outside the text starts (a window that opens inside it): the
    lines before its END line that hold key material, the one right above it possibly a shorter last
    line, back to ``floor`` at most."""
    start = max(text.rfind("\n", floor, end_marker) + 1, floor)
    last_line = True
    while start > floor:
        line_start = max(text.rfind("\n", floor, start - 1) + 1, floor)
        line = text[line_start : start - 1]
        if not (_KEY_BODY_RUN.search(line) or (last_line and _KEY_TAIL_RUN.search(line))):
            break
        start, last_line = line_start, False
    return start


def _line_end(text: str, position: int) -> int:
    end = text.find("\n", position)
    return len(text) if end == -1 else end


def merged_spans(spans: list[Span]) -> list[Span]:
    """The spans in order, with overlapping or touching spans joined into one."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


_RULES: tuple[Callable[[str], list[Span]], ...] = (
    _private_key_spans,
    _matches(_KEY_MARKER_LINE),
    *(_matches(shape) for shape in _TOKEN_SHAPES),
    _matches(_BEARER_VALUE, _bearer_value),
    _matches(_URL_PASSWORD, _url_password),
    _matches(_QUERY_VALUE, _query_secret),
    _matches(_SHELL_ASSIGNMENT, _keyed(_shell_literal)),
    _matches(_INLINE_ENV_ASSIGNMENT, _keyed(_inline_env_literal)),
    _matches(_CLI_SECRET_FLAG, _keyed(_inline_env_literal)),
    _quoted_spans,
    flow_spans,
    yaml_block_spans,
    _matches(_PLAIN_VALUE, _keyed(_plain_literal, scalar=False)),
    _matches(_BARE_VALUE, _keyed(_bare_literal)),
    _matches(_LITERAL_FALLBACK, _keyed(_quoted_literal)),
    _call_literal_spans,
    _matches(_QUOTED_ASSIGNMENT, _high_entropy_value),
)
_CONFIG_RULES = (*_RULES, _matches(_CONFIG_SCALAR, _keyed(_config_literal)))
