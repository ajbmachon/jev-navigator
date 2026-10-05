"""What a secret value looks like in code and config, and what stays code.

``hide_secrets`` applies every rule in order and returns the masked text with the values it
hid: token and hash shapes, credentials in URLs, and values under secret-named keys (shell and
env-file words, quoted, plain, bare and fallback values, structured values in
``secret_structures``), plus literal arguments to secret-named calls and high-entropy quoted
values. Which keys are secret and which values are code is ``secret_values``. The structural
rules follow the analysis engine's audit masker, whose corpus both sides test.
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

_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----|\Z)", re.S
)
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


_RULES: tuple[Callable[[str], list[Span]], ...] = (
    _matches(_PRIVATE_KEY_BLOCK),
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
