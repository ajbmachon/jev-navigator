"""Keep secrets out of every Jev request: mask what is found, then scan the final request and refuse on a hit.

Masking works by content: every value the masker hides anywhere in a request is hidden everywhere
in it, so a token found in an assignment is also hidden where a relation text or another item quotes
it. The built-in masker and scanner are lightweight and on by default. A host with a stronger scanner
passes its own objects; turning either off must be explicit (``masker=None`` or ``scanner=None``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Protocol

MASK = "[MASKED]"
BY_CONTENT_MIN_CHARS = 8
RANDOM_VALUE_MIN_CHARS = 16
HIGH_ENTROPY_BITS_PER_CHAR = 4.0
HIGH_ENTROPY_MIN_CHARS = 20
TOKEN_CHARACTER_CLASS = r"[A-Za-z0-9+/=_\-]"

_KEY = r"(?:(?<![\w$.\\-])|(?<=\\[nrt]))(?P<key>[A-Za-z_$][\w$.-]*+)"
_SEPARATOR = r"[\"']?\s*[:=]\s*"
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----|\Z)", re.S
)
_KEY_MARKER_LINE = re.compile(r"^.*-----(?:BEGIN|END) [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*$", re.M)
_TOKEN_SHAPES = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\$2[aby]?\$\d{2}\$[./A-Za-z0-9]{53}"),
    re.compile(r"\$argon2(?:id|i|d)\$v=\d+\$m=\d+,t=\d+,p=\d+\$[A-Za-z0-9+/]+\$[A-Za-z0-9+/]+"),
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+(?P<value>[A-Za-z0-9._~+/=-]{16,})")
_URL_PASSWORD = re.compile(
    r"(?i)(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*+://[^/\s:@'\"`]++:(?P<value>[^/\s@'\"`]++)@"
)
_QUERY_VALUE = re.compile(r"[?&](?P<key>[A-Za-z_][\w.-]*+)=(?P<value>[^&\s#'\"`]++)")
_ENV_FILE_VALUE = re.compile(
    r"(?:^|(?<=[ \t;&|\"'`])|(?<=\\[nrt]))(?:export[ \t]+)?(?P<key>[A-Z_][A-Z0-9_]*+)="
    r"(?P<value>[^\s\"'`#$()\[\]{}\\][^\s\"'`#()\[\]{}\\]*+)(?=[ \t;&|\"'`\\]|$)",
    re.M,
)
_CLI_SECRET_FLAG = re.compile(
    r"(?:^|(?<=\s))--?(?P<key>[A-Za-z][\w-]*+)(?:=|[ \t]++)(?P<value>[^\s\"'`=-][^\s\"'`]*+)"
)
_USAGE_PLACEHOLDER = re.compile(r"\.\.\.|…|<[^<>]*>|\*+|x+", re.I)
_CONFIG_SCALAR = re.compile(
    rf"^[ \t]*+(?:-[ \t]+)?(?:(?:export|ENV|ARG)[ \t]+)?[\"']?{_KEY}[\"']?[ \t]*+[:=][ \t]*+"
    r"(?P<value>[^\s#\"'`{\[|>&*!](?:[^\n#]*[^\s#])?)[ \t]*+(?:#[^\n]*)?$",
    re.M,
)
_SHORT_NUMBER = re.compile(r"[\d.,:_+-]{1,7}")
_CONFIG_SUFFIXES = (".yml", ".yaml", ".env", ".ini", ".cfg", ".conf", ".properties", ".toml", ".dockerfile")
_CONFIG_NON_VALUES = frozenset({"null", "~", "true", "false", "yes", "no", "on", "off"})
_CI_EXPRESSION = re.compile(r"\$\{\{[^{}]*\}\}")
_QUOTED_SECRET_VALUE = re.compile(rf"{_KEY}{_SEPARATOR}(?P<quote>[\"'`])(?P<value>[^\"'`\n]++)(?P=quote)")
_LITERAL_FALLBACK = re.compile(
    rf"{_KEY}{_SEPARATOR}[^\n,;:=]*?(?:\|\||\?\?|\bor\b)\s*(?P<quote>[\"'`])(?P<value>[^\"'`\n]+)(?P=quote)"
)
_BARE_SECRET_VALUE = re.compile(
    rf"{_KEY}{_SEPARATOR}(?P<value>[\w.$@%+/~-][\w.$@%+/~=-]*+)(?=[ \t]*(?:$|[,;}})\]&]|#|//))", re.M
)
_QUOTED_ASSIGNMENT = re.compile(
    rf"""[:=]\s*["'](?P<value>{TOKEN_CHARACTER_CLASS}{{{HIGH_ENTROPY_MIN_CHARS},}}+)["']"""
)
_CALL = re.compile(r"(?<![\w.$])(?P<name>[\w.$]++)\((?P<arguments>[^(){}\[\]\n]*+)\)")
_SECRET_CALL_WORD = re.compile(r"(?i)secret|token|password|passwd|credential|api_?key|hmac")
_QUOTED_LITERAL = re.compile(r"(?P<quote>[\"'`])(?P<value>[^\"'`\n]+)(?P=quote)")
_KEY_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SECRET_WORDS = (
    ("secret", "access", "key"),
    ("secret", "key"),
    ("access", "key"),
    ("private", "key"),
    ("api", "key"),
    ("apikey",),
    ("secretkey",),
    ("password",),
    ("passwd",),
    ("pass",),
    ("credentials",),
    ("pwd",),
    ("secret",),
    ("token",),
    ("credential",),
)
_NAMING_SUFFIXES = frozenset(
    {
        "name", "id", "ref", "path", "dir", "directory", "file", "url", "uri", "endpoint", "host", "header",
        "label", "annotation", "mount", "env", "type", "kind", "field", "count", "length", "size", "prefix",
        "pattern", "patterns", "rule", "rules", "regex",
    }
)  # fmt: skip
_CODE_REFERENCE = re.compile(
    r"\$?[A-Za-z_]\w*(?:\.\$?[A-Za-z_]\w*)*|\d+(?:-\d+)+|-?(?:0x[\da-fA-F]+|\d[\d_]*(?:\.\d+)*[A-Za-z%]{0,4})"
)
_INTERPOLATION = re.compile(r"\$\{[^{}]*\}|\$\([^()]*\)")
_VARIABLE = re.compile(r"\$[A-Za-z_]\w*")
_DOTTED_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_NAME_SHAPED = re.compile(r"[A-Za-z_][A-Za-z_.\-]*")
_PATH_SHAPED = re.compile(r"(?:~|\.{1,2})?/?[\w.@-]+(?:/[\w.@\[\]-]+)+/?|/[\w.@-]*")
_URL_SHAPED = re.compile(r"[a-z][a-z0-9+.-]*://\S+")
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
_MESSAGE_SUFFIXES = frozenset(
    {
        "message",
        "msg",
        "error",
        "err",
        "text",
        "hint",
        "title",
        "description",
        "placeholder",
        "prompt",
        "help",
    }
)
_DEFAULT_PASSWORDS = frozenset({"password", "passwd", "pwd", "secret", "admin", "root"})
_KEY_SEPARATORS = re.compile(r"[._\-$\s]+")
_NAME_LITERAL = re.compile(r"[A-Z][A-Z0-9_]*|[a-z]+(?:[-_./][a-z]+)*")
_ALGORITHM_NAME = re.compile(r"(?i)(?:sha|md|blake2[bs]?|hs|rs|es|ps)-?\d+")


class Masker(Protocol):
    """``mask`` hides secrets in one text; ``masked_values`` lists the values it hides there, so they
    can be hidden everywhere else in the request too."""

    def mask(self, text: str, path: str | None = None) -> str: ...

    def masked_values(self, text: str, path: str | None = None) -> list[str]: ...


class Scanner(Protocol):
    def findings(self, text: str, path: str | None = None) -> list[str]: ...


class SecretInRequestError(RuntimeError):
    """The final pre-send scan found a secret; the request was not sent."""


@dataclass(frozen=True)
class SecretMasker:
    """Masks secret values and keeps code: private-key blocks, common token shapes, Bearer values,
    env-file values, quoted, bare and fallback values under secret-named keys, literal arguments to
    secret-named calls, and high-entropy quoted values in assignments. A value that is a reference
    (an identifier, dotted path, call, env lookup or interpolation) is code and stays.

    ``masked_values`` lists every masked value but a short number (``"1.5"``, ``"0"``), so request masking
    hides each copy elsewhere too: anywhere for a value of ``BY_CONTENT_MIN_CHARS`` or more characters,
    as a whole word for a shorter one (see ``_copy_pattern``).

    ``path`` is the file the text comes from. In a config file, or in text from no file, an unquoted value
    under a secret key is a value (``POSTGRES_PASSWORD: example``); in code it stays (``token: str``)."""

    def mask(self, text: str, path: str | None = None) -> str:
        return _hide_secrets(text, path)[0]

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return [value for value in _hide_secrets(text, path)[1] if not _SHORT_NUMBER.fullmatch(value)]


@dataclass(frozen=True)
class SecretScanner:
    """Reports what the built-in masker would have masked; used as the final check before sending."""

    def findings(self, text: str, path: str | None = None) -> list[str]:
        return [value[:4] for value in _hide_secrets(text, path)[1]]


def mask_request(state: Mapping, questions: Mapping, masker: Masker) -> tuple[Mapping, Mapping, frozenset]:
    """The masked state and questions, and every value that was hidden in either."""
    values = masked_values([state, questions], masker)
    return mask_everywhere(state, masker, values), mask_everywhere(questions, masker, values), values


def mask_by_content(value: object, masker: Masker) -> object:
    """Masks nested JSON-like data so that a value hidden in one string is hidden in all of them."""
    return mask_everywhere(value, masker, masked_values(value, masker))


def masked_values(value: object, masker: Masker) -> frozenset[str]:
    """Every value the masker hides anywhere inside nested JSON-like data, keys included."""
    return frozenset(
        found for text, path in dict.fromkeys(_strings(value)) for found in masker.masked_values(text, path)
    )


def mask_everywhere(value: object, masker: Masker, values: frozenset[str]) -> object:
    """Masks every string by the masker's rules, then hides each of ``values`` wherever it still
    appears. Keys are left as they are; ``refuse_if_secret`` refuses a request with one in a key."""
    copies = [_copy_pattern(secret) for secret in sorted(values - {MASK}, key=len, reverse=True)]

    @cache
    def hide(text: str, path: str | None) -> str:
        text = masker.mask(text, path)
        for copy in copies:
            text = copy.sub(MASK, text)
        return text

    return _each_string(value, hide)


def safe_options(options: Mapping[str, str], masker: Masker | None) -> dict[str, str]:
    """Options whose key would change under masking are dropped: a secret is never offered as a choice."""
    if masker is None:
        return dict(options)
    return {key: masker.mask(text) for key, text in options.items() if masker.mask(key) == key}


def refuse_if_secret(
    state: Mapping, questions: Mapping, scanner: Scanner | None, masked: frozenset[str] = frozenset()
) -> None:
    """Refuses when a value masked elsewhere is still in the request (it can only sit in a key), or
    when the scanner finds a secret."""
    texts = _strings(state) + _strings(questions)
    copies = [_copy_pattern(value) for value in masked - {MASK}]
    if any(copy.search(text) for text, _ in texts for copy in copies):
        raise SecretInRequestError("a masked value is still in the request, in a key; nothing was sent")
    if scanner is None:
        return
    for text, path in texts:
        if scanner.findings(text, path):
            raise SecretInRequestError("the final scan found a secret in the request; nothing was sent")


def _copy_pattern(value: str) -> re.Pattern[str]:
    """Where a masked value's copies stand: anywhere for a value of ``BY_CONTENT_MIN_CHARS`` or more
    characters, and as a whole word for a shorter one, so ``hunter2`` is hidden but ``hunter2x`` stays."""
    if len(value) >= BY_CONTENT_MIN_CHARS:
        return re.compile(re.escape(value))
    return re.compile(rf"(?<![\w$]){re.escape(value)}(?![\w$])")


def _each_string(value: object, change: Callable[[str, str | None], str], path: str | None = None) -> object:
    """Changes every string, each with the file it comes from (see ``_file_of``)."""
    if isinstance(value, str):
        return change(value, path)
    if isinstance(value, Mapping):
        inner = _file_of(value, path)
        return {key: _each_string(item, change, inner) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_each_string(item, change, path) for item in value]
    return value


def _strings(value: object, path: str | None = None) -> list[tuple[str, str | None]]:
    """Every string, keys included, with the file it comes from."""
    if isinstance(value, str):
        return [(value, path)]
    if isinstance(value, Mapping):
        inner = _file_of(value, path)
        return [pair for key, item in value.items() for pair in [(str(key), None), *_strings(item, inner)]]
    if isinstance(value, list | tuple):
        return [pair for item in value for pair in _strings(item, path)]
    return []


def _file_of(mapping: Mapping, outer: str | None) -> str | None:
    """A mapping's ``file`` names the file its strings come from (a slice's code, a candidate's lines)."""
    file = mapping.get("file")
    return file if isinstance(file, str) else outer


def _is_config_shaped(path: str | None) -> bool:
    """Text from no file, or from a config file, a dotenv file or a Dockerfile."""
    if path is None:
        return True
    name = path.rsplit("/", 1)[-1].lower()
    return name.endswith(_CONFIG_SUFFIXES) or name.startswith((".env", "dockerfile"))


def _hide_secrets(text: str, path: str | None = None) -> tuple[str, list[str]]:
    """The text with every secret value masked, and the values that were masked, in rule order."""
    hidden: list[str] = []
    for rule in _CONFIG_RULES if _is_config_shaped(path) else _RULES:
        spans = rule(text)
        hidden += [text[start:end] for start, end in spans]
        text = _masked_spans(text, spans)
    return text, [value for value in hidden if value != MASK]


def _masked_spans(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + MASK + text[end:]
    return text


def _matches(pattern: re.Pattern[str], hides: Callable[[re.Match[str]], bool] = bool):
    """Spans of the pattern's ``value`` group (the whole match when it has none) that ``hides`` accepts."""
    group = "value" if "value" in pattern.groupindex else 0

    def spans(text: str) -> list[tuple[int, int]]:
        return [match.span(group) for match in pattern.finditer(text) if hides(match)]

    return spans


def _call_literal_spans(text: str) -> list[tuple[int, int]]:
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
        _is_literal(value, quote)
        and not _NAME_LITERAL.fullmatch(value)
        and not _ALGORITHM_NAME.fullmatch(value)
        and (len(value) >= BY_CONTENT_MIN_CHARS or any(character.isdigit() for character in value))
    )


def _key_kind(key: str) -> str | None:
    """The key's kind when a secret word is one of its parts: "secret" when the secret word ends the key
    (``DB_PASSWORD``, ``authToken``, ``credentials``), "naming" when a naming word ends it (``SECRET_ENV``,
    ``token_url``, ``secretAccessKeyId``), "message" when a message word ends it (``PASSWORD_ERROR``), else
    "suffixed" (``SECRET_KEY_BASE``, ``GH_TOKEN_RO``). None when
    no part is a secret word (``max_tokens``, ``tokenizer``, ``bypass``)."""
    parts = _key_parts(key)
    end = _last_secret_word_end(parts)
    if end is None:
        return None
    if end == len(parts):
        return "secret"
    if parts[-1] in _NAMING_SUFFIXES:
        return "naming"
    return "message" if parts[-1] in _MESSAGE_SUFFIXES else "suffixed"


def _key_parts(key: str) -> list[str]:
    return [part.lower() for piece in re.split(r"[._\-$]+", key) for part in _KEY_PART.findall(piece)]


def _last_secret_word_end(parts: list[str]) -> int | None:
    """Where the last secret word among the parts ends, or None when no part is one."""
    for start in range(len(parts) - 1, -1, -1):
        for word in _SECRET_WORDS:
            if tuple(parts[start : start + len(word)]) == word:
                return start + len(word)
    return None


def _keyed_value_hides(kind_holds: Callable[[str], bool]) -> Callable[[re.Match[str]], bool]:
    """A literal under a key that holds a secret is hidden as ``_hides_under`` decides."""

    def hides(match: re.Match[str]) -> bool:
        kind = _key_kind(match["key"])
        return kind is not None and kind_holds(match) and _hides_under(kind, match["key"], match["value"])

    return hides


def _hides_under(kind: str, key: str, value: str) -> bool:
    """A value that repeats its key (``PASS: "PASS"``) shows nothing the key does not. Otherwise a secret
    key hides every literal; a suffixed key every literal but an environment variable's name, a path or a
    URL; a message key the same, except a sentence; a naming key only a credential-looking word, one word
    of eight or more characters that names nothing."""
    if _repeats_its_key(key, value):
        return False
    if kind == "secret":
        return True
    if kind == "message" and any(character.isspace() for character in value):
        return False
    if kind in ("suffixed", "message"):
        return not (
            _ENVIRONMENT_NAME.fullmatch(value)
            or _PATH_SHAPED.fullmatch(value)
            or _URL_SHAPED.fullmatch(value)
        )
    return (
        len(value) >= BY_CONTENT_MIN_CHARS
        and not any(character.isspace() for character in value)
        and not _names_something(value)
    )


def _repeats_its_key(key: str, value: str) -> bool:
    """Whether the value is its key's name or last word (``FAIL = "fail"``), unless that word is a common
    default password (``password = "password"`` is a credential)."""
    written = _KEY_SEPARATORS.sub("", value).lower()
    return (
        bool(written)
        and written not in _DEFAULT_PASSWORDS
        and written in (_KEY_SEPARATORS.sub("", key).lower(), _key_parts(key)[-1])
    )


def _names_something(value: str) -> bool:
    return bool(
        _NAME_SHAPED.fullmatch(value) or _PATH_SHAPED.fullmatch(value) or _URL_SHAPED.fullmatch(value)
    )


def _quoted_literal(match: re.Match[str]) -> bool:
    quote = match["quote"] if "quote" in match.re.groupindex else '"'
    return _is_literal(match["value"], quote)


def _bare_literal(match: re.Match[str]) -> bool:
    value = match["value"]
    return any(c.isalnum() for c in value) and (
        not _CODE_REFERENCE.fullmatch(value) or _looks_generated(value)
    )


def _looks_generated(value: str) -> bool:
    """A long undotted run of letters and digits (a hex key, ``whsec_`` plus 32 characters) is a value,
    even though it parses as an identifier."""
    return (
        len(value) >= RANDOM_VALUE_MIN_CHARS
        and "." not in value
        and any(c.isdigit() for c in value)
        and any(c.isalpha() for c in value)
    )


def _env_file_literal(match: re.Match[str]) -> bool:
    """A written value, not a dotted path or a usage placeholder (``KEY=...``, ``KEY=<credential>``)."""
    value = match["value"]
    return not _DOTTED_PATH.fullmatch(value) and not _USAGE_PLACEHOLDER.fullmatch(value)


def _url_password(match: re.Match[str]) -> bool:
    return _is_literal(match["value"], '"')


def _query_secret(match: re.Match[str]) -> bool:
    return _key_kind(match["key"]) == "secret" and _is_literal(match["value"], '"')


def _is_literal(value: str, quote: str) -> bool:
    """A whole ``${...}`` or ``$(...)``, or ``$NAME`` outside single quotes, refers to a variable: code."""
    if _INTERPOLATION.fullmatch(value):
        return False
    return quote == "'" or not _VARIABLE.fullmatch(value)


def _config_literal(match: re.Match[str]) -> bool:
    """An unquoted config value, unless it is empty, a boolean, or a whole reference: ``${VAR}``, ``$VAR``
    or a CI expression (``${{ secrets.TOKEN }}``)."""
    value = match["value"]
    return (
        value.lower() not in _CONFIG_NON_VALUES
        and not _CI_EXPRESSION.fullmatch(value)
        and _is_literal(value, '"')
    )


def _bearer_value(match: re.Match[str]) -> bool:
    return any(character.isdigit() for character in match["value"])


def _high_entropy_value(match: re.Match[str]) -> bool:
    return is_high_entropy(match["value"])


def is_high_entropy(value: str) -> bool:
    """Whether a value's characters are spread like a random token's: at least
    ``HIGH_ENTROPY_BITS_PER_CHAR`` bits of Shannon entropy per character."""
    counts = Counter(value)
    bits = -sum(count / len(value) * math.log2(count / len(value)) for count in counts.values())
    return bits >= HIGH_ENTROPY_BITS_PER_CHAR


_RULES = (
    _matches(_PRIVATE_KEY_BLOCK),
    _matches(_KEY_MARKER_LINE),
    *(_matches(shape) for shape in _TOKEN_SHAPES),
    _matches(_BEARER_VALUE, _bearer_value),
    _matches(_URL_PASSWORD, _url_password),
    _matches(_QUERY_VALUE, _query_secret),
    _matches(_ENV_FILE_VALUE, _keyed_value_hides(_env_file_literal)),
    _matches(_CLI_SECRET_FLAG, _keyed_value_hides(_env_file_literal)),
    _matches(_QUOTED_SECRET_VALUE, _keyed_value_hides(_quoted_literal)),
    _matches(_LITERAL_FALLBACK, _keyed_value_hides(_quoted_literal)),
    _call_literal_spans,
    _matches(_BARE_SECRET_VALUE, _keyed_value_hides(_bare_literal)),
    _matches(_QUOTED_ASSIGNMENT, _high_entropy_value),
)
_CONFIG_RULES = (*_RULES, _matches(_CONFIG_SCALAR, _keyed_value_hides(_config_literal)))
