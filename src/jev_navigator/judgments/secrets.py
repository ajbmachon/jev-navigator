"""Keep secrets out of every Jev request: mask what is found, then scan the final request and refuse on a hit.

Masking works by content: every value the masker hides anywhere in a request is hidden everywhere
in it, so a token found in an assignment is also hidden where a relation text or another item quotes
it. The built-in masker and scanner are lightweight and on by default. A host with a stronger scanner
passes its own objects; turning either off must be explicit (``masker=None`` or ``scanner=None``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Protocol

from ..directives.place_signatures import located_file
from .secret_shapes import (
    BY_CONTENT_MIN_CHARS,
    HIGH_ENTROPY_MIN_CHARS,
    MASK,
    TOKEN_CHARACTER_CLASS,
    hide_secrets,
    is_high_entropy,
)

_SHORT_NUMBER = re.compile(r"[\d.,:_+-]{1,4}")
_CHOICE_QUESTION = "choice"

__all__ = [
    "BY_CONTENT_MIN_CHARS",
    "HIGH_ENTROPY_MIN_CHARS",
    "MASK",
    "TOKEN_CHARACTER_CLASS",
    "Masker",
    "Scanner",
    "SecretInRequestError",
    "SecretMasker",
    "SecretScanner",
    "is_high_entropy",
    "mask_by_content",
    "mask_everywhere",
    "mask_request",
    "masked_values",
    "refuse_if_secret",
    "safe_options",
]


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
    as a whole word for a shorter one (see ``copy_pattern``).

    ``path`` is the file the text comes from. In a config file, or in text from no file, an unquoted value
    under a secret key is a value (``POSTGRES_PASSWORD: example``); in code it stays (``token: str``)."""

    def mask(self, text: str, path: str | None = None) -> str:
        return hide_secrets(text, path)[0]

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return [value for value in hide_secrets(text, path)[1] if _is_copied(value)]


DEFAULT_MASKER = SecretMasker()
"""What every request is masked with unless its caller names another masker."""


@dataclass(frozen=True)
class SecretScanner:
    """Reports what the built-in masker would have masked; used as the final check before sending."""

    def findings(self, text: str, path: str | None = None) -> list[str]:
        return [value[:4] for value in hide_secrets(text, path)[1]]


def mask_request(state: Mapping, questions: Mapping, masker: Masker) -> tuple[Mapping, Mapping, frozenset]:
    """The masked state and questions, and every value that was hidden in either. JVN's own question
    wording keeps its words (see ``_wording_keys``)."""
    values = masked_values([state, questions], masker)
    return (
        mask_everywhere(state, masker, values),
        mask_everywhere(questions, masker, values, questions=True),
        values,
    )


def mask_by_content(value: object, masker: Masker) -> object:
    """Masks nested JSON-like data so that a value hidden in one string is hidden in all of them."""
    return mask_everywhere(value, masker, masked_values(value, masker))


def masked_values(value: object, masker: Masker) -> frozenset[str]:
    """Every value the masker hides anywhere inside nested JSON-like data, keys included."""
    return frozenset(
        found
        for text in dict.fromkeys(_strings(value))
        for found in masker.masked_values(text.text, text.path)
    )


def mask_everywhere(value: object, masker: Masker, values: frozenset[str], questions: bool = False) -> object:
    """Masks every string by the masker's rules, then hides each of ``values`` wherever it still
    appears. Keys are left as they are. In ``questions``, JVN's own wording hides no copies."""
    copies = copies_of(values)

    @cache
    def hide(text: str, path: str | None, role: str) -> str:
        text = masker.mask(text, path)
        return text if role == "wording" else copies.sub(text)

    return _each_string(value, hide, questions=questions)


def safe_options(options: Mapping[str, str], masker: Masker | None) -> dict[str, str]:
    """Options whose key would change under masking are dropped: a secret is never offered as a choice."""
    if masker is None:
        return dict(options)
    return {key: masker.mask(text) for key, text in options.items() if masker.mask(key) == key}


def refuse_if_secret(
    state: Mapping, questions: Mapping, scanner: Scanner | None, masked: frozenset[str] = frozenset()
) -> None:
    """Refuses when a value masked elsewhere is still in the request, or when the scanner finds a secret.
    A key counts only for a value of ``BY_CONTENT_MIN_CHARS`` or more characters (a short value such as
    ``"false"`` equals JVN's own keys), and question wording, which keeps its words, never counts."""
    texts = _strings(state) + _strings(questions, questions=True)
    copies = copies_of(masked)
    holding = next(
        (text for text in texts if any(_holds_copy(text, value) for value in copies.found(text.text))), None
    )
    if holding is not None:
        raise SecretInRequestError(
            f"a masked value is still in the request, in a {holding.role}; nothing was sent"
        )
    if scanner is None:
        return
    for text in texts:
        if scanner.findings(text.text, text.path):
            raise SecretInRequestError("the final scan found a secret in the request; nothing was sent")


def copy_pattern(value: str) -> re.Pattern[str]:
    """Where a masked value's copies stand: anywhere for a value of ``BY_CONTENT_MIN_CHARS`` or more
    characters, and as a whole word for a shorter one, so ``hunter2`` is hidden but ``hunter2x`` stays."""
    if len(value) >= BY_CONTENT_MIN_CHARS:
        return re.compile(re.escape(value))
    return re.compile(rf"(?<![\w$]){re.escape(value)}(?![\w$])")


class Copies:
    """Where any of ``values`` stands in a text, found in one pass however many values there are: a
    value of ``BY_CONTENT_MIN_CHARS`` or more characters is looked up by its first that many
    characters, a shorter one by itself at a word start, and every candidate is confirmed by
    ``copy_pattern``, which alone decides where a copy stands. Overlapping copies yield the leftmost,
    then the longest. A value found inside ``MASK`` itself is left out: masking its copy would
    make a new one."""

    def __init__(self, values: Iterable[str]) -> None:
        kept = sorted({value for value in values if value and value not in MASK}, key=len, reverse=True)
        self._patterns: dict[str, re.Pattern[str]] = {}
        self._long: dict[str, list[str]] = {}
        self._short: dict[int, set[str]] = {}
        for value in kept:
            if len(value) >= BY_CONTENT_MIN_CHARS:
                self._long.setdefault(value[:BY_CONTENT_MIN_CHARS], []).append(value)
            else:
                self._short.setdefault(len(value), set()).add(value)
        self._short_lengths = sorted(self._short, reverse=True)

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(start, end) for start, end, _ in self._copies(text)]

    def found(self, text: str) -> list[str]:
        return [value for _, _, value in self._copies(text)]

    def sub(self, text: str, keep_lines: bool = False) -> str:
        """``text`` with every copy replaced by ``MASK``, again until none is left: masking one copy
        can leave a short value glued after it at a word start. With ``keep_lines`` each line break a
        copy covered stays after its mask, so the text keeps its line count."""
        while copies := self.spans(text):
            text = _copies_replaced(text, copies, keep_lines)
        return text

    def _copies(self, text: str) -> list[tuple[int, int, str]]:
        word_starts = self._word_starts(text)
        copies, position = [], 0
        while position < len(text):
            value = self._copy_at(text, position, position in word_starts)
            if value is None:
                position += 1
                continue
            copies.append((position, position + len(value), value))
            position += len(value)
        return copies

    def _word_starts(self, text: str) -> frozenset[int]:
        """Where a short value may begin: no word character or ``$`` stands right before."""
        if not self._short:
            return frozenset()
        return frozenset(match.start() for match in _WORD_START.finditer(text))

    def _copy_at(self, text: str, position: int, word_start: bool) -> str | None:
        candidates = self._long.get(text[position : position + BY_CONTENT_MIN_CHARS], ())
        if word_start:
            candidates = [*candidates, *self._short_candidates(text, position)]
        if not candidates:
            return None
        matched = [value for value in candidates if self._pattern(value).match(text, position)]
        return max(matched, key=len, default=None)

    def _pattern(self, value: str) -> re.Pattern[str]:
        if value not in self._patterns:
            self._patterns[value] = copy_pattern(value)
        return self._patterns[value]

    def _short_candidates(self, text: str, position: int) -> list[str]:
        return [
            text[position : position + length]
            for length in self._short_lengths
            if text[position : position + length] in self._short[length]
        ]


def _copies_replaced(text: str, spans: list[tuple[int, int]], keep_lines: bool) -> str:
    parts, position = [], 0
    for start, end in spans:
        line_breaks = "\n" * text.count("\n", start, end) if keep_lines else ""
        parts += [text[position:start], MASK + line_breaks]
        position = end
    return "".join([*parts, text[position:]])


@lru_cache(maxsize=16)
def _copies_of(values: frozenset[str]) -> Copies:
    return Copies(values)


def copies_of(values: frozenset[str]) -> Copies:
    """The one-pass ``Copies`` of ``values``, built once for each set a process masks with."""
    return _copies_of(frozenset(values))


_WORD_START = re.compile(r"(?<![\w$])(?=.)", re.DOTALL)


@dataclass(frozen=True)
class _RequestText:
    """A string of a request, the file it comes from, and its role: a "value", a "key" of the request's
    structure, or question "wording"."""

    text: str
    path: str | None
    role: str


def _holds_copy(text: _RequestText, value: str) -> bool:
    return not (text.role == "wording" or (text.role == "key" and len(value) < BY_CONTENT_MIN_CHARS))


def _is_copied(value: str) -> bool:
    """A masked value is hidden everywhere else too, unless it is a number of at most four characters or
    holds no letter or digit (``"<"``)."""
    return any(character.isalnum() for character in value) and not _SHORT_NUMBER.fullmatch(value)


def _wording_keys(mapping: Mapping, questions: bool) -> frozenset[str]:
    """The keys of a question that hold JVN's own wording, written from code: its instructions, and its
    criteria unless it is a choice, whose criteria are the options code supplies (signatures)."""
    if not questions or "instructions" not in mapping:
        return frozenset()
    return frozenset(
        {"instructions"} if mapping.get("type") == _CHOICE_QUESTION else {"instructions", "criteria"}
    )


def _each_string(
    value: object,
    change: Callable[[str, str | None, str], str],
    path: str | None = None,
    role: str = "value",
    questions: bool = False,
) -> object:
    """Changes every string, each with the file it comes from (see ``_file_of``) and its role."""
    if isinstance(value, str):
        return change(value, path, role)
    if isinstance(value, Mapping):
        inner, wording = _file_of(value, path), _wording_keys(value, questions)
        return {
            key: _each_string(item, change, inner, "wording" if key in wording else role, questions)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_each_string(item, change, path, role, questions) for item in value]
    return value


def _strings(
    value: object, path: str | None = None, role: str = "value", questions: bool = False
) -> list[_RequestText]:
    """Every string, keys included, with the file it comes from and its role."""
    if isinstance(value, str):
        return [_RequestText(value, path, role)]
    if isinstance(value, Mapping):
        inner, wording = _file_of(value, path), _wording_keys(value, questions)
        return [
            text
            for key, item in value.items()
            for text in [
                _RequestText(str(key), None, "key"),
                *_strings(item, inner, "wording" if key in wording else role, questions),
            ]
        ]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _strings(item, path, role, questions)]
    return []


def _file_of(mapping: Mapping, outer: str | None) -> str | None:
    """The file a mapping's strings come from: its ``file`` (a slice's code), or the file its
    ``signature`` names (a candidate's preview), else the enclosing mapping's."""
    file = mapping.get("file")
    if isinstance(file, str):
        return file
    signature = mapping.get("signature")
    named = located_file(signature) if isinstance(signature, str) else None
    return named or outer
