"""Keep secrets out of every Jev request: mask what is found, then scan the final request and refuse on a hit.

Masking works by content: every value the masker hides anywhere in a request is hidden everywhere
in it, so a token found in an assignment is also hidden where a relation text or another item quotes
it. The built-in masker and scanner are lightweight and on by default. A host with a stronger scanner
passes its own objects; turning either off must be explicit (``masker=None`` or ``scanner=None``).
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Protocol

from ..directives.places import located_file
from ..directives.shown import CUT_MARKS
from .secret_shapes import (
    BY_CONTENT_MIN_CHARS,
    HIGH_ENTROPY_MIN_CHARS,
    MASK,
    TOKEN_CHARACTER_CLASS,
    hide_secrets,
    is_high_entropy,
    merged_spans,
)

_SHORT_NUMBER = re.compile(r"[\d.,:_+-]{1,4}")
COPY_MIN_CHARS = 4
EDGE_MAX_CHARS = 512
"""A known value up to this length also has the edges a mask cut it at hidden (see ``_KnownEdges``); a
longer one, in practice code an unclosed quote swallowed, is hidden by its copies only. A cut mark finds
the start of a value of any length (``split_starts``)."""
MASK_TOKEN = re.compile(re.escape(MASK))
_PLAIN_WORD = re.compile(r"[A-Z]?[a-z]+(?:_[a-z]+)*|[A-Z]+(?:_[A-Z]+)*")
# The role of a request string: a value, a key of the request's structure, the request's point, or
# JVN's own question wording.
VALUE, KEY, POINT, WORDING = "value", "key", "point", "wording"
TARGET = "target"
TARGETS = "targets"
WORKFLOW = "workflow"
POINT_KEYS = frozenset({TARGET, TARGETS, WORKFLOW})
"""The request's own keys that hold its point, the text Jev is asked about: a search's target, a
find-all's targets and a trace's workflow question."""
_CHOICE_QUESTION = "choice"

__all__ = [
    "BY_CONTENT_MIN_CHARS",
    "EDGE_MAX_CHARS",
    "HIGH_ENTROPY_MIN_CHARS",
    "MASK",
    "MASK_TOKEN",
    "POINT_KEYS",
    "TARGET",
    "TARGETS",
    "WORKFLOW",
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
    "split_starts",
    "value_starts",
    "ValueStarts",
]


class Masker(Protocol):
    """``mask`` hides secrets in one text; ``masked_values`` lists the values it hides there, so they
    can be hidden everywhere else in the request too. A masker that writes tokens of its own besides
    ``MASK`` names them all in a ``token_pattern`` attribute, so the rest of a known value beside any
    of them is found (``MASK_TOKEN`` when it has none)."""

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

    ``masked_values`` lists every masked value of ``COPY_MIN_CHARS`` or more characters but a short number
    (``"1234"``), so request masking hides each copy elsewhere too: anywhere for a value of
    ``BY_CONTENT_MIN_CHARS`` or more characters, as a whole word for a shorter one (see ``copy_pattern``).
    A value under ``COPY_MIN_CHARS`` (``"x"``) is too short to identify a secret and is hidden only where
    a rule finds it.

    ``path`` is the file the text comes from. In a config file, or in text from no file, an unquoted value
    under a secret key is a value (``POSTGRES_PASSWORD: example``); in code it stays (``token: str``)."""

    def mask(self, text: str, path: str | None = None) -> str:
        return hide_secrets(text, path)[0]

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return [form for value in hide_secrets(text, path)[1] for form in _copied_forms(value)]


DEFAULT_MASKER = SecretMasker()
"""What every request is masked with unless its caller names another masker."""


@dataclass(frozen=True)
class SecretScanner:
    """Reports what the built-in masker would have masked; used as the final check before sending."""

    def findings(self, text: str, path: str | None = None) -> list[str]:
        return [value[:4] for value in hide_secrets(text, path)[1]]


def mask_request(state: Mapping, questions: Mapping, masker: Masker) -> tuple[Mapping, Mapping, frozenset]:
    """The masked state and questions, and every value that was hidden in either. The request's point
    and JVN's own question wording keep their words (see ``_wording_keys``)."""
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
    """Masks every string by the masker's rules and hides each of ``values`` wherever it still appears.
    The rules read the text first, so a secret only the masker recognizes is hidden whole before a known
    value inside it could break its shape. A rule can also cut a known value short, ending it early (a
    URL password at its first ``@``) or starting it late (an email address after a ``#``); what is left
    of the value beside the mask is hidden too (``_KnownEdges``). A string the hiding changed is masked
    once more, because a hidden copy can turn a kept value into one the rules hide (``sessionToken:
    "[MASKED]-token"`` no longer repeats its key), and the request sent must be one the rules leave as
    it is. Keys are left as they are. JVN's own question wording hides no copies. The request's point
    hides copies of a value of ``BY_CONTENT_MIN_CHARS`` or more characters only, and never of a plain word
    (``_is_plain_word``): its ordinary words stay (a secret value such as ``"shared"`` or ``"described"``
    can equal one), while a long secret its writer copied from the code is hidden."""
    token = getattr(masker, "token_pattern", MASK_TOKEN)
    known = sorted(values - {MASK}, key=len, reverse=True)
    long_known = [
        secret for secret in known if len(secret) >= BY_CONTENT_MIN_CHARS and not _is_plain_word(secret)
    ]
    hiders = {
        VALUE: (_known_edges(frozenset(known), token), [copy_pattern(secret) for secret in known]),
        POINT: (_known_edges(frozenset(long_known), token), [copy_pattern(secret) for secret in long_known]),
    }

    @cache
    def hide(text: str, path: str | None, role: str) -> str:
        masked = masker.mask(text, path)
        if role not in hiders:
            return masked
        edges, copies = hiders[role]
        copied = edges.hide(_hide_copies(masked, copies))
        return masked if copied == masked else masker.mask(copied, path)

    return _each_string(value, hide, questions=questions)


@lru_cache(maxsize=16)
def _known_edges(known: frozenset[str], token: re.Pattern[str]) -> _KnownEdges:
    """One request's edges, built once although a request is masked again whenever it is split."""
    return _KnownEdges(known, token)


class _KnownEdges:
    """What a mask left of a known value it cut short: the value's rest after a mask token, from one of
    its non-alphanumeric characters on, and its start before one, up to such a character. Rules begin
    and end at such characters, so these are the shapes a cut leaves. Each edge is checked in place
    beside each token, longest first, so the work grows with the tokens and the edges, not with every
    suffix of every value."""

    def __init__(self, known: Iterable[str], token: re.Pattern[str]) -> None:
        self._token = token
        rests: defaultdict[str, set[str]] = defaultdict(set)
        starts: defaultdict[str, set[str]] = defaultdict(set)
        for secret in known:
            if len(secret) > EDGE_MAX_CHARS:
                continue
            for cut in range(1, len(secret)):
                if not secret[cut].isalnum() and _has_alnum(secret[cut:]):
                    rests[secret[cut]].add(secret[cut:])
                if not secret[cut - 1].isalnum() and _has_alnum(secret[:cut]):
                    starts[secret[cut - 1]].add(secret[:cut])
        self._rests = {first: sorted(edges, key=len, reverse=True) for first, edges in rests.items()}
        self._starts = {last: sorted(edges, key=len, reverse=True) for last, edges in starts.items()}
        self._split = value_starts(frozenset(known))

    def hide(self, text: str) -> str:
        if not self._rests and not self._starts and not self._split.heads:
            return text
        spans = [(begin, end) for begin, end, _ in split_starts(text, self._split)]
        for token in self._token.finditer(text):
            begin, end = token.span()
            before = begin and next(
                (edge for edge in self._starts.get(text[begin - 1], ()) if text.endswith(edge, 0, begin)), ""
            )
            after = end < len(text) and next(
                (edge for edge in self._rests.get(text[end], ()) if text.startswith(edge, end)), ""
            )
            if before or after:
                spans.append((begin - len(before or ""), end + len(after or "")))
        for begin, end in reversed(merged_spans(spans)):
            text = text[:begin] + MASK + text[end:]
        return text


def _has_alnum(text: str) -> bool:
    return any(character.isalnum() for character in text)


@dataclass(frozen=True)
class ValueStarts:
    """Values longer than ``COPY_MIN_CHARS`` characters, sorted, the first ``COPY_MIN_CHARS`` characters of
    each, and the longest one's length: where a cut that split one of them can begin (see
    ``split_starts``)."""

    ordered: tuple[str, ...]
    heads: frozenset[str]
    longest: int

    def owners(self, kept: str) -> tuple[str, ...]:
        """The values ``kept`` is a proper start of: a run of the sorted values, found by bisection."""
        owners = []
        for index in range(bisect_left(self.ordered, kept), len(self.ordered)):
            if not self.ordered[index].startswith(kept):
                break
            if len(self.ordered[index]) > len(kept):
                owners.append(self.ordered[index])
        return tuple(owners)


@lru_cache(maxsize=16)
def value_starts(values: frozenset[str]) -> ValueStarts:
    ordered = tuple(sorted(value for value in values if len(value) > COPY_MIN_CHARS))
    return ValueStarts(
        ordered, frozenset(value[:COPY_MIN_CHARS] for value in ordered), max(map(len, ordered), default=0)
    )


def split_starts(text: str, starts: ValueStarts) -> list[tuple[int, int, tuple[str, ...]]]:
    """Where ``text`` holds a proper start of ``COPY_MIN_CHARS`` or more characters of a value right before
    a cut mark: the longest such start there, and the values it starts. A cut that keeps a text's start (a
    long line, a history section, a slice's first lines) can split a value, and the start it keeps matches
    no copy of the whole value, so request masking hides it here. Each character before a mark is read
    once for a value's first characters, however long the values are."""
    if not starts.heads or "cut" not in text:
        return []
    marks = [mark.start() for mark in CUT_MARKS.finditer(text)]
    heads: list[int] = []
    scanned = 0
    for end in marks:
        for begin in range(max(scanned, end - starts.longest + 1), end - COPY_MIN_CHARS + 1):
            if text[begin : begin + COPY_MIN_CHARS] in starts.heads:
                heads.append(begin)
        scanned = max(scanned, end - COPY_MIN_CHARS + 1)
    found = []
    for end in marks:
        for begin in heads[
            bisect_left(heads, end - starts.longest + 1) : bisect_right(heads, end - COPY_MIN_CHARS)
        ]:
            if owners := starts.owners(text[begin:end]):
                found.append((begin, end, owners))
                break
    return found


def _hide_copies(text: str, copies: list[re.Pattern[str]]) -> str:
    text = _hide_overlapping_copies(text, copies)
    for copy in copies:
        text = copy.sub(MASK, text)
    return text


def _hide_overlapping_copies(text: str, copies: list[re.Pattern[str]]) -> str:
    """Copies that share characters, hidden one after another, would leave part of whichever came second,
    and which comes first among values of one length is a set's order. Each run of such copies is hidden
    as one mask instead; a copy inside a longer one joins it, as hiding the longer first would."""
    runs: list[list[int]] = []
    for begin, end in sorted(match.span() for copy in copies for match in copy.finditer(text)):
        if runs and begin < runs[-1][1]:
            runs[-1][1] = max(runs[-1][1], end)
            runs[-1][2] += 1
        else:
            runs.append([begin, end, 1])
    for begin, end, count in reversed(runs):
        if count > 1:
            text = text[:begin] + MASK + text[end:]
    return text


def safe_options(options: Mapping[str, str], masker: Masker | None) -> dict[str, str]:
    """Options whose key would change under masking are dropped: a secret is never offered as a choice."""
    if masker is None:
        return dict(options)
    return {key: masker.mask(text) for key, text in options.items() if masker.mask(key) == key}


def refuse_if_secret(
    state: Mapping, questions: Mapping, scanner: Scanner | None, masked: frozenset[str] = frozenset()
) -> None:
    """Refuses when a value masked elsewhere is still in the request, or when the scanner finds a secret.
    A key or the request's point counts only for a value of ``BY_CONTENT_MIN_CHARS`` or more characters (a
    short value such as ``"false"`` equals JVN's own keys and ordinary words), and JVN's own question
    wording, which keeps its words, never counts."""
    texts = _strings(state) + _strings(questions, questions=True)
    copies = [(value, copy_pattern(value)) for value in masked - {MASK}]
    if any(_holds_copy(text, value, copy) for text in texts for value, copy in copies):
        raise SecretInRequestError(
            "a masked value is still in the request, in a key or its point; nothing was sent"
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


@dataclass(frozen=True)
class _RequestText:
    """A string of a request, the file it comes from, and its role: a "value", a "key" of the request's
    structure, the request's "point", or JVN's own question "wording"."""

    text: str
    path: str | None
    role: str


def _holds_copy(text: _RequestText, value: str, copy: re.Pattern[str]) -> bool:
    if text.role == WORDING or (
        text.role in (KEY, POINT) and (len(value) < BY_CONTENT_MIN_CHARS or _is_plain_word(value))
    ):
        return False
    return bool(copy.search(text.text))


def _is_plain_word(value: str) -> bool:
    """A value of plain words (``description``, ``read_only``, ``Described``, ``READ_ONLY``): as a key of
    the request's structure or in its point it is a word, not a copied secret (``instructions``), so
    hiding it there would rewrite the point or refuse every request whose structure names it. Its copies
    in code are hidden like any value's. Mixed case, as in a generated token, or a digit is not plain
    words."""
    return bool(_PLAIN_WORD.fullmatch(value))


def _copied_forms(value: str) -> list[str]:
    """A masked value and, for a value wrapped in quotes (a shell word such as ``"pa55word"``), the value
    between them, which is the same secret where it stands unquoted (the bare value in a command or URL)."""
    inner = value[1:-1] if len(value) > 2 and value[0] == value[-1] and value[0] in "\"'" else None
    return [form for form in (value, inner) if form is not None and _is_copied(form)]


def _is_copied(value: str) -> bool:
    """A masked value is hidden everywhere else too, unless it is shorter than ``COPY_MIN_CHARS``, a number
    of at most four characters, or holds no letter or digit (``"<"``)."""
    return (
        len(value) >= COPY_MIN_CHARS
        and any(character.isalnum() for character in value)
        and not _SHORT_NUMBER.fullmatch(value)
    )


def _wording_role(questions: bool) -> str:
    """Wording keeps its words: in a question it is JVN's own text, in the state the request's point."""
    return WORDING if questions else POINT


def _wording_keys(mapping: Mapping, questions: bool, root: bool) -> frozenset[str]:
    """The keys that hold wording: in the state, its own point (``POINT_KEYS`` at the top level, never a
    key inside an item); in a question, JVN's own text written from code, its instructions, and its
    criteria unless it is a choice, whose criteria are the options code supplies (signatures)."""
    if not questions:
        return POINT_KEYS & mapping.keys() if root else frozenset()
    if "instructions" not in mapping:
        return frozenset()
    return frozenset(
        {"instructions"} if mapping.get("type") == _CHOICE_QUESTION else {"instructions", "criteria"}
    )


def _each_string(
    value: object,
    change: Callable[[str, str | None, str], str],
    path: str | None = None,
    role: str = VALUE,
    questions: bool = False,
    root: bool = True,
) -> object:
    """Changes every string, each with the file it comes from (see ``_file_of``) and its role."""
    if isinstance(value, str):
        return change(value, path, role)
    if isinstance(value, Mapping):
        inner, wording = _file_of(value, path), _wording_keys(value, questions, root)
        return {
            key: _each_string(
                item, change, inner, _wording_role(questions) if key in wording else role, questions, False
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_each_string(item, change, path, role, questions, False) for item in value]
    return value


def _strings(
    value: object, path: str | None = None, role: str = VALUE, questions: bool = False, root: bool = True
) -> list[_RequestText]:
    """Every string, keys included, with the file it comes from and its role."""
    if isinstance(value, str):
        return [_RequestText(value, path, role)]
    if isinstance(value, Mapping):
        inner, wording = _file_of(value, path), _wording_keys(value, questions, root)
        return [
            text
            for key, item in value.items()
            for text in [
                _RequestText(str(key), None, KEY),
                *_strings(
                    item, inner, _wording_role(questions) if key in wording else role, questions, False
                ),
            ]
        ]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _strings(item, path, role, questions, False)]
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
