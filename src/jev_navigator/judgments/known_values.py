"""Hide a known secret's bare copies in every request a search sends, whichever request carries the code
a rule finds the secret in.

Request masking (``secrets.mask_request``) hides a value everywhere in the one request where a rule finds
it, but a search spreads its code over many requests: ``password = "<value>"`` in one unit is hidden,
while ``dial(order, "<value>")``, judged in another request, holds nothing a rule could recognise. So
before a search sends anything, JVN's rules (``secrets.SecretMasker``) read every file of the search's
index once (``repository_values``), and the search's Judge masks with a ``KnownValuesMasker`` around its
own masker, which lists each of those values wherever a request holds a copy. Request masking then
hides the copy as it hides a copy of a value found in the request, wherever it hides those, and the
final pre-send check (``secrets.refuse_if_secret``) refuses a request still holding one. The values
depend on the files of the index's scope only, never on what a search reached or in which order, so a
request's bytes are the same in every search and run over the same scope, and code holding no copy is
sent exactly as before.

Every file of the scope is read but a binary one and one the index cannot read, which no request can
carry; an env file in the scope is read here too, though never sent. Reading fixes each file as the
index first read it from the search's start, so a file changed later counts as changed even if no
request opened it, and the index keeps every file it read, compressed, while it lives.

Only a value of ``BY_CONTENT_MIN_CHARS`` or more characters standing in at most ``KNOWN_VALUE_MAX_FILES``
files of the scope is known this way. A shorter value is hidden within its own request only, as a whole
word. A value in more files is a placeholder rather than a secret (``password: password``, a model name
a test passes as a key), and hiding it across the repository would blank ordinary code; it too is
hidden only in the requests that carry code a rule finds it in. A secret copied bare into more files
than that is therefore not hidden across requests. A known value of plain words (``description``,
``read_only``) is hidden in code like any other, but never in a point or a key of a request's
structure, where it is a word (``secrets._is_plain_word``). A host masker that hides more than JVN's
rules (``Masker.masked_values`` listing values the rules miss) hides those within each request only.

A cut that keeps a text's start (a long line, a history section, a slice's first lines) can split a
known value; the masker
also lists a value whose start stands right before the cut's mark, and request masking hides that start
(``secrets.split_starts``).
"""

from __future__ import annotations

import asyncio
import re
import threading
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..index.code_index import CodeIndex
from ..index.scope import BINARY_SNIFF_BYTES
from .judge import Judge
from .secrets import (
    BY_CONTENT_MIN_CHARS,
    DEFAULT_MASKER,
    MASK_TOKEN,
    Masker,
    ValueStarts,
    split_starts,
    value_starts,
)

KNOWN_VALUE_MAX_FILES = 5
"""A value standing in more files than this is a placeholder, not a secret (see the module docstring)."""

__all__ = [
    "KNOWN_VALUE_MAX_FILES",
    "KnownValuesMasker",
    "known_values_masker",
    "repository_values",
    "with_known_values",
    "with_known_values_async",
]


@dataclass(frozen=True)
class KnownValuesMasker:
    """``inner`` knowing ``known``: ``masked_values`` also lists each known value a text holds, so request
    masking (``secrets.mask_everywhere``) hides its copies wherever it hides a value found in the request
    and the final check refuses one left behind. ``mask`` is ``inner``'s, so JVN's own question wording,
    which request masking leaves as it is, stays byte for byte."""

    inner: Masker
    known: frozenset[str]
    _starts: ValueStarts = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_starts", value_starts(self.known))

    def mask(self, text: str, path: str | None = None) -> str:
        return self.inner.mask(text, path)

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        """The inner masker's values, each known value the text holds, and each known value whose start
        a cut kept (``secrets.split_starts``), so request masking hides that start too."""
        whole = [value for value in self.known if value in text]
        split = [owner for _, _, owners in split_starts(text, self._starts) for owner in owners]
        return [*self.inner.masked_values(text, path), *whole, *split]

    @property
    def token_pattern(self) -> re.Pattern[str]:
        return getattr(self.inner, "token_pattern", MASK_TOKEN)


def with_known_values(judge: Judge, index: CodeIndex) -> Judge:
    """A scope of ``judge`` masking with ``known_values_masker``, or ``judge`` itself when that is its
    own masker already (it masks nothing, or already knows every value ``index`` holds)."""
    masker = known_values_masker(judge.masker, index)
    if masker is judge.masker:
        return judge
    scope = judge.scope()
    scope.masker = masker
    return scope


async def with_known_values_async(judge: Judge, index: CodeIndex) -> Judge:
    """``with_known_values`` in a worker thread, so reading the repository leaves the event loop free."""
    return await asyncio.to_thread(with_known_values, judge, index)


def known_values_masker(masker: Masker | None, index: CodeIndex) -> Masker | None:
    """``masker`` knowing the values ``index`` holds (``repository_values``): ``masker`` itself when it
    is None or already knows them all, else a ``KnownValuesMasker`` around the masker it wraps."""
    if masker is None:
        return None
    wrapped = isinstance(masker, KnownValuesMasker)
    inner, known = (masker.inner, masker.known) if wrapped else (masker, frozenset())
    found = repository_values(index)
    return masker if found <= known else KnownValuesMasker(inner, known | found)


def repository_values(index: CodeIndex) -> frozenset[str]:
    """Every value JVN's rules hide in a file of ``index`` that has ``BY_CONTENT_MIN_CHARS`` or more
    characters and stands in at most ``KNOWN_VALUE_MAX_FILES`` of its files, read once per index."""
    scan = _scan_of(index)
    with scan.lock:
        if scan.values is None:
            scan.values = _values_in(index)
        return scan.values


class _Scan:
    """One index's known values, read once: a second search waits for the first's reading."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.values: frozenset[str] | None = None


_SCANS: weakref.WeakKeyDictionary[CodeIndex, _Scan] = weakref.WeakKeyDictionary()
_SCANS_LOCK = threading.Lock()


def _scan_of(index: CodeIndex) -> _Scan:
    with _SCANS_LOCK:
        return _SCANS.setdefault(index, _Scan())


def _values_in(index: CodeIndex) -> frozenset[str]:
    """The values the rules hide that are long enough, read file by file so no more than one file's
    text is held at a time, kept when ``_holders`` counts them in few enough files."""
    found: dict[str, None] = {}
    text_files: set[str] = set()
    for file, text in _texts(index):
        text_files.add(file)
        found.update(
            dict.fromkeys(
                value
                for value in DEFAULT_MASKER.masked_values(text, file)
                if len(value) >= BY_CONTENT_MIN_CHARS
            )
        )
    holders = _holders(index, list(found), text_files)
    return frozenset(value for value in found if holders[value] <= KNOWN_VALUE_MAX_FILES)


def _holders(index: CodeIndex, values: list[str], text_files: set[str]) -> dict[str, int]:
    """How many of ``text_files`` hold each value, found by the index's text search
    (``CodeIndex.search_texts``), whose ripgrep reads its patterns on standard input, never from its
    command line, which process listings and error messages show. It searches binary files too,
    which ``text_files`` leaves out. A value on one line is counted from its hits; one spanning lines
    is searched by its longest line, then confirmed in each file holding that line. The index keeps
    those searches' lines, values among them, while it lives, as it keeps the files."""
    keys = {value: max(value.split("\n"), key=len) for value in values}
    hits = index.search_texts(dict.fromkeys(keys.values())) if keys else {}
    counts = {}
    for value, key in keys.items():
        files = {hit.file for hit in hits[key]} & text_files
        counts[value] = (
            len(files) if key == value else sum(value in "\n".join(index.lines(file)) for file in files)
        )
    return counts


def _texts(index: CodeIndex) -> Iterator[tuple[str, str]]:
    """Every file of the index as the index first read it, but a binary one (a NUL among its first
    ``BINARY_SNIFF_BYTES``, as a text search decides) and one it cannot read, which no request can carry."""
    for file in index.files:
        try:
            text = "\n".join(index.lines(file))
        except OSError:
            continue
        if "\0" not in text[:BINARY_SNIFF_BYTES]:
            yield file, text
