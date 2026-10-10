"""BM25 over identifiers, strings and paths, with a separate file-name signal.

Documents hold compact word counts, never parser output or source text. Construct them one unit at
a time. The existing spelling map owns case/separator/part expansion; plural folding is local to
retrieval and does not invent new source-search terms.

``unit_scent`` holds one index over every unit a ``CodeIndex`` lists, one document per whole unit,
built once per CodeIndex and bounded: building refuses more than ``max_units`` units, and the index
lives no longer than the CodeIndex it was built from.
"""

from __future__ import annotations

import heapq
import math
import re
import threading
import weakref
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..index.code_index import CodeIndex
from ..index.imports import without_comments
from ..index.languages import split_lines
from ..index.units import LineAnchor, Unit, list_units
from ..mentions import spelling_variants

DEFAULT_MAX_UNITS = 25_000
"""The most units ``unit_scent`` indexes for one CodeIndex; a larger repository is refused, never
indexed in part."""
_WHOLE_UNITS = 2**31 - 1
"""Room per unit large enough that the listing cuts no unit: a document is a whole unit."""

_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    ]
)


def words(text: str) -> tuple[str, ...]:
    """Case, separators, acronym/camel parts and regular plurals, in occurrence order."""
    result = []
    for token in _TOKEN.findall(text):
        variants = spelling_variants(_ACRONYM.sub(r"\1_\2", token))
        expanded = dict.fromkeys(part.casefold() for variant in variants for part in variant.split("_"))
        for part in expanded:
            if len(part) > 2 and part not in _STOP:
                result.append(_singular(part))
    return tuple(result)


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


@dataclass(frozen=True)
class ScentDocument:
    id: str
    path: str
    terms: Mapping[str, int]
    filename_terms: Mapping[str, int]
    test: bool = False


def scent_document(id: str, path: str, symbol: str, source: str, *, test: bool = False) -> ScentDocument:
    """Names and string contents of one unit; comments do not supply lexical scent."""
    return uncommented_scent_document(id, path, symbol, without_comments(source, path), test=test)


def uncommented_scent_document(
    id: str, path: str, symbol: str, code: str, *, test: bool = False
) -> ScentDocument:
    """``scent_document`` for ``code`` whose comments are already removed, such as a unit's lines of
    ``CodeIndex.source_without_comments``, so no parser runs per unit."""
    return ScentDocument(
        id,
        path,
        Counter(words(symbol + " " + code + " " + path)),
        Counter(words(PurePosixPath(path).stem)),
        test,
    )


class ScentIndex:
    """An inverted index of compact counts. Query cost follows matched terms, not source bytes."""

    def __init__(self, documents: Iterable[ScentDocument], *, k1: float = 1.2, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 > 0 and b between zero and one")
        self.documents = {document.id: document for document in documents}
        self.k1, self.b = k1, b
        self.lengths = {id: sum(doc.terms.values()) for id, doc in self.documents.items()}
        self.average_length = sum(self.lengths.values()) / max(1, len(self.lengths))
        self.postings: dict[str, dict[str, int]] = {}
        self.filenames: dict[str, dict[str, int]] = {}
        for id, document in self.documents.items():
            for term, count in document.terms.items():
                self.postings.setdefault(term, {})[id] = count
            for term, count in document.filename_terms.items():
                self.filenames.setdefault(term, {})[id] = count

    def signals(self, text: str) -> tuple[dict[str, float], dict[str, float]]:
        """BM25 and IDF-weighted file-name matches. Absent matches have score zero."""
        scent: dict[str, float] = {}
        filename: dict[str, float] = {}
        count = len(self.documents)
        for term in set(words(text)):
            matches = self.postings.get(term, {})
            idf = math.log1p((count - len(matches) + 0.5) / (len(matches) + 0.5))
            for id, frequency in matches.items():
                norm = 1 - self.b + self.b * self.lengths[id] / (self.average_length or 1)
                score = idf * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
                scent[id] = scent.get(id, 0) + score
            for id in self.filenames.get(term, {}):
                filename[id] = filename.get(id, 0) + idf
        return scent, filename

    def scores(self, text: str, *, filename_weight: float = 2.0) -> dict[str, float]:
        scent, filename = self.signals(text)
        return {id: scent.get(id, 0) + filename_weight * filename.get(id, 0) for id in self.documents}


class ScentIndexTooLargeError(ValueError):
    """A CodeIndex lists more units than its scent index may hold."""


@dataclass(frozen=True)
class UnitScent:
    """BM25 over a CodeIndex's units, one document per whole unit keyed by unit id, with the line each
    unit starts at."""

    index: ScentIndex
    starts: Mapping[str, LineAnchor]

    def ranked(self, text: str, limit: int) -> list[tuple[LineAnchor, float]]:
        """The first line of the ``limit`` units scoring above zero for ``text``, best first, each with
        its score; a tie goes to the smaller unit id."""
        scored = ((-score, unit_id) for unit_id, score in self.index.scores(text).items() if score > 0)
        return [(self.starts[unit_id], -negative) for negative, unit_id in heapq.nsmallest(limit, scored)]


def unit_scent(index: CodeIndex, *, max_units: int = DEFAULT_MAX_UNITS) -> UnitScent:
    """The scent index of every unit ``index`` lists, built on first use and kept while ``index``
    lives. Building it raises ``ScentIndexTooLargeError`` when ``index`` lists more than ``max_units``
    units, before any document is made; an index already built is returned as it is."""
    with _BUILDING:
        scent = _BUILT.get(index)
        if scent is None:
            scent = _BUILT[index] = _built(index, max_units)
        return scent


_BUILT: weakref.WeakKeyDictionary[CodeIndex, UnitScent] = weakref.WeakKeyDictionary()
_BUILDING = threading.Lock()


def _built(index: CodeIndex, max_units: int) -> UnitScent:
    units = list_units(index, index.files, box_chars=_WHOLE_UNITS).units
    if len(units) > max_units:
        raise ScentIndexTooLargeError(
            f"the repository lists {len(units):,} units, more than the scent index's bound of "
            f"{max_units:,}; narrow the CodeIndex's scope or raise max_units"
        )
    starts = {unit.id: LineAnchor(unit.path, unit.start) for unit in units}
    return UnitScent(ScentIndex(_documents(index, units)), starts)


def _documents(index: CodeIndex, units: Sequence[Unit]) -> Iterator[ScentDocument]:
    """One document per unit, from its lines of its file's comment-free source; units come in file
    order, so only one file's lines are held at a time."""
    file, lines = "", ()
    for unit in units:
        if unit.path != file:
            file, lines = unit.path, split_lines(index.source_without_comments(unit.path))
        code = "\n".join("\n".join(lines[start - 1 : end]) for start, end in unit.ranges)
        yield uncommented_scent_document(unit.id, unit.path, unit.symbol, code, test=unit.test)
