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
import os
import re
import sqlite3
import threading
import weakref
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

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


class _SavedPostings:
    """One table of a saved index, read a term at a time, so its postings cost memory only while a query
    holds the ones it matched. Each thread reads on its own connection."""

    def __init__(self, path: Path, table: str, connections: threading.local) -> None:
        self._path, self._connections = path, connections
        self._query = f"SELECT numbers FROM {table} WHERE term = ?"

    def get(self, term: str, default: tuple[()] = ()) -> array[int] | tuple[()]:
        connection = getattr(self._connections, "connection", None)
        if connection is None:
            connection = self._connections.connection = sqlite3.connect(
                f"file:{quote(str(self._path))}?mode=ro&immutable=1", uri=True
            )
        row = connection.execute(self._query, (term,)).fetchone()
        if row is None:
            return default
        numbers = array("I")
        numbers.frombytes(row[0])
        return numbers


class ScentIndex:
    """An inverted index of compact counts, built from ``documents`` one at a time and keeping none of
    them. Each term's postings are one flat array of document numbers and counts. Query cost follows
    matched terms, not source bytes. ``save`` and ``load`` keep the postings in a file."""

    def __init__(self, documents: Iterable[ScentDocument], *, k1: float = 1.2, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 > 0 and b between zero and one")
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tests = bytearray()
        self.lengths = array("I")
        self.postings: dict[str, array[int]] | _SavedPostings = {}
        self.filenames: dict[str, array[int]] | _SavedPostings = {}
        for document in documents:
            number = len(self.ids)
            self.ids.append(document.id)
            self.tests.append(document.test)
            self.lengths.append(sum(document.terms.values()))
            for term, count in document.terms.items():
                self.postings.setdefault(term, array("I")).extend((number, count))
            for term, count in document.filename_terms.items():
                self.filenames.setdefault(term, array("I")).extend((number, count))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def save(self, path: Path) -> None:
        """Write this index to ``path``, which appears only once it is complete."""
        partial = path.with_name(path.name + ".partial")
        partial.unlink(missing_ok=True)
        with closing(sqlite3.connect(partial)) as database, database:
            database.execute("PRAGMA journal_mode = OFF")
            database.execute("CREATE TABLE settings (k1 REAL, b REAL)")
            database.execute("INSERT INTO settings VALUES (?, ?)", (self.k1, self.b))
            database.execute(
                "CREATE TABLE documents (number INTEGER PRIMARY KEY, id TEXT, test INTEGER, length INTEGER)"
            )
            database.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                (
                    (number, *row)
                    for number, row in enumerate(zip(self.ids, self.tests, self.lengths, strict=True))
                ),
            )
            for table, postings in (("postings", self.postings), ("filenames", self.filenames)):
                database.execute(f"CREATE TABLE {table} (term TEXT PRIMARY KEY, numbers BLOB) WITHOUT ROWID")
                database.executemany(
                    f"INSERT INTO {table} VALUES (?, ?)",
                    ((term, postings[term].tobytes()) for term in sorted(postings)),
                )
        os.replace(partial, path)

    @classmethod
    def load(cls, path: Path) -> ScentIndex:
        """The index ``save`` wrote to ``path``, answering as the saved one did. Its documents are in
        memory; its postings stay in the file, which must outlive the index."""
        index = cls.__new__(cls)
        with closing(sqlite3.connect(f"file:{quote(str(path))}?mode=ro&immutable=1", uri=True)) as database:
            index.k1, index.b = database.execute("SELECT k1, b FROM settings").fetchone()
            rows = database.execute("SELECT id, test, length FROM documents ORDER BY number").fetchall()
        index.ids = [row[0] for row in rows]
        index.tests = bytearray(row[1] for row in rows)
        index.lengths = array("I", (row[2] for row in rows))
        index.average_length = sum(index.lengths) / max(1, len(index.lengths))
        connections = threading.local()
        index.postings = _SavedPostings(path, "postings", connections)
        index.filenames = _SavedPostings(path, "filenames", connections)
        return index

    def signals(self, text: str) -> tuple[dict[str, float], dict[str, float]]:
        """BM25 and IDF-weighted file-name matches. Absent matches have score zero."""
        scent: dict[str, float] = {}
        filename: dict[str, float] = {}
        count = len(self.ids)
        for term in set(words(text)):
            matches = self.postings.get(term, ())
            found = len(matches) // 2
            idf = math.log1p((count - found + 0.5) / (found + 0.5))
            for number, frequency in zip(matches[::2], matches[1::2], strict=True):
                norm = 1 - self.b + self.b * self.lengths[number] / (self.average_length or 1)
                score = idf * frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
                id = self.ids[number]
                scent[id] = scent.get(id, 0) + score
            for number in self.filenames.get(term, ())[::2]:
                id = self.ids[number]
                filename[id] = filename.get(id, 0) + idf
        return scent, filename

    def scores(self, text: str, *, filename_weight: float = 2.0) -> dict[str, float]:
        scent, filename = self.signals(text)
        return {id: scent.get(id, 0) + filename_weight * filename.get(id, 0) for id in self.ids}


class ScentIndexTooLargeError(ValueError):
    """A CodeIndex lists more units than its scent index may hold."""


@dataclass(frozen=True)
class UnitScent:
    """BM25 over a CodeIndex's units, one document per whole unit keyed by unit id, with the line each
    unit starts at."""

    index: ScentIndex
    starts: Mapping[str, LineAnchor]

    def ranked(
        self, text: str, limit: int, in_scope: Callable[[str], bool] | None = None
    ) -> list[tuple[LineAnchor, float]]:
        """The first line of the ``limit`` units scoring above zero for ``text``, best first, each with
        its score, counting only units whose file ``in_scope`` keeps; a tie goes to the smaller unit id."""
        scored = (
            (-score, unit_id)
            for unit_id, score in self.index.scores(text).items()
            if score > 0 and (in_scope is None or in_scope(self.starts[unit_id].file))
        )
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
