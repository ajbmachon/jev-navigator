"""BM25 over identifiers, strings and paths, with a separate file-name signal.

Documents hold compact word counts, never parser output or source text. Construct them one unit at
a time. The existing spelling map owns case/separator/part expansion; plural folding is local to
retrieval and does not invent new source-search terms.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..index.imports import without_comments
from ..mentions import spelling_variants

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
    code = without_comments(source, path)
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
