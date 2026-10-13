"""A bounded ranked queue with a small share for existing words and source diversity.

This composes the public source, unit-resolution and ranking owners. It never judges, expands
from an answer or invents a search term. Candidate admission belongs to the caller.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..index.code_index import CodeIndex
from ..index.units import Unit, UnitReader
from ..sources import (
    CALLEES,
    CALLERS,
    FILES,
    IMPORTERS,
    IMPORTS,
    NAMED_FILES,
    NAMES,
    TEXT_NAMED_FILES,
    Seeds,
)
from .rank import RankFeatures, RankWeights, rank
from .scent import ScentIndex, words

_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_SOURCES = (NAMED_FILES, TEXT_NAMED_FILES, FILES, IMPORTS, IMPORTERS, CALLERS, CALLEES, NAMES)


@dataclass(frozen=True)
class CollectionSelection:
    ids: tuple[str, ...]
    terms: tuple[str, ...]
    routes: Mapping[str, str]


def _terms(text: str, scent: ScentIndex) -> tuple[str, ...]:
    """Eight rare repeatable surface words, preserving the literal spelling from the input."""
    counts = {}
    for token in dict.fromkeys(_TOKEN.findall(text)):
        frequencies = [len(scent.postings.get(word, {})) for word in words(token)]
        repeated = [n for n in frequencies if n >= 2]
        if repeated:
            counts[token] = min(repeated)
    return tuple(sorted(counts, key=lambda token: (counts[token], token))[:8])


def _take(queues: Sequence[Sequence[str]], count: int, held: set[str]) -> list[str]:
    """One unseen candidate from each route before a second from that route."""
    pending = [deque(queue) for queue in queues]
    selected = []
    while pending and len(selected) < count:
        remaining = []
        for queue in pending:
            while queue and queue[0] in held:
                queue.popleft()
            if queue and len(selected) < count:
                candidate = queue.popleft()
                selected.append(candidate)
                held.add(candidate)
            if queue:
                remaining.append(queue)
        pending = remaining
    return selected


def collect_diverse(
    index: CodeIndex,
    reader: UnitReader,
    seeds: Seeds,
    scent: ScentIndex,
    candidates: Sequence[str],
    units: Mapping[str, Unit],
    features: Mapping[str, RankFeatures],
    weights: RankWeights,
    *,
    text: str,
    cited_text: str,
    limit: int = 192,
    reserve: int = 32,
) -> CollectionSelection:
    """Retain the ranked prefix; reserve equal shares for known words and existing sources.

    Each route preserves the same combined ranking. Duplicates consume no reserved position.
    Exhausted routes give their unused room back to ranking. Only caller-admitted candidates can
    enter, including when a source reaches a file outside that candidate universe.
    """
    if limit < 0 or not 0 <= reserve <= limit:
        raise ValueError("Need 0 <= reserve <= limit")
    ordered = rank(candidates, features, weights)
    selected = ordered[: limit - reserve]
    held = set(selected)
    routes = dict.fromkeys(selected, "rank")
    by_file = defaultdict(list)
    for candidate in ordered:
        by_file[units[candidate].path].append(candidate)
    resolved = {}

    def members(reaches):
        result = set()
        for reach in reaches:
            key = reach.at
            file = key if isinstance(key, str) else key.file
            if file not in by_file:
                continue
            if key not in resolved:
                resolved[key] = (
                    reader.list_files([key]).units if isinstance(key, str) else reader.resolve(key)[0]
                )
            for unit in resolved[key]:
                for candidate in by_file[file]:
                    if any(
                        max(a, c) <= min(b, d)
                        for a, b in unit.ranges
                        for c, d in units[candidate].ranges
                    ):
                        result.add(candidate)
        return [candidate for candidate in ordered if candidate in result]

    terms = tuple(dict.fromkeys((*_terms(text, scent), *_terms(cited_text, scent))))
    hits = defaultdict(list)
    for reach in NAMES.reach(index, Seeds(names=terms)):
        hits[reach.seed].append(reach)
    word_queues = [members(hits[term]) for term in terms]
    word_selected = _take(word_queues, reserve // 2, held)
    selected.extend(word_selected)
    routes.update(dict.fromkeys(word_selected, "known word"))
    source_queues = [members(source.reach(index, seeds)) for source in _SOURCES]
    source_selected = _take(source_queues, reserve - reserve // 2, held)
    selected.extend(source_selected)
    routes.update(dict.fromkeys(source_selected, "source"))
    for candidate in ordered:
        if len(selected) >= limit:
            break
        if candidate not in held:
            selected.append(candidate)
            held.add(candidate)
            routes[candidate] = "rank backfill"
    return CollectionSelection(tuple(selected), terms, routes)
