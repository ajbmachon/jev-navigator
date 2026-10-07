"""Selective-focus measurements using case 1's scent owner and JVN structure.

This is an offline study adapter, not a library API. Corpus construction reuses
the persisted case-1 bindings, its canonicalization owner and its ScentIndex.
Deciding labels are never inputs to focus ranking or selection.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import excerpt_rules

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import Unit, read_ranges

CASE1 = Path.home() / "Projects/jev-navigator-case1"
RULES = tuple(
    [f"R{k}" for k in (3, 5, 8)]
    + [f"S{k}" for k in (3, 5, 8)]
    + [f"C{k}_{cap}" for k in (3, 5, 8) for cap in (20, 40, 80)]
    + [f"W{k}" for k in (3, 5, 8)]
)


@lru_cache(maxsize=1)
def owners():
    """Load the existing sibling implementation without copying its algorithms."""
    import jev_navigator

    sibling = str(CASE1 / "src/jev_navigator")
    if sibling not in jev_navigator.__path__:
        jev_navigator.__path__.append(sibling)
    scent = importlib.import_module("jev_navigator.selection.scent")
    assert Path(scent.__file__).resolve() == CASE1 / "src/jev_navigator/selection/scent.py"
    path = CASE1 / "measurements/selection/collect.py"
    spec = importlib.util.spec_from_file_location("excerpt_case1_collect", path)
    collect = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = collect
    spec.loader.exec_module(collect)
    return scent, collect


def repository_scent(root, catalogue):
    """Reconstruct the original rarity corpus from already-bound source units."""
    scent, collect = owners()
    index = CodeIndex.from_git(Path(root))
    documents, units = [], {}
    with sqlite3.connect(f"file:{catalogue}?mode=ro", uri=True) as db:
        for key, path, code, binding in db.execute(
            "select id,path,code,binding from documents where root=?", (str(root),)
        ):
            bound = None if binding is None else json.loads(binding)
            if bound is None:
                documents.append(scent.scent_document(key, path, "", code))
            else:
                bound["ranges"] = tuple(tuple(r) for r in bound["ranges"])
                units[key] = Unit(**bound)
    canonical, _ = collect.canonical_source_units(units)
    documents.extend(
        scent.scent_document(
            key, unit.path, unit.symbol, read_ranges(index, unit.path, unit.ranges), test=unit.test
        )
        for key, unit in canonical.items()
    )
    return scent.ScentIndex(documents)


def load_case_inputs(proof, case_ids):
    """Read original finder point citations, never the reference/gold citation fields."""
    requested = set(case_ids)
    queries, citations, paths = {}, {}, []
    for population, ids in (
        ("dev110", {case for case in requested if case.startswith("analysis-engine:")}),
        ("hard27", {case for case in requested if not case.startswith("analysis-engine:")}),
    ):
        if not ids:
            continue
        assert population == "dev110" or ids <= {
            *(f"P{i}" for i in range(1, 8)),
            *(f"U{i}" for i in range(1, 7)),
        }
        path = Path(proof) / f"runs/pack-49b78955/pack-inputs-{population}.json"
        rows = json.loads(path.read_text())["cases"]
        paths.append(path)
        for case in sorted(ids):
            claim = rows[case]["claim"]
            queries[case] = claim["statement"]
            citations[case] = {}
            for item in claim.get("evidence", []):
                line = item.get("line")
                if isinstance(line, int) and not isinstance(line, bool) and line > 0:
                    citations[case].setdefault(item["file"], []).append([line, line])
    assert queries.keys() == requested
    return queries, citations, paths


def rank_terms(query, index):
    """Ascending DF is descending IDF within one fixed corpus; ties follow mention order."""
    scent, _ = owners()
    ordered = dict.fromkeys(scent.words(query))
    rows = [
        {"term": term, "df": len(index.postings.get(term, {})), "mention": mention}
        for mention, term in enumerate(ordered)
    ]
    return sorted((row for row in rows if row["df"]), key=lambda r: (r["df"], r["mention"])), [
        row["term"] for row in rows if not row["df"]
    ]


def scope_at(line, facts):
    return min(
        (tuple(f["range"]) for f in facts.get("functions", []) if f["range"][0] <= line <= f["range"][1]),
        key=lambda r: r[1] - r[0],
        default=None,
    )


@lru_cache(maxsize=65536)
def line_words(text):
    return frozenset(owners()[0].words(text))


def select_lines(source, facts, terms, citations, rule):
    """Select local structure, then optionally cap each innermost function."""
    visible = set(source)
    headers = excerpt_rules.selected_lines(source, facts, "", "A")
    term_rank = {term: rank for rank, term in enumerate(terms)}
    ranked_hits = {
        n: min(term_rank[word] for word in line_words(text) & term_rank.keys())
        for n, text in source.items()
        if line_words(text) & term_rank.keys()
    }
    cited = {n for a, b in citations for n in range(a, b + 1) if n in visible}
    focus = cited | ranked_hits.keys()
    selected = headers | focus
    if rule.startswith("W"):
        for line in focus:
            excerpt_rules._add(selected, [line - 40, line + 48], visible)
    elif rule.startswith(("S", "C")):
        for line in sorted(focus):
            exits = [
                e
                for e in facts.get("exits", [])
                if e["kind"] in {"return_statement", "raise_statement", "throw_statement"}
                and visible.intersection(range(e["range"][0], e["range"][1] + 1))
                and scope_at(e["range"][0], facts) == scope_at(line, facts)
            ]
            if exits:
                nearest = min(
                    exits,
                    key=lambda e: (
                        min(abs(line - e["range"][0]), abs(line - e["range"][1])),
                        e["range"][0] < line,
                        e["range"][0],
                    ),
                )
                excerpt_rules._add(selected, nearest["range"], visible)
        for call in facts.get("calls", []):
            if focus.intersection(range(call["range"][0], call["range"][1] + 1)):
                excerpt_rules._add(selected, call["range"], visible)
        excerpt_rules._condition_closure(selected, facts, visible)
    if rule.startswith("C"):
        cap = int(rule.split("_")[1])
        grouped = {}
        for line in selected:
            grouped.setdefault(scope_at(line, facts), []).append(line)

        def priority(line):
            category = 0 if line in headers else 1 if line in cited else 2 if line in ranked_hits else 3
            return (
                category,
                ranked_hits.get(line, -1),
                min((abs(line - f) for f in focus), default=0),
                line,
            )

        selected = {line for lines in grouped.values() for line in sorted(lines, key=priority)[:cap]}
    return selected, {"rare_hits": excerpt_rules.ranges(ranked_hits), "cited": excerpt_rules.ranges(cited)}


class Selector:
    """One immutable query/citation context per original finding and repository."""

    def __init__(self, index, queries_by_case, citations_by_case):
        self.terms = {case: rank_terms(query, index) for case, query in queries_by_case.items()}
        self.citations = citations_by_case
        self.documents = len(index.documents)

    def render(self, source, facts, case_id, rule, *, file):
        assert rule in RULES
        k = int(rule.split("_")[0][1:])
        terms = tuple(row["term"] for row in self.terms[case_id][0][:k])
        selected, focus = select_lines(
            source, facts, terms, self.citations.get(case_id, {}).get(file, []), rule
        )
        return {**excerpt_rules.render_selection(source, selected), "focus": focus, "terms": terms}
