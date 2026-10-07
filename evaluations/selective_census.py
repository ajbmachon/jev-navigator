"""Offline selective-focus survival on the original 229 deciding-line instances."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import excerpt_rules
import selective_rules
from census import PLANNING, PROOF, deny_network

OUT = PLANNING / "search-design/excerpts/selective"


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    sys.addaudithook(deny_network)
    OUT.mkdir(parents=True, exist_ok=True)
    prior = OUT.parent
    truth = [json.loads(line) for line in (prior / "census.jsonl").read_text().splitlines()]
    assert len(truth) == 229
    case_ids = {row["case"] for row in truth}
    queries, citations, inputs = selective_rules.load_case_inputs(PROOF, case_ids)
    data = PLANNING / "search-design/case1/data"
    catalogue = data / "catalogue.sqlite"
    manifests = {case: load(data / (case.replace(":", "_") + ".json")) for case in case_ids}
    roots = {case: manifest["root"] for case, manifest in manifests.items()}
    facts = {root: load(prior / ("structure-" + Path(root).name + ".json")) for root in set(roots.values())}
    ordering = PROOF / "runs/roles-compare-20261006/efficiency/ordering-method.json"
    excerpt_rules.STOP_WORDS.update(load(ordering)["stop_words"])
    excerpt_rules.STOP_WORDS.update({"py", "json", "yaml", "md", "mjs", "test", "tests"})
    selectors, context, term_rows = {}, {}, []
    for root in sorted(set(roots.values())):
        index = selective_rules.repository_scent(root, catalogue)
        expected = {
            "masked-analysis-engine-5a501e77-git": 16696,
            "parse-server-f7b91ad": 5086,
            "umami-ec0ff50": 37874,
        }
        assert len(index.documents) == expected[Path(root).name]
        selector = selective_rules.Selector(
            index,
            {case: query for case, query in queries.items() if roots[case] == root},
            citations,
        )
        selectors[root] = selector
        for case, (ranked, absent) in selector.terms.items():
            context[case] = {
                "root": root,
                "query": queries[case],
                "documents": selector.documents,
                "terms": ranked,
                "absent": absent,
                "citations": citations[case],
            }
            term_rows.extend(
                {"case": case, "root": root, "documents": selector.documents, "rank": rank, **row}
                for rank, row in enumerate(ranked, 1)
            )
        print(f"Reused scent corpus {Path(root).name}: {len(index.documents)} documents", flush=True)
        del index
    (OUT / "focus-context.json").write_text(json.dumps(context, indent=2) + "\n")
    table(OUT / "focus-terms.tsv", term_rows)
    views = {}
    for row in truth:
        key = (row["case"], row["file"], tuple(tuple(r) for r in row["unit"]["ranges"]))
        views.setdefault(key, row)
    variants = ("whole", "D", *selective_rules.RULES)
    unit_rows, line_rows, selected = [], [], {}
    for (case, file, runs), row in views.items():
        root = roots[case]
        lines = (Path(root) / file).read_text().splitlines()
        source = {n: lines[n - 1] for a, b in runs for n in range(a, b + 1)}
        whole_body = "\n".join(f"{n}: {text}" for n, text in source.items())
        for rule in variants:
            if rule == "whole":
                excerpt = excerpt_rules.render_selection(source, source)
            elif rule == "D":
                excerpt = excerpt_rules.render_excerpt(source, facts[root][file], queries[case], rule)
            else:
                excerpt = selectors[root].render(source, facts[root][file], case, rule, file=file)
            shown = {n for a, b in excerpt["runs"] for n in range(a, b + 1)}
            selected[(case, file, runs, rule)] = shown
            assert shown <= source.keys()
            if rule.startswith("C"):
                counts = defaultdict(int)
                for n in shown:
                    counts[selective_rules.scope_at(n, facts[root][file])] += 1
                assert max(counts.values(), default=0) <= int(rule.split("_")[1])
            unit_rows.append(
                {
                    "case": case,
                    "population": row["population"],
                    "file": file,
                    "unit": row["unit"],
                    "rule": rule,
                    "whole_lines": len(source),
                    "kept_lines": len(shown),
                    "whole_tokens": len(whole_body) / 4,
                    "kept_tokens": len(excerpt["body"]) / 4,
                    "excerpt": excerpt,
                }
            )
    for row in truth:
        key = (row["case"], row["file"], tuple(tuple(r) for r in row["unit"]["ranges"]))
        for rule in variants:
            line_rows.append(
                {
                    "population": row["population"],
                    "case": row["case"],
                    "file": row["file"],
                    "line": row["line"],
                    "rule": rule,
                    "kept": row["line"] in selected[(*key, rule)],
                }
            )
    summaries = []
    for population, expected in (("dev110", 201), ("hard27", 28), ("all", 229)):
        labels = [r for r in line_rows if population == "all" or r["population"] == population]
        units = [r for r in unit_rows if population == "all" or r["population"] == population]
        for rule in variants:
            labeled = [r for r in labels if r["rule"] == rule]
            measured = [r for r in units if r["rule"] == rule]
            assert len(labeled) == expected
            whole_tokens = sum(r["whole_tokens"] for r in measured)
            whole_lines = sum(r["whole_lines"] for r in measured)
            kept_tokens = sum(r["kept_tokens"] for r in measured)
            kept_lines = sum(r["kept_lines"] for r in measured)
            summaries.append(
                {
                    "population": population,
                    "rule": rule,
                    "kept": sum(r["kept"] for r in labeled),
                    "labels": expected,
                    "recall": sum(r["kept"] for r in labeled) / expected,
                    "holding_units": len(measured),
                    "whole_lines": whole_lines,
                    "kept_lines": kept_lines,
                    "source_share": kept_lines / whole_lines,
                    "whole_tokens": whole_tokens,
                    "kept_tokens": kept_tokens,
                    "token_share": kept_tokens / whole_tokens,
                }
            )
    controls = {(s["population"], s["rule"]): s for s in summaries}
    assert controls[("dev110", "D")]["kept"] == 196 and controls[("hard27", "D")]["kept"] == 28
    for population in ("dev110", "hard27"):
        old = next(
            s for s in load(prior / "rule-summary.json") if s["population"] == population and s["rule"] == "D"
        )
        assert controls[(population, "D")]["kept_tokens"] == old["kept_tokens_estimate"]
    for name, rows in (("census-lines", line_rows), ("census-units", unit_rows)):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "rule-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    table(OUT / "rules.tsv", summaries)
    table(OUT / "census-lines.tsv", line_rows)
    paths = [
        *inputs,
        prior / "census.jsonl",
        prior / "rule-summary.json",
        catalogue,
        ordering,
        Path(__file__),
        Path(selective_rules.__file__),
        Path(excerpt_rules.__file__),
        Path(selective_rules.owners()[0].__file__),
        Path(selective_rules.owners()[1].__file__),
    ]
    paths.extend(prior / ("structure-" + Path(root).name + ".json") for root in facts)
    (OUT / "census-provenance.json").write_text(
        json.dumps(
            {
                "provider_calls": 0,
                "spend_usd": 0,
                "inputs": {str(path): digest(path) for path in paths},
                "holding_units": len(views),
                "labels": 229,
                "tokens": "numbered body characters / 4 including inline omission markers",
                "citation_owner": "original claim.evidence integer line points; file-only is not a line seed",
                "rarity_owner": (
                    "case1 ScentIndex on canonical persisted admitted ranges and unbound request documents"
                ),
                "DF_order": ("present terms by ascending DF, first-mention ties; descending IDF at fixed N"),
            },
            indent=2,
        )
        + "\n"
    )
    for row in summaries:
        print(
            row["population"],
            row["rule"],
            f"{row['kept']}/{row['labels']}",
            f"tokens {row['token_share']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
