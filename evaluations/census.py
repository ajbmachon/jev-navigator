"""Zero-call deciding-line census and cumulative excerpt measurements."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import excerpt_rules as rules

from jev_navigator.index.units import LineAnchor, Reading, UnitReader

PLANNING = Path.home() / ".local/share/jvn-takeover/2026-10-03"
PROOF = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
OUT = PLANNING / "search-design/excerpts"
DATA = PLANNING / "search-design/case1/data"


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Excerpt census is zero cost and offline.")


def load(path):
    return json.loads(path.read_text())


def in_range(line, span):
    return span[0] <= line <= span[1]


def classify(line, source, facts, query):
    text = source[line]
    hit = rules.matches(text, query)
    categories = []
    if any(in_range(line, s["header"]) for s in facts["signatures"]) or any(
        in_range(line, s["range"]) for s in facts["decorators"]
    ):
        categories.append("signature or decorator")
    if facts["language"] == "python":
        for s in facts["signatures"]:
            if any(
                e["range"][0] == s["body_start"]
                and in_range(line, e["range"])
                and re.match(r"(?is)^(?:[ru]|ur|ru)?[\'\"]", e["text"])
                for e in facts["strings"]
            ):
                categories.append("docstring")
                break
        if "docstring" not in categories:
            for e in facts["strings"]:
                if in_range(line, e["range"]) and re.match(r"(?is)^(?:[ru]|ur|ru)?[\'\"]", e["text"]):
                    preceding = source.get(e["range"][0] - 1, "")
                    if e["range"][0] <= 2 or not preceding.strip() or preceding.lstrip().startswith("#"):
                        categories.append("docstring")
                        break
    guards = [
        c
        for c in facts["conditions"]
        if c.get("kind") in {"if_statement", "elif_clause"}
        and c["range"][1] - c["body_start"] <= 5
        and any(c["range"][0] <= e["range"][0] <= c["range"][1] for e in facts["exits"])
        and c["range"][0] <= min(source) + 10
    ]
    if any(in_range(line, c["range"]) for c in guards):
        categories.append("early guard clause")
    if any(in_range(line, c["header"]) for c in facts["conditions"]):
        categories.append("condition line")
    if any(
        in_range(line, c["range"])
        and any(in_range(n, c["range"]) and rules.matches(t, query) for n, t in source.items())
        for c in facts["conditions"]
    ):
        categories.append("inside a condition enclosing a word match")
    if any(in_range(line, e["range"]) for e in facts["exits"]):
        categories.append("return or raise or throw")
    callees = [d for c in facts["calls"] if in_range(line, c["range"]) for d in c["delegates"]]
    if callees:
        categories.append("call to an in-repo function")
    # These are conservative observable external API cues, not a semantic
    # claim that every arbitrary save/execute call has an outside effect.
    effect_pattern = (
        r"\b(?:open|read_text|write_text|read_bytes|write_bytes|unlink|mkdir|rmdir|fetch|readFile|writeFile|"
        r"unlinkSync|execSync|spawn|system|Popen|check_output|check_call)\s*\("
        r"|\b(?:requests|httpx|subprocess|os|fs|shutil|prisma|knex|sequelize|database|db|"
        r"collection|clickhouse|conn)\s*[.\[]"
    )
    effect = bool(re.search(effect_pattern, text))
    if any(
        in_range(line, c["range"])
        and re.search(
            r"\.(?:none|any|query|execute|insert|updateMany|deleteMany|extractall|copy2|communicate)\s*\(",
            c["text"],
        )
        for c in facts["calls"]
    ):
        effect = True
    if "/queries/prisma/" in str(facts.get("file", "")) and any(
        in_range(line, c["range"]) and re.search(r"\.(?:create|update|delete)\s*\(", c["text"])
        for c in facts["calls"]
    ):
        effect = True
    if effect:
        categories.append("outside-effect API cue")
    assignment = [
        a
        for a in facts["assignments"]
        if in_range(line, a["range"])
        and any(i["range"][0] > a["range"][1] and i["text"] in a["targets"] for i in facts["identifiers"])
    ]
    if assignment:
        categories.append("assignment of a value used later")
    if not categories:
        categories.append("other")
    return {
        "categories": categories,
        "primary": categories[0],
        "word_or_name_match": hit,
        "cited_name_match": any(
            re.search(r"(?<![\w$])" + re.escape(v) + r"(?![\w$])", text)
            for name in rules.names_from_text(query).code
            for v in rules.spelling_variants(name)
        ),
        "callees": callees,
        "effect_cue": effect,
    }


def main():
    sys.addaudithook(deny_network)
    OUT.mkdir(parents=True, exist_ok=True)
    rules.STOP_WORDS.update(
        load(PROOF / "runs/roles-compare-20261006/efficiency/ordering-method.json")["stop_words"]
    )
    rules.STOP_WORDS.update({"py", "json", "yaml", "md", "mjs", "test", "tests"})
    ledger = {
        (r["case"], r["file"], r["line"]): r
        for r in (json.loads(s) for s in (PLANNING / "discovery/lines.jsonl").read_text().splitlines())
        if r["case"].startswith("analysis-engine:") or re.fullmatch(r"[PU][1-7]", r["case"])
    }
    db = sqlite3.connect(f"file:{DATA / 'catalogue.sqlite'}?mode=ro", uri=True)
    case_ids = db.execute("select id from cases order by id").fetchall()
    cases = {cid: load(DATA / (cid.replace(":", "_") + ".json")) for (cid,) in case_ids}
    root_files = defaultdict(set)
    for case in cases.values():
        root_files[case["root"]].update(label["file"] for label in case["labels"])
    structures = {}
    for root, files in root_files.items():
        structures[root] = rules.Structure(root, sorted(files))
        (OUT / ("structure-" + Path(root).name + ".json")).write_text(
            json.dumps({f: structures[root].facts(f) for f in sorted(files)}) + "\n"
        )
        print("Parsed", len(files), "deciding files in", Path(root).name, flush=True)
    readers = {
        root: UnitReader(s.index, 2_000_000_000, False, Reading.MIXED) for root, s in structures.items()
    }
    census, unit_rows = [], {}
    for cid, case in cases.items():
        case["dataset"] = "dev110" if cid.startswith("analysis-engine:") else "hard27"
        root = case["root"]
        for label in case["labels"]:
            file, line = label["file"], label["line"]
            record = ledger[(cid, file, line)]
            query = record["finding"]
            holders = []
            for key in label["units"]:
                canonical = case["identities"].get(key, key)
                row = db.execute(
                    "select binding from documents where root=? and id=?", (root, key)
                ).fetchone()
                if row is None:
                    row = db.execute(
                        "select binding from documents where root=? and id=?", (root, canonical)
                    ).fetchone()
                if row is None:
                    for alias, target in case["identities"].items():
                        if target == canonical:
                            row = db.execute(
                                "select binding from documents where root=? and id=?", (root, alias)
                            ).fetchone()
                            if row is not None:
                                break
                if row:
                    binding = json.loads(row[0])
                    if any(in_range(line, r) for r in binding["ranges"]):
                        holders.append(binding)
            if holders:
                holder = min(holders, key=lambda u: sum(b - a + 1 for a, b in u["ranges"]))
                runs = holder["ranges"]
                symbol, kind = holder["symbol"], holder["kind"]
                registered = True
            else:
                # Resolve real holding units for floors and withheld tests.
                # A target citation never supplies an excerpt window.
                resolved, problem = readers[root].resolve(LineAnchor(file, line))
                if resolved:
                    unit = min(resolved, key=lambda u: sum(b - a + 1 for a, b in u.ranges))
                    runs = [list(r) for r in unit.ranges]
                    symbol, kind = unit.symbol, str(unit.kind)
                else:
                    assert file.endswith("source-inventory-contract.mjs"), (cid, file, problem)
                    runs = [[1, len(structures[root].index.lines(file))]]
                    symbol, kind = "<unparsed file>", "unparsed"
                registered = False
            lines = structures[root].index.lines(file)
            source = {n: lines[n - 1] for a, b in runs for n in range(a, b + 1)}
            assert line in source, (cid, label)
            facts = {**structures[root].facts(file), "file": file}
            unit_id = f"{cid}|{file}|{runs}"
            selected = {rule: rules.render_excerpt(source, facts, query, rule) for rule in rules.RULES}
            row = {
                "case": cid,
                "population": case["dataset"],
                "file": file,
                "line": line,
                "unit": {"symbol": symbol, "kind": kind, "ranges": runs, "registered_candidate": registered},
                "source": source[line],
                "unit_lines": len(source),
                **classify(line, source, facts, query),
                "kept_by": {r: any(in_range(line, span) for span in s["runs"]) for r, s in selected.items()},
            }
            census.append(row)
            if unit_id not in unit_rows:
                whole_body = "\n".join(f"{n}: {t}" for n, t in source.items())
                unit_rows[unit_id] = {
                    "population": case["dataset"],
                    "case": cid,
                    "file": file,
                    "symbol": symbol,
                    "whole_lines": len(source),
                    "whole_tokens_estimate": len(whole_body) / 4,
                    "variants": {
                        r: {
                            "lines": s["shown_lines"],
                            "tokens_estimate": len(s["body"]) / 4,
                            "ranges": s["runs"],
                        }
                        for r, s in selected.items()
                    },
                }
    summaries = []
    for pop, expected in [("dev110", 201), ("hard27", 28)]:
        labels = [r for r in census if r["population"] == pop]
        units = [r for r in unit_rows.values() if r["population"] == pop]
        assert len(labels) == expected, (pop, len(labels))
        for rule in rules.RULES:
            summaries.append(
                {
                    "population": pop,
                    "rule": rule,
                    "labels": expected,
                    "kept": sum(r["kept_by"][rule] for r in labels),
                    "holding_units": len(units),
                    "whole_lines": sum(u["whole_lines"] for u in units),
                    "kept_lines": sum(u["variants"][rule]["lines"] for u in units),
                    "whole_tokens_estimate": sum(u["whole_tokens_estimate"] for u in units),
                    "kept_tokens_estimate": sum(u["variants"][rule]["tokens_estimate"] for u in units),
                }
            )
    (OUT / "census.jsonl").write_text("".join(json.dumps(r) + "\n" for r in census))
    (OUT / "holding-units.json").write_text(json.dumps(unit_rows, indent=2) + "\n")
    (OUT / "rule-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    counts = {
        pop: {
            "primary": dict(Counter(r["primary"] for r in census if r["population"] == pop)),
            "overlapping": dict(
                Counter(c for r in census if r["population"] == pop for c in r["categories"])
            ),
            "word_or_name_match": sum(r["word_or_name_match"] for r in census if r["population"] == pop),
            "cited_name_match": sum(r["cited_name_match"] for r in census if r["population"] == pop),
        }
        for pop in ["dev110", "hard27"]
    }
    (OUT / "census-summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    input_paths = [
        PLANNING / "discovery/lines.jsonl",
        PLANNING / "search-design/case1/deciding-ranks.jsonl",
        DATA / "input-manifest.json",
    ]
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "provider_calls": 0,
                "spend_usd": 0,
                "source_head": "a6ee11ce",
                "parsers": "JVN ast_grep_rules with grammar_of/sgconfig_of and CodeIndex",
                "token_method": "numbered rendered characters / 4, including exact elision markers",
                "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
            },
            indent=2,
        )
        + "\n"
    )
    for s in summaries:
        print(
            s["population"],
            s["rule"],
            s["kept"],
            f"lines {s['kept_lines'] / s['whole_lines']:.1%}",
            f"tokens {s['kept_tokens_estimate'] / s['whole_tokens_estimate']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
