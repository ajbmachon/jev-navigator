"""Execute retained model plans and measure reach. Labels are read only to score outputs."""

import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import msgspec
from paid_planner import resources
from receipts import write_json

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import items_to_judge, read_ranges
from jev_navigator.search_plan import decode_plan, execute_plan

BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
TUNING = {*(f"P{i}" for i in range(1, 8)), *(f"U{i}" for i in range(1, 7))}


def term_origin(approach, context):
    terms = [p.term for p in approach.provenance]
    if terms and any(term.casefold() in context["finding"].casefold() for term in terms):
        return "copied finding word"
    if any("hypothesis" in (p.source + " " + p.transformation).casefold() for p in approach.provenance):
        return "model proposed convention"
    visible = json.dumps({k: v for k, v in context.items() if k != "finding"})
    if terms and all(term in visible for term in terms):
        return "copied cited code or outline"
    return "new term or unverified provenance"


def binding_status(outcome):
    if outcome.reaches:
        return "bound to real code"
    invalid = [
        problem
        for problem in outcome.problems
        if problem != "no matching places" and "left out" not in problem
    ]
    return "invalid or unbound argument" if invalid else "valid empty search"


def main():
    import ijson

    source, out = (Path(arg).expanduser() for arg in sys.argv[1:3])
    # Reuse the pushed Case 1 implementation unchanged, pinned by its source receipt.
    spec = importlib.util.spec_from_file_location("jev_navigator.selection.scent", out / "scent.py")
    scent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scent
    spec.loader.exec_module(scent)
    for dataset in ("dev110", "hard27"):
        with (BASE / f"runs/pack-49b78955/pack-inputs-{dataset}.json").open("rb") as stream:
            inputs = {
                cid: row
                for cid, row in ijson.kvitems(stream, "cases")
                if dataset == "dev110" or cid in TUNING
            }
        name = "replay-dev110" if dataset == "dev110" else "hard27"
        with (BASE / f"cases/{name}.json").open("rb") as stream:
            cases = [c for c in ijson.items(stream, "item") if c["case_id"] in inputs]
        roots = dict.fromkeys(inputs[c["case_id"]]["repository"] for c in cases)
        for root in roots:
            members = [c for c in cases if inputs[c["case_id"]]["repository"] == root]
            withheld = set(inputs[members[0]["case_id"]]["withheld"])
            with CodeIndex.from_git(Path(root)) as inventory:
                index = CodeIndex(
                    Path(root),
                    tuple(f for f in inventory.files if f not in withheld),
                    commit=inventory.commit,
                )
                for case in members:
                    cid = case["case_id"]
                    folder = out / "cases" / cid.replace(":", "_")
                    if not (folder / "plan.json").exists() or (folder / "execution.json").exists():
                        continue
                    resources()
                    context = json.loads(
                        (source / "cases" / folder.name / "planner-context.json").read_text()
                    )
                    plan = decode_plan((folder / "plan.json").read_bytes())
                    start = time.perf_counter()
                    result = execute_plan(index, plan, box_chars=70000)
                    documents = (
                        scent.scent_document(
                            c.unit.id,
                            c.unit.path,
                            c.unit.symbol,
                            read_ranges(index, c.unit.path, c.unit.ranges),
                            test=c.unit.test,
                        )
                        for c in result.candidates
                    )
                    scores = scent.ScentIndex(documents).scores(context["finding"])
                    ranked = sorted(
                        result.candidates,
                        key=lambda c: (
                            min(a.rank for a in c.approaches),
                            -scores[c.unit.id],
                            c.unit.path,
                            c.unit.ranges,
                            c.unit.id,
                        ),
                    )
                    outcomes = []
                    for outcome in result.outcomes:
                        status = binding_status(outcome)
                        outcomes.append(
                            {
                                **msgspec.to_builtins(outcome),
                                "binding": status,
                                "term_origin": term_origin(outcome.approach, context),
                            }
                        )
                    labels = []
                    prefix_counts = {4: 0, 8: 0}
                    with (folder / "ranked-candidates.jsonl").open("w") as saved:
                        for candidate in ranked:
                            unit = candidate.unit
                            items = [
                                {**asdict(item), "code": read_ranges(index, item.file, item.ranges)}
                                for item in items_to_judge(unit)
                            ]
                            saved.write(
                                json.dumps(
                                    {
                                        "unit": asdict(unit),
                                        "approach_ranks": [a.rank for a in candidate.approaches],
                                        "scent": scores[unit.id],
                                        "items": items,
                                    }
                                )
                                + "\n"
                            )
                    flattened = [(c, item) for c in ranked for item in items_to_judge(c.unit)]
                    for label in case["labels"]:
                        reached = [
                            c
                            for c in ranked
                            if c.unit.path == label["file"]
                            and any(
                                a <= label["first_line"] and b >= label["last_line"] for a, b in c.unit.ranges
                            )
                        ]
                        first = reached[0] if reached else None
                        origin = (
                            term_origin(min(first.approaches, key=lambda a: a.rank), context)
                            if first
                            else None
                        )
                        record = {
                            **label,
                            "reached": bool(reached),
                            "term_origin": origin,
                            "first_unit": first.unit.id if first else None,
                            "approach_ranks": [a.rank for a in first.approaches] if first else [],
                        }
                        for cap in prefix_counts:
                            hit = any(
                                item.file == label["file"]
                                and any(
                                    a <= label["first_line"] and b >= label["last_line"]
                                    for a, b in item.ranges
                                )
                                for _, item in flattened[: 16 * cap]
                            )
                            record[f"first_{cap}_groups_reach"] = hit
                            prefix_counts[cap] += hit
                        labels.append(record)
                    record = {
                        "case": cid,
                        "dataset": dataset,
                        "candidates": len(ranked),
                        "items": len(flattened),
                        "judge_groups_needed": math.ceil(len(flattened) / 16),
                        "outcomes": outcomes,
                        "labels": labels,
                        "prefix_reach": prefix_counts,
                        "seconds": time.perf_counter() - start,
                        "commit": index.commit,
                        "delivery": "unjudged; no Jev candidate call authorized before guard review",
                    }
                    write_json(folder / "execution.json", record)
                    print(cid, len(ranked), sum(label["reached"] for label in labels), flush=True)
    records = [json.loads(p.read_text()) for p in sorted((out / "cases").glob("*/execution.json"))]
    write_json(out / "execution-summary.json", {"status": "development measurement", "cases": records})


if __name__ == "__main__":
    main()
