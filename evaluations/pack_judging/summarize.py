"""Reconcile source coverage and billing from immutable answers and literal packets."""

from __future__ import annotations

import statistics
from collections import defaultdict
from decimal import Decimal

from paid_support import Ledger

from .frozen import load, sha256
from .prepare import write
from .refresh import ADMITTED
from .run import OUT, diagnostics, folder, records


def invoice(name, cid):
    receipts = [load(p) for p in (OUT / "answers" / name / folder(cid)).rglob("receipt.json")]
    # Exact canary reuse keeps the original attempt id; count its bill only once.
    unique = {r["id"]: r for r in receipts}
    return {
        "physical_calls": len(unique),
        "usd": float(sum((Decimal(r.get("usd", "0")) for r in unique.values()), Decimal(0))),
        "service_work_seconds": sum(r["seconds"] for r in unique.values()),
    }


def aggregate(rows):
    total = {}
    for dataset in ("dev110", "hard27"):
        for arm in sorted({row["arm"] for row in rows}):
            subset = [row for row in rows if (row["dataset"], row["arm"]) == (dataset, arm)]
            if not subset:
                continue
            metrics = {
                key: sum(r["metrics"][key] for r in subset)
                for key in subset[0]["metrics"]
                if key != "first_rank"
            }
            first = [r["metrics"]["first_rank"] for r in subset if r["metrics"]["first_rank"] is not None]
            metrics.update(
                median_first_rank_found=statistics.median(first) if first else None,
                no_witness_points=len(subset) - len(first),
                deciding_share_at_080=metrics["high_deciding_units"] / max(1, metrics["deciding_units"]),
                citation_precision_proxy_at_080=metrics["high_deciding_units"]
                / max(1, metrics["high_units"]),
            )
            money = {key: sum(r["invoice"][key] for r in subset) for key in subset[0]["invoice"]}
            total[f"{dataset}:{arm}"] = {
                "metrics": metrics,
                "lab": {
                    room: sum(label["delivered"] for r in subset for label in r["lab"][room]["labels"])
                    for room in ("7200", "20000", "36000")
                },
                "native": {
                    room: sum(r["native"][room]["delivered"].count(True) for r in subset)
                    for room in ("7200", "20000", "36000")
                },
                "invoice": money,
                "usd_per_finding": money["usd"] / len(subset),
                "service_work_seconds_per_finding": money["service_work_seconds"] / len(subset),
                "packing_seconds_per_finding": sum(r["packing_seconds"] for r in subset) / len(subset),
                "completed_groups": sum(r["completed_groups"] for r in subset),
                "failed_groups": sum(r["failed_groups"] for r in subset),
            }
    return total


def main():
    cases = load(ADMITTED / "frozen-cases.json")
    rows = load(OUT / "per-finding-results.json")
    finalists = load(OUT / "finalists.json")
    names = [*finalists["jev"], finalists["llm"]]
    by_case = {c["context"]["case"]: c for c in cases}
    coverage = []
    for name in names:
        prepared = {(r["case"], r["group"]): r for r in records(name)}
        for row in rows:
            if row["arm"] != name:
                continue
            cid = row["case"]
            paths = [
                OUT / "answers" / name / folder(cid) / str(g["ordinal"]) / "observation.json"
                for g in by_case[cid]["groups"]
            ]
            actual = [load(p) for p in paths if p.exists()]
            row["invoice"] = invoice(name, cid)
            row["completed_groups"] = sum(a["failure"] is None for a in actual)
            row["failed_groups"] = sum(a["failure"] is not None for a in actual)
            assert all(a["request_sha256"] == prepared[cid, a["group"]]["request_sha256"] for a in actual)
            coverage.append(
                {
                    "arm": name,
                    "case": cid,
                    "expected_groups": len(paths),
                    "present_groups": len(actual),
                    "failed_groups": row["failed_groups"],
                }
            )
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap="7")
    categories = defaultdict(lambda: defaultdict(Decimal))
    for event in ledger.latest().values():
        categories[event["category"]][event["status"]] += Decimal(event["usd"])
    baseline_pilot = []
    chosen = load(ADMITTED / "pilot.json")["groups"]
    for case in cases:
        cid = case["context"]["case"]
        if cid in chosen:
            baseline_pilot.append(
                {
                    "case": cid,
                    **diagnostics(
                        case,
                        {
                            g["ordinal"]: g["baseline_rows"]
                            for g in case["groups"]
                            if g["ordinal"] in chosen[cid]
                        },
                    ),
                }
            )
    write(OUT / "baseline-pilot.json", baseline_pilot)
    write(OUT / "per-finding-results.json", rows)
    summary = {
        "status": "COMPLETE"
        if all(r["expected_groups"] == r["present_groups"] and not r["failed_groups"] for r in coverage)
        else "INCOMPLETE",
        "arms": aggregate(rows),
        "coverage": coverage,
        "money": {
            "cap_usd": "7.00",
            "available_usd": str(ledger.balance()),
            "categories": {k: {s: str(v) for s, v in costs.items()} for k, costs in categories.items()},
            "settled_usd": str(
                sum(
                    (Decimal(e["usd"]) for e in ledger.latest().values() if e["status"] == "settled"),
                    Decimal(0),
                )
            ),
            "uncertain_reserved_usd": str(
                sum(
                    (Decimal(e["usd"]) for e in ledger.latest().values() if e["status"] == "reserved"),
                    Decimal(0),
                )
            ),
        },
        "identities": {
            name: sha256(OUT / name)
            for name in (
                "engine-proof.json",
                "guard-hand-inspection.json",
                "pilot-summary.json",
                "pilot-hand-inspection.json",
                "finalists.json",
                "spend-ledger.jsonl",
                "per-finding-results.json",
            )
        },
    }
    write(OUT / "summary.json", summary)
    print(summary["status"], summary["money"])


if __name__ == "__main__":
    main()
