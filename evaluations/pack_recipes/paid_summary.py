"""Summarize saved recipe judgments and room curves without any provider access."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize(out):
    records = json.loads((out / "paid/results.json").read_text())
    ledger = [json.loads(line) for line in (out / "paid/spend-ledger.jsonl").read_text().splitlines()]
    usage = [row for row in ledger if row["event"] == "usage"]
    summary = {
        "findings": len(records),
        "usage": {},
        "arms": {},
        "failures": Counter(row["failure"] for row in records if row["failure"]),
        "rooms_pending": sum("measurement" not in row for row in records),
    }
    for category in ("planner", "agent", "guard", "jev"):
        observed = [row for row in usage if row["category"] == category]
        summary["usage"][category] = {
            "requests": len(observed),
            "usd": sum(float(row["usd"]) for row in observed),
            "input_tokens": sum(row["input_tokens"] for row in observed),
            "output_tokens": sum(row.get("output_tokens") or 0 for row in observed),
            "models": sorted({row["model"] for row in observed}),
        }
    slices = defaultdict(list)
    for row in records:
        slices[(row["dataset"], row["recipe"])].append(row)
        slices[(row["dataset"], "primary")].append(row)
    for (dataset, recipe), selected in slices.items():
        arm = {
            "findings": len(selected),
            "references": sum(len(row["labels"]) for row in selected),
            "judged_reach": 0,
            "unbound_bodies": 0,
            "rooms": {},
        }
        for field in ("requests", "usd", "judgment_seconds", "http_sum_seconds", "units_judged"):
            values = [row.get(field, 0) for row in selected]
            arm[field] = {
                "total": sum(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "maximum": max(values),
            }
        for row in selected:
            reach = row.get("measurement", {}).get("judged_reach", {})
            arm["judged_reach"] += reach.get("references", 0)
            arm["unbound_bodies"] += len(reach.get("unbound", []))
        arm["native_selection_losses"] = (
            sum(
                judged["judged"] and not eligible["selected"]
                for row in selected
                for judged, eligible in zip(
                    row.get("measurement", {}).get("judged_reach", {}).get("labels", []),
                    row.get("measurement", {}).get("native_before_fitting", {}).get("labels", []),
                    strict=True,
                )
            )
            if all(
                "native_before_fitting" in row.get("measurement", {}) or row["failure"] for row in selected
            )
            else None
        )
        arm["native_before_fitting"] = sum(
            row.get("measurement", {}).get("native_before_fitting", {}).get("references", 0)
            for row in selected
        )
        for room in ("7200", "20000", "36000"):
            arm["rooms"][room] = {}
            for path in ("lab", "native"):
                arm["rooms"][room][path] = sum(
                    label["delivered"]
                    for row in selected
                    for label in row.get("measurement", {})
                    .get("rooms", {})
                    .get(room, {})
                    .get(path, {})
                    .get("labels", [])
                )
            arm["rooms"][room]["native_fitting_losses"] = sum(
                row.get("measurement", {})
                .get("rooms", {})
                .get(room, {})
                .get("native", {})
                .get("fitting_lost_references", 0)
                for row in selected
            )
        summary["arms"][f"{dataset}/{recipe}"] = arm
    (out / "paid/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    summarize(parser.parse_args().out)


if __name__ == "__main__":
    main()
