"""Aggregate development reach, frozen delivery and actual packet evidence."""

import argparse
import json
from pathlib import Path
from statistics import median

from positions import request_rank

from jev_navigator.index.units import Piece, Unit


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def units(record):
    return tuple(
        Unit(
            **{
                **unit,
                "ranges": tuple(map(tuple, unit["ranges"])),
                "pieces": tuple(Piece(**piece) for piece in unit["pieces"]),
            }
        )
        for unit in record["units"]
    )


def aggregate(records):
    result = {}
    for name in ("pack-local", "pack-named", "pack-convention", "primary"):
        selected = [r for r in records if (r["primary"] if name == "primary" else r["recipe"] == name)]
        labels = [label for r in selected for label in r["labels"]]
        result[name] = {
            "findings": len(selected),
            "references": len(labels),
            "reached": sum(label["reached"] for label in labels),
            "within_1_request": sum(
                label["request_rank"] is not None and label["request_rank"] <= 1 for label in labels
            ),
            "within_2_requests": sum(
                label["request_rank"] is not None and label["request_rank"] <= 2 for label in labels
            ),
            "within_8_requests": sum(
                label["request_rank"] is not None and label["request_rank"] <= 8 for label in labels
            ),
            "median_items": median(r["units_to_judge"] for r in selected),
            "median_requests": median(r["requests_16"] for r in selected),
            "total_requests": sum(r["requests_16"] for r in selected),
            "max_requests": max(r["requests_16"] for r in selected),
            "median_gather_seconds": median(r["local_seconds"] for r in selected),
            "median_all_recipe_prepare_seconds": median(
                r["case_prepare_seconds"] for r in selected if r["recipe"] == "pack-convention"
            )
            if name in {"primary", "pack-convention"}
            and any(r["recipe"] == "pack-convention" for r in selected)
            else None,
            "median_flat_forecast_usd": median(r["estimated_usd_flat"] for r in selected),
            "provider_calls": 0,
            "spent_usd": 0,
        }
    return result


def main(out):
    summary = {"status": "development measures, source order placeholder"}
    for dataset in ("dev110", "hard27"):
        records = list(rows(out / f"{dataset}-candidates.jsonl"))
        # Recompute old diagnostic positions from the actual pieces. No search reruns.
        for record in records:
            actual = units(record)
            for label in record["labels"]:
                label["request_rank"] = request_rank(actual, label["file"], label["first_line"])
        (out / f"{dataset}-candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        summary[dataset] = aggregate(records)
    frozen = json.loads((out / "frozen-development.json").read_text())
    native = json.loads((out / "native-replay.json").read_text())
    reference = {r["case"]: r["labels"] for r in rows(out / "dev110-candidates.jsonl")}
    summary["paths"] = {}
    for name in ("pack-local", "pack-named", "pack-convention", "primary"):
        lab = [r for r in frozen if (r["primary"] if name == "primary" else r["recipe"] == name)]
        real = [r for r in native if (r["primary"] if name == "primary" else r["recipe"] == name)]
        delivered = sum(
            any(
                w["file"] == label["file"] and w["first_line"] <= label["first_line"] <= w["last_line"]
                for w in r["consumer_windows"]
            )
            for r in real
            for label in reference[r["case"]]
        )
        summary["paths"][name] = {
            "frozen_lab": {
                "findings": len(lab),
                "delivered": sum(label["delivered"] for r in lab for label in r["labels"]),
                "median_requests": median(r["requests"] for r in lab),
                "median_historical_usd": median(r["reported_usd"] for r in lab),
                "median_historical_http_seconds": median(r["reported_http_seconds"] for r in lab),
            },
            "native": {
                "findings": len(real),
                "floor_only_delivered": delivered,
                "median_attempted_requests": median(r["requests"] for r in real),
                "exact_answer_hits": sum(r["exact_hits"] for r in real),
                "failures": sum(bool(r["failure"]) for r in real),
                "median_local_seconds": median(r["local_seconds"] for r in real),
                "judged_delivery": "unknown where exact requests have no stored answer",
            },
        }
    strict = json.loads((out / "strict-request-coverage.json").read_text())
    summary["strict"] = {
        "requests": sum(r["requests"] for r in strict),
        "exact_hits": sum(r["exact_hits"] for r in strict),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
