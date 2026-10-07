"""Post-run literal deciding-line attribution and provider spend reconciliation. No provider calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

from replay import EVAL, emitted_labels


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-folder", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    cases = {c["case_id"]: c for c in load(EVAL / "cases/replay-dev110.json")}
    packs = {p["case"]: p for p in load(args.case_folder / "trial-final/agent-inputs.json")}
    sample = {
        row["case"]: row["stratum"] for row in load(args.case_folder / "trial-final/sample.json")["sample"]
    }
    baselines = {
        r["case"]: r for r in csv.DictReader((args.case_folder / "findings.tsv").open(), delimiter="\t")
    }
    baseline_lines = {
        (r["case"], r["file"], int(r["line"])): r
        for r in csv.DictReader((args.case_folder / "lines.tsv").open(), delimiter="\t")
    }
    results = load(args.run / "results.json")
    assert len(results) == 20 and {r["case"] for r in results} == set(sample)
    line_rows, finding_rows, raw_agent_cost = [], [], Decimal(0)
    for row in sorted(results, key=lambda r: r["case"]):
        cid = row["case"]
        folder = args.run / cid.replace(":", "_")
        shown = {}
        sources = {}
        with (folder / "source-lines.jsonl").open() as source:
            for line in source:
                item = json.loads(line)
                file = item["file"]
                assert file not in packs[cid]["withheld"] and Path(file).name not in (
                    "AGENTS.md",
                    "CLAUDE.md",
                )
                if file not in sources:
                    sources[file] = (Path(packs[cid]["repository"]) / file).read_text().splitlines()
                assert sources[file][item["line"] - 1] == item["text"], (cid, file, item["line"])
                shown[(file, item["line"])] = item["first_call"]
        labels = emitted_labels(cases[cid], shown)
        delivered = sum(label["shown"] for label in labels)
        complete = row["status"] == "agent_final"
        normalized = {
            **row,
            "stratum": sample[cid],
            "registered": len(labels),
            "delivered": delivered,
            "all_registered_delivered": delivered == len(labels) if labels else None,
            "agent_completed": complete,
            "calls_in_target": 2 <= row["jvn_calls"] <= 5,
            "completed_in_target": complete and 2 <= row["jvn_calls"] <= 5,
            "total_usd": str(Decimal(row["agent_usd"]) + Decimal(row["jev_usd"])),
            "active_wall_seconds": row.get("active_wall_seconds", row["wall_seconds"]),
            "lab81_delivered": int(baselines[cid]["lab_halfcent_delivered"]),
            "native74_delivered": int(baselines[cid]["native_default_delivered"]),
            "lab_requests": baselines[cid]["lab_halfcent_requests"],
            "lab_usd": baselines[cid]["lab_halfcent_usd"],
            "lab_http_seconds": baselines[cid]["lab_halfcent_http_seconds"],
            "native_replay_attempts": baselines[cid]["native_replay_attempts"],
        }
        finding_rows.append(normalized)
        for label in labels:
            old = baseline_lines[(cid, label["file"], label["line"])]
            line_rows.append(
                {
                    "case": cid,
                    **label,
                    "lab81_shown": old["lab_halfcent_shown"],
                    "native74_shown": old["native_default_shown"],
                    "replay_ceiling_shown": old["compression_shown"],
                }
            )
        for file in folder.glob("agent-*-response.json"):
            response = load(file)
            raw_agent_cost += Decimal(str(response["usage"]["cost"]))
        for file in folder.glob("tool-*-response.json"):
            assert len(file.read_text().rstrip("\n")) <= 24000
    agent_usd = sum((Decimal(r["agent_usd"]) for r in finding_rows), Decimal(0))
    assert abs(agent_usd - raw_agent_cost) < Decimal("0.000000000001")
    provenance, after = load(args.run / "provenance.json"), load(args.run / "source-after.json")
    assert provenance["source_commit_before"] == after["commit"]
    assert provenance["source_status_before"] == after["status"]
    summary = {
        "measurement": "development adaptive agent search over frozen dev110 stratified sample",
        "findings": len(finding_rows),
        "registered": len(line_rows),
        "delivered": sum(r["delivered"] for r in finding_rows),
        "labelled_findings": sum(r["registered"] > 0 for r in finding_rows),
        "full_delivery_findings": sum(r["all_registered_delivered"] is True for r in finding_rows),
        "agent_completed": sum(r["agent_completed"] for r in finding_rows),
        "calls_in_target": sum(r["calls_in_target"] for r in finding_rows),
        "completed_in_target": sum(r["completed_in_target"] for r in finding_rows),
        "status_counts": dict(Counter(r["status"] for r in finding_rows)),
        "agent_requests": sum(r["agent_requests"] for r in finding_rows),
        "jvn_calls": sum(r["jvn_calls"] for r in finding_rows),
        "jev_requests": sum(r["jev_requests"] for r in finding_rows),
        "agent_usd": str(agent_usd),
        "jev_usd": str(sum((Decimal(r["jev_usd"]) for r in finding_rows), Decimal(0))),
        "lab81_delivered": sum(r["lab81_delivered"] for r in finding_rows),
        "native74_delivered": sum(r["native74_delivered"] for r in finding_rows),
        "tool_error_count": sum(len(r["tool_errors"]) for r in finding_rows),
        "resumed_findings": sum(r.get("resumed_response_ceiling", False) for r in finding_rows),
        "source_unchanged": True,
    }
    for field in [
        "jvn_calls",
        "agent_requests",
        "returned_tokens_cl100k",
        "agent_input_tokens",
        "agent_output_tokens",
        "agent_cached_tokens",
        "source_files",
        "source_lines",
        "active_wall_seconds",
        "wall_seconds",
    ]:
        values = [r[field] for r in finding_rows]
        summary[field] = {
            "median": statistics.median(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
        }
    fields = [
        "case",
        "stratum",
        "registered",
        "delivered",
        "all_registered_delivered",
        "status",
        "agent_completed",
        "jvn_calls",
        "calls_in_target",
        "completed_in_target",
        "agent_requests",
        "jev_requests",
        "returned_tokens_cl100k",
        "agent_input_tokens",
        "agent_output_tokens",
        "agent_cached_tokens",
        "jev_input_tokens",
        "jev_output_tokens",
        "agent_usd",
        "jev_usd",
        "total_usd",
        "wall_seconds",
        "active_wall_seconds",
        "source_files",
        "source_lines",
        "lab81_delivered",
        "native74_delivered",
        "lab_requests",
        "lab_usd",
        "lab_http_seconds",
        "native_replay_attempts",
    ]
    with (args.case_folder / "trial-findings.tsv").open("w") as out:
        writer = csv.DictWriter(
            out, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(finding_rows)
    with (args.case_folder / "trial-lines.tsv").open("w") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "case",
                "file",
                "line",
                "shown",
                "first_call",
                "lab81_shown",
                "native74_shown",
                "replay_ceiling_shown",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(line_rows)
    (args.run / "scored-findings.json").write_text(json.dumps(finding_rows, indent=2) + "\n")
    (args.run / "scored-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        str(f.relative_to(args.case_folder)): {
            "bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        for f in args.run.rglob("*")
        if f.is_file() and f.suffix != ".sqlite"
    }
    manifest["spend-ledger.jsonl"] = {
        "bytes": (args.case_folder / "spend-ledger.jsonl").stat().st_size,
        "sha256": hashlib.sha256((args.case_folder / "spend-ledger.jsonl").read_bytes()).hexdigest(),
    }
    (args.case_folder / "trial-provenance.yaml").write_text(
        json.dumps(
            {
                "summary": summary,
                "run": str(args.run),
                "provenance": provenance,
                "continuation": load(args.run / "resume-provenance.json"),
                "artifacts": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
