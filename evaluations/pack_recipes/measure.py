"""Offline reach and real-shape request preparation for the versioned recipes.

Run with this checkout's src and the recorded Engine e733ea0c on PYTHONPATH.
Labels are read only after gathering and never become search arguments.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from configurations import RECIPES, primary_recipe
from positions import request_rank
from resources import check_resources

from jev_navigator.index.units import RangeAnchor, Reading, items_to_judge, read_ranges, resolve_anchors
from jev_navigator.sources import Seeds

BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
TAKEOVER = Path.home() / ".local/share/jvn-takeover/2026-10-03"
ROLES = BASE / "runs/roles-compare-20261006"


def offline(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Recipes measurement permits zero provider calls")


sys.addaudithook(offline)


def load(path):
    return json.loads(path.read_text())


def covers(units, label):
    return any(
        unit.path == label["file"] and any(start <= label["first_line"] <= end for start, end in unit.ranges)
        for unit in units
    )


def requests(index, units, statement, templates):
    items = []
    for unit in units:
        for item in items_to_judge(unit):
            items.append({"file": item.file, "code": read_ranges(index, item.file, item.ranges)})
            if len(items) == 16:
                yield request(items, statement, templates)
                items = []
    if items:
        yield request(items, statement, templates)


def request(items, statement, templates):
    questions = {}
    for slot in range(len(items)):
        for name, question in templates.items():
            if not name.endswith("#0"):
                continue
            # Bind the approved wording exactly, as #148's bind_question does.
            question = json.loads(json.dumps(question).replace("items[0]", f"items[{slot}]"))
            questions[f"{name.split('#')[0]}#{slot}"] = question
    return {"state": {"targets": {"p0": statement}, "items": items}, "questions": questions}


def source_seeds(index, row, pack):
    statement = row["claim"]["statement"]
    if pack is None:
        # Missing line citations still allow whole cited files to be searched.
        files = tuple(item["file"] for item in row["claim"].get("evidence", ()) if item.get("file"))
        return Seeds(texts=(statement,), files=files)
    anchors = tuple(RangeAnchor(*anchor) for anchor in pack.anchors)
    units = resolve_anchors(index, anchors, box_chars=76_800, reading=Reading.MIXED).units
    files = tuple(
        dict.fromkeys(
            item["file"] for item in row["claim"].get("evidence", ()) if item.get("file") in index.files
        )
    )
    return Seeds(names=pack.names, texts=(statement,), files=files, anchors=anchors, units=units)


def measure(dataset, out, limit):
    from enginepy.workflows.document_analysis import evidence_pack as ep
    from enginepy.workflows.document_analysis.code_relations import CodeRelations
    from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps

    case_path = BASE / ("cases/replay-dev110.json" if dataset == "dev110" else "cases/hard27.json")
    cases = load(case_path)
    cases = [case for case in cases if dataset == "dev110" or case["case_id"].startswith(("P", "U"))]
    input_path = BASE / f"runs/pack-49b78955/pack-inputs-{dataset}.json"
    rows = load(input_path)["cases"]
    templates = load(ROLES / "step-3/roles16-candidate.json")["request"]["questions"]
    resources = check_resources()
    scopes = {}
    records = []
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dataset}-candidates.jsonl"
    with path.open("w") as output:
        for case in cases[:limit]:
            resources = check_resources()
            cid = case["case_id"]
            row = rows[cid]
            root = row["repository"]
            started = time.monotonic()
            if root not in scopes:
                withheld = frozenset(row["withheld"])
                relations = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
                scopes[root] = relations, ep.pack_index(relations, root)
            relations, index = scopes[root]
            pack = ep.pack_request(relations, root, row["claim"], row.get("points"))
            seeds = source_seeds(index, row, pack)
            primary = primary_recipe(row["claim"]["statement"], seeds.names).name
            for recipe in RECIPES:
                began = time.monotonic()
                gathered = recipe.gather(index, seeds, box_chars=76_800)
                seconds = time.monotonic() - began
                # All pieces have the real 16-item shape. Preparing is free.
                prepared = out / "requests" / dataset / recipe.name / f"{cid.replace(':', '_')}.jsonl.gz"
                prepared.parent.mkdir(parents=True, exist_ok=True)
                body_chars = 0
                groups = 0
                with gzip.open(prepared, "wt") as handle:
                    for group in requests(index, gathered.units, row["claim"]["statement"], templates):
                        handle.write(json.dumps(group, ensure_ascii=False) + "\n")
                        body_chars += len(json.dumps(group, ensure_ascii=False, separators=(",", ":")))
                        groups += 1
                labels = []
                for label in case["labels"]:
                    ranks = [
                        number for number, unit in enumerate(gathered.units, 1) if covers((unit,), label)
                    ]
                    labels.append(
                        {
                            **label,
                            "reached": bool(ranks),
                            "unit_rank": min(ranks, default=None),
                            "request_rank": request_rank(gathered.units, label["file"], label["first_line"]),
                        }
                    )
                record = {
                    "dataset": dataset,
                    "case": cid,
                    "recipe": recipe.name,
                    "version": recipe.version,
                    "primary": primary == recipe.name,
                    "ranker": "stable source order placeholder",
                    "units": [asdict(unit) for unit in gathered.units],
                    "units_to_judge": sum(len(items_to_judge(unit)) for unit in gathered.units),
                    "requests_16": groups,
                    "request_body_chars": body_chars,
                    "estimated_usd_flat": groups * 0.000656705,
                    "local_seconds": seconds,
                    "case_prepare_seconds": time.monotonic() - started,
                    "labels": labels,
                    "unresolved": [(asdict(reach), reason) for reach, reason in gathered.unresolved],
                    "sources": dict(
                        Counter(reach.source for reaches in gathered.reached_by.values() for reach in reaches)
                    ),
                    "requests_path": str(prepared),
                    "requests_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
                    "provider_calls": 0,
                    "spent_usd": 0,
                }
                output.write(json.dumps(record, default=str) + "\n")
                output.flush()
                records.append(
                    {key: value for key, value in record.items() if key not in {"units", "unresolved"}}
                )
            print(
                json.dumps(
                    {"case": cid, "primary": primary, "seconds": round(time.monotonic() - started, 2)}
                ),
                flush=True,
            )
    (out / f"{dataset}-summary.json").write_text(
        json.dumps(
            {
                "records": records,
                "resources": resources,
                "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "inputs": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [case_path, input_path]
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("dev110", "hard27"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    measure(args.dataset, args.out, args.limit)
