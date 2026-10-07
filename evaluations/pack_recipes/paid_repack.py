"""Repack saved paid receipts for free, without rebuilding or judging the search.

Limited runs write case checkpoints only and can overlap paid collection. A full
run requires a stable spend ledger before replacing the combined results once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from paid_rooms import measure_case
from resources import check_resources

BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
HARD_TUNING = {*(f"P{i}" for i in range(1, 8)), *(f"U{i}" for i in range(1, 7))}


def offline(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Saved-pack repacking permits zero network calls")


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def ledger_snapshot(path):
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
            length += len(block)
    return {"sha256": digest.hexdigest(), "bytes": length}


def restore_pack(value):
    """Reverse dataclasses.asdict using the archived Engine's actual value types."""
    from enginepy.workflows.document_analysis.evidence_pack import (
        EvidencePack,
        PackCost,
        PackUnit,
        RankedUnit,
    )
    from enginepy.workflows.document_analysis.skeptic_packet import (
        PacketBudget,
        PacketRegion,
        RelationReceipt,
    )

    units = []
    for selected in value["units"]:
        ranked = selected["unit"]
        unit = RankedUnit(
            **{
                **ranked,
                "runs": tuple(map(tuple, ranked["runs"])),
                "extent": tuple(map(tuple, ranked["extent"])),
            }
        )
        units.append(PackUnit(unit, tuple(selected["reasons"]), selected["text"]))
    cost = value["cost"]
    return EvidencePack(
        claim_id=value["claim_id"],
        floor=tuple(PacketRegion(**region) for region in value["floor"]),
        units=tuple(units),
        facts=tuple(value["facts"]),
        cost=PackCost(**{**cost, "stopped_by": tuple(cost["stopped_by"])}),
        searches=tuple(RelationReceipt(**receipt) for receipt in value["searches"]),
        trimmed=value["trimmed"],
        failure=value["failure"],
        budget=PacketBudget(**value["budget"]),
    )


def repack(out, limit=None):
    from enginepy.workflows.document_analysis import evidence_pack as ep
    from enginepy.workflows.document_analysis.code_relations import CodeRelations
    from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps

    ledger = out / "paid/spend-ledger.jsonl"
    before = ledger_snapshot(ledger)
    result_path = out / "paid/results.json"
    results = load(result_path)
    by_case = {(result["dataset"], result["case"]): position for position, result in enumerate(results)}
    protected = [{key: value for key, value in result.items() if key != "measurement"} for result in results]
    scopes, measured, skipped, seen = {}, [], [], set()
    started = time.monotonic()
    for dataset in ("dev110", "hard27"):
        inputs = load(BASE / f"runs/pack-49b78955/pack-inputs-{dataset}.json")["cases"]
        for record in rows(out / "combined" / f"{dataset}-candidates.jsonl"):
            if not record["primary"]:
                continue
            cid = record["case"]
            if dataset == "hard27" and cid not in HARD_TUNING:
                raise ValueError(f"Held-out case in primary inventory: {cid}")
            if record.get("version") != 2:
                raise ValueError(f"Expected v2 primary recipe for {dataset}/{cid}")
            identity = dataset, cid
            if identity in seen:
                raise ValueError(f"Duplicate primary recipe for {dataset}/{cid}")
            seen.add(identity)
            if limit is not None and len(measured) >= limit:
                break
            if identity not in by_case:
                raise ValueError(f"Paid result not complete for {dataset}/{cid}")
            check_resources()
            position = by_case[identity]
            folder = out / "paid/cases" / dataset / cid.replace(":", "_")
            if not (folder / "pack.json").exists():
                if results[position].get("failure") != "no readable anchor":
                    raise FileNotFoundError(folder / "pack.json")
                skipped.append({"dataset": dataset, "case": cid, "reason": "no readable anchor"})
                continue
            row = inputs[cid]
            root = row["repository"]
            if root not in scopes:
                withheld = frozenset(row["withheld"])
                scopes[root] = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
            relations = scopes[root]
            request = ep.pack_request(relations, root, row["claim"], row.get("points"))
            if request is None:
                raise ValueError(f"Saved pack no longer has readable anchor for {dataset}/{cid}")
            pack = restore_pack(load(folder / "pack.json"))
            # Saved bytes are delivery authority; recreated context only binds point ids.
            request = replace(
                request, floor=pack.floor, searches=pack.searches, trimmed=pack.trimmed, budget=pack.budget
            )
            group_path = folder / "groups.jsonl"
            groups = [] if not group_path.exists() and pack.cost.calls == 0 else list(rows(group_path))
            measurement = measure_case(record, pack, relations, request, groups, record["labels"])
            save(folder / "measurements.json", measurement)
            results[position]["measurement"] = measurement
            measured.append(
                {
                    "dataset": dataset,
                    "case": cid,
                    "requests": len(groups),
                    "judged_references": measurement["judged_reach"]["references"],
                }
            )
            print(json.dumps(measured[-1]), flush=True)
        if limit is not None and len(measured) >= limit:
            break
    after = ledger_snapshot(ledger)
    stable = before == after
    receipt = {
        "provider_calls": 0,
        "spent_usd": 0,
        "ledger_before": before,
        "ledger_after": after,
        "ledger_unchanged": stable,
        "measured": measured,
        "skipped": skipped,
        "seconds": time.monotonic() - started,
        "combined_results_updated": limit is None,
    }
    unchanged = [{key: value for key, value in result.items() if key != "measurement"} for result in results]
    if protected != unchanged:
        raise RuntimeError("Free repacking changed original paid metrics")
    if limit is None:
        if not stable:
            raise RuntimeError("Spend ledger changed during free repack; combined results were not replaced")
        if len(seen) != 123:
            raise ValueError(f"Expected 123 primary tuning findings, received {len(seen)}")
        save(result_path, results)
        save(out / "paid/repack-receipt.json", receipt)
    else:
        save(out / "paid" / f"repack-limit-{limit}-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    sys.addaudithook(offline)
    receipt = repack(args.out, args.limit)
    print(
        json.dumps({key: value for key, value in receipt.items() if key not in {"measured", "skipped"}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
