"""One-hop development continuation from source-identical stored confirmations."""

import argparse
import json
import time
from pathlib import Path

from configurations import NAMED
from measure import BASE, covers, load
from resources import check_resources
from summarize import units

from jev_navigator.index.units import items_to_judge, read_ranges
from jev_navigator.judgments.questions import content_hash


def main(out):
    from enginepy.workflows.document_analysis import evidence_pack as ep
    from enginepy.workflows.document_analysis.code_relations import CodeRelations
    from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps

    inputs = load(BASE / "runs/pack-49b78955/pack-inputs-dev110.json")["cases"]
    confirmations = {
        r["case"]: r["confirmed"] for r in load(out / "frozen-development.json") if r["recipe"] == NAMED.name
    }
    scopes = {}
    results = []
    with (out / "dev110-candidates.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["recipe"] != NAMED.name:
                continue
            check_resources()
            began = time.monotonic()
            cid = record["case"]
            row = inputs[cid]
            root = row["repository"]
            if root not in scopes:
                withheld = frozenset(row["withheld"])
                relations = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
                scopes[root] = ep.pack_index(relations, root)
            index = scopes[root]
            gathered = units(record)
            by_id = {u.id: u for u in gathered[:128]}
            confirmed = []
            mismatches = []
            for old in confirmations[cid]:
                unit = by_id.get(old["place"])
                if unit is None:
                    continue
                raw = {"file": unit.path, "code": read_ranges(index, unit.path, unit.ranges)}
                if content_hash(raw) == old["raw_entry_sha256"]:
                    confirmed.append(unit)
                else:
                    mismatches.append(unit.id)
            continuation = NAMED.continue_from(index, confirmed, box_chars=76_800)
            union = (*gathered, *continuation.units)
            results.append(
                {
                    "case": cid,
                    "confirmed": len(confirmed),
                    "source_mismatches": mismatches,
                    "hop_units": len(continuation.units),
                    "hop_items": sum(len(items_to_judge(u)) for u in continuation.units),
                    "before": sum(covers(gathered, label) for label in record["labels"]),
                    "after": sum(covers(union, label) for label in record["labels"]),
                    "local_seconds": time.monotonic() - began,
                }
            )
    (out / "confirmed-hops.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "findings": len(results),
                "confirmed": sum(r["confirmed"] for r in results),
                "source_mismatches": sum(len(r["source_mismatches"]) for r in results),
                "before": sum(r["before"] for r in results),
                "after": sum(r["after"] for r in results),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
