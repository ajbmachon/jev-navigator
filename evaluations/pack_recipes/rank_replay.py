"""Named v2 combined-rank configurations over the exact frozen v1 recipe reach."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from measure import BASE, ROLES, covers, load, requests, source_seeds
from positions import request_rank
from ranking import RecipeRanker
from resources import check_resources
from summarize import units


def main(dataset, source, out):
    from enginepy.workflows.document_analysis import evidence_pack as ep
    from enginepy.workflows.document_analysis.code_relations import CodeRelations
    from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps

    inputs = load(BASE / f"runs/pack-49b78955/pack-inputs-{dataset}.json")["cases"]
    templates = load(ROLES / "step-3/roles16-candidate.json")["request"]["questions"]
    scopes = {}
    records = []
    out.mkdir(parents=True, exist_ok=True)
    with (
        (source / f"{dataset}-candidates.jsonl").open() as inputs_file,
        (out / f"{dataset}-candidates.jsonl").open("w") as output,
    ):
        for line in inputs_file:
            check_resources()
            record = json.loads(line)
            cid = record["case"]
            row = inputs[cid]
            root = row["repository"]
            if root not in scopes:
                withheld = frozenset(row["withheld"])
                relations = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
                scopes[root] = relations, ep.pack_index(relations, root)
            relations, index = scopes[root]
            pack = ep.pack_request(relations, root, row["claim"], row.get("points"))
            seeds = source_seeds(index, row, pack)
            admitted = units(record)
            # Replay the exact gathered bodies, never silently rank changed source.
            for unit in admitted:
                if hashlib.sha256(read_source(index, unit).encode()).hexdigest() != unit.content_sha256:
                    raise ValueError(f"Source changed for {cid} {unit.id}")
            began = time.monotonic()
            ranked = RecipeRanker(index, row["claim"]["statement"], seeds.units)(admitted)
            rank_seconds = time.monotonic() - began
            from dataclasses import asdict

            record.update(
                version=2,
                configuration=f"{record['recipe']}-v2-combined",
                ranker="case1-combined-default-v1, cochange=0",
                rank_seconds=rank_seconds,
                local_seconds=record["local_seconds"] + rank_seconds,
                units=[asdict(unit) for unit in ranked],
            )
            for label in record["labels"]:
                positions = [i for i, unit in enumerate(ranked, 1) if covers((unit,), label)]
                label.update(
                    unit_rank=min(positions, default=None),
                    request_rank=request_rank(ranked, label["file"], label["first_line"]),
                )
            prepared = out / "requests" / dataset / record["recipe"] / f"{cid.replace(':', '_')}.jsonl.gz"
            prepared.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(prepared, "wt") as handle:
                for group in requests(index, ranked, row["claim"]["statement"], templates):
                    handle.write(json.dumps(group, ensure_ascii=False) + "\n")
            record.update(
                requests_path=str(prepared), requests_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest()
            )
            output.write(json.dumps(record) + "\n")
            output.flush()
            records.append({k: v for k, v in record.items() if k not in {"units", "unresolved"}})
            print(
                json.dumps({"case": cid, "recipe": record["recipe"], "rank_seconds": round(rank_seconds, 3)}),
                flush=True,
            )
    summary = load(source / f"{dataset}-summary.json")
    summary.update(records=records, ranking_source="case1 5fa412eb and a621da93", resources=check_resources())
    (out / f"{dataset}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def read_source(index, unit):
    from jev_navigator.index.units import read_ranges

    return read_ranges(index, unit.path, unit.ranges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("dev110", "hard27"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.source, args.out)
