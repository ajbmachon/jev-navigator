"""Zero-provider preparation of the canonical paid-plan/#149-census union.

The saved Case 1 float32 features and finding-held-out weights are immutable inputs.
Unknown source bindings are counted, never supplied with invented ranges or bodies.
Labels are opened only after all candidate files and orders have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import Piece, Unit, items_to_judge, read_ranges
from jev_navigator.judgments.questions import content_hash

CAPS = (128, 256, 384)
TAKEOVER = Path.home() / ".local/share/jvn-takeover/2026-10-03/search-design"
SOURCE_REVISION = "7c273774cb5b0f111d9f8eb486a7144018bf9c67"
FALLBACK_POLICY = (
    "Exact census geometry uses frozen Case1 combined_cv original ordinal; outside-census plan units "
    "use their ordinal after sorting by minimum approach rank then saved candidate position. "
    "Merge these ordinals, known Case1 first on ties. No scent or walk is invented for absent features."
)


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Union preparation permits zero provider calls")


def resources():
    # Reuse the actual trial resource floor, without constructing its provider transport.
    from paid_planner import resources as check

    check()


def selected_fields(path, fields):
    """Read rank metadata without admitting a labels collection into the ranking path."""
    import ijson

    result = {}
    for field in fields:
        with path.open("rb") as stream:
            result[field] = next(ijson.items(stream, field, use_float=True))
    return result


def canonical_id(unit):
    return content_hash({"file": unit["path"], "ranges": unit["ranges"]})


def unit_record(record):
    return Unit(
        **{
            **record,
            "ranges": tuple(map(tuple, record["ranges"])),
            "pieces": tuple(Piece(**piece) for piece in record.get("pieces", [])),
        }
    )


def pin_source(checkout, expected):
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != expected:
        raise ValueError(f"Case1 revision changed: expected {expected}, found {head}")
    paths = (
        "measurements/selection/collect.py",
        "measurements/selection/evaluate.py",
        "measurements/selection/prepare_trial.py",
        "src/jev_navigator/selection/scent.py",
        "src/jev_navigator/selection/graph.py",
        "src/jev_navigator/selection/rank.py",
    )
    hashes = {}
    for name in paths:
        source = (checkout / name).read_bytes()
        committed = subprocess.check_output(["git", "-C", str(checkout), "show", f"{head}:{name}"])
        if source != committed:
            raise ValueError(f"Uncommitted Case1 source: {name}")
        hashes[name] = hashlib.sha256(source).hexdigest()
    return {"head": head, "files": hashes, "method": "unchanged saved Case1 float32 features and weights"}


def frozen_order(ids, features, weights):
    import numpy as np

    # This dtype and stable tie order are the saved evaluator's public artifact contract.
    scores = features @ np.asarray(weights, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    return order, scores


def census_candidates(db, cid, root, ids, features, weights):
    order, scores = frozen_order(ids, features, weights)
    positions = {ids[int(i)]: rank for rank, i in enumerate(order, 1)}
    vectors = {key: features[i] for i, key in enumerate(ids)}
    score_by_id = dict(zip(ids, scores.tolist(), strict=True))
    candidates, unavailable = {}, []
    records = db.execute(
        "select c.id,c.position,d.binding from candidates c join documents d on d.id=c.id "
        "and d.root=? where c.case_id=? order by c.position",
        (root, cid),
    )
    for request_id, position, binding in records:
        unit = json.loads(binding) if binding else None
        if unit is None:
            unavailable.append({"request_id": request_id, "position": position, "reason": "unknown binding"})
            continue
        key = canonical_id(unit)
        vector = vectors[request_id]
        rank = positions[request_id]
        if key in candidates:
            candidates[key]["origin"]["ranking"]["request_ids"].append(request_id)
            continue
        candidates[key] = {
            "unit": {**unit, "id": key},
            "item_unit": unit,
            "approach_ranks": [],
            "scent": float(vector[0]),
            "walk": float(vector[1]),
            "features": vector.tolist(),
            "combined_score": score_by_id[request_id],
            "rank_source": "case1_combined_cv",
            "ranking_ordinal": rank,
            "source_flags": {"plan": False, "ranking": True},
            "origin": {"ranking": {"request_ids": [request_id], "position": position, "rank": rank}},
        }
    return candidates, unavailable, positions


def merge_plan(census, plan_rows):
    """Merge exact source ranges; a second spelling never consumes another unit slot."""
    candidates = dict(census)
    plan_order = []
    for position, record in enumerate(plan_rows):
        key = canonical_id(record["unit"])
        if key not in candidates:
            candidates[key] = {
                "unit": {**record["unit"], "id": key},
                "approach_ranks": [],
                "scent": None,
                "walk": None,
                "features": None,
                "combined_score": None,
                "rank_source": "plan_ordinal_fallback",
                "source_flags": {"plan": True, "ranking": False},
                "origin": {},
            }
        candidate = candidates[key]
        ranks = sorted(set(candidate["approach_ranks"]) | set(record["approach_ranks"]))
        candidate.update(
            approach_ranks=ranks,
            source_flags={"plan": True, "ranking": candidate["source_flags"]["ranking"]},
            # The paid-plan reader owns its actual whole-unit/piece Item metadata.
            items=record["items"],
            unit={**record["unit"], "id": key},
        )
        candidate["origin"].setdefault(
            "plan", {"unit_id": record["unit"]["id"], "position": position, "scent": record.get("scent")}
        )
        if key not in plan_order:
            plan_order.append(key)
    outside = sorted(
        (key for key in plan_order if not candidates[key]["source_flags"]["ranking"]),
        key=lambda key: (
            min(candidates[key]["approach_ranks"], default=sys.maxsize),
            candidates[key]["origin"]["plan"]["position"],
        ),
    )
    for ordinal, key in enumerate(outside, 1):
        candidates[key]["ranking_ordinal"] = ordinal
    ranked = sorted(
        candidates,
        key=lambda key: (
            candidates[key]["ranking_ordinal"],
            not candidates[key]["source_flags"]["ranking"],
        ),
    )
    ranking_order = sorted(census, key=lambda key: census[key]["ranking_ordinal"])
    return candidates, {"plan_only": plan_order, "ranking_only": ranking_order, "union": ranked}


def actual_record(candidate, index):
    """Read real bodies one unit at a time; preserve the original item geometry."""
    record = {key: value for key, value in candidate.items() if key != "item_unit"}
    unit = record["unit"]
    raw = read_ranges(index, unit["path"], unit["ranges"])
    actual_hash = hashlib.sha256(raw.encode()).hexdigest()
    # Case1 historical segment bindings retain their holder's old content hash.
    # Hash the emitted geometry itself, rather than forwarding that stale metadata.
    if actual_hash != unit["content_sha256"]:
        record["origin"]["binding_content_sha256"] = unit["content_sha256"]
    if "items" in record:
        for item in record["items"]:
            if read_ranges(index, item["file"], item["ranges"]) != item["code"]:
                raise ValueError(f"Saved plan body differs from real source: {item['id']}")
    else:
        bound = unit_record(candidate["item_unit"])
        record["items"] = [
            {**asdict(item), "code": read_ranges(index, item.file, item.ranges)}
            for item in items_to_judge(bound)
        ]
    record["unit"] = {**unit, "revision": index.commit, "content_sha256": actual_hash}
    return record


def totals(records):
    items = [item for record in records for item in record["items"]]
    body_bytes = sum(len(item["code"].encode()) for item in items)
    return {
        "canonical_units": len(records),
        "items": len(items),
        "body_bytes": body_bytes,
        "estimated_body_tokens_at_4_chars": sum(len(item["code"]) for item in items) / 4,
        "token_method": "source characters / 4; estimate, not tokenizer or billed tokens",
    }


def prepare_case(db, feature_path, folder, out, index, weights):
    import numpy as np

    metadata = selected_fields(feature_path.with_suffix(".json"), ("case", "root"))
    with np.load(feature_path) as arrays:
        ids, features = arrays["ids"].tolist(), arrays["features"]
    census, unavailable, positions = census_candidates(
        db, metadata["case"], metadata["root"], ids, features, weights
    )
    plan_path = folder / "ranked-candidates.jsonl"
    if not plan_path.exists():
        raise ValueError(f"Missing paid-plan candidates: {plan_path}")
    candidates, orders = merge_plan(census, rows(plan_path))
    out.mkdir(parents=True, exist_ok=True)
    write(out / "source-identities.json", {key: candidate["origin"] for key, candidate in candidates.items()})
    write(out / "unavailable-bindings.json", unavailable)
    with (out / "source-locations.jsonl").open("w") as locations:
        for key, candidate in candidates.items():
            items = candidate.get("items")
            if items is None:
                items = [asdict(item) for item in items_to_judge(unit_record(candidate["item_unit"]))]
            locations.write(
                json.dumps(
                    {
                        "id": key,
                        "source_flags": candidate["source_flags"],
                        "items": [{"file": item["file"], "ranges": item["ranges"]} for item in items],
                    }
                )
                + "\n"
            )
    # Only materialize the declared prefixes; census discovery remains unrestricted.
    top = {}
    needed = {key for order in orders.values() for key in order[: max(CAPS)]}
    byte_count = 0
    item_count = 0
    with (out / "union-candidates.jsonl").open("w") as stream:
        for number, key in enumerate(orders["union"]):
            if number % 512 == 0:
                resources()
            if key not in needed:
                continue
            record = actual_record(candidates[key], index)
            top[key] = record
            if number < max(CAPS):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                item_count += len(record["items"])
                byte_count += sum(len(item["code"].encode()) for item in record["items"])
    curves = {
        name: {str(cap): totals([top[key] for key in order[:cap]]) for cap in CAPS}
        for name, order in orders.items()
    }
    receipt = {
        "case": metadata["case"],
        "population": "dev110" if metadata["case"].startswith("analysis-engine:") else "hard27",
        "source_root": str(index.root),
        "source_revision": index.commit,
        "weights": weights,
        "fallback_policy": FALLBACK_POLICY,
        "units": len(candidates),
        "emitted_union_units": min(max(CAPS), len(candidates)),
        "body_totals_scope": "emitted union prefix of at most 384 canonical units",
        "items": item_count,
        "body_bytes": byte_count,
        "only_plan": sum(c["source_flags"] == {"plan": True, "ranking": False} for c in candidates.values()),
        "only_ranking": sum(
            c["source_flags"] == {"plan": False, "ranking": True} for c in candidates.values()
        ),
        "both": sum(all(c["source_flags"].values()) for c in candidates.values()),
        "unavailable_census_bindings": len(unavailable),
        "source_totals": curves,
        "provider_calls": 0,
        "usd": 0,
    }
    write(out / "preparation.json", receipt)
    return receipt, orders, top, positions


def reaches(records, label):
    return any(
        item["file"] == label["file"] and any(a <= label["line"] <= b for a, b in item["ranges"])
        for record in records
        for item in record["items"]
    )


def overlap_name(plan, ranking):
    return "both" if plan and ranking else "only_plan" if plan else "only_ranking" if ranking else "neither"


def measure(prepared, labels_path, ranking_path, out, provenance_path=None):
    # Ranking and all candidate output are complete before this first reference-truth access.
    labels = defaultdict(list)
    for label in rows(labels_path):
        if label["configuration"] == "combined_cv":
            labels[label["case"]].append(label)
    published = load(ranking_path)["summary"]
    plan_provenance = {
        (row["case"], row["file"], row["first_line"]): row
        for row in (load(provenance_path) if provenance_path is not None else [])
    }
    reproduction = defaultdict(lambda: defaultdict(int))
    summary = {}
    per_line = []
    for receipt, orders, top, positions in prepared:
        cid, population = receipt["case"], receipt["population"]
        source_flags = [{"plan": False, "ranking": False} for _ in labels[cid]]
        for location in rows(out / "cases" / cid.replace(":", "_") / "source-locations.jsonl"):
            for label, flags in zip(labels[cid], source_flags, strict=True):
                if reaches([location], label):
                    for source in flags:
                        flags[source] |= location["source_flags"][source]
        for label, full_flags in zip(labels[cid], source_flags, strict=True):
            rank = (
                0
                if label["floor"]
                else min((positions[key] for key in label["units"] if key in positions), default=None)
            )
            if rank != label["rank"]:
                raise ValueError(f"Frozen Case1 rank differs for {cid} {label['file']}:{label['line']}")
            for cap in CAPS:
                reproduction[population][str(cap)] += rank is not None and rank <= cap
            flags = {
                name: reaches([top[key] for key in order[: max(CAPS)]], label)
                for name, order in orders.items()
            }
            entry = {
                "case": cid,
                "population": population,
                "file": label["file"],
                "line": label["line"],
                "floor": label["floor"],
                "frozen_case1_rank": rank,
                "prefix_reach": {},
                "source_overlap_full": overlap_name(full_flags["plan"], full_flags["ranking"]),
                "origin": {
                    "plan": plan_provenance.get((cid, label["file"], label["line"])),
                    "ranking": {
                        "witnessed": bool(full_flags["ranking"]),
                        "lineage": "frozen finding-word scent and cited-anchor/file-name graph seeds",
                        "model_convention_inference": "not used by the code ranking",
                    },
                    "pretraining_influence": "unmeasured; terms cannot establish pretraining origin",
                },
                "source_overlap_at_384": (
                    "both"
                    if flags["plan_only"] and flags["ranking_only"]
                    else "only_plan"
                    if flags["plan_only"]
                    else "only_ranking"
                    if flags["ranking_only"]
                    else "neither"
                ),
            }
            pop = summary.setdefault(
                population,
                {
                    "labels": 0,
                    "floor": 0,
                    "arms": {},
                    "overlap_at_384": defaultdict(int),
                    "overlap_full": defaultdict(int),
                },
            )
            pop["labels"] += 1
            pop["floor"] += bool(label["floor"])
            pop["overlap_at_384"][entry["source_overlap_at_384"]] += 1
            pop["overlap_full"][entry["source_overlap_full"]] += 1
            for name, order in orders.items():
                entry["prefix_reach"][name] = {}
                arm = pop["arms"].setdefault(name, {})
                for cap in CAPS:
                    hit = reaches([top[key] for key in order[:cap]], label)
                    entry["prefix_reach"][name][str(cap)] = hit
                    counts = arm.setdefault(
                        str(cap), {"actual_body_lines": 0, "including_unchanged_floor": 0}
                    )
                    counts["actual_body_lines"] += hit
                    counts["including_unchanged_floor"] += hit or bool(label["floor"])
            entry["new_union_reach_vs_plan"] = {
                str(cap): entry["prefix_reach"]["union"][str(cap)]
                and not entry["prefix_reach"]["plan_only"][str(cap)]
                for cap in CAPS
            }
            per_line.append(entry)
    for population, counts in reproduction.items():
        for cap in (128, 256):
            expected = published[population]["combined_cv"]["recall_counts"][str(cap)]
            if counts[str(cap)] != expected:
                raise ValueError(f"Published Case1 {population}@{cap}: {counts[str(cap)]} != {expected}")
    if reproduction["dev110"]["128"] != 143:
        raise ValueError("The required frozen development 143/201 baseline was not reproduced")
    result = {
        "provider_calls": 0,
        "usd": 0,
        "status": "zero-cost source reach; delivery remains unmeasured",
        "fallback_policy": FALLBACK_POLICY,
        "frozen_case1_reproduction": dict(reproduction),
        "frozen_curve_note": "Published order includes unknown bindings and unchanged floor lines.",
        "comparison_note": "Equal canonical source-unit caps; only real bodies count as actual_body_lines.",
        "summary": summary,
        "cases": [receipt for receipt, _, _, _ in prepared],
    }
    write(out / "reach-summary.json", result)
    write(out / "deciding-line-reach.json", per_line)
    return result


def supplement_plan_ranking(out, paid):
    """Add the label-free union order restricted to paid-plan candidates, without editing the union."""
    selected, receipts = {}, {}
    started = time.perf_counter()
    for folder in sorted((out / "cases").iterdir()):
        resources()
        identities = load(folder / "source-identities.json")
        candidates = {}
        for record in rows(paid / "cases" / folder.name / "ranked-candidates.jsonl"):
            key = canonical_id(record["unit"])
            candidates.setdefault(key, record)
        outside = sorted(
            (key for key in candidates if "ranking" not in identities[key]),
            key=lambda key: (
                min(candidates[key]["approach_ranks"], default=sys.maxsize),
                identities[key]["plan"]["position"],
            ),
        )
        fallback = {key: ordinal for ordinal, key in enumerate(outside, 1)}
        order = sorted(
            candidates,
            key=lambda key: (
                identities[key]["ranking"]["rank"] if "ranking" in identities[key] else fallback[key],
                "ranking" not in identities[key],
            ),
        )
        top = []
        with (folder / "plan-case1-ranked-candidates.jsonl").open("w") as stream:
            for key in order[: max(CAPS)]:
                record = {
                    **candidates[key],
                    "unit": {**candidates[key]["unit"], "id": key},
                    "rank_source": "case1_combined_cv"
                    if "ranking" in identities[key]
                    else "plan_ordinal_fallback",
                    "origin": identities[key],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                top.append(record)
        receipt = load(folder / "preparation.json")
        selected[receipt["case"]] = top
        receipts[receipt["case"]] = {str(cap): totals(top[:cap]) for cap in CAPS}
    # Reference truth is admitted only once every supplementary order has been emitted.
    result = load(out / "reach-summary.json")
    labels = load(out / "deciding-line-reach.json")
    for label in labels:
        cid, population = label["case"], label["population"]
        arm = result["summary"][population]["arms"].setdefault("plan_case1_ranked", {})
        label["prefix_reach"]["plan_case1_ranked"] = {}
        for cap in CAPS:
            hit = reaches(selected[cid][:cap], label)
            label["prefix_reach"]["plan_case1_ranked"][str(cap)] = hit
            counts = arm.setdefault(str(cap), {"actual_body_lines": 0, "including_unchanged_floor": 0})
            counts["actual_body_lines"] += hit
            counts["including_unchanged_floor"] += hit or label["floor"]
    for receipt in result["cases"]:
        receipt["source_totals"]["plan_case1_ranked"] = receipts[receipt["case"]]
    result["plan_order_note"] = (
        "plan_only retains the original paid earliest-approach/scent order. "
        "plan_case1_ranked restricts the declared frozen union order to plan-admitted units."
    )
    result["supplement_seconds"] = time.perf_counter() - started
    write(out / "reach-summary.json", result)
    write(out / "deciding-line-reach.json", labels)
    return result


def main():
    started = time.perf_counter()
    sys.addaudithook(deny_network)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=TAKEOVER / "case2")
    parser.add_argument("--paid", type=Path, default=TAKEOVER / "case2/paid-trial-20261007")
    parser.add_argument("--case1", type=Path, default=TAKEOVER / "case1")
    parser.add_argument("--case1-checkout", type=Path, default=Path.home() / "Projects/jev-navigator-case1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.out = args.out or args.paid / "union-stage-20261007"
    resources()
    args.out.mkdir(parents=True, exist_ok=True)
    source_receipt = pin_source(args.case1_checkout, SOURCE_REVISION)
    fit = selected_fields(args.case1 / "ranking.json", ("folds", "hard27_frozen_weights"))
    weights = {cid: fold["weights"] for fold in fit["folds"] for cid in fold["held_out_cases"]}
    db = sqlite3.connect(f"file:{args.case1 / 'data/catalogue.sqlite'}?mode=ro", uri=True)
    prepared = []
    current_root, index = None, None
    try:
        for feature_path in sorted((args.case1 / "data").glob("*.npz")):
            resources()
            metadata = selected_fields(feature_path.with_suffix(".json"), ("case", "root"))
            folder_name = metadata["case"].replace(":", "_")
            if metadata["root"] != current_root:
                if index is not None:
                    index.close()
                # Source geometry is already bound. No whole-repository parser output is requested.
                index = CodeIndex.from_git(Path(metadata["root"]))
                current_root = metadata["root"]
            receipt = prepare_case(
                db,
                feature_path,
                args.paid / "cases" / folder_name,
                args.out / "cases" / folder_name,
                index,
                weights.get(metadata["case"], fit["hard27_frozen_weights"]),
            )
            prepared.append(receipt)
            print(
                json.dumps({key: receipt[0][key] for key in ("case", "units", "only_plan", "both")}),
                flush=True,
            )
        result = measure(
            prepared,
            args.case1 / "deciding-ranks.jsonl",
            args.case1 / "ranking.json",
            args.out,
            args.paid / "line-provenance.json",
        )
        result = supplement_plan_ranking(args.out, args.paid)
        write(
            args.out / "manifest.json",
            {
                "case1_source": source_receipt,
                "seconds": time.perf_counter() - started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                "source": str(args.source),
                "paid": str(args.paid),
                "case1_data": str(args.case1 / "data"),
                "feature_receipts": "Saved Case1 .npz features; labels used only in final measure().",
                "frozen_ranking_sha256": hashlib.sha256(
                    (args.case1 / "ranking.json").read_bytes()
                ).hexdigest(),
                "provider_calls": 0,
                "usd": 0,
            },
        )
        print(
            json.dumps({"frozen": result["frozen_case1_reproduction"], "summary": result["summary"]}),
            flush=True,
        )
    finally:
        db.close()
        if index is not None:
            index.close()


if __name__ == "__main__":
    main()
