"""Stream #149's free census into a disk catalogue, then derive code-only ranking features.

Run with the checkout on PYTHONPATH and numpy installed. Inputs and output are explicit CLI paths;
the default proof root contains only dev110 and hard27's P/U tuning side. No provider is available.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.scope import is_test_file
from jev_navigator.index.units import Reading, UnitReader, read_ranges
from jev_navigator.judgments.questions import content_hash
from jev_navigator.selection import ScentIndex, graph_from_index, rank_features, scent_document


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("The selection replay permits zero provider calls")


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def resources(path):
    disk = shutil.disk_usage(path).free / 1024**3
    memory = subprocess.check_output(["vm_stat"], text=True)
    pages = {
        line.split(":")[0]: int(line.split(":")[1].strip().rstrip("."))
        for line in memory.splitlines()[1:]
        if ":" in line
    }
    available = (
        sum(pages.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative"))
        * 16384
        / 1024**3
    )
    if disk < 30 or available < 8:
        raise RuntimeError(
            f"Resource floor reached: {disk:.1f} GiB disk, {available:.1f} GiB free/reclaimable"
        )
    return {
        "disk_free_gib": disk,
        "memory_available_gib": available,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }


def request_path(census, dataset, cid):
    name = cid.replace(":", "_") + ".jsonl.gz"
    paths = list((census / "requests" / dataset).glob("final-after*/" + name))
    if len(paths) != 1:
        raise ValueError(f"{cid}: expected one final #149 request recording, found {len(paths)}")
    return paths[0]


def populate(db, census, cases, inputs, dataset):
    """Each body is decoded and released separately. Candidate order is actual census request order."""
    for case in cases:
        cid = case["case_id"]
        root = inputs[cid]["repository"]
        if db.execute("select 1 from cases where id=?", (cid,)).fetchone():
            continue
        started = time.monotonic()
        path = request_path(census, dataset, cid)
        seen = set()
        with gzip.open(path, "rt") as stream, db:
            for number, line in enumerate(stream):
                request = json.loads(line)
                for item in request["state"]["items"]:
                    key = content_hash(item)
                    if key in seen:
                        continue
                    seen.add(key)
                    db.execute(
                        "insert or ignore into documents values (?,?,?,?,null)",
                        (root, key, item["file"], item["code"]),
                    )
                    db.execute("insert into candidates values (?,?,?,?)", (cid, key, len(seen) - 1, number))
            db.execute("insert into cases values (?,?,?,?,?)", (cid, dataset, root, str(path), len(seen)))
        print(
            json.dumps({"collected": cid, "candidates": len(seen), "seconds": time.monotonic() - started}),
            flush=True,
        )


def bind_file(index, path):
    """One file's possible whole-unit or historical 60-line bodies, with ambiguity retained."""
    reader = UnitReader(index, 76800, listed_only=True, reading=Reading.MIXED)
    result = {}
    for unit in reader.list_files([path]).units:
        variants = [unit.ranges]
        variants.extend(
            ((first, min(first + 59, end)),)
            for start, end in unit.ranges
            for first in range(start, end + 1, 60)
        )
        for ranges in variants:
            key = content_hash({"file": path, "code": read_ranges(index, path, ranges)})
            if key in result and result[key] is not None and result[key].ranges != ranges:
                result[key] = None
            elif key not in result:
                result[key] = replace(unit, id=key, ranges=tuple(ranges), pieces=())
    return result


def bind_shown(index, path, key, code, bound):
    """Reuse exact masked-source equivalence, retaining nonunique locations as unknown."""
    if key in bound:
        return bound[key]
    candidates = []
    if "[MASKED]" in code:
        from find_eval.frozen_inputs import source_matches_masked

        prefix, suffix = code.split("[MASKED]")[0], code.split("[MASKED]")[-1]
        for unit in bound.values():
            if unit is None:
                continue
            raw = read_ranges(index, path, unit.ranges)
            # Masking replaces literal spans. Source outside the first and last markers
            # must already match. Reject impossible bodies before the expensive masker.
            if not raw.startswith(prefix) or not raw.endswith(suffix):
                continue
            if len(raw.splitlines()) == len(code.splitlines()) and source_matches_masked(raw, code, path):
                candidates.append(unit)
    else:
        # Structured text pieces can split at keys instead of historical 60-line edges.
        source = "\n".join(index.lines(path))
        start = source.find(code)
        if start >= 0 and source.find(code, start + 1) < 0:
            first = source.count("\n", 0, start) + 1
            last = first + code.count("\n")
            holders = {
                unit.id: unit
                for unit in bound.values()
                if unit is not None and any(a <= first <= last <= b for a, b in unit.ranges)
            }
            if holders:
                unit = min(holders.values(), key=lambda unit: unit.end - unit.start)
                candidates.append(replace(unit, ranges=((first, last),)))
    locations = {unit.ranges: unit for unit in candidates}
    return replace(next(iter(locations.values())), id=key) if len(locations) == 1 else None


def input_anchors(row):
    """Anchors come only from original inputs and text, never deciding labels."""
    import re

    query = row["claim"]["statement"]
    anchors = []
    for evidence in row["claim"].get("evidence", []):
        path = evidence.get("file")
        if not path:
            continue
        explicit = evidence.get("first_line") or evidence.get("start") or evidence.get("line")
        lines = evidence.get("lines")
        if lines:
            explicit = lines[0]
        mentioned = re.findall(re.escape(Path(path).name) + r":(\d+)", query)
        anchors.extend((path, int(line)) for line in mentioned)
        if explicit:
            anchors.append((path, explicit))
        if not explicit and not mentioned:
            anchors.append((path, None))
    return anchors


def canonical_source_units(units):
    """Request mask spellings are aliases, never additional physical graph nodes."""
    canonical, identities = {}, {}
    for key, unit in units.items():
        identity = content_hash({"file": unit.path, "ranges": unit.ranges})
        identities[key] = identity
        canonical.setdefault(identity, replace(unit, id=identity))
    return canonical, identities


def cached_features_current(path, row):
    """Reuse features only when their receipt records the current citation seeds."""
    receipt_path = path.with_suffix(".json")
    if not path.exists() or not receipt_path.exists():
        return False
    recorded = load(receipt_path).get("anchors")
    # Receipts use JSON lists; order and repeated citations do not change seeds.
    return recorded is not None and {tuple(anchor) for anchor in recorded} == set(input_anchors(row))


def build_features(db, cases, inputs, out, *, limit=None):
    import numpy as np

    grouped = defaultdict(list)
    for case in cases:
        grouped[inputs[case["case_id"]]["repository"]].append(case)
    timings = []
    for root, root_cases in grouped.items():
        if all(
            cached_features_current(
                out / (case["case_id"].replace(":", "_") + ".npz"), inputs[case["case_id"]]
            )
            for case in root_cases
        ):
            print(json.dumps({"resumed_complete_root": root}), flush=True)
            continue
        files = [row[0] for row in db.execute("select distinct path from documents where root=?", (root,))]
        index = CodeIndex.from_git(Path(root), ["."])
        withheld = set(inputs[root_cases[0]["case_id"]].get("withheld", []))
        if withheld.intersection(files):
            raise ValueError("Census contains a withheld input")
        # The graph never admits a withheld file, including as a binding target.
        index = CodeIndex(Path(root), [file for file in index.files if file not in withheld])
        units = {}
        documents = []
        corpus_start = time.monotonic()
        for number, path in enumerate(files):
            if path in index.files:
                aliases = [path]
            else:
                from find_eval.frozen_inputs import source_matches_masked

                aliases = [
                    original for original in index.files if source_matches_masked(original, path, original)
                ]
            sources = [(original, bind_file(index, original)) for original in aliases]
            rows = db.execute(
                "select id,code from documents where root=? and path=?", (root, path)
            ).fetchall()
            for key, code in rows:
                matches = [
                    unit
                    for original, bound in sources
                    if (unit := bind_shown(index, original, key, code, bound)) is not None
                ]
                locations = {(unit.path, unit.ranges): unit for unit in matches}
                unit = next(iter(locations.values())) if len(locations) == 1 else None
                if unit is not None:
                    units[key] = unit
                record = None if unit is None else asdict(unit)
                db.execute(
                    "update documents set binding=? where root=? and id=?", (json.dumps(record), root, key)
                )
                if unit is None:
                    documents.append(scent_document(key, path, "", code, test=is_test_file(path)))
            if number % 100 == 0:
                db.commit()
                print(
                    json.dumps({"binding_root": Path(root).name, "files": number, **resources(out)}),
                    flush=True,
                )
        db.commit()
        canonical, identities = canonical_source_units(units)
        documents.extend(
            scent_document(
                key,
                unit.path,
                unit.symbol,
                read_ranges(index, unit.path, unit.ranges),
                test=is_test_file(unit.path),
            )
            for key, unit in canonical.items()
        )
        scent = ScentIndex(documents)
        graph = graph_from_index(index, list(canonical.values()))
        graph_path = out / ("graph-" + hashlib.sha256(root.encode()).hexdigest()[:12] + ".json")
        write(
            graph_path,
            {
                "root": root,
                "adjacency": graph.adjacency,
                "edge_counts": graph.edge_counts,
                "unresolved": graph.unresolved,
            },
        )
        corpus_seconds = time.monotonic() - corpus_start
        print(
            json.dumps(
                {
                    "graph": Path(root).name,
                    "units": len(canonical),
                    "request_aliases": len(units),
                    "edges": graph.edge_counts,
                    "unresolved": graph.unresolved,
                    "seconds": corpus_seconds,
                }
            ),
            flush=True,
        )
        for case in root_cases[:limit]:
            cid = case["case_id"]
            path = out / (cid.replace(":", "_") + ".npz")
            if cached_features_current(path, inputs[cid]):
                continue
            started = time.monotonic()
            row = inputs[cid]
            ids = [
                r[0]
                for r in db.execute("select id from candidates where case_id=? order by position", (cid,))
            ]
            candidate_ids = set(ids)
            anchors = input_anchors(row)
            seeds = {
                identities[key]: 1.0
                for key, unit in units.items()
                if key in candidate_ids
                and any(
                    unit.path == file and (line is None or any(a <= line <= b for a, b in unit.ranges))
                    for file, line in anchors
                )
            }
            # File-name scent seeds are query facts, independent of reference truth.
            _, filename = scent.signals(row["claim"]["statement"])
            candidates = list(dict.fromkeys(identities.get(key, key) for key in ids))
            seed_names = sorted(
                (key for key in candidates if filename.get(key, 0)), key=lambda key: -filename[key]
            )[:16]
            seeds.update({key: 0.25 for key in seed_names if key not in seeds})
            features = rank_features(scent, row["claim"]["statement"], graph, seeds)
            vectors = np.asarray(
                [
                    [f.scent, f.walk, f.path, f.prior, f.hub, f.test]
                    for key in ids
                    for f in [features[identities.get(key, key)]]
                ],
                dtype=np.float32,
            )
            np.savez_compressed(path, ids=np.asarray(ids), features=vectors)
            labels = []
            for label in case["labels"]:
                holding = [
                    key
                    for key in ids
                    if key in units
                    and units[key].path == label["file"]
                    and any(a <= label["first_line"] <= b for a, b in units[key].ranges)
                ]
                labels.append({"file": label["file"], "line": label["first_line"], "units": holding})
            receipt = {
                "case": cid,
                "dataset": case["set_name"],
                "root": root,
                "features": str(path),
                "graph": str(graph_path),
                "identities": {key: identities[key] for key in ids if key in identities},
                "labels": labels,
                "anchors": anchors,
                "seeds": seeds,
                "unbound_candidates": sum(key not in units for key in ids),
                "seconds": time.monotonic() - started,
                "corpus_seconds": corpus_seconds,
                "resources": resources(out),
                "provider_calls": 0,
                "usd": 0,
            }
            write(path.with_suffix(".json"), receipt)
            timings.append(receipt)
            print(
                json.dumps({key: receipt[key] for key in ("case", "seconds", "unbound_candidates")}),
                flush=True,
            )
        del scent, graph, units, index, documents
    return timings


def main():
    sys.addaudithook(deny_network)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proof", type=Path, default=Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
    )
    parser.add_argument(
        "--census", type=Path, default=Path.home() / ".local/share/jvn-takeover/2026-10-03/unlock"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    resources(args.out)
    db = sqlite3.connect(args.out / "catalogue.sqlite")
    db.executescript("""
        create table if not exists documents(root text,id text,path text,code text,binding text,
                                             primary key(root,id));
        create index if not exists documents_by_file on documents(root,path);
        create table if not exists candidates(case_id text,id text,position integer,request_number integer,
                                              primary key(case_id,id));
        create table if not exists cases(id text primary key,dataset text,root text,
                                        requests_file text,n integer);
    """)
    all_cases, inputs = [], {}
    for dataset, casefile in (("dev110", "replay-dev110.json"), ("hard27", "hard27.json")):
        cases = load(args.proof / "cases" / casefile)
        if dataset == "hard27":
            cases = [
                case
                for case in cases
                if case["case_id"] in {*(f"P{i}" for i in range(1, 8)), *(f"U{i}" for i in range(1, 7))}
            ]
        if dataset == "hard27" and len(cases) != 13:
            raise ValueError("Hard27 tuning must contain exactly P1..P7 and U1..U6")
        rows = load(args.proof / f"runs/pack-49b78955/pack-inputs-{dataset}.json")["cases"]
        inputs.update({case["case_id"]: rows[case["case_id"]] for case in cases})
        populate(db, args.census, cases, inputs, dataset)
        all_cases.extend(cases)
    write(
        args.out / "input-manifest.json",
        {
            "provider_calls": 0,
            "usd": 0,
            "cases": len(all_cases),
            "files": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [args.census / "manifest.json", args.census / "final-summary.json"]
            },
        },
    )
    build_features(db, all_cases, inputs, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
