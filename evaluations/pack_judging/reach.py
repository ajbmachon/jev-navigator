"""Unchanged J1-3 over the corrected top-256 queue, with exact group reuse."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import paid_prepare
from paid_client import JevSession
from paid_native import source_unit
from paid_run import scopes
from paid_support import CapStopError, Ledger, resources

from jev_navigator.index.units import items_to_judge, read_ranges
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.secrets import SecretMasker, SecretScanner, mask_request, refuse_if_secret

from .consume import judge_group
from .designs import identity, jev_request
from .frozen import load, sha256
from .prepare import PLANNING, free_meta, meta_candidate, price, write
from .refresh import ADMITTED
from .run import BY_NAME, credentials, folder, records
from .run import OUT as PREVIOUS

OUT = PLANNING / "search-design/judging/reach-20261007"
DATA = PLANNING / "search-design/case1/citation-rebuild-20261007"
DESIGN = BY_NAME["J1-3"]
PREFIXES = (8, 12, 16)


def ordered_key(request):
    return hashlib.sha256(
        json.dumps({k: request[k] for k in ("state", "questions")}, ensure_ascii=False).encode()
    ).hexdigest()


def old_answers():
    """Index complete ordered requests, never individual unit or canonical-only hashes."""
    lookup = {}
    for record in records(DESIGN.name):
        base = PREVIOUS / "answers" / DESIGN.name / folder(record["case"]) / str(record["group"])
        if not (base / "observation.json").exists() or load(base / "observation.json")["failure"]:
            continue
        for receipt_path in (base / "http").glob("jev_*/receipt.json"):
            receipt = load(receipt_path)
            wire = load(receipt_path.with_name("request.json"))
            if (
                wire["state"] != record["request"]["state"]
                or wire["questions"] != record["request"]["questions"]
            ):
                continue
            raw = load(receipt_path.with_name("response.json"))
            if receipt["status"] != 200 or "usd" not in receipt or "answers" not in raw:
                continue
            lookup[ordered_key(wire)] = {
                "request": record["request"],
                "response": str(receipt_path.with_name("response.json")),
                "receipt": str(receipt_path),
                "wire": str(receipt_path.with_name("request.json")),
                "observation": str(base / "observation.json"),
            }
    return lookup


def prepare():
    if (OUT / "prepared.json").exists():
        raise ValueError("Frozen reach preparation already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").symlink_to(DATA, target_is_directory=True)
    ranking = load(DATA / "ranking.json")
    weights = ranking["hard27_frozen_weights"]
    assert weights == [0.5, 1, 0, 0, 0, -0.1]
    previous = old_answers()
    cached, totals, cases, shapes = {}, defaultdict(lambda: defaultdict(float)), [], set()
    old_questions = defaultdict(dict)
    for record in records(DESIGN.name):
        old_questions[record["case"]].update(record["request"]["questions"])
    with gzip.open(OUT / "requests.jsonl.gz", "wt", compresslevel=1) as archive:
        for original in load(ADMITTED / "frozen-cases.json"):
            resources()
            context = original["context"]
            cid, dataset = context["case"], context["dataset"]
            targets = original["groups"][0]["request"]["state"]["targets"]
            # Configure the existing ranking owner against the immutable corrected data.
            with patch.object(paid_prepare, "ROOT", OUT):
                candidates = list(paid_prepare.ranked_units(cid, weights, 256))
            assert len(candidates) == 256, (cid, len(candidates))
            relations, index = scopes(context, cached)
            items, places = [], []
            for candidate in candidates:
                for item in items_to_judge(source_unit(candidate["unit"])):
                    items.append({"file": item.file, "code": read_ranges(index, item.file, item.ranges)})
                    places.append(
                        {"id": item.id, "file": item.file, "ranges": [list(r) for r in item.ranges]}
                    )
            assert len(items) == 256, (cid, "unit pieces would change the registered group shape")
            groups = []
            for ordinal in range(16):
                baseline = {"state": {"targets": targets, "items": items[ordinal * 16 : (ordinal + 1) * 16]}}
                request = jev_request(DESIGN, baseline)
                state, questions, _ = mask_request(request["state"], request["questions"], SecretMasker())
                request.update(state=state, questions=questions)
                refuse_if_secret(state, questions, SecretScanner())
                assert all(q == old_questions[cid][key] for key, q in questions.items()), (
                    cid,
                    "question hash changed",
                )
                outside_box = JEV_INPUT_LIMITS.exceeded_by(state, questions)
                shapes.add((tuple(state), tuple(state["items"][0]), len(state["items"])))
                key = ordered_key(request)
                reuse = previous.get(key)
                if reuse is not None:
                    assert request == reuse["request"], "Model or ordered request differs"
                cost = price(request, False, {})
                record = {
                    "case": cid,
                    "dataset": dataset,
                    "group": ordinal,
                    "request": request,
                    "request_sha256": identity(request),
                    "ordered_sha256": key,
                    "price": cost,
                    "reuse": reuse,
                    "eligible": not outside_box,
                    "outside_box": str(outside_box) if outside_box else None,
                }
                archive.write(json.dumps(record, ensure_ascii=False) + "\n")
                groups.append(
                    {
                        "ordinal": ordinal,
                        "request": request,
                        "request_sha256": record["request_sha256"],
                        "eligible": not outside_box,
                        "places": {
                            str(i): p for i, p in enumerate(places[ordinal * 16 : (ordinal + 1) * 16])
                        },
                    }
                )
                for prefix in PREFIXES:
                    if ordinal < prefix:
                        total = totals[f"{dataset}:{prefix}"]
                        total["requests"] += 1
                        total["reused"] += reuse is not None and not outside_box
                        total["outside_box"] += bool(outside_box)
                        for field in ("low_usd", "high_usd", "reservation_usd"):
                            total[field] += 0 if reuse or outside_box else cost[field]
            result = {
                "context": {**context, "points": targets, "weights": weights},
                "groups": groups,
                "labels": original["labels"],
                "bindings": [c["unit"] for c in candidates],
                "candidate_ids": [c["candidate_id"] for c in candidates],
            }
            path = OUT / "cases" / (folder(cid) + ".json")
            write(path, result)
            cases.append({"case": cid, "dataset": dataset, "path": str(path), "sha256": sha256(path)})
            print(json.dumps({"prepared": cid, "groups": 16}), flush=True)
    assert len(cases) == 123
    assert len(shapes) == 1
    guards = [
        r
        for r in load(PREVIOUS / "guard-hand-inspection.json")["prototypes"]
        if r["prototype"].startswith("J1-3-")
    ]
    assert len(guards) == 2 and all(g["admitted"] for g in guards)
    write(
        OUT / "prepared.json",
        {
            "cap_usd": "2.00",
            "provider_calls": 0,
            "weights": weights,
            "weight_policy": (
                "User-specified frozen global refit, identical on all 123. No fit or tuning in this step."
            ),
            "ranking_commits": ["f624ad86", "1c13a7b"],
            "ranking_sha256": sha256(DATA / "ranking.json"),
            "groups": 1968,
            "cases": cases,
            "prices": dict(totals),
            "requests_sha256": sha256(OUT / "requests.jsonl.gz"),
            "shapes": [list(s) for s in shapes],
            "guards": [
                {k: g[k] for k in ("prototype", "report_sha256", "dimension_min_scores", "admitted")}
                for g in guards
            ],
            "guard_policy": (
                "Unchanged 16-item request shape and byte-identical questions. "
                "Reuse the already-read guards under explicit user instruction."
            ),
        },
    )


def prepare_meta():
    manifest = load(OUT / "prepared.json")
    destination = OUT / "meta/free-prepare.json"
    if destination.exists():
        result = load(destination)
        assert result["provider_calls"] == 0
        return
    prototypes = []
    for dataset in ("dev110", "hard27"):
        entry = next(e for e in manifest["cases"] if e["dataset"] == dataset)
        case = load(Path(entry["path"]))
        group = next(g for g in case["groups"] if g["eligible"])
        candidate = OUT / "meta" / f"{dataset}-candidate.json"
        write(candidate, meta_candidate(DESIGN, group["request"], dataset, group["ordinal"]))
        prepared = candidate.with_name(candidate.stem + "-prepared.json")
        result = free_meta(candidate, prepared)
        prototypes.append(
            {
                "case": entry["case"],
                "group": group["ordinal"],
                "request_sha256": identity(group["request"]),
                "candidate_sha256": sha256(candidate),
                "prepared_sha256": sha256(prepared),
                **result,
            }
        )
    write(destination, {"provider_calls": 0, "prototypes": prototypes})


def prepared_records():
    manifest = load(OUT / "prepared.json")
    assert sha256(OUT / "requests.jsonl.gz") == manifest["requests_sha256"]
    with gzip.open(OUT / "requests.jsonl.gz", "rt") as archive:
        for line in archive:
            record = json.loads(line)
            assert identity(record["request"]) == record["request_sha256"]
            assert ordered_key(record["request"]) == record["ordered_sha256"]
            yield record


def retained_overload(base, observation):
    if not observation["failure"].startswith("TypeSafeInternalServerError:"):
        return None
    paths = list((base / "http").glob("reach_*/receipt.json"))
    if not paths:
        return None
    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    receipt = load(path)
    response = load(path.with_name("response.json"))
    if receipt["status"] != 529 or response.get("detail", {}).get("error_type") != "system_overloaded":
        return None
    return receipt


def group_invoice(base, ledger):
    receipts = [load(p) for p in (base / "http").glob("reach_*/receipt.json")]
    entries = ledger.latest()
    known = sum((Decimal(r["usd"]) for r in receipts if "usd" in r), Decimal(0))
    pending = sum((Decimal(entries[r["id"]]["usd"]) for r in receipts if "usd" not in r), Decimal(0))
    return known, pending


async def run_paid():
    assert load(OUT / "engine-proof.json")["status"] == "PASS"
    assert load(OUT / "meta/free-prepare.json")["provider_calls"] == 0
    credentials()
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap="2.00")
    semaphore, halted = asyncio.Semaphore(2), asyncio.Event()

    async def attempt(record, base):
        result = base / "observation.json"
        started = time.perf_counter()
        session = JevSession(ledger, base / "http", "reach")
        try:
            if record["reuse"]:
                reuse = record["reuse"]
                wire = load(Path(reuse["wire"]))
                assert ordered_key(wire) == record["ordered_sha256"]
                assert all(wire[k] == record["request"][k] for k in ("state", "questions", "model"))
                write(
                    base / "http/exact" / (record["ordered_sha256"] + ".json"), load(Path(reuse["response"]))
                )
            rows = await judge_group(Judge(session, masker=None, max_calls=1), DESIGN, record["request"])
            if record["reuse"]:
                assert not session.receipts
                receipt = load(Path(record["reuse"]["receipt"]))
                known, pending = Decimal(receipt["usd"]), Decimal(0)
            else:
                assert len(session.receipts) == 1
                receipt = session.receipts[0]
                if receipt["status"] != 200 or "usd" not in receipt:
                    raise RuntimeError("Uncertain provider usage or failed transport stops dispatch")
                known, pending = group_invoice(base, ledger)
        except Exception as error:
            write(
                result,
                {
                    "case": record["case"],
                    "group": record["group"],
                    "request_sha256": record["request_sha256"],
                    "rows": [],
                    "failure": f"{type(error).__name__}: {error}",
                    "seconds": time.perf_counter() - started,
                },
            )
            raise
        write(
            result,
            {
                "case": record["case"],
                "group": record["group"],
                "request_sha256": record["request_sha256"],
                "rows": rows,
                "seconds": time.perf_counter() - started,
                "failure": None,
                "reused": bool(record["reuse"]),
                "receipt": receipt,
                "known_spend_usd": str(known),
                "pending_reserved_usd": str(pending),
                "cash_usd": "0" if record["reuse"] else str(known + pending),
                "configuration_usd": str(known + pending),
            },
        )

    def preserve_failure(base, observation, receipt):
        write(
            base / "failures" / (receipt["id"].replace(":", "_") + ".json"),
            {
                **observation,
                "receipt": receipt,
                "retry_reason": (
                    "Confirmed provider system_overloaded response; old reservation stays counted."
                ),
            },
        )

    async def one(record):
        base = OUT / "answers" / folder(record["case"]) / str(record["group"])
        result = base / "observation.json"
        if result.exists():
            observation = load(result)
            assert observation["request_sha256"] == record["request_sha256"]
            if not observation["failure"]:
                return
            receipt = retained_overload(base, observation)
            if receipt is None:
                raise RuntimeError("Previous failed attempt must be reconciled before resuming")
            preserve_failure(base, observation, receipt)
        if not record["eligible"]:
            write(
                result,
                {
                    "case": record["case"],
                    "group": record["group"],
                    "request_sha256": record["request_sha256"],
                    "rows": [],
                    "failure": None,
                    "skipped": "input_box",
                    "outside_box": record["outside_box"],
                    "reused": False,
                    "receipt": None,
                    "cash_usd": "0",
                    "configuration_usd": "0",
                },
            )
            return
        async with semaphore:
            while not halted.is_set():
                try:
                    await attempt(record, base)
                    return
                except Exception:
                    observation = load(result)
                    receipt = retained_overload(base, observation)
                    if receipt is None:
                        halted.set()
                        raise
                    preserve_failure(base, observation, receipt)
                    print(
                        json.dumps(
                            {
                                "overload_retry": record["case"],
                                "group": record["group"],
                                "reserved": receipt["id"],
                            }
                        ),
                        flush=True,
                    )
                    await asyncio.sleep(2)

    lower = 0
    started = time.perf_counter()
    for prefix in PREFIXES:
        work = [r for r in prepared_records() if lower <= r["group"] < prefix]
        outcomes = await asyncio.gather(*(one(record) for record in work), return_exceptions=True)
        failures = [str(value) for value in outcomes if isinstance(value, BaseException)]
        write(
            OUT / f"paid-prefix-{prefix}.json",
            {
                "prefix": prefix,
                "scheduled": len(work),
                "failures": failures,
                "seconds_cumulative": time.perf_counter() - started,
                "remaining_usd": str(ledger.balance()),
                "status": "STOP" if failures else "PASS",
            },
        )
        print(
            json.dumps({"prefix": prefix, "failures": failures, "remaining": str(ledger.balance())}),
            flush=True,
        )
        if failures:
            if any(isinstance(value, CapStopError) for value in outcomes):
                return
            raise RuntimeError("Paid dispatch halted. See retained observations and ledger.")
        lower = prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "meta", "paid"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "meta":
        prepare_meta()
    else:
        asyncio.run(run_paid())


if __name__ == "__main__":
    main()
