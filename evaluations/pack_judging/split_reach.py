"""Judge only the 259 refused reach groups through the pinned main Judge."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from jev_navigator.judgments.profiles import J1
from paid_client import JEV_RATE
from paid_support import CapStopError, Ledger, load, resources, write

from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import request_sha256

from .reach_transport import (
    RetainedJevSession,
    group_invoice,
    restore_unreceipted_failure,
    retained_retryable_failure,
)

PLANNING = Path.home() / ".local/share/jvn-takeover/2026-10-03"
REACH = PLANNING / "search-design/judging/reach-20261007"
OUT = PLANNING / "search-design/judging/split-reach-20261007"
MAIN = "7d1686ee"
CAP = "0.40"


def records():
    manifest = load(REACH / "prepared.json")
    assert digest(REACH / "requests.jsonl.gz") == manifest["requests_sha256"]
    with gzip.open(REACH / "requests.jsonl.gz", "rt") as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_folder(record):
    return OUT / "answers" / record["case"].replace(":", "_") / str(record["group"])


def frozen_plan(judge, request, refusals):
    """Start from the original group; main owns admission, halves and slot rebinding.

    The public planner normally closes groups greedily before they exceed its box.
    This follow-up starts with an already frozen, refused sixteen-item group. Use
    its existing plan and batch builder, then its ordinary refusal/splitting waves.
    No request sizing, question rewriting or split algorithm is reimplemented.
    """
    targets = request["state"]["targets"]
    checks = [check for check, _, _ in J1.asked(targets).values()]
    plan = judge._check_plan(
        checks,
        request["state"]["items"],
        {"targets": targets},
        "items",
        judge.thresholds,
        refusals=refusals,
        keep_order=True,
    )
    original = judge._batch(plan, list(range(len(plan.items))))
    assert original.state == request["state"]
    assert original.questions == request["questions"]
    plan.batches = [original]
    return plan


def measured_leaves(judge, plan, batch):
    outside = judge.input_limits.exceeded_by(batch.state, batch.questions)
    if not outside:
        yield batch, False
        return
    halves = judge._halves(plan, batch)
    if halves is None:
        yield batch, True
    else:
        for half in halves:
            yield from measured_leaves(judge, plan, half)


class NoSend:
    model = "jev-latest"

    def ask(self, *args, **kwargs):
        raise AssertionError("Preparation attempted a provider call")


def prepare():
    destination = OUT / "prepared.json"
    if destination.exists():
        return
    counts, prices, groups = Counter(), Counter(), []
    with gzip.open(OUT / "requests.jsonl.gz", "wt") as stream:
        for record in records():
            if record["eligible"]:
                counts["unchanged_groups_reused"] += 1
                continue
            judge = Judge(NoSend(), masker=None, max_calls=0)
            plan = frozen_plan(judge, record["request"], [])
            leaves = []
            for batch, refused in measured_leaves(judge, plan, plan.batches[0]):
                request = {"model": "jev-latest", "state": batch.state, "questions": batch.questions}
                wire_bytes = len(json.dumps(request, ensure_ascii=False).encode())
                for qid, question in batch.questions.items():
                    assert question == record["request"]["questions"][qid]
                leaf = {
                    "members": list(batch.members),
                    "request": request,
                    "request_sha256": request_sha256(batch.state, batch.questions),
                    "singleton_refused": refused,
                    "reservation_usd": str(Decimal(wire_bytes + 1024) * JEV_RATE),
                    "forecast_low_usd": str(Decimal(wire_bytes) * Decimal("0.23") * JEV_RATE),
                    "forecast_high_usd": str((Decimal(wire_bytes) * Decimal("0.30") + 254) * JEV_RATE),
                }
                stream.write(
                    json.dumps({"case": record["case"], "group": record["group"], **leaf}, ensure_ascii=False)
                    + "\n"
                )
                leaves.append({k: v for k, v in leaf.items() if k != "request"})
                counts["refused_singletons" if refused else "fresh_requests"] += 1
                if not refused:
                    for key in ("reservation_usd", "forecast_low_usd", "forecast_high_usd"):
                        prices[key] += Decimal(leaf[key])
            groups.append(
                {
                    "case": record["case"],
                    "dataset": record["dataset"],
                    "group": record["group"],
                    "leaves": leaves,
                }
            )
    assert len(groups) == 259 and counts["unchanged_groups_reused"] == 1709
    write(
        destination,
        {
            "main": MAIN,
            "judge_sha256": digest(
                Path(__import__("jev_navigator.judgments.judge", fromlist=["Judge"]).__file__)
            ),
            "cap_usd": CAP,
            "provider_calls": 0,
            "counts": dict(counts),
            "prices": {k: str(v) for k, v in prices.items()},
            "groups": groups,
            "reach_requests_sha256": digest(REACH / "requests.jsonl.gz"),
            "split_requests_sha256": digest(OUT / "requests.jsonl.gz"),
            "guards": load(REACH / "prepared.json")["guards"],
            "guard_policy": (
                "Existing admitted request-shape guards cover item splitting under explicit user instruction."
            ),
            "question_identity": J1.identity(),
        },
    )
    print(json.dumps(load(destination)["counts"]), flush=True)
    print(json.dumps(load(destination)["prices"]), flush=True)


def credentials():
    for path in (Path.home() / "Projects/analysis-engine/.env", Path.home() / ".config/jgrep/env"):
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in {"TYPESAFE_API_KEY", "TYPESAFE_BASE_URL"}:
                os.environ.setdefault(key, value.strip().strip("\"'"))


async def paid_group(record, ledger):
    base = group_folder(record)
    result = base / "observation.json"
    if result.exists() and not load(result)["failure"]:
        assert load(result)["request_sha256"] == record["request_sha256"]
        return
    if result.exists():
        previous = load(result)
        assert previous["request_sha256"] == record["request_sha256"]
        restore_unreceipted_failure(base, previous, ledger)
        receipt = retained_retryable_failure(base, previous)
        if receipt is None:
            raise RuntimeError("Previous failed attempt needs reconciliation before resuming")
        write(
            base / "failures" / (receipt["id"].replace(":", "_") + ".json"), {**previous, "receipt": receipt}
        )
    session = RetainedJevSession(ledger, base / "http", "reach")
    while True:
        started, rows, refusals = time.perf_counter(), [], []
        judge = Judge(session, masker=None, max_concurrency=1)
        plan = frozen_plan(judge, record["request"], refusals)
        try:
            async for wave in judge._settled_waves(plan):
                for batch in wave:
                    for qid, original_slot in batch.slots.items():
                        check = next(c for c in plan.checks if qid.split("#")[0] == c.question_id)
                        answer = plan.answered[(original_slot, qid)]
                        point = next(
                            p for p in record["request"]["state"]["targets"] if check.name == f"match_{p}"
                        )
                        rows.append(
                            {
                                "point": point,
                                "slot": original_slot,
                                "score": answer.probability,
                                "score_kind": "noul_yes_probability",
                                "raw_question_id": qid,
                                "answer_request_sha256": answer.request_sha256,
                            }
                        )
            known, pending = group_invoice(base, ledger)
            write(
                result,
                {
                    "case": record["case"],
                    "group": record["group"],
                    "request_sha256": record["request_sha256"],
                    "rows": rows,
                    "failure": None,
                    "reused": False,
                    "receipt": None,
                    "refused_slots": [
                        position
                        for position, item in enumerate(plan.items)
                        if any(item is r.item for r in refusals)
                    ],
                    "seconds": time.perf_counter() - started,
                    "known_spend_usd": str(known),
                    "pending_reserved_usd": str(pending),
                    "cash_usd": str(known + pending),
                    "configuration_usd": str(known + pending),
                },
            )
            return
        except Exception as error:
            failure = {
                "case": record["case"],
                "group": record["group"],
                "request_sha256": record["request_sha256"],
                "rows": rows,
                "failure": f"{type(error).__name__}: {error}",
                "seconds": time.perf_counter() - started,
            }
            write(result, failure)
            receipt = retained_retryable_failure(base, failure)
            if receipt is None or isinstance(error, CapStopError):
                raise
            write(
                base / "failures" / (receipt["id"].replace(":", "_") + ".json"),
                {**failure, "receipt": receipt},
            )
            print(json.dumps({"retry": receipt["id"], "reserved_stays_counted": True}), flush=True)
            await asyncio.sleep(2)


async def paid():
    prepared = load(OUT / "prepared.json")
    assert prepared["split_requests_sha256"] == digest(OUT / "requests.jsonl.gz")
    assert prepared["judge_sha256"] == digest(
        Path(__import__("jev_navigator.judgments.judge", fromlist=["Judge"]).__file__)
    )
    credentials()
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap=CAP)
    completed = 0
    try:
        for record in sorted(
            (r for r in records() if not r["eligible"]), key=lambda r: (r["group"], r["case"])
        ):
            resources()
            await paid_group(record, ledger)
            completed += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "case": record["case"],
                        "group": record["group"],
                        "remaining_usd": str(ledger.balance()),
                    }
                ),
                flush=True,
            )
    finally:
        write(
            OUT / "paid-status.json",
            {
                "completed": completed,
                "status": "PASS" if completed == 259 else "STOP",
                "remaining_usd": str(ledger.balance()),
                "physical_attempts": len(ledger.latest()),
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "paid"))
    stage = parser.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    if stage == "prepare":
        prepare()
    else:
        asyncio.run(paid())


if __name__ == "__main__":
    main()
