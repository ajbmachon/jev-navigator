"""Judge a ranked union once, with durable spend and complete checkpoint rounds."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from paid_planner import resources
from paid_shape import entries, prepare
from spend import SpendLedger

from jev_navigator.judgments.answers import reported_input_tokens, response_from_raw
from jev_navigator.judgments.questions import content_hash
from jev_navigator.judgments.secrets import SecretScanner, refuse_if_secret

RATE = Decimal("0.000000042")
ROOT = Path.home() / ".local/share/jvn-takeover/2026-10-03/search-design/case2"
PAID = ROOT / "paid-trial-20261007"
OUT = PAID / "union-stage-20261007"
PROFILE = Path.home() / "Projects/jev-navigator-role-profile/src/jev_navigator/judgments/profiles.py"


def rows(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def reserve_price(request):
    # UTF-8 bytes are a conservative token bound. The documented request limit
    # bounds any successful evaluation at 64k tokens. Include framing overhead.
    size = len(json.dumps(request, ensure_ascii=False).encode()) + 1024
    return RATE * min(size, 64000)


def profile():
    spec = importlib.util.spec_from_file_location("jev_navigator.judgments.profiles", PROFILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ROLES_V2


def prepare_union():
    role_profile = profile()
    summaries = []
    for folder in sorted((OUT / "cases").iterdir()):
        resources()
        context = json.loads((ROOT / "cases" / folder.name / "planner-context.json").read_text())
        source = folder / "union-candidates.jsonl"
        item_info = {}
        for row in rows(source):
            for item in row["items"]:
                item_info[item["id"]] = {
                    "unit_id": row["unit"]["id"],
                    "source_flags": row.get("source_flags", row.get("sources", {})),
                }
        requests, members, refusals = prepare(
            role_profile.questions("p0"),
            itertools.islice(entries(source), 384),
            {"targets": {"p0": context["finding"]}},
        )
        records = []
        for ordinal, (request, group) in enumerate(zip(requests[:24], members[:24], strict=True), 1):
            records.append(
                {
                    "ordinal": ordinal,
                    "request": request,
                    "members": [{**member, **item_info[member["id"]]} for member in group],
                    "request_sha256": content_hash(
                        {"state": request["state"], "questions": request["questions"]}
                    ),
                }
            )
        path = folder / "prepared-requests.jsonl"
        with path.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        summaries.append(
            {
                "folder": folder.name,
                "requests": len(records),
                "items": sum(len(r["members"]) for r in records),
                "refusals": [str(r) for r in refusals],
                "reservation_usd": str(sum((reserve_price(r["request"]) for r in records), Decimal(0))),
            }
        )
        if not (OUT / "guard-candidate.json").exists():
            full = next((r for r in records if len(r["members"]) == 16), None)
            if full:
                write(
                    OUT / "guard-candidate.json",
                    {
                        "case_id": folder.name,
                        "revision_id": "case2-union-ranked-six-role-16-v1",
                        "request": full["request"],
                        "intended_uses": {
                            name: (
                                "Retain the raw role probability. Code ranks by the maximum of "
                                "decide, guard, value, effect and satisfied, and retains the best "
                                "unit per role. Delegates selects forwarding only. This is "
                                "candidate relevance, not a finding verdict."
                            )
                            for name in full["request"]["questions"]
                        },
                    },
                )
    write(OUT / "shape-summary.json", summaries)
    print(
        "Prepared",
        len(summaries),
        "findings",
        sum(r["requests"] for r in summaries),
        "physical requests",
        flush=True,
    )


def guard(ledger):
    identifier = "union-guard-v1"
    if any(e["id"] == identifier and e["status"] == "settled" for e in ledger.events):
        return
    prepared = OUT / "guard-prepared.json"
    with prepared.open("w") as stream:
        subprocess.run(
            ["system-one-meta-builder", "prepare", str(OUT / "guard-candidate.json")],
            stdout=stream,
            check=True,
        )
    requests = json.loads(prepared.read_text())["requests"]
    quote = sum((reserve_price(r["submitted"]) for r in requests), Decimal(0))
    ledger.reserve(identifier, "meta", str(quote))
    started = time.perf_counter()
    with (OUT / "guard-report.json").open("w") as stream:
        result = subprocess.run(
            [
                "system-one-meta-builder",
                "guard",
                str(OUT / "guard-candidate.json"),
                "--output",
                str(OUT / "guard-receipts.jsonl"),
            ],
            stdout=stream,
            check=False,
        )
    receipts = [r for r in rows(OUT / "guard-receipts.jsonl") if r.get("status") == "ok"]
    tokens = sum(reported_input_tokens(r.get("response", {})) or 0 for r in receipts)
    if not receipts or not tokens:
        raise RuntimeError("Guard usage is unresolved; its reservation remains charged")
    ledger.settle(
        identifier,
        str(RATE * tokens),
        input_tokens=tokens,
        calls=len(receipts),
        seconds=time.perf_counter() - started,
    )
    report = json.loads((OUT / "guard-report.json").read_text())
    print(
        "Union guard",
        result.returncode,
        "usd",
        RATE * tokens,
        "remaining",
        ledger.balance(),
        "route",
        report.get("route"),
        flush=True,
    )
    if result.returncode not in (0, 3):
        raise RuntimeError("Guard failed; retain receipts and stop dispatch")


def send(ledger, folder, record):
    import httpx2
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    from jev_navigator.adapters.typesafe import CapturingTransport

    request = record["request"]
    digest = content_hash({"state": request["state"], "questions": request["questions"]})
    if record["request_sha256"] != digest:
        raise ValueError("Prepared request changed before dispatch")
    refuse_if_secret(request["state"], request["questions"], SecretScanner())
    identifier = f"union:{folder.name}:{record['ordinal']}"
    ledger.reserve(identifier, "jev", str(reserve_price(request)))
    started = time.perf_counter()
    # Retries are explicitly disabled: every physical send must have a reservation.
    capture = CapturingTransport(httpx2.HTTPTransport())

    def retain_attempt(attempt):
        append(
            folder / "transport-attempts.jsonl",
            {
                "ordinal": record["ordinal"],
                "request_sha256": record["request_sha256"],
                "sent_body_base64": base64.b64encode(attempt.sent_body).decode(),
                "body_base64": base64.b64encode(attempt.response.body).decode() if attempt.response else None,
                "status": attempt.response.status if attempt.response else None,
                "content_type": attempt.response.content_type if attempt.response else None,
                "duration_ms": attempt.duration_ms,
            },
        )

    error = None
    with capture.collecting(retain_attempt) as collection:
        try:
            with TypeSafeClient(
                model=request["model"], retry=RetryPolicy(max_retries=0), transport=capture
            ) as sdk:
                sdk.system_one(request["state"], request["questions"])
        except Exception as caught:
            error = caught
    try:
        raw = json.loads(collection.attempts[-1].response.body)
    except (IndexError, AttributeError, ValueError):
        raw = None
    if (raw is None or reported_input_tokens(raw) is None) and error is None:
        error = RuntimeError("Provider usage is unresolved; reservation remains charged")
    if error is not None:
        append(
            folder / "failures.jsonl",
            {
                "ordinal": record["ordinal"],
                "request_sha256": record["request_sha256"],
                "error": type(error).__name__,
                "message": str(error),
                "seconds": time.perf_counter() - started,
            },
        )
        if raw is None or reported_input_tokens(raw) is None:
            raise error
    tokens = reported_input_tokens(raw)
    receipt = {
        "ordinal": record["ordinal"],
        "request_sha256": record["request_sha256"],
        "response": raw,
        "usage": raw.get("usage"),
        "usd": str(RATE * tokens) if tokens is not None else None,
        "seconds": time.perf_counter() - started,
    }
    append(folder / "responses.jsonl", receipt)
    if tokens is None:
        raise RuntimeError("Provider did not report usage; reservation remains charged")
    ledger.settle(
        identifier,
        receipt["usd"],
        input_tokens=tokens,
        seconds=receipt["seconds"],
        served_model=raw.get("model"),
    )
    if error is not None:
        raise error
    parsed = response_from_raw(raw)
    missing = set(request["questions"]) - set(parsed.answers)
    if missing:
        raise RuntimeError(f"Provider omitted {len(missing)} answers; charged receipt retained")
    return tokens, receipt["seconds"]


def dispatch(ledger):
    queues = {
        folder: sum(1 for _ in rows(folder / "prepared-requests.jsonl"))
        for folder in sorted((OUT / "cases").iterdir())
    }
    complete = {folder: {r["ordinal"] for r in rows(folder / "responses.jsonl")} for folder in queues}
    unresolved = [e["id"] for e in {e["id"]: e for e in ledger.events}.values() if e["status"] == "reserved"]
    if unresolved:
        raise RuntimeError(f"Reconcile {len(unresolved)} reserved attempts before another send")
    started = time.perf_counter()
    stop = None
    # A barrier between checkpoints keeps the lower curve complete for all cases.
    for checkpoint in (4, 8, 16, 24):
        pending = [
            (folder, ordinal)
            for ordinal in range(1, checkpoint + 1)
            for folder, count in queues.items()
            if ordinal <= count and ordinal not in complete[folder]
        ]
        if not pending:
            continue
        print(
            "Starting checkpoint",
            checkpoint,
            "pending",
            len(pending),
            "remaining",
            ledger.balance(),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=3) as pool:
            for batch_start in range(0, len(pending), 3):
                resources()
                batch = [
                    (
                        folder,
                        next(r for r in rows(folder / "prepared-requests.jsonl") if r["ordinal"] == ordinal),
                    )
                    for folder, ordinal in pending[batch_start : batch_start + 3]
                ]
                projected = sum((reserve_price(r["request"]) for _, r in batch), Decimal(0))
                if projected > ledger.balance():
                    stop = f"cap reserve stop at checkpoint {checkpoint}"
                    break
                futures = [
                    (folder, record, pool.submit(send, ledger, folder, record)) for folder, record in batch
                ]
                for folder, record, future in futures:
                    future.result()
                    complete[folder].add(record["ordinal"])
                if batch_start % 30 == 0:
                    print(
                        "Checkpoint",
                        checkpoint,
                        "done",
                        sum(len(s) for s in complete.values()),
                        "remaining",
                        ledger.balance(),
                        flush=True,
                    )
        write(
            OUT / "dispatch-summary.json",
            {
                "cap_usd": str(ledger.cap),
                "remaining_usd": str(ledger.balance()),
                "seconds": time.perf_counter() - started,
                "stop": stop,
                "per_finding": {
                    folder.name: {"answered": sorted(complete[folder]), "prepared": count}
                    for folder, count in queues.items()
                },
            },
        )
        if stop:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "guard", "judge"))
    args = parser.parse_args()
    resources()
    ledger = SpendLedger(PAID / "spend-ledger.jsonl", cap="3.00")
    if args.phase == "prepare":
        prepare_union()
    elif args.phase == "guard":
        guard(ledger)
    else:
        guard(ledger)
        dispatch(ledger)


if __name__ == "__main__":
    main()
