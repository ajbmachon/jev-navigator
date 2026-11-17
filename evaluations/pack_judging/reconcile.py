"""Settle failed transport reservations from retained read-only provider usage records."""

from __future__ import annotations

import gzip
from datetime import datetime
from decimal import Decimal

from paid_support import Ledger

from .frozen import load, sha256
from .prepare import write
from .repack import FOLLOWUP, offline_owner
from .run import OUT


def usage_record(path):
    values = list(load(path)["usage"].values())
    assert len(values) == 1
    value = values[0]
    assert len(value["grouped_data"]) == 1
    group = value["grouped_data"][0]
    assert group["group_by_values"] == {
        "model_requested": "deepseek-v4-flash-0731",
        "model_used": "deepseek-v4-flash-0731",
        "provider_used": "sference",
    }
    for name in ("total_requests", "input_tokens", "output_tokens", "total_tokens", "spend"):
        assert group[name] == value[name]
    return value


def run():
    offline_owner()
    billing = FOLLOWUP / "billing"
    assert not (billing / "reconciliation.json").exists(), "Already reconciled"
    failed_usage = usage_record(billing / "failed-window-usage.json")
    assert failed_usage["completions_requests"] == failed_usage["total_requests"] == 98
    assert Decimal(failed_usage["spend"]) == 0
    assert all(failed_usage[name] == 0 for name in ("input_tokens", "output_tokens", "total_tokens"))
    control = load(OUT / "canary-typed-json-object/response.json")
    paid_usage = usage_record(billing / "paid-control-usage.json")
    assert paid_usage["completions_requests"] == paid_usage["total_requests"] == 1
    assert Decimal(paid_usage["spend"]) == Decimal(str(control["usage"]["cost"]))
    assert paid_usage["input_tokens"] == control["usage"]["prompt_tokens"]
    assert paid_usage["output_tokens"] == control["usage"]["completion_tokens"]
    window = load(billing / "failed-window-usage-request.json")["body"]
    start, end = (datetime.fromisoformat(window[name]).timestamp() for name in ("start", "end"))
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap="7.00")
    latest = ledger.latest()
    audit = load(OUT / "failed-transport-reservation-audit.json")["rows"]
    rows, response_ids, reservations = [], set(), set()
    for entry in audit:
        assert len(entry["request_paths"]) == len(entry["ledger_ids"]) == 1
        request = OUT / entry["request_paths"][0]
        identity = entry["ledger_ids"][0]
        assert identity not in reservations
        reservations.add(identity)
        assert latest[identity]["status"] == "reserved"
        assert latest[identity]["usd"] == entry["reservation_usd"]
        response = request.with_name("response.json")
        row = {
            "id": identity,
            "request": str(request.relative_to(OUT)),
            "request_file_sha256": sha256(request),
            "reserved_usd": entry["reservation_usd"],
            "billed_usd": "0",
        }
        if response.exists():
            raw = load(response)
            assert raw["id"] not in response_ids
            response_ids.add(raw["id"])
            assert start <= raw["created"] < end
            assert not raw.get("usage")
            assert not raw["choices"][0]["message"].get("content")
            row.update(
                evidence="zero-cost aggregate provider usage window",
                provider_response_id=raw["id"],
                response_file_sha256=sha256(response),
                provider_created=raw["created"],
            )
        else:
            observation = request.with_name("observation.json")
            assert "nodename nor servname provided" in load(observation)["failure"]
            row.update(
                evidence="DNS resolution failed before HTTP dispatch", observation_sha256=sha256(observation)
            )
        rows.append(row)
    assert len(rows) == 99 and len(response_ids) == 98
    assert reservations == {key for key, value in latest.items() if value["status"] == "reserved"}
    released = sum((Decimal(row["reserved_usd"]) for row in rows), Decimal(0))
    assert released == Decimal("0.914999400")
    before_hash = sha256(ledger.path)
    (billing / "ledger-before.jsonl.gz").write_bytes(gzip.compress(ledger.path.read_bytes(), mtime=0))
    usage_hash = sha256(billing / "failed-window-usage.json")
    for row in rows:
        ledger.settle(
            row["id"],
            "0",
            reconciliation="engine-fixed-20261007/billing/reconciliation.json",
            reconciliation_evidence=row["evidence"],
            provider_usage_sha256=usage_hash,
        )
    assert all(value["status"] == "settled" for value in ledger.latest().values())
    total = ledger.cap - ledger.balance()
    assert total == Decimal("4.395335798")
    write(
        billing / "reconciliation.json",
        {
            "status": "SETTLED",
            "provider_inference_calls": 0,
            "provider_management_reads": 3,
            "reconciliation_inference_usd": "0",
            "usage_evidence_granularity": "aggregate narrow time window, not per-request provider logs",
            "provider_zero_billed_responses": 98,
            "dns_failures_before_dispatch": 1,
            "positive_control_usd": paid_usage["spend"],
            "released_reservations_usd": str(released),
            "settled_study_usd": str(total),
            "remaining_reserved_usd": "0",
            "cap_usd": str(ledger.cap),
            "ledger_before_sha256": before_hash,
            "ledger_after_sha256": sha256(ledger.path),
            "provider_usage_sha256": usage_hash,
            "positive_control_usage_sha256": sha256(billing / "paid-control-usage.json"),
            "rows": rows,
        },
    )
    print(f"Released ${released}; settled ${total}; no uncertain reservations remain")


if __name__ == "__main__":
    run()
