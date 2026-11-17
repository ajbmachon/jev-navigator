"""Join every complete-group answer, physical attempt, invoice and breadth packet."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from paid_client import JEV_RATE
from paid_support import Ledger

from jev_navigator.judgments.answers import response_from_raw

from .consume import consume_jev
from .designs import identity
from .frozen import load, sha256
from .prepare import write
from .reach import DESIGN, OUT, PREFIXES, ordered_key, prepared_records
from .run import folder


def main():
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap="2.00")
    entries = ledger.latest()
    seen_attempts, models, counts = set(), set(), defaultdict(int)
    invoices = defaultdict(Decimal)
    for record in prepared_records():
        base = OUT / "answers" / folder(record["case"]) / str(record["group"])
        observation = load(base / "observation.json")
        assert observation["request_sha256"] == record["request_sha256"]
        assert not observation["failure"]
        receipts = list((base / "http").glob("reach_*/receipt.json"))
        if not record["eligible"]:
            assert not receipts and not observation["rows"] and observation["skipped"] == "input_box"
            counts["input_box_exclusions"] += 1
            continue
        counts["successful_groups"] += 1
        successful = []
        for path in receipts:
            receipt = load(path)
            wire_path, raw_path = path.with_name("request.json"), path.with_name("response.json")
            wire = load(wire_path)
            raw = load(raw_path) if raw_path.exists() else None
            assert ordered_key(wire) == record["ordered_sha256"]
            assert identity(wire) == record["request_sha256"]
            assert sha256(wire_path) == receipt["request_sha256"]
            assert len(wire["state"]["items"]) == 16
            event = entries[receipt["id"]]
            assert receipt["id"] not in seen_attempts
            seen_attempts.add(receipt["id"])
            counts[f"status:{receipt['status']}"] += 1
            if receipt["status"] == 200:
                assert event["status"] == "settled"
                assert Decimal(event["usd"]) == Decimal(raw["usage"]["input_tokens"]) * JEV_RATE
                models.add(raw["model"])
                successful.append((receipt, raw))
            else:
                if receipt["status"] == 529:
                    assert raw["detail"]["error_type"] == "system_overloaded"
                else:
                    assert receipt["status"] is None
                    assert receipt["failure"].startswith("TypeSafeAPIConnectionError:")
                    assert raw is None
                assert event["status"] == "reserved"
                assert (base / "failures" / (receipt["id"].replace(":", "_") + ".json")).exists()
            for prefix in PREFIXES:
                if record["group"] < prefix:
                    invoices[f"{record['dataset']}:{prefix}:configuration_usd"] += Decimal(event["usd"])
        assert len(successful) == 1
        receipt, raw = successful[0]
        expected = consume_jev(DESIGN, record["request"], response_from_raw(raw))
        assert len(expected) == len(observation["rows"]) == len(record["request"]["questions"])
        for actual, wanted in zip(observation["rows"], expected, strict=True):
            assert all(actual[key] == wanted[key] for key in ("point", "slot", "score", "raw_question_id"))
        assert observation["receipt"]["id"] == receipt["id"]
    assert seen_attempts == set(entries)
    summary = load(OUT / "summary.json")
    assert summary["packing_provider_calls"] == 0 and summary["rows"] == 369
    assert summary["literal_packets"] == 1107
    for key, amount in invoices.items():
        assert Decimal(summary["costs"][key]) == amount, (key, amount, summary["costs"][key])
    assert sum(Decimal(e["usd"]) for e in entries.values()) == Decimal(summary["step_counted_usd"])
    for packet in load(OUT / "packet-hashes.json"):
        path = OUT / "packs" / folder(packet["case"]) / str(packet["prefix"]) / f"native-{packet['room']}.txt"
        assert sha256(path) == packet["sha256"]
    results = load(OUT / "results.json")
    assert len({(row["case"], row["prefix"]) for row in results}) == 369
    lookup = {(row["case"], row["prefix"]): row for row in results}
    increments, visibility = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for row in results:
        for room, measures in row["rooms"].items():
            key = f"{row['dataset']}:{row['prefix']}:{room}"
            for seen, delivered in zip(row["seen"], measures["native"], strict=True):
                visibility[key]["seen_delivered"] += seen and delivered
                visibility[key]["seen_not_delivered"] += seen and not delivered
                visibility[key]["unseen_delivered"] += not seen and delivered
                visibility[key]["unseen_not_delivered"] += not seen and not delivered
        if row["prefix"] == 8:
            continue
        before = lookup[row["case"], row["prefix"] - 4]
        key = f"{row['dataset']}:{row['prefix'] - 4}->{row['prefix']}"
        for room, measures in row["rooms"].items():
            for path in ("lab", "native"):
                for earlier, later in zip(before["rooms"][room][path], measures[path], strict=True):
                    increments[key][f"{path}:{room}:gained"] += not earlier and later
                    increments[key][f"{path}:{room}:lost"] += earlier and not later
    write(
        OUT / "audit.json",
        {
            "status": "PASS",
            "counts": dict(counts),
            "all_physical_attempts_reserved_and_joined": True,
            "served_models": sorted(models),
            "packing_provider_calls": 0,
            "packet_identities_verified": 1107,
            "increments": {k: dict(v) for k, v in increments.items()},
            "visibility": {k: dict(v) for k, v in visibility.items()},
            "request_archive_sha256": sha256(OUT / "requests.jsonl.gz"),
            "spend_ledger_sha256": sha256(OUT / "spend-ledger.jsonl"),
        },
    )
    print("Audit PASS: exact groups, all attempts, every invoice, 1,107 packet identities")


if __name__ == "__main__":
    main()
