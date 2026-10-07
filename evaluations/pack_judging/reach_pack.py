"""Offline breadth-prefix comparison through the existing lab and patched Engine."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from enginepy.workflows.document_analysis import evidence_pack as ep
from paid_report import COMPARE, HARD_HISTORY, cases_only, historical_case
from paid_run import scopes
from paid_support import Ledger, resources

from jev_navigator.judgments.judge import Judge

from .frozen import load, sha256
from .packing import lab_rooms, native_pack, native_rooms
from .prepare import write
from .reach import OUT, PREFIXES, prepared_records
from .repack import FOLLOWUP, NoProvider, offline_owner
from .run import folder


async def main():
    owner = offline_owner()  # Installs socket/getaddrinfo refusal before any packing.
    assert Path(ep.__file__).resolve().is_relative_to((FOLLOWUP / "scratch").resolve())
    assert hasattr(ep.EvidencePack, "packing_units")
    prepared = load(OUT / "prepared.json")
    assert sum(1 for _ in prepared_records()) == prepared["groups"]
    dev, hard = cases_only(COMPARE / "step-3/metadata.json"), cases_only(HARD_HISTORY / "windows.json")
    cached, results = {}, []
    totals = defaultdict(lambda: defaultdict(int))
    costs = defaultdict(Decimal)
    packet_hashes = []
    for entry in prepared["cases"]:
        resources()
        path = Path(entry["path"])
        assert sha256(path) == entry["sha256"]
        complete = load(path)
        cid, dataset = entry["case"], entry["dataset"]
        context = complete["context"]
        relations, index = scopes(context, cached)
        history = historical_case(context, complete["labels"], dev, hard)
        for prefix in PREFIXES:
            case = {**complete, "groups": complete["groups"][:prefix]}
            source = complete["bindings"][: prefix * 16]
            rows, receipts = {}, []
            for group in case["groups"]:
                observed = load(OUT / "answers" / folder(cid) / str(group["ordinal"]) / "observation.json")
                assert observed["request_sha256"] == group["request_sha256"]
                assert not observed["failure"]
                rows[group["ordinal"]] = observed["rows"]
                receipts.append(observed)
                costs[f"{dataset}:{prefix}:counted_usd"] += Decimal(observed["cash_usd"])
                costs[f"{dataset}:{prefix}:configuration_usd"] += Decimal(observed["configuration_usd"])
                costs[f"{dataset}:{prefix}:known_spend_usd"] += Decimal(
                    observed.get("known_spend_usd", observed["cash_usd"])
                )
                costs[f"{dataset}:{prefix}:pending_reserved_usd"] += Decimal(
                    observed.get("pending_reserved_usd", "0")
                )
            judge = Judge(NoProvider(), masker=None, max_calls=0)
            pack, result = await native_pack(relations, index, case, source, judge, rows)
            assert judge.calls == 0 and result.failure is None
            assert pack is None or not pack.failure
            lab = lab_rooms(history, source, case, rows, index)
            destination = OUT / "packs" / folder(cid) / str(prefix)
            destination.mkdir(parents=True, exist_ok=True)
            native = native_rooms(pack, relations, context, case["labels"], destination)
            # Verify the deciding line itself was serialized to Jev, not merely in a nominal span.
            visible = set()
            for group in case["groups"]:
                if not rows[group["ordinal"]]:
                    continue
                for slot, item in enumerate(group["request"]["state"]["items"]):
                    place = group["places"][str(slot)]
                    line_numbers = [i for first, last in place["ranges"] for i in range(first, last + 1)]
                    sent_lines = item["code"].split("\n")
                    assert len(line_numbers) == len(sent_lines), (cid, group["ordinal"], slot)
                    original = (Path(context["repository"]) / place["file"]).read_text().splitlines()
                    for line, sent in zip(line_numbers, sent_lines, strict=True):
                        # Redacted lines do not establish that Jev saw the original deciding text.
                        if sent == original[line - 1]:
                            visible.add((place["file"], line))
            shown = [(label["file"], label["first_line"]) in visible for label in case["labels"]]
            key = f"{dataset}:{prefix}"
            totals[key]["cases"] += 1
            totals[key]["registered_lines"] += len(shown)
            totals[key]["seen_lines"] += sum(shown)
            totals[key]["groups"] += prefix
            totals[key]["reused"] += sum(o["reused"] for o in receipts)
            totals[key]["outside_box"] += sum(o.get("skipped") == "input_box" for o in receipts)
            rooms = {}
            for room in ("7200", "20000", "36000"):
                packet = destination / f"native-{room}.txt"
                literal = owner.reader(packet.read_text())
                assert native[room]["delivered"] == [
                    (label["file"], label["first_line"]) in literal for label in case["labels"]
                ]
                assert native[room]["within_room"]
                lab_delivered = [label["delivered"] for label in lab[room]["labels"]]
                totals[key][f"lab:{room}"] += sum(lab_delivered)
                totals[key][f"native:{room}"] += sum(native[room]["delivered"])
                rooms[room] = {
                    "lab": lab_delivered,
                    "native": native[room]["delivered"],
                    "native_chars": native[room]["chars"],
                }
                packet_hashes.append({"case": cid, "prefix": prefix, "room": room, "sha256": sha256(packet)})
            results.append(
                {
                    "case": cid,
                    "dataset": dataset,
                    "prefix": prefix,
                    "seen": shown,
                    "rooms": rooms,
                    "provider_calls": 0,
                    "cash_usd": str(sum((Decimal(o["cash_usd"]) for o in receipts), Decimal(0))),
                    "configuration_usd": str(
                        sum((Decimal(o["configuration_usd"]) for o in receipts), Decimal(0))
                    ),
                    "service_seconds": sum(o["receipt"]["seconds"] for o in receipts if o["receipt"]),
                }
            )
        print(f"packed: {cid}", flush=True)
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap="2.00")
    latest = ledger.latest()
    known = sum((Decimal(e["usd"]) for e in latest.values() if e["status"] == "settled"), Decimal(0))
    pending = sum((Decimal(e["usd"]) for e in latest.values() if e["status"] == "reserved"), Decimal(0))
    assert known + pending <= ledger.cap
    pending_ids = {e["id"] for e in latest.values() if e["status"] == "reserved"}
    verified_overloads = set()
    for receipt_path in (OUT / "answers").glob("*/*/http/reach_*/receipt.json"):
        receipt = load(receipt_path)
        if receipt["id"] in pending_ids:
            assert receipt["status"] == 529
            raw = load(receipt_path.with_name("response.json"))
            assert raw["detail"]["error_type"] == "system_overloaded"
            verified_overloads.add(receipt["id"])
    assert pending_ids == verified_overloads
    write(OUT / "results.json", results)
    write(OUT / "packet-hashes.json", packet_hashes)
    write(
        OUT / "summary.json",
        {
            "status": "PASS",
            "packing_provider_calls": 0,
            "rows": len(results),
            "literal_packets": len(packet_hashes),
            "all_packets_within_room": True,
            "weights": prepared["weights"],
            "engine_sha256": sha256(Path(ep.__file__)),
            "paid_attempts": len(ledger.latest()),
            "step_counted_usd": str(ledger.cap - ledger.balance()),
            "known_spend_usd": str(known),
            "pending_reserved_usd": str(pending),
            "overload_attempts": len(pending_ids),
            "totals": {k: dict(v) for k, v in totals.items()},
            "costs": {k: str(v) for k, v in costs.items()},
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
