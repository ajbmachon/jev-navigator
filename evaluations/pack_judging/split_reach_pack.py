"""Matched before/after split-admission replay at four rooms, with no provider calls."""

from __future__ import annotations

import argparse
import asyncio
import csv
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
from .prepare import PLANNING, write
from .reach import OUT as REACH
from .reach import PREFIXES, prepared_records
from .repack import FOLLOWUP, NoProvider, offline_owner
from .run import folder

OUT = PLANNING / "search-design/judging/split-reach-20261007"
ROOMS = (7200, 16000, 20000, 36000)


def census_lines():
    with (PLANNING / "search-design/collection/unseen-lines.tsv").open() as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t") if row["input_box_excluded"] == "True"]
    assert len(rows) == 11
    return rows


async def main(variant):
    owner = offline_owner()  # Installs socket/getaddrinfo refusal before any packing.
    assert Path(ep.__file__).resolve().is_relative_to((FOLLOWUP / "scratch").resolve())
    assert hasattr(ep.EvidencePack, "packing_units")
    prepared = load(REACH / "prepared.json")
    census = census_lines()
    census_results = []
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
                old_path = REACH / "answers" / folder(cid) / str(group["ordinal"]) / "observation.json"
                new_path = OUT / "answers" / folder(cid) / str(group["ordinal"]) / "observation.json"
                observed = load(new_path if variant == "after" and new_path.exists() else old_path)
                if variant == "after" and not group["eligible"]:
                    assert new_path.exists() and not observed["failure"]
                elif group["eligible"]:
                    assert observed == load(old_path)
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
            lab = lab_rooms(history, source, case, rows, index, rooms=ROOMS)
            destination = OUT / variant / "packs" / folder(cid) / str(prefix)
            destination.mkdir(parents=True, exist_ok=True)
            native = native_rooms(pack, relations, context, case["labels"], destination, rooms=ROOMS)
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
            for room in map(str, ROOMS):
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
            for census_row in census:
                if census_row["case"] == cid:
                    position = int(census_row["label_index"])
                    label = case["labels"][position]
                    assert (label["file"], label["first_line"]) == (
                        census_row["file"],
                        int(census_row["line"]),
                    )
                    census_results.append(
                        {
                            "id": census_row["id"],
                            "case": cid,
                            "prefix": prefix,
                            "seen": shown[position],
                            "rooms": {
                                room: {path: masks[path][position] for path in ("lab", "native")}
                                for room, masks in rooms.items()
                            },
                        }
                    )
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
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap="0.40")
    latest = ledger.latest()
    known = sum((Decimal(e["usd"]) for e in latest.values() if e["status"] == "settled"), Decimal(0))
    pending = sum((Decimal(e["usd"]) for e in latest.values() if e["status"] == "reserved"), Decimal(0))
    assert known + pending <= ledger.cap
    if variant == "before":
        old_results = {(r["case"], r["prefix"]): r for r in load(REACH / "results.json")}
        old_hashes = {
            (r["case"], r["prefix"], r["room"]): r["sha256"] for r in load(REACH / "packet-hashes.json")
        }
        for row in results:
            prior = old_results[(row["case"], row["prefix"])]
            assert row["seen"] == prior["seen"]
            for room in ("7200", "20000", "36000"):
                assert row["rooms"][room] == prior["rooms"][room]
        for row in packet_hashes:
            if row["room"] != "16000":
                assert row["sha256"] == old_hashes[(row["case"], row["prefix"], row["room"])]
    write(OUT / variant / "census-lines.json", census_results)
    write(OUT / variant / "results.json", results)
    write(OUT / variant / "packet-hashes.json", packet_hashes)
    write(
        OUT / variant / "summary.json",
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
            "unpriced_attempts": sum(e["status"] == "reserved" for e in latest.values()),
            "totals": {k: dict(v) for k, v in totals.items()},
            "costs": {k: str(v) for k, v in costs.items()},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("before", "after"))
    asyncio.run(main(parser.parse_args().variant))
