"""Free role coverage on the original matched R0/J1 groups and literal Engine selection."""

from __future__ import annotations

import asyncio
import hashlib
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

from enginepy.workflows.document_analysis import evidence_pack as ep
from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLE_QUESTIONS
from paid_native import packet_in_room
from paid_run import scopes
from paid_support import resources

from jev_navigator.judgments.judge import Judge

from .frozen import load, sha256
from .packing import native_pack
from .prepare import PLANNING, write
from .refresh import ADMITTED
from .repack import FOLLOWUP, NoProvider, offline_owner
from .run import bindings, observations

DESTINATION = PLANNING / "search-design/judging/selected-roles-20261007"
ROOMS = (16000, 20000, 36000)
ROLES = tuple(ROLE_QUESTIONS)


def deciding_lines(region, labels, reader):
    """Require literal numbered source delivery, not the enclosing region header."""
    shown = reader(region.render())
    return [label for label in labels if (label["file"], label["first_line"]) in shown]


async def run():
    owner = offline_owner()
    assert Path(ep.__file__).resolve().is_relative_to((FOLLOWUP / "scratch").resolve())
    cases = load(ADMITTED / "frozen-cases.json")
    assert len(cases) == 123 and sum(len(c["groups"]) for c in cases) == 984
    truth_path = Path.home() / (
        ".local/share/system-one-proof/jvn-eval-2026-10-03/"
        "runs/roles-compare-20261006/role-need/truth-table.json"
    )
    truth = {row["case"]: row for row in load(truth_path)}
    results, receipts, cache = [], [], {}
    for case in cases:
        resources()
        context = case["context"]
        cid = context["case"]
        targets = case["groups"][0]["request"]["state"]["targets"]
        context = {**context, "points": targets}
        baseline = {
            (row["place"]["id"], row["point"]): row
            for group in case["groups"]
            for row in group["baseline_rows"]
        }
        for group in case["groups"]:
            assert 1 <= len(group["request"]["state"]["items"]) <= 16
            if len(group["request"]["state"]["items"]) == 16:
                receipts.append(group["receipt"])
        relations, index = scopes(context, cache)
        judge = Judge(NoProvider(), masker=None, max_calls=0)
        pack, result = await native_pack(
            relations, index, {**case, "context": context}, bindings(case), judge, observations("J1-3", case)
        )
        assert judge.calls == 0 and result.failure is None
        candidates = {}
        if pack:
            for unit in (*pack.units, *(pack.packing_units or ())):
                region = unit.region()
                candidates.setdefault(region, unit)
        for room in ROOMS:
            packet = packet_in_room(pack, relations, context, room)
            selected = []
            if packet:
                assert len(packet.render()) <= room * 4
                for region in packet.regions:
                    if region.role != "ranked":
                        continue
                    unit = candidates[region]
                    points = []
                    for point in targets:
                        row = baseline[unit.unit.place, point]
                        assert set(row["probabilities"]) == set(ROLES)
                        points.append(
                            {
                                "point": point,
                                "probabilities": row["probabilities"],
                                "roles_at_050": [r for r in ROLES if row["probabilities"][r] >= 0.5],
                                "request_sha256": row["request_sha256"],
                            }
                        )
                    selected.append(
                        {
                            "place_id": unit.unit.place,
                            "file": region.file,
                            "ranges": unit.unit.runs,
                            "text_sha256": hashlib.sha256(region.text.encode()).hexdigest(),
                            "deciding_lines": deciding_lines(region, case["labels"], owner.reader),
                            "points": points,
                        }
                    )
            results.append({"case": cid, "dataset": context["dataset"], "room": room, "pieces": selected})
        print(cid, flush=True)
    totals = {}
    for dataset in ("dev110", "hard27"):
        for room in ROOMS:
            rows = [r for r in results if r["dataset"] == dataset and r["room"] == room]
            deciding = [p for r in rows for p in r["pieces"] if p["deciding_lines"]]
            all_roles, local_roles, matched = Counter(), Counter(), Counter()
            role_cases = {r: set() for r in ROLES}
            any_count = local_count = 0
            for row in rows:
                for piece in row["pieces"]:
                    if not piece["deciding_lines"]:
                        continue
                    roles = {r for p in piece["points"] for r in p["roles_at_050"]}
                    all_roles.update(roles)
                    local_roles.update(roles & set(LOCAL_ROLES))
                    any_count += bool(roles)
                    local_count += bool(roles & set(LOCAL_ROLES))
                    for role in roles:
                        role_cases[role].add(row["case"])
            if dataset == "dev110":
                for row in rows:
                    for role in truth[row["case"]]["registered_code_roles"]:
                        if any(
                            role in point["roles_at_050"]
                            for piece in row["pieces"]
                            if piece["deciding_lines"]
                            for point in piece["points"]
                        ):
                            matched[role] += 1
            totals[f"{dataset}:{room}"] = {
                "selected_pieces": sum(len(r["pieces"]) for r in rows),
                "deciding_pieces": len(deciding),
                "any_of_six_at_050": any_count,
                "any_local_role_at_050": local_count,
                "deciding_piece_role_counts": dict(all_roles),
                "deciding_piece_local_counts": dict(local_roles),
                "deciding_case_role_counts": {r: len(c) for r, c in role_cases.items()},
                "historical_host_role_cases_matched": dict(matched),
            }
    costs = [Decimal(r["usd"]) for r in receipts]
    summary = {
        "status": "PASS",
        "provider_calls": 0,
        "usd": 0,
        "cases": len(cases),
        "matched_groups": sum(len(c["groups"]) for c in cases),
        "measured_full_groups": len(receipts),
        "rooms": ROOMS,
        "totals": totals,
        "measured_original_16_piece_role_request_usd": {
            "mean": str(sum(costs) / len(costs)),
            "median": str(statistics.median(costs)),
            "min": str(min(costs)),
            "max": str(max(costs)),
            "total": str(sum(costs)),
        },
        "role_need_reference_counts": {
            role: sum(role in r["registered_code_roles"] for r in truth.values()) for role in LOCAL_ROLES
        },
        "source_hashes": {str(p): sha256(p) for p in (ADMITTED / "frozen-cases.json", truth_path)},
        "limitations": [
            "R0 observed original pieces in their original groups, not fresh selected-only batches.",
            "Role probabilities can concern other lines in a piece, not its registered deciding line.",
            "Role-need references are provisional host readings; their point counts differ from pieces.",
            "Actual new selected-only labelling request price has not been measured without provider calls.",
        ],
    }
    write(DESTINATION / "results.json", results)
    write(DESTINATION / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    asyncio.run(run())
