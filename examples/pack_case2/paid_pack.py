"""Pack original paid union observations without issuing or regrouping requests.

Use the pinned role-profile JVN, Engine and find_eval owners on PYTHONPATH.
This module constructs no transport and imports no offline launcher's audit hook.
Labels belong only to ``score_windows`` after packing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from enginepy.workflows.document_analysis import evidence_pack as owner
from enginepy.workflows.document_analysis.skeptic_packet import PacketBudget, SkepticPacket
from find_eval.simulate import composed_case, consumer_windows
from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2, RoleAnswers, compose_roles, retain_roles

from jev_navigator.directives.find_all import UnitScore
from jev_navigator.index.units import Item, Piece, Unit, UnitKind
from jev_navigator.judgments.answers import response_from_raw
from jev_navigator.judgments.judge import CheckResult
from jev_navigator.judgments.questions import content_hash, item_path
from jev_navigator.judgments.thresholds import Thresholds

VIEWS = ("union", "plan-only", "ranking-only")
ROOMS = (7200, 20000, 36000)
CHECKPOINTS = (4, 8, 16, 24)
BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"


@dataclass(frozen=True)
class Observation:
    """A real answer with its original physical request and asked source piece."""

    ordinal: int
    target: str
    target_text: str
    unit: Unit
    answer: CheckResult
    code: str
    source_flags: frozenset[str]
    usd: float
    seconds: float
    input_tokens: int | None

    @property
    def window(self):
        return [
            {"file": self.answer.place.file, "first_line": start, "last_line": end}
            for start, end in self.answer.place.ranges
        ]


def _unit(raw):
    fields = dict(raw)
    fields["ranges"] = tuple(map(tuple, fields["ranges"]))
    fields["kind"] = UnitKind(fields["kind"])
    fields["pieces"] = tuple(Piece(**piece) for piece in fields.get("pieces", ()))
    return Unit(**fields)


def _flags(raw):
    return (
        frozenset(key for key, included in raw.items() if included)
        if isinstance(raw, dict)
        else frozenset(raw)
    )


def answered_observations(prepared, responses, candidates):
    """Join receipts to EXACT original bodies, then decode the six original answers.

    Only answered requests contribute. Missing answers are an error, never 0.5.
    Geometry comes from source membership, never from an enclosing unit header.
    """
    source_items = {}
    for candidate in candidates:
        unit = _unit(candidate["unit"])
        for item in candidate["items"]:
            source_items[(unit.id, item["id"])] = (unit, item, candidate.get("source_flags", ()))
    receipts = {}
    for receipt in responses:
        ordinal = receipt["ordinal"]
        if ordinal in receipts:
            raise ValueError(f"duplicate response ordinal {ordinal}")
        receipts[ordinal] = receipt
    observations = []
    seen = set()
    thresholds = Thresholds()
    for record in prepared:
        ordinal = record["ordinal"]
        if not isinstance(ordinal, int) or ordinal < 1:
            raise ValueError("physical request ordinals must be positive and one-based")
        if ordinal in seen:
            raise ValueError(f"duplicate prepared ordinal {ordinal}")
        seen.add(ordinal)
        if ordinal not in receipts:
            continue
        receipt = receipts[ordinal]
        request = record["request"]
        state, questions = request["state"], request["questions"]
        digest = content_hash({"state": state, "questions": questions})
        if record["request_sha256"] != digest or receipt["request_sha256"] != digest:
            raise ValueError(f"request/response identity mismatch at ordinal {ordinal}")
        members = record["members"]
        if len(members) != len(state["items"]):
            raise ValueError(f"membership count mismatch at ordinal {ordinal}")
        raw_response = response_from_raw(receipt["response"])
        if set(raw_response.answers) != set(questions):
            raise ValueError(f"incomplete original answers at ordinal {ordinal}")
        for slot, member in enumerate(members):
            unit, item, flags = source_items[(member["unit_id"], member["id"])]
            place = Item(member["id"], member["file"], tuple(map(tuple, member["ranges"])))
            if (place.file, place.ranges) != (item["file"], tuple(map(tuple, item["ranges"]))):
                raise ValueError(f"source geometry mismatch for {place.id}")
            if member.get("unit_id", unit.id) != unit.id:
                raise ValueError(f"source unit mismatch for {place.id}")
            # Masking can change code, but the original physical state's path is fixed.
            if state["items"][slot]["file"] != place.file:
                raise ValueError(f"original request member mismatch for {place.id}")
            for target, target_text in state["targets"].items():
                components = {}
                for check in ROLES_V2.questions(target):
                    qid = f"{check.question_id}#{slot}"
                    if questions.get(qid) != check.to_question(item_path("items", slot)):
                        raise ValueError(f"original question contract mismatch: {qid}")
                    probability = raw_response.noul(qid).probability
                    if not 0 <= probability <= 1:
                        raise ValueError(f"invalid probability: {qid}")
                    role = check.name.removesuffix(f"_{target}")
                    components[role] = CheckResult(
                        state["items"][slot],
                        probability,
                        thresholds.noul_verdict(probability),
                        True,
                        digest,
                        qid,
                        place,
                    )
                observations.append(
                    Observation(
                        ordinal,
                        target,
                        target_text,
                        unit,
                        ROLES_V2.compose(components, thresholds),
                        item["code"],
                        _flags(member.get("source_flags", flags)),
                        float(receipt["usd"]),
                        float(receipt["seconds"]),
                        raw_response.input_tokens,
                    )
                )
    if missing := set(receipts) - seen:
        raise ValueError(f"responses without original prepared requests: {sorted(missing)}")
    return tuple(observations)


def filter_observations(observations, view, checkpoint):
    """Filter answers only; original wire state and question companions stay untouched."""
    if view not in VIEWS or checkpoint not in CHECKPOINTS:
        raise ValueError("unknown view or physical request checkpoint")
    flag = {"plan-only": "plan", "ranking-only": "ranking"}.get(view)
    return tuple(
        o for o in observations if o.ordinal <= checkpoint and (flag is None or flag in o.source_flags)
    )


def score_windows(windows, labels):
    """Score actual source runs; disjoint pieces never cover their envelope's gap."""
    results = []
    for label in labels:
        intervals = sorted((w["first_line"], w["last_line"]) for w in windows if w["file"] == label["file"])
        next_line = label["first_line"]
        for start, end in intervals:
            if start > next_line:
                break
            next_line = max(next_line, end + 1)
        results.append({**label, "delivered": next_line > label["last_line"]})
    return results


def receipt_metrics(observations):
    requests = {o.ordinal: o for o in observations}
    return {
        "original_requests": len(requests),
        "original_usd": sum(o.usd for o in requests.values()),
        "original_seconds": sum(o.seconds for o in requests.values()),
        "original_input_tokens": sum(o.input_tokens or 0 for o in requests.values()),
        "original_tokens_not_reported": sum(o.input_tokens is None for o in requests.values()),
        "asked_windows": [window for o in observations for window in o.window],
        "observed_pieces": len(observations),
        "provider_calls": 0,
        "usd": 0,
    }


def pack_lab(lab_case, observations, room_tokens):
    """Use the frozen lab's actual role allocation on freshly answered source pieces."""
    started = time.monotonic()
    pairs, units, probabilities = {}, {}, {}
    for order, observed in enumerate(observations):
        place = observed.answer.place
        key = f"{observed.target}:{place.id}"
        # The numbered source body, not the complete unit envelope, determines room.
        lines = iter(observed.code.split("\n"))
        numbered = "\n".join(
            f"{n}: {next(lines)}" for start, end in place.ranges for n in range(start, end + 1)
        )
        units[key] = {
            "place": place.id,
            "file": place.file,
            "runs": place.ranges,
            "extent": observed.unit.ranges,
            "name": observed.unit.symbol or "<top level>",
            "numbered_chars": len(numbered),
        }
        pairs[key] = {"point_id": observed.target, "unit_key": key, "order": order}
        probabilities[key] = {r: a.probability for r, a in observed.answer.components.items()}
    case = {**lab_case, "labels": [], "ranked_capacity_chars": room_tokens * 4}
    result = composed_case(case, pairs, units, probabilities, (RoleAnswers, compose_roles), LOCAL_ROLES)
    return {
        **result,
        "seconds": time.monotonic() - started,
        "room_tokens": room_tokens,
        "owner_path": "find_eval.simulate.composed_case",
        "room_axis": "ranked code allowance; frozen lab floor is additional",
        "ranked_estimated_tokens": result["ranked_chars"] / 4,
        **receipt_metrics(observations),
    }


def _rankings(observations):
    """Actual profile retention over observed pieces, preserving original Item IDs.

    The union's canonical geometry ID can differ from an original Item's ID.
    UnitScore is the native owner boundary that does not mistake that alias for
    an unasked body, or change the source receipt to force an ID lookup.
    """
    scores = defaultdict(list)
    for observed in observations:
        runs = observed.answer.place.ranges
        piece = None
        if runs != observed.unit.ranges:
            piece = next((p for p in observed.unit.pieces if ((p.start, p.end),) == runs), None)
            if piece is None:
                raise ValueError(f"asked piece has no source geometry: {observed.answer.place.id}")
        scores[observed.target].append(UnitScore(observed.unit, observed.answer, piece))
    return {
        point: tuple(owner._ranked(score, 1) for score in retain_roles(entries, LOCAL_ROLES))
        for point, entries in scores.items()
    }


def _coverage(observations):
    by_target = defaultdict(list)
    for observed in observations:
        by_target[observed.target].append(
            RoleAnswers(
                observed.answer.place.id,
                {role: answer.probability for role, answer in observed.answer.components.items()},
            )
        )
    result = {}
    for target, answers in by_target.items():
        composed = compose_roles(answers, LOCAL_ROLES)
        result[target] = {
            "best_by_role": {role: answer.unit_id for role, answer in composed["best_by_role"].items()},
            "uncovered_roles": sorted(composed["uncovered_roles"]),
            "follow_units": composed["follow_units"],
        }
    return result


def _small_packet(pack, room_chars):
    """Actual owner lower-level allocator, accounting for its rendered metadata."""
    packet = SkepticPacket(
        pack.claim_id,
        (*pack.floor, *(u.region() for u in pack.units)),
        pack.searches,
        pack.trimmed,
        pack.facts,
        pack.budget,
    )
    candidate = packet.within_budget(room_chars)
    while len(candidate.render()) > room_chars and candidate.regions:
        region_chars = sum(len(r.render()) + 1 for r in candidate.regions)
        overhead = len(candidate.render()) - region_chars
        next_packet = candidate.within_budget(max(0, room_chars - overhead))
        if next_packet.regions == candidate.regions:
            next_packet = candidate.within_budget(max(0, region_chars - 1))
        candidate = next_packet
    return candidate


def pack_native(row, relations, index, observations, room_tokens):
    """Native selection and rendered consumer packet from original CheckResults.

    No native source traversal runs: that would create different physical groups.
    The original trial requested p0 only; the claim's optional additional points
    cannot acquire inferred answers here. Native no-anchor behavior remains intact.
    """
    started = time.monotonic()
    budget = PacketBudget(max(20000, room_tokens))
    request = owner.pack_request(relations, row["repository"], row["claim"], budget=budget)
    if request is None:
        return {
            "no_anchor": True,
            "consumer_windows": [],
            "selected_units": 0,
            "selected_chars": 0,
            "owner_path": "evidence_pack.pack_request (no anchor)",
            "room_axis": "total rendered consumer packet allowance",
            "coverage": _coverage(observations),
            "packet": "",
            "packet_chars": 0,
            "packet_estimated_tokens": 0,
            "seconds": time.monotonic() - started,
            "room_tokens": room_tokens,
            **receipt_metrics(observations),
        }
    if room_tokens < 20000:
        floor = owner.build_packet(relations, row["repository"], row["claim"], budget_chars=sys.maxsize)
        floor = floor.within_budget(room_tokens * 2)
        request = replace(request, floor=floor.regions, searches=floor.receipts, trimmed=floor.trimmed)
    rankings = _rankings(observations)
    settings = owner.PackSettings(callee_round=False, question_profile=ROLES_V2, required_roles=LOCAL_ROLES)
    shown = (
        (owner._with_registration(relations, unit), reasons)
        for unit, reasons in owner.select_units(rankings, settings)
    )
    units = tuple(owner.PackUnit(unit, reasons, owner._numbered(index, unit)) for unit, reasons in shown)
    facts = (
        f"original union observations: {len(observations)} answered source piece(s); "
        "view filtered after judgment; no new groups or callee round",
    )
    pack = owner.EvidencePack(
        request.claim_id,
        request.floor,
        units,
        facts,
        owner.PackCost(0, 0, ("original_request_checkpoint",)),
        request.searches,
        request.trimmed,
        budget=budget,
    )
    packet = _small_packet(pack, room_tokens * 4) if room_tokens < 20000 else pack.packet()
    rendered = packet.render()
    return {
        "no_anchor": False,
        "consumer_windows": consumer_windows(pack, packet),
        "selected_units": len(units),
        "delivered_regions": len(packet.regions),
        "selected_chars": sum(len(u.render()) for u in units),
        "packet": rendered,
        "packet_chars": len(rendered),
        "packet_estimated_tokens": len(rendered) / 4,
        "packet_sha256": content_hash(rendered),
        "facts": list(facts),
        "coverage": _coverage(observations),
        "selected_sources": [
            {
                "place": observed.answer.place.id,
                "unit_id": observed.unit.id,
                "request_sha256": observed.answer.request_sha256,
                "ordinal": observed.ordinal,
                "runs": observed.answer.place.ranges,
                "roles": {
                    role: {"question_id": answer.question_id, "probability": answer.probability}
                    for role, answer in observed.answer.components.items()
                },
            }
            for observed in observations
            if observed.answer.place.id in {unit.unit.place for unit in units}
        ],
        "seconds": time.monotonic() - started,
        "room_tokens": room_tokens,
        "room_axis": "total rendered consumer packet allowance",
        "owner_path": (
            "SkepticPacket.within_budget(explicit legacy room)"
            if room_tokens < 20000
            else "EvidencePack.packet/within_pack_budget"
        ),
        **receipt_metrics(observations),
    }


def _rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _workload():
    """Read only admitted case input/label records; never deserialize the unit census."""
    import ijson

    tuning = {*(f"P{i}" for i in range(1, 8)), *(f"U{i}" for i in range(1, 7))}
    inputs, labels, lab = {}, {}, {}
    for dataset in ("dev110", "hard27"):
        with (BASE / f"runs/pack-49b78955/pack-inputs-{dataset}.json").open("rb") as stream:
            for cid, row in ijson.kvitems(stream, "cases", use_float=True):
                if dataset == "dev110" or cid in tuning:
                    inputs[cid] = (dataset, row)
        name = "replay-dev110" if dataset == "dev110" else "hard27"
        with (BASE / f"cases/{name}.json").open("rb") as stream:
            for case in ijson.items(stream, "item", use_float=True):
                if case["case_id"] in inputs:
                    labels[case["case_id"]] = case["labels"]
    metadata = BASE / "runs/roles-compare-20261006/step-3/metadata.json"
    with metadata.open("rb") as stream:
        for cid, case in ijson.kvitems(stream, "cases", use_float=True):
            if cid in inputs:
                lab[cid] = {key: value for key, value in case.items() if key != "labels"}
    return inputs, labels, lab


def main():
    """Persist every packing checkpoint as receipts land; rerunning incurs zero cost."""
    from evidence_pack_mode import _floor_window, _scope

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=lambda value: Path(value).expanduser())
    parser.add_argument("--case", action="append", dest="cases")
    options = parser.parse_args()
    inputs, labels, lab = _workload()
    by_repository = defaultdict(list)
    for cid, (dataset, row) in inputs.items():
        folder = options.stage / "cases" / cid.replace(":", "_")
        if options.cases and cid not in options.cases:
            continue
        if not (folder / "prepared-requests.jsonl").exists():
            continue
        by_repository[row["repository"]].append((cid, dataset, row, folder))
    summary = []
    for cases in by_repository.values():
        relations, index = _scope(cases[0][2])
        try:
            for cid, dataset, row, folder in cases:
                prepared = tuple(_rows(folder / "prepared-requests.jsonl"))
                response_path = folder / "responses.jsonl"
                receipts = tuple(_rows(response_path)) if response_path.exists() else ()
                prepared_ordinals = {record["ordinal"] for record in prepared}
                if prepared_ordinals != set(range(1, len(prepared) + 1)):
                    raise ValueError(f"prepared physical ordinals are not contiguous: {cid}")
                answered_ordinals = {record["ordinal"] for record in receipts}
                observations = answered_observations(
                    prepared,
                    receipts,
                    _rows(folder / "union-candidates.jsonl"),
                )
                lab_case = lab.get(cid)
                frozen_lab = lab_case is not None
                if lab_case is None:
                    floor_request = owner.pack_request(relations, row["repository"], row["claim"])
                    lab_case = {
                        "floor_windows": []
                        if floor_request is None
                        else [_floor_window(region) for region in floor_request.floor]
                    }
                results = []
                for checkpoint in CHECKPOINTS:
                    complete = set(range(1, min(checkpoint, len(prepared)) + 1)) <= answered_ordinals
                    for view in VIEWS:
                        selected = filter_observations(observations, view, checkpoint)
                        reached = score_windows([w for o in selected for w in o.window], labels[cid])
                        for room in ROOMS:
                            for path, packed in (
                                ("lab", pack_lab(lab_case, selected, room)),
                                ("native", pack_native(row, relations, index, selected, room)),
                            ):
                                if path == "native":
                                    packet_name = f"{view}-{checkpoint}-{room}-native-packet.txt"
                                    (folder / packet_name).write_text(packed.pop("packet") + "\n")
                                    packed["packet_file"] = packet_name
                                windows = packed["windows"] if path == "lab" else packed["consumer_windows"]
                                scored = score_windows(windows, labels[cid])
                                result = {
                                    "case": cid,
                                    "dataset": dataset,
                                    "path": path,
                                    "view": view,
                                    "checkpoint": checkpoint,
                                    "checkpoint_complete": complete,
                                    "checkpoint_status": "complete" if complete else "observed_lower_bound",
                                    "prepared_requests": len(prepared),
                                    "answered_ordinals": sorted(answered_ordinals),
                                    "room_tokens": room,
                                    **packed,
                                    "labels": scored,
                                    "reached_labels": reached,
                                    "delivered": sum(label["delivered"] for label in scored),
                                    "reached": sum(label["delivered"] for label in reached),
                                    "label_count": len(scored),
                                    "lab_floor_origin": "frozen lab" if frozen_lab else "actual Engine floor",
                                    "frozen_lab_baseline_available": frozen_lab,
                                }
                                results.append(result)
                _write(folder / "packing-results.json", {"case": cid, "results": results})
                summary.extend(
                    {
                        key: result[key]
                        for key in (
                            "case",
                            "dataset",
                            "path",
                            "view",
                            "checkpoint",
                            "checkpoint_complete",
                            "checkpoint_status",
                            "room_tokens",
                            "delivered",
                            "reached",
                            "label_count",
                            "seconds",
                            "original_requests",
                            "original_usd",
                        )
                    }
                    for result in results
                )
                _write(
                    options.stage / "packing-summary.json",
                    {
                        "cases": summary,
                        "provider_calls": 0,
                        "usd": 0,
                        "engine": "91f57ab4",
                        "jvn_profile": "677ef3e1",
                        "lab_room_axis": "ranked code room plus original floor",
                        "native_room_axis": "total rendered consumer packet room",
                    },
                )
                print(f"{cid}: packed {len(results)} view/checkpoint/room/path results", flush=True)
        finally:
            if index is not None:
                index.close()


if __name__ == "__main__":
    main()
