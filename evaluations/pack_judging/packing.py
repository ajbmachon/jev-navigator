"""Bind scalar observations to the existing lab allocator and real Engine pack builder."""

from __future__ import annotations

from enginepy.workflows.document_analysis import evidence_pack as ep
from find_eval.simulate import composed_case, proposal_reducer
from jev_navigator.judgments.profiles import MATCH
from paid_native import packet_in_room, source_unit
from paid_report import lab_inputs
from paid_run import shown_lines
from paid_support import ROOMS

from jev_navigator.directives.find_all import NOT_REACHED, FindAllResult
from jev_navigator.index.units import items_to_judge
from jev_navigator.judgments.judge import CheckResult
from jev_navigator.judgments.thresholds import Thresholds

from .consume import judge_group


class FrozenRecipe:
    """A caller recipe, not a Judge: preserve frozen groups and return JVN's existing result type."""

    def __init__(self, case, bindings, rows, design=None, requests=None):
        self.case, self.rows, self.design, self.requests = case, rows, design, requests
        self.units = tuple(source_unit(b) for b in bindings)
        self.result = None

    async def search(self, index, judge, targets, **kwargs):
        rows = self.rows
        before = judge.calls
        if rows is None:
            rows = {}
            for ordinal, request in self.requests.items():
                rows[ordinal] = await judge_group(judge, self.design, request)
        places = {item.id: item for unit in self.units for item in items_to_judge(unit)}
        observed = {point: [] for point in targets}
        for group in self.case["groups"]:
            for row in rows.get(group["ordinal"], []):
                place = group["places"][str(row["slot"])]
                item = places[place["id"]]
                assert item.file == place["file"] and item.ranges == tuple(map(tuple, place["ranges"]))
                observed[row["point"]].append(
                    CheckResult(
                        group["request"]["state"]["items"][row["slot"]],
                        row["score"],
                        Thresholds().noul_verdict(row["score"]),
                        True,
                        row.get("request_sha256", group["request_sha256"]),
                        row.get("raw_question_id"),
                        item,
                    )
                )
        answered = {answer.place.id for values in observed.values() for answer in values}
        self.result = FindAllResult(
            dict(targets),
            76800,
            1,
            self.units,
            {key: tuple(values) for key, values in observed.items()},
            {key: NOT_REACHED for key in places if key not in answered},
            {},
            (),
            {},
            frozenset(),
            "scope_examined",
            judge.calls - before,
            entered_by={unit.id: "ranked_selection" for unit in self.units},
            question_profile=MATCH,
            required_roles=(),
        )
        return self.result


async def native_pack(relations, index, case, bindings, judge, rows=None, design=None, requests=None):
    recipe = FrozenRecipe(case, bindings, rows, design, requests)
    context = case["context"]
    request = ep.pack_request(relations, context["repository"], context["claim"], context.get("points"))
    if request is None:
        await recipe.search(index, judge, case["groups"][0]["request"]["state"]["targets"])
        return None, recipe.result
    previous = ep.find_all_async
    ep.find_all_async = recipe.search
    try:
        pack = await ep.build_pack(
            relations,
            request,
            judge,
            index,
            ep.PackSettings(callee_round=False, question_profile=MATCH, required_roles=()),
        )
    finally:
        ep.find_all_async = previous
    return pack, recipe.result


def lab_rooms(history, bindings, case, rows, index):
    observations = []
    for group in case["groups"]:
        for row in rows.get(group["ordinal"], []):
            observations.append(
                {
                    "place": group["places"][str(row["slot"])],
                    "point": row["point"],
                    "probabilities": {"match": row["score"]},
                }
            )
    # Reuse geometry/source binding from Case 1; its scalar-independent renderer needs only a score.
    keys = {(o["point"], o["place"]["id"]) for o in observations}
    geometry = [
        row
        for group in case["groups"]
        for row in group["baseline_rows"]
        if (row["point"], row["place"]["id"]) in keys
    ]
    pairs, units, _ = lab_inputs(bindings, geometry, index)
    answers = {f"{o['point']}:{o['place']['id']}": o["probabilities"] for o in observations}
    overhead = history["pack_chars"] - history["ranked_capacity_chars"]
    return {
        str(room): composed_case(
            {
                **history,
                "ranked_capacity_chars": max(
                    0, (history["pack_chars"] if room == 7200 else room * 4) - overhead
                ),
            },
            pairs,
            units,
            answers,
            proposal_reducer(None),
            (),
        )
        for room in ROOMS
    }


def native_rooms(pack, relations, context, labels, destination):
    result = {}
    for room in ROOMS:
        packet = packet_in_room(pack, relations, context, room)
        text = packet.render() if packet else ""
        (destination / f"native-{room}.txt").write_text(text)
        shown = shown_lines(packet)
        result[str(room)] = {
            "delivered": [(label["file"], label["first_line"]) in shown for label in labels],
            "chars": len(text),
            "selected_units": len(pack.units) if pack else 0,
            "within_room": len(text) <= room * 4,
            "regions": [
                {
                    "role": region.role,
                    "file": region.file,
                    "first_line": region.start,
                    "last_line": region.end,
                }
                for region in packet.regions
            ]
            if packet
            else [],
        }
    return result
