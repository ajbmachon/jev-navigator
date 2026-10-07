"""Free, post-judging room curves using the archived allocator and Engine packet.

Only successful actual request bodies supply lab observations. Labels score literal
source windows after allocation; they never participate in binding or selection.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, replace

ROOMS = (7_200, 20_000, 36_000)
_NUMBERED = re.compile(r"^(\d+): ?(.*)$")


def measurement_budget(tokens):
    """Keep production reservations while admitting the historical evaluation room."""
    from enginepy.workflows.document_analysis.skeptic_packet import PacketBudget

    @dataclass(frozen=True)
    class EvaluationPacketBudget(PacketBudget):
        def __post_init__(self):
            if self.tokens <= 0:
                raise ValueError("evaluation room must be positive")

    return EvaluationPacketBudget(tokens)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _runs(numbers):
    runs = []
    for number in sorted(set(numbers)):
        if runs and number == runs[-1][1] + 1:
            runs[-1][1] = number
        else:
            runs.append([number, number])
    return runs


def _windows(file, numbered, **metadata):
    """Intervals are derived from printed line numbers, never region headers."""
    printed = [(int(match[1]), line) for line in numbered.splitlines() if (match := _NUMBERED.match(line))]
    return [
        {
            "file": file,
            "first_line": start,
            "last_line": end,
            "text": "\n".join(line for number, line in printed if start <= number <= end),
            **metadata,
        }
        for start, end in _runs(number for number, _line in printed)
    ]


def _labels(labels, windows, key="delivered"):
    result = []
    for label in labels:
        covered = {
            line
            for window in windows
            if window["file"] == label["file"]
            for line in range(
                max(label["first_line"], window["first_line"]),
                min(label["last_line"], window["last_line"]) + 1,
            )
        }
        result.append({**label, key: len(covered) >= (label["last_line"] - label["first_line"] + 1) / 2})
    return result


def _source_variants(record, index, shown_files):
    """Materialize only files sent in actual groups, from the frozen candidate identities."""
    from jev_navigator.index.units import read_ranges

    variants = defaultdict(list)
    for unit in record["units"]:
        file = unit["path"]
        if file not in shown_files or file not in index.files:
            continue
        ranges = [unit["ranges"], *[[[piece["start"], piece["end"]]] for piece in unit.get("pieces", ())]]
        for runs in ranges:
            raw = read_ranges(index, file, runs)
            numbered = "\n".join(f"{n}: {index.lines(file)[n - 1]}" for a, b in runs for n in range(a, b + 1))
            variants[file].append(
                {
                    "place": unit["id"],
                    "file": file,
                    "runs": runs,
                    "extent": unit["ranges"],
                    "name": unit["symbol"] or "<top level>",
                    "raw_code": raw,
                    "numbered": numbered,
                    "numbered_chars": len(numbered),
                }
            )
    return variants


def _bind(item, variants, index):
    from find_eval.frozen_inputs import source_matches_masked
    from find_eval.masked_labels import _same_file

    file, shown = item["file"], item["code"]
    matches = [
        unit
        for canonical_file in variants
        if _same_file(file, canonical_file)
        for unit in variants[canonical_file]
        if unit["raw_code"] == shown
        or ("[MASKED]" in shown and source_matches_masked(unit["raw_code"], shown, canonical_file))
    ]
    # Limit-driven splitting can create a contiguous piece not frozen in the candidate file.
    if not matches and "[MASKED]" not in shown and file in index.files:
        source = "\n".join(index.lines(file))
        offset = source.find(shown)
        if offset >= 0 and (offset == 0 or source[offset - 1] == "\n"):
            end_offset = offset + len(shown)
            if (end_offset == len(source) or source[end_offset] == "\n") and source.find(
                shown, offset + 1
            ) < 0:
                start, end = source[:offset].count("\n") + 1, source[:end_offset].count("\n") + 1
                owners = {
                    unit["place"]: unit
                    for unit in variants[file]
                    if any(a <= start <= end <= b for a, b in unit["extent"])
                }
                if len(owners) == 1:
                    owner = next(iter(owners.values()))
                    numbered = "\n".join(f"{n}: {index.lines(file)[n - 1]}" for n in range(start, end + 1))
                    matches = [
                        {
                            **owner,
                            "runs": [[start, end]],
                            "raw_code": shown,
                            "numbered": numbered,
                            "numbered_chars": len(numbered),
                        }
                    ]
    distinct = {
        (unit["file"], tuple(map(tuple, unit["runs"])), tuple(map(tuple, unit["extent"]))): unit
        for unit in matches
    }
    if len(distinct) != 1:
        return None, "ambiguous body" if distinct else "unbound body"
    unit = next(iter(distinct.values()))
    numbers = [n for a, b in unit["runs"] for n in range(a, b + 1)]
    raw_lines, shown_lines = unit["raw_code"].split("\n"), shown.split("\n")
    if len(numbers) != len(raw_lines):
        return None, "source range no longer matches bound file"
    if len(raw_lines) != len(shown_lines):
        return None, "mask changed source line count"
    literal = "\n".join(
        f"{n}: {raw}" for n, raw, sent in zip(numbers, raw_lines, shown_lines, strict=True) if raw == sent
    )
    return {**unit, "literal_numbered": literal}, "literal body bound to frozen source"


def _observations(record, relations, groups, request):
    from find_eval.masked_labels import _same_file
    from jev_navigator.judgments.profiles import ROLES_V2

    from jev_navigator.index.code_index import CodeIndex

    sent_files = {item["file"] for group in groups for item in group["state"]["items"]}
    shown_files = {
        unit["path"] for unit in record["units"] if any(_same_file(sent, unit["path"]) for sent in sent_files)
    }
    index = CodeIndex(relations._repo, sorted(file for file in shown_files if relations.readable(file)))
    variants = _source_variants(record, index, shown_files)
    units, pairs, observations, bindings, unbound = {}, {}, {}, [], []
    asked = ROLES_V2.asked(request.points)
    candidate_order = {unit["id"]: order for order, unit in enumerate(record["units"])}
    point_order = {point: order for order, point in enumerate(request.points)}
    for group_number, group in enumerate(groups):
        answers = group.get("response", {}).get("answers", {})
        slot_points = defaultdict(dict)
        for question_id in group["questions"]:
            base, slot = question_id.rsplit("#", 1)
            _check, role, point = asked[base.split("@", 1)[0]]
            answer = answers.get(question_id)
            if answer is not None:
                if answer["type"] != "noul":
                    raise ValueError(f"expected Noul answer for {question_id}")
                slot_points[(int(slot), point)][role] = answer["noul"]
        for (slot, point), probabilities in slot_points.items():
            item = group["state"]["items"][slot]
            unit, reason = _bind(item, variants, index)
            witness = {
                "group": group_number,
                "key": group["key"],
                "slot": slot,
                "point": point,
                "shown": item,
                "binding": reason,
            }
            if unit is None:
                unbound.append(witness)
                continue
            unit_key = _hash([unit["place"], unit["runs"], unit["extent"]])
            pair = _hash([group["key"], slot, point])
            units[unit_key] = unit
            bindings.append(
                {
                    **witness,
                    "unit_key": unit_key,
                    "source": unit,
                    "roles": probabilities,
                    "complete": set(probabilities) == set(ROLES_V2.templates),
                }
            )
            if set(probabilities) == set(ROLES_V2.templates):
                # HTTP completion order is incidental. Frozen candidate order and
                # piece position reproduce the configured search's logical order.
                order = (
                    candidate_order[unit["place"]],
                    tuple(map(tuple, unit["runs"])),
                    point_order[point],
                    group["key"],
                    slot,
                )
                pairs[pair] = {"unit_key": unit_key, "point_id": point, "order": order}
                observations[pair] = probabilities
    return units, pairs, observations, bindings, unbound


def measure_case(record, pack, relations, request, groups, labels):
    """Return serializable literal delivery at three rooms without judging again."""
    from enginepy.workflows.document_analysis.skeptic_packet import SkepticPacket
    from find_eval.simulate import composed_case, proposal_reducer
    from jev_navigator.judgments.profiles import LOCAL_ROLES

    units, pairs, observed, bindings, unbound = _observations(record, relations, groups, request)
    judged_windows = [
        window
        for binding in bindings
        for window in _windows(
            binding["source"]["file"],
            binding["source"]["literal_numbered"],
            group=binding["group"],
            slot=binding["slot"],
            point=binding["point"],
        )
    ]
    reached = _labels(labels, judged_windows, "judged")
    selected_windows = [
        window
        for region in (*pack.floor, *(unit.region() for unit in pack.units))
        for window in _windows(region.file, region.text, role=region.role)
    ]
    selected_labels = _labels(labels, selected_windows, "selected")
    selected_references = sum(label["selected"] for label in selected_labels)
    result = {
        "case": record["case"],
        "recipe": record["recipe"],
        "primary": record["primary"],
        "judged_reach": {
            "labels": reached,
            "references": sum(row["judged"] for row in reached),
            "windows": judged_windows,
            "bindings": bindings,
            "unbound": unbound,
            "complete_pairs": len(observed),
        },
        "native_before_fitting": {
            "windows": selected_windows,
            "labels": selected_labels,
            "references": selected_references,
        },
        "rooms": {},
        "provider_calls": 0,
        "spent_usd": 0,
    }
    for tokens in ROOMS:
        budget = measurement_budget(tokens)
        floor_packet = SkepticPacket(
            request.claim_id, request.floor, request.searches, request.trimmed, budget=budget
        ).within_budget(budget.floor_chars)
        resized = replace(pack, floor=floor_packet.regions, trimmed=floor_packet.trimmed, budget=budget)
        packet = resized.packet()
        floor_windows = [
            window
            for region in floor_packet.regions
            for window in _windows(region.file, region.text, role=region.role)
        ]
        # The same metadata and reservation owner supplies the lab's ranked allowance.
        base_packet = replace(resized, units=()).packet()
        header = replace(
            base_packet, regions=(), notes=(), trimmed=base_packet.trimmed + len(base_packet.regions)
        ).render()
        metadata = max(budget.header_chars, len(header) + 1)
        metadata += max(budget.facts_chars, sum(len(note) + 1 for note in base_packet.notes))
        floor_chars = len(base_packet.render()) - len(replace(base_packet, regions=()).render())
        case = {
            "floor_windows": floor_windows,
            "labels": labels,
            "ranked_capacity_chars": max(0, budget.chars - metadata - floor_chars),
        }
        allocated = composed_case(case, pairs, units, observed, proposal_reducer(None), LOCAL_ROLES)
        lab_windows = [
            *floor_windows,
            *[
                window
                for pair in allocated["selected"]
                for window in _windows(
                    units[pairs[pair]["unit_key"]]["file"],
                    units[pairs[pair]["unit_key"]]["numbered"],
                    role="ranked",
                )
            ],
        ]
        native_windows = [
            window
            for region in packet.regions
            for window in _windows(region.file, region.text, role=region.role)
        ]
        rendered = packet.render()
        result["rooms"][str(tokens)] = {
            "total_tokens_estimated": tokens,
            "budget": {
                "chars": budget.chars,
                "floor_chars": budget.floor_chars,
                "header_chars": budget.header_chars,
                "facts_chars": budget.facts_chars,
            },
            "lab": {**allocated, "windows": lab_windows, "labels": _labels(labels, lab_windows)},
            "native": {
                "windows": native_windows,
                "labels": _labels(labels, native_windows),
                "rendered": rendered,
                "chars": len(rendered),
                "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "trimmed": packet.trimmed,
                "failure": pack.failure,
                "selected_references": selected_references,
                "fitting_lost_references": selected_references
                - sum(label["delivered"] for label in _labels(labels, native_windows)),
            },
            "floor_regions": [asdict(region) for region in floor_packet.regions],
        }
    return result
