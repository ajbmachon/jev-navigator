"""Report completed original union judgments and their real lab/native packing results.

This reader has no provider transport. Conditional views share the original union
request companions and cost; unfinished checkpoints are never reported as complete.
Run: python examples/pack_case2/trial_report.py PATH/union-stage-20261007
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from jev_navigator.judgments.answers import reported_input_tokens

CHECKPOINTS = (0, 4, 8, 16, 24)
DISPLAY_CHECKPOINTS = (4, 8, 16, 24)
ROOMS = (7200, 20000, 36000)
VIEWS = ("union", "plan-only", "ranking-only")
PATHS = ("lab", "native")
POPULATIONS = {"dev110": (110, 201), "hard27": (13, 28)}
CAP = Decimal("3.00")
PRE_CANDIDATE = Decimal("0.533934660")


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Trial reporting permits zero provider calls")


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    if not path.exists():
        return
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if not line.endswith("\n"):
                print(f"Snapshot waits for complete appended line: {path}:{number}", file=sys.stderr)
                return
            yield json.loads(line)


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def usd(value):
    return format(Decimal(str(value)), ".9f")


def median(values):
    return statistics.median(values) if values else None


def label_key(cid, label):
    return cid, label["file"], label["first_line"], label["last_line"]


def covers(windows, label):
    next_line = label["first_line"]
    intervals = sorted(
        (a, b) for window in windows if window["file"] == label["file"] for a, b in window["ranges"]
    )
    for start, end in intervals:
        if start > next_line:
            break
        next_line = max(next_line, end + 1)
    return next_line > label["last_line"]


def money(paid):
    latest = {row["id"]: row for row in rows(paid / "spend-ledger.jsonl")}
    settled = [row for row in latest.values() if row["status"] == "settled"]
    unresolved = [row for row in latest.values() if row["status"] == "reserved"]
    groups = defaultdict(lambda: Decimal(0))
    for row in settled:
        category = "candidate_jev" if row["id"].startswith("union:") else row["category"]
        groups[category] += Decimal(row["usd"])
    actual = sum((Decimal(row["usd"]) for row in settled), Decimal(0))
    reservations = sum((Decimal(row["usd"]) for row in unresolved), Decimal(0))
    return {
        "global_cap_usd": usd(CAP),
        "settled_total_usd": usd(actual),
        "unresolved_reservations_usd": usd(reservations),
        "remaining_admission_usd": usd(CAP - actual - reservations),
        "category_usd": {key: usd(value) for key, value in groups.items()},
        "candidate_jev_calls": sum(row["id"].startswith("union:") for row in settled),
        "candidate_jev_usd": usd(groups["candidate_jev"]),
        "pre_candidate_settled_usd": usd(PRE_CANDIDATE),
        "unresolved_ids": [row["id"] for row in unresolved],
        "first_guard": latest.get("meta-guard-v1"),
        "union_guard": latest.get("union-guard-v1"),
    }


def receipt_prefix(prepared, responses, transports, checkpoint):
    asked = {row["ordinal"]: row for row in prepared}
    if set(asked) != set(range(1, len(asked) + 1)):
        raise ValueError("Prepared physical request ordinals are not contiguous")
    receipts = {}
    for row in responses:
        ordinal = row["ordinal"]
        if ordinal in receipts:
            raise ValueError(f"Duplicate physical response ordinal {ordinal}")
        if ordinal not in asked or asked[ordinal]["request_sha256"] != row["request_sha256"]:
            raise ValueError(f"Response has no matching original physical request: {ordinal}")
        receipts[ordinal] = row
    wanted = set(range(1, min(checkpoint, len(asked)) + 1))
    answered = wanted & receipts.keys()
    durations = defaultdict(list)
    for attempt in transports:
        if attempt["ordinal"] in answered and attempt.get("duration_ms") is not None:
            durations[attempt["ordinal"]].append(attempt["duration_ms"] / 1000)
    missing_duration = sorted(answered - durations.keys())
    http_seconds = sum(
        sum(durations[ordinal]) if ordinal in durations else receipts[ordinal]["seconds"]
        for ordinal in answered
    )
    cost = sum((Decimal(receipts[n]["usd"]) for n in answered), Decimal(0))
    return {
        "required_physical_requests": len(wanted),
        "answered_physical_requests": len(answered),
        "response_checkpoint_complete": wanted <= answered,
        "physical_request_ordinals": sorted(answered),
        "candidate_jev_usd": usd(cost),
        "service_http_seconds": http_seconds,
        "service_seconds_method": "transport duration_ms; receipt wall fallback for missing durations",
        "http_duration_missing_ordinals": missing_duration,
        "receipt_wall_seconds": sum(receipts[n]["seconds"] for n in answered),
        "candidate_input_tokens": sum(reported_input_tokens(receipts[n]["response"]) or 0 for n in answered),
        "tokens_unreported_requests": sum(
            reported_input_tokens(receipts[n]["response"]) is None for n in answered
        ),
        "asked_pieces": sum(len(asked[n]["members"]) for n in answered),
    }


def final_collection_wall(path, responses, prepared_count):
    """Final observed first-to-last call span, never wall time at an earlier checkpoint."""
    ordinals = {row["ordinal"] for row in responses}
    required = set(range(1, min(24, prepared_count) + 1))
    first = min(responses, key=lambda row: row["ordinal"]) if responses else None
    metadata = path.stat() if path.exists() else None
    birth = getattr(metadata, "st_birthtime", None) if metadata else None
    modified = metadata.st_mtime if metadata else None
    result = {
        "final_candidate_collection_wall_seconds": None,
        "final_candidate_collection_wall_method": "unreported",
        "final_candidate_collection_first_call_duration": first["seconds"] if first else None,
        "responses_birthtime_epoch": birth,
        "responses_last_mtime_epoch": modified,
        "final_candidate_collection_start_epoch": None,
        "final_candidate_collection_end_epoch": None,
        "final_candidate_collection_answered_requests": len(ordinals),
        "final_candidate_collection_max_ordinal": max(ordinals, default=0),
        "final_candidate_collection_24_requests_answered": set(range(1, 25)) <= ordinals,
        "final_candidate_collection_prepared_queue_complete": required <= ordinals,
        "final_candidate_collection_status": (
            "24 physical requests answered"
            if set(range(1, 25)) <= ordinals
            else "prepared queue exhausted below 24"
            if required <= ordinals
            else "observed partial queue"
        ),
        "final_candidate_collection_wall_scope": (
            "Final observed first candidate call through last response, including checkpoint waits. "
            "Not wall time for earlier 4/8/16 checkpoints or complete planner-to-packet elapsed time."
        ),
    }
    if not responses:
        result["final_candidate_collection_wall_reason"] = "no candidate responses"
        return result
    if all(row.get("started_at") is not None and row.get("finished_at") is not None for row in responses):
        start = min(float(row["started_at"]) for row in responses)
        end = max(float(row["finished_at"]) for row in responses)
        method = (
            "absolute receipt timestamps; first started_at through last finished_at; "
            "includes checkpoint waiting"
        )
    elif birth is not None and modified is not None:
        start = birth - float(first["seconds"])
        end = modified
        method = (
            "filesystem first-response birth minus first-call duration through last-response modification; "
            "approximate; includes checkpoint waiting"
        )
    else:
        result["final_candidate_collection_wall_reason"] = (
            "no complete receipt timestamps or filesystem birthtime"
        )
        return result
    if end < start:
        raise ValueError(f"Candidate collection timestamps run backwards: {path}")
    result.update(
        final_candidate_collection_wall_seconds=end - start,
        final_candidate_collection_wall_method=method,
        final_candidate_collection_start_epoch=start,
        final_candidate_collection_end_epoch=end,
    )
    return result


def final_wall_summary(case_facts):
    result = {}
    for dataset in POPULATIONS:
        selected = [row for row in case_facts if row["dataset"] == dataset]
        measured = [row for row in selected if row["final_candidate_collection_wall_seconds"] is not None]
        complete = [row for row in measured if row["final_candidate_collection_prepared_queue_complete"]]
        result[dataset] = {
            "findings_observed": len(selected),
            "wall_reported_findings": len(measured),
            "final_candidate_collection_wall_seconds_median": median(
                [row["final_candidate_collection_wall_seconds"] for row in measured]
            ),
            "completed_prepared_queues": len(complete),
            "completed_prepared_queue_wall_seconds_median": median(
                [row["final_candidate_collection_wall_seconds"] for row in complete]
            ),
            "methods": dict(Counter(row["final_candidate_collection_wall_method"] for row in selected)),
            "statuses": dict(Counter(row["final_candidate_collection_status"] for row in selected)),
        }
    return result


def collect(stage, paid):
    per_finding, audits, case_facts = [], [], []
    baseline = {label_key(row["case"], row): row for row in load(paid / "line-provenance.json", [])}
    reach = {
        (row["case"], row["file"], row["line"]): row for row in load(stage / "deciding-line-reach.json", [])
    }
    folders = sorted((stage / "cases").iterdir()) if (stage / "cases").exists() else []
    missing = []
    for folder in folders:
        results_path = folder / "packing-results.json"
        if not results_path.exists():
            missing.append(folder.name)
            continue
        document = load(results_path)
        cid = document["case"]
        prepared = list(rows(folder / "prepared-requests.jsonl"))
        responses = list(rows(folder / "responses.jsonl"))
        collection_wall = final_collection_wall(folder / "responses.jsonl", responses, len(prepared))
        transports = list(rows(folder / "transport-attempts.jsonl"))
        planner = load(paid / "cases" / folder.name / "planner-receipt.json", {})
        execution = load(paid / "cases" / folder.name / "execution.json", {})
        prefixes = {cap: receipt_prefix(prepared, responses, transports, cap) for cap in CHECKPOINTS}
        source_candidates = {row["unit"]["id"]: row for row in rows(folder / "union-candidates.jsonl")}
        approaches = {row["approach"]["rank"]: row for row in execution.get("outcomes", [])}
        case_facts.append(
            {
                "case": cid,
                "dataset": document["results"][0]["dataset"],
                **collection_wall,
                "planner_seconds": planner.get("seconds"),
                "executor_seconds": execution.get("seconds"),
                "planner_usd": planner.get("usd"),
                "native_source_scope": next(
                    (
                        result.get("native_source_scope")
                        for result in document["results"]
                        if result["path"] == "native"
                    ),
                    None,
                ),
                "prepared_physical_requests": len(prepared),
                "responded_physical_requests": len(responses),
            }
        )
        for result in document["results"]:
            if (
                len(result["labels"]) != result["label_count"]
                or sum(label["delivered"] for label in result["labels"]) != result["delivered"]
                or sum(label["delivered"] for label in result["reached_labels"]) != result["reached"]
            ):
                raise ValueError(f"Packing aggregate differs from its per-label audit: {cid}")
            cap = result["checkpoint"]
            prefix = prefixes[cap]
            complete = result["checkpoint_complete"] and prefix["response_checkpoint_complete"]
            expected_ordinals = prefix["physical_request_ordinals"]
            packed_ordinals = [n for n in result["answered_ordinals"] if n <= cap]
            if packed_ordinals != expected_ordinals:
                complete = False  # Packing is an older receipt snapshot; regenerate it before completion.
            row = {
                "case": cid,
                "dataset": result["dataset"],
                "path": result["path"],
                "view": result["view"],
                "checkpoint": cap,
                "room_tokens": result["room_tokens"],
                "checkpoint_complete": complete,
                "checkpoint_status": "floor_only_baseline"
                if cap == 0 and complete
                else "complete"
                if complete
                else "observed_lower_bound",
                "label_count": result["label_count"],
                "asked_deciding_lines": result["reached"],
                "delivered_deciding_lines": result["delivered"],
                **prefix,
                "planner_usd": planner.get("usd"),
                "planner_seconds": planner.get("seconds"),
                "executor_seconds": execution.get("seconds"),
                "pack_seconds": result["seconds"],
                **collection_wall,
                "work_normalized_seconds": sum(
                    (
                        planner.get("seconds", 0),
                        execution.get("seconds", 0),
                        prefix["service_http_seconds"],
                        result["seconds"],
                    )
                ),
                "packed_physical_requests": len(packed_ordinals),
                "conditional_view_touched_requests": result["original_requests"],
                "conditional_view_touched_request_usd": usd(result["original_usd"]),
                "packet_estimated_tokens": result.get("packet_estimated_tokens"),
                "ranked_estimated_tokens": result.get("ranked_estimated_tokens"),
                "no_anchor": result.get("no_anchor", False),
                "room_axis": result["room_axis"],
                "native_eligible_observations": result.get("eligible_observation_count"),
                "native_excluded_observations": result.get("excluded_observation_count"),
                "native_excluded_reasons": dict(
                    Counter(item["reason"] for item in result.get("excluded_observations", []))
                ),
                "native_added_files": len(result.get("native_source_scope", {}).get("added_files", [])),
                "native_scope_excluded_files": len(
                    result.get("native_source_scope", {}).get("excluded_files", [])
                ),
            }
            per_finding.append(row)
            seen_labels = {label_key(cid, label): label for label in result["reached_labels"]}
            for label in result["labels"]:
                key = label_key(cid, label)
                old = baseline.get(key, {})
                lineage = reach.get((cid, label["file"], label["first_line"]), {})
                witnesses = []
                windows = []
                source = {"plan": False, "ranking": False}
                for request in sorted(prepared, key=lambda row: row["ordinal"]):
                    if request["ordinal"] not in packed_ordinals:
                        continue
                    for member in request["members"]:
                        flags = member.get("source_flags", {})
                        included = flags if isinstance(flags, dict) else dict.fromkeys(flags, True)
                        required = {"plan-only": "plan", "ranking-only": "ranking"}.get(result["view"])
                        if required and not included.get(required):
                            continue
                        windows.append(member)
                        if member["file"] == label["file"] and any(
                            a <= label["last_line"] and b >= label["first_line"] for a, b in member["ranges"]
                        ):
                            witnesses.append(
                                {
                                    "ordinal": request["ordinal"],
                                    "request_sha256": request["request_sha256"],
                                    "item_id": member["id"],
                                    "unit_id": member["unit_id"],
                                    "file": member["file"],
                                    "ranges": member["ranges"],
                                    "source_flags": included,
                                    "candidate_origin": source_candidates[member["unit_id"]].get("origin"),
                                    "approach_ranks": source_candidates[member["unit_id"]].get(
                                        "approach_ranks", []
                                    ),
                                }
                            )
                            for name in source:
                                source[name] |= bool(included.get(name))
                asked = seen_labels[key]["delivered"]
                if asked != covers(windows, label):
                    raise ValueError(f"Asked-source audit differs from packing result: {key}")
                first_plan = next((w for w in witnesses if w["source_flags"].get("plan")), None)
                proposal = (
                    approaches.get(min(first_plan["approach_ranks"]))
                    if (first_plan and first_plan["approach_ranks"])
                    else None
                )
                audits.append(
                    {
                        "case": cid,
                        "dataset": result["dataset"],
                        "file": label["file"],
                        "first_line": label["first_line"],
                        "last_line": label["last_line"],
                        "role": label.get("role"),
                        "path": result["path"],
                        "view": result["view"],
                        "checkpoint": cap,
                        "room_tokens": result["room_tokens"],
                        "checkpoint_complete": complete,
                        "asked": asked,
                        "delivered": label["delivered"],
                        "baseline_lab81_delivered": old.get("baseline_lab81_delivered"),
                        "baseline_native74_delivered": old.get("baseline_native74_delivered"),
                        "new_delivered_vs_lab81": label["delivered"]
                        and old.get("baseline_lab81_delivered") is False,
                        "new_delivered_vs_native74": label["delivered"]
                        and old.get("baseline_native74_delivered") is False,
                        "source_flags": source,
                        "first_witness": witnesses[0] if witnesses else None,
                        "asked_source_witnesses": witnesses,
                        "unasked_floor_delivery": label["delivered"] and not asked,
                        "plan_term_origin": proposal.get("term_origin") if proposal else None,
                        "first_witness_term_origin": proposal.get("term_origin")
                        if proposal and witnesses and witnesses[0]["source_flags"].get("plan")
                        else None,
                        "saved_plan_line_term_origin": old.get("term_origin"),
                        "actual_asked_plan_approach": proposal.get("approach") if proposal else None,
                        "plan_first_approach": old.get("actual_first_approach") if source["plan"] else None,
                        "code_ranking_lineage": lineage.get("origin", {}).get("ranking")
                        if source["ranking"]
                        else None,
                        "pretraining_origin": "unmeasured; witnessed terms do not establish knowledge origin",
                        "packet_file": result.get("packet_file"),
                        "native_exclusion_receipts": result.get("excluded_observations", []),
                    }
                )
    return per_finding, audits, case_facts, missing


def aggregate(per_finding):
    groups = defaultdict(list)
    for row in per_finding:
        key = row["dataset"], row["path"], row["view"], row["room_tokens"], row["checkpoint"]
        groups[key].append(row)
    results = []
    for dataset, (expected_cases, labels) in POPULATIONS.items():
        for path in PATHS:
            for view in VIEWS:
                for room in ROOMS:
                    previous = None
                    for checkpoint in CHECKPOINTS:
                        selected = groups[(dataset, path, view, room, checkpoint)]
                        if len({row["case"] for row in selected}) != len(selected):
                            raise ValueError(
                                f"Duplicate finding in curve cell: "
                                f"{dataset}/{path}/{view}/{room}/{checkpoint}"
                            )
                        complete = len(selected) == expected_cases and all(
                            row["checkpoint_complete"] for row in selected
                        )
                        if complete and sum(row["label_count"] for row in selected) != labels:
                            raise ValueError(
                                f"Completed {dataset} curve has an incorrect deciding-label total"
                            )
                        observed_cost = sum(
                            (Decimal(row["candidate_jev_usd"]) for row in selected), Decimal(0)
                        )
                        asked = sum(row["asked_deciding_lines"] for row in selected)
                        delivered = sum(row["delivered_deciding_lines"] for row in selected)
                        record = {
                            "dataset": dataset,
                            "path": path,
                            "view": view,
                            "room_tokens": room,
                            "checkpoint": checkpoint,
                            "status": "complete" if complete else "incomplete",
                            "findings_expected": expected_cases,
                            "findings_packed": len(selected),
                            "findings_checkpoint_complete": sum(
                                row["checkpoint_complete"] for row in selected
                            ),
                            "labels_expected": labels,
                            "asked_deciding_lines": asked if complete else None,
                            "delivered_deciding_lines": delivered if complete else None,
                            "observed_lower_bound_asked": asked,
                            "observed_lower_bound_delivered": delivered,
                            "candidate_jev_usd": usd(observed_cost) if complete else None,
                            "observed_candidate_jev_usd": usd(observed_cost),
                            "physical_requests": sum(row["answered_physical_requests"] for row in selected),
                            "native_scope_audit_complete": bool(selected)
                            and all(row["native_excluded_observations"] is not None for row in selected)
                            if path == "native"
                            else None,
                            "native_excluded_observation_count": sum(
                                row["native_excluded_observations"] or 0 for row in selected
                            )
                            if path == "native"
                            and selected
                            and all(row["native_excluded_observations"] is not None for row in selected)
                            else None,
                            "native_excluded_reasons": dict(
                                sum((Counter(row["native_excluded_reasons"]) for row in selected), Counter())
                            )
                            if path == "native"
                            else None,
                            "service_http_seconds_sum": sum(row["service_http_seconds"] for row in selected),
                            "work_normalized_seconds_median": median(
                                [row["work_normalized_seconds"] for row in selected]
                            ),
                            "packet_estimated_tokens_median": median(
                                [
                                    row["packet_estimated_tokens"]
                                    for row in selected
                                    if row["packet_estimated_tokens"] is not None
                                ]
                            ),
                            "ranked_estimated_tokens_median": median(
                                [
                                    row["ranked_estimated_tokens"]
                                    for row in selected
                                    if row["ranked_estimated_tokens"] is not None
                                ]
                            ),
                        }
                        if complete:
                            prior_cost = Decimal(previous["candidate_jev_usd"]) if previous else Decimal(0)
                            added_cost = observed_cost - prior_cost
                            prior_delivered = previous["delivered_deciding_lines"] if previous else None
                            record["marginal"] = {
                                "from_checkpoint": previous["checkpoint"] if previous else None,
                                "candidate_usd_increment": usd(added_cost),
                                "delivered_lines_increment": delivered - prior_delivered
                                if previous
                                else None,
                                "lines_per_extra_dollar": (
                                    (delivered - prior_delivered) / float(added_cost)
                                    if previous and added_cost > 0
                                    else None
                                ),
                                "baseline_note": "Measured zero-request floor baseline"
                                if checkpoint == 0
                                else "Zero-request floor baseline not yet available"
                                if not previous
                                else None,
                            }
                            previous = record
                        else:
                            record["marginal"] = None
                        results.append(record)
    return results


def first_witness_origin(row):
    """Attribute arrival to its earliest actual asked piece, preserving later plan evidence separately."""
    first = row["first_witness"]
    if first:
        flags = first["source_flags"]
        if flags.get("plan"):
            return row["first_witness_term_origin"] or "plan provenance unavailable"
        if flags.get("ranking"):
            return "code ranking lineage"
    return "code-built floor" if row["unasked_floor_delivery"] else "source provenance unavailable"


def delivered_provenance(audits):
    groups = defaultdict(list)
    for row in audits:
        if row["checkpoint_complete"]:
            key = (row["dataset"], row["path"], row["view"], row["room_tokens"], row["checkpoint"])
            groups[key].append(row)
    results = []
    for key, selected in sorted(groups.items()):
        record = dict(zip(("dataset", "path", "view", "room_tokens", "checkpoint"), key, strict=True))
        for baseline in ("lab81", "native74"):
            known = [row for row in selected if row[f"baseline_{baseline}_delivered"] is not None]
            fresh = [row for row in known if row[f"new_delivered_vs_{baseline}"]]
            origins = Counter()
            for row in fresh:
                origin = first_witness_origin(row)
                origins[origin] += 1
            record[baseline] = {
                "baseline_known_labels": len(known),
                "new_delivered": len(fresh) if known else None,
                "term_origins": dict(origins),
            }
        results.append(record)
    return results


def trial_summary(stage, paid, per_finding, case_facts, missing, audits):
    first = load(paid / "trial-summary.json", {})
    planner = [load(path) for path in (paid / "cases").glob("*/planner-receipt.json")]
    zero = load(stage / "reach-summary.json", {})
    guards = {"first": load(paid / "guard-scores.json", {}), "union": load(stage / "guard-scores.json", {})}
    curve = aggregate(per_finding)
    money_record = money(paid)
    return {
        "status": "complete" if curve and all(row["status"] == "complete" for row in curve) else "incomplete",
        "development_measure": True,
        "scope": {
            "dev110": {"findings": 110, "labels": 201},
            "hard27_tuning": {"findings": 13, "labels": 28},
            "held_out_evaluation": False,
        },
        "provider_calls_from_reporting": 0,
        "reporting_usd": 0,
        "missing_packing_cases": missing,
        "financials": money_record,
        "planner": {
            "calls": len(planner),
            "usd": usd(first.get("planner_usd", "0.434860440")),
            "input_tokens": sum(row["usage"]["prompt_tokens"] for row in planner),
            "output_tokens": sum(row["usage"]["completion_tokens"] for row in planner),
            "seconds_median": median([row["seconds"] for row in planner]),
            "binding_counts": first.get("binding_counts"),
            "malformed_rejected": first.get("rejected_schema_approaches"),
            "invalid_argument_audit": load(paid / "invalid-argument-audit.json", {}).get("counts"),
            "argument_note": (
                "Nine unbound approaches use an argument absent from context; not all "
                "failures are invented terms."
            ),
            "full_plan_unit_reach": {"dev110": 118, "hard27": 26},
            "full_plan_actual_body_reach": {"dev110": 117, "hard27": 26},
        },
        "guard_scores": guards,
        "new_delivery_provenance": delivered_provenance(audits),
        "packing_owner": {
            key: value for key, value in load(stage / "packing-summary.json", {}).items() if key != "cases"
        },
        "guard_note": (
            "Route is advisory review_required; aggregate atomicity and consumer remain unreviewed."
        ),
        "curve": curve,
        "floor_only_baselines": [row for row in curve if row["checkpoint"] == 0],
        "zero_cost_comparison": zero,
        "case_work": case_facts,
        "final_candidate_collection_timing": final_wall_summary(case_facts),
        "collection": load(stage / "dispatch-summary.json", {}),
        "baselines": {
            "lab_historical": 80,
            "lab_half_cent": 81,
            "native_historical": 68,
            "native_36000": 74,
            "unchanged_dev_floor_lab": 22,
            "unchanged_dev_floor_native": 35,
            "agent_usd_range": [0.029, 0.039],
            "agent_seconds_approx": 60,
            "agent_files_median": 11,
            "agent_lines_approx": 480,
            "perfect_planner_retrospective": 191,
            "perfect_planner_strict_provenance": 174,
            "recorded_planner_first_64_body_reach": 84,
            "recorded_planner_source": str(paid.parent / "REPORT.md"),
        },
        "comparison_note": (
            "Plan-only and ranking-only are conditional views of the same paid union "
            "answers and companions; not independent paid arms."
        ),
        "room_note": (
            "Lab rooms cover ranked code plus its original floor; native rooms cover the "
            "total rendered packet. Native 7,200 uses an explicit lower owner "
            "allocator; 20,000/36,000 use the standard owner. No new native traversal or "
            "request groups."
        ),
        "timing_note": (
            "Work-normalized seconds = planner + original executor + summed service HTTP "
            "+ packing per finding. Global barriers mean this is not per-finding elapsed "
            "latency. Final first-to-last candidate collection wall includes checkpoint waiting and is "
            "reported separately; filesystem-based values are approximate and cannot recover "
            "earlier checkpoint wall or complete planner-to-packet elapsed time."
        ),
        "native_scope_note": (
            "Native rendering adds explicitly admitted Git-tracked generated/vendor source to its normal "
            "Engine scope. Withheld, untracked and noncanonical source remain excluded. Asked-source reach "
            "is retained separately; native exclusion counts and exact reasons are in every curve cell "
            "and the per-line audit."
        ),
        "production_note": (
            "This trial uses retained evaluation source only. A production-approved "
            "customer-code LLM route remains open; no adoption or held-out claim "
            "follows."
        ),
        "artifacts": {
            "packing_summary": str(stage / "packing-summary.json"),
            "ledger": str(paid / "spend-ledger.jsonl"),
            "response_glob": str(stage / "cases/*/responses.jsonl"),
            "per_line_audit": str(stage / "line-provenance-final.json"),
            "invalid_arguments": str(paid / "invalid-argument-audit.json"),
        },
    }


def measured(row, key):
    return str(row[key]) if row["status"] == "complete" else "pending"


def timing(row):
    value = row["work_normalized_seconds_median"]
    return f"{value:.2f}" if row["status"] == "complete" and value is not None else "pending"


def learned_text(summary):
    completed = [
        row
        for row in summary["curve"]
        if row["status"] == "complete" and row["view"] == "union" and row["checkpoint"] > 0
    ]
    if not completed:
        return (
            "The fresh delivery curve is incomplete, so there is no measured result to adopt. "
            "See trial-summary.json and finish retained-receipt packing before choosing the next design."
        )
    observations = []
    for dataset in POPULATIONS:
        for path in PATHS:
            candidates = [row for row in completed if row["dataset"] == dataset and row["path"] == path]
            if not candidates:
                continue
            final = max(candidates, key=lambda row: (row["checkpoint"], row["room_tokens"]))
            observations.append(
                f"The {dataset} {path} union at {final['checkpoint']} physical requests "
                f"and {final['room_tokens']:,} tokens shows Jev {final['asked_deciding_lines']} "
                f"deciding lines and delivers {final['delivered_deciding_lines']} "
                f"of {final['labels_expected']}. "
                "The remaining gap identifies selection and room losses separately from search reach. "
                "See marginal-curve.json and line-provenance-final.json before enlarging discovery."
            )
    observations.extend(
        [
            "Plan and ranking views share original union companions. Their difference measures filtering "
            "after one judgment and cannot establish the price of separate paid workflows. "
            "See per-finding.csv before selecting a separate production arm.",
            "Finding words, cited code, outlines and proposed conventions are witnessed term provenance. "
            "The trial cannot identify pretraining knowledge. See line-provenance-final.json when choosing "
            "whether a rule or a planning call owns the next search.",
        ]
    )
    return "\n\n".join(observations)


def zero_cost_tables(zero):
    lines = [
        "",
        "## Zero-cost source comparison",
        "",
        "These use equal canonical-unit caps of 128, 256 and 384 per finding, with no new Jev calls. "
        "Short plan queues are exhausted without padding, so actual admitted counts can differ. "
        "A canonical unit may yield several source items or no judgeable item. Bodies are exact emitted "
        "source pieces; geometry alone does not count as a body. Source-character token estimates are "
        "not tokenizer measurements or billed input tokens.",
        "",
        "Saved plan order is the primary plan baseline. The supplementary plan_case1_ranked arm "
        "restricts frozen union ordering to plan units. Census units retain frozen Case1 scent and walk "
        "order; outside-census plan units use the disclosed ordinal fallback. No labels retune ranking.",
        "",
        zero.get("fallback_policy", "Fallback policy unavailable."),
        "",
        "| Population | Unit cap | Arm | Actual units summed | Actual body lines | "
        "With unchanged floor | Body bytes | Estimated body tokens |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in POPULATIONS:
        arms = zero.get("summary", {}).get(dataset, {}).get("arms", {})
        cases = [row for row in zero.get("cases", []) if row["population"] == dataset]
        for cap in (128, 256, 384):
            for arm in ("plan_only", "ranking_only", "union", "plan_case1_ranked"):
                reach = arms.get(arm, {}).get(str(cap), {})
                totals = [row.get("source_totals", {}).get(arm, {}).get(str(cap)) for row in cases]
                available = bool(totals) and all(total is not None for total in totals)
                units = sum(total["canonical_units"] for total in totals) if available else "unreported"
                body = sum(total["body_bytes"] for total in totals) if available else "unreported"
                tokens = (
                    f"{sum(total['estimated_body_tokens_at_4_chars'] for total in totals):,.2f}"
                    if available
                    else "unreported"
                )
                lines.append(
                    f"| {dataset} | {cap} | {arm} | {units} | "
                    f"{reach.get('actual_body_lines', 'unreported')} | "
                    f"{reach.get('including_unchanged_floor', 'unreported')} | {body} | {tokens} |"
                )
    lines.extend(
        [
            "",
            "The published frozen Case1 curve includes unchanged floor lines and unavailable source "
            "bindings in its ordering. It is reproduced separately below and must not be read as "
            "actual-body-only reach. In particular, the published development value 143 at 128 differs "
            "from actual-body ranking reach 122.",
            "",
            "| Population | Frozen Case1 at 128 | At 256 | At 384 |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in POPULATIONS:
        curve = zero.get("frozen_case1_reproduction", {}).get(dataset, {})
        values = " | ".join(str(curve.get(str(cap), "unreported")) for cap in (128, 256, 384))
        lines.append(f"| {dataset} | {values} |")
    lines.extend(
        [
            "",
            "Deciding-line overlap counts use actual bodies across each full source pool, independently "
            "of the paid prefix. They count line instances, not overlapping candidate units.",
            "",
            "| Population | Only plan | Only ranking | Both | Neither |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in POPULATIONS:
        overlap = zero.get("summary", {}).get(dataset, {}).get("overlap_full", {})
        values = " | ".join(
            str(overlap.get(key, 0)) for key in ("only_plan", "only_ranking", "both", "neither")
        )
        lines.append(f"| {dataset} | {values} |")
    return lines


def markdown(summary, stage):
    finance = summary["financials"]
    planner = summary["planner"]
    lines = [
        "# Case 2 fresh union trial",
        "",
        f"Status: **{summary['status']}**. All measures are development measures: "
        "110 development findings with 201 deciding-line instances, and 13 P/U tuning findings "
        "with 28 lines. There is no held-out or production adoption result.",
        "",
        f"Candidate Jev made {finance['candidate_jev_calls']} physical calls "
        f"for ${finance['candidate_jev_usd']}. Settled total spend is "
        f"${finance['settled_total_usd']} of $3.00. Unresolved reservations hold "
        f"${finance['unresolved_reservations_usd']}, leaving ${finance['remaining_admission_usd']} "
        "for admission. Planner and guard spend before candidate calls was $0.533934660.",
        "",
        "An incomplete cell means the whole population has not completed that checkpoint. "
        "Observed lower bounds remain in JSON and CSV. They are not completed 24-request results.",
        "",
        "## Fresh union delivery",
        "",
        "| Population | Requests | Room | Jev saw | Lab | Native | Candidate $ | Work s lab/native |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {
        (row["dataset"], row["path"], row["view"], row["room_tokens"], row["checkpoint"]): row
        for row in summary["curve"]
    }
    for dataset in POPULATIONS:
        for checkpoint in DISPLAY_CHECKPOINTS:
            for room in ROOMS:
                lab = by_key[(dataset, "lab", "union", room, checkpoint)]
                native = by_key[(dataset, "native", "union", room, checkpoint)]
                lines.append(
                    f"| {dataset} | {checkpoint} | {room:,} | {measured(lab, 'asked_deciding_lines')} | "
                    f"{measured(lab, 'delivered_deciding_lines')} | "
                    f"{measured(native, 'delivered_deciding_lines')} | "
                    f"{measured(lab, 'candidate_jev_usd')} | "
                    f"{timing(lab)}/{timing(native)} |"
                )
    lines.extend(
        [
            "",
            "## Conditional views in every room",
            "",
            "| Population | Requests | Room | View | Jev saw | Lab | Native | "
            "Shared candidate $ | Work s lab/native |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in POPULATIONS:
        for checkpoint in DISPLAY_CHECKPOINTS:
            for room in ROOMS:
                for view in ("plan-only", "ranking-only"):
                    lab = by_key[(dataset, "lab", view, room, checkpoint)]
                    native = by_key[(dataset, "native", view, room, checkpoint)]
                    lines.append(
                        f"| {dataset} | {checkpoint} | {room:,} | {view} | "
                        f"{measured(lab, 'asked_deciding_lines')} | "
                        f"{measured(lab, 'delivered_deciding_lines')} | "
                        f"{measured(native, 'delivered_deciding_lines')} | "
                        f"{measured(lab, 'candidate_jev_usd')} | {timing(lab)}/{timing(native)} |"
                    )
    lines.extend(
        [
            "",
            "## Marginal delivery in every room",
            "",
            "Each increment compares the preceding completed physical checkpoint, beginning with the "
            "measured zero-request floor. Every view shares the full original union request cost. "
            "A negative gain records a packing change rather than hiding it.",
            "",
            "| Population | Requests | Room | View | Extra $ | Lab gain | Lab lines/$ | "
            "Native gain | Native lines/$ |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in POPULATIONS:
        for checkpoint in DISPLAY_CHECKPOINTS:
            for room in ROOMS:
                for view in VIEWS:
                    lab = by_key[(dataset, "lab", view, room, checkpoint)]
                    native = by_key[(dataset, "native", view, room, checkpoint)]
                    values = []
                    for row in (lab, native):
                        change = row.get("marginal") or {}
                        gain = change.get("delivered_lines_increment")
                        efficiency = change.get("lines_per_extra_dollar")
                        values.extend(
                            [
                                str(gain) if gain is not None else "unmeasured",
                                f"{efficiency:.2f}" if efficiency is not None else "unmeasured",
                            ]
                        )
                    extra = (lab.get("marginal") or {}).get("candidate_usd_increment", "pending")
                    lines.append(
                        f"| {dataset} | {checkpoint} | {room:,} | {view} | {extra} | "
                        + " | ".join(values)
                        + " |"
                    )
    lines.extend(zero_cost_tables(summary["zero_cost_comparison"]))
    lines.extend(
        [
            "",
            "## Measured zero-request floors",
            "",
            "| Population | Room | Lab floor | Native floor |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in POPULATIONS:
        for room in ROOMS:
            lab = by_key[(dataset, "lab", "union", room, 0)]
            native = by_key[(dataset, "native", "union", room, 0)]
            lines.append(
                f"| {dataset} | {room:,} | {measured(lab, 'delivered_deciding_lines')} | "
                f"{measured(native, 'delivered_deciding_lines')} |"
            )
    lines.extend(
        [
            "",
            "## Native source admission audit at 24 requests",
            "",
            "Asked-source reach includes the actual retained pieces even when a native consumer excludes "
            "an observation. The following counts and reasons describe that consumer boundary.",
            "",
            "| Population | Room | Scope audit | Excluded observations | Reasons |",
            "|---|---:|---|---:|---|",
        ]
    )
    for dataset in POPULATIONS:
        for room in ROOMS:
            row = by_key[(dataset, "native", "union", room, 24)]
            complete = row["status"] == "complete" and row["native_scope_audit_complete"]
            count = row["native_excluded_observation_count"] if complete else "pending"
            reasons = "; ".join(f"{key}: {count}" for key, count in row["native_excluded_reasons"].items())
            lines.append(
                f"| {dataset} | {room:,} | {'complete' if complete else 'pending'} | {count} | "
                f"{reasons or ('none' if complete else 'pending')} |"
            )
    lines.extend(
        [
            "",
            "## New delivery and first witnessed source at 24 requests",
            "",
            "Each new delivered line is compared with its path's saved baseline using exact label "
            "geometry. Arrival attribution uses the earliest actual asked piece. A later plan witness "
            "is retained separately and cannot override an earlier ranking-only arrival. Copied finding "
            "words, cited code, outlines and suggested conventions describe observable term provenance; "
            "they do not identify pretraining knowledge. P/U has no saved baseline comparison.",
            "",
            "| Population | Path | Room | Baseline | New delivered lines | First-witness origins |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    lineage = {
        (row["dataset"], row["path"], row["view"], row["room_tokens"], row["checkpoint"]): row
        for row in summary["new_delivery_provenance"]
    }
    for dataset in POPULATIONS:
        for path in PATHS:
            for room in ROOMS:
                key = (dataset, path, "union", room, 24)
                baseline = "lab81" if path == "lab" else "native74"
                detail = lineage.get(key, {}).get(baseline, {})
                complete = by_key[key]["status"] == "complete"
                count = detail.get("new_delivered") if complete else None
                origins = "; ".join(
                    f"{origin}: {count}" for origin, count in detail.get("term_origins", {}).items()
                )
                lines.append(
                    f"| {dataset} | {path} | {room:,} | {baseline} | "
                    f"{count if count is not None else 'unreported' if complete else 'pending'} | "
                    f"{origins or ('unreported' if complete else 'pending')} |"
                )
    elapsed = summary["collection"].get("seconds")
    elapsed_text = f"{elapsed:.2f} seconds" if elapsed is not None else "pending"
    lines.extend(
        [
            "",
            f"Recorded candidate collection elapsed time: {elapsed_text}. "
            "Work medians above are sums of planner, executor, original HTTP and packing work; "
            "they are not elapsed latency across global checkpoint barriers.",
        ]
    )
    lines.extend(
        [
            "",
            "## Final per-finding candidate collection wall",
            "",
            "| Population | Timed findings | Final observed wall median seconds | Complete prepared queues |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset, result in summary["final_candidate_collection_timing"].items():
        value = result["final_candidate_collection_wall_seconds_median"]
        text = f"{value:.2f}" if value is not None else "unreported"
        lines.append(
            f"| {dataset} | {result['wall_reported_findings']} | {text} | "
            f"{result['completed_prepared_queues']} |"
        )
    lines.extend(
        [
            "",
            "Current receipts lack absolute call timestamps. Their final wall estimate uses original "
            "response-file birth time minus the first call duration through last file modification. "
            "This is approximate and includes checkpoint waiting after the first call. It does not "
            "recover wall time at earlier 4/8/16 checkpoints or complete planner-to-packet elapsed time. "
            "Future fully timestamped receipts use first started_at through last finished_at instead.",
            "",
            "The planner's median per-call service duration remains separate. No recorded batch "
            "wall time is available here, and call durations are not summed to invent an end-to-end latency.",
        ]
    )
    lines.extend(
        [
            "",
            summary["comparison_note"],
            "",
            summary["room_note"],
            "",
            summary["timing_note"],
            "",
            "The full [marginal curve](marginal-curve.json) records extra candidate dollars, "
            "extra delivered lines and lines per extra dollar at each completed checkpoint. "
            "Marginal delivery uses the room-specific zero-request floor baseline when it is complete.",
            "",
            f"The planner made {planner['calls']} calls, used {planner['input_tokens']:,} input "
            f"and {planner['output_tokens']:,} output tokens, cost ${planner['usd']}, "
            f"and took a median of {planner['seconds_median']:.2f} seconds. "
            "Of 1,203 proposals, 1,008 bound to code, 140 were empty, 46 were invalid or unbound, "
            "and 9 malformed proposals were rejected. Nine unbound approaches used an argument absent "
            "from context; this does not show that every failed argument was invented.",
            "",
            "Plan unit geometry reaches 118 development lines; actual emitted bodies reach 117. "
            "The bundled source-inventory-contract.mjs:1 piece exceeds the input room and emits no item. "
            "The hard tuning plan reaches 26 lines.",
            "",
            "The first guard made 96 calls for $0.059654028. The union guard made 96 calls "
            "for $0.039420192. Union median scores were question 60.5, state 84, primitive 90, "
            "evidence 89 and task 92, with maximum missing-evidence risk 0.71. " + summary["guard_note"],
            "",
            "Baselines are lab 80 historically and 81 at half a cent; native 68 historically "
            "and 74 at 36,000 tokens. The unchanged development floors supplied 22 lab and 35 native lines. "
            "A searching agent costs $0.029 to $0.039, takes about a minute and reads a median of 11 files "
            "and about 480 lines. Perfect-planner replay reaches 191 retrospectively and 174 with strict "
            "provenance. Those ceilings do not measure the cheap planner.",
            "",
            "Frozen free-source curves, source overlap, body bytes and estimated tokens are in "
            "[reach-summary.json](reach-summary.json). New delivered lines and their exact asked source "
            "pieces are audited against lab81 and native74 in "
            "[line-provenance-final.json](line-provenance-final.json).",
            "",
            summary["production_note"],
            "",
            summary["native_scope_note"],
            "",
            "## What we learned",
            "",
            learned_text(summary),
        ]
    )
    learned = (
        "# Case 2 trial lessons\n\n"
        "These are development observations from retained original union requests.\n\n"
        "## What we learned\n\n" + learned_text(summary) + "\n"
    )
    return "\n".join(lines) + "\n", learned


def main():
    sys.addaudithook(deny_network)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=lambda value: Path(value).expanduser())
    parser.add_argument("--paid", type=lambda value: Path(value).expanduser())
    args = parser.parse_args()
    paid = args.paid or args.stage.parent
    args.stage.mkdir(parents=True, exist_ok=True)
    per_finding, audits, facts, missing = collect(args.stage, paid)
    summary = trial_summary(args.stage, paid, per_finding, facts, missing, audits)
    write(args.stage / "trial-summary.json", summary)
    write(args.stage / "marginal-curve.json", {"development_measure": True, "curves": summary["curve"]})
    write(args.stage / "line-provenance-final.json", audits)
    csv_path = args.stage / "per-finding.csv"
    fields = list(per_finding[0]) if per_finding else ["case", "checkpoint_status"]
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_finding)
    trial, learned = markdown(summary, args.stage)
    (args.stage / "GENERATED-TRIAL.md").write_text(trial)
    (args.stage / "GENERATED-LEARNED.md").write_text(learned)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "packed_results": len(per_finding),
                "missing_cases": len(missing),
                "candidate_usd": summary["financials"]["candidate_jev_usd"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
