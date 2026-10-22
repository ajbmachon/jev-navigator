"""Zero-cost freeze, candidate generation, free Meta prepare and explicit pricing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.questions import serialized_chars
from jev_navigator.judgments.secrets import SecretScanner, refuse_if_secret

from .consume import consume_jev
from .designs import DESIGNS, identity, jev_request, llm_request, scan_llm_request
from .frozen import freeze_case, load, metrics, richer_state, sha256

PLANNING = Path.home() / ".local/share/jvn-takeover/2026-10-03"
DEFAULT_CASE1 = PLANNING / "search-design/case1/paid-trial-20261007"
DEFAULT_OUT = PLANNING / "search-design/judging/design-20261007"


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def pilot_cases(cases: list[dict]) -> tuple[set[str], dict]:
    """All thirteen allowed hard cases plus seventeen dev cases across input-derived strata."""
    hard = {case["context"]["case"] for case in cases if case["context"]["dataset"] == "hard27"}
    expected = {*(f"P{i}" for i in range(1, 8)), *(f"U{i}" for i in range(1, 7))}
    if hard != expected:
        raise ValueError("Hard tuning population differs from the authorized P/U set")
    strata = defaultdict(list)
    for case in cases:
        context = case["context"]
        if context["dataset"] != "dev110":
            continue
        files = [e["file"] for e in context["claim"].get("evidence", [])]
        language = "python" if any(file.endswith(".py") for file in files) else "other"
        absence = bool(
            re.search(
                r"\b(missing|absent|never|no |does not|without|only)\b", context["claim"]["statement"], re.I
            )
        )
        stratum = f"{context['case'].split(':')[0]}/{language}/{absence}"
        strata[stratum].append(context["case"])
    for ids in strata.values():
        ids.sort(key=lambda cid: hashlib.sha256(cid.encode()).hexdigest())
    chosen, assignment = set(hard), {cid: "hard27/all_allowed" for cid in hard}
    while len(chosen) < 30:
        for stratum in sorted(strata):
            if strata[stratum] and len(chosen) < 30:
                cid = strata[stratum].pop(0)
                chosen.add(cid)
                assignment[cid] = stratum
    return chosen, assignment


def meta_candidate(design, request, dataset, group):
    return {
        "case_id": f"judging-{dataset}-{group}",
        "revision_id": design.name,
        "request": request,
        "intended_uses": {
            qid: (
                "Retain raw typed observations for this exact group. Rank by the Noul probability "
                "or Choice option probability. The 0.80 diagnostic is not calibrated accuracy. "
                "Choice winner confirmation separately uses provider confidence at 0.70. "
                "Select supplied source for pack assembly, never a verdict on the finding."
            )
            for qid in request["questions"]
        },
        "workflow": {
            "purpose": "Rank the frozen units for the existing lab and Engine pack allocators.",
            # Supply the real JVN consumer, even when Meta cannot bind its dynamic question IDs.
            "consumer_code": inspect.getsource(consume_jev),
        },
    }


def free_meta(candidate: Path, destination: Path) -> dict:
    result = subprocess.run(["system-one-meta-builder", "prepare", str(candidate)], capture_output=True)
    destination.write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Free Meta prepare failed for {candidate}: {result.stderr.decode()}")
    prepared = json.loads(result.stdout)
    requests = prepared["requests"]
    unresolved = [
        path
        for row in requests
        for path in row.get("state_path_coverage", {}).get("unresolved_backticked_paths", [])
    ]
    if unresolved:
        raise ValueError(f"Unresolved state paths: {unresolved}")
    bodies = [row["submitted"] for row in requests if "submitted" in row]
    return {
        "review_requests": len(bodies),
        "prepare_sha256": sha256(destination),
        "review_bytes": sum(len(json.dumps(body, ensure_ascii=False).encode()) for body in bodies),
        "forecast_usd": sum((254 + 0.30 * len(json.dumps(body).encode())) * 0.042 / 1e6 for body in bodies),
        "reservation_usd": sum((len(json.dumps(body).encode()) + 1024) * 0.042 / 1e6 for body in bodies),
        "atomicity": "not_reviewed_without_source_backed_independent_proposition_pair",
        "workflow_coverage": [
            row.get("workflow_coverage", {}) for row in requests if row.get("scope") == "workflow"
        ],
        "unreviewed": [row for row in requests if "submitted" not in row],
    }


def price(request: dict, llm: bool, rates: dict) -> dict:
    wire_bytes = len(json.dumps(request, ensure_ascii=False).encode())
    if llm:
        # Derived from the *measured* Case 1 Requesty input density, without assuming cache hits.
        input_bytes = len(request["messages"][0]["content"].encode())
        output = request["max_tokens"]
        return {
            "bytes": wire_bytes,
            "low_usd": input_bytes * 0.284 * rates["input_price"] + output * rates["output_price"],
            "high_usd": input_bytes * 0.335 * rates["input_price"] + output * rates["output_price"],
            "reservation_usd": (wire_bytes + 1024) * rates["input_price"] + output * rates["output_price"],
        }
    body_bytes = len(json.dumps({"state": request["state"], "questions": request["questions"]}).encode())
    return {
        "bytes": wire_bytes,
        "low_usd": (254 + 0.23 * body_bytes) * 0.042 / 1e6,
        "high_usd": (254 + 0.30 * body_bytes) * 0.042 / 1e6,
        "reservation_usd": (wire_bytes + 1024) * 0.042 / 1e6,
    }


def aggregate_metrics(rows):
    result = {}
    for dataset in ("dev110", "hard27"):
        values = [row for row in rows if row["dataset"] == dataset]
        totals = {
            key: sum(row[key] for row in values)
            for key in (
                "registered_lines",
                "units",
                "cited_deciding_units",
                "high_cited_deciding_units",
                "high_rows",
                "high_cited_rows",
                "judged_lines",
                "high_lines",
            )
        }
        totals.update(
            cases=len(values),
            deciding_unit_share_at_080=totals["high_cited_deciding_units"] / totals["cited_deciding_units"],
            citation_precision_at_080=totals["high_cited_rows"] / totals["high_rows"],
        )
        result[dataset] = totals
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case1", type=Path, default=DEFAULT_CASE1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a fresh output directory; frozen requests must not be overwritten")
    args.out.mkdir(parents=True)
    version = subprocess.check_output(["system-one-meta-builder", "--version"], text=True).strip()
    cases = [freeze_case(folder) for folder in sorted((args.case1 / "cases").iterdir())]
    if len(cases) != 123:
        raise ValueError("Expected the complete authorized 123-case population")
    selected, strata = pilot_cases(cases)
    # Read-only catalogue refresh is separate from preparation. A saved current EU catalogue is required.
    rates = load(args.out.parent / "provider-model.json")
    if rates["id"] != "sference/deepseek-v4-flash-0731" or rates["geolocation"] != "eu":
        raise ValueError("Model catalogue does not identify the authorized EU model")
    write(
        args.out / "pilot.json",
        {
            "cases": sorted(selected),
            "strata": strata,
            "selection": "all 13 P/U plus 17 dev, input-derived round-robin strata",
        },
    )
    baseline_metrics = [metrics(case) for case in cases]
    write(args.out / "baseline-per-case.json", baseline_metrics)
    write(args.out / "baseline-summary.json", aggregate_metrics(baseline_metrics))
    write(args.out / "frozen-cases.json", cases)
    price_rows, prototypes = [], []
    scanner = SecretScanner()
    for design in DESIGNS:
        totals = defaultdict(float)
        violations, done = [], set()
        stream_path = args.out / f"{design.name}-requests.jsonl.gz"
        with gzip.open(stream_path, "wt", encoding="utf-8", compresslevel=1) as stream:
            for case in cases:
                context = case["context"]
                rich = richer_state(context) if design.richer else None
                for group in case["groups"]:
                    request = (
                        llm_request(design, group["request"])
                        if design.family == "L1"
                        else jev_request(design, group["request"], rich)
                    )
                    eligible = design.family == "L1" or not JEV_INPUT_LIMITS.exceeded_by(
                        request["state"], request["questions"]
                    )
                    if not eligible:
                        violations.append(
                            {
                                "case": context["case"],
                                "group": group["ordinal"],
                                "state_chars": serialized_chars(request["state"]),
                            }
                        )
                    # Existing final scanner is reused; no copied secret handling.
                    if group["ordinal"] == 0:
                        if design.family == "L1":
                            scan_llm_request(request)
                        else:
                            refuse_if_secret(request, {}, scanner)
                    cost = price(request, design.family == "L1", rates)
                    record = {
                        "case": context["case"],
                        "dataset": context["dataset"],
                        "group": group["ordinal"],
                        "baseline_sha256": group["request_sha256"],
                        "request_sha256": identity(request),
                        "eligible": eligible,
                        "price": cost,
                        "request": request,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    for field in ("low_usd", "high_usd", "reservation_usd"):
                        totals["full_" + field] += cost[field]
                        if group["ordinal"] < 4:
                            totals["first4_" + field] += cost[field]
                        if context["case"] in selected and group["ordinal"] == 0:
                            totals["pilot_" + field] += cost[field]
                    totals["groups"] += 1
                    if (
                        design.family != "L1"
                        and context["dataset"] not in done
                        and context["case"] in selected
                        and len(request["state"]["items"]) == 16
                        and eligible
                    ):
                        prototype = args.out / f"{design.name}-{context['dataset']}-candidate.json"
                        candidate = meta_candidate(design, request, context["dataset"], group["ordinal"])
                        write(prototype, candidate)
                        review = free_meta(prototype, prototype.with_name(prototype.stem + "-prepared.json"))
                        prototypes.append(
                            {
                                "design": design.name,
                                "dataset": context["dataset"],
                                "case": context["case"],
                                "group": group["ordinal"],
                                "candidate_sha256": sha256(prototype),
                                "request_sha256": identity(request),
                                **review,
                            }
                        )
                        done.add(context["dataset"])
        price_rows.append(
            {
                "design": design.name,
                **totals,
                "box_violations": violations,
                "archive_sha256": sha256(stream_path),
            }
        )
        print(
            json.dumps(
                {
                    "design": design.name,
                    "box_violations": len(violations),
                    "full_high_usd": totals["full_high_usd"],
                }
            ),
            flush=True,
        )
    write(
        args.out / "prices.json",
        {
            "cap_usd": 4.0,
            "arms": price_rows,
            "meta": prototypes,
            "meta_high_usd": sum(row["forecast_usd"] for row in prototypes),
            "meta_reservation_usd": sum(row["reservation_usd"] for row in prototypes),
            "price_policy": "no cache discount; sensitivity forecasts are not bills",
        },
    )
    write(
        args.out / "manifest.json",
        {
            "status": "STOP_FOR_LEAD_DESIGN_CHECK",
            "provider_calls": 0,
            "spend_usd": 0,
            "recorded": datetime.now().astimezone().isoformat(),
            "meta_version": version,
            "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "case1": str(args.case1),
            "case1_summary_sha256": sha256(args.case1 / "summary.json"),
            "groups": sum(len(case["groups"]) for case in cases),
            "group_sizes": dict(
                Counter(len(group["request"]["state"]["items"]) for case in cases for group in case["groups"])
            ),
            "candidate_designs": len(DESIGNS),
            "pilot_findings": len(selected),
            "artifacts": {path.name: sha256(path) for path in args.out.iterdir() if path.is_file()},
            "label_limit": "Citations are not exhaustive negative labels. All current cases have one point; "
            "future multi-point precision needs independent point-to-citation labels.",
        },
    )
    print("STOP: free preparation complete; no guard or candidate calls authorized in this phase.")


if __name__ == "__main__":
    main()
