"""Bind fresh baseline observations to the exact physical requests Case 1 sent."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from find_eval.frozen_inputs import source_matches_masked

from jev_navigator.judgments.questions import request_sha256
from jev_navigator.judgments.secrets import SecretMasker

from .consume import baseline_scores


def load(path: Path):
    return json.loads(path.read_bytes())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_case(folder: Path) -> dict:
    context = load(folder / "context.json")
    shape = load(folder / "code-shape.json")["requests"]
    observations = load(folder / "code/observations.json")
    wires = {}
    for path in sorted((folder / "code/http").glob("jev_*/request.json")):
        request = load(path)
        key = request_sha256(request["state"], request["questions"])
        if key in wires:
            raise ValueError(f"Repeated baseline request in {folder}: {key}")
        wires[key] = path
    if len(shape) != len(wires) or len(shape) > 8:
        raise ValueError(f"Incomplete frozen baseline at {folder}")
    by_request = defaultdict(list)
    for row in observations:
        by_request[row["request_sha256"]].append(row)
    groups = []
    for ordinal, request in enumerate(shape):
        key = request_sha256(request["state"], request["questions"])
        path = wires[key]
        actual = load(path)
        if actual["state"] != request["state"] or actual["questions"] != request["questions"]:
            raise ValueError("Prepared group differs from the actual paid group")
        raw = load(path.with_name("response.json"))
        slots, rows = {}, []
        for point in request["state"]["targets"]:
            parts = [row for row in by_request[key] if row["point"] == point]
            if len(parts) != len(request["state"]["items"]):
                raise ValueError("Baseline observations do not cover every slot")
            for slot, row in enumerate(parts):
                if not source_matches_masked(
                    row["place"]["file"], request["state"]["items"][slot]["file"], row["place"]["file"]
                ):
                    raise ValueError("Baseline item and source address disagree")
                # Independently check the slot join against all six raw provider fields.
                for role, probability in row["probabilities"].items():
                    names = [
                        name
                        for name in request["questions"]
                        if name.startswith(f"{role}_{point}@") and name.endswith(f"#{slot}")
                    ]
                    if len(names) != 1 or raw["answers"][names[0]]["noul"] != probability:
                        raise ValueError("Baseline slot join does not reproduce the raw response")
                if slot in slots and slots[slot] != row["place"]:
                    raise ValueError("Source identity differs across points")
                slots[slot] = row["place"]
                rows.append(
                    {
                        **row,
                        "slot": slot,
                        "score": baseline_scores(row["probabilities"]),
                        "score_kind": "composed_role_probability",
                    }
                )
        groups.append(
            {
                "ordinal": ordinal,
                "request": request,
                "request_sha256": key,
                "wire_sha256": sha256(path),
                "response_sha256": sha256(path.with_name("response.json")),
                "receipt": load(path.with_name("receipt.json")),
                "places": slots,
                "baseline_rows": rows,
            }
        )
    return {
        "context": context,
        "groups": groups,
        "labels": load(folder / "labels.json"),
        "baseline_result": load(folder / "code/result.json"),
    }


def richer_state(context: dict) -> dict:
    """Only finder-supplied citations, never deciding labels, supply additional evidence."""
    cited = []
    root = Path(context["repository"])
    withheld = set(context["withheld"])
    for evidence in context["claim"].get("evidence", []):
        file, line = evidence["file"], evidence.get("line")
        if file in withheld:
            cited.append({"file": file, "line": line, "unavailable": "withheld in original corpus"})
            continue
        source = root / file
        if line is None or not source.is_file():
            cited.append({"file": file, "line": line, "unavailable": "no resolvable cited line"})
            continue
        lines = source.read_text().splitlines()
        if not 1 <= line <= len(lines):
            cited.append({"file": file, "line": line, "unavailable": "citation outside source"})
            continue
        start = end = line
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        cited.append(
            {
                "file": file,
                "cited_line": line,
                "first_line": start,
                "last_line": end,
                "code": SecretMasker().mask(body),
            }
        )
    return {"claim": context["claim"]["statement"], "cited_code": cited}


def hits(place: dict, label: dict) -> bool:
    return place["file"] == label["file"] and any(
        start <= label["first_line"] <= end for start, end in place["ranges"]
    )


def metrics(case: dict) -> dict:
    """Citation-based diagnostics, separated from evidence-independent semantic accuracy."""
    rows = [row for group in case["groups"] for row in group["baseline_rows"]]
    unique_places = {row["place"]["id"]: row["place"] for row in rows}
    deciding = {
        key for key, place in unique_places.items() if any(hits(place, label) for label in case["labels"])
    }
    high_places = {row["place"]["id"] for row in rows if row["score"] >= 0.80}
    high_rows = [row for row in rows if row["score"] >= 0.80]
    high_cited = sum(row["place"]["id"] in deciding for row in high_rows)
    per_point = {}
    for point in case["groups"][0]["request"]["state"]["targets"]:
        ranked = sorted((row for row in rows if row["point"] == point), key=lambda row: -row["score"])
        ranks = [i + 1 for i, row in enumerate(ranked) if row["place"]["id"] in deciding]
        per_point[point] = {
            "first_cited_deciding_rank": min(ranks, default=None),
            "recall16": sum(
                any(hits(row["place"], label) for row in ranked[:16]) for label in case["labels"]
            ),
            "recall32": sum(
                any(hits(row["place"], label) for row in ranked[:32]) for label in case["labels"]
            ),
        }
    return {
        "case": case["context"]["case"],
        "dataset": case["context"]["dataset"],
        "registered_lines": len(case["labels"]),
        "units": len(unique_places),
        "cited_deciding_units": len(deciding),
        "high_cited_deciding_units": len(high_places & deciding),
        "high_rows": len(high_rows),
        "high_cited_rows": high_cited,
        "judged_lines": sum(
            any(hits(place, label) for place in unique_places.values()) for label in case["labels"]
        ),
        "high_lines": sum(
            any(row["score"] >= 0.80 and hits(row["place"], label) for row in rows)
            for label in case["labels"]
        ),
        "per_point": per_point,
    }
