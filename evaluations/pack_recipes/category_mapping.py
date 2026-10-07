"""Join existing Jev category answers to the 46-subcategory recipe priors. No new labels."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from configurations import GUARD_CATEGORIES, PRESENCE_CATEGORIES, VALUE_CATEGORIES, category_recipes

ROOT = Path.home() / ".local/share/jvn-takeover/2026-10-03"
BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
EVAL = Path.home() / "Projects/heedvane-evals-theme-agent/experiments"
sys.path.insert(0, str(EVAL))
from theme_agent.e_classification.compose import RoutingPolicy, route_first_answers  # noqa: E402


def mapping():
    definitions = json.loads((ROOT / "categories/categories.json").read_text())
    cases = json.loads((BASE / "cases/replay-dev110.json").read_text())
    ids = {case["case_id"].removeprefix("analysis-engine:") for case in cases}
    choices = {}
    path = EVAL / "theme_agent/results/e-ae-scan-answers.jsonl"
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["case_id"] not in ids:
                continue
            if row["route"] == "templated-by-code":
                choices[row["case_id"]] = {"category": row["category"], "origin": "code template"}
                continue
            routed = route_first_answers(
                row["first"]["answers"], row["filed_lens"], RoutingPolicy(0.1, 0.3, 0.5, True)
            )
            stage = (
                "second" if routed.kind == "reclassify" and row.get("second_lens") == routed.lens else "first"
            )
            answer = row[stage]["answers"]["root_cause_category"]
            choices[row["case_id"]] = {
                "category": answer["choice"],
                "confidence": answer["confidence"],
                "origin": f"stored Jev {stage}",
            }
    counts = Counter(
        definitions["fine_to_subcategory"].get(row["category"], "unknown") for row in choices.values()
    )
    result = []
    for category in definitions["subcategories"]:
        identity = category["id"]
        blocks = [
            "same file and direct callees",
            "owner-qualified names and named files",
            "confirmed one hop",
        ]
        if identity in GUARD_CATEGORIES:
            blocks.append("downward guard chain with wiring attachments")
        if identity in VALUE_CATEGORIES:
            blocks.append("setting definition default environment override and docs")
        if identity in PRESENCE_CATEGORIES:
            blocks.append("literal presence in enumerated expected places")
        result.append(
            {
                "subcategory": identity,
                "name": category["name"],
                "recipes": category_recipes(identity),
                "blocks": blocks,
                "dev110_findings": counts[identity],
            }
        )
    return {
        "status": "development priors using mapped fine-category answers, not a measured 46-way Choice",
        "mapped": result,
        "answers": choices,
        "missing_category_answers": sorted(ids - choices.keys()),
        "none_or_unmapped": counts["unknown"],
        "mapped_count": sum(row["dev110_findings"] for row in result),
        "hard27": "No compatible stored category Choice. Do not invent categories for P/U findings.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(mapping(), indent=2) + "\n")
