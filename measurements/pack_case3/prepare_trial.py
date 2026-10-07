"""Freeze the proposed development sample and an exact free rank request. Never contacts a provider."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from replay import EVAL, TAKEOVER, check_resources, load

from jev_navigator.batch import Operation, run_batch
from jev_navigator.batch_operations import _rank_output
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.answers import JevResponse, NoulAnswer

SEED = "case3-agent-trial-20261007"
QUOTAS = {"local": 6, "cross_boundary": 8, "discovered_terms": 4, "no_trace": 2}


def sample_ids(cases, commands, ledger):
    by_case = defaultdict(list)
    for row in ledger:
        if row["slice"] == "replay-dev110":
            by_case[row["case"]].append(row)
    pools = defaultdict(list)
    for case in cases:
        cid = case["case_id"]
        rows = by_case[cid]
        if cid not in commands:
            stratum = "no_trace"
        elif any(
            row.get("origin")
            in (
                "identifier or literal from earlier read",
                "path from earlier read or listing",
                "framework or instruction convention",
            )
            for row in rows
        ):
            stratum = "discovered_terms"
        elif any(
            row.get("operation")
            in (
                "named files and text",
                "references and callers",
                "definitions and callees",
                "spelling normalization",
            )
            for row in rows
        ):
            stratum = "cross_boundary"
        else:
            stratum = "local"
        pools[stratum].append(cid)
    selected = []
    for stratum, quota in QUOTAS.items():
        ordered = sorted(pools[stratum], key=lambda cid: hashlib.sha256(f"{SEED}:{cid}".encode()).hexdigest())
        if len(ordered) < quota:
            raise ValueError(f"not enough {stratum} cases")
        selected.extend({"case": cid, "stratum": stratum} for cid in ordered[:quota])
    return selected, {stratum: len(pool) for stratum, pool in pools.items()}


class Capture:
    """Capture Judge's real prepared request; dummy answers are never used as measurement."""

    model = "case3-prepare-stand-in"

    def __init__(self):
        self.requests = []

    def ask(self, state, questions):
        self.requests.append({"model": "jev-latest", "state": state, "questions": questions})
        return JevResponse({key: NoulAnswer(0.5) for key in questions}, self.model)


def main():
    from jev_navigator.judgments.judge import Judge

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    resources = check_resources()
    cases = load(EVAL / "cases/replay-dev110.json")
    commands = load(TAKEOVER / "discovery/commands.json")
    ledger = (json.loads(row) for row in (TAKEOVER / "discovery/lines.jsonl").read_text().splitlines())
    selected, population = sample_ids(cases, commands, ledger)
    packs = load(EVAL / "runs/pack-49b78955/pack-inputs-dev110.json")["cases"]
    manifest = {
        "seed": SEED,
        "quotas": QUOTAS,
        "population": population,
        "sample": selected,
        "measurement": "proposed development sample; trial not run",
        "resources": resources,
    }
    (args.out / "sample.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Agent input carries only its claim and valid masked checkout. No discovery route, label or answer.
    inputs = [{"case": row["case"], **packs[row["case"]]} for row in selected]
    (args.out / "agent-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    pack = inputs[0]
    with CodeIndex.from_git(Path(pack["repository"])) as tracked:
        files = [
            file
            for file in tracked.files
            if file not in pack["withheld"] and Path(file).name not in ("AGENTS.md", "CLAUDE.md")
        ]
        with CodeIndex(tracked.root, files, commit=tracked.commit) as index:
            cited = list(dict.fromkeys(e["file"] for e in pack["claim"]["evidence"]))
            candidates = []
            for file in cited:
                if file not in files:
                    continue
                for span in index.symbols_in(file):
                    for start in range(span.start, span.end + 1, 60):
                        candidates.append(Span(file, start, min(span.end, start + 59)))
                        if len(candidates) == 16:
                            break
                    if len(candidates) == 16:
                        break
                if len(candidates) == 16:
                    break
            capture = Capture()
            operation = Operation("rank", query=pack["claim"]["statement"], candidates=tuple(candidates))
            result = run_batch(index, [operation], judge=Judge(capture, max_calls=1))
            if not capture.requests or any(page.error for page in result.pages):
                raise RuntimeError("could not prepare a rank request")
            request = capture.requests[0]
    candidate = {
        "case_id": "case3-rank-preparation",
        "group_id": "case3-development",
        "revision_id": "batch-v1",
        "request": request,
        "intended_uses": {
            key: (
                "Order this supplied candidate by raw query-match probability. "
                "Return evidence locations without declaring the claim confirmed."
            )
            for key in request["questions"]
        },
        "workflow": {
            "purpose": "Return reusable per-candidate query-match probabilities to the searching agent.",
            "consumer_code": inspect.getsource(_rank_output),
        },
    }
    (args.out / "rank-candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    (args.out / "rank-operation.json").write_text(json.dumps(asdict(operation), indent=2) + "\n")
    print(
        json.dumps(
            {
                "sample": len(selected),
                "population": population,
                "prepared_items": len(request["state"]["items"]),
                "provider_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
