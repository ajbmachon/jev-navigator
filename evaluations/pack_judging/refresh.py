"""Apply the admitted design changes while retaining unchanged frozen artifacts."""

from __future__ import annotations

import contextlib
import gzip
import json
import os
from collections import defaultdict

from jev_navigator.judgments.client import JEV_INPUT_LIMITS

from .designs import DESIGNS, identity, jev_request, llm_request, scan_llm_request
from .frozen import hits, load, richer_state, sha256
from .prepare import PLANNING, free_meta, meta_candidate, price, write

PREVIOUS = PLANNING / "search-design/judging/design-20261007-admitted-v4"
ADMITTED = PLANNING / "search-design/judging/design-20261007-admitted-v5"


def main():
    if ADMITTED.exists():
        raise ValueError("Admitted requests must not be overwritten")
    ADMITTED.mkdir()
    previous = load(PREVIOUS / "prices.json")
    cases = load(PREVIOUS / "frozen-cases.json")
    selected = load(PREVIOUS / "pilot.json")
    pilot = {}
    for case in cases:
        cid = case["context"]["case"]
        if cid in selected["cases"]:
            first = next(
                (
                    g["ordinal"]
                    for g in case["groups"]
                    if any(hits(place, label) for place in g["places"].values() for label in case["labels"])
                ),
                None,
            )
            pilot[cid] = sorted({0, *([] if first is None else [first])})
    write(
        ADMITTED / "pilot.json",
        {
            **selected,
            "groups": pilot,
            "selection": "first plus first registered deciding group if different; labels never in state",
        },
    )
    for name in ("frozen-cases.json", "baseline-per-case.json", "baseline-summary.json"):
        os.link(PREVIOUS / name, ADMITTED / name)
    rates = load(ADMITTED.parent / "provider-model.json")
    arms, meta = [], []
    for design in DESIGNS:
        filename = f"{design.name}-requests.jsonl.gz"
        changed = design.family == "L1"
        if not changed:
            os.link(PREVIOUS / filename, ADMITTED / filename)
        totals, violations, done = defaultdict(float), [], set()
        with contextlib.ExitStack() as stack:
            old = stack.enter_context(gzip.open(PREVIOUS / filename, "rt"))
            records = iter(json.loads(line) for line in old)
            new = (
                stack.enter_context(gzip.open(ADMITTED / filename, "wt", compresslevel=1))
                if changed
                else None
            )
            try:
                for case in cases:
                    context = case["context"]
                    rich = richer_state(context) if design.richer else None
                    for group in case["groups"]:
                        record = next(records)
                        assert (record["case"], record["group"]) == (context["case"], group["ordinal"])
                        if changed:
                            request = (
                                llm_request(design, group["request"])
                                if design.family == "L1"
                                else jev_request(design, group["request"], rich)
                            )
                            if design.family == "L1":
                                scan_llm_request(request)
                            record.update(
                                request=request,
                                request_sha256=identity(request),
                                eligible=design.family == "L1"
                                or not JEV_INPUT_LIMITS.exceeded_by(
                                    request.get("state", {}), request.get("questions", {})
                                ),
                                price=price(request, design.family == "L1", rates),
                            )
                            new.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if not record["eligible"]:
                            violations.append({"case": context["case"], "group": group["ordinal"]})
                        for field in ("low_usd", "high_usd", "reservation_usd"):
                            totals["full_" + field] += record["price"][field]
                            if group["ordinal"] in pilot.get(context["case"], []):
                                totals["pilot_" + field] += record["price"][field]
                        if design.family == "L1" or context["dataset"] in done:
                            continue
                        prototype = next(
                            m
                            for m in previous["meta"]
                            if m["design"] == design.name and m["dataset"] == context["dataset"]
                        )
                        if (prototype["case"], prototype["group"]) != (context["case"], group["ordinal"]):
                            continue
                        candidate = ADMITTED / f"{design.name}-{context['dataset']}-candidate.json"
                        prepared = candidate.with_name(candidate.stem + "-prepared.json")
                        if changed:
                            write(
                                candidate,
                                meta_candidate(
                                    design, record["request"], context["dataset"], group["ordinal"]
                                ),
                            )
                            review = free_meta(candidate, prepared)
                            meta.append(
                                {
                                    **prototype,
                                    **review,
                                    "candidate_sha256": sha256(candidate),
                                    "request_sha256": record["request_sha256"],
                                }
                            )
                        else:
                            os.link(PREVIOUS / candidate.name, candidate)
                            os.link(PREVIOUS / prepared.name, prepared)
                            meta.append(prototype)
                        done.add(context["dataset"])
            finally:
                if new:
                    new.close()
        arms.append(
            {
                "design": design.name,
                **totals,
                "box_violations": violations,
                "archive_sha256": sha256(ADMITTED / filename),
                "reprepared": changed,
            }
        )
        print(
            json.dumps({"design": design.name, "changed": changed, "outside_box": len(violations)}),
            flush=True,
        )
    write(ADMITTED / "prices.json", {"cap_usd": 7, "arms": arms, "meta": meta})
    write(
        ADMITTED / "manifest.json",
        {
            "status": "ADMITTED_BY_LEAD",
            "lead_check_sha256": sha256(ADMITTED.parent / "LEAD-DESIGN-CHECK.md"),
            "groups": 984,
            "pilot_groups": sum(map(len, pilot.values())),
            "provider_calls": 0,
            "artifacts": {p.name: sha256(p) for p in ADMITTED.iterdir() if p.is_file()},
        },
    )


if __name__ == "__main__":
    main()
