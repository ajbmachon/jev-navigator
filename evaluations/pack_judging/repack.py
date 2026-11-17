"""Replay the four frozen judges on a separately selected scratch Engine, with no network."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from enginepy.workflows.document_analysis import evidence_pack as ep
from find_eval.simulate import proposal_reducer
from paid_native import build
from paid_report import COMPARE, HARD_HISTORY, cases_only, historical_case
from paid_report import lab_rooms as baseline_lab_rooms
from paid_run import scopes
from paid_support import resources

from jev_navigator.judgments.judge import Judge

from .frozen import load, sha256
from .packing import lab_rooms, native_pack, native_rooms
from .prepare import DEFAULT_CASE1, PLANNING, write
from .refresh import ADMITTED
from .run import OUT, bindings, folder, observations

FOLLOWUP = PLANNING / "search-design/judging/engine-fixed-20261007"


class NoProvider:
    model = "offline-frozen-observations"

    def ask(self, *args, **kwargs):
        raise AssertionError("Re-packing attempted inference")


def offline_owner():
    path = PLANNING / "search-design/case1/engine-leak-20261007/replay.py"
    spec = importlib.util.spec_from_file_location("case1_leak_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This existing replay owner installs its socket.connect/getaddrinfo audit refusal.
    return module


async def run(variant, limit=None):
    owner = offline_owner()
    engine = Path(ep.__file__).resolve()
    assert "analysis-engine-theme-agent-last-phase" not in str(engine)
    if variant == "fixed":
        assert engine.is_relative_to((FOLLOWUP / "scratch").resolve())
        assert hasattr(ep.EvidencePack, "packing_units")
    else:
        assert not hasattr(ep.EvidencePack, "packing_units")
    cases = load(ADMITTED / "frozen-cases.json")
    baseline = {r["case"]: r for r in load(DEFAULT_CASE1 / "per-finding-summary.json")}
    previous = {(r["case"], r["arm"]): r for r in load(OUT / "per-finding-results.json")}
    dev = cases_only(COMPARE / "step-3/metadata.json")
    hard = cases_only(HARD_HISTORY / "windows.json")
    reducer = proposal_reducer(COMPARE / "step-1b/roles-PROPOSAL.md")
    required = load(COMPARE / "step-2/selection-policy.json")["required_roles"]
    totals = defaultdict(lambda: defaultdict(int))
    cache, results = {}, []
    raw_lookup_count = 0
    for case in cases[:limit]:
        resources()
        original = case["context"]
        # Target identity is frozen with the answers, including the original no-anchor fallback.
        context = {**original, "points": case["groups"][0]["request"]["state"]["targets"]}
        replay_case = {**case, "context": context}
        cid, dataset = context["case"], context["dataset"]
        relations, index = scopes(context, cache)
        source = bindings(case)
        history = historical_case(context, case["labels"], dev, hard)
        for arm in ("R0", "J1-3", "J4-1", "L1-3"):
            destination = FOLLOWUP / variant / "packets" / folder(cid) / arm
            destination.mkdir(parents=True, exist_ok=True)
            if arm == "R0":
                prior = DEFAULT_CASE1 / "cases" / folder(cid) / "code"
                identifiers = load(prior / "selected-candidates.json")
                candidates = [
                    {"unit": unit, "candidate_id": identifier}
                    for unit, identifier in zip(source, identifiers, strict=True)
                ]
                answers = owner.OriginalAnswers(prior / "http")
                judge = Judge(answers, max_calls=8, items_per_request=16, max_concurrency=1)
                pack, result = await build(relations, index, context, candidates, judge)
                expected = [
                    {
                        "key": hashlib.sha256(
                            json.dumps(
                                {"state": g["request"]["state"], "questions": g["request"]["questions"]},
                                ensure_ascii=False,
                            ).encode()
                        ).hexdigest(),
                        "items": len(g["request"]["state"]["items"]),
                        "questions": len(g["request"]["questions"]),
                    }
                    for g in case["groups"]
                ]
                assert answers.requests == expected, (cid, "R0 changed original groups")
                raw_lookup_count += len(answers.requests)
                lab = baseline_lab_rooms(
                    history, source, load(prior / "observations.json"), index, reducer, required
                )
                old_lab = baseline[cid]["arms"]["code"]["lab_rooms"]
                old_native = baseline[cid]["arms"]["code"]["rooms"]
            else:
                rows = observations(arm, case)
                assert len(rows) == len(case["groups"])
                judge = Judge(NoProvider(), masker=None, max_calls=0)
                pack, result = await native_pack(relations, index, replay_case, source, judge, rows)
                assert judge.calls == 0
                lab = lab_rooms(history, source, case, rows, index)
                old_lab, old_native = previous[cid, arm]["lab"], previous[cid, arm]["native"]
            assert result.failure is None
            assert pack is None or not pack.failure
            native = native_rooms(pack, relations, context, case["labels"], destination)
            paired = {}
            for room in ("7200", "20000", "36000"):
                literal = owner.reader((destination / f"native-{room}.txt").read_text())
                assert native[room]["delivered"] == [
                    (label["file"], label["first_line"]) in literal for label in case["labels"]
                ], (cid, arm, room, "literal reader disagrees")
                delivered = (
                    lab[room]["delivered"]
                    if arm == "R0"
                    else [label["delivered"] for label in lab[room]["labels"]]
                )
                old_delivered = (
                    old_lab[room]["delivered"]
                    if arm == "R0"
                    else [label["delivered"] for label in old_lab[room]["labels"]]
                )
                assert delivered == old_delivered, (cid, arm, room, "lab changed")
                assert native[room]["within_room"]
                before = old_native[room]["delivered"]
                after = native[room]["delivered"]
                assert all(not b or a for b, a in zip(before, after, strict=True)), (
                    cid,
                    arm,
                    room,
                    "original deciding line lost",
                )
                if variant == "before":
                    original_packet = (
                        DEFAULT_CASE1 / "cases" / folder(cid) / "code" / f"packet-{room}.txt"
                        if arm == "R0"
                        else OUT / "packs" / arm / folder(cid) / f"native-{room}.txt"
                    )
                    assert (destination / f"native-{room}.txt").read_bytes() == original_packet.read_bytes()
                paired[room] = {
                    "kept": sum(b and a for b, a in zip(before, after, strict=True)),
                    "gained": sum(not b and a for b, a in zip(before, after, strict=True)),
                    "lost": sum(b and not a for b, a in zip(before, after, strict=True)),
                }
                totals[f"{dataset}:{arm}"][f"lab:{room}"] += sum(delivered)
                totals[f"{dataset}:{arm}"][f"native:{room}"] += sum(after)
                totals[f"{dataset}:{arm}"][f"before_native:{room}"] += sum(before)
            results.append(
                {
                    "case": cid,
                    "dataset": dataset,
                    "arm": arm,
                    "lab": lab,
                    "native": native,
                    "paired": paired,
                    "provider_calls": 0,
                    "no_anchor": pack is None,
                }
            )
        print(f"{variant}: {cid}", flush=True)
    write(FOLLOWUP / variant / "results.json", results)
    write(
        FOLLOWUP / variant / "summary.json",
        {
            "status": "PASS",
            "provider_calls": 0,
            "usd": 0,
            "engine_file": str(engine),
            "engine_sha256": sha256(engine),
            "rows": len(results),
            "r0_exact_group_lookups": raw_lookup_count,
            "literal_native_packets": len(results) * 3,
            "all_lab_delivery_unchanged": True,
            "all_original_native_deciding_lines_preserved": True,
            "totals": {k: dict(v) for k, v in totals.items()},
        },
    )


def audit():
    """Independently read every literal packet and compare fixed judges on the same labels."""
    owner = offline_owner()
    cases = {case["context"]["case"]: case for case in load(ADMITTED / "frozen-cases.json")}
    baseline = load(DEFAULT_CASE1 / "per-finding-summary.json")
    prices = load(OUT / "summary.json")["arms"]
    patched = {}
    literal_count, max_chars = 0, 0
    for variant in ("before", "fixed"):
        summary = load(FOLLOWUP / variant / "summary.json")
        results = load(FOLLOWUP / variant / "results.json")
        assert summary["provider_calls"] == summary["usd"] == 0
        assert summary["rows"] == len(results) == 492
        assert summary["r0_exact_group_lookups"] == 984
        assert len({(row["case"], row["arm"]) for row in results}) == 492
        for row in results:
            assert row["provider_calls"] == 0
            if variant == "fixed":
                patched[row["case"], row["arm"]] = row
            for room in ("7200", "20000", "36000"):
                packet = (
                    FOLLOWUP / variant / "packets" / folder(row["case"]) / row["arm"] / f"native-{room}.txt"
                )
                text = packet.read_text()
                shown = owner.reader(text)
                assert len(text) == row["native"][room]["chars"] <= int(room) * 4
                assert row["native"][room]["delivered"] == [
                    (label["file"], label["first_line"]) in shown for label in cases[row["case"]]["labels"]
                ]
                literal_count += 1
                max_chars = max(max_chars, len(text))
    totals = load(FOLLOWUP / "fixed/summary.json")["totals"]
    assert [totals["dev110:R0"][f"native:{room}"] for room in (7200, 20000, 36000)] == [77, 109, 129]
    assert [totals["hard27:R0"][f"native:{room}"] for room in (7200, 20000, 36000)] == [14, 13, 14]
    costs, paired, pooled = {}, defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for key, values in totals.items():
        dataset, arm = key.split(":")
        cost = (
            sum(
                (Decimal(row["arms"]["code"]["usd"]) for row in baseline if row["dataset"] == dataset),
                Decimal(0),
            )
            if arm == "R0"
            else Decimal(str(prices[key]["invoice"]["usd"]))
        )
        costs[key] = {
            "inference_usd": str(cost),
            "fixed_native_lines_per_dollar": {
                str(room): float(Decimal(values[f"native:{room}"]) / cost) for room in (7200, 20000, 36000)
            },
        }
        pooled[arm]["inference_usd"] += cost
        for room in (7200, 20000, 36000):
            pooled[arm][f"native:{room}"] += values[f"native:{room}"]
    for (cid, arm), row in patched.items():
        if arm == "R0":
            continue
        for room in ("7200", "20000", "36000"):
            before = patched[cid, "R0"]["native"][room]["delivered"]
            after = row["native"][room]["delivered"]
            for b, a in zip(before, after, strict=True):
                paired[f"{row['dataset']}:{arm}:{room}"][
                    "kept" if b and a else "gained" if a else "lost" if b else "neither"
                ] += 1
    for values in pooled.values():
        cost = values["inference_usd"]
        values["inference_usd"] = str(cost)
        values["fixed_native_lines_per_dollar"] = {
            str(room): float(Decimal(values[f"native:{room}"]) / cost) for room in (7200, 20000, 36000)
        }
    identities = {
        str(path.relative_to(PLANNING)): sha256(path)
        for path in (
            *(ADMITTED / f"{arm}-requests.jsonl.gz" for arm in ("J1-3", "J4-1", "L1-3")),
            OUT / "artifact-index.json",
            *(FOLLOWUP / variant / "results.json" for variant in ("before", "fixed")),
            *(
                PLANNING / "search-design/case1" / name
                for name in ("01-engine-packet-candidates.patch", "02-engine-generated-citation.patch")
            ),
        )
    }
    write(
        FOLLOWUP / "audit.json",
        {
            "status": "PASS",
            "provider_calls": 0,
            "usd": 0,
            "literal_packets_verified": literal_count,
            "max_packet_chars": max_chars,
            "unpatched_packets_identical_to_original": 1476,
            "baseline_matches_case1_leak_followup": True,
            "costs": costs,
            "pooled": dict(pooled),
            "fixed_judges_paired_against_fixed_R0": dict(paired),
            "identities": identities,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("before", "fixed", "audit"))
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    if arguments.variant == "audit":
        audit()
    else:
        asyncio.run(run(arguments.variant, arguments.limit))
