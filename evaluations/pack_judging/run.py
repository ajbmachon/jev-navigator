"""Admitted paid stages, sharing Case 1's ledger, Jev transport and native owners."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from decimal import Decimal
from functools import cmp_to_key
from pathlib import Path
from unittest.mock import patch

import paid_client
from enginepy.workflows.document_analysis import evidence_pack as ep
from paid_client import JevSession
from paid_report import COMPARE, HARD_HISTORY, cases_only, historical_case
from paid_run import scopes
from paid_support import REQUESTY, CapStopError, Ledger, resources

from jev_navigator.judgments.answers import ChoiceAnswer, JevResponse, NoulAnswer
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.secrets import SecretScanner

from .consume import consume_jev, consume_llm, judge_group
from .designs import DESIGNS, identity, scan_llm_request
from .frozen import hits, load, sha256
from .packing import lab_rooms, native_pack, native_rooms
from .prepare import DEFAULT_CASE1, PLANNING, write
from .refresh import ADMITTED

OUT = PLANNING / "search-design/judging/paid-20261007"
BY_NAME = {d.name: d for d in DESIGNS}


class ProviderContractError(RuntimeError):
    """A systemic reply/usage failure stops dispatch after retaining the attempt."""


def credentials():
    for path in (Path.home() / "Projects/analysis-engine/.env", Path.home() / ".config/jgrep/env"):
        for line in path.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in {"REQUESTY_API_KEY", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL"}:
                os.environ.setdefault(key, value.strip().strip("\"'"))


def records(name):
    with gzip.open(ADMITTED / f"{name}-requests.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if identity(record["request"]) != record["request_sha256"]:
                raise ValueError("Prepared request identity changed")
            yield record


def folder(cid):
    return cid.replace(":", "_")


def bindings(case):
    return load(DEFAULT_CASE1 / "cases" / folder(case["context"]["case"]) / "code/units.json")


async def proof(cases):
    """Exercise a real frozen group through Judge inside Engine's async search boundary."""
    case = next(c for c in cases if c["context"]["case"] == "P1")
    context = case["context"]
    relations, index = scopes(context, {})
    source = bindings(case)
    receipts = []
    destination = OUT / "proof"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("J1-1", "J2-2", "J4-1"):
        design = BY_NAME[name]
        record = next(r for r in records(name) if r["case"] == "P1" and r["group"] == 0)
        request = record["request"]
        winner = len(request["state"]["items"]) - 1

        class Provider:
            model = "independent-port-fixture"

            def __init__(self):
                self.sent = []

            async def ask(self, state, questions, winner=winner):
                self.sent.append({"state": state, "questions": questions})
                answers = {
                    qid: (
                        ChoiceAnswer.from_probabilities(
                            {key: float(key == str(winner)) for key in q["criteria"]}
                        )
                        if q["type"] == "choice"
                        else NoulAnswer(0.99 if qid.endswith(f"#{winner}") else 0.01)
                    )
                    for qid, q in questions.items()
                }
                self.response = JevResponse(answers, self.model)
                return self.response

        provider = Provider()
        judge = Judge(provider, masker=None, max_calls=1)
        pack, result = await native_pack(
            relations, index, case, source, judge, design=design, requests={0: request}
        )
        assert provider.sent == [{"state": request["state"], "questions": request["questions"]}]
        assert pack is not None and not pack.failure and judge.calls == 1
        point = next(iter(result.targets))
        expected = case["groups"][0]["places"][str(winner)]["id"]
        ranking = ep.point_rankings(result, 1)[point]
        assert ranking[0].place == expected
        original = ep.pack_request(relations, context["repository"], context["claim"], context.get("points"))
        assert pack.floor == original.floor
        native = native_rooms(pack, relations, context, case["labels"], destination)
        history = historical_case(
            context,
            case["labels"],
            cases_only(COMPARE / "step-3/metadata.json"),
            cases_only(HARD_HISTORY / "windows.json"),
        )
        scalar = {0: consume_jev(design, request, provider.response)}
        lab = lab_rooms(history, source, case, scalar, index)
        assert set(lab) == set(native) == {"7200", "20000", "36000"}
        assert judge.calls == 1 and len(provider.sent) == 1
        receipts.append(
            {
                "design": name,
                "first_place": expected,
                "calls": 1,
                "floor_regions": len(pack.floor),
                "rooms": [7200, 20000, 36000],
                "source_binding": "pass",
                "scalar_rank": "pass",
                "floor": "pass",
                "packing_additional_calls": 0,
                "request_sha256": record["request_sha256"],
                "lab_rooms": lab,
                "native_rooms": native,
            }
        )
    write(OUT / "engine-proof.json", {"status": "PASS", "case": "P1", "receipts": receipts})


class LocatedReviewScanner:
    """Restore provenance for Meta's copies of exact, already located source bodies."""

    def __init__(self, prepared):
        self.paths = {}
        for row in prepared["requests"]:
            state = row.get("submitted", {}).get("state", {})
            original = state.get("request_state", {})
            for item in original.get("items", []):
                if "code" in item and "file" in item:
                    self.paths[item["code"]] = item["file"]
        self.scanner = SecretScanner()

    def findings(self, text, path=None):
        return self.scanner.findings(text, path or self.paths.get(text))


def guards(names, ledger):
    if load(OUT / "engine-proof.json")["status"] != "PASS":
        raise ValueError("Engine proof must pass first")
    sys.path.insert(0, str(Path.home() / "Projects/system-one-meta-builder/src"))
    from system_one_meta_builder import cli

    for name in names:
        for dataset in ("dev110", "hard27"):
            path = ADMITTED / f"{name}-{dataset}-candidate.json"
            destination = OUT / "guards" / f"{name}-{dataset}"
            destination.mkdir(parents=True, exist_ok=True)
            if (destination / "identity.json").exists() and load(destination / "identity.json")["exit"] in (
                0,
                3,
            ):
                continue
            if (destination / "report.json").exists():
                (destination / "report.json").rename(destination / "failed-report.json")
            prepared = load(path.with_name(path.stem + "-prepared.json"))
            allowed = {identity(row["submitted"]) for row in prepared["requests"] if "submitted" in row}
            session = JevSession(ledger, destination / "http", "guard")

            def call(submitted, questions, *, record_event, session=session, allowed=allowed):
                if identity(submitted) not in allowed:
                    raise ValueError("Guard request differs from the matching free preparation")
                raw, cached = session.ask_raw(submitted["state"], submitted["questions"])
                record_event(
                    {"event": "reserved_exact_wire_receipt", "folder": str(session.folder), "cached": cached}
                )
                return raw, session.receipts[-1]["seconds"] if session.receipts else 0

            cli.call_once = call
            output = io.StringIO()
            with (
                contextlib.redirect_stdout(output),
                patch.object(
                    paid_client, "SecretScanner", lambda prepared=prepared: LocatedReviewScanner(prepared)
                ),
            ):
                status = cli.main(["guard", str(path), "--output", str(destination / "receipts.jsonl")])
            (destination / "report.json").write_text(output.getvalue())
            write(destination / "identity.json", {"candidate_sha256": sha256(path), "exit": status})
            if status not in (0, 3):
                raise RuntimeError(f"Guard failed: {name}/{dataset}, exit {status}")
            print(
                json.dumps({"guard": name, "dataset": dataset, "balance_usd": str(ledger.balance())}),
                flush=True,
            )


def llm_call(record, destination, ledger):
    request = record["request"]
    scan_llm_request(request)
    wire = json.dumps(request, ensure_ascii=False).encode()
    rates = load(ADMITTED.parent / "provider-model.json")
    if rates["geolocation"] != "eu":
        raise ValueError("Expected the admitted EU route")
    resources()
    attempt = "llm:" + uuid.uuid4().hex
    reserve = Decimal(len(wire) + 1024) * Decimal(str(rates["input_price"]))
    reserve += Decimal(request["max_tokens"]) * Decimal(str(rates["output_price"]))
    ledger.reserve(attempt, "llm", reserve)
    write(
        destination / "attempt.json",
        {
            "id": attempt,
            "case": record.get("case"),
            "group": record.get("group"),
            "request_sha256": identity(request),
            "wire_sha256": hashlib.sha256(wire).hexdigest(),
            "reservation_usd": str(reserve),
            "model": request["model"],
        },
    )
    (destination / "request.json").write_bytes(wire)
    started = time.perf_counter()
    req = urllib.request.Request(
        REQUESTY + "/chat/completions",
        data=wire,
        headers={
            "Authorization": "Bearer " + os.environ["REQUESTY_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raw = error.read()
        (destination / "response.json").write_bytes(raw)
        write(
            destination / "failure-receipt.json",
            {
                "id": attempt,
                "status": error.code,
                "seconds": time.perf_counter() - started,
                "model": request["model"],
                "unpriced_reserved": True,
            },
        )
        raise ProviderContractError(f"HTTP {error.code}; response retained and bill reserved") from error
    except urllib.error.URLError as error:
        write(
            destination / "failure-receipt.json",
            {
                "id": attempt,
                "seconds": time.perf_counter() - started,
                "model": request["model"],
                "unpriced_reserved": True,
                "reason": str(error),
            },
        )
        raise ProviderContractError("Transport failed; bill remains reserved") from error
    (destination / "response.json").write_bytes(raw)
    reply = json.loads(raw)
    if not reply.get("usage") or not (reply.get("choices") or [{}])[0].get("message", {}).get("content"):
        write(
            destination / "failure-receipt.json",
            {
                "id": attempt,
                "seconds": time.perf_counter() - started,
                "model": reply.get("model"),
                "unpriced_reserved": True,
                "reason": "Missing answer content or usage",
            },
        )
        raise ProviderContractError("Provider returned no answer content or usage; stop dispatch")
    usage = reply["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    calculated = Decimal(usage["prompt_tokens"] - cached) * Decimal(str(rates["input_price"]))
    calculated += Decimal(cached) * Decimal(str(rates["cached_price"]))
    calculated += Decimal(usage["completion_tokens"]) * Decimal(str(rates["output_price"]))
    usd = Decimal(str(usage.get("cost", usage.get("total_cost", calculated))))
    elapsed = time.perf_counter() - started
    ledger.settle(attempt, usd, usage=usage, seconds=elapsed, model=reply.get("model"))
    write(
        destination / "receipt.json",
        {"id": attempt, "seconds": elapsed, "usage": usage, "usd": str(usd), "model": reply.get("model")},
    )
    choice = reply["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("Partial LLM output is not a selection")
    return consume_llm(
        choice["message"]["content"],
        json.loads(request["messages"][0]["content"].split("STATE_JSON:\n")[1])["targets"],
        len(json.loads(request["messages"][0]["content"].split("STATE_JSON:\n")[1])["items"]),
    )


async def evaluate(names, ledger, pilot=True):
    chosen = load(ADMITTED / "pilot.json")["groups"]
    for name in names:
        if not name.startswith("L1"):
            reviewed = load(OUT / "guard-hand-inspection.json")["prototypes"]
            admitted = {row["prototype"] for row in reviewed if row["admitted"]}
            if any(f"{name}-{dataset}" not in admitted for dataset in ("dev110", "hard27")):
                raise ValueError("Matching guards must be read by hand before candidate calls")
        semaphore = asyncio.Semaphore(4)
        halted = asyncio.Event()

        async def one(record, name=name, semaphore=semaphore, halted=halted):
            cid, ordinal = record["case"], record["group"]
            destination = OUT / "answers" / name / folder(cid) / str(ordinal)
            destination.mkdir(parents=True, exist_ok=True)
            result = destination / "observation.json"
            if result.exists():
                if load(result)["request_sha256"] != record["request_sha256"]:
                    raise ValueError("A different request cannot reuse the observation")
                return
            async with semaphore:
                if halted.is_set():
                    return
                start = time.perf_counter()
                started_at = time.time()
                try:
                    if name.startswith("L1"):
                        rows = await asyncio.to_thread(llm_call, record, destination, ledger)
                    else:
                        session = JevSession(ledger, destination / "http")
                        rows = await judge_group(
                            Judge(session, masker=None, max_calls=1), BY_NAME[name], record["request"]
                        )
                    failure = None
                    stop = None
                except CapStopError:
                    halted.set()
                    raise
                except Exception as error:
                    rows, failure = [], f"{type(error).__name__}: {error}"
                    stop = error if isinstance(error, ProviderContractError) else None
                    if stop is not None:
                        halted.set()
                write(
                    result,
                    {
                        "request_sha256": record["request_sha256"],
                        "case": cid,
                        "group": ordinal,
                        "rows": rows,
                        "seconds": time.perf_counter() - start,
                        "started_at": started_at,
                        "finished_at": time.time(),
                        "failure": failure,
                    },
                )
                if stop is not None:
                    raise stop

        work = [
            r
            for r in records(name)
            if r["eligible"] and (not pilot or r["group"] in chosen.get(r["case"], []))
        ]
        await asyncio.gather(*(one(record) for record in work))
        print(
            json.dumps(
                {"arm": name, "pilot": pilot, "groups": len(work), "balance_usd": str(ledger.balance())}
            ),
            flush=True,
        )


def observations(name, case):
    result = {}
    for group in case["groups"]:
        path = (
            OUT
            / "answers"
            / name
            / folder(case["context"]["case"])
            / str(group["ordinal"])
            / "observation.json"
        )
        if path.exists():
            result[group["ordinal"]] = load(path)["rows"]
    return result


def diagnostics(case, rows):
    bound = [
        dict(row, place=g["places"][str(row["slot"])])
        for g in case["groups"]
        for row in rows.get(g["ordinal"], [])
    ]
    labels = case["labels"]
    deciding = [r for r in bound if any(hits(r["place"], label) for label in labels)]
    high = [r for r in bound if r["score"] >= 0.80]
    ranks = sorted(bound, key=lambda r: -r["score"])
    witnesses = [i + 1 for i, r in enumerate(ranks) if any(hits(r["place"], label) for label in labels)]
    return {
        "cases": 1,
        "units": len(bound),
        "deciding_units": len(deciding),
        "high_deciding_units": sum(r["score"] >= 0.80 for r in deciding),
        "high_units": len(high),
        "registered_lines": len(labels),
        "judged_lines": sum(any(hits(r["place"], label) for r in bound) for label in labels),
        "high_lines": sum(any(hits(r["place"], label) for r in high) for label in labels),
        "recall16": sum(any(hits(r["place"], label) for r in ranks[:16]) for label in labels),
        "recall32": sum(any(hits(r["place"], label) for r in ranks[:32]) for label in labels),
        "first_rank": min(witnesses, default=None),
        "witness_points": bool(witnesses),
    }


def pilot_summary(cases):
    chosen = load(ADMITTED / "pilot.json")["groups"]
    per_case, total = {}, {}
    for name in BY_NAME:
        rows = {
            c["context"]["case"]: diagnostics(c, observations(name, c))
            for c in cases
            if c["context"]["case"] in chosen
        }
        if not any(row["units"] for row in rows.values()):
            continue
        per_case[name] = rows
        totals = {
            key: sum(row[key] for row in rows.values())
            for key in (
                "units",
                "deciding_units",
                "high_deciding_units",
                "high_units",
                "registered_lines",
                "judged_lines",
                "high_lines",
                "recall16",
                "recall32",
                "witness_points",
            )
        }
        totals["mean_first_rank_found"] = sum(row["first_rank"] or 0 for row in rows.values()) / max(
            1, totals["witness_points"]
        )
        totals["high_deciding_share"] = totals["high_deciding_units"] / max(1, totals["deciding_units"])
        total[name] = totals
    write(OUT / "pilot-summary.json", {"arms": total, "per_case": per_case})
    common_rich_comparisons(cases)
    print(json.dumps(total, indent=2))


def common_rich_comparisons(cases):
    selected = load(ADMITTED / "pilot.json")["groups"]
    comparisons = []
    for design in DESIGNS:
        if not design.richer:
            continue
        name = design.name
        parent = name.removeprefix("J3-")
        eligibility = {(r["case"], r["group"]): r["eligible"] for r in records(name)}
        pairs, exclusions = [], []
        for case in cases:
            cid = case["context"]["case"]
            if cid not in selected:
                continue
            rich = observations(name, case)
            if not rich:
                continue
            ordinary = observations(parent, case)
            common = {g for g in selected[cid] if eligibility[cid, g]}
            pairs.append(
                {
                    "case": cid,
                    "groups": sorted(common),
                    "parent": diagnostics(case, {g: ordinary[g] for g in common if g in ordinary}),
                    "rich": diagnostics(case, {g: rich[g] for g in common if g in rich}),
                }
            )
            for group in case["groups"]:
                if not eligibility[cid, group["ordinal"]]:
                    exclusions.append(
                        {
                            "case": cid,
                            "group": group["ordinal"],
                            "pilot": group["ordinal"] in selected[cid],
                            "deciding_lines": [
                                label
                                for label in case["labels"]
                                if any(hits(place, label) for place in group["places"].values())
                            ],
                        }
                    )
        if pairs:
            comparisons.append(
                {
                    "rich": name,
                    "parent": parent,
                    "scope": "pilot original common admissible groups",
                    "pairs": pairs,
                    "excluded_groups": exclusions,
                }
            )
    write(OUT / "j3-common-pilot.json", comparisons)


def primary_parent():
    summary = load(OUT / "pilot-summary.json")["arms"]

    def score(name):
        a = summary[name]
        return (a["high_deciding_share"], -a["mean_first_rank_found"], a["recall16"])

    return max((name for name in summary if name.startswith(("J1-", "J2-"))), key=score).split("-")[0]


def freeze():
    inspected = load(OUT / "pilot-hand-inspection.json")
    if len({row["case"] for row in inspected["cases"]}) < 10:
        raise ValueError("At least ten cases require actual hand inspection")
    summary = load(OUT / "pilot-summary.json")["arms"]
    price_rows = {row["design"]: row for row in load(ADMITTED / "prices.json")["arms"]}

    def order(name):
        a = summary[name]
        return (
            a["recall16"],
            a["witness_points"],
            0 if name.startswith("J4") else a["high_deciding_share"],
            -price_rows[name]["full_high_usd"],
        )

    families = defaultdict(list)
    for name in summary:
        families["J3" if name.startswith("J3") else name.split("-")[0]].append(name)
    best = {family: max(names, key=order) for family, names in families.items()}

    def compare(left, right):
        a, b = summary[left], summary[right]
        rank_a, rank_b = (a["recall16"], a["witness_points"]), (b["recall16"], b["witness_points"])
        if rank_a != rank_b:
            return (rank_a > rank_b) - (rank_a < rank_b)
        if (
            not left.startswith("J4")
            and not right.startswith("J4")
            and a["high_deciding_share"] != b["high_deciding_share"]
        ):
            return (a["high_deciding_share"] > b["high_deciding_share"]) - (
                a["high_deciding_share"] < b["high_deciding_share"]
            )
        cost_a, cost_b = price_rows[left]["full_high_usd"], price_rows[right]["full_high_usd"]
        return (cost_b > cost_a) - (cost_b < cost_a)

    jev = sorted(
        (name for family, name in best.items() if family != "L1"), key=cmp_to_key(compare), reverse=True
    )[:2]
    write(
        OUT / "finalists.json",
        {
            "jev": jev,
            "llm": best["L1"],
            "family_winners": best,
            "order": "recall16, witness coverage, high-band share except J4, cost",
            "pilot_inspection_sha256": sha256(OUT / "pilot-hand-inspection.json"),
        },
    )
    print(json.dumps(load(OUT / "finalists.json")))


async def report(cases):
    finalists = load(OUT / "finalists.json")
    names = [*finalists["jev"], finalists["llm"]]
    cached, outputs = {}, []
    dev = cases_only(COMPARE / "step-3/metadata.json")
    hard = cases_only(HARD_HISTORY / "windows.json")

    class Offline:
        model = "no-provider-calls-in-packing"

        def ask(self, *args):
            raise AssertionError("Packing tried to send")

    for case in cases:
        context = case["context"]
        relations, index = scopes(context, cached)
        history = historical_case(context, case["labels"], dev, hard)
        for name in names:
            destination = OUT / "packs" / name / folder(context["case"])
            destination.mkdir(parents=True, exist_ok=True)
            rows = observations(name, case)
            judge = Judge(Offline(), masker=None, max_calls=0)
            started = time.perf_counter()
            pack, result = await native_pack(relations, index, case, bindings(case), judge, rows)
            assert judge.calls == 0
            lab = lab_rooms(history, bindings(case), case, rows, index)
            native = native_rooms(pack, relations, context, case["labels"], destination)
            row = {
                "case": context["case"],
                "dataset": context["dataset"],
                "arm": name,
                "metrics": diagnostics(case, rows),
                "lab": lab,
                "native": native,
                "packing_seconds": time.perf_counter() - started,
                "no_anchor": pack is None,
                "provider_calls": 0,
                "failure": pack.failure if pack else None,
            }
            write(destination / "result.json", row)
            outputs.append(row)
        print(json.dumps({"packed": context["case"]}), flush=True)
    write(OUT / "per-finding-results.json", outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "proof",
            "guards",
            "pilot",
            "rich-guards",
            "rich-pilot",
            "summary",
            "freeze",
            "full",
            "report",
        ),
    )
    mode = parser.parse_args().mode
    OUT.mkdir(parents=True, exist_ok=True)
    credentials()
    ledger = Ledger(OUT / "spend-ledger.jsonl", cap="7.00")
    cases = load(ADMITTED / "frozen-cases.json")
    primary = [d.name for d in DESIGNS if not d.richer and d.family != "L1"]
    if mode == "proof":
        asyncio.run(proof(cases))
    elif mode == "guards":
        guards(primary, ledger)
    elif mode == "pilot":
        asyncio.run(evaluate([*primary, *(d.name for d in DESIGNS if d.family == "L1")], ledger))
        pilot_summary(cases)
    elif mode == "rich-guards":
        guards([d.name for d in DESIGNS if d.richer and d.family == primary_parent()], ledger)
    elif mode == "rich-pilot":
        asyncio.run(evaluate([d.name for d in DESIGNS if d.richer and d.family == primary_parent()], ledger))
        pilot_summary(cases)
    elif mode == "summary":
        pilot_summary(cases)
    elif mode == "freeze":
        freeze()
    elif mode == "full":
        chosen = load(OUT / "finalists.json")
        asyncio.run(evaluate([*chosen["jev"], chosen["llm"]], ledger, pilot=False))
    elif mode == "report":
        asyncio.run(report(cases))


if __name__ == "__main__":
    main()
