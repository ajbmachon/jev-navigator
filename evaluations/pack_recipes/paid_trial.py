"""Authorized v2 recipe trial through the actual Engine pack entry point.

One primary recipe per finding, eight physical requests at most, one shared cap.
All rooms are repacked from saved answers without another send.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from pathlib import Path

from paid_budget import SpendLedger, SpendStopError
from replay import BASE, CandidateSource, load, request_key, rows
from resources import check_resources


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class PaidClient:
    model = "jev-latest"

    def __init__(self, ledger, cid, folder):
        self.ledger, self.cid, self.folder = ledger, cid, folder
        self.lock = threading.RLock()
        self.groups = []
        self.calls = 0
        folder.mkdir(parents=True, exist_ok=True)
        self.cached = {}
        # A response hook saves the raw receipt before SDK parsing. Recover it
        # even if the process stopped before appending a completed group.
        journal = folder / "groups.jsonl"
        if journal.exists():
            self.cached.update((group["key"], group) for group in rows(journal))
        for source in folder.glob("*.request.json"):
            response_path = source.with_name(source.name.replace(".request.json", ".response.json"))
            if response_path.exists():
                group = load(source)
                group["response"] = load(response_path)
                group["usd"] = group["response"]["usage"]["input_tokens"] * 0.000000042
                if set(group["response"]["answers"]) == set(group["questions"]):
                    self.cached.setdefault(group["key"], group)
            self.calls += 1

    def ask(self, state, questions):
        import httpx2
        from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

        from jev_navigator.judgments.answers import response_from_raw
        from jev_navigator.judgments.secrets import SecretScanner, refuse_if_secret

        key = request_key(state, questions)
        with self.lock:
            if key in self.cached:
                group = self.cached[key]
                self.groups.append(group)
                return response_from_raw(group["response"])
            if self.calls >= 8:
                raise SpendStopError("Eight physical sends already reached for this finding")
            if not 1 <= len(state["items"]) <= 16:
                raise ValueError("Prepared shape is one to sixteen ordered units")
            refuse_if_secret(state, questions, SecretScanner())
            ticket = self.ledger.reserve("jev", self.cid, key)
            self.calls += 1
            number = self.calls
        prefix = self.folder / f"{number:02d}"
        group = {"state": state, "questions": questions, "key": key, "number": number}
        save(prefix.with_suffix(".request.json"), group)
        accounted = False
        started = time.monotonic()

        def request_hook(request):
            prefix.with_suffix(".request.bin").write_bytes(request.read())

        def response_hook(response):
            nonlocal accounted
            prefix.with_suffix(".response.bin").write_bytes(response.read())
            raw = json.loads(response.read())
            save(prefix.with_suffix(".response.json"), raw)
            if "usage" in raw:
                accounted = True
                group["usd"] = self.ledger.settle(
                    ticket, raw, category="jev", case=self.cid, request_sha256=key, number=number
                )

        try:
            with (
                httpx2.Client(
                    timeout=None, event_hooks={"request": [request_hook], "response": [response_hook]}
                ) as transport,
                TypeSafeClient(retry=RetryPolicy(max_retries=0), http_client=transport) as sdk,
            ):
                response = sdk.system_one(
                    state=state,
                    questions={name: Noul.model_validate(q) for name, q in questions.items()},
                    model=self.model,
                )
                raw = json.loads(response.raw_http_response.content)
                if set(raw["answers"]) != set(questions):
                    raise ValueError("Provider response has incomplete answer bindings")
                group.update(response=raw, seconds=time.monotonic() - started)
                with self.lock:
                    self.groups.append(group)
                    self.cached[key] = group
                    with (self.folder / "groups.jsonl").open("a") as handle:
                        handle.write(json.dumps(group, ensure_ascii=False) + "\n")
                return response_from_raw(raw)
        except BaseException:
            if not accounted:
                self.ledger.unknown(ticket, category="jev", case=self.cid, number=number)
            raise


async def trial(out, limit):
    from enginepy.workflows.document_analysis import evidence_pack as ep
    from enginepy.workflows.document_analysis.code_relations import CodeRelations
    from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps
    from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2

    from jev_navigator.directives.frontier import Policy
    from jev_navigator.environment import load_typesafe_environment
    from jev_navigator.judgments.judge import Judge

    os.environ["TYPESAFE_BASE_URL"] = "https://api.typesafe.ai"
    load_typesafe_environment()
    guard = load(out / "paid/guard-report.json")
    if guard["status"] != "complete":
        raise ValueError("Prepared guard has no complete result")
    ledger = SpendLedger(out / "paid/spend-ledger.jsonl")
    if ledger.halted:
        raise SpendStopError("Unreconciled usage stops all candidate calls")
    results = []
    path = out / "paid/results.json"
    if path.exists():
        results = load(path)
    done = {(r["dataset"], r["case"]) for r in results}
    original_search = ep.find_all_async
    scopes = {}
    total = 0
    for dataset in ("dev110", "hard27"):
        inputs = load(BASE / f"runs/pack-49b78955/pack-inputs-{dataset}.json")["cases"]
        for record in rows(out / "combined" / f"{dataset}-candidates.jsonl"):
            if not record["primary"]:
                continue
            cid = record["case"]
            if dataset == "hard27" and cid not in {
                *(f"P{i}" for i in range(1, 8)),
                *(f"U{i}" for i in range(1, 7)),
            }:
                raise ValueError("Held-out case in tuning inventory")
            total += 1
            if limit and total > limit:
                return results
            if (dataset, cid) in done:
                continue
            check_resources()
            row = inputs[cid]
            root = row["repository"]
            started = time.monotonic()
            if root not in scopes:
                withheld = frozenset(row["withheld"])
                relations = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
                scopes[root] = relations, ep.pack_index(relations, root)
            relations, index = scopes[root]
            request = ep.pack_request(relations, root, row["claim"], row.get("points"))
            folder = out / "paid/cases" / dataset / cid.replace(":", "_")
            client = PaidClient(ledger, cid, folder)
            if request is None:
                result = {
                    "dataset": dataset,
                    "case": cid,
                    "recipe": record["recipe"],
                    "failure": "no readable anchor",
                    "requests": 0,
                    "labels": record["labels"],
                }
            else:
                source = CandidateSource(tuple(record["units"]))

                async def configured(index, judge, targets, _source=source, **options):
                    options.update(
                        sources=(_source,), hops=(), policy=Policy("recipe-v2-combined", ranked=False)
                    )
                    return await original_search(index, judge, targets, **options)

                judge = Judge(client, scanner=None, max_calls=8, items_per_request=16)
                ep.find_all_async = configured
                try:
                    pack = await ep.build_pack(
                        relations,
                        request,
                        judge,
                        index,
                        ep.PackSettings(
                            question_profile=ROLES_V2, required_roles=LOCAL_ROLES, callee_round=False
                        ),
                    )
                finally:
                    ep.find_all_async = original_search
                save(folder / "pack.json", __import__("dataclasses").asdict(pack))
                packet = pack.packet()
                (folder / "consumer-packet.txt").write_text(packet.render())
                result = {
                    "dataset": dataset,
                    "case": cid,
                    "recipe": record["recipe"],
                    "failure": pack.failure,
                    "requests": len(client.groups),
                    "labels": record["labels"],
                    "judgment_seconds": time.monotonic() - started,
                    "http_sum_seconds": sum(g.get("seconds", 0) for g in client.groups),
                    "usd": sum(g["usd"] for g in client.groups),
                    "units_judged": sum(len(g["state"]["items"]) for g in client.groups),
                }
                try:
                    from paid_rooms import measure_case

                    result["measurement"] = measure_case(
                        record, pack, relations, request, client.groups, record["labels"]
                    )
                except ImportError:
                    result["rooms_pending"] = True
            results = [r for r in results if (r["dataset"], r["case"]) != (dataset, cid)]
            results.append(result)
            save(path, results)
            print(
                json.dumps(
                    {
                        "done": len(results),
                        "case": cid,
                        "recipe": record["recipe"],
                        "requests": result["requests"],
                        "failure": result["failure"],
                        "total_usd": str(ledger.spent),
                    }
                ),
                flush=True,
            )
            if ledger.halted:
                raise SpendStopError("Provider usage needs reconciliation before continuing")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(trial(args.out, args.limit))


if __name__ == "__main__":
    main()
