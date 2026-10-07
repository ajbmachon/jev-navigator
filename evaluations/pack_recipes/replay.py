"""Strict cached-request replay and frozen-group development allocation.

Run separately with JVN #148, Engine #1475 and the archived find-eval harness.
The reach run uses #149. Keeping these processes separate preserves both owners.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
ROLES = BASE / "runs/roles-compare-20261006"


def offline(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Recipes replay permits zero provider calls")


sys.addaudithook(offline)


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load(path):
    return json.loads(path.read_text())


def request_key(state, questions):
    return hashlib.sha256(
        json.dumps({"state": state, "questions": questions}, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def cached_groups(metadata):
    from find_eval.simulate import pair_probabilities

    questions = load(ROLES / "step-3/roles16-candidate.json")["request"]["questions"]
    answers, models = pair_probabilities(ROLES / "step-3/paid/roles16/answer-table.jsonl", questions)
    if models != {"jev-1.13.0"}:
        raise ValueError("Unexpected stored served model")
    receipts = load(ROLES / "efficiency/physical-receipts.json")["roles16"]
    costs = {row["members_key"]: row for row in receipts}
    groups = defaultdict(list)
    exact = {}
    for row in rows(ROLES / "step-3/roles16-requests.jsonl"):
        cost = costs[json.dumps(row["members"], sort_keys=True)]
        pair_ids = tuple(dict.fromkeys(member["pair_id"] for member in row["members"].values()))
        group = {
            "key": request_key(row["state"], row["questions"]),
            "pairs": pair_ids,
            "members": row["members"],
            "usd": cost["usd"],
            "ms": cost["ms"],
            "tokens": cost["tokens"],
            "order": (row["source_request_number"], row["split"]),
        }
        groups[row["case_id"]].append(group)
        exact[group["key"]] = group
    for values in groups.values():
        values.sort(key=lambda group: group["order"])
    return groups, exact, answers


class MissingRequestError(RuntimeError):
    pass


class ExactCacheClient:
    model = "jev-1.13.0"

    def __init__(self, exact, answers):
        self.exact, self.answers = exact, answers
        self.requests = []
        self.hits = []

    def ask(self, state, questions):
        from jev_navigator.judgments.answers import JevResponse, NoulAnswer

        key = request_key(state, questions)
        self.requests.append({"state": state, "questions": questions, "key": key})
        group = self.exact.get(key)
        if group is None:
            raise MissingRequestError("No stored answer for this exact ordered request of 16 companions")
        self.hits.append(group)
        return JevResponse(
            {
                question: NoulAnswer(self.answers[member["pair_id"]][member["role"]])
                for question, member in group["members"].items()
            },
            self.model,
            group["tokens"],
        )


@dataclass(frozen=True)
class CandidateSource:
    units: tuple[dict, ...]
    name: str = "recipe"
    label: str = "recipe candidates"

    def reach(self, _index, _seeds):
        from jev_navigator.index.units import RangeAnchor
        from jev_navigator.sources import Reach

        for unit in self.units:
            for start, end in unit["ranges"]:
                yield Reach(RangeAnchor(unit["path"], start, end), self.name, unit["id"], 0)


def frozen_development(record, metadata, groups, answers):
    """Keep complete historical companions in original search order, never splice their answers."""
    from find_eval.simulate import composed_case, proposal_reducer

    cid = record["case"]
    candidates = {unit["id"] for unit in record["units"][:128]}
    pairs = {key: pair for key, pair in metadata["pairs"].items() if pair["case_id"] == cid}
    admitted = []
    spend = 0
    for group in groups[cid]:
        if not any(
            metadata["units"][pairs[pair]["unit_key"]]["place"] in candidates for pair in group["pairs"]
        ):
            continue
        if spend + group["usd"] > 0.005:
            break
        admitted.append(group)
        spend += group["usd"]
    observed = {pair: answers[pair] for group in admitted for pair in group["pairs"]}
    result = composed_case(
        metadata["cases"][cid],
        pairs,
        metadata["units"],
        observed,
        proposal_reducer(ROLES / "step-1b/roles-PROPOSAL.md"),
        ("decide", "guard", "value", "effect"),
    )
    return {
        "path": "frozen original groups development allocation",
        "requests": len(admitted),
        "reported_usd": spend,
        "reported_http_seconds": sum(g["ms"] for g in admitted) / 1000,
        "observed_pairs": len(observed),
        "labels": result["labels"],
        "windows": result["windows"],
        "confirmed": [
            metadata["units"][pairs[pair]["unit_key"]]
            for pair, probabilities in observed.items()
            if max(probabilities[role] for role in ("decide", "guard", "value", "effect", "satisfied")) >= 0.8
        ],
    }


async def native_replay(records, exact, answers, inputs, out):
    from enginepy.workflows.document_analysis import evidence_pack as ep
    from enginepy.workflows.document_analysis.code_relations import CodeRelations
    from enginepy.workflows.document_analysis.import_neighbors import repository_import_maps
    from find_eval.simulate import consumer_windows
    from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2

    from jev_navigator.judgments.judge import Judge

    scopes = {}
    results = []
    original_search = ep.find_all_async
    for record in records:
        from resources import check_resources

        check_resources()
        cid = record["case"]
        row = inputs[cid]
        root = row["repository"]
        started = time.monotonic()
        if root not in scopes:
            withheld = frozenset(row["withheld"])
            relations = CodeRelations(root, repository_import_maps(root, withheld), withheld=withheld)
            scopes[root] = relations, ep.pack_index(relations, root)
        relations, index = scopes[root]
        request = ep.pack_request(relations, root, row["claim"], row.get("points"))
        if request is None:
            result = {
                "case": cid,
                "recipe": record["recipe"],
                "primary": record["primary"],
                "failure": "no readable anchor",
                "consumer_windows": [],
                "requests": 0,
                "exact_hits": 0,
                "local_seconds": time.monotonic() - started,
                "provider_calls": 0,
                "spent_usd": 0,
            }
            results.append(result)
            continue
        source = CandidateSource(tuple(record["units"]))

        async def configured(index, judge, targets, _source=source, **options):
            options["sources"] = (_source,)
            options["hops"] = ()
            return await original_search(index, judge, targets, **options)

        client = ExactCacheClient(exact, answers)
        judge = Judge(client, scanner=None, max_calls=8, items_per_request=16)
        ep.find_all_async = configured
        try:
            pack = await ep.build_pack(
                relations,
                request,
                judge,
                index,
                ep.PackSettings(question_profile=ROLES_V2, required_roles=LOCAL_ROLES, callee_round=False),
            )
        finally:
            ep.find_all_async = original_search
        consumer = pack.packet()
        folder = out / "native" / record["recipe"] / cid.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "consumer-packet.txt").write_text(consumer.render())
        with gzip.open(folder / "requests.jsonl.gz", "wt") as handle:
            for group in client.requests:
                handle.write(json.dumps(group, ensure_ascii=False) + "\n")
        windows = consumer_windows(pack, consumer)
        result = {
            "case": cid,
            "recipe": record["recipe"],
            "primary": record["primary"],
            "failure": pack.failure,
            "consumer_windows": windows,
            "consumer_chars": len(consumer.render()),
            "consumer_sha256": hashlib.sha256(consumer.render().encode()).hexdigest(),
            "requests": len(client.requests),
            "exact_hits": len(client.hits),
            "reported_usd": sum(group["usd"] for group in client.hits),
            "local_seconds": time.monotonic() - started,
            "provider_calls": 0,
            "spent_usd": 0,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "case": cid,
                    "recipe": record["recipe"],
                    "hits": len(client.hits),
                    "requests": len(client.requests),
                    "seconds": round(result["local_seconds"], 2),
                }
            ),
            flush=True,
        )
        (out / "native-replay.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    metadata = load(ROLES / "step-3/metadata.json")
    groups, exact, answers = cached_groups(metadata)
    records = list(rows(args.out / "dev110-candidates.jsonl"))[: args.limit]
    development = []
    strict = []
    for record in records:
        result = frozen_development(record, metadata, groups, answers)
        development.append(
            {"case": record["case"], "recipe": record["recipe"], "primary": record["primary"], **result}
        )
        count = 0
        hits = 0
        with gzip.open(record["requests_path"], "rt") as handle:
            for row in handle:
                group = json.loads(row)
                count += 1
                hits += request_key(group["state"], group["questions"]) in exact
        strict.append(
            {
                "case": record["case"],
                "recipe": record["recipe"],
                "requests": count,
                "exact_hits": hits,
            }
        )
    (args.out / "frozen-development.json").write_text(json.dumps(development, indent=2) + "\n")
    (args.out / "strict-request-coverage.json").write_text(json.dumps(strict, indent=2) + "\n")
    if args.native:
        inputs = load(BASE / "runs/pack-49b78955/pack-inputs-dev110.json")["cases"]
        asyncio.run(native_replay(records, exact, answers, inputs, args.out))


if __name__ == "__main__":
    main()
