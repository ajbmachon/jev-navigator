"""Consume planned candidates through the pinned Engine packet owner at zero cost.

Run on JVN #148 and Engine #1475 paths, exactly like the anchored baseline. The
candidate adapter is evaluation configuration; no Engine source is edited.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import evidence_pack_mode
from enginepy.workflows.document_analysis import evidence_pack as owner
from find_eval.simulate import consumer_windows
from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2
from offline_mask import OfflineMasker

from jev_navigator.index.units import LineAnchor
from jev_navigator.judgments.answers import response_from_raw
from jev_navigator.judgments.client import MissingAnswerError
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import content_hash
from jev_navigator.sources import Reach

BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
STORE = BASE / "runs/roles-compare-20261006"
MASKER = OfflineMasker()


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Native replay permits zero paid calls")


sys.addaudithook(deny_network)


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class StoredPlanSource:
    name = "plan"
    label = "planned candidates"

    def __init__(self, path):
        self.path = path

    def reach(self, index, seeds):
        for candidate in rows(self.path):
            unit = candidate["unit"]
            rank = min(candidate["approach_ranks"])
            yield Reach(LineAnchor(unit["path"], unit["ranges"][0][0]), f"plan:{rank}", unit["id"], 1)


class ExactClient:
    model = "jev-latest"

    def __init__(self, cache):
        self.cache = cache
        self.requests = []
        self.exact_hits = 0

    def ask(self, state, questions):
        request = {"state": state, "questions": questions}
        self.requests.append(request)
        response = self.cache.get(content_hash(request))
        if response is None:
            raise MissingAnswerError(
                "No stored answer with identical state, questions and ordered companions"
            )
        self.exact_hits += 1
        return response_from_raw(response)


async def main():
    out = Path(sys.argv[1]).expanduser()
    summary = load(out / "replay-summary.json")
    cache = {}
    for record in rows(STORE / "step-3/paid/roles16/journal.jsonl"):
        if record["kind"] == "http_attempt" and record.get("status") == 200:
            request = json.loads(base64.b64decode(record["sent_body_base64"]))
            cache[content_hash({"state": request["state"], "questions": request["questions"]})] = json.loads(
                base64.b64decode(record["body_base64"])
            )
    inputs = {}
    for path in (BASE / "runs/real-pack-e733ea0c-jvn0a73a59c-value-full/inputs").glob("dev110-[01].json"):
        inputs.update(load(path)["cases"])
    scope = evidence_pack_mode._scope(next(iter(inputs.values())))
    original_sources = owner._FIRST_ROUND_SOURCES
    checkpoint = out / "native-checkpoint.json"
    results = load(checkpoint) if checkpoint.exists() else []
    completed = {(row["case"], row["arm"]) for row in results}
    try:
        for case in summary["cases"]:
            if case["dataset"] != "dev110":
                continue
            cid, arm = case["case"], case["arm"]
            if (cid, arm) in completed:
                continue
            prefix = out / "cases" / cid.replace(":", "_")
            row = inputs[cid]
            relations, index = scope
            request = owner.pack_request(relations, row["repository"], row["claim"], row.get("points"))
            if request is None:
                result = {
                    "case": cid,
                    "arm": arm,
                    "no_anchor": True,
                    "delivered": 0,
                    "labels": [{**label, "delivered": False} for label in case["labels"]],
                    "seconds": 0,
                    "exact_hits": 0,
                    "requests": 0,
                }
            else:
                owner._FIRST_ROUND_SOURCES = (StoredPlanSource(prefix / f"{arm}-candidates.jsonl"),)
                client = ExactClient(cache)
                judge = Judge(client, scanner=None, max_calls=48, items_per_request=16, masker=MASKER)
                started = time.monotonic()
                pack = await owner.build_pack(
                    relations,
                    request,
                    judge,
                    index,
                    owner.PackSettings(question_profile=ROLES_V2, required_roles=LOCAL_ROLES),
                )
                packet = pack.packet()
                windows = consumer_windows(pack, packet)
                labels = [
                    {
                        **label,
                        "delivered": any(
                            window["file"] == label["file"]
                            and window["first_line"] <= label["first_line"]
                            and window["last_line"] >= label["last_line"]
                            for window in windows
                        ),
                    }
                    for label in case["labels"]
                ]
                rendered = packet.render()
                (prefix / f"{arm}-native-packet.txt").write_text(rendered + "\n")
                result = {
                    "case": cid,
                    "arm": arm,
                    "labels": labels,
                    "delivered": sum(label["delivered"] for label in labels),
                    "consumer_windows": windows,
                    "packet_chars": len(rendered),
                    "packet_sha256": content_hash(rendered),
                    "facts": list(pack.facts),
                    "seconds": time.monotonic() - started,
                    "exact_hits": client.exact_hits,
                    "requests": len(client.requests),
                    "no_anchor": False,
                    "stop": list(pack.cost.stopped_by),
                }
                write(prefix / f"{arm}-native-requests.json", client.requests)
            result.update(provider_calls=0, usd=0)
            write(prefix / f"{arm}-native-result.json", result)
            results.append(result)
            write(out / "native-checkpoint.json", results)
            print(
                cid, arm, "native delivered", result["delivered"], "exact", result["exact_hits"], flush=True
            )
    finally:
        owner._FIRST_ROUND_SOURCES = original_sources
    write(
        out / "native-summary.json",
        {"cases": results, "provider_calls": 0, "usd": 0, "engine": "91f57ab4", "jvn_profile": "677ef3e1"},
    )


if __name__ == "__main__":
    asyncio.run(main())
