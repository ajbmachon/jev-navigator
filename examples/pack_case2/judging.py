"""Prepare real 16-unit request shapes and replay only unchanged stored groups.

Run with the JVN #148 profile owner on PYTHONPATH, followed by the frozen harness
and Engine paths. No provider client is constructed. This stage consumes Case 2
candidate receipts produced on #149; it never edits either prerequisite branch.
"""

from __future__ import annotations

import base64
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from find_eval.simulate import composed_case, pair_probabilities, proposal_reducer
from jev_navigator.judgments.profiles import ROLES_V2
from offline_mask import OfflineMasker
from request_batches import judge_in_order

from jev_navigator.index.units import Item
from jev_navigator.judgments.answers import JevResponse, NoulAnswer
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import content_hash

BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
STORE = BASE / "runs/roles-compare-20261006"
MASKER = OfflineMasker()


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("No paid calls are permitted")


sys.addaudithook(deny_network)


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def exact_key(state, questions):
    # Keys preserve ordered items and full questions. The complete request body is
    # also retained; comparing this hash does not relax batch-company eligibility.
    return content_hash({"state": state, "questions": questions})


class RequestRecorder:
    model = "jev-latest"

    def __init__(self, destination, exact):
        self.destination = destination
        self.exact = exact
        self.requests = []

    def ask(self, state, questions):
        key = exact_key(state, questions)
        body = {"model": self.model, "state": state, "questions": questions}
        size = len(json.dumps(body, ensure_ascii=False).encode())
        row = {
            "hash": key,
            "items": len(state["items"]),
            "bytes": size,
            "exact_stored_request": key in self.exact,
        }
        self.requests.append(row)
        self.destination.write(json.dumps(body, ensure_ascii=False) + "\n")
        # Shape rehearsal only. These values never enter delivery or ranking.
        return JevResponse({name: NoulAnswer(0.5) for name in questions}, self.model)


def prepare_requests(prefix, arm, context, exact):
    def entries():
        for record in rows(prefix / f"{arm}-candidates.jsonl"):
            for item in record["items"]:
                yield (
                    {"file": item["file"], "code": item["code"]},
                    Item(item["id"], item["file"], tuple(map(tuple, item["ranges"]))),
                )

    with gzip.open(prefix / f"{arm}-requests.jsonl.gz", "wt") as saved:
        recorder = RequestRecorder(saved, exact)
        judge = Judge(recorder, items_per_request=16, scanner=None, masker=MASKER)
        refusals = []
        # The request is the masking and companion boundary, as in real search.
        # Do not form one repository-wide call then mask every body with secrets
        # collected from thousands of unrelated future request companions.
        for _ in judge_in_order(
            judge, ROLES_V2.questions("p0"), entries(), {"targets": {"p0": context["finding"]}}, refusals
        ):
            pass
    return {
        "requests": len(recorder.requests),
        "request_sizes": [r["items"] for r in recorder.requests],
        "request_body_bytes": sum(r["bytes"] for r in recorder.requests),
        "exact_queue_requests": sum(r["exact_stored_request"] for r in recorder.requests),
        "refusals": [str(r) for r in refusals],
    }


def main():
    out = Path(sys.argv[1]).expanduser()
    summary = load(out / "replay-summary.json")
    metadata = load(STORE / "step-3/metadata.json")
    templates = load(STORE / "step-3/roles16-candidate.json")["request"]["questions"]
    observations, models = pair_probabilities(STORE / "step-3/paid/roles16/answer-table.jsonl", templates)
    assert models == {"jev-1.13.0"}
    requests = defaultdict(list)
    exact = set()
    for request in rows(STORE / "step-3/roles16-requests.jsonl"):
        requests[request["case_id"]].append(request)
        exact.add(exact_key(request["state"], request["questions"]))
    for case_requests in requests.values():
        case_requests.sort(key=lambda r: (r["source_request_number"], r["split"]))
    # Independent actual sent-body receipts, indexed by exact full request context.
    paid = {}
    for record in rows(STORE / "step-3/paid/roles16/journal.jsonl"):
        if record["kind"] == "http_attempt" and record.get("status") == 200:
            body = json.loads(base64.b64decode(record["sent_body_base64"]))
            response = json.loads(base64.b64decode(record["body_base64"]))
            paid[exact_key(body["state"], body["questions"])] = {
                "usd": response["usage"]["input_tokens"] * 0.042 / 1e6,
                "http_ms": record["duration_ms"],
            }
    reducer = proposal_reducer(STORE / "step-1b/roles-PROPOSAL.md")
    required = load(STORE / "step-2/selection-policy.json")["required_roles"]
    results = []
    for case in summary["cases"]:
        cid, arm = case["case"], case["arm"]
        prefix = out / "cases" / cid.replace(":", "_")
        context = load(prefix / "planner-context.json")
        physical = prepare_requests(prefix, arm, context, exact)
        physical["unjudged_queue_items"] = sum(
            r for r in physical["request_sizes"]
        )  # No queue answer is consumed unless exact full context is available.
        if physical["exact_queue_requests"]:
            raise RuntimeError("Exact queue answers found: implement their native consumer before reporting")
        entries = {
            content_hash({"file": item["file"], "code": item["code"]})
            for candidate in rows(prefix / f"{arm}-candidates.jsonl")
            for item in candidate["items"]
        }
        pairs = {}
        usd = http_ms = retained_requests = 0
        for request in requests.get(cid, ()):
            # This separate development arm preserves every original companion,
            # question and search order. It is not the new plan's queue delivery.
            if not all(content_hash(item) in entries for item in request["state"]["items"]):
                continue
            members = {member["pair_id"] for member in request["members"].values()}
            pairs.update({pair: metadata["pairs"][pair] for pair in members})
            receipt = paid[exact_key(request["state"], request["questions"])]
            usd += receipt["usd"]
            http_ms += receipt["http_ms"]
            retained_requests += 1
        if cid in metadata["cases"]:
            composed = composed_case(
                metadata["cases"][cid],
                pairs,
                metadata["units"],
                {pair: observations[pair] for pair in pairs},
                reducer,
                required,
            )
            write(prefix / f"{arm}-historical-groups-delivery.json", composed)
            delivered = sum(label["delivered"] for label in composed["labels"])
        else:
            delivered = None
        record = {
            "case": cid,
            "dataset": case["dataset"],
            "arm": arm,
            "physical": physical,
            "historical_groups_development": {
                "retained_requests": retained_requests,
                "exactly_judged_pairs": len(pairs),
                "delivered": delivered,
                "reported_historical_usd": usd,
                "historical_http_seconds": http_ms / 1000,
            },
            "provider_calls": 0,
            "usd": 0,
        }
        results.append(record)
        write(out / "judging-checkpoint.json", results)
        print(
            cid, arm, "requests", physical["requests"], "historical retained", retained_requests, flush=True
        )
    write(
        out / "judging-summary.json",
        {"cases": results, "provider_calls": 0, "usd": 0, "profile_owner": "JVN #148 at 677ef3e1"},
    )


if __name__ == "__main__":
    main()
