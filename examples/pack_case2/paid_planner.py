"""Authorized 123-call measurement. No Jev candidate transport is constructed."""

import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import msgspec
from planner import MODEL, PlannerContract, PlannerInput
from receipts import write_json
from spend import SpendLedger
from trial_resources import resources

from jev_navigator.search_plan import Approach, SearchPlan

ENDPOINT = "https://router.eu.requesty.ai/v1"
MAX_OUTPUT = 3000


def validated_approaches(text, context):
    """Reject malformed approaches independently. Never rewrite their arguments."""
    try:
        return PlannerContract().parse(text, context), []
    except ValueError as error:
        document = json.loads(text)
        if set(document) != {"approaches"} or not isinstance(document["approaches"], list):
            raise error
        if len(document["approaches"]) > 10:
            raise error
        accepted, rejected, ranks = [], [], set()
        for position, proposal in enumerate(document["approaches"]):
            try:
                approach = msgspec.convert(proposal, type=Approach)
                if approach.rank in ranks:
                    raise ValueError("duplicate rank")
                ranks.add(approach.rank)
                accepted.append(approach)
            except (ValueError, TypeError) as failure:
                rejected.append({"position": position, "proposal": proposal, "error": str(failure)})
        return SearchPlan(tuple(accepted)), rejected


def provider_cost(usage, model):
    # Requesty reports total_cost; preserve a separate catalog calculation for audit.
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    calculated = (
        Decimal(usage["prompt_tokens"] - cached) * Decimal(str(model["input_price"]))
        + Decimal(cached) * Decimal(str(model["cached_price"]))
        + Decimal(usage["completion_tokens"]) * Decimal(str(model["output_price"]))
    )
    reported = usage.get("total_cost", usage.get("cost"))
    return Decimal(str(reported)) if reported is not None else calculated, calculated


def main():
    source, out = (Path(arg).expanduser() for arg in sys.argv[1:3])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 123
    out.mkdir(parents=True, exist_ok=True)
    ledger = SpendLedger(out / "spend-ledger.jsonl")
    headers = {
        "Authorization": "Bearer " + os.environ["REQUESTY_API_KEY"],
        "Content-Type": "application/json",
    }
    with urllib.request.urlopen(urllib.request.Request(ENDPOINT + "/models", headers=headers)) as response:
        catalog = json.load(response)
    model = next(row for row in catalog["data"] if row["id"] == MODEL)
    write_json(out / "provider-model.json", model)
    calls = json.loads((source / "trial-manifest.json").read_text())["calls"][:limit]

    def run(call):
        resources()
        cid = call["case"]
        folder = out / "cases" / cid.replace(":", "_")
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "planner-receipt.json").exists():
            return
        prompt_path = Path(call["prompt"])
        prompt = prompt_path.read_text()
        assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == call["prompt_sha256"]
        context = PlannerInput(**json.loads(prompt_path.with_name("planner-context.json").read_text()))
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_OUTPUT,
            "response_format": {"type": "json_object"},
        }
        # UTF-8 bytes conservatively reserve input, independent of the proxy tokenizer.
        reserve = Decimal(len(prompt.encode()) + 1024) * Decimal(str(model["input_price"]))
        reserve += Decimal(MAX_OUTPUT) * Decimal(str(model["output_price"]))
        ledger.reserve(cid, "planner", str(reserve))
        wire = json.dumps(body, ensure_ascii=False).encode()
        (folder / "planner-request.json").write_bytes(wire)
        start = time.perf_counter()
        request = urllib.request.Request(ENDPOINT + "/chat/completions", data=wire, headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
        (folder / "planner-response.json").write_bytes(raw)
        seconds = time.perf_counter() - start
        reply = json.loads(raw)
        usage = reply["usage"]
        usd, calculated = provider_cost(usage, model)
        ledger.settle(cid, str(usd), usage=usage, catalog_usd=str(calculated), model=reply.get("model"))
        receipt = {
            **call,
            "usage": usage,
            "usd": str(usd),
            "catalog_usd": str(calculated),
            "seconds": seconds,
            "served_model": reply.get("model"),
            "finish_reason": reply["choices"][0].get("finish_reason"),
        }
        try:
            plan, rejected = validated_approaches(reply["choices"][0]["message"]["content"], context)
            write_json(folder / "plan.json", plan)
            receipt["approaches"] = len(plan.approaches)
            receipt["rejected_approaches"] = rejected
        except (ValueError, TypeError, RuntimeError) as error:
            receipt["parse_error"] = str(error)
        write_json(folder / "planner-receipt.json", receipt)
        print(
            json.dumps(
                {
                    "case": cid,
                    "usd": str(usd),
                    "seconds": round(seconds, 2),
                    "approaches": receipt.get("approaches"),
                    "error": receipt.get("parse_error"),
                    "balance": str(ledger.balance()),
                }
            ),
            flush=True,
        )

    # The first receipt checks actual usage and schema before concurrent scaling.
    pending = [
        call
        for call in calls
        if not (out / "cases" / call["case"].replace(":", "_") / "planner-receipt.json").exists()
    ]
    if pending:
        run(pending[0])
    with ThreadPoolExecutor(max_workers=3) as pool:
        for _result in pool.map(run, pending[1:]):
            pass


if __name__ == "__main__":
    main()
