"""Run the existing Meta guard with the trial ledger around every physical send."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

from paid_budget import SpendLedger
from resources import check_resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from system_one_meta_builder import cli

    from jev_navigator.environment import load_typesafe_environment

    # Development trial uses the vendor route, not the retired local gateway.
    os.environ["TYPESAFE_BASE_URL"] = "https://api.typesafe.ai"
    load_typesafe_environment()

    out = args.out
    ledger = SpendLedger(out / "paid/spend-ledger.jsonl")
    original_call = cli.call_once

    def reserved_call(submitted, questions, *, record_event):
        check_resources()
        key = hashlib.sha256(json.dumps(submitted, sort_keys=True).encode()).hexdigest()
        ticket = ledger.reserve("guard", "v2-engine-role-group", key)
        accounted = False

        def event(value):
            nonlocal accounted
            record_event(value)
            if value["event"] == "http_response":
                import base64

                raw = json.loads(base64.b64decode(value["body_base64"]))
                if "usage" in raw:
                    accounted = True
                    ledger.settle(ticket, raw, category="guard", case="v2-engine-role-group")

        try:
            return original_call(submitted, questions, record_event=event)
        except BaseException:
            if not accounted:
                ledger.unknown(ticket, category="guard")
            raise

    cli.call_once = reserved_call
    receipts = out / "paid/guard-receipts.jsonl"
    with (out / "paid/guard-report.json").open("w") as handle, contextlib.redirect_stdout(handle):
        code = cli.main(
            ["guard", str(out / "paid-candidate.json"), "--output", str(receipts), "--reuse", str(receipts)]
        )
    print(json.dumps({"guard_exit": code, "spent_usd": str(ledger.spent), "pending": len(ledger.pending)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
