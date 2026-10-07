"""Durable reservations for the recipe trial's one-dollar cumulative cap."""

from __future__ import annotations

import json
import os
import threading
import uuid
from decimal import Decimal
from pathlib import Path

RATE = Decimal("0.000000042")
MAX_REQUEST_USD = Decimal(64000) * RATE


class SpendStopError(RuntimeError):
    """The next paid send cannot be admitted."""


class SpendLedger:
    def __init__(self, path: Path, cap: str = "1.00"):
        self.path, self.cap = path, Decimal(cap)
        self.lock = threading.RLock()
        self.spent = Decimal(0)
        self.pending = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["event"] == "reserve":
                self.pending[row["ticket"]] = Decimal(row["usd"])
            elif row["event"] == "release_unsent":
                self.pending.pop(row["ticket"])
            elif row["event"] == "usage":
                self.spent += Decimal(row["usd"])
                self.pending.pop(row["ticket"])
        self.halted = bool(self.pending)

    def append(self, row):
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(self, category, case, request_sha256, maximum=MAX_REQUEST_USD):
        with self.lock:
            if self.halted or self.spent + sum(self.pending.values()) + maximum > self.cap:
                self.append(
                    {
                        "event": "stop",
                        "category": category,
                        "case": case,
                        "spent_usd": str(self.spent),
                        "next_usd": str(maximum),
                    }
                )
                raise SpendStopError("Next reservation would exceed cap or usage needs reconciliation")
            ticket = uuid.uuid4().hex
            self.pending[ticket] = maximum
            self.append(
                {
                    "event": "reserve",
                    "ticket": ticket,
                    "category": category,
                    "case": case,
                    "request_sha256": request_sha256,
                    "usd": str(maximum),
                }
            )
            return ticket

    def settle(self, ticket, raw, **context):
        with self.lock:
            usage = raw["usage"]
            tokens = usage["input_tokens"]
            if type(tokens) is not int or tokens < 0:
                raise ValueError("Missing valid provider input usage")
            usd = Decimal(tokens) * RATE
            maximum = self.pending.pop(ticket)
            self.spent += usd
            self.append(
                {
                    "event": "usage",
                    "ticket": ticket,
                    "usd": str(usd),
                    "total_usd": str(self.spent),
                    "model": raw["model"],
                    "input_tokens": tokens,
                    "output_tokens": usage.get("output_tokens"),
                    **context,
                }
            )
            if usd > maximum or self.spent + sum(self.pending.values()) > self.cap:
                self.halted = True
                raise SpendStopError("Reported bill exceeded reservation or cumulative cap")
            return float(usd)

    def unknown(self, ticket, **context):
        with self.lock:
            self.halted = True
            self.append({"event": "usage_unknown", "ticket": ticket, **context})
