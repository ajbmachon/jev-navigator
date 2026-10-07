"""Durable trial reservations. An interrupted or unpriced call stays reserved."""

import json
import os
from decimal import Decimal
from pathlib import Path
from threading import Lock


class SpendLedger:
    def __init__(self, path: Path, cap: str = "1.00"):
        self.path = path
        self.cap = Decimal(cap)
        self.lock = Lock()
        self.events = [json.loads(line) for line in path.open()] if path.exists() else []

    def _append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)

    def balance(self):
        latest = {event["id"]: event for event in self.events}
        return self.cap - sum((Decimal(e["usd"]) for e in latest.values()), Decimal(0))

    def reserve(self, id: str, category: str, usd: str):
        with self.lock:
            if any(event["id"] == id for event in self.events):
                raise RuntimeError(f"Attempt already recorded: {id}; reconcile its receipt before resuming")
            amount = Decimal(usd)
            if amount < 0 or amount > self.balance():
                raise RuntimeError(f"Cap stop: requested ${amount}, available ${self.balance()}")
            self._append({"id": id, "category": category, "status": "reserved", "usd": str(amount)})

    def settle(self, id: str, usd: str, **usage):
        with self.lock:
            prior = next(event for event in reversed(self.events) if event["id"] == id)
            if prior["status"] != "reserved":
                raise RuntimeError(f"Attempt already settled: {id}")
            amount = Decimal(usd)
            if amount < 0:
                raise ValueError("Negative spend")
            self._append({**prior, "status": "settled", "usd": str(amount), **usage})
            if self.balance() < 0:
                raise RuntimeError(f"Reported usage exceeded cap by ${-self.balance()}; stop all dispatch")
