"""Persistent request reservations and provider usage for the authorized Case 3 evaluation."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo


class SpendStopError(RuntimeError):
    """No further paid request is admissible until the recorded spend is reconciled."""


class SpendLedger:
    def __init__(self, path: Path, cap: str = "1.25") -> None:
        self.path = path
        self.cap = Decimal(cap)
        self.lock = threading.RLock()
        self.spent = Decimal(0)
        self.reserved: dict[str, Decimal] = {}
        self.halted = False
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                if row["event"] == "usage":
                    self.spent += Decimal(row["usd"])
                if row["event"] == "trial_reserved":
                    self.reserved[row["ticket"]] = Decimal(row["reserve_usd"])
                if row["event"] == "usage" and row.get("ticket"):
                    self.reserved.pop(row["ticket"], None)
                if row["event"] == "usage_unknown":
                    self.halted = True
        if self.reserved or self.halted or self.spent >= self.cap:
            raise SpendStopError("Ledger needs reconciliation before another paid call")

    def append(self, row: dict) -> None:
        with self.lock, self.path.open("a") as output:
            output.write(
                json.dumps({"at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(), **row}) + "\n"
            )
            output.flush()

    def reserve(self, category: str, case: str, maximum: Decimal) -> str:
        with self.lock:
            if self.halted or self.spent + sum(self.reserved.values()) + maximum > self.cap:
                self.halted = True
                self.append(
                    {
                        "event": "cap_stop",
                        "category": category,
                        "case": case,
                        "next_reservation_usd": str(maximum),
                        "spent_usd": str(self.spent),
                    }
                )
                raise SpendStopError("Next possible bill would cross the cumulative cap")
            ticket = uuid.uuid4().hex
            self.reserved[ticket] = maximum
            self.append(
                {
                    "event": "trial_reserved",
                    "ticket": ticket,
                    "category": category,
                    "case": case,
                    "reserve_usd": str(maximum),
                }
            )
            return ticket

    def settle(self, ticket: str, *, usd: Decimal, **usage) -> None:
        with self.lock:
            maximum = self.reserved.pop(ticket)
            self.spent += usd
            self.append(
                {"event": "usage", "ticket": ticket, "usd": str(usd), "total_usd": str(self.spent), **usage}
            )
            if usd > maximum or self.spent + sum(self.reserved.values()) > self.cap:
                self.halted = True
                raise SpendStopError("Reported usage exceeds its reservation or the cumulative cap")

    def unknown(self, ticket: str, **context) -> None:
        with self.lock:
            self.halted = True
            self.append(
                {
                    "event": "usage_unknown",
                    "ticket": ticket,
                    "reservation_retained_usd": str(self.reserved[ticket]),
                    **context,
                }
            )

    def case_usage(self, case: str) -> list[dict]:
        with self.lock, self.path.open() as source:
            return [
                row
                for line in source
                if (row := json.loads(line)).get("case") == case and row["event"] == "usage"
            ]
