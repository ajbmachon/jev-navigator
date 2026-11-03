"""Retained reach transport failures and conservative retry accounting."""

import hashlib
import time
from decimal import Decimal

from paid_client import JevSession
from paid_support import load, write


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RetainedJevSession(JevSession):
    """Keep a terminal receipt when the reused transport loses its response."""

    def ask(self, state, questions):
        try:
            return super().ask(state, questions)
        except Exception as error:
            current = getattr(self.active, "current", None)
            if current is not None:
                attempt_id, path, started, wire = current
                if not (path / "receipt.json").exists():
                    receipt = {
                        "id": attempt_id,
                        "category": self.category,
                        "status": None,
                        "seconds": time.perf_counter() - started,
                        "usage": {},
                        "model": None,
                        "attempted_model": self.model,
                        "request_sha256": hashlib.sha256(wire).hexdigest(),
                        "failure": f"{type(error).__name__}: {error}",
                        "unpriced_reserved": True,
                    }
                    write(path / "receipt.json", receipt)
                    self.receipts.append(receipt)
            raise


def restore_unreceipted_failure(base, observation, ledger):
    """Join the earlier reset's retained wire and reserved identity without inventing a reply."""
    if not observation["failure"].startswith("TypeSafeAPIConnectionError:"):
        return
    entries = ledger.latest()
    for wire in (base / "http").glob("reach_*/request.json"):
        receipt_path = wire.with_name("receipt.json")
        if receipt_path.exists():
            continue
        attempt_id = wire.parent.name.replace("reach_", "reach:", 1)
        assert entries[attempt_id]["status"] == "reserved"
        assert not wire.with_name("response.json").exists()
        write(
            receipt_path,
            {
                "id": attempt_id,
                "category": "reach",
                "status": None,
                "seconds": observation["seconds"],
                "timing_source": "failed_group_elapsed",
                "usage": {},
                "model": None,
                "attempted_model": "jev-latest",
                "request_sha256": sha256(wire),
                "failure": observation["failure"],
                "unpriced_reserved": True,
                "recovered_from_retained_wire_and_group_failure": True,
            },
        )


def retained_retryable_failure(base, observation):
    paths = list((base / "http").glob("reach_*/receipt.json"))
    if not paths:
        return None
    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    receipt = load(path)
    if observation["failure"].startswith("TypeSafeAPIConnectionError:"):
        if receipt["status"] is None and receipt.get("failure") == observation["failure"]:
            return receipt
        return None
    if not observation["failure"].startswith("TypeSafeInternalServerError:"):
        return None
    response = load(path.with_name("response.json"))
    if receipt["status"] != 529 or response.get("detail", {}).get("error_type") != "system_overloaded":
        return None
    return receipt


def group_invoice(base, ledger):
    receipts = [load(p) for p in (base / "http").glob("reach_*/receipt.json")]
    entries = ledger.latest()
    known = sum((Decimal(r["usd"]) for r in receipts if "usd" in r), Decimal(0))
    pending = sum((Decimal(entries[r["id"]]["usd"]) for r in receipts if "usd" not in r), Decimal(0))
    return known, pending
