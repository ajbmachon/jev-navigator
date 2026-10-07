"""The experiment's dollar cap survives concurrent sends and interruption."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "recipe_paid_budget", Path(__file__).parents[1] / "evaluations/pack_recipes/paid_budget.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reservations_admit_only_affordable_concurrent_requests(tmp_path):
    ledger = MODULE.SpendLedger(tmp_path / "usage.jsonl", cap="0.005376")

    def reserve(number):
        try:
            return ledger.reserve("jev", f"finding-{number}", str(number))
        except MODULE.SpendStopError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = [ticket for ticket in pool.map(reserve, range(8)) if ticket]
    assert len(admitted) == 2
    assert sum(ledger.pending.values()) == Decimal("0.005376")
    for ticket in admitted:
        ledger.settle(
            ticket,
            {"usage": {"input_tokens": 10000, "output_tokens": 0}, "model": "jev-1.13.0"},
            category="jev",
        )
    resumed = MODULE.SpendLedger(ledger.path, cap="0.005376")
    assert resumed.spent == Decimal("0.000840000")
    assert not resumed.pending
    resumed.reserve("guard", "shape", "new")


def test_unreported_attempt_blocks_restart_instead_of_disappearing(tmp_path):
    ledger = MODULE.SpendLedger(tmp_path / "usage.jsonl")
    ticket = ledger.reserve("guard", "shape", "hash")
    ledger.unknown(ticket, category="guard")
    resumed = MODULE.SpendLedger(ledger.path)
    assert resumed.pending[ticket] == Decimal("0.002688000")
    with pytest.raises(MODULE.SpendStopError):
        resumed.reserve("jev", "finding", "hash2")
    assert resumed.spent == 0


def test_reported_usage_cannot_silently_exceed_reserved_projection(tmp_path):
    ledger = MODULE.SpendLedger(tmp_path / "usage.jsonl")
    ticket = ledger.reserve("jev", "finding", "hash")
    with pytest.raises(MODULE.SpendStopError):
        ledger.settle(ticket, {"usage": {"input_tokens": 64001}, "model": "jev-1.13.0"})
    assert ledger.halted
    with pytest.raises(MODULE.SpendStopError):
        ledger.reserve("jev", "second", "hash2")


def test_unparseable_reported_usage_keeps_reservation_and_stops_further_sends(tmp_path):
    ledger = MODULE.SpendLedger(tmp_path / "usage.jsonl")
    ticket = ledger.reserve("jev", "finding", "hash")
    with pytest.raises(ValueError):
        ledger.settle(ticket, {"usage": {"input_tokens": None}, "model": "jev-1.13.0"})
    with pytest.raises(MODULE.SpendStopError):
        ledger.reserve("jev", "next", "hash2")
    assert ticket in ledger.pending
