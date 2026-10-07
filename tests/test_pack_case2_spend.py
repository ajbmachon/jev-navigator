"""The paid recipe must retain reservations across a process restart."""

import importlib.util
from pathlib import Path

import pytest


def ledger(path):
    spec = importlib.util.spec_from_file_location(
        "spend", Path(__file__).parents[1] / "examples/pack_case2/spend.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpendLedger(path)


def test_pending_call_survives_restart_and_cannot_be_paid_twice(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger(path).reserve("finding-1", "planner", "0.70")
    resumed = ledger(path)
    with pytest.raises(RuntimeError, match="Cap stop"):
        resumed.reserve("finding-2", "planner", "0.31")
    with pytest.raises(RuntimeError, match="already recorded"):
        resumed.reserve("finding-1", "planner", "0.01")
    resumed.settle("finding-1", "0.20", input_tokens=100)
    ledger(path).reserve("guard", "meta", "0.80")
    assert ledger(path).balance() == 0


def test_reported_overrun_is_retained_and_blocks_another_dispatch(tmp_path):
    path = tmp_path / "ledger.jsonl"
    trial = ledger(path)
    trial.reserve("finding-1", "planner", "0.90")
    with pytest.raises(RuntimeError, match="exceeded cap"):
        trial.settle("finding-1", "1.01")
    with pytest.raises(RuntimeError, match="Cap stop"):
        ledger(path).reserve("finding-2", "planner", "0.001")
