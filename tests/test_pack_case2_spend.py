"""The paid recipe must retain reservations across a process restart."""

import importlib.util
import json
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


def test_one_invalid_approach_keeps_valid_neighbors_and_the_original_failure(monkeypatch):
    directory = Path(__file__).parents[1] / "examples/pack_case2"
    monkeypatch.syspath_prepend(str(directory))
    import paid_planner
    from planner import PlannerInput

    invalid = {
        "rank": 2,
        "call": {"operation": "file_units", "path": "invented.py"},
        "provenance": [],
        "source": "unsupported field",
    }
    reply = json.dumps(
        {
            "approaches": [
                {"rank": 1, "call": {"operation": "file_units", "path": "real.py"}, "provenance": []},
                invalid,
            ]
        }
    )
    plan, rejected = paid_planner.validated_approaches(reply, PlannerInput("", (), "", (), {}))
    assert [approach.call.path for approach in plan.approaches] == ["real.py"]
    assert rejected[0]["proposal"] == invalid
    assert "unknown field" in rejected[0]["error"]


def test_binding_report_distinguishes_real_empty_search_from_invented_path(monkeypatch, tmp_path):
    from shop_search import shop_index

    from jev_navigator.search_plan import Approach, FileUnits, FindText, SearchPlan, execute_plan

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/pack_case2"))
    from paid_execute import binding_status

    index = shop_index(tmp_path, {"real.py": "def alpha():\n    return 7\n"})
    result = execute_plan(
        index,
        SearchPlan(
            (
                Approach(1, FindText("absent_word"), ()),
                Approach(2, FileUnits("invented.py"), ()),
                Approach(3, FileUnits("real.py"), ()),
            )
        ),
        box_chars=70000,
    )
    assert [binding_status(outcome) for outcome in result.outcomes] == [
        "valid empty search",
        "invalid or unbound argument",
        "bound to real code",
    ]
