"""Integration at the actual Engine boundary, using a retained group and independent answers."""

import asyncio

from . import run
from .frozen import load
from .refresh import ADMITTED


def test_scalar_recipes_reach_async_engine_and_preserve_source_floor_and_rooms(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    asyncio.run(run.proof(load(ADMITTED / "frozen-cases.json")))
    proof = load(tmp_path / "engine-proof.json")
    assert proof["status"] == "PASS"
    assert len(proof["receipts"]) == 3


def test_meta_fragment_scan_uses_original_source_path_and_still_rejects_literals():
    import pytest

    from jev_navigator.judgments.secrets import SecretInRequestError, refuse_if_secret

    code = "interface Fence {\n  fencingToken: string;\n}"
    state = {
        "request_state": {"items": [{"file": "owner.ts", "code": code}]},
        "state_components": [{"path": "items[0].code", "value": code}],
    }
    prepared = {"requests": [{"submitted": {"state": state}}]}
    with pytest.raises(SecretInRequestError):
        refuse_if_secret(state, {}, run.SecretScanner())
    refuse_if_secret(state, {}, run.LocatedReviewScanner(prepared))
    state["state_components"][0]["value"] = "api_key = 'fixture-secret-value'"
    with pytest.raises(SecretInRequestError):
        refuse_if_secret(state, {}, run.LocatedReviewScanner(prepared))


def test_scalar_lab_accepts_groups_without_six_role_baseline_answers():
    from paid_report import COMPARE, HARD_HISTORY, cases_only, historical_case
    from paid_run import scopes

    from .packing import lab_rooms

    previous = {(r["case"], r["arm"]): r for r in load(run.OUT / "per-finding-results.json")}
    dev, hard = cases_only(COMPARE / "step-3/metadata.json"), cases_only(HARD_HISTORY / "windows.json")
    cases = load(ADMITTED / "frozen-cases.json")
    chosen = [
        next(c for c in cases if c["context"]["case"] == "P1"),
        next(c for c in cases if c["context"]["dataset"] == "dev110"),
    ]
    cached = {}
    for original in chosen:
        context = original["context"]
        _, index = scopes(context, cached)
        case = {
            **original,
            "groups": [{k: v for k, v in g.items() if k != "baseline_rows"} for g in original["groups"]],
        }
        history = historical_case(context, case["labels"], dev, hard)
        actual = lab_rooms(history, run.bindings(case), case, run.observations("J1-3", case), index)
        expected = previous[context["case"], "J1-3"]["lab"]
        assert {
            room: [label["delivered"] for label in value["labels"]] for room, value in actual.items()
        } == {room: [label["delivered"] for label in value["labels"]] for room, value in expected.items()}
