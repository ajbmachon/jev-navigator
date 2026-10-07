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
