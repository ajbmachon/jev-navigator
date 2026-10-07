"""The paid recipe owns each physical send and completes lower checkpoints first."""

import json
from pathlib import Path
from threading import Lock

import pytest


@pytest.fixture
def trial(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/pack_case2"))
    import paid_trial

    monkeypatch.setattr(paid_trial, "resources", lambda: None)
    return paid_trial


def test_actual_sdk_dispatch_reserves_before_send_and_finishes_lower_rounds(trial, monkeypatch, tmp_path):
    import httpx2

    seen = []
    lock = Lock()
    ledger = trial.SpendLedger(tmp_path / "ledger.jsonl", cap="3.00")

    def provider(request):
        body = json.loads(request.content)
        finding, ordinal = body["state"]["finding"], body["state"]["ordinal"]
        identifier = f"union:{finding}:{ordinal}"
        assert any(e["id"] == identifier and e["status"] == "reserved" for e in ledger.events)
        with lock:
            seen.append((finding, ordinal))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 100, "output_tokens": 0},
                "answers": {name: {"type": "noul", "noul": 0.8} for name in body["questions"]},
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(httpx2, "HTTPTransport", lambda: httpx2.MockTransport(provider))
    monkeypatch.setattr(trial, "OUT", tmp_path)
    for name in ("a", "b", "c"):
        folder = tmp_path / "cases" / name
        folder.mkdir(parents=True)
        for ordinal in range(1, 7):
            request = {
                "model": "jev-latest",
                "state": {"finding": name, "ordinal": ordinal},
                "questions": {"match#0": {"type": "noul", "instructions": "Is this a match?"}},
            }
            trial.append(
                folder / "prepared-requests.jsonl",
                {
                    "ordinal": ordinal,
                    "request": request,
                    "request_sha256": trial.content_hash(
                        {"state": request["state"], "questions": request["questions"]}
                    ),
                },
            )
    trial.dispatch(ledger)
    assert set(seen[:12]) == {(name, n) for name in "abc" for n in range(1, 5)}
    assert set(seen[12:]) == {(name, n) for name in "abc" for n in (5, 6)}
    assert len(seen) == 18
    for name in "abc":
        folder = tmp_path / "cases" / name
        receipts = list(trial.rows(folder / "responses.jsonl"))
        attempts = list(trial.rows(folder / "transport-attempts.jsonl"))
        assert len(receipts) == len(attempts) == 6
        assert {r["ordinal"] for r in receipts} == set(range(1, 7))
    trial.dispatch(trial.SpendLedger(tmp_path / "ledger.jsonl", cap="3.00"))
    assert len(seen) == 18


def test_failed_physical_send_is_not_retried_and_keeps_unknown_spend_reserved(trial, monkeypatch, tmp_path):
    import httpx2
    from typesafe_sdk import TypeSafeInternalServerError

    seen = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def provider(request):
        seen.append(request.content)
        return httpx2.Response(503, json={"detail": "temporary failure"})

    monkeypatch.setattr(httpx2, "HTTPTransport", lambda: httpx2.MockTransport(provider))
    ledger = trial.SpendLedger(tmp_path / "ledger.jsonl", cap="3.00")
    record = {
        "ordinal": 1,
        "request": {
            "model": "jev-latest",
            "state": "sample",
            "questions": {
                "match#0": {"type": "noul", "instructions": "Is this a match?"},
            },
        },
    }
    record["request_sha256"] = trial.content_hash(
        {
            "state": record["request"]["state"],
            "questions": record["request"]["questions"],
        }
    )
    with pytest.raises(TypeSafeInternalServerError):
        trial.send(ledger, tmp_path, record)
    assert len(seen) == 1
    assert list(trial.rows(tmp_path / "transport-attempts.jsonl"))[0]["status"] == 503
    assert ledger.events[-1]["status"] == "reserved"
    with pytest.raises(RuntimeError, match="already recorded"):
        trial.send(trial.SpendLedger(tmp_path / "ledger.jsonl", cap="3.00"), tmp_path, record)
    assert len(seen) == 1
