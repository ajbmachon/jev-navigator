"""Replay the observed blank provider reply at a real HTTP boundary without paid sends."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from paid_support import Ledger

from . import run
from .designs import DESIGNS, llm_request


def test_blank_reply_preserves_wire_and_bill_and_valid_typed_reply_settles(tmp_path, monkeypatch):
    blank = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant"}}], "model": "fixture"}
    valid = {
        "choices": [{"finish_reason": "stop", "message": {"content": '{"p0":["2"]}'}}],
        "model": "fixture",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.00001},
    }
    replies, sent = iter((blank, valid)), []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            sent.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(next(replies)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    request = llm_request(
        DESIGNS[-3],
        {"state": {"targets": {"p0": "Reject negative quantity"}, "items": [{"code": "return q"}] * 16}},
    )
    ledger = Ledger(tmp_path / "ledger.jsonl", cap="7")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(run, "REQUESTY", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(run, "resources", lambda: None)
    monkeypatch.setenv("REQUESTY_API_KEY", "fixture")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    try:
        with pytest.raises(run.ProviderContractError):
            run.llm_call({"request": request}, first, ledger)
        assert json.loads((first / "response.json").read_text()) == blank
        assert json.loads((first / "request.json").read_text()) == request
        assert next(iter(ledger.latest().values()))["status"] == "reserved"
        assert json.loads((first / "attempt.json").read_text())["id"] in ledger.latest()
        rows = run.llm_call({"request": request}, second, ledger)
        assert [row["slot"] for row in rows if row["score"]] == [2]
        assert sorted(event["status"] for event in ledger.latest().values()) == ["reserved", "settled"]
        assert sent == [request, request]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_systemic_provider_failure_stops_pending_groups(tmp_path, monkeypatch):
    import asyncio
    import time

    baseline = {
        "state": {"targets": {"p0": "Reject negative quantity"}, "items": [{"code": "return q"}] * 16}
    }
    request = llm_request(DESIGNS[-3], baseline)
    from .designs import identity

    records = [
        {
            "case": "fixture",
            "group": group,
            "eligible": True,
            "request": request,
            "request_sha256": identity(request),
        }
        for group in range(12)
    ]
    sent = []

    def broken_provider(record, destination, ledger):
        sent.append(record["group"])
        time.sleep(0.01)
        raise run.ProviderContractError("Recorded blank provider reply")

    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "records", lambda name: iter(records))
    monkeypatch.setattr(run, "load", lambda path: {"groups": {"fixture": [0]}})
    monkeypatch.setattr(run, "llm_call", broken_provider)
    with pytest.raises(run.ProviderContractError):
        asyncio.run(run.evaluate(["L1-1"], Ledger(tmp_path / "ledger.jsonl", cap="7"), pilot=False))
    assert 1 <= len(sent) <= 4
    assert not any(group >= 4 for group in sent)
    assert list(tmp_path.glob("answers/L1-1/fixture/*/observation.json"))
