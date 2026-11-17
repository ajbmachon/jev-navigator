"""Frozen-group splitting at the real main Judge and HTTP reservation boundary."""

import asyncio
import json
import socket
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import paid_client
import pytest
from paid_support import CapStopError, Ledger, load
from typesafe_sdk import TypeSafeAPIResponseValidationError, TypeSafeClient

from . import split_reach


@pytest.mark.parametrize("missing_usage", [False, True])
def test_split_group_retries_keep_wire_reserves_and_original_slots(tmp_path, monkeypatch, missing_usage):
    record = next(r for r in split_reach.records() if not r["eligible"])
    sent = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            wire = self.rfile.read(int(self.headers["Content-Length"]))
            sent.append(wire)
            request = json.loads(wire)
            if len(sent) == 1:
                status, raw = 529, {"detail": {"error_type": "system_overloaded"}}
            elif len(sent) == 2:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            else:
                status = 200
                raw = {
                    "model": "fixture",
                    "usage": {"input_tokens": 100},
                    "answers": {qid: {"type": "noul", "noul": 0.9} for qid in request["questions"]},
                }
            if missing_usage and status == 200:
                raw.pop("usage")
            body = json.dumps(raw).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setattr(split_reach, "OUT", tmp_path)
    monkeypatch.setattr(paid_client, "resources", lambda: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        paid_client,
        "TypeSafeClient",
        partial(TypeSafeClient, base_url=f"http://127.0.0.1:{server.server_port}"),
    )
    ledger = Ledger(tmp_path / "ledger.jsonl", cap="0.40")
    try:
        if missing_usage:
            with pytest.raises(TypeSafeAPIResponseValidationError, match="usage"):
                asyncio.run(split_reach.paid_group(record, ledger))
            assert len(sent) == 3
            assert all(event["status"] == "reserved" for event in ledger.latest().values())
            base = split_reach.group_folder(record)
            assert any(load(path)["status"] == 200 for path in (base / "http").glob("*/receipt.json"))
            assert all(
                path.with_name("response.json").exists()
                for path in (base / "http").glob("*/receipt.json")
                if load(path)["status"] == 200
            )
            return
        asyncio.run(split_reach.paid_group(record, ledger))
        assert len(sent) == 5 and sent[0] == sent[1] == sent[2]
        requests = [json.loads(wire) for wire in sent[2:]]
        assert [len(request["state"]["items"]) for request in requests] == [4, 4, 8]
        assert [item for request in requests for item in request["state"]["items"]] == record["request"][
            "state"
        ]["items"]
        observation = load(split_reach.group_folder(record) / "observation.json")
        assert observation["failure"] is None
        assert sorted(row["slot"] for row in observation["rows"]) == list(range(16))
        assert all(row["score"] == 0.9 for row in observation["rows"])
        assert sorted(event["status"] for event in ledger.latest().values()) == [
            "reserved",
            "reserved",
            "settled",
            "settled",
            "settled",
        ]
        assert float(observation["pending_reserved_usd"]) > 0
        prior_events = len(ledger.events)
        asyncio.run(split_reach.paid_group(record, ledger))
        assert len(sent) == 5 and len(ledger.events) == prior_events
        changed = {**record, "case": "changed-company"}
        with pytest.raises(CapStopError):
            asyncio.run(split_reach.paid_group(changed, Ledger(tmp_path / "empty.jsonl", cap="0")))
        assert len(sent) == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
