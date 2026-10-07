"""Offline transport proof for paid accounting, masked source delivery and cap admission."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from shop_search import shop_index

sys.path.insert(0, str(Path(__file__).parents[1] / "measurements/pack_case3"))
from trial import AgentProvider, RankingProvider, run_case
from trial_budget import SpendLedger, SpendStopError


class CharacterCounter:
    def encode(self, text, **_):
        return text


@pytest.fixture
def endpoint():
    received = []
    responses = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(responses.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received, responses
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def ledger_at(root):
    path = root / "ledger.jsonl"
    path.write_text(json.dumps({"event": "usage", "category": "jev_guard", "usd": "0.005944176"}) + "\n")
    return SpendLedger(path)


def provider_at(ledger, url):
    return AgentProvider(
        ledger,
        {"context_window": 1048576, "input_price": 2.8e-7, "cached_price": 7e-8, "output_price": 5.6e-7},
        url,
        "offline-test",
    )


def response(message, *, usage=True):
    raw = {
        "id": "local-response",
        "model": "test-model",
        "choices": [
            {"finish_reason": "tool_calls" if message.get("tool_calls") else "stop", "message": message}
        ],
    }
    if usage:
        raw["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
    return raw


def test_real_batch_source_and_withheld_exclusion_reach_agent_with_reconciled_billing(
    tmp_path, endpoint, monkeypatch
):
    url, received, responses = endpoint
    root = tmp_path / "repo"
    root.mkdir()
    with shop_index(
        root,
        {
            "limit.py": "MAX = 4\ndef check(n):\n    return n <= MAX\n",
            "withheld.py": "SECRET = 'withheld'\n",
            "AGENTS.md": "hidden instructions",
        },
    ) as index:
        pack = {
            "case": "test-finding",
            "repository": str(root),
            "commit": index.commit,
            "withheld": ["withheld.py"],
            "claim": {"statement": "check limits", "evidence": [{"file": "limit.py"}]},
        }
    args = {
        "operations": [
            {"op": "show", "file": "limit.py", "line": 1, "end": 3},
            {"op": "named_files"},
            {"op": "show", "file": "withheld.py"},
        ]
    }
    responses.extend(
        [
            response(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "batched_jvn", "arguments": json.dumps(args)},
                        }
                    ],
                }
            ),
            response({"role": "assistant", "content": "limit.py:1-3 decides the limit. No remaining gap."}),
        ]
    )
    ledger = ledger_at(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    original_check_output = subprocess.check_output

    def without_mac_memory_probe(command, *args, **kwargs):
        if command[0] == "vm_stat":
            raise FileNotFoundError("vm_stat is absent on Linux")
        return original_check_output(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", without_mac_memory_probe)
    result = run_case(pack, out, "Find deciding source.", provider_at(ledger, url), CharacterCounter())
    assert result["status"] == "agent_final"
    assert received[0]["max_tokens"] == 8000
    assert result["jvn_calls"] == 1 and result["agent_requests"] == 2 and result["jev_requests"] == 0
    assert Decimal(result["agent_usd"]) == Decimal("0.0000812")
    assert ledger.spent == Decimal("0.006025376")
    tool = json.loads(next(m["content"] for m in received[1]["messages"] if m["role"] == "tool"))
    assert tool["pages"][0]["items"][0]["text"] == "MAX = 4\ndef check(n):\n    return n <= MAX"
    assert [x["file"] for x in tool["pages"][1]["items"]] == ["limit.py"]
    assert tool["pages"][2]["error"]
    rows = [json.loads(line) for line in (out / "test-finding/source-lines.jsonl").read_text().splitlines()]
    assert [(r["file"], r["line"]) for r in rows] == [("limit.py", 1), ("limit.py", 2), ("limit.py", 3)]
    assert (out / "test-finding/agent-01-request.bin").exists()
    assert (out / "test-finding/agent-02-response.bin").exists()
    assert not ledger.reserved


def test_next_possible_bill_is_denied_before_http_and_unknown_usage_retains_reservation(tmp_path, endpoint):
    url, received, responses = endpoint
    ledger = ledger_at(tmp_path)
    ledger.cap = Decimal("0.01")
    provider = provider_at(ledger, url)
    with pytest.raises(SpendStopError):
        provider.ask("case", tmp_path, 1, [], 1000, True)
    assert received == []
    assert ledger.spent == Decimal("0.005944176")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    ledger = ledger_at(fresh)
    provider = provider_at(ledger, url)
    responses.append(response({"role": "assistant", "content": "unknown usage"}, usage=False))
    with pytest.raises(KeyError):
        provider.ask("case", fresh, 1, [], 1000, True)
    assert len(received) == 1 and ledger.halted and ledger.reserved
    with pytest.raises(SpendStopError):
        provider.ask("case", fresh, 2, [], 1000, True)
    assert len(received) == 1
    with pytest.raises(SpendStopError):
        SpendLedger(ledger.path)


def test_rank_sdk_preserves_wire_receipts_and_accounts_usage_before_answer_decoding(
    tmp_path, endpoint, monkeypatch
):
    pytest.importorskip("typesafe_sdk")
    url, received, responses = endpoint
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test")
    monkeypatch.setenv("TYPESAFE_BASE_URL", url)
    key = "match_query@349c91fb#0"
    state = {
        "items": [{"id": "x.py:1-2", "code": "def check(): return 4"}],
        "targets": {"query": "checks the limit"},
    }
    questions = {
        key: {
            "type": "noul",
            "instructions": "Look only at items[0]. Does that code match the description in targets.query?",
        }
    }
    responses.append(
        {
            "model": "jev-test",
            "usage": {"input_tokens": 200, "output_tokens": 5},
            "answers": {key: {"type": "noul", "noul": 0.7}},
        }
    )
    ledger = ledger_at(tmp_path)
    ranking = RankingProvider(ledger, "case", tmp_path)
    result = ranking.ask(state, questions)
    assert result.noul(key).probability == 0.7
    assert received[0]["state"] == state and received[0]["questions"] == questions
    assert ledger.spent == Decimal("0.005952576")
    assert not ledger.reserved
    assert json.loads((tmp_path / "jev-01.request.bin").read_bytes())["state"] == state
    assert json.loads((tmp_path / "jev-01.response.bin").read_bytes())["usage"]["input_tokens"] == 200


def test_continuation_preserves_five_used_calls_source_receipts_and_cumulative_usage(tmp_path, endpoint):
    url, received, responses = endpoint
    root = tmp_path / "repo"
    root.mkdir()
    with shop_index(root, {"limit.py": "MAX = 4\ndef check(n):\n    return n <= MAX\n"}) as index:
        pack = {
            "case": "continued",
            "repository": str(root),
            "commit": index.commit,
            "withheld": [],
            "claim": {"statement": "check limits", "evidence": [{"file": "limit.py"}]},
        }
    arguments = json.dumps({"operations": [{"op": "show", "file": "limit.py", "line": 1, "end": 3}]})
    tool_calls = [
        {"id": f"call-{i}", "type": "function", "function": {"name": "batched_jvn", "arguments": arguments}}
        for i in range(5)
    ]
    partial = response({"role": "assistant", "content": "The final answer was cut"})
    partial["choices"][0]["finish_reason"] = "length"
    responses.extend([response({"role": "assistant", "tool_calls": tool_calls}), partial])
    ledger = ledger_at(tmp_path)
    provider = provider_at(ledger, url)
    out = tmp_path / "output"
    out.mkdir()
    first = run_case(pack, out, "Find evidence.", provider, CharacterCounter())
    assert first["status"] == "agent_output_cap" and first["jvn_calls"] == 5
    original_request = (out / "continued/agent-01-request.bin").read_bytes()
    original_input_time = (out / "continued/input.json").stat().st_mtime_ns
    responses.append(response({"role": "assistant", "content": "limit.py:1-3. Enough evidence."}))
    final = run_case(pack, out, "Find evidence.", provider, CharacterCounter(), resume=True)
    assert final["status"] == "agent_final" and final["jvn_calls"] == 5
    assert final["agent_requests"] == 3 and final["source_lines"] == 3
    assert Decimal(final["agent_usd"]) == Decimal("0.0001218")
    assert "tools" not in received[-1] and received[-1]["max_tokens"] == 7940
    assert (out / "continued/agent-01-request.bin").read_bytes() == original_request
    assert (out / "continued/input.json").stat().st_mtime_ns == original_input_time
    assert json.loads((out / "continued/result-before-resume.json").read_text())["jvn_calls"] == 5
    assert not ledger.reserved


def test_reported_agent_dollars_are_ledger_authority_when_the_catalog_calculation_differs(tmp_path, endpoint):
    url, _, responses = endpoint
    raw = response({"role": "assistant", "content": "Done"})
    raw["usage"]["cost"] = 0.00006
    responses.append(raw)
    ledger = ledger_at(tmp_path)
    provider_at(ledger, url).ask("case", tmp_path, 1, [], 1000, False)
    assert ledger.spent == Decimal("0.006004176")
    [usage] = ledger.case_usage("case")
    assert usage["usd"] == "0.00006" and usage["usd_source"] == "provider_usage.cost"
    assert Decimal(usage["catalog_usd"]) == Decimal("0.0000406")
