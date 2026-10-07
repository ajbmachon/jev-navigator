"""Run the frozen cheap-agent search trial through the public batched JVN library.

Only the transport is provider-specific. Labels are read by the separate post-run scorer.
The host exposes no shell, external files, web, historical answers or deciding-line labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

from replay import check_resources, source_rows
from trial_budget import SpendLedger, SpendStopError

from jev_navigator.batch import Operation, run_batch
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.answers import response_from_raw
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.store import SqliteAnswerStore

MODEL = "sference/deepseek-v4-flash-0731"
MAX_TOOL_CALLS = 5
MAX_INPUT = 80_000
MAX_OUTPUT = 8_000
MAX_JEV = 8


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def tool_contract() -> dict:
    """Expose exactly the documented Operation fields, with host-owned budgets omitted."""
    properties = {key: {"type": "string"} for key in ("file", "name", "query")}
    properties.update({key: {"type": "integer"} for key in ("line", "end", "window", "limit")})
    properties.update({key: {"type": "array", "items": {"type": "string"}} for key in ("patterns", "scopes")})
    properties.update(
        {
            "op": {
                "type": "string",
                "enum": [
                    "outline",
                    "names",
                    "def",
                    "refs",
                    "callers",
                    "callees",
                    "named_files",
                    "show",
                    "cochange",
                    "tests_of",
                    "rank",
                ],
            },
            "regex": {"type": "boolean"},
            "cursor": {
                "type": "object",
                "properties": {"row": {"type": "integer"}, "character": {"type": "integer"}},
                "additionalProperties": False,
            },
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                    "required": ["file", "start", "end"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return {
        "type": "function",
        "function": {
            "name": "batched_jvn",
            "description": "Run numbered, pageable code operations over the supplied masked repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": properties,
                            "required": ["op"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["operations"],
                "additionalProperties": False,
            },
        },
    }


class AgentProvider:
    def __init__(self, ledger, catalog, base_url, api_key):
        self.ledger = ledger
        self.catalog = catalog
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def ask(self, case, directory, turn, messages, output_limit, tools):
        payload = {"model": MODEL, "messages": messages, "max_tokens": output_limit, "stream": False}
        if tools:
            payload.update(tools=[tool_contract()], parallel_tool_calls=False)
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        # Reserve the catalog's complete context window, not an estimated tokenizer count.
        maximum = Decimal(self.catalog["context_window"]) * Decimal(
            str(self.catalog["input_price"])
        ) + Decimal(output_limit) * Decimal(str(self.catalog["output_price"]))
        ticket = self.ledger.reserve("agent", case, maximum)
        save(directory / f"agent-{turn:02d}-request.json", payload)
        (directory / f"agent-{turn:02d}-request.bin").write_bytes(encoded)
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=encoded,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        try:
            try:
                with urllib.request.urlopen(request) as response:
                    body = response.read()
            except urllib.error.HTTPError as error:
                body = error.read()
                (directory / f"agent-{turn:02d}-response.bin").write_bytes(body)
                raise
            (directory / f"agent-{turn:02d}-response.bin").write_bytes(body)
            raw = json.loads(body)
            save(directory / f"agent-{turn:02d}-response.json", raw)
            usage = raw["usage"]
            input_tokens, output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if not all(type(n) is int and n >= 0 for n in (input_tokens, output_tokens, cached)):
                raise ValueError("Invalid reported token counts")
            if cached > input_tokens:
                raise ValueError("Cached input exceeds total prompt usage")
            catalog_usd = (
                (Decimal(input_tokens - cached) * Decimal(str(self.catalog["input_price"])))
                + Decimal(cached) * Decimal(str(self.catalog["cached_price"]))
                + Decimal(output_tokens) * Decimal(str(self.catalog["output_price"]))
            )
            reported_usd = usage.get("cost")
            usd = catalog_usd if reported_usd is None else Decimal(str(reported_usd))
            if not usd.is_finite() or usd < 0:
                raise ValueError("Invalid reported provider cost")
        except Exception:
            self.ledger.unknown(ticket, category="agent", case=case, turn=turn)
            raise
        self.ledger.settle(
            ticket,
            usd=usd,
            category="agent",
            case=case,
            turn=turn,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached,
            reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            request_id=raw.get("id"),
            model=raw.get("model"),
            catalog_usd=str(catalog_usd),
            usd_source="catalog_usage" if reported_usd is None else "provider_usage.cost",
        )
        return raw


class RankingProvider:
    """Keep exact SDK wire receipts and account before decoding into JVN answers."""

    model = "jev-latest"

    def __init__(self, ledger, case, directory):
        self.ledger, self.case, self.directory = ledger, case, directory
        self.calls = 0

    def ask(self, state, questions):
        import httpx2
        from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

        prepared_questions = {key: Noul.model_validate(q) for key, q in questions.items()}
        if not 1 <= len(state["items"]) <= 16:
            raise ValueError("Rank requests must contain one to sixteen supplied candidates")
        if self.calls >= MAX_JEV:
            raise SpendStopError("Eight candidate Jev requests have already been sent")
        ticket = self.ledger.reserve("jev_candidate", self.case, Decimal("0.002688"))
        self.calls += 1
        number = self.calls
        prefix = self.directory / f"jev-{number:02d}"
        save(
            prefix.with_suffix(".request.json"), {"state": state, "questions": questions, "model": self.model}
        )
        accounted = False

        def request_hook(request):
            prefix.with_suffix(".request.bin").write_bytes(request.read())

        def response_hook(response):
            nonlocal accounted
            body = response.read()
            prefix.with_suffix(".response.bin").write_bytes(body)
            raw = json.loads(body)
            save(prefix.with_suffix(".response.json"), raw)
            usage = raw["usage"]
            tokens = usage["input_tokens"]
            out = usage["output_tokens"]
            if not all(type(n) is int and n >= 0 for n in (tokens, out)):
                raise ValueError("Invalid reported Jev usage")
            accounted = True
            self.ledger.settle(
                ticket,
                usd=Decimal(tokens) * Decimal("0.000000042"),
                category="jev_candidate",
                case=self.case,
                input_tokens=tokens,
                output_tokens=out,
                model=raw.get("model"),
                request_id=response.headers.get("x-typesafe-request-id"),
                request_number=number,
            )

        try:
            with (
                httpx2.Client(
                    timeout=None, event_hooks={"request": [request_hook], "response": [response_hook]}
                ) as transport,
                TypeSafeClient(retry=RetryPolicy(max_retries=0), http_client=transport) as sdk,
            ):
                response = sdk.system_one(
                    state=state,
                    questions=prepared_questions,
                    model=self.model,
                )
                return response_from_raw(json.loads(response.raw_http_response.content))
        except Exception:
            if not accounted:
                self.ledger.unknown(ticket, category="jev_candidate", case=self.case, request_number=number)
            raise


def run_case(pack, out, prompt, provider, encoding, *, resume=False):
    """One adaptive search. Its observed output is scored only after all runs complete."""
    cid = pack["case"]
    directory = out / cid.replace(":", "_")
    if not resume:
        directory.mkdir()
    start = time.perf_counter()
    if not resume:
        save(directory / "input.json", pack)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(pack["claim"], ensure_ascii=False)},
    ]
    fragments, shown = defaultdict(str), {}
    calls, returned_tokens, final, status = 0, 0, "", "searching"
    tool_errors, uninspected_pages, turn = [], [], 0
    ranking = RankingProvider(provider.ledger, cid, directory)
    previous = None
    started_at = time.time()
    if resume:
        previous = json.loads((directory / "result.json").read_text())
        if previous["status"] != "agent_output_cap":
            raise ValueError("Only the prematurely capped response is eligible for this continuation")
        save(directory / "result-before-resume.json", previous)
        started_at = (directory / "input.json").stat().st_mtime
        messages = json.loads((directory / "messages.json").read_text())
        save(directory / "messages-before-resume.json", messages)
        # An incomplete model tool request was never executed and must not become an action.
        if messages[-1].get("tool_calls"):
            messages[-1].pop("tool_calls")
            messages[-1]["content"] = (messages[-1].get("content") or "") + (
                "\nThe incomplete tool request was not executed."
            )
        calls = previous["jvn_calls"]
        returned_tokens = previous["returned_tokens_cl100k"]
        tool_errors = previous["tool_errors"]
        uninspected_pages = previous["issued_continuations"]
        turn = previous["agent_requests"]
        ranking.calls = previous["jev_requests"]
        with (directory / "source-lines.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                shown[(row["file"], row["line"])] = row
        for number in range(1, calls + 1):
            request = json.loads((directory / f"tool-{number:02d}-request.json").read_text())
            payload = json.loads((directory / f"tool-{number:02d}-response.json").read_text())
            if "pages" not in payload:
                continue
            operations = tuple(
                Operation.from_dict(op) for op in json.loads(request["function"]["arguments"])["operations"]
            )
            list(source_rows(payload, fragments, operations))
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response hit a per-response ceiling below the case's allowance. "
                    "Continue from this saved transcript without repeating inspected evidence. "
                    f"You have {MAX_TOOL_CALLS - calls} batched tool calls remaining. "
                    "Finish with exact source ranges and remaining gaps "
                    "under the existing case token allowance."
                ),
            }
        )
    try:
        with CodeIndex.from_git(Path(pack["repository"])) as tracked:
            files = [
                f
                for f in tracked.files
                if f not in pack["withheld"] and Path(f).name not in ("AGENTS.md", "CLAUDE.md")
            ]
            with CodeIndex(tracked.root, files, commit=tracked.commit) as index:
                save(
                    directory / "revision.json",
                    {"index_commit": tracked.commit, "pack_commit": pack["commit"]},
                )
                judge = Judge(
                    ranking,
                    max_calls=MAX_JEV - ranking.calls,
                    items_per_request=16,
                    max_concurrency=1,
                    store=SqliteAnswerStore(directory / "answers.sqlite"),
                )
                while True:
                    if provider.ledger.halted:
                        raise SpendStopError("Shared ledger stopped further spend")
                    usage = provider.ledger.case_usage(cid)
                    agent = [r for r in usage if r["category"] == "agent"]
                    billed_input = sum(r["input_tokens"] for r in agent)
                    billed_output = sum(r["output_tokens"] for r in agent)
                    if billed_input >= MAX_INPUT or billed_output >= MAX_OUTPUT:
                        status = "agent_token_cap"
                        break
                    turn += 1
                    raw = provider.ask(
                        cid,
                        directory,
                        turn,
                        messages,
                        MAX_OUTPUT - billed_output,
                        calls < MAX_TOOL_CALLS,
                    )
                    choice = raw["choices"][0]
                    message = choice["message"]
                    messages.append(
                        {
                            k: message[k]
                            for k in ("role", "content", "tool_calls", "reasoning_content")
                            if k in message and message[k] is not None
                        }
                    )
                    latest = provider.ledger.case_usage(cid)
                    agent_usage = [r for r in latest if r["category"] == "agent"]
                    if sum(r["input_tokens"] for r in agent_usage) > MAX_INPUT:
                        final, status = message.get("content") or "", "agent_token_cap"
                        break
                    if choice["finish_reason"] == "length":
                        final, status = message.get("content") or "", "agent_output_cap"
                        break
                    requests = message.get("tool_calls") or []
                    if not requests:
                        final, status = message.get("content") or "", "agent_final"
                        break
                    if calls >= MAX_TOOL_CALLS:
                        status = "tool_call_cap"
                        break
                    for request in requests:
                        if calls >= MAX_TOOL_CALLS:
                            status = "tool_call_cap"
                            break
                        calls += 1
                        save(directory / f"tool-{calls:02d}-request.json", request)
                        try:
                            if request["function"]["name"] != "batched_jvn":
                                raise ValueError("Only batched_jvn is available")
                            args = json.loads(request["function"]["arguments"])
                            if set(args) != {"operations"} or not isinstance(args["operations"], list):
                                raise ValueError("Expected an operations array")
                            operations = tuple(Operation.from_dict(op) for op in args["operations"])
                            tool_start = time.perf_counter()
                            result = run_batch(index, operations, judge=judge, max_chars=24_000)
                            text = result.render()
                            save(
                                directory / f"tool-{calls:02d}-timing.json",
                                {"wall_seconds": time.perf_counter() - tool_start},
                            )
                            payload = json.loads(text)
                            tool_errors.extend(
                                {"call": calls, "error": page.error} for page in result.pages if page.error
                            )
                            uninspected_pages.extend(
                                {"call": calls, "operation": p.operation, "next": p.to_dict()["next"]}
                                for p in result.pages
                                if p.next
                            )
                            for source in source_rows(payload, fragments, operations):
                                key = (source["file"], source["line"])
                                shown.setdefault(key, {**source, "first_call": calls})
                        except (ValueError, TypeError, KeyError) as error:
                            text = json.dumps({"error": str(error)})
                            tool_errors.append({"call": calls, "error": str(error)})
                        (directory / f"tool-{calls:02d}-response.json").write_text(text + "\n")
                        returned_tokens += len(encoding.encode(text, disallowed_special=()))
                        messages.append({"role": "tool", "tool_call_id": request["id"], "content": text})
                    if status == "tool_call_cap":
                        break
                    if calls == MAX_TOOL_CALLS:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "All five batched calls are used. Return your final source ranges "
                                    "and remaining gaps now."
                                ),
                            }
                        )
    except Exception as error:
        status = "spend_stop" if isinstance(error, SpendStopError) else "error"
        tool_errors.append({"error": f"{type(error).__name__}: {error}"})
    usage = provider.ledger.case_usage(cid)
    agent = [r for r in usage if r["category"] == "agent"]
    jev = [r for r in usage if r["category"] == "jev_candidate"]
    save(directory / "messages.json", messages)
    with (directory / "source-lines.jsonl").open("w") as output:
        for source in shown.values():
            output.write(json.dumps(source, ensure_ascii=False) + "\n")
    result = {
        "case": cid,
        "status": status,
        "jvn_calls": calls,
        "agent_requests": len(agent),
        "jev_requests": len(jev),
        "returned_tokens_cl100k": returned_tokens,
        "agent_input_tokens": sum(r["input_tokens"] for r in agent),
        "agent_output_tokens": sum(r["output_tokens"] for r in agent),
        "agent_cached_tokens": sum(r.get("cached_tokens", 0) for r in agent),
        "agent_usd": str(sum((Decimal(r["usd"]) for r in agent), Decimal(0))),
        "jev_usd": str(sum((Decimal(r["usd"]) for r in jev), Decimal(0))),
        "jev_input_tokens": sum(r["input_tokens"] for r in jev),
        "jev_output_tokens": sum(r["output_tokens"] for r in jev),
        "wall_seconds": time.time() - started_at if resume else time.perf_counter() - start,
        "active_wall_seconds": time.perf_counter() - start + (previous["wall_seconds"] if previous else 0),
        "resumed_response_ceiling": resume,
        "source_lines": len(shown),
        "source_files": len({file for file, _ in shown}),
        "final": final,
        "tool_errors": tool_errors,
        "issued_continuations": uninspected_pages,
        "unfinished_fragments": len(fragments),
    }
    save(directory / "result.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ("case", "status", "jvn_calls", "agent_usd", "jev_usd", "wall_seconds")}
        ),
        flush=True,
    )
    return result


def main():
    import tiktoken

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-output-caps", action="store_true")
    args = parser.parse_args()
    if not args.resume_output_caps:
        args.out.mkdir(exist_ok=False)
    ledger = SpendLedger(args.case_folder / "spend-ledger.jsonl")
    catalog = json.loads((args.case_folder / "requesty-model-catalog.json").read_text())["models"][0]
    assert catalog["id"] == MODEL and catalog["geolocation"] == "eu"
    endpoint = os.environ["REQUESTY_BASE_URL"].rstrip("/")
    assert endpoint == "https://router.eu.requesty.ai/v1"
    provider = AgentProvider(ledger, catalog, endpoint, os.environ["REQUESTY_API_KEY"])
    packs = json.loads((args.case_folder / "trial-final/agent-inputs.json").read_text())
    assert len(packs) == 20
    frozen_prompt = (args.case_folder / "TRIAL-PROMPT.md").read_text()
    prompt = frozen_prompt.split("# Host measurement contract")[0]
    prompt += "\nPublic tool contract supplied inline:\n" + Path("docs/batch.md").read_text()
    save(
        args.out / ("resume-provenance.json" if args.resume_output_caps else "provenance.json"),
        {
            "model": MODEL,
            "provider": "requesty",
            "endpoint": endpoint,
            "catalog": catalog,
            "starting_usd": str(ledger.spent),
            "workers": 3,
            "jvn_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_commit_before": subprocess.check_output(
                ["git", "-C", packs[0]["repository"], "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_status_before": subprocess.check_output(
                ["git", "-C", packs[0]["repository"], "status", "--porcelain"], text=True
            ),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "inputs_sha256": hashlib.sha256(
                (args.case_folder / "trial-final/agent-inputs.json").read_bytes()
            ).hexdigest(),
            "limits": {
                "tool_calls": 5,
                "batch_chars": 24000,
                "candidate_jev_requests": 8,
                "agent_input_tokens": 80000,
                "agent_output_tokens": 8000,
            },
        },
    )
    encoding = tiktoken.get_encoding("cl100k_base")
    unchanged = []
    if args.resume_output_caps:
        existing = json.loads((args.out / "results.json").read_text())
        unchanged = [row for row in existing if row["status"] != "agent_output_cap"]
        selected = {row["case"] for row in existing if row["status"] == "agent_output_cap"}
        packs = [pack for pack in packs if pack["case"] in selected]

    def checked_case(pack):
        resources = check_resources()
        save(args.out / f"resources-{pack['case'].replace(':', '_')}.json", resources)
        return run_case(pack, args.out, prompt, provider, encoding, resume=args.resume_output_caps)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(checked_case, pack) for pack in packs]
        results = [future.result() for future in as_completed(futures)]
    results.extend(unchanged)
    save(args.out / "results.json", results)
    save(
        args.out / "source-after.json",
        {
            "commit": subprocess.check_output(
                ["git", "-C", packs[0]["repository"], "rev-parse", "HEAD"], text=True
            ).strip(),
            "status": subprocess.check_output(
                ["git", "-C", packs[0]["repository"], "status", "--porcelain"], text=True
            ),
        },
    )
    save(
        args.out / "spend-summary.json",
        {
            "total_usd": str(ledger.spent),
            "remaining_usd": str(ledger.cap - ledger.spent - sum(ledger.reserved.values())),
            "reserved_usd": str(sum(ledger.reserved.values())),
            "halted": ledger.halted,
            "resources": check_resources(),
        },
    )


if __name__ == "__main__":
    main()
