"""Zero-provider Case 2 development study. No planner or judge connector exists here.

Run with: uv run --with ijson --with tiktoken python examples/pack_case2/replay.py OUT
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import ijson
import msgspec
import tiktoken
from planner import PlannerContract, PlannerInput
from receipts import merge_candidates
from receipts import write_json as write
from trace_translate import repair

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import items_to_judge, read_ranges
from jev_navigator.plan_outline import outline_lines
from jev_navigator.search_plan import Approach, SearchPlan, TermProvenance, decode_plan, execute_plan

BASE = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"
TAKEOVER = Path.home() / ".local/share/jvn-takeover/2026-10-03"
TUNING = {*(f"P{i}" for i in range(1, 8)), *(f"U{i}" for i in range(1, 7))}
TOKENS = tiktoken.get_encoding("cl100k_base")
WORD = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
SYNTAX = set(
    [
        "def",
        "class",
        "from",
        "import",
        "return",
        "if",
        "else",
        "and",
        "or",
        "not",
        "true",
        "false",
        "null",
        "int",
        "str",
        "bool",
        "self",
        "os",
        "py",
        "md",
        "ts",
        "js",
        "json",
        "yml",
        "yaml",
        "toml",
        "conf",
        "github",
        "workflows",
    ]
)


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("This development replay permits zero provider calls")


sys.addaudithook(deny_network)


def resource_check():
    disk = shutil.disk_usage(BASE).free
    vm = subprocess.check_output(["vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+)", vm)[1])
    counts = {name: int(number) for name, number in re.findall(r"(Pages [^:]+):\s+(\d+)", vm)}
    available = sum(counts.get(f"Pages {name}", 0) for name in ("free", "inactive", "speculative")) * page
    if disk < 30 * 10**9 or available < 8 * 10**9:
        raise RuntimeError(f"Resource stop: disk={disk}, available memory={available}")
    return {
        "disk_free_bytes": disk,
        "available_memory_bytes": available,
        "memory_measure": "vm_stat free + inactive + speculative pages",
    }


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases(dataset):
    with (BASE / f"cases/{'replay-dev110' if dataset == 'dev110' else 'hard27'}.json").open("rb") as stream:
        return [
            case for case in ijson.items(stream, "item") if dataset == "dev110" or case["case_id"] in TUNING
        ]


def inputs(dataset):
    with (BASE / f"runs/pack-49b78955/pack-inputs-{dataset}.json").open("rb") as stream:
        return {
            cid: row for cid, row in ijson.kvitems(stream, "cases") if dataset == "dev110" or cid in TUNING
        }


def words(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return {word.casefold() for word in WORD.findall(text)} | {
        part.casefold() for word in WORD.findall(text) for part in word.split("_") if part
    }


def scoped_files(index, claim):
    cited = {entry["file"] for entry in claim.get("evidence", ()) if entry.get("file") in index.files}
    directories = {str(Path(file).parent) for file in cited}
    vocabulary = words(claim["statement"])
    return tuple(
        file
        for file in index.files
        if file in cited or str(Path(file).parent) in directories or words(Path(file).stem) & vocabulary
    )


def token_count(text):
    return len(TOKENS.encode(text, disallowed_special=()))


def measure_outline(index, out, key):
    saved_map = out / f"outline-{key}.txt"
    saved_record = out / f"outline-{key}.json"
    if saved_map.exists() and saved_record.exists():
        record = json.loads(saved_record.read_text())
        if record["commit"] == index.commit:
            files = {}
            current = None
            chunks = []
            for line in saved_map.open():
                if not line.startswith("  "):
                    if current is not None:
                        files[current] = "".join(chunks)
                    current, chunks = line.strip(), []
                chunks.append(line)
            if current is not None:
                files[current] = "".join(chunks)
            if set(files) == set(index.files):
                return files, record
    directories = defaultdict(lambda: {"files": 0, "tokens": 0, "characters": 0})
    files = {}
    current = None
    chunks = []
    started = time.monotonic()
    with (out / f"outline-{key}.txt").open("w") as saved:
        for line in outline_lines(index):
            saved.write(line)
            if not line.startswith("  "):
                if current is not None:
                    files[current] = "".join(chunks)
                current, chunks = line.strip(), []
            chunks.append(line)
        if current is not None:
            files[current] = "".join(chunks)
    # Retain compact map facts, never parser output or source bodies. Per-file
    # token sums are an explicitly labelled boundary-count proxy.
    for file, text in files.items():
        row = directories[str(Path(file).parent)]
        row["files"] += 1
        row["tokens"] += token_count(text)
        row["characters"] += len(text)
    record = {
        "files": len(files),
        "tokens": sum(r["tokens"] for r in directories.values()),
        "characters": sum(r["characters"] for r in directories.values()),
        "directories": dict(directories),
        "seconds": time.monotonic() - started,
        "commit": index.commit,
        "tokenizer": "cl100k_base per-file sum proxy, not DeepSeek tokens",
    }
    write(out / f"outline-{key}.json", record)
    return files, record


def cited_code(index, claim):
    result = []
    for entry in claim.get("evidence", ()):
        file = entry.get("file")
        if file not in index.files:
            continue
        lines = index.lines(file)
        mentioned = re.findall(re.escape(Path(file).name) + r":(\d+)(?:-(\d+))?", claim["statement"])
        extents = [(max(1, int(a) - 10), min(len(lines), int(b or a) + 10)) for a, b in mentioned]
        extents = extents or [(1, min(len(lines), 40))]
        for start, end in extents:
            if start <= end:
                result.append(
                    {"file": file, "start": start, "end": end, "code": "\n".join(lines[start - 1 : end])}
                )
    return tuple(result)


def context_for(index, maps, claim):
    scope = scoped_files(index, claim)
    full_map = "".join(maps[file] for file in scope)
    code = cited_code(index, claim)
    # The context includes an explicit pending page when the compact scope does
    # not fit. Keep cited paths first; omitted map facts are not available to round 1.
    ordered = sorted(scope, key=lambda file: (file not in {r["file"] for r in code}, file))
    kept = []
    used = 0
    base_context = PlannerInput(
        claim["statement"],
        code,
        "",
        (),
        {"roles": "unknown", "candidates": "unjudged", "outline_files_pending": len(scope)},
    )
    outline_room = max(0, min(6000, 12000 - token_count(PlannerContract().render(base_context)) - 100))
    for file in ordered:
        size = token_count(maps[file])
        if used + size > outline_room:
            continue
        kept.append(file)
        used += size
    context = PlannerInput(
        claim["statement"],
        code,
        "".join(maps[file] for file in kept),
        (),
        {"roles": "unknown", "candidates": "unjudged", "outline_files_pending": len(scope) - len(kept)},
    )
    return context, {
        "scope_files": len(scope),
        "scope_tokens": token_count(full_map),
        "outline_files_supplied": len(kept),
        "outline_tokens_supplied": used,
        "prompt_tokens_proxy": token_count(PlannerContract().render(context)),
    }


def normalize(path, index, recorded_root):
    for root in (str(index.root), recorded_root):
        if root and path.startswith(root.rstrip("/") + "/"):
            return path[len(root.rstrip("/")) + 1 :]
    return path.removeprefix("./")


def translate(proposal, index, recorded_root, command):
    args = dict(proposal["arguments"])
    if "path" in args:
        args["path"] = normalize(args["path"], index, recorded_root)
    if "scopes" in args:
        args["scopes"] = [normalize(scope, index, recorded_root) for scope in args["scopes"]]
    if proposal["operation"] == "find_text":
        args["pattern_kind"] = (
            "literal" if "--fixed-strings" in command or re.search(r"\s-[a-zA-Z]*F", command) else "regex"
        )
        args["ignore_case"] = bool("--ignore-case" in command or re.search(r"\s-[a-zA-Z]*i", command))
    record = {
        "approaches": [
            {
                "rank": proposal["rank"],
                "call": {"operation": proposal["operation"], **args},
                "provenance": [
                    {"term": str(value), "source": f"recorded command step {proposal['command_step']}"}
                    for value in args.values()
                    if value
                ],
                "reason": "recorded chronological approach",
            }
        ]
    }
    return decode_plan(json.dumps(record)).approaches[0]


def argument_terms(call):
    raw = msgspec.to_builtins(call)
    for key, value in raw.items():
        if key in {"operation", "pattern_kind", "ignore_case"}:
            continue
        for term in value if isinstance(value, list) else [value]:
            if isinstance(term, str) and term not in {"", ".", "./"}:
                yield term


def admit(approach, available, known_paths, witness):
    provenance = []
    missing = []
    for term in argument_terms(approach.call):
        if term in known_paths:
            provenance.append(TermProvenance(term, known_paths[term]))
        elif term == getattr(approach.call, "path", None) or term == getattr(approach.call, "owner", None):
            missing.append({"term": term, "unknown_components": ["exact path not yet observed"]})
        elif (needed := words(re.sub(r"\\[bBdDsSwWZAztnr]", "", term)) - SYNTAX) <= available:
            provenance.append(TermProvenance(term, witness, "case/separator/regex syntax"))
        else:
            missing.append({"term": term, "unknown_components": sorted(needed - available)})
    return (Approach(approach.rank, approach.call, tuple(provenance), approach.reason), missing)


def run_arm(index, approaches, context, strict):
    available = words(context.finding + context.outline + json.dumps(context.cited_code))
    known_paths = {
        line: "supplied outline" for line in context.outline.splitlines() if not line.startswith("  ")
    }
    known_paths.update({entry["file"]: "cited code" for entry in context.cited_code})
    remaining = list(approaches)
    union = {}
    rounds = []
    blocked = []
    started = time.monotonic()
    for number in range(1, 4 if strict else 2):
        admitted = []
        pending = []
        blocked = []
        for approach in remaining:
            checked, missing = admit(
                approach,
                available,
                known_paths,
                "supplied context" if number == 1 else "completed prior-round candidates",
            )
            if strict and missing:
                pending.append(approach)
                blocked.append({"rank": approach.rank, "missing": missing})
            else:
                admitted.append(checked if strict else approach)
        result = execute_plan(index, SearchPlan(tuple(admitted)), box_chars=70000)
        merge_candidates(union, result)
        rounds.append(
            {
                "round": number,
                "admitted": [a.rank for a in admitted],
                "blocked": blocked,
                "candidates": len(result.candidates),
                "outcomes": [msgspec.to_builtins(o) for o in result.outcomes],
            }
        )
        if not pending or not admitted:
            break
        for candidate in result.candidates:
            unit = candidate.unit
            available |= words(read_ranges(index, unit.path, unit.ranges))
            known_paths[unit.path] = f"completed round {number}"
        remaining = pending
    return tuple(union.values()), {
        "rounds": rounds,
        "blocked": blocked,
        "seconds": time.monotonic() - started,
    }


def covers(candidates, label):
    return any(
        c.unit.path == label["file"] and any(a <= label["first_line"] <= b for a, b in c.unit.ranges)
        for c in candidates
    )


def main():
    out = Path(sys.argv[1]).expanduser()
    corrected_extraction = "--correct-extraction" in sys.argv[2:]
    out.mkdir(parents=True, exist_ok=True)
    start_resources = resource_check()
    plans = {}
    commands = {}
    for name, target in (("planning-approaches", plans), ("commands", commands)):
        with (TAKEOVER / f"discovery/{name}.json").open("rb") as stream:
            for cid, value in ijson.kvitems(stream, ""):
                target[cid] = value[:10] if isinstance(value, list) else value
    summaries = []
    outline_records = {}
    # Each repository is released before another is indexed.
    for dataset in ("dev110", "hard27"):
        workload = inputs(dataset)
        selected = cases(dataset)
        roots = list(dict.fromkeys(workload[c["case_id"]]["repository"] for c in selected))
        for root in roots:
            members = [c for c in selected if workload[c["case_id"]]["repository"] == root]
            first = workload[members[0]["case_id"]]
            key = "dev110" if dataset == "dev110" else Path(root).name
            withheld = set(first["withheld"])
            with CodeIndex.from_git(Path(root)) as index:
                admissible = tuple(file for file in index.files if file not in withheld)
                index = CodeIndex(Path(root), admissible, commit=index.commit)
                maps, record = measure_outline(index, out, key)
                outline_records[key] = record
                for case in members:
                    resource_check()
                    cid = case["case_id"]
                    row = workload[cid]
                    context, context_sizes = context_for(index, maps, row["claim"])
                    prefix = out / "cases" / cid.replace(":", "_")
                    prefix.mkdir(parents=True, exist_ok=True)
                    write(prefix / "planner-context.json", asdict(context))
                    (prefix / "planner-prompt.txt").write_text(PlannerContract().render(context))
                    source_commands = commands.get(cid, {}).get("commands", [])
                    command_by_step = {r["step"]: r["command"] for r in source_commands}
                    recorded_root = next(
                        (
                            r["output"].splitlines()[0]
                            for r in source_commands
                            if r["command"].startswith("pwd") and r["output"].startswith("/")
                        ),
                        "",
                    )
                    translated = []
                    errors = []
                    for proposal in plans.get(cid, []):
                        try:
                            command = command_by_step.get(proposal["command_step"], "")
                            if corrected_extraction:
                                proposal, correction = repair(proposal, command)
                                if correction:
                                    errors.append({"rank": proposal["rank"], "extraction_note": correction})
                            translated.append(
                                translate(
                                    proposal,
                                    index,
                                    recorded_root,
                                    command,
                                )
                            )
                        except (ValueError, TypeError) as error:
                            errors.append({"rank": proposal["rank"], "error": str(error)})
                    for strict in (False, True):
                        arm = "strict" if strict else "retrospective"
                        candidates, run = run_arm(index, translated, context, strict)
                        with (prefix / f"{arm}-candidates.jsonl").open("w") as saved:
                            for candidate in candidates:
                                unit = candidate.unit
                                record = {
                                    "unit": asdict(unit),
                                    "approach_ranks": [a.rank for a in candidate.approaches],
                                    "items": [
                                        {**asdict(item), "code": read_ranges(index, item.file, item.ranges)}
                                        for item in items_to_judge(unit)
                                    ],
                                }
                                saved.write(json.dumps(record) + "\n")
                        labels = [{**label, "reached": covers(candidates, label)} for label in case["labels"]]
                        record = {
                            "case": cid,
                            "dataset": dataset,
                            "arm": arm,
                            "surrogate_available": cid in plans,
                            "candidates": len(candidates),
                            "items": sum(len(items_to_judge(c.unit)) for c in candidates),
                            "labels": labels,
                            "context": context_sizes,
                            "translation_errors": errors,
                            "seconds": run["seconds"],
                            "blocked": run["blocked"],
                            "provider_calls": 0,
                            "usd": 0,
                        }
                        write(prefix / f"{arm}-execution.json", {**record, "rounds": run["rounds"]})
                        summaries.append(record)
                    write(out / "replay-checkpoint.json", summaries)
                    print(
                        cid,
                        "candidates",
                        summaries[-2]["candidates"],
                        summaries[-1]["candidates"],
                        flush=True,
                    )
    write(
        out / "schema.json", __import__("jev_navigator.search_plan", fromlist=["plan_schema"]).plan_schema()
    )
    write(
        out / "replay-summary.json",
        {
            "provider_calls": 0,
            "usd": 0,
            "cases": summaries,
            "outlines": outline_records,
            "start_resources": start_resources,
            "end_resources": resource_check(),
        },
    )
    print("Development replay complete", len(summaries), flush=True)


if __name__ == "__main__":
    main()
