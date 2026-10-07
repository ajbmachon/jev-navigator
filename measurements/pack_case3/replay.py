"""Development replay of recorded bash approaches through the public batch surface, without a provider.

This is retrospective trace compression, not a measured cheap-agent policy. Future recorded inputs
are available to the compression arm. A conservative arm starts another round for inputs not yet
visible. No command from a receipt is ever executed. Labels only score literal emitted source rows.

Run: uv run --with tiktoken python measurements/pack_case3/replay.py --out /path/to/case3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from jev_navigator.batch import Operation, run_batch
from jev_navigator.index.code_index import CodeIndex

TAKEOVER = Path.home() / ".local/share/jvn-takeover/2026-10-03"
EVAL = Path.home() / ".local/share/system-one-proof/jvn-eval-2026-10-03"


def load(path):
    return json.loads(path.read_text())


def check_resources():
    free_disk = __import__("shutil").disk_usage(Path.cwd()).free
    vm = subprocess.check_output(["vm_stat"], text=True)
    counts = {key: int(value) for key, value in re.findall(r"([\w ]+):\s+(\d+)\.", vm)}
    available = sum(counts.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative"))
    free_memory = available * 16384
    if free_disk < 30_000_000_000 or free_memory < 8_000_000_000:
        raise RuntimeError(f"resource stop: disk {free_disk}, available memory {free_memory}")
    return {"free_disk_bytes": free_disk, "available_memory_bytes": free_memory}


def discovery_parser():
    spec = importlib.util.spec_from_file_location("discovery_parser", TAKEOVER / "discovery/analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shell_pipelines(command, parser):
    """Separate unquoted newlines, preserving quoted regex pipes and sed range lists."""
    lexer = shlex.shlex(parser.shell(command), posix=True, punctuation_chars=";&|\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    result, pipeline, words = [], [], []
    for token in lexer:
        if token and all(character in ";&|\n" for character in token):
            if words:
                pipeline.append(words)
                words = []
            if token.strip("\n") != "|" and pipeline:
                result.append(pipeline)
                pipeline = []
        else:
            words.append(token)
    if words:
        pipeline.append(words)
    if pipeline:
        result.append(pipeline)
    return result


@dataclass(frozen=True)
class Move:
    operation: Operation
    step: int
    started: int
    completed: int
    command: str


def path_in(token, files, receipt_root):
    if token.startswith("/"):
        try:
            token = str(Path(token).relative_to(receipt_root))
        except ValueError:
            return None
    token = token.removeprefix("./")
    if basename(token) in ("AGENTS.md", "CLAUDE.md"):
        return None
    return token if token in files else None


def basename(token):
    return token.rpartition("/")[2]


def option(words, short, long=None, default=0):
    for i, word in enumerate(words):
        if word in (short, long) and i + 1 < len(words):
            try:
                return int(words[i + 1])
            except ValueError:
                return default
        if word.startswith(short) and word[len(short) :].isdigit():
            return int(word[len(short) :])
    return default


def text_arguments(words, parser):
    value_options = {
        "-g",
        "--glob",
        "-e",
        "-t",
        "-T",
        "-m",
        "-A",
        "-B",
        "-C",
        "--type",
        "--max-count",
        "--max-columns",
        "--max-columns-preview",
    }
    patterns = [words[i + 1] for i, word in enumerate(words[:-1]) if word in ("-g", "--glob")]
    positionals = []
    skip = False
    for word in words[1:]:
        if skip:
            skip = False
        elif word in value_options:
            skip = True
        elif not word.startswith("-"):
            positionals.append(word)
    if "--files" in words:
        return {"patterns": tuple(patterns), "scopes": tuple(positionals)}
    pattern = parser.pattern(words, "")
    regex = not any(word in ("-F", "--fixed-strings") for word in words)
    if any(word in ("-i", "--ignore-case", "-ni", "-in") for word in words):
        pattern = "(?i)" + (pattern if regex else re.escape(pattern))
        regex = True
    scopes = positionals[1:] if "-e" not in words else positionals
    scopes = [scope for scope in scopes if scope != "|" and not scope.isdigit()]
    context = max(
        option(words, "-C", "--context"),
        option(words, "-A", "--after-context"),
        option(words, "-B", "--before-context"),
    )
    return {
        "query": pattern,
        "regex": regex,
        "scopes": tuple(scopes),
        "patterns": tuple(patterns),
        "window": context,
    }


def translate(record, index, parser):
    first = record["commands"][0]["output"].splitlines()
    receipt_root = Path(first[0]) if first and first[0].startswith("/") else index.root
    moves, rejected = [], []
    seen = set()
    files = frozenset(index.files)
    for command in sorted(record["commands"], key=lambda c: c.get("started_line", c["receipt_line"])):
        admitted = False
        try:
            pipelines = shell_pipelines(command["command"], parser)
        except ValueError:
            rejected.append({"step": command["step"], "reason": "shell syntax could not be translated"})
            continue
        for pipeline in pipelines:
            words = pipeline[0]
            ops = []
            if words[0] in ("rg", "grep"):
                args = text_arguments(words, parser)
                if args.get("query"):
                    ops = [Operation("refs", **args, limit=400)]
                elif "--files" in words:
                    args["patterns"] = tuple(
                        p for p in args["patterns"] if p not in ("AGENTS.md", "CLAUDE.md")
                    )
                    ops = [Operation("named_files", **args, limit=400)]
            elif words[0] in ("nl", "sed", "cat", "head", "tail"):
                paths = [path_in(word, files, receipt_root) for word in words[1:]]
                paths = [path for path in paths if path]
                intervals = [
                    (int(match[1]), int(match[2] or match[1]))
                    for argv in pipeline
                    for word in argv
                    for match in re.finditer(r"(?:^|;)(\d+)(?:,(\d+))?p(?=;|$)", word)
                ]
                for path in paths:
                    count = len(index.lines(path))
                    if not count:
                        continue
                    for start, end in intervals or [(1, count)]:
                        end = min(end, count)
                        if words[0] == "head":
                            end = min(end, option(words, "-n", "--lines", 10))
                        elif words[0] == "tail":
                            start = max(1, count - option(words, "-n", "--lines", 10) + 1)
                        if end >= start:
                            ops.append(Operation("show", file=path, line=start, end=end, limit=400))
            for op in ops:
                admitted = True
                identity = json.dumps(asdict(op), sort_keys=True)
                if identity in seen:
                    continue
                seen.add(identity)
                moves.append(
                    Move(
                        op,
                        command["step"],
                        command.get("started_line", command["receipt_line"]),
                        command["receipt_line"],
                        command["command"],
                    )
                )
        if not admitted:
            rejected.append(
                {"step": command["step"], "reason": "instruction, excluded file or unsupported shell"}
            )
    return moves, rejected


def ready(move, known, first_completion):
    if move.started < first_completion:
        return True
    op = move.operation
    if op.file:
        return op.file in known or basename(op.file) in known
    terms = [
        word
        for word in re.findall(r"[\w$]+", op.query or " ".join(op.patterns))
        if len(word) > 3 and word not in ("from", "import", "return", "function")
    ]
    return all(word in known for word in terms)


def source_rows(payload, fragments, pending):
    for page, op in zip(payload["pages"], pending, strict=True):
        for item in page["items"]:
            if "row_json" in item:
                identity = asdict(op)
                identity.pop("cursor")
                key = json.dumps(identity, sort_keys=True), item["number"]
                fragments[key] += item["row_json"]
                if len(fragments[key]) != item["row_json_chars"]:
                    continue
                item = json.loads(fragments.pop(key))
            if "line" in item and "text" in item:
                for offset, text in enumerate(item["text"].split("\n")):
                    yield {"file": item["file"], "line": item["line"] + offset, "text": text}


def replay(index, moves, initial, conservative, capacity, out, encoding):
    queue = list(moves)
    known = initial
    calls, tokens, chars, errors = 0, 0, 0, []
    shown = {}
    fragments = defaultdict(str)
    batches = []
    start = time.monotonic()
    while queue:
        group = [queue.pop(0)]
        while queue and len(group) < 10:
            if conservative and not ready(queue[0], known, min(m.completed for m in group)):
                break
            group.append(queue.pop(0))
        pending = [move.operation for move in group]
        batches.append([{"step": move.step, "operation": asdict(move.operation)} for move in group])
        # Every page is a real library call. No hidden free follow-up to inflate the compression.
        while pending:
            response = run_batch(index, pending, max_chars=capacity)
            text = response.render()
            calls += 1
            chars += len(text)
            tokens += len(encoding.encode(text, disallowed_special=()))
            known += "\n" + text
            payload = response.to_dict()
            for row in source_rows(payload, fragments, pending):
                shown.setdefault((row["file"], row["line"]), calls)
            errors.extend(
                {"operation": page.operation, "error": page.error, "items": page.items}
                for page in response.pages
                if page.error
            )
            out.write(
                json.dumps(
                    {"call": calls, "operation_steps": [m.step for m in group], "response": payload},
                    ensure_ascii=False,
                )
                + "\n"
            )
            pending = [
                replace(op, cursor=page.next)
                for op, page in zip(pending, response.pages, strict=True)
                if page.next is not None
            ]
    return {
        "calls": calls,
        "tokens_cl100k": tokens,
        "chars": chars,
        "wall_seconds": time.monotonic() - start,
        "errors": errors,
        "batches": batches,
    }, shown


def emitted_labels(case, shown):
    return [
        {
            "file": label["file"],
            "line": line,
            "shown": (label["file"], line) in shown,
            "first_call": shown.get((label["file"], line)),
        }
        for label in case["labels"]
        if label.get("role") == "deciding"
        for line in range(label["first_line"], label["last_line"] + 1)
    ]


def statistics_of(rows, field):
    values = sorted(row[field] for row in rows)
    return {
        "median": statistics.median(values),
        "p90": values[math.ceil(len(values) * 0.9) - 1],
        "total": sum(values),
    }


def main():
    import tiktoken

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=64_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [
        EVAL / "cases/replay-dev110.json",
        EVAL / "runs/pack-49b78955/pack-inputs-dev110.json",
        TAKEOVER / "discovery/commands.json",
        TAKEOVER / "discovery/lines.jsonl",
        TAKEOVER / "discovery/planning-approaches.json",
        TAKEOVER / "discovery/term-proposals.jsonl",
        TAKEOVER / "unlock/manifest.json",
        TAKEOVER / "unlock/final-summary.json",
        Path(__file__),
    ]
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("freeze and commit the replay code before measuring")
    resources = check_resources()
    cases = load(EVAL / "cases/replay-dev110.json")
    packs = load(EVAL / "runs/pack-49b78955/pack-inputs-dev110.json")["cases"]
    commands = load(TAKEOVER / "discovery/commands.json")
    shell_parser = discovery_parser()
    encoding = tiktoken.get_encoding("cl100k_base")

    # A socket audit guards only the evaluation stage after the free tokenizer asset is loaded.
    def offline(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.bind"):
            raise RuntimeError(f"offline replay forbids {event}")

    sys.addaudithook(offline)
    results = []
    for case in cases:
        resources = check_resources()
        cid = case["case_id"]
        if cid not in commands:
            results.append(
                {
                    "case": cid,
                    "agent_calls": None,
                    "registered": 0,
                    "status": "no recorded discovery trace; workload retained",
                }
            )
            continue
        pack = packs[cid]
        root = Path(pack["repository"])
        with CodeIndex.from_git(root) as tracked:
            files = [
                file
                for file in tracked.files
                if file not in pack["withheld"] and basename(file) not in ("AGENTS.md", "CLAUDE.md")
            ]
            with CodeIndex(root, files, commit=tracked.commit) as index:
                moves, rejected = translate(commands[cid], index, shell_parser)
                row = {
                    "case": cid,
                    "agent_calls": len(commands[cid]["commands"]),
                    "agent_tokens_cl100k": sum(
                        len(encoding.encode(c["output"], disallowed_special=()))
                        for c in commands[cid]["commands"]
                    ),
                    "receipt": commands[cid]["receipt"],
                    "untranslated": rejected,
                    "translated_operations": len(moves),
                }
                initial = case["target"] + "\n" + json.dumps(pack["claim"])
                for arm in ("compression", "conservative"):
                    with (args.out / f"{cid.replace(':', '_')}-{arm}.jsonl").open("w") as out:
                        measured, shown = replay(
                            index, moves, initial, arm == "conservative", args.capacity, out, encoding
                        )
                    labels = emitted_labels(case, shown)
                    row[arm] = {
                        **measured,
                        "labels": labels,
                        "delivered": sum(label["shown"] for label in labels),
                        "delivered_first_five": sum(
                            label["shown"] and label["first_call"] <= 5 for label in labels
                        ),
                        "all_labels_call": (
                            max(label["first_call"] for label in labels)
                            if labels and all(label["shown"] for label in labels)
                            else None
                        ),
                    }
                row["registered"] = len(row["compression"]["labels"])
                results.append(row)
        print(
            cid,
            row["compression"]["calls"],
            row["conservative"]["calls"],
            row["compression"]["delivered"],
            "/",
            row["registered"],
            flush=True,
        )
        (args.out / "cases.json").write_text(json.dumps(results, indent=2) + "\n")
    traced = [row for row in results if "compression" in row]
    summary = {
        "provider_calls": 0,
        "spend_usd": 0,
        "capacity_chars_per_call": args.capacity,
        "findings": len(results),
        "traced_findings": len(traced),
        "registered_lines": sum(row["registered"] for row in results),
        "resources": resources,
        "tokenizer": "cl100k_base reference tokenizer, not the provider's DeepSeek tokenizer",
        "measurement": "development retrospective trace compression; no new agent decisions",
    }
    for arm in ("compression", "conservative"):
        summary[arm] = {
            "delivered": sum(row[arm]["delivered"] for row in traced),
            "within_five_calls": sum(row[arm]["calls"] <= 5 for row in traced),
            "delivered_first_five": sum(row[arm]["delivered_first_five"] for row in traced),
            "all_labels_first_five": sum(
                row[arm]["all_labels_call"] is not None and row[arm]["all_labels_call"] <= 5 for row in traced
            ),
            "statistics": {
                field: statistics_of([row[arm] for row in traced], field)
                for field in ("calls", "tokens_cl100k", "chars", "wall_seconds")
            },
        }
    summary["input_sha256"] = input_hashes
    summary["jvn_head"] = head
    summary["source_unchanged"] = all(
        hashlib.sha256(path.read_bytes()).hexdigest() == input_hashes[str(path)] for path in paths
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
