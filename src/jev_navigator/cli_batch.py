"""JSON transport for the batched library entry. Facts need no credentials."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters.routes import system_one_client
from .batch import Operation, run_batch
from .environment import load_typesafe_environment
from .index.code_index import CodeIndex
from .judgments.client import ReplayOnlyClient
from .judgments.judge import Judge
from .judgments.store import SqliteAnswerStore, shared_store_path


def add_batch_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("batch", help="run several explicit, pageable code operations in one call")
    parser.add_argument("request", help="JSON object, request file or - for stdin")
    parser.add_argument("--repo", default=".", help="Source directory, Git optional")
    parser.add_argument("--max-chars", type=int, default=24_000, help="Complete response size in characters")
    parser.add_argument(
        "--max-calls", type=int, default=0, help="Explicit live rank request allowance (default 0)"
    )
    parser.add_argument(
        "--replay", action="store_true", help="Rank only from exact stored answers, zero live calls"
    )


def run_batch_command(args: argparse.Namespace) -> int:
    client = None
    try:
        if hasattr(args, "payload"):
            payload = args.payload
        elif args.request.lstrip().startswith("{"):
            payload = json.loads(args.request)
        elif args.request == "-":
            payload = json.load(sys.stdin)
        else:
            payload = json.loads(Path(args.request).expanduser().read_text())
        unknown = set(payload) - {"command", "operations", "repo", "max_chars", "max_calls", "replay"}
        if unknown:
            raise ValueError(f"unknown batch fields: {sorted(unknown)}")
        if not isinstance(payload["operations"], list):
            raise ValueError("operations must be an array")
        operations = tuple(Operation.from_dict(row) for row in payload["operations"])
        max_calls = payload.get("max_calls", args.max_calls)
        if type(max_calls) is not int or max_calls < 0:
            raise ValueError("max_calls must be non-negative")
        judge = None
        replay = payload.get("replay", args.replay)
        if type(replay) is not bool:
            raise ValueError("replay must be a boolean")
        if (max_calls or replay) and any(op.op == "rank" for op in operations):
            if replay:
                client = ReplayOnlyClient()
            else:
                load_typesafe_environment(os.environ)
                client = system_one_client(os.environ)
            judge = Judge(
                client,
                max_calls=max_calls,
                items_per_request=16,
                store=SqliteAnswerStore(shared_store_path()),
            )
        with CodeIndex.from_directory(Path(payload.get("repo", args.repo)).expanduser().resolve()) as index:
            result = run_batch(
                index, operations, judge=judge, max_chars=payload.get("max_chars", args.max_chars)
            )
            print(result.render())
        return 1 if any(page.error for page in result.pages) else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"jvn batch: {error}", file=sys.stderr)
        return 2
    finally:
        if client is not None and hasattr(client, "close"):
            client.close()
