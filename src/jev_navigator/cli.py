"""Command-line evidence pack for one live ``find_code`` search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from .adapters.typesafe import TypeSafeJevClient
from .directives.entry import EntrySelection, choose_initial_candidates
from .directives.find_code import FindResult, SearchBudget, Visit, find_code
from .directives.places import Place, place_for_line
from .index.code_index import CodeIndex
from .judgments.client import JevClient
from .judgments.judge import Judge
from .judgments.store import JsonlAnswerStore
from .judgments.thresholds import Thresholds
from .progress import ProgressJournal, TerminalProgress

SCHEMA_VERSION = "jev-navigator.evidence-pack/v1"
NON_NEGATIVE_BUDGET_FIELDS = ("max_depth", "max_steps", "max_calls", "neighbours_per_kind", "preview_lines")
POSITIVE_BUDGET_FIELDS = ("beam_width", "max_slice_chars", "max_line_chars")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "schema":
        print(json.dumps(_request_schema(_command_parser(_parser(), args.topic)), indent=2))
        return 0
    if args.command == "help":
        parser = _parser()
        (_command_parser(parser, args.topic) if args.topic else parser).print_help()
        return 0
    if args.command != "find":
        raise AssertionError(f"unhandled command: {args.command}")
    budget = SearchBudget(
        max_depth=args.max_depth,
        max_steps=args.max_steps,
        max_calls=args.max_calls,
        beam_width=args.beam_width,
        neighbours_per_kind=args.neighbours_per_kind,
        preview_lines=args.preview_lines,
        max_slice_chars=args.max_slice_chars,
        max_line_chars=args.max_line_chars,
    )
    try:
        _validate_budget(budget)
    except ValueError as error:
        _parser().error(str(error))
    repository = Path(args.repo).resolve()
    output = Path(args.out).expanduser() if args.out else _default_output(repository)
    client: TypeSafeJevClient | None = None
    try:
        _load_typesafe_environment(os.environ)
        client = TypeSafeJevClient()  # model=None resolves TYPESAFE_DEFAULT_MODEL in the adapter
        manifest = create_evidence_pack(
            repository,
            tuple(args.prefix),
            args.target,
            tuple(args.start),
            output,
            budget,
            client,
            thresholds=Thresholds.from_env(),
            verbose=args.verbose,
        )
    except Exception as error:
        print(f"jvn find: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("jvn find: cancelled", file=sys.stderr)
        return 130
    finally:
        if client is not None:
            client.close()
    search_outcome = manifest["search"]["outcome"]
    if args.json:
        print(
            json.dumps(
                {
                    "output_directory": str(output.resolve()),
                    "manifest": str(output.resolve() / "manifest.json"),
                    "report": str(output.resolve() / "report.md"),
                    "search": manifest["search"],
                    "provider": manifest["provider"],
                }
            )
        )
    else:
        print(f"evidence pack: {output.resolve()}")
        print(f"outcome: {search_outcome} ({manifest['search']['calls']} live calls)")
    return 130 if search_outcome == "cancelled" else 0


def create_evidence_pack(
    repository: Path,
    prefixes: tuple[str, ...],
    target: str,
    starts: tuple[str, ...],
    output: Path,
    budget: SearchBudget,
    client: JevClient,
    *,
    thresholds: Thresholds | None = None,
    verbose: bool = False,
    fact_cache_dir: Path | None = None,
) -> dict:
    """Run the real index/search owners and persist their reviewable evidence."""
    repository = repository.resolve()
    output = output.resolve()
    _validate_budget(budget)
    _prepare_output(output)
    journal_path = output / "journal.jsonl"
    journal_path.touch()
    progress = TerminalProgress(journal_path, verbose=verbose)
    journal = ProgressJournal(journal_path, progress)
    progress.start()
    outcome = "failed"
    try:
        progress.phase("indexing files")
        index = CodeIndex.from_directory(
            repository,
            prefixes=prefixes,
            exclude_paths=(output, Path.cwd() / "jvn-results"),
            scan_observer=progress.scan,
            fact_cache_dir=fact_cache_dir,
        )
        if warning := _scope_warning(len(index.files)):
            print(warning, file=sys.stderr)
        thresholds = thresholds or Thresholds()
        judge = Judge(
            client,
            thresholds=thresholds,
            max_calls=budget.max_calls,
            journal=journal,
            store=JsonlAnswerStore(output / "answers.jsonl"),
        )
        selection: EntrySelection | None = None
        if starts:
            start_places = [_parse_start(index, start) for start in starts]
            initial_candidates: tuple[tuple[Place, float], ...] = ()
        else:
            progress.phase("choosing an entry point")
            start_places = []
            selection = choose_initial_candidates(index, judge, target)
            initial_candidates = tuple(
                (
                    candidate.place,
                    candidate.selection_probability if candidate.selection_probability is not None else 0.0,
                )
                for candidate in selection.candidates
            )
        progress.phase("navigating code")
        started = monotonic()
        result = find_code(
            index,
            judge,
            target,
            start_places,
            budget=budget,
            commit=None,
            initial_candidates=initial_candidates,
        )
        duration_seconds = monotonic() - started
        progress.phase("writing evidence pack")
        manifest = _manifest(
            repository,
            prefixes,
            target,
            starts,
            budget,
            thresholds,
            index,
            result,
            requested_model=getattr(client, "model", "unknown"),
            served_model=judge.served_model,
            input_tokens=judge.input_tokens,
            duration_seconds=duration_seconds,
            total_calls=judge.calls,
            entry_selection=selection,
        )
        _write_json(output / "manifest.json", manifest)
        (output / "report.md").write_text(_report(manifest))
        outcome = str(result.outcome)
        return manifest
    except KeyboardInterrupt:
        outcome = "cancelled"
        raise
    finally:
        journal.record_terminal(outcome)
        progress.close(outcome)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jvn",
        description="Find code by behavior. Code follows relationships; Jev judges concrete evidence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  jvn find "where are evidence quotes rejected?"
  jvn find "where is an order's item count checked?" --repo /path/to/repository
  jvn --json request.json
  cat request.json | jvn --json -
  jvn --json '{"target":"where are evidence quotes rejected?"}'

For agents: jvn schema find prints the request's JSON Schema without making model calls.
JSON mode writes results to stdout; progress goes to stderr. Ctrl-C cancels.
Results default to ./jvn-results/<directory>-<timestamp> in the invocation directory.
Credentials: process environment, then ~/.config/jvn/env (TYPESAFE_API_KEY / TYPESAFE_BASE_URL).
Use jvn help find for options and examples. Exit codes: 0 completed, 1 failed, 2 invalid input, 130 cancelled.
A completed search can have a non-found outcome; inspect search.outcome in JSON output.""",
    )
    parser.add_argument(
        "--json", metavar="REQUEST", help="JSON object, request file, or - for stdin; emit JSON"
    )
    commands = parser.add_subparsers(dest="command")
    find = commands.add_parser(
        "find",
        help="find semantically described code and write a versioned evidence pack",
        description="Find semantically described code and save a reviewable evidence pack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  jvn find "the check that limits items per order"
  jvn find "the order limit" --prefix app/ --prefix tests/
  jvn find "the order limit" --start app/orders.py:42 --out ./order-evidence
  jvn find "the order limit" --max-calls 8 --max-depth 3 --max-steps 8
  jvn find "the order limit" --beam-width 1 --neighbours-per-kind 8
  jvn find "the order limit" --preview-lines 8 --max-slice-chars 12000 --max-line-chars 240
  jvn find "the order limit" --verbose

All flags are optional. No default depth, step, call or neighbour-count cap.
Explicit limits may leave work unexplored; inspect the result's outcome and not_inspected entries.
Find stops on a match; it is not an exhaustive find-all or an end-to-end trace.
For JSON field names, types and defaults: jvn schema find. Full examples: docs/cli.md.""",
    )
    schema = commands.add_parser("schema", help="print a command's JSON request schema (no model calls)")
    schema.add_argument("topic", choices=("find",), help="command whose request schema to show")
    help_command = commands.add_parser("help", help="show general or command-specific help")
    help_command.add_argument("topic", nargs="?", choices=("find", "schema"))
    find.add_argument(
        "target", help="Behavior to locate; name the concrete check, decision or transformation"
    )
    scope = find.add_argument_group("Scope and results")
    limits = find.add_argument_group("Optional search limits")
    evidence = find.add_argument_group("Search scheduling and model context")
    scope.add_argument(
        "--repo", default=".", help="Directory to inspect (default: current directory; Git optional)"
    )
    scope.add_argument(
        "--prefix", action="append", default=[], help="Optional file or directory scope; repeatable"
    )
    scope.add_argument(
        "--start",
        action="append",
        default=[],
        metavar="PATH:LINE",
        help="Known entry or caller line; repeatable. Without one, jvn chooses a narrow entry point.",
    )
    scope.add_argument(
        "--out",
        help="New or empty output directory (default: a unique run under ./jvn-results)",
    )
    defaults = SearchBudget()
    limits.add_argument(
        "--max-depth",
        type=int,
        default=defaults.max_depth,
        help="Maximum relationship hops from a start (0 means starts only; default: unlimited)",
    )
    limits.add_argument(
        "--max-steps",
        type=int,
        default=defaults.max_steps,
        help="Maximum distinct code openings during navigation (default: unlimited)",
    )
    limits.add_argument(
        "--max-calls",
        type=int,
        default=defaults.max_calls,
        help="Maximum model requests, including entry selection (default: unlimited; not a token cap)",
    )
    evidence.add_argument(
        "--beam-width",
        type=int,
        default=defaults.beam_width,
        help="Places opened per round (default: 3; 1 makes navigation sequential)",
    )
    limits.add_argument(
        "--neighbours-per-kind",
        type=int,
        default=defaults.neighbours_per_kind,
        help="Candidates retained per relationship kind per opening (default: unlimited)",
    )
    evidence.add_argument(
        "--preview-lines",
        type=int,
        default=defaults.preview_lines,
        help="Leading lines shown for each candidate preview (default: 8; 0 hides preview code)",
    )
    evidence.add_argument(
        "--max-slice-chars",
        type=int,
        default=defaults.max_slice_chars,
        help="Characters allowed in one opened code slice (default: 12000; not the whole request)",
    )
    evidence.add_argument(
        "--max-line-chars",
        type=int,
        default=defaults.max_line_chars,
        help="Characters shown per source/preview/signature line (default: 240)",
    )
    find.add_argument(
        "--verbose",
        action="store_true",
        help="show expanded masked requests on stderr (default: concise live progress)",
    )
    return parser


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.json is None:
        if args.command is None:
            parser.error("a command or --json FILE is required")
        return args
    if args.command is not None:
        parser.error("use --json FILE on its own; put command options in the JSON request")
    try:
        if args.json.lstrip().startswith(("{", "[")):
            payload = json.loads(args.json)
        elif args.json == "-":
            payload = json.load(sys.stdin)
        else:
            with Path(args.json).expanduser().open(encoding="utf-8") as source:
                payload = json.load(source)
    except (OSError, ValueError) as error:
        parser.error(f"cannot read JSON request: {error}")
    if not isinstance(payload, dict):
        parser.error("JSON request must be an object")
    command = payload.get("command", "find")
    # The command parser is the option schema for both input formats.
    if command != "find":
        parser.error(f"unknown JSON command: {command!r}; expected find")
    actions = _request_actions(_command_parser(parser, command))
    arguments = [command]
    for name, value in payload.items():
        if name == "command":
            continue
        action = actions.get(name)
        if action is None:
            parser.error(f"unknown JSON field: {name}")
        if value is None and action.default is None and action.option_strings:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                parser.error(f"JSON field {name} must be a boolean")
            if value:
                arguments.append(action.option_strings[0])
            continue
        values = value if isinstance(action, argparse._AppendAction) else [value]
        if not isinstance(values, list):
            parser.error(f"JSON field {name} must be an array")
        expected_type = action.type or str
        for item in values:
            if expected_type is int and isinstance(item, float) and item.is_integer():
                item = int(item)
            if type(item) is not expected_type:
                parser.error(f"JSON field {name} must contain {expected_type.__name__} values")
            if action.option_strings:
                arguments.append(f"{action.option_strings[0]}={item}")
    if "target" in payload:
        # A target beginning with '--' is still text, never another option.
        arguments.extend(["--", payload["target"]])
    parsed = parser.parse_args(arguments)
    parsed.json = args.json
    return parsed


def _command_parser(parser: argparse.ArgumentParser, command: str) -> argparse.ArgumentParser:
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return commands.choices[command]


def _request_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {action.dest: action for action in parser._actions if action.dest != "help"}


def _request_schema(parser: argparse.ArgumentParser) -> dict:
    properties = {"command": {"type": "string", "const": "find", "default": "find"}}
    required = []
    for name, action in _request_actions(parser).items():
        field = {"description": action.help, "type": "integer" if action.type is int else "string"}
        if isinstance(action, argparse._StoreTrueAction):
            field["type"] = "boolean"
        elif isinstance(action, argparse._AppendAction):
            field.update(type="array", items={"type": field["type"]})
        if action.option_strings:
            field["default"] = action.default
            if action.default is None:
                field["type"] = [field["type"], "null"]
        else:
            required.append(name)
        if name in NON_NEGATIVE_BUDGET_FIELDS:
            field["minimum"] = 0
        elif name in POSITIVE_BUDGET_FIELDS:
            field["minimum"] = 1
        properties[name] = field
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "jvn find request",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "examples": [{"target": "the check that limits how many items an order may have"}],
    }


def _validate_budget(budget: SearchBudget) -> None:
    invalid = [
        name
        for name in NON_NEGATIVE_BUDGET_FIELDS
        if (value := getattr(budget, name)) is not None and value < 0
    ]
    invalid += [name for name in POSITIVE_BUDGET_FIELDS if getattr(budget, name) < 1]
    if invalid:
        raise ValueError(f"invalid search budget fields: {', '.join(invalid)}")


def _prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _default_output(repository: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path.cwd() / "jvn-results" / f"{repository.name}-{stamp}"


def _scope_warning(file_count: int) -> str | None:
    if file_count <= 20_000:
        return None
    return f"jvn: large scope contains {file_count:,} tracked files; indexing may take longer"


def _load_typesafe_environment(
    environment: MutableMapping[str, str],
    path: Path | None = None,
) -> None:
    """Load official TypeSafe SDK settings: process environment, then checkout `.env`,
    then the legacy `~/.config/jvn/env`; a process value always takes precedence."""
    from .environment import load_typesafe_environment

    load_typesafe_environment(environment, legacy=path)


def _parse_start(index: CodeIndex, value: str) -> Place:
    path, separator, raw_line = value.rpartition(":")
    if not separator or not path:
        raise ValueError(f"start must be PATH:LINE, got {value!r}")
    try:
        line = int(raw_line)
    except ValueError as error:
        raise ValueError(f"start line must be an integer, got {value!r}") from error
    if line < 1 or line > len(index.lines(path)):
        raise ValueError(f"start line is outside {path}: {line}")
    return place_for_line(index, path, line, "caller-provided start")


def _manifest(
    repository: Path,
    prefixes: tuple[str, ...],
    target: str,
    starts: tuple[str, ...],
    budget: SearchBudget,
    thresholds: Thresholds,
    index: CodeIndex,
    result: FindResult,
    *,
    requested_model: str,
    served_model: str | None,
    input_tokens: int,
    duration_seconds: float,
    total_calls: int,
    entry_selection: EntrySelection | None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "navigator": _navigator_provenance(),
        "source": {
            "repository": str(repository),
            "revision": index.commit,
            "prefixes": list(prefixes),
            "tracked_files": len(index.files),
        },
        "target": target,
        "requested_starts": list(starts),
        "entry_selection": entry_selection.to_json() if entry_selection else None,
        "budget": asdict(budget),
        "thresholds": thresholds.as_dict(),
        "provider": {
            "requested_model": requested_model,
            "served_model": served_model,
            "input_tokens": input_tokens,
        },
        "search": {
            "outcome": result.outcome,
            "steps": result.steps,
            "calls": total_calls,
            "entry_calls": total_calls - result.calls,
            "navigation_calls": result.calls,
            "duration_seconds": round(duration_seconds, 3),
            "moves": list(result.moves),
            "found": [_visit(visit) for visit in result.found],
            "starts": [_visit(visit) for visit in result.starts],
            "searched": [_visit(visit) for visit in result.searched],
            "unsure": [_visit(visit) for visit in result.unsure],
            "not_inspected": [
                {
                    "place": entry.place_key,
                    "signature": entry.signature,
                    "reason": entry.reason,
                    "priority": entry.priority,
                    "depth": entry.depth,
                    "path": list(entry.path),
                    "tier": entry.tier.name.lower(),
                    "included_in_opened_span": _included_in_opened_span(entry.place.open(), result),
                }
                for entry in result.not_inspected
            ],
            "unparsed_files": sorted(result.unparsed_files),
            "parser_scans": {
                "completed": list(result.parser_scans_completed),
                "pending": list(result.parser_scans_pending),
            },
            "unavailable_files": dict(result.unavailable_files),
            "history": [step.to_json() for step in result.history.steps] if result.history else [],
        },
    }


def _navigator_provenance() -> dict:
    package_root = Path(__file__).resolve().parent
    source_files = sorted(package_root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    repository = next((parent for parent in package_root.parents if (parent / ".git").exists()), None)
    revision = None
    dirty = None
    if repository is not None:
        revision = _git(repository, "rev-parse", "HEAD")
        dirty = bool(_git(repository, "status", "--porcelain", "--untracked-files=all"))
    return {
        "package_version": importlib.metadata.version("jev-navigator"),
        "source_revision": revision,
        "source_dirty": dirty,
        "source_tree_sha256": digest.hexdigest(),
    }


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _visit(visit: Visit) -> dict:
    return {
        "place": visit.place_key,
        "source": visit.code.source(),
        "code": visit.code.text,
        "probability": visit.probability,
        "verdict": visit.verdict,
        "path": list(visit.path),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _included_in_opened_span(candidate, result: FindResult) -> str | None:
    span = candidate.span
    for visit in (*result.found, *result.starts, *result.unsure, *result.searched):
        opened = visit.code.span
        if opened.file == span.file and opened.start <= span.start and span.end <= opened.end:
            return visit.place_key
    return None


_FRONTIER_REASONS = {
    "target_found": "Search stopped after finding a match",
    "deprioritized": "Candidate score did not exceed the opening threshold",
    "budget": "Configured search limit reached",
    "depth": "Configured depth limit reached",
    "cancelled": "Search cancelled",
    "stop_rule": "Caller stop condition met",
    "scope_incomplete": "Source scope incomplete",
    "neighbours_per_kind": "Configured neighbour limit reached",
}


def _report(manifest: dict) -> str:
    source = manifest["source"]
    search = manifest["search"]
    lines = [
        "# Jev navigator evidence pack",
        "",
        f"- Schema: `{manifest['schema_version']}`",
        f"- Navigator: `{manifest['navigator']['package_version']}` at "
        f"`{manifest['navigator']['source_revision'] or manifest['navigator']['source_tree_sha256']}`",
        f"- Revision: `{source['revision']}`",
        f"- Scope: {', '.join(f'`{prefix}`' for prefix in source['prefixes']) or 'whole directory'}",
        f"- Target: {manifest['target']}",
        f"- Outcome: **{search['outcome']}**",
        f"- Search: {search['steps']} opened places, {search['calls']} live calls",
        f"- Provider: requested `{manifest['provider']['requested_model']}`, served "
        f"`{manifest['provider']['served_model']}`",
        f"- Navigation elapsed: {search['duration_seconds']:.3f} seconds "
        "(indexing and entry selection excluded)",
        f"- Coverage caveat: {len(search['not_inspected'])} candidates were not independently opened; "
        f"{len(search['unparsed_files'])} files failed a completed parser scan. "
        f"Pending parser scans: {', '.join(search['parser_scans']['pending']) or 'none'}.",
        f"- Files that disappeared after inventory: {len(search['unavailable_files'])}.",
        "",
        "## Opened code",
        "",
        "| Set | Probability | Verdict | Source |",
        "| --- | ---: | --- | --- |",
    ]
    for name in ("found", "starts", "unsure", "searched"):
        for visit in search[name]:
            source = visit["source"]
            lines.append(
                f"| {name} | {visit['probability']:.3f} | {visit['verdict']} | "
                f"`{source['file']}:{source['lines'][0]}-{source['lines'][1]}` |"
            )
    if not any(search[name] for name in ("found", "starts", "unsure", "searched")):
        lines.append("| — | — | — | No code was opened. |")
    lines += ["", "## Found spans", ""]
    if not search["found"]:
        lines.append("No span crossed the configured yes threshold.")
    for visit in search["found"]:
        source = visit["source"]
        language = Path(source["file"]).suffix.removeprefix(".")
        lines += [
            f"### `{source['file']}:{source['lines'][0]}-{source['lines'][1]}`",
            "",
            f"Raw P(contains target): **{visit['probability']:.3f}**. Reached by `{source['reached_by']}`.",
            "",
            f"```{language}",
            visit["code"],
            "```",
            "",
        ]
    lines += [
        "## Candidates not independently opened",
        "",
        "This records separate candidate evaluations, not unseen text. Some candidates were already "
        "included in a larger opened span; that does not give them an independent model judgment.",
        "",
    ]
    if not search["not_inspected"]:
        lines.append("The search left no candidates awaiting an independent opening.")
    else:
        lines += [
            "| Why no separate opening | Recorded candidate score | Code coverage | Place |",
            "| --- | ---: | --- | --- |",
        ]
        for entry in search["not_inspected"]:
            reason = _FRONTIER_REASONS.get(entry["reason"], entry["reason"])
            included = entry.get("included_in_opened_span")
            coverage = (
                f"Lines included in opened span `{included}`"
                if included
                else "No containing opened span recorded"
            )
            lines.append(f"| {reason} | {entry['priority']:.3f} | {coverage} | `{entry['place']}` |")
    lines += [
        "",
        "The complete source spans, raw probabilities, decisions, and history are in "
        "`manifest.json`; provider response records and request hashes are in `journal.jsonl`. "
        "Each response record’s `exact` flag distinguishes wire capture from SDK-decoded data.",
        "",
    ]
    return "\n".join(lines)
