"""Command-line evidence pack for one live ``find_code`` search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic

from .adapters.typesafe import TypeSafeJevClient
from .cli_resume import SavedSearch, load_resume, save_resume
from .cli_statistics import STATISTICS_KINDS, STATISTICS_OPERATIONS, create_statistics_pack
from .cli_trace import create_trace_evidence_pack
from .directives.entry import EntrySelection, choose_initial_candidates
from .directives.find_all import CONTAINS_IMPLEMENTATION, FindAllResult, find_all
from .directives.find_code import FindResult, Outcome, SearchBudget, Visit, find_code
from .directives.places import Place, place_for_line
from .index.code_index import CodeIndex
from .index.languages import language_of
from .judgments.answers import TokenTotal
from .judgments.client import JevClient
from .judgments.judge import CallCapReachedError, Judge
from .judgments.store import SHARED_STORE_VARIABLE, default_shared_store, run_answer_store, shared_store_path
from .judgments.thresholds import Thresholds
from .operations import TraceGraph
from .progress import ProgressJournal, TerminalProgress
from .run_files import place_label, shown_relation, step_without_key_mentions

SCHEMA_VERSION = "jev-navigator.evidence-pack/v1"
KEEP_REQUESTS_HELP = (
    "Keep the code and full request text in the run folder (default: code locations and request "
    "hashes only); for your own or open-source code"
)
NON_NEGATIVE_BUDGET_FIELDS = ("max_depth", "max_steps", "max_calls", "neighbours_per_kind", "preview_lines")
POSITIVE_BUDGET_FIELDS = ("beam_width", "max_slice_chars", "max_line_chars")
# Each call is a paid request, so a bare `jvn find` stops at this many; `--max-calls none` lifts it.
DEFAULT_MAX_CALLS = 24
DEFAULT_FIND_ALL_MAX_CALLS = 2 * DEFAULT_MAX_CALLS


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "schema":
        print(json.dumps(_request_schema(_command_parser(_parser(), args.topic)), indent=2))
        return 0
    if args.command == "help":
        parser = _parser()
        (_command_parser(parser, args.topic) if args.topic else parser).print_help()
        return 0
    if args.command == "stats":
        return _run_statistics(args)
    if args.command not in ("find", "findall", "trace"):
        raise AssertionError(f"unhandled command: {args.command}")
    budget = SearchBudget(
        **{
            name: getattr(args, name)
            for name in (*NON_NEGATIVE_BUDGET_FIELDS, *POSITIVE_BUDGET_FIELDS)
            if hasattr(args, name)
        }
    )
    try:
        _validate_budget(budget)
    except ValueError as error:
        _parser().error(str(error))
    repository = Path(args.repo).resolve()
    output = Path(args.out).expanduser() if args.out else _default_output(repository)
    answer_store = _answer_store(args)
    client: TypeSafeJevClient | None = None
    try:
        _load_typesafe_environment(os.environ)
        client = TypeSafeJevClient()  # model=None resolves TYPESAFE_DEFAULT_MODEL in the adapter
        if args.command == "trace":
            manifest = create_trace_evidence_pack(
                repository,
                args.target,
                tuple(args.start),
                output,
                client,
                prefixes=tuple(args.prefix),
                thresholds=Thresholds.from_env(),
                depth=budget.max_depth,
                max_calls=budget.max_calls,
                verbose=args.verbose,
                answer_store=answer_store,
                keep_requests=args.keep_requests,
            )
        else:
            resume_from = Path(args.resume).expanduser() if getattr(args, "resume", None) else None
            while True:
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
                    answer_store=answer_store,
                    keep_requests=args.keep_requests,
                    workflow=args.command,
                    resume_from=resume_from,
                )
                if not _continue_search(args, budget, manifest, output):
                    break
                resume_from = output
                output = output.parent / _default_output(repository).name
    except Exception as error:
        print(f"jvn {args.command}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"jvn {args.command}: cancelled", file=sys.stderr)
        return 130
    finally:
        if client is not None:
            client.close()
    result_key = "trace" if args.command == "trace" else "search"
    result = manifest[result_key]
    search_outcome = result["outcome"]
    calls = manifest["provider"]["calls"] if args.command == "trace" else result["calls"]
    resume_directory = (
        str(output.resolve())
        if args.command in ("find", "findall") and search_outcome in ("budget", "cancelled")
        else None
    )
    if args.json:
        print(
            json.dumps(
                {
                    "output_directory": str(output.resolve()),
                    "manifest": str(output.resolve() / "manifest.json"),
                    "report": str(output.resolve() / "report.md"),
                    result_key: result,
                    "provider": manifest["provider"],
                    "resume": resume_directory,
                }
            )
        )
    else:
        print(f"evidence pack: {output.resolve()}")
        print(f"outcome: {_outcome_summary(result)} ({calls} live calls)")
        if resume_directory is not None:
            print(f"resume: use --resume {output.resolve()} with the same target and repository")
    return 130 if search_outcome == "cancelled" else 0


def _continue_search(args: argparse.Namespace, budget: SearchBudget, manifest: dict, output: Path) -> bool:
    """Offer another paid-call allowance only after the search has saved its resumable state."""
    if (
        args.command not in ("find", "findall")
        or args.json
        or not all(stream.isatty() for stream in (sys.stdin, sys.stdout, sys.stderr))
        or not budget.max_calls
        or manifest["search"]["outcome"] != "budget"
        or manifest["search"]["calls_this_invocation"] < budget.max_calls
    ):
        return False
    print(f"Partial results saved: {output.resolve()}", file=sys.stderr, flush=True)
    print(
        f"Continue the saved search with up to {budget.max_calls} more live model calls? [y/N] ",
        end="",
        file=sys.stderr,
        flush=True,
    )
    return sys.stdin.readline().strip().casefold() in ("y", "yes")


def _run_statistics(args: argparse.Namespace) -> int:
    repository = Path(args.repo).resolve()
    output = Path(args.out).expanduser() if args.out else _default_output(repository)
    progress = TerminalProgress(None)
    progress.start()
    outcome = "failed"
    try:
        progress.phase("indexing files for structural measurements")
        index = CodeIndex.from_directory(
            repository,
            prefixes=tuple(args.prefix),
            exclude_paths=(output, Path.cwd() / "jvn-results"),
            scan_observer=progress.scan,
        )
        pack = create_statistics_pack(
            repository,
            tuple(args.prefix),
            output,
            args.operation or STATISTICS_OPERATIONS,
            kinds=args.kind or STATISTICS_KINDS,
            held=not args.top_level,
            limit=args.limit,
            min_lines=args.min_lines,
            max_lines=args.max_lines,
            index=index,
        )
        outcome = "completed" if pack["coverage"]["complete"] else "partial"
        if args.json:
            print(
                json.dumps(
                    {
                        "output_directory": str(output.resolve()),
                        "manifest": str(output.resolve() / "statistics.json"),
                        "report": str(output.resolve() / "statistics.md"),
                        "statistics": pack,
                    }
                )
            )
        else:
            print(f"statistics pack: {output.resolve()}")
            if "counts" in pack:
                print("counts: " + ", ".join(f"{k}={v}" for k, v in pack["counts"]["totals"].items()))
            if "largest" in pack:
                for symbol in pack["largest"]["biggest"]:
                    print(
                        f"largest: {symbol['name']} ({symbol['size']} lines) "
                        f"{symbol['file']}:{symbol['lines'][0]}-{symbol['lines'][1]}"
                    )
            print(f"coverage: {outcome}; 0 model calls")
        return 0
    except KeyboardInterrupt:
        outcome = "cancelled"
        print("jvn stats: cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"jvn stats: {error}", file=sys.stderr)
        return 1
    finally:
        progress.close(outcome)


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
    answer_store: Path | None = None,
    workflow: str = "find",
    resume_from: Path | None = None,
    keep_requests: bool = False,
) -> dict:
    """Run the real index/search owners and persist their reviewable evidence. By default the pack
    keeps code locations and request hashes; ``keep_requests`` also keeps the code and request text."""
    if workflow not in ("find", "findall"):
        raise ValueError(f"unknown search workflow: {workflow}")
    repository = repository.resolve()
    output = output.resolve()
    _validate_budget(budget)
    thresholds = thresholds or Thresholds()
    previous = _previous_pack(resume_from, repository, prefixes, target, starts, thresholds, client, workflow)
    _prepare_output(output)
    if resume_from is not None:
        for name in ("answers.jsonl", "journal.jsonl"):
            source = resume_from.resolve() / name
            if source.is_file():
                shutil.copyfile(source, output / name)
    journal_path = output / "journal.jsonl"
    journal_path.touch()
    progress = TerminalProgress(journal_path, verbose=verbose)
    journal = ProgressJournal(journal_path, progress, keep_request_text=keep_requests)
    progress.start()
    outcome = "failed"
    try:
        progress.phase("indexing files")
        excluded = (output, Path.cwd() / "jvn-results")
        if resume_from is not None:
            excluded += (resume_from.resolve(),)
        index = CodeIndex.from_directory(
            repository,
            prefixes=prefixes,
            exclude_paths=excluded,
            scan_observer=progress.scan,
            fact_cache_dir=fact_cache_dir,
        )
        journal.place_label = partial(place_label, index)
        if warning := _scope_warning(len(index.files)):
            print(warning, file=sys.stderr)
        checkpoint = SavedSearch(None)
        if previous is not None:
            if previous["source"]["revision"] != index.commit:
                raise ValueError("repository revision changed since the evidence pack")
            checkpoint = load_resume(resume_from.resolve() / "resume.json", index)
        resume = checkpoint.result
        if resume is not None:
            journal.remember_moves({entry.place_key: entry.place.move for entry in resume.not_inspected})
        resuming_enumeration = checkpoint.completed is not None
        if resuming_enumeration and checkpoint.check_id != CONTAINS_IMPLEMENTATION.question_id:
            raise ValueError("Find All question changed since the evidence pack; start a new search")
        judge = Judge(
            client,
            thresholds=thresholds,
            max_calls=budget.max_calls,
            served_model=previous["provider"]["served_model"] if previous else None,
            journal=journal,
            store=run_answer_store(output / "answers.jsonl", answer_store),
        )
        selection: EntrySelection | None = None
        started = monotonic()
        entry_pending = False
        if resume is not None:
            start_places = []
            initial_candidates = ()
        elif starts:
            start_places = [_parse_start(index, start) for start in starts]
            initial_candidates: tuple[tuple[Place, float], ...] = ()
        else:
            progress.phase("choosing an entry point")
            start_places = []
            try:
                selection = choose_initial_candidates(index, judge, target)
            except CallCapReachedError:
                entry_pending = True
            initial_candidates = (
                tuple(
                    (
                        candidate.place,
                        candidate.selection_probability
                        if candidate.selection_probability is not None
                        else 0.0,
                    )
                    for candidate in selection.candidates
                )
                if selection
                else ()
            )
        if resuming_enumeration:
            assert resume is not None
            result = resume
        elif entry_pending:
            result = FindResult(Outcome.BUDGET, (), (), (), (), 0, 0)
        else:
            progress.phase("navigating code")
            result = find_code(
                index,
                judge,
                target,
                start_places,
                budget=budget,
                commit=None,
                initial_candidates=initial_candidates,
                resume=resume,
            )
        seed_calls = judge.calls
        seed_duration_seconds = monotonic() - started
        enumeration = None
        if workflow == "findall" and result.outcome not in (Outcome.BUDGET, Outcome.CANCELLED):
            progress.phase("expanding seed and checking remaining functions")
            enumeration = find_all(
                index,
                judge,
                target,
                [visit.code.span for visit in result.found],
                completed=checkpoint.completed or (),
                check=CONTAINS_IMPLEMENTATION,
            )
        duration_seconds = monotonic() - started
        progress.phase("writing evidence pack")
        scope_unavailable: dict[str, str] = {}
        needs_resume = (
            enumeration.stopped_by in ("budget", "cancelled")
            if enumeration is not None
            else result.outcome in (Outcome.BUDGET, Outcome.CANCELLED)
        )
        if needs_resume:
            scope_unavailable = save_resume(
                output / "resume.json",
                index,
                result,
                entry_pending=entry_pending,
                completed=enumeration.judged if enumeration is not None else None,
                check_id=CONTAINS_IMPLEMENTATION.question_id if enumeration is not None else None,
            )
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
            input_total=judge.input_total,
            duration_seconds=seed_duration_seconds,
            total_calls=seed_calls,
            entry_selection=selection,
            previous=previous,
            resume_from=resume_from,
            entry_pending=entry_pending,
            scope_unavailable=scope_unavailable,
        )
        manifest["workflow"] = workflow
        if workflow == "findall" and enumeration is None:
            enumeration = FindAllResult(
                target,
                TraceGraph((), (), (), "not_started"),
                (),
                tuple(index.files),
                index.observed_unparsed_files,
                tuple(file for file in index.files if not language_of(file)),
                index.unavailable_files,
                str(result.outcome),
                0,
            )
        if enumeration is not None:
            manifest["seed_search"] = previous["seed_search"] if resuming_enumeration else manifest["search"]
            manifest["search"] = _find_all_summary(enumeration, judge.calls, duration_seconds, previous)
        if not keep_requests:
            _drop_code(manifest, index)
        _write_json(output / "manifest.json", manifest)
        (output / "report.md").write_text(
            _find_all_report(manifest) if enumeration is not None else _report(manifest)
        )
        outcome = str(manifest["search"]["outcome"])
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

All flags are optional. Live calls stop at 24 unless --max-calls sets another cap ('none' lifts it);
there is no default depth, step or neighbour-count cap.
In an interactive terminal, a call-budget stop saves the frontier and offers another allowance.
JSON and piped invocations never prompt; use --resume for a follow-up invocation.
Explicit limits may leave work unexplored; inspect the result's outcome and not_inspected entries.
Find stops on a match; it is not an exhaustive find-all or an end-to-end trace.
For JSON field names, types and defaults: jvn schema find. Full examples: docs/cli.md.""",
    )
    schema = commands.add_parser("schema", help="print a command's JSON request schema (no model calls)")
    schema.add_argument(
        "topic", choices=("find", "findall", "trace", "stats"), help="command whose request schema to show"
    )
    help_command = commands.add_parser("help", help="show general or command-specific help")
    help_command.add_argument("topic", nargs="?", choices=("find", "findall", "trace", "stats", "schema"))
    _add_search_arguments(find)
    find.add_argument(
        "--resume",
        help="Prior budget-stopped evidence pack; continue its saved frontier into a new output pack",
    )
    findall = commands.add_parser(
        "findall",
        help="find matching function bodies, including disconnected implementations",
        description="Find a seed, expand code relationships, then judge remaining function bodies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='jvn findall "functions that reject an order exceeding the item limit"\n'
        "Default: 48 live model calls; --max-calls none lifts it. No file cap.\n"
        "Coverage describes functions examined, not proof of absence.\n"
        "All existing scope, output and request-display options are available.\n"
        'JSON: {"command":"findall","target":"functions checking the order item limit"}',
    )
    _add_search_arguments(findall, max_calls=DEFAULT_FIND_ALL_MAX_CALLS)
    findall.add_argument("--resume", help="Prior partial Find All pack; continue without repeating judgments")
    trace = commands.add_parser(
        "trace",
        help="trace a known entry point and judge source evidence in batches",
        description="Follow static code relationships and inspect five atomic evidence obligations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='jvn trace "how an order becomes an HTTP result" --start app/orders.py:42\n'
        'JSON: {"command":"trace","target":"order to HTTP result","start":["app/orders.py:42"]}\n'
        "Start paths are relative to --repo. Unknown bindings remain visible.\n"
        "No default depth or model-call cap; --max-calls sets an optional request allowance.\n"
        "A completed trace is not proof of semantic completeness. "
        "Saved continuation belongs to find and findall.",
    )
    trace.add_argument("target", help="Concrete workflow, transformation or outcome to trace")
    trace.add_argument(
        "--start",
        action="append",
        required=True,
        metavar="PATH:LINE",
        help="Known function entry line; required and repeatable",
    )
    trace.add_argument("--repo", default=".", help="Source directory (default: current directory)")
    trace.add_argument("--prefix", action="append", default=[], help="Optional source scope; repeatable")
    trace.add_argument("--out", help="New or empty output directory (default: ./jvn-results/<run>)")
    trace.add_argument("--max-depth", type=int, help="Optional maximum static relationship hops")
    trace.add_argument(
        "--max-calls", type=_count_or_none, help="Optional model-request cap; none is unlimited"
    )
    trace.add_argument("--verbose", action="store_true", help="Print expanded masked model requests")
    _add_answer_store_argument(trace)
    trace.add_argument("--keep-requests", action="store_true", help=KEEP_REQUESTS_HELP)
    stats = commands.add_parser(
        "stats",
        help="count and rank parsed functions/classes without model calls",
        description="Measure parser facts locally; no model or API key is used.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="jvn stats\njvn stats --kind function --limit 1\n"
        "jvn stats --operation range --kind function --min-lines 20 --max-lines 80\n"
        "jvn stats --operation count --prefix src/\n"
        'JSON: {"command":"stats","kind":["function"],"limit":1}\n'
        "Counts cover all parsed symbols. --limit changes only the displayed ranking.\n"
        "Methods and nested functions are included unless --top-level is supplied.",
    )
    stats.add_argument("--repo", default=".", help="Source directory (default: current directory)")
    stats.add_argument("--prefix", action="append", default=[], help="Source scope; repeatable")
    stats.add_argument("--out", help="New or empty output directory (default: ./jvn-results/<run>)")
    stats.add_argument(
        "--operation",
        action="append",
        choices=STATISTICS_OPERATIONS,
        help="Sections to produce; repeatable, default: all",
    )
    stats.add_argument(
        "--kind",
        action="append",
        choices=STATISTICS_KINDS,
        help="Symbol kind; repeatable, default: functions and classes",
    )
    stats.add_argument("--limit", type=int, help="Number of ranked symbols to display; counts stay complete")
    stats.add_argument("--min-lines", type=int, help="Inclusive minimum symbol size in the range listing")
    stats.add_argument("--max-lines", type=int, help="Inclusive maximum symbol size in the range listing")
    stats.add_argument(
        "--top-level", action="store_true", help="Exclude nested symbols from ranking and ranges"
    )
    return parser


def _answer_store(args: argparse.Namespace) -> Path:
    """The shared store this run uses: ``--answer-store`` when given, else ``shared_store_path()``.
    The run says which on stderr."""
    path = Path(args.answer_store).expanduser().resolve() if args.answer_store else shared_store_path()
    print(f"answer store: {path}", file=sys.stderr)
    return path


def _add_answer_store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--answer-store",
        metavar="PATH",
        help=(
            f"Shared answer store file (default: ${SHARED_STORE_VARIABLE}, else {default_shared_store()}); "
            "a new file keeps this run from replaying another run's answers"
        ),
    )


def _add_search_arguments(find: argparse.ArgumentParser, *, max_calls: int = DEFAULT_MAX_CALLS) -> None:
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
    _add_answer_store_argument(find)
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
        type=_count_or_none,
        default=max_calls,
        metavar="N|none",
        help=(
            f"Maximum model requests, including entry selection (default: {max_calls}; "
            "'none' for no cap; not a token cap)"
        ),
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
    find.add_argument("--keep-requests", action="store_true", help=KEEP_REQUESTS_HELP)


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
    if command not in ("find", "findall", "trace", "stats"):
        parser.error(f"unknown JSON command: {command!r}; expected find, findall, trace or stats")
    actions = _request_actions(_command_parser(parser, command))
    arguments = [command]
    for name, value in payload.items():
        if name == "command":
            continue
        action = actions.get(name)
        if action is None:
            parser.error(f"unknown JSON field: {name}")
        if value is None and action.type is _count_or_none:
            arguments.append(f"{action.option_strings[0]}=none")
            continue
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
        expected_type = int if action.type is _count_or_none else action.type or str
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
    command = parser.prog.rsplit(" ", 1)[-1]
    properties = {"command": {"type": "string", "const": command, "default": command}}
    required = []
    for name, action in _request_actions(parser).items():
        numeric = action.type in (int, _count_or_none)
        field = {"description": action.help, "type": "integer" if numeric else "string"}
        if isinstance(action, argparse._StoreTrueAction):
            field["type"] = "boolean"
        elif isinstance(action, argparse._AppendAction):
            field.update(type="array", items={"type": field["type"]})
            if action.required:
                field["minItems"] = 1
        if action.choices is not None:
            choice_field = field["items"] if isinstance(action, argparse._AppendAction) else field
            choice_field["enum"] = list(action.choices)
        if action.option_strings:
            field["default"] = action.default
            if not action.required and (action.default is None or action.type is _count_or_none):
                field["type"] = [field["type"], "null"]
        if not action.option_strings or action.required:
            required.append(name)
        if name in NON_NEGATIVE_BUDGET_FIELDS:
            field["minimum"] = 0
        elif name in (*POSITIVE_BUDGET_FIELDS, "limit", "min_lines", "max_lines"):
            field["minimum"] = 1
        properties[name] = field
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"jvn {command} request",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "examples": [
            {
                "command": command,
                **({"target": "the order item limit"} if command != "stats" else {}),
                **({"start": ["app/orders.py:42"]} if command == "trace" else {}),
            }
        ],
    }


def _count_or_none(value: str) -> int | None:
    """A ``--max-calls`` value: a number, or ``none`` for no cap."""
    if value == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'none', got {value!r}") from None


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


def _previous_pack(
    resume_from: Path | None,
    repository: Path,
    prefixes: tuple[str, ...],
    target: str,
    starts: tuple[str, ...],
    thresholds: Thresholds,
    client: JevClient,
    workflow: str,
) -> dict | None:
    if resume_from is None:
        return None
    source = resume_from.resolve()
    previous = json.loads((source / "manifest.json").read_text())
    if not (source / "resume.json").is_file():
        raise ValueError(f"no saved search frontier in {source}")
    if previous["search"]["outcome"] not in ("budget", "cancelled"):
        raise ValueError("only a budget-stopped or cancelled search can resume")
    if (
        previous.get("workflow", "find") != workflow
        or previous["source"]["repository"] != str(repository)
        or previous["source"]["prefixes"] != list(prefixes)
        or previous["target"] != target
        or previous["requested_starts"] != list(starts)
        or previous["thresholds"] != thresholds.as_dict()
        or previous["provider"]["requested_model"] != getattr(client, "model", "unknown")
    ):
        raise ValueError(
            "resume must use the same workflow, repository, scope, target, starts, thresholds and model"
        )
    return previous


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
    input_total: TokenTotal,
    duration_seconds: float,
    total_calls: int,
    entry_selection: EntrySelection | None,
    previous: dict | None = None,
    resume_from: Path | None = None,
    entry_pending: bool = False,
    scope_unavailable: dict[str, str] | None = None,
) -> dict:
    old_search = previous.get("seed_search", previous["search"]) if previous else {}
    entry_receipt = entry_selection.to_json() if entry_selection else None
    if entry_receipt is None and previous:
        entry_receipt = previous.get("entry_selection")
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
        "entry_selection": entry_receipt,
        "resume_from": str(resume_from.resolve()) if resume_from else None,
        "budget": asdict(budget),
        "thresholds": thresholds.as_dict(),
        "provider": {
            "requested_model": requested_model,
            "served_model": served_model,
            "input_tokens": (previous["provider"]["input_tokens"] if previous else 0) + input_total.reported,
            "responses_without_usage": _plus_known(_carried_unreported(previous), input_total.not_reported),
        },
        "search": {
            "outcome": result.outcome,
            "entry_selection_pending": entry_pending,
            "steps": old_search.get("steps", 0) + result.steps,
            "calls": old_search.get("calls", 0) + total_calls,
            "calls_this_invocation": total_calls,
            "entry_calls": old_search.get("entry_calls", 0) + total_calls - result.calls,
            "navigation_calls": old_search.get("navigation_calls", 0) + result.calls,
            "duration_seconds": round(old_search.get("duration_seconds", 0) + duration_seconds, 3),
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
            "files_judged": result.files_judged,
            "files_read": result.files_read,
            "code_files": result.code_files,
            "parser_scans": {
                "completed": list(result.parser_scans_completed),
                "pending": list(result.parser_scans_pending),
            },
            "unavailable_files": {**result.unavailable_files, **(scope_unavailable or {})},
            "history": [
                *old_search.get("history", []),
                *([step.to_json() for step in result.history.steps] if result.history else []),
            ],
        },
    }


def _drop_code(manifest: dict, index: CodeIndex) -> None:
    """Leave each place as its location: the code it held stays in the repository."""
    _drop_entry_code(manifest.get("entry_selection") or {}, index)
    for name in ("search", "seed_search"):
        _drop_search_code(manifest.get(name) or {}, index)


def _drop_entry_code(entry_selection: dict, index: CodeIndex) -> None:
    for decision in entry_selection.get("decisions", []):
        for option in decision.get("options", []):
            option.pop("description", None)
    for candidate in entry_selection.get("candidates", []):
        candidate["signature"] = place_label(index, candidate["place"])


def _drop_search_code(search: dict, index: CodeIndex) -> None:
    visits = [visit for group in ("found", "starts", "searched", "unsure") for visit in search.get(group, [])]
    moves = {visit["place"]: visit.get("move") for visit in visits if "place" in visit}
    for visit in visits:
        visit.pop("code", None)
        source = visit.get("source") or {}
        if "place" in visit and "reached_by" in source:
            source["reached_by"] = shown_relation(visit.get("move"), source["reached_by"], visit["place"])
    for entry in search.get("not_inspected", []):
        entry["signature"] = place_label(index, entry["place"])
    search["history"] = [_step_without_code(step, moves, index) for step in search.get("history", [])]


def _step_without_code(step: dict, moves: dict, index: CodeIndex) -> dict:
    shown = step_without_key_mentions(step, moves)
    for fetched in shown["fetched"]:
        fetched.pop("code", None)
    for offered in shown["judgments"].get("could_contain", []):
        offered["signature"] = place_label(index, offered["place"])
    return shown


def _find_all_summary(result: FindAllResult, calls: int, elapsed: float, previous: dict | None) -> dict:
    old_search = previous["search"] if previous else {}

    def answer(value):
        return {
            "source": {key: value.item[key] for key in ("file", "lines", "commit", "file_sha256")},
            "code": value.item["code"],
            "name": value.item["name"],
            "probability": value.probability,
            "verdict": value.verdict,
            "request_sha256": value.request_sha256,
            "from_store": value.from_store,
        }

    return {
        "outcome": result.stopped_by,
        "coverage": result.coverage,
        "unit": "function",
        "calls": old_search.get("calls", 0) + calls,
        "calls_this_invocation": calls,
        "enumeration_calls": old_search.get("enumeration_calls", 0) + result.calls,
        "duration_seconds": round(old_search.get("duration_seconds", 0) + elapsed, 3),
        "found": [answer(value) for value in result.matched],
        "unsure": [answer(value) for value in result.uncertain],
        "searched": [answer(value) for value in result.negative],
        "remaining_files": list(result.remaining_files),
        "unparsed_files": sorted(result.unparsed_files),
        "unsupported_files": list(result.unsupported_files),
        "unavailable_files": dict(result.unavailable_files),
        "graph": asdict(result.graph),
    }


def _find_all_report(manifest: dict) -> str:
    search = manifest["search"]
    lines = [
        "# Find All evidence pack",
        "",
        f"Target: {manifest['target']}",
        "",
        f"Outcome: **{search['outcome']}**. Coverage: **{search['coverage']}**.",
        f"{search['calls']} live requests; "
        f"{search['duration_seconds']:.3f}s for seed search and enumeration.",
        "",
        "Coverage counts function bodies examined. It does not prove model accuracy, behavioral "
        "equivalence or absence of other implementations. Uncertain graph bindings remain uncertain.",
        "",
        "| Result | P(contains target) | Function | Source |",
        "| --- | ---: | --- | --- |",
    ]
    for group in ("found", "unsure", "searched"):
        for value in search[group]:
            source = value["source"]
            lines.append(
                f"| {group} | {value['probability']:.3f} | `{value['name']}` | "
                f"`{source['file']}:{source['lines'][0]}-{source['lines'][1]}` |"
            )
    lines += ["", "## Coverage gaps", ""]
    for field in ("remaining_files", "unparsed_files", "unsupported_files", "unavailable_files"):
        lines.append(f"- {field}: {', '.join(search[field]) or 'none'}")
    lines += ["", "## Matching bodies", ""]
    for value in search["found"]:
        source = value["source"]
        lines += [f"### {source['file']}:{source['lines'][0]}-{source['lines'][1]}", ""]
        lines += _code_block(value, "")
    return "\n".join(lines) + "\n"


def _carried_unreported(previous: dict | None) -> int | None:
    """The unreported-response count of an earlier receipt, ``None`` when it predates the field."""
    return 0 if previous is None else previous["provider"].get("responses_without_usage")


def _unreported_text(count: int | None) -> str:
    return "not known (earlier receipt)" if count is None else str(count)


def _plus_known(carried: int | None, added: int) -> int | None:
    return None if carried is None else carried + added


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
        "move": visit.move,
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


def _outcome_summary(search: dict) -> str:
    """The outcome name; an empty Find also says how much of the scope Jev judged and code read."""
    outcome = str(search["outcome"])
    if "files_judged" not in search or outcome not in ("nothing_left", "scope_incomplete"):
        return outcome
    judged, read, total = search["files_judged"], search["files_read"], search["code_files"]
    seen = f"Jev judged code in {judged} of {total} files"
    if outcome == "nothing_left":
        return f"{outcome} (nothing left worth opening: {seen}; all {total} were read)"
    parts = [f"not found: {seen}", f"{read - judged} more were read only to list links"]
    parts.append(f"{total - read} never reached")
    if search["unparsed_files"]:
        parts.append(f"{len(search['unparsed_files'])} parsed only partly")
    if search["unavailable_files"]:
        parts.append(f"{len(search['unavailable_files'])} gone from disk")
    return f"{outcome} ({'; '.join(parts)})"


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


def _code_block(place: dict, language: str) -> list[str]:
    """The place's code, when the pack kept it (``--keep-requests``); otherwise only its location."""
    return [f"```{language}", place["code"], "```", ""] if "code" in place else []


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
        f"- Outcome: **{_outcome_summary(search)}**",
        *(["- Entry selection awaits another call allowance."] if search["entry_selection_pending"] else []),
        f"- Search: {search['steps']} opened places, {search['calls']} live calls",
        f"- Provider: requested `{manifest['provider']['requested_model']}`, served "
        f"`{manifest['provider']['served_model']}`",
        f"- Responses without usage: {_unreported_text(manifest['provider']['responses_without_usage'])}",
        f"- Navigation elapsed: {search['duration_seconds']:.3f} seconds "
        "(indexing and entry selection excluded)",
        f"- Coverage caveat: {len(search['not_inspected'])} candidates were not independently opened; "
        f"{len(search['unparsed_files'])} files failed a completed parser scan. "
        f"Pending parser scans: {', '.join(search['parser_scans']['pending']) or 'none'}.",
        "- Files unavailable (disappeared or changed on disk, or too large to parse): "
        f"{len(search['unavailable_files'])}.",
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
            *_code_block(visit, language),
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
