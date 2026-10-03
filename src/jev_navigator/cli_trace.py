"""Evidence pack for one static workflow trace.

``create_trace_evidence_pack`` is the ordinary callable behind a future ``jvn trace`` command: it
runs the real index owner, the existing ``trace_workflow`` directive and the shared journal, answer
store and progress owners, then persists their reviewable JSON and Markdown evidence. The index owns
connectivity; Jev only judges the five typed evidence obligations, and a positive judgment never
turns a candidate or unresolved static link into proven handoff.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .directives.trace import EvidenceStatus, TraceObligation, TraceResult, trace_workflow
from .index.code_index import CodeIndex
from .index.spans import Span
from .judgments.client import JevClient
from .judgments.judge import CheckResult, Judge
from .judgments.store import JsonlAnswerStore
from .judgments.thresholds import Thresholds
from .progress import ProgressJournal, TerminalProgress

SCHEMA_VERSION = "jev-navigator.trace-evidence-pack/v1"


def create_trace_evidence_pack(
    repository: Path,
    question: str,
    starts: Sequence[str],
    output: Path,
    client: JevClient,
    *,
    prefixes: Sequence[str] = (),
    thresholds: Thresholds | None = None,
    depth: int | None = None,
    max_calls: int | None = None,
    verbose: bool = False,
    fact_cache_dir: Path | None = None,
    cancelled: Callable[[], bool] | None = None,
    served_model: str | None = None,
    answers_from: Path | None = None,
) -> dict:
    """Trace the workflow around ``starts`` and write the reviewable evidence pack to ``output``.

    Inputs follow ``create_evidence_pack`` where they apply: ``repository`` is the directory to
    index, ``prefixes`` narrow the scope, ``starts`` are ``PATH:LINE`` entry points (each line must
    lie inside a function), ``output`` must be a new or empty directory, and ``client`` is the
    caller's ``JevClient``. ``thresholds``, ``max_calls``, ``verbose`` and ``fact_cache_dir`` behave
    as there; ``depth`` and ``cancelled`` pass through to the static walk. ``served_model`` pins the
    model identity that ``answers.jsonl`` replays against, as a resumed pack does; ``answers_from``
    seeds this pack's answer store from a prior pack's, so identical questions about identical code
    replay without a new request. ``question`` is
    the workflow question every obligation is asked about.

    Returns the manifest that is persisted as ``manifest.json`` next to ``report.md``,
    ``answers.jsonl`` (the shared answer store) and ``journal.jsonl`` (the shared request journal
    and terminal progress). A cancelled walk still writes the pack: every obligation is then
    ``unresolved`` and no request was made. A call budget that stops a later batch also writes the
    pack with outcome ``budget``: answered obligations keep their evidence, obligations with
    unexamined spans stay ``unresolved``, and cached answers replay without a live call.
    """
    if not question.strip():
        raise ValueError("trace needs a workflow question")
    if not starts:
        raise ValueError("trace needs at least one start as PATH:LINE")
    repository = repository.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if answers_from is not None:
        source = answers_from.resolve()
        if not source.is_file():
            raise ValueError(f"no answer store to replay at {source}")
        shutil.copyfile(source, output / "answers.jsonl")
    thresholds = thresholds or Thresholds()
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
        start_spans = tuple(_start_span(index, start) for start in starts)
        judge = Judge(
            client,
            thresholds=thresholds,
            max_calls=max_calls,
            served_model=served_model,
            journal=journal,
            store=JsonlAnswerStore(output / "answers.jsonl"),
        )
        progress.phase("tracing workflow")
        result = trace_workflow(index, judge, question, start_spans, depth=depth, cancelled=cancelled)
        outcome = _outcome(result)
        manifest = _manifest(
            repository, question, tuple(starts), prefixes, thresholds, depth, index, judge, result
        )
        progress.phase("writing evidence pack")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        (output / "report.md").write_text(_report(manifest))
        return manifest
    except KeyboardInterrupt:
        outcome = "cancelled"
        raise
    finally:
        journal.record_terminal(outcome)
        progress.close(outcome)


def _start_span(index: CodeIndex, start: str) -> Span:
    """The concrete function span that contains the caller's ``PATH:LINE`` start."""
    path, separator, raw_line = start.rpartition(":")
    if not separator or not path:
        raise ValueError(f"start must be PATH:LINE, got {start!r}")
    try:
        line = int(raw_line)
    except ValueError as error:
        raise ValueError(f"start line must be an integer, got {start!r}") from error
    if line < 1 or line > len(index.lines(path)):
        raise ValueError(f"start line is outside {path}: {line}")
    span = index.enclosing_symbol(path, line)
    if span is None:
        raise ValueError(f"start line {line} of {path} is not inside a function")
    return span


def _manifest(
    repository: Path,
    question: str,
    starts: tuple[str, ...],
    prefixes: Sequence[str],
    thresholds: Thresholds,
    depth: int | None,
    index: CodeIndex,
    judge: Judge,
    result: TraceResult,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": question,
        "requested_starts": list(starts),
        "source": {
            "repository": str(repository),
            "revision": index.commit,
            "prefixes": list(prefixes),
            "tracked_files": len(index.files),
        },
        "thresholds": thresholds.as_dict(),
        "depth": depth,
        "provider": {
            "requested_model": getattr(judge.client, "model", "unknown"),
            "served_model": judge.served_model,
            "calls": judge.calls,
            "input_tokens": judge.input_total.reported,
            "responses_without_usage": judge.input_total.not_reported,
        },
        "trace": {
            "outcome": _outcome(result),
            "budget_stopped": result.budget_stopped,
            "graph_stop": result.graph.stop,
            "roots": [_span_json(span) for span in result.graph.roots],
            "functions": [_span_json(span) for span in result.graph.functions],
            "links": [_link_json(link) for link in result.graph.links],
            "obligations": [_obligation_json(obligation) for obligation in result.obligations],
            "included": [_span_json(span) for span in result.included],
            "excluded": [_span_json(span) for span in result.excluded],
            "unresolved_links": [_link_json(link) for link in result.unresolved_links],
        },
    }


def _outcome(result: TraceResult) -> str:
    """Keep cancellation, call budget and explicit traversal depth distinct."""
    if result.cancelled:
        return "cancelled"
    if result.budget_stopped:
        return "budget"
    return "depth" if result.graph.stop == "depth" else "completed"


def _obligation_json(obligation: TraceObligation) -> dict:
    return {
        "name": obligation.name,
        "status": obligation.status.value,
        "examined": obligation.examined,
        "evidence": [_result_json(item) for item in obligation.evidence],
        "unresolved": [_result_json(item) for item in obligation.unresolved],
        "checked": len(obligation.checked),
    }


def _result_json(result: CheckResult) -> dict:
    item = result.item
    return {
        "source": {key: item[key] for key in ("file", "lines", "commit", "file_sha256")},
        "code": item["code"],
        "span_key": item["span_key"],
        "probability": result.probability,
        "verdict": str(result.verdict),
        "from_store": result.from_store,
        "request_sha256": result.request_sha256,
    }


def _span_json(span: Span) -> dict:
    return {
        "key": span.key,
        "file": span.file,
        "lines": [span.start, span.end],
        "name": span.name,
    }


def _link_json(link) -> dict:
    binding = link.binding
    return {
        "relation": link.relation,
        "name": link.name,
        "site": f"{link.file}:{link.line}",
        "source": _span_json(link.source) if link.source else None,
        "target": _span_json(link.target) if link.target else None,
        "binding": None
        if binding is None
        else {"status": binding.status, "reason": binding.reason, "proven": binding.proven},
    }


def _report(manifest: dict) -> str:
    trace = manifest["trace"]
    lines = [
        "# Workflow trace evidence pack",
        "",
        f"- Schema: `{manifest['schema_version']}`",
        f"- Question: {manifest['question']}",
        f"- Revision: `{manifest['source']['revision']}`",
        f"- Starts: {', '.join(f'`{start}`' for start in manifest['requested_starts'])}",
        f"- Outcome: **{trace['outcome']}** (static walk: {trace['graph_stop']})",
        f"- Provider: requested `{manifest['provider']['requested_model']}`, served "
        f"`{manifest['provider']['served_model']}`, {manifest['provider']['calls']} live calls",
        "",
        "Connectivity is the index's static view. It is not proof of a correct handoff: only the "
        "obligations below carry source-identified Jev evidence, and uncertain or missing static "
        "links stay listed as gaps. No claim is made about model accuracy.",
        "",
        "## Evidence obligations",
        "",
        "| Obligation | Status | Evidence | Unresolved |",
        "| --- | --- | ---: | ---: |",
    ]
    for obligation in trace["obligations"]:
        lines.append(
            f"| {obligation['name']} | {obligation['status']} "
            f"| {len(obligation['evidence'])} | {len(obligation['unresolved'])} |"
        )
    unexamined = [obligation["name"] for obligation in trace["obligations"] if not obligation["examined"]]
    if unexamined:
        lines += [
            "",
            f"Judging stopped ({trace['outcome']}) before every span was examined, so "
            f"{', '.join(unexamined)} keep their unexamined spans `unresolved`; unexamined spans "
            "are never reported as a negative or a gap.",
            "",
        ]
    lines += ["", "## Evidence", ""]
    for obligation in trace["obligations"]:
        lines += [f"### {obligation['name']} — {obligation['status']}", ""]
        if not obligation["evidence"]:
            lines.append("No supplied source crossed the yes threshold for this obligation.")
            lines.append("")
        for evidence in obligation["evidence"]:
            source = evidence["source"]
            lines += [
                f"- `{source['file']}:{source['lines'][0]}-{source['lines'][1]}` "
                f"P(yes) {evidence['probability']:.3f} ({evidence['verdict']})"
            ]
    lines += ["", "## Unresolved static links", ""]
    if not trace["unresolved_links"]:
        lines.append("Every static link in the walked component is resolved.")
    else:
        lines += ["| Relation | Name | Site | Binding |", "| --- | --- | --- | --- |"]
        for link in trace["unresolved_links"]:
            binding = link["binding"]
            status = "no binding" if binding is None else binding["status"]
            lines.append(f"| {link['relation']} | {link['name']} | `{link['site']}` | {status} |")
    lines += [
        "",
        "## Component coverage",
        "",
        f"- Included in the evidence backbone: {len(trace['included'])} functions",
        f"- Excluded from the backbone: {len(trace['excluded'])} functions",
        "",
    ]
    lines += [
        f"- `{span['file']}:{span['lines'][0]}-{span['lines'][1]}` ({span['name']})"
        for span in trace["excluded"]
    ]
    lines += [
        "",
        "The complete obligations, evidence items with request hashes, and unresolved links are in "
        "`manifest.json`; provider responses are in `journal.jsonl`, answers in `answers.jsonl`.",
        "",
    ]
    return "\n".join(lines) + "\n"


# EvidenceStatus is re-exported for callers that inspect persisted statuses.
__all__ = ["SCHEMA_VERSION", "EvidenceStatus", "create_trace_evidence_pack"]
