"""Mechanical adapters over JVN's existing index, bindings, file resolver and Judge."""

from __future__ import annotations

import fnmatch
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import PurePosixPath

from . import operations
from .batch import Operation
from .directives.find_all import match_check
from .index.bindings import Binding, names_exactly
from .index.code_index import CodeIndex
from .index.scope import is_test_file
from .index.spans import Span
from .index.units import PIECE_LINES
from .judgments.client import MissingAnswerError, ReplayOnlyClient
from .judgments.judge import CallCapReachedError, Judge
from .judgments.questions import Check
from .mentions import spelling_variants


def rows_for(
    index: CodeIndex,
    op: Operation,
    judge: Judge | None,
    checks: Sequence[Check] | None,
    shared: Mapping | None,
) -> Iterator[dict]:
    if op.file and op.file not in index.files:
        raise ValueError(f"file is outside the index: {op.file}")
    if op.op == "outline":
        for file in selected_files(index, op):
            yield {"file": file, "imports": index.imports(file)}
            for span in index.symbols_in(file):
                yield {**location(span), "name": span.name}
    elif op.op == "show":
        yield from source_rows(index, op.file, op.line - op.window, (op.end or op.line) + op.window)
    elif op.op == "named_files":
        if op.patterns or op.scopes or not op.query:
            yield from ({"file": file} for file in selected_files(index, op))
        else:
            named = operations.files_named_by(index, (op.query,), (op.file,) if op.file else ())
            yield from ({"file": file, "named_by": token} for file, token in named.named_by.items())
    elif op.op == "names":
        yield from name_rows(index, op)
    elif op.op == "refs" and op.query:
        yield from text_rows(index, op)
    elif op.op in ("def", "refs", "callers", "callees"):
        yield from relation_rows(index, op)
    elif op.op == "cochange":
        yield from (
            {"file": file, "commits": count}
            for file, count in index.co_changed_files(op.file, limit=len(index.files))
        )
    elif op.op == "tests_of":
        yield from test_rows(index, op)
    elif op.op == "rank":
        yield from rank_rows(index, op, judge, checks, shared)


def selected_files(index: CodeIndex, op: Operation) -> Iterator[str]:
    positive = [pattern for pattern in op.patterns if not pattern.startswith("!")]
    negative = [pattern[1:] for pattern in op.patterns if pattern.startswith("!")]
    for file in index.files:
        if op.file and file != op.file:
            continue
        if op.scopes and not any(
            scope in (".", "./", "")
            or file == scope.removeprefix("./")
            or file.startswith(scope.removeprefix("./").rstrip("/") + "/")
            for scope in op.scopes
        ):
            continue
        if (not positive or any(_matches(file, p) for p in positive)) and not any(
            _matches(file, p) for p in negative
        ):
            yield file


def _matches(file: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(file, pattern) or fnmatch.fnmatchcase(PurePosixPath(file).name, pattern)


def source_rows(index: CodeIndex, file: str, start: int, end: int) -> Iterator[dict]:
    lines = index.lines(file)
    end = min(len(lines), end)
    for line in range(max(1, start), end + 1, PIECE_LINES):
        last = min(end, line + PIECE_LINES - 1)
        yield {"file": file, "line": line, "end": last, "text": "\n".join(lines[line - 1 : last])}


def location(span: Span) -> dict:
    return {"file": span.file, "start": span.start, "end": span.end}


def binding_data(binding: Binding | None) -> dict:
    if binding is None:
        return {"status": "unknown", "reason": "no binding was supplied"}
    return {
        "status": binding.status,
        "reason": binding.reason,
        "target": location(binding.target) if binding.target else None,
    }


def subjects(index: CodeIndex, op: Operation) -> tuple[Span, ...]:
    if op.op == "callees":
        span = index.enclosing_symbol(op.file, op.line)
        if span is None:
            raise ValueError(f"no enclosing symbol at {op.file}:{op.line}")
        return (span,)
    found = index.find_definition(op.name)
    if not op.file:
        return found
    own = tuple(span for span in found if span.file == op.file)
    if own:
        return own
    binding = index.binding_of(op.file, op.line, op.name)
    return (binding.target,) if binding.target else ()


def relation_rows(index: CodeIndex, op: Operation) -> Iterator[dict]:
    if op.op == "def":
        for span in subjects(index, op):
            yield {**location(span), "name": span.name}
        return
    if op.op == "callees":
        for span in subjects(index, op):
            for edge in index.callee_edges(span):
                yield {
                    "file": span.file,
                    "line": edge.line,
                    "name": edge.name,
                    "binding": binding_data(edge.binding),
                }
        return
    # Without an owner every same-named use remains a candidate. An owner filters proven other targets.
    owners = subjects(index, op) if op.file else ()
    if op.file and not owners:
        raise ValueError(f"no definition bound to {op.name} in {op.file}")
    sites = (
        tuple(site for owner in owners for site in index.callers_of(owner))
        if owners
        else index.find_callers(op.name)
    )
    for site in dict.fromkeys(sites):
        if not owners or any(names_exactly(site.binding, owner) for owner in owners):
            yield {
                "file": site.file,
                "line": site.line,
                "kind": "call",
                "name": op.name,
                "binding": binding_data(site.binding),
            }
    if op.op == "refs":
        references = (
            tuple(ref for owner in owners for ref in index.references_to(owner))
            if owners
            else index.find_references(op.name)
        )
        for reference in dict.fromkeys(references):
            if not owners or any(names_exactly(reference.binding, owner) for owner in owners):
                yield {
                    "file": reference.file,
                    "line": reference.line,
                    "kind": reference.role,
                    "name": reference.name,
                    "binding": binding_data(reference.binding),
                }


def text_rows(index: CodeIndex, op: Operation) -> Iterator[dict]:
    """Exact source, including context, rather than ripgrep's abbreviated long-line preview."""
    pattern = re.compile(op.query if op.regex else re.escape(op.query))
    for file in selected_files(index, op):
        lines = index.lines(file)
        last = 0
        for line, text in enumerate(lines, 1):
            if pattern.search(text):
                start, end = max(last + 1, line - op.window), min(len(lines), line + op.window)
                yield from source_rows(index, file, start, end)
                last = max(last, end)


def name_rows(index: CodeIndex, op: Operation) -> Iterator[dict]:
    variants = spelling_variants(op.name)
    hits = index.search_texts(variants)
    allowed = frozenset(selected_files(index, op))
    counts: Counter[str] = Counter()
    for variant in variants:
        counts[variant] = sum(hit.file in allowed for hit in hits[variant])
    yield from (
        {"name": name, "hits": count} for name, count in sorted(counts.items(), key=lambda x: x[1]) if count
    )


def test_rows(index: CodeIndex, op: Operation) -> Iterator[dict]:
    """Candidate tests by sibling stem, importing file, or exact supplied name. No coverage claim."""
    imports = frozenset(index.dependents(op.file)) if op.file else frozenset()
    stem = PurePosixPath(op.file).stem if op.file else ""
    hits = {hit.file for hit in index.search_text(op.name, whole_word=True)} if op.name else set()
    for file in selected_files(index, Operation("outline", scopes=op.scopes)):
        if not is_test_file(file):
            continue
        test_stem = re.sub(r"^(?:test_)", "", PurePosixPath(file).stem)
        test_stem = re.sub(r"(?:_test|\.test|\.spec)$", "", test_stem)
        reasons = [
            reason
            for condition, reason in (
                (stem and stem == test_stem, "sibling name"),
                (file in imports, "imports source"),
                (file in hits, "mentions name"),
            )
            if condition
        ]
        if reasons:
            yield {"file": file, "reasons": reasons}


def _rank_inputs(index, op, judge, checks, shared):
    if judge is None:
        raise ValueError("rank requires a caller-supplied Judge; no model client is created automatically")
    scoped = judge.scope()
    if op.cursor.row or op.cursor.character:
        if judge.store is None:
            raise ValueError("rank continuation requires the original Judge's answer store")
        replay = ReplayOnlyClient()
        replay.model = judge.client.model
        replay.replays_any_model = getattr(judge.client, "replays_any_model", False)
        scoped.client = replay
    scoped.items_per_request = 16
    scoped.max_concurrency = 1
    checks = tuple(checks) if checks is not None else (match_check("query"),)
    state = dict(shared) if shared is not None else {"targets": {"query": op.query}}
    items = []
    for span in dict.fromkeys(op.candidates):
        if span.start < 1 or span.end < span.start or span.end > len(index.lines(span.file)):
            raise ValueError(f"candidate range is outside its file: {span.key}")
        source = index.read_slice(span, "rank")
        items.append(
            {"id": span.key, "file": span.file, "start": span.start, "end": span.end, "code": source.text}
        )
    return scoped, checks, items, state


def _rank_output(items, checks, answers, refusals, stop):
    ranked = {
        item["id"]: {"candidate": item["id"], "answers": {}, "score": None, "status": "not_judged"}
        for item in items
    }
    for check, answer in answers:
        row = ranked[answer.item["id"]]
        row["answers"][check] = {"probability": answer.probability, "request_sha256": answer.request_sha256}
        row["score"] = max(row["score"] or 0.0, answer.probability)
        row["status"] = "judged" if len(row["answers"]) == len(checks) else "partial"
    refused = {refusal.item["id"]: str(refusal.error) for refusal in refusals}
    for row in ranked.values():
        if row["status"] != "judged":
            row["reason"] = refused.get(row["candidate"], stop or "not every requested check was answered")
    return sorted(ranked.values(), key=lambda row: (row["score"] is None, -(row["score"] or 0.0)))


def rank_rows(index, op, judge, checks, shared) -> Iterator[dict]:
    scoped, checks, items, state = _rank_inputs(index, op, judge, checks, shared)
    answers, refusals, stop = [], [], ""
    try:
        for answer in scoped.iter_check_every(checks, items, state, keep_order=True, refusals=refusals):
            answers.append(answer)
    except (CallCapReachedError, MissingAnswerError) as error:
        stop = str(error)
    yield from _rank_output(items, checks, answers, refusals, stop)


async def rank_rows_async(index, op, judge, checks, shared) -> list[dict]:
    import asyncio

    scoped, checks, items, state = await asyncio.to_thread(_rank_inputs, index, op, judge, checks, shared)
    answers, refusals, stop = [], [], ""
    try:
        async for answer in scoped.iter_check_every_async(
            checks, items, state, keep_order=True, refusals=refusals
        ):
            answers.append(answer)
    except (CallCapReachedError, MissingAnswerError) as error:
        stop = str(error)
    return _rank_output(items, checks, answers, refusals, stop)
