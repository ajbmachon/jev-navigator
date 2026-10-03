"""Persist a find evidence pack's frontier so a later CLI invocation can continue it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .directives.find_code import FindResult, NotInspected, Outcome, QueueTier, Visit
from .directives.places import Place, place_relationship
from .index.bindings import Binding
from .index.code_index import CodeIndex
from .index.spans import Span
from .judgments.judge import CheckResult
from .judgments.thresholds import NoulVerdict

STATE_VERSION = 1


@dataclass(frozen=True)
class SavedSearch:
    result: FindResult | None
    completed: tuple[CheckResult, ...] | None = None
    check_id: str | None = None


def scope_identity(index: CodeIndex) -> tuple[str, dict[str, str]]:
    """Identity of scoped bytes, with explicit markers for files that cannot be read."""
    digest = hashlib.sha256()
    available = set(index.available_files)
    unavailable = index.unavailable_files
    for file in sorted(index.files):
        if file in available:
            try:
                file_hash = index.read_slice(Span(file, 1, 1)).file_sha256
            except OSError as error:
                unavailable[file] = f"{type(error).__name__}: {error}"
                file_hash = ""
            if file_hash:
                unavailable.pop(file, None)
            else:
                unavailable.setdefault(
                    file, index.unavailable_files.get(file, "unavailable during resume snapshot")
                )
        else:
            file_hash = ""
            unavailable[file] = index.unavailable_files.get(file, "unavailable during resume snapshot")
        digest.update(file.encode())
        digest.update(b"\0")
        identity = f"unavailable:{unavailable[file]}" if file in unavailable else f"sha256:{file_hash}"
        digest.update(identity.encode())
        digest.update(b"\0")
    return digest.hexdigest(), unavailable


def save_resume(
    path: Path,
    index: CodeIndex,
    result: FindResult,
    *,
    entry_pending: bool,
    completed: tuple[CheckResult, ...] | None = None,
    check_id: str | None = None,
) -> dict[str, str]:
    """Save only the state find_code needs to reopen its frontier; manifest owns past evidence."""
    digest, unavailable = scope_identity(index)
    state = {
        "version": STATE_VERSION,
        "scope_digest": digest,
        "unavailable_files": unavailable,
        "stage": "enumeration" if completed is not None else "entry" if entry_pending else "navigation",
        "check_id": check_id,
        "completed": [asdict(answer) for answer in completed] if completed is not None else None,
        "result": None if entry_pending else _result_record(result),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return unavailable


def load_resume(path: Path, index: CodeIndex) -> SavedSearch:
    """Return a fresh-index frontier, or None when entry selection must be replayed."""
    state = json.loads(path.read_text())
    if state.get("version") != STATE_VERSION:
        raise ValueError("unsupported find resume state version")
    if state.get("scope_digest") != scope_identity(index)[0]:
        raise ValueError("repository source or scope changed since the evidence pack")
    if state.get("stage") == "entry":
        return SavedSearch(None)
    if state.get("stage") not in ("navigation", "enumeration"):
        raise ValueError("invalid find resume stage")
    record = state["result"]
    result = FindResult(
        outcome=Outcome(record.get("outcome", Outcome.BUDGET)),
        found=tuple(_read_visit(item, index) for item in record.get("found", [])),
        searched=tuple(_read_visit(item, index) for item in record["searched"]),
        unsure=tuple(_read_visit(item, index) for item in record["unsure"]),
        not_inspected=tuple(_read_frontier(item, index) for item in record["not_inspected"]),
        steps=record["steps"],
        calls=record["calls"],
        visited=frozenset(record["visited"]),
        judged_code=frozenset(record["judged_code"]),
        starts=tuple(_read_visit(item, index) for item in record["starts"]),
    )
    completed = None
    if state["stage"] == "enumeration":
        completed = tuple(
            CheckResult(**{**item, "verdict": NoulVerdict(item["verdict"])}) for item in state["completed"]
        )
    return SavedSearch(result, completed, state.get("check_id"))


def _result_record(result: FindResult) -> dict:
    return {
        "outcome": result.outcome,
        "found": [_visit_record(item) for item in result.found],
        "steps": result.steps,
        "calls": result.calls,
        "visited": sorted(result.visited),
        "judged_code": sorted(result.judged_code),
        "searched": [_visit_record(item) for item in result.searched],
        "unsure": [_visit_record(item) for item in result.unsure],
        "starts": [_visit_record(item) for item in result.starts],
        "not_inspected": [_frontier_record(item) for item in result.not_inspected],
    }


def _visit_record(visit: Visit) -> dict:
    """A location, never code text: Resume re-reads the code from the unchanged scope."""
    return {
        "place_key": visit.place_key,
        "code": {"span": asdict(visit.code.span), "origin": visit.code.origin},
        "path": list(visit.path),
        "probability": visit.probability,
        "verdict": visit.verdict,
    }


def _read_visit(record: dict, index: CodeIndex) -> Visit:
    code = record["code"]
    return Visit(
        record["place_key"],
        index.read_slice(Span(**code["span"]), origin=code["origin"]),
        tuple(record["path"]),
        record["probability"],
        NoulVerdict(record["verdict"]),
    )


def _frontier_record(entry: NotInspected) -> dict:
    code = entry.place.open()
    return {
        "place_key": entry.place_key,
        "signature": entry.signature,
        "kind": entry.place.kind,
        "span": asdict(code.span),
        "origin": code.origin,
        "reason": entry.reason,
        "priority": entry.priority,
        "depth": entry.depth,
        "path": list(entry.path),
        "tier": entry.tier.value,
        "relationship": place_relationship(entry.place),
    }


def _read_frontier(record: dict, index: CodeIndex) -> NotInspected:
    span = Span(**record["span"])
    origin = record["origin"]
    relationship = record.get("relationship") or {}
    binding_record = relationship.get("binding")
    binding = None
    if binding_record is not None:
        target = binding_record.get("target")
        binding = Binding(
            binding_record["status"], binding_record["reason"], Span(**target) if target is not None else None
        )
    place = Place(
        record["place_key"],
        record["kind"],
        record["signature"],
        lambda: index.read_slice(span, origin=origin),
        relationship.get("relation"),
        binding,
        relationship.get("move"),
    )
    return NotInspected(
        record["place_key"],
        record["signature"],
        record["reason"],
        record["priority"],
        record["depth"],
        tuple(record["path"]),
        place,
        QueueTier(record["tier"]),
    )
