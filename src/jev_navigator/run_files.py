"""How places and relations appear in a CLI run folder without ``--keep-requests``.

A place is labelled ``path:line name`` from structured fields only, the place key (``path:start-end``
or ``path:line~radius``) and the enclosing symbol the index knows, never from a signature, whose
quoted code line may itself contain backticks. Requests never carry a stored signature: each opening
lists its neighbours afresh, so a resumed frontier keeps only this label.

Relations and ``reached_by`` texts go through ``judgments.relations.without_quoted_code``, the one
owner of how a key mention is shown without its literal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .index.code_index import CodeIndex
from .judgments.journal import error_text_digested, keeps_request_text, message_fields
from .judgments.relations import without_quoted_code

_LINE_RANGE = re.compile(r"[-~]")
_LABEL_NAME = re.compile(r"(?: [^\s`]+)?")
_NEIGHBOUR_LISTS = ("could_contain", "not_inspected", "not_opened")


def require_kept_request_text(run_file: Path, keep_requests: bool) -> None:
    """A run continues ``run_file``, an earlier journal or answer store, without ``keep_requests``
    only when the file kept no request's text, so a folder written without the flag holds no code."""
    if not keep_requests and run_file.is_file() and keeps_request_text(run_file):
        raise ValueError(
            f"{run_file} keeps the text of its requests, so a run that continues it must keep it too: "
            'add --keep-requests (JSON "keep_requests": true)'
        )


def place_location(place_key: str) -> str:
    """``path:line``: the first line a place key names."""
    file, _, lines = place_key.rpartition(":")
    return f"{file}:{_LINE_RANGE.split(lines, maxsplit=1)[0]}"


def place_label(index: CodeIndex, place_key: str) -> str:
    """``path:line name``, or ``path:line`` where no symbol encloses that line or the index holds no
    facts of the file yet. A label shows only facts already in memory, so writing the journal or the
    evidence pack starts no parse, which could fail."""
    location = place_location(place_key)
    file, _, line = location.rpartition(":")
    symbol = index.known_enclosing_symbol(file, int(line)) if line.isdigit() else None
    return f"{location} {symbol.name}" if symbol is not None and symbol.name else location


def is_place_label(place_key: str, text: str) -> bool:
    """Whether ``text`` has the form ``place_label`` gives ``place_key``: its location, then optionally
    one name. A signature, which quotes a line of code, does not."""
    location = place_location(place_key)
    return text.startswith(location) and _LABEL_NAME.fullmatch(text, len(location)) is not None


def carried_over_journal_line(line: str, *, keep_error_text: bool) -> str:
    """A line of an earlier pack's journal as this pack keeps it. A history step neighbour whose
    signature is not a label, the code signature an older pack wrote, shows its location instead.
    Without ``keep_error_text`` every error message and error body reads as this pack writes them,
    however the earlier pack kept them. A line that is not a JSON record stays as written."""
    try:
        record = json.loads(line)
    except ValueError:
        return line
    carried = _carried_record(record, keep_error_text)
    return line if carried == record else json.dumps(carried, sort_keys=True) + "\n"


def _carried_record(record: dict, keep_error_text: bool) -> dict:
    if record["kind"] != "history_step":
        return record if keep_error_text else error_text_digested(record)
    step = _with_labelled_neighbours(record["step"])
    return {**record, "step": step if keep_error_text else failure_digested(step)}


def _with_labelled_neighbours(step: Mapping) -> Mapping:
    judgments = step.get("judgments", {})
    offered = judgments.get("could_contain", [])
    if all(is_place_label(entry["place"], entry["signature"]) for entry in offered):
        return step
    labelled = [
        entry
        if is_place_label(entry["place"], entry["signature"])
        else {**entry, "signature": place_location(entry["place"])}
        for entry in offered
    ]
    return {**step, "judgments": {**judgments, "could_contain": labelled}}


@dataclass(frozen=True)
class PlaceLabels:
    """The labels one run writes. A place an earlier save labelled keeps that label: the earlier save
    owns it, and a resumed run may never parse the place's file. Any other place gets ``place_label``."""

    index: CodeIndex
    saved: Mapping[str, str] = field(default_factory=dict)

    def __call__(self, place_key: str) -> str:
        saved = self.saved.get(place_key)
        return saved if saved is not None else place_label(self.index, place_key)


def relation_shown(relation: str, place_key: str) -> str:
    """A place's relation as a run file keeps it, located at the line its key names (a window's
    mention line, a function's first line), so one place shows one line everywhere."""
    file, _, line = place_location(place_key).rpartition(":")
    return without_quoted_code(relation, file, int(line))


def relationship_shown(relationship: Mapping | None, place_key: str) -> Mapping | None:
    """A place's relationship with its relation as a run file keeps it."""
    if not relationship or "relation" not in relationship:
        return relationship
    return {**relationship, "relation": relation_shown(relationship["relation"], place_key)}


def source_shown(source: Mapping, place_key: str) -> dict:
    """The code source of the place ``place_key`` with its ``reached_by`` as a run file keeps it."""
    if "reached_by" not in source:
        return dict(source)
    return {**source, "reached_by": relation_shown(source["reached_by"], place_key)}


def step_shown(step: Mapping) -> dict:
    """A history step with every neighbour relation and fetched ``reached_by`` as a run file keeps it."""
    judgments = {
        name: [_entry_shown(entry) for entry in value] if name in _NEIGHBOUR_LISTS else value
        for name, value in step.get("judgments", {}).items()
    }
    fetched = [source_shown(source, step["arguments"]["place"]) for source in step.get("fetched", [])]
    return {**step, "judgments": judgments, "fetched": fetched}


def failure_digested(step: Mapping) -> Mapping:
    """A history step whose failure keeps its message only as a digest (``--no-error-text``). A step
    read back from a saved pack may already hold the digest, and stays as it is."""
    failure = step.get("judgments", {}).get("failure", {})
    if "message" not in failure:
        return step
    digest = {"type": failure["type"], **message_fields(failure["message"], keep_text=False)}
    return {**step, "judgments": {**step["judgments"], "failure": digest}}


def _entry_shown(entry: Mapping) -> Mapping:
    if not entry.get("relationship"):
        return entry
    return {**entry, "relationship": relationship_shown(entry["relationship"], entry["place"])}
