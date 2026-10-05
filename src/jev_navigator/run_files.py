"""How places and relations appear in a CLI run folder without ``--keep-requests``.

A place is labelled ``path:line name`` from structured fields only, the place key (``path:start-end``
or ``path:line~radius``) and the enclosing symbol the index knows, never from a signature, whose
quoted code line may itself contain backticks. Requests never carry a stored signature: each opening
lists its neighbours afresh, so a resumed frontier keeps only this label.

Relations and ``reached_by`` texts go through ``judgments.relations.without_quoted_code``, the one
owner of how a key mention is shown without its literal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .index.code_index import CodeIndex
from .judgments.journal import message_fields
from .judgments.relations import without_quoted_code

_LINE_RANGE = re.compile(r"[-~]")
_NEIGHBOUR_LISTS = ("could_contain", "not_inspected", "not_opened")


def place_location(place_key: str) -> str:
    """``path:line``: the first line a place key names."""
    file, _, lines = place_key.rpartition(":")
    return f"{file}:{_LINE_RANGE.split(lines, maxsplit=1)[0]}"


def place_label(index: CodeIndex, place_key: str) -> str:
    """``path:line name``, or ``path:line`` where no symbol encloses that line."""
    location = place_location(place_key)
    file, _, line = location.rpartition(":")
    symbol = index.enclosing_symbol(file, int(line)) if file in index.files and line.isdigit() else None
    return f"{location} {symbol.name}" if symbol is not None and symbol.name else location


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
