"""How places and relations appear in a CLI run folder without ``--keep-requests``.

A place is labelled ``path:line name`` from structured fields only, the place key (``path:start-end``
or ``path:line~radius``) and the enclosing symbol the index knows, never from a signature, whose
quoted code line may itself contain backticks. Requests never carry a stored signature: each opening
lists its neighbours afresh, so a resumed frontier keeps only this label.

A relation listed by ``KEY_MENTION_MOVE`` quotes a string literal from the code; run files show it as
``mentions a key (path:line)``, chosen by the move name, never by reading the relation text. Every
other relation names only symbols and stays readable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .index.code_index import CodeIndex

KEY_MENTION_MOVE = "keys_mentioned"
_FRONTIER_LISTS = ("could_contain", "not_inspected", "not_opened")

_LINE_RANGE = re.compile(r"[-~]")


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


def shown_relation(move: str | None, relation: str, place_key: str) -> str:
    """The relation a run file may store for a place the move listed."""
    return f"mentions a key ({place_location(place_key)})" if move == KEY_MENTION_MOVE else relation


def step_without_key_mentions(step: Mapping, moves: Mapping[str, str | None]) -> dict:
    """A history step whose key-mention relations, in neighbour lists and in the opened place's
    ``reached_by``, are shown by location; ``moves`` names the move that listed each opened place."""
    judgments = {
        name: [_entry_without_key_mention(entry) for entry in value] if name in _FRONTIER_LISTS else value
        for name, value in step.get("judgments", {}).items()
    }
    place = (step.get("arguments") or {}).get("place")
    fetched = [
        {**source, "reached_by": shown_relation(moves.get(place), source.get("reached_by", ""), place)}
        if place is not None and "reached_by" in source
        else source
        for source in step.get("fetched", [])
    ]
    return {**step, "judgments": judgments, "fetched": fetched}


def shown_relationship(relationship: Mapping | None, place_key: str) -> Mapping | None:
    """A place's relationship with a key-mention relation shown by location."""
    if not relationship or relationship.get("move") != KEY_MENTION_MOVE or "relation" not in relationship:
        return relationship
    return {**relationship, "relation": shown_relation(KEY_MENTION_MOVE, relationship["relation"], place_key)}


def _entry_without_key_mention(entry: Mapping) -> Mapping:
    shown = shown_relationship(entry.get("relationship"), entry["place"])
    return entry if shown is entry.get("relationship") else {**entry, "relationship": shown}
