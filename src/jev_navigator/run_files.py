"""How a place appears in a CLI run folder without ``--keep-requests``: ``path:line name``.

The label comes from structured fields only, the place key (``path:start-end`` or
``path:line~radius``) and the enclosing symbol the index knows, never from a signature, whose quoted
code line may itself contain backticks. Requests never carry a stored signature: each opening lists
its neighbours afresh, so a resumed frontier keeps only this label.
"""

from __future__ import annotations

import re

from .index.code_index import CodeIndex

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
