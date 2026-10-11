"""A static configuration: the other lines that use the name a value on one line is assigned to.

The anchor is a file, a line and the column where a value starts on that line. The name is the
variable, key or argument the text before the column assigns the value to (``bound_name``). Its uses
are the scope's whole-word text hits of that name outside the lines the caller already shows around
the anchor, ordered the anchor's file first, then other files, then test files (``is_test_file``),
each by path and line. A name assigned on an indented line of a parsed code file belongs to its block,
and a name hit more than ``max_scope_hits`` times is too common to follow across files; both are
looked up in the anchor's file only. No model call; the caller masks what it shows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.scope import is_test_file
from ..index.spans import TextHit

_BINDING = re.compile(
    r"""([A-Za-z_][A-Za-z0-9_\-]*)["'\]]?\s*(?:=>|:=|=|:)\s*(?:[A-Za-z]+\s+)?[\["'`({fbr]*\s*$"""
)
MIN_NAME_CHARS = 3


def bound_name(before: str) -> str | None:
    """The name ``before``, the text of a line up to a value, assigns the value to: ``API_KEY = "``,
    ``"api_key": "``, ``token: `` or ``password=`` give that name; a name under ``MIN_NAME_CHARS``
    characters, or text that assigns nothing, gives None."""
    match = _BINDING.search(before)
    return match.group(1) if match and len(match.group(1)) >= MIN_NAME_CHARS else None


@dataclass(frozen=True)
class ValueUses:
    """The bound name (None when the line binds the value to no name), how many lines use it, and the
    first ``max_uses`` of them in order."""

    name: str | None
    found: int
    hits: tuple[TextHit, ...]


@dataclass(frozen=True)
class ValueUsesConfiguration:
    """How many uses to return, when a name is too common to follow across files, and how many lines
    either side of the anchor the caller shows itself."""

    max_uses: int = 10
    max_scope_hits: int = 50
    shown_radius: int = 12

    def find(self, index: CodeIndex, file: str, line: int, column: int) -> ValueUses:
        """The uses of the name the value starting at ``column`` of ``file``'s ``line`` is assigned to."""
        lines = index.lines(file)
        text = lines[line - 1] if 0 < line <= len(lines) else ""
        name = bound_name(text[:column])
        if name is None:
            return ValueUses(None, 0, ())
        hits = [
            hit
            for hit in index.search_text(name, whole_word=True)
            if not (hit.file == file and abs(hit.line - line) <= self.shown_radius)
        ]
        local = language_of(file) is not None and text[:1].isspace()
        if local or len(hits) > self.max_scope_hits:
            hits = [hit for hit in hits if hit.file == file]
        ordered = sorted(hits, key=lambda hit: (_place(hit, file), hit.file, hit.line))
        return ValueUses(name, len(ordered), tuple(ordered[: self.max_uses]))


def _place(hit: TextHit, file: str) -> int:
    return 0 if hit.file == file else 2 if is_test_file(hit.file) else 1


VALUE_USES = ValueUsesConfiguration()
