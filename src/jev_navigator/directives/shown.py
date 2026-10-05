"""The code a request shows: long lines cut, and a slice cut at a line boundary only when its requests
would not fit the box of the client's input limits, so a function that fits goes whole. Every cut
stays visible."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import replace

from ..index.spans import CodeSlice

MAX_LINE_CHARS = 240
LINE_CUT_MARK = " [line cut]"


def cut_long_line(line: str, max_chars: int = MAX_LINE_CHARS) -> str:
    return line if len(line) <= max_chars else f"{line[:max_chars]}{LINE_CUT_MARK}"


def shown_slice(
    code: CodeSlice, fits: Callable[[CodeSlice], bool], max_line_chars: int = MAX_LINE_CHARS
) -> CodeSlice | None:
    """``code`` with each line cut at ``max_line_chars``: whole when ``fits`` accepts it, else its
    longest start that ``fits`` accepts, whose span ends at the last shown line and whose text ends
    with a note naming what was left out. Returns None when not even the first line fits, so the
    caller retains the source as uninspected. ``fits`` accepts every shorter start of a slice it
    accepts, as a size box does."""
    lines = [cut_long_line(line, max_line_chars) for line in code.text.split("\n")]
    whole = _first_lines(code, lines, len(lines))
    if fits(whole):
        return whole
    counts = range(1, len(lines))
    kept = bisect_left(counts, True, key=lambda count: not fits(_first_lines(code, lines, count)))
    return _first_lines(code, lines, kept) if kept else None


def _first_lines(code: CodeSlice, lines: list[str], count: int) -> CodeSlice:
    text = "\n".join(lines[:count])
    if count < len(lines):
        text += f"\n[cut after {count} of {len(lines)} lines to fit the request size limit]"
    return replace(code, span=replace(code.span, end=code.span.start + count - 1), text=text)
