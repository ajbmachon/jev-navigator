"""The grammar a place's signature is written in, read back without the index: ``located_file`` parses
what the signature builders in ``places`` write, so masking can tell which file a candidate shows."""

from __future__ import annotations

import re

from .shown import LINE_CUT_MARK

_PLACE_LINES = re.compile(r":\d+(?:-\d+)? ")
_WINDOW_LINE = re.compile(r"line \d+ ")


def located_file(signature: str) -> str | None:
    """The file a place's signature names, parsed by the grammar the signature builders write: the
    file, ``:lines`` and a space at each separator, then the place's text (see ``_is_place_text``).
    None when no split fits, and when more than one does (a path or a quoted code line that holds a
    separator itself), so a caller that needs the file reads such a signature as config."""
    files: list[str] = []
    for separator in _PLACE_LINES.finditer(signature):
        if separator.start() and _is_place_text(signature[separator.end() :]):
            files.append(signature[: separator.start()])
            if len(files) > 1:
                return None
    return files[0] if files else None


def _is_place_text(text: str) -> bool:
    """A place's text: an optional ``line N `` then quoted code, ending with the closing quote or a
    parenthesised relation, or anywhere when ``cut_long_line`` cut it."""
    body = text.removesuffix(LINE_CUT_MARK)
    window_line = _WINDOW_LINE.match(body)
    quoted = body[window_line.end() :] if window_line else body
    if not quoted.startswith("`"):
        return False
    return body != text or quoted.endswith("`") or ("` (" in quoted and quoted.endswith(")"))
