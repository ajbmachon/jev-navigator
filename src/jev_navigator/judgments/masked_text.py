"""The one owner of the text a request may show. ``masked_lines`` masks a whole file once, so every
slice, window, preview, excerpt or line cut takes masked text: a cut can no longer split a secret from
the key that marks it. A value whose key is in another file is left to the request's own copy masking."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .secrets import BY_CONTENT_MIN_CHARS, MASK, Masker


def masked_lines(lines: Sequence[str], file: str | None, masker: Masker) -> tuple[str, ...]:
    """``lines`` with every value ``masker`` finds anywhere in them masked wherever it stands, read as
    ``file``. The line count stays: a value spanning lines becomes ``MASK`` on its first line and
    leaves the lines it covered empty up to what follows it on its last line."""
    text = "\n".join(lines)
    return tuple(_masked(text, _secret_spans(text, masker, file)).split("\n"))


def _secret_spans(text: str, masker: Masker, file: str | None) -> list[tuple[int, int]]:
    values = sorted(set(masker.masked_values(text, file)) - {MASK}, key=len, reverse=True)
    if not values:
        return []
    copies = re.compile("|".join(_copy_pattern(value) for value in values))
    return [match.span() for match in copies.finditer(text)]


def _copy_pattern(value: str) -> str:
    """Where the masker hides a value's copies: anywhere for ``BY_CONTENT_MIN_CHARS`` or more
    characters, as a whole word for a shorter one."""
    if len(value) >= BY_CONTENT_MIN_CHARS:
        return re.escape(value)
    return rf"(?<![\w$]){re.escape(value)}(?![\w$])"


def _masked(text: str, spans: list[tuple[int, int]]) -> str:
    parts, position = [], 0
    for start, end in spans:
        parts += [text[position:start], MASK + "\n" * text.count("\n", start, end)]
        position = end
    return "".join([*parts, text[position:]])
