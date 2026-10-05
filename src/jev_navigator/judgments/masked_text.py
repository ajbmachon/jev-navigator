"""The one owner of the text a request may show. ``masked_lines`` masks a whole file once, so every
slice, window, preview, excerpt or line cut takes masked text: a cut can no longer split a secret from
the key that marks it."""

from __future__ import annotations

from collections.abc import Sequence

from .secrets import MASK, Masker, copies_of


def masked_lines(lines: Sequence[str], file: str | None, masker: Masker) -> tuple[str, ...]:
    """``lines`` with every value ``masker`` finds anywhere in them masked wherever it stands, read as
    ``file``. The line count stays: a value spanning lines becomes ``MASK`` on its first line and
    leaves the lines it covered empty up to what follows it on its last line."""
    text = "\n".join(lines)
    values = frozenset(masker.masked_values(text, file)) - {MASK}
    return tuple(copies_of(values).sub(text, keep_lines=True).split("\n"))
