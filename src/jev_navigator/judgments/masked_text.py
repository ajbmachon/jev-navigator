"""The one owner of the text a request may show. ``masked_lines`` masks a whole file once, so every
slice, window, preview, excerpt or line cut takes masked text: a cut can no longer split a secret from
the key that marks it. The values it hides join the caller's ``HiddenValues``, so a request hides their
copies in other files' slices too."""

from __future__ import annotations

from collections.abc import Sequence

from .secrets import MASK, HiddenValues, Masker, copies_of


def masked_lines(
    lines: Sequence[str], file: str | None, masker: Masker, hidden: HiddenValues
) -> tuple[str, ...]:
    """``lines`` with every value ``masker`` finds anywhere in them masked wherever it stands, read as
    ``file``, and those values added to ``hidden``. The line count stays: a value spanning lines
    becomes ``MASK`` on its first line and leaves the lines it covered empty up to what follows it
    on its last line."""
    text = "\n".join(lines)
    values = frozenset(masker.masked_values(text, file)) - {MASK}
    hidden.add(values)
    return tuple(_masked(text, copies_of(values).spans(text)).split("\n"))


def _masked(text: str, spans: list[tuple[int, int]]) -> str:
    parts, position = [], 0
    for start, end in spans:
        parts += [text[position:start], MASK + "\n" * text.count("\n", start, end)]
        position = end
    return "".join([*parts, text[position:]])
