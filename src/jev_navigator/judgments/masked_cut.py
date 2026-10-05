"""The one way a region of a source is cut for a request. A cut can split a secret from the key that
marks it, so the masker reads the whole source, and the cut only blanks what it found there."""

from __future__ import annotations

import re

from .secrets import MASK, Masker

_PIECE_CHARACTER = re.compile(r"""[^\s"']""")


def masked_cut(
    source: str,
    start: int,
    end: int,
    masker: Masker | None,
    file: str | None = None,
    keep: tuple[int, int] | None = None,
) -> str:
    """``source[start:end]`` as it may go into a request. An edge that splits a run of characters other
    than whitespace and quotes moves inward past that run, never into ``keep``; then every part of the
    range that holds a value ``masker`` finds anywhere in ``source`` (read as ``file``) is masked.
    ``start``, ``end`` and ``keep`` are offsets into ``source``."""
    inner_start, inner_end = keep or (end, start)
    start = _past_split_piece(source, start, min(inner_start, end))
    end = _before_split_piece(source, end, max(inner_end, start))
    secrets = _secret_spans(source, masker, file) if masker else []
    return _masked_range(source, start, end, secrets)


def _past_split_piece(source: str, start: int, limit: int) -> int:
    if not _splits_piece(source, start):
        return start
    while start < limit and _is_piece_character(source[start]):
        start += 1
    return start


def _before_split_piece(source: str, end: int, limit: int) -> int:
    if not _splits_piece(source, end):
        return end
    while end > limit and _is_piece_character(source[end - 1]):
        end -= 1
    return end


def _splits_piece(source: str, index: int) -> bool:
    return (
        0 < index < len(source)
        and _is_piece_character(source[index - 1])
        and _is_piece_character(source[index])
    )


def _is_piece_character(character: str) -> bool:
    return bool(_PIECE_CHARACTER.match(character))


def _secret_spans(source: str, masker: Masker, file: str | None) -> list[tuple[int, int]]:
    values = sorted(set(masker.masked_values(source, file)) - {MASK}, key=len, reverse=True)
    if not values:
        return []
    copies = re.compile("|".join(re.escape(value) for value in values))
    return [match.span() for match in copies.finditer(source)]


def _masked_range(source: str, start: int, end: int, secrets: list[tuple[int, int]]) -> str:
    parts, position = [], start
    for secret_start, secret_end in secrets:
        if secret_end <= position or secret_start >= end:
            continue
        parts += [source[position : max(position, secret_start)], MASK]
        position = min(secret_end, end)
    return "".join([*parts, source[position:end]])
