"""How a neighbour's signature appears in a CLI run folder without ``--keep-requests``.

A signature (built in ``directives.places``) starts with its location, ``path:line`` or
``path:start-end``, and quotes one line of that file between backticks. Run files keep the location
and the symbol name and drop the quoted code. Requests never carry a stored signature: each opening
lists its neighbours afresh, so a resumed frontier keeps only this label.
"""

from __future__ import annotations

import re

from .index.code_index import CodeIndex

_QUOTE = re.compile(r" ?`[^`]*`")


def unquoted(signature: str) -> str:
    """``path:line (relation)``: the signature without its quoted code or a symbol name."""
    return _QUOTE.sub("", signature, count=1)


def location_label(index: CodeIndex, signature: str) -> str:
    """``path:line name (relation)``. A signature without a quote, such as one already labelled,
    stays as it is."""
    if not _QUOTE.search(signature):
        return signature
    location = signature.split(" ", 1)[0]
    name = _symbol_name(index, location)
    label = unquoted(signature)
    return label.replace(location, f"{location} {name}", 1) if name else label


def _symbol_name(index: CodeIndex, location: str) -> str:
    file, _, lines = location.rpartition(":")
    first = lines.split("-", 1)[0]
    if file not in index.files or not first.isdigit():
        return ""
    symbol = index.enclosing_symbol(file, int(first))
    return symbol.name if symbol is not None and symbol.name else ""
