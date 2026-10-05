"""Locations and pieces of code that every layer passes around. Lines are 1-based and inclusive."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bindings import Binding


@dataclass(frozen=True, order=True)
class Span:
    file: str
    start: int
    end: int
    name: str = ""

    @property
    def key(self) -> str:
        return f"{self.file}:{self.start}-{self.end}"

    def contains(self, line: int) -> bool:
        return self.start <= line <= self.end

    def size(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: Span) -> bool:
        return self.file == other.file and self.start <= other.end and other.start <= self.end


def holder_of(symbols: Iterable[Span], symbol: Span) -> Span | None:
    """The smallest of ``symbols`` whose lines contain ``symbol``'s and fill more of them: a method's
    class, a nested function's outer function. Lines are all a span knows, so a symbol on the same
    lines as another (a method on a one-line class) has no holder."""
    holding = [
        other
        for other in symbols
        if other != symbol
        and other.size() > symbol.size()
        and other.contains(symbol.start)
        and other.contains(symbol.end)
    ]
    return min(holding, key=Span.size, default=None)


@dataclass(frozen=True)
class CodeSlice:
    """The text of a span and its source: ``origin`` says how it was reached (the operation or the
    relation followed), ``commit`` the revision it was read at. A commit ending in ``+worktree``
    means the file had uncommitted changes when read."""

    span: Span
    text: str
    origin: str = ""
    commit: str = ""
    file_sha256: str = ""

    def source(self) -> dict:
        return {
            "file": self.span.file,
            "lines": [self.span.start, self.span.end],
            "commit": self.commit,
            "file_sha256": self.file_sha256,
            "reached_by": self.origin,
        }

    @property
    def key(self) -> str:
        return self.span.key


@dataclass(frozen=True)
class CallSite:
    """A call to a name: where it is, the function it sits in (None at module level), and how sure
    the index is that it reaches the definition (see ``bindings``)."""

    file: str
    line: int
    caller: Span | None
    binding: Binding | None = None


@dataclass(frozen=True)
class CallEdge:
    """A call made inside a function: the called name, its line and its binding."""

    name: str
    line: int
    binding: Binding


@dataclass(frozen=True)
class Reference:
    """A use of a name that is not a call: passed as an argument, stored in a collection, assigned,
    used as a decorator, exported or returned. ``holder`` is the function it sits in (None at module
    level); ``binding`` says how sure the index is that the name reaches its definition."""

    name: str
    file: str
    line: int
    role: str
    holder: Span | None
    binding: Binding | None = None


@dataclass(frozen=True, order=True)
class TextHit:
    file: str
    line: int
    text: str
