"""Locations and pieces of code that every layer passes around. Lines are 1-based and inclusive."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

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

    @property
    def is_named(self) -> bool:
        """Whether the syntax gave the span a name: an anonymous function's is `<anonymous>`."""
        return bool(self.name) and not self.name.startswith("<")

    def contains(self, line: int) -> bool:
        return self.start <= line <= self.end

    def size(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: Span) -> bool:
        return self.file == other.file and self.start <= other.end and other.start <= self.end


def merged_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """The ranges in order, each one that starts at or before the end of the one before joined into it:
    character spans that overlap or touch, or line ranges that share a line."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class Lines(Protocol):
    """Anything that sits on lines ``start`` to ``end`` of one file: a span or a unit."""

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...


Held = TypeVar("Held", bound=Lines)


def _extent(held: Lines) -> int:
    return held.end - held.start


class Nesting(Generic[Held]):
    """Which of one file's spans holds a line, and which holds one of them, as lookups. Each line's
    spans are kept smallest first, and of equal size in the order given, built once, so neither
    question scans every span again.

    ``innermost`` is the smallest span holding a line. ``holder`` is the smallest span whose lines
    contain a span's and fill more of them: a method's class, a nested function's outer function.
    Lines are all a span knows, so a span on the same lines as another (a method on a one-line class)
    has no holder."""

    def __init__(self, spans: Iterable[Held]) -> None:
        self._by_line: dict[int, list[Held]] = {}
        for held in sorted(spans, key=_extent):
            for line in range(held.start, held.end + 1):
                self._by_line.setdefault(line, []).append(held)

    def innermost(self, line: int) -> Held | None:
        holding = self._by_line.get(line)
        return holding[0] if holding else None

    def holder(self, held: Lines) -> Held | None:
        holding = self._by_line.get(held.start, [])
        larger = bisect_right(holding, _extent(held), key=_extent)
        return next((other for other in holding[larger:] if other.end >= held.end), None)


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
