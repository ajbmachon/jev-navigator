"""Several explicit code operations in one bounded, pageable agent response.

The caller owns the recipe and any Judge. Mechanical operations never create a model client.
Source rows remain numbered and a character cursor can continue even a single very long line.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace

from .index.code_index import CodeIndex
from .index.spans import Span
from .judgments.judge import Judge
from .judgments.questions import Check

OPERATIONS = (
    "outline",
    "names",
    "def",
    "refs",
    "callers",
    "callees",
    "named_files",
    "show",
    "cochange",
    "tests_of",
    "rank",
)


@dataclass(frozen=True)
class Cursor:
    row: int = 0
    character: int = 0

    def __post_init__(self) -> None:
        if type(self.row) is not int or type(self.character) is not int or min(self.row, self.character) < 0:
            raise ValueError("cursor row and character must be non-negative integers")


@dataclass(frozen=True)
class Operation:
    op: str
    file: str = ""
    name: str = ""
    query: str = ""
    line: int = 1
    end: int | None = None
    window: int = 0
    patterns: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    regex: bool = False
    candidates: tuple[Span, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)
    limit: int = 80

    def __post_init__(self) -> None:
        if self.op not in OPERATIONS:
            raise ValueError(f"unknown operation {self.op!r}; choose from {', '.join(OPERATIONS)}")
        for name in ("line", "limit", "window"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "window" else 1):
                raise ValueError(f"{name} must be {'non-negative' if name == 'window' else 'positive'}")
        if self.end is not None and (type(self.end) is not int or self.end < self.line):
            raise ValueError("end must be an integer at or after line")
        if self.op in ("show", "cochange", "callees") and not self.file:
            raise ValueError(f"{self.op} requires file")
        if self.op in ("names", "def", "callers") and not self.name:
            raise ValueError(f"{self.op} requires name")
        if self.op == "refs" and not (self.name or self.query):
            raise ValueError("refs requires name or query")
        if self.op == "tests_of" and not (self.file or self.name):
            raise ValueError("tests_of requires file or name")
        if self.op == "rank" and not self.query:
            raise ValueError("rank requires query")

    @classmethod
    def from_dict(cls, value: Mapping) -> Operation:
        values = dict(value)
        for name in ("patterns", "scopes"):
            if name in values:
                if not isinstance(values[name], list) or not all(isinstance(x, str) for x in values[name]):
                    raise ValueError(f"{name} must be an array of strings")
                values[name] = tuple(values[name])
        if "cursor" in values:
            values["cursor"] = Cursor(**values["cursor"])
        if "candidates" in values:
            values["candidates"] = tuple(Span(**item) for item in values["candidates"])
        for name in ("op", "file", "name", "query"):
            if name in values and not isinstance(values[name], str):
                raise ValueError(f"{name} must be a string")
        if "regex" in values and type(values["regex"]) is not bool:
            raise ValueError("regex must be a boolean")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(str(error)) from error


@dataclass(frozen=True)
class Page:
    number: int
    operation: str
    total: int
    items: tuple[dict, ...]
    next: Cursor | None
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BatchResult:
    pages: tuple[Page, ...]
    calls: int
    replayed_answers: int

    def to_dict(self) -> dict:
        return {
            "pages": [page.to_dict() for page in self.pages],
            "calls": self.calls,
            "replayed_answers": self.replayed_answers,
        }

    def render(self) -> str:
        """The same bounded payload at the CLI and library boundaries, numbered by operation/row."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def run_batch(
    index: CodeIndex,
    operations: Sequence[Operation],
    *,
    judge: Judge | None = None,
    checks: Sequence[Check] | None = None,
    shared: Mapping | None = None,
    max_chars: int = 24_000,
) -> BatchResult:
    """Run independent facts concurrently. Rank operations share the caller's Judge and run in order.

    ``max_chars`` bounds the complete JSON response, including numbering and continuation metadata.
    Each operation gets an equal display share. A failed operation is explicit beside successful ones.
    Repeat the same operation with its ``next`` cursor to continue. Keep the index/source revision fixed.
    """
    from .batch_operations import rank_rows

    if not operations:
        return BatchResult((), 0, 0)
    share = _display_share(operations, max_chars)
    before = judge.calls if judge else 0
    replayed = judge.replayed_answers if judge else 0

    pages = _fact_pages(index, operations, share)
    for number, operation in enumerate(operations, 1):
        if operation.op != "rank":
            continue
        try:
            pages.append(_page(number, operation, rank_rows(index, operation, judge, checks, shared), share))
        except (OSError, RuntimeError, ValueError, LookupError) as error:
            pages.append(_error_page(number, operation, error, share))

    return BatchResult(
        tuple(sorted(pages, key=lambda p: p.number)),
        (judge.calls - before) if judge else 0,
        (judge.replayed_answers - replayed) if judge else 0,
    )


async def run_batch_async(
    index: CodeIndex,
    operations: Sequence[Operation],
    *,
    judge: Judge | None = None,
    checks: Sequence[Check] | None = None,
    shared: Mapping | None = None,
    max_chars: int = 24_000,
) -> BatchResult:
    """The same contract with native async judging, for hosts such as an agent runtime."""
    from .batch_operations import rank_rows_async

    if not operations:
        return BatchResult((), 0, 0)
    share = _display_share(operations, max_chars)
    before, replayed = (judge.calls, judge.replayed_answers) if judge else (0, 0)
    pages = await asyncio.to_thread(_fact_pages, index, operations, share)
    for number, operation in enumerate(operations, 1):
        if operation.op != "rank":
            continue
        try:
            rows = await rank_rows_async(index, operation, judge, checks, shared)
            pages.append(_page(number, operation, rows, share))
        except (OSError, RuntimeError, ValueError, LookupError) as error:
            pages.append(_error_page(number, operation, error, share))
    return BatchResult(
        tuple(sorted(pages, key=lambda p: p.number)),
        (judge.calls - before) if judge else 0,
        (judge.replayed_answers - replayed) if judge else 0,
    )


def _display_share(operations: Sequence[Operation], max_chars: int) -> int:
    if type(max_chars) is not int or max_chars < 1024 * len(operations) + 128:
        raise ValueError("max_chars must allow at least 1024 characters per operation plus 128")
    return (max_chars - 128) // len(operations)


def _error_page(number: int, operation: Operation, error: Exception, share: int) -> Page:
    page = _page(number, replace(operation, cursor=Cursor()), [{"error": str(error)}], share)
    return replace(page, error=type(error).__name__)


def _fact_pages(index: CodeIndex, operations: Sequence[Operation], share: int) -> list[Page]:
    from .batch_operations import rows_for

    def run(number_operation: tuple[int, Operation]) -> Page:
        number, operation = number_operation
        try:
            return _page(number, operation, rows_for(index, operation, None, None, None), share)
        except (OSError, RuntimeError, ValueError, LookupError) as error:
            return _error_page(number, operation, error, share)

    facts = [(n, op) for n, op in enumerate(operations, 1) if op.op != "rank"]
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(facts)))) as executor:
        return list(executor.map(run, facts))


def _size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _page(number: int, operation: Operation, rows: Iterable[dict], share: int) -> Page:
    items: list[dict] = []
    next_cursor = None
    total = 0
    # Reserve metadata, the count and a cursor even for a large repository.
    room = share - 300
    for position, row in enumerate(rows):
        total += 1
        if position < operation.cursor.row or next_cursor is not None:
            continue
        start = operation.cursor.character if position == operation.cursor.row else 0
        # Page the entire row as JSON text only when it cannot fit as a structured item.
        item = {"number": position + 1, **row}
        size = _size(item) + 1
        if not start and size <= room and len(items) < operation.limit:
            items.append(item)
            room -= size
            continue
        if items or len(items) >= operation.limit:
            next_cursor = Cursor(position, start)
            continue
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if start >= len(encoded):
            raise ValueError("cursor character is outside its row")
        fragment = {
            "number": position + 1,
            "row_json_offset": start,
            "row_json_chars": len(encoded),
            "row_json": "",
        }
        low, high = 0, len(encoded) - start
        while low < high:
            middle = (low + high + 1) // 2
            fragment["row_json"] = encoded[start : start + middle]
            if _size(fragment) + 1 <= room:
                low = middle
            else:
                high = middle - 1
        if not low:
            raise ValueError("response share is too small for a row fragment")
        fragment["row_json"] = encoded[start : start + low]
        items.append(fragment)
        next_cursor = Cursor(position, start + low) if start + low < len(encoded) else Cursor(position + 1)
    if operation.cursor.row > total or (operation.cursor.row == total and operation.cursor.character):
        raise ValueError("cursor row is outside this result")
    if next_cursor == Cursor(total):
        next_cursor = None
    return Page(number, operation.op, total, tuple(items), next_cursor)
