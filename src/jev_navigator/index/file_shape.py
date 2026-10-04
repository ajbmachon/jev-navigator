"""What a file's bytes say about the cost of parsing it, measured without parsing it.

ast-grep's memory follows the length of each line of code, roughly with its square: a 120 KB bundle on
one line peaked at 681 MB, the same bundle cut into lines of about 1,000 characters at 35 MB. Size
alone does not drive it (703 KB of short lines peaked at 28 MB). The estimate below fits every real
file measured on 03.10.2026 within about 10%; a line of data such as one image string holds few syntax
nodes and the estimate over-counts it, which errs on the safe side.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

BASE_PEAK_MB = 25.0
PEAK_MB_PER_SQUARED_THOUSAND_CHARACTERS = 0.05
MAX_PARSE_PEAK_MB = 250.0
"""A file whose estimated parse peak exceeds this is never parsed. For a one-line file that is a line
of about 70,000 bytes. The largest parse measured under it peaked at 122 MB, and ast-grep scans
files in parallel, so several can be in memory at once."""

PARSEABLE_UP_TO_BYTES = int(
    1000 * ((MAX_PARSE_PEAK_MB - BASE_PEAK_MB) / PEAK_MB_PER_SQUARED_THOUSAND_CHARACTERS) ** 0.5
)
"""A file this small can never be over the bound, whatever its lines: the sum of the squared line
lengths is at most the square of the file's size. The size alone clears it, without a read."""


LONG_LINE_CHARS = 10_000
DENSE_AVERAGE_LINE_CHARS = 110
DENSE_MINIMUM_BYTES = 4_096
LARGE_FILE_BYTES = 500_000


class Trigger(StrEnum):
    """A reason to ask whether a file is generated. A trigger never refuses a parse: only the memory
    bound does that. Starting values from the census of 12,976 files (37 flagged)."""

    LONG_LINE = "long_line"
    """Some line is longer than ``LONG_LINE_CHARS`` characters."""
    DENSE_LINES = "dense_lines"
    """More than ``DENSE_AVERAGE_LINE_CHARS`` characters per line (the average GitHub Linguist uses for
    minified files) on a file of at least ``DENSE_MINIMUM_BYTES``."""
    LARGE_FILE = "large_file"
    """More than ``LARGE_FILE_BYTES`` bytes."""


@dataclass(frozen=True)
class FileShape:
    size_bytes: int
    line_count: int
    longest_line: int
    squared_thousands: float

    @property
    def chars_per_line(self) -> float:
        return 0.0 if self.line_count == 0 else self.size_bytes / self.line_count

    @property
    def parse_peak_mb(self) -> float:
        return BASE_PEAK_MB + PEAK_MB_PER_SQUARED_THOUSAND_CHARACTERS * self.squared_thousands

    @property
    def triggers(self) -> tuple[Trigger, ...]:
        fired = (
            (Trigger.LONG_LINE, self.longest_line > LONG_LINE_CHARS),
            (
                Trigger.DENSE_LINES,
                self.size_bytes >= DENSE_MINIMUM_BYTES and self.chars_per_line > DENSE_AVERAGE_LINE_CHARS,
            ),
            (Trigger.LARGE_FILE, self.size_bytes > LARGE_FILE_BYTES),
        )
        return tuple(trigger for trigger, tripped in fired if tripped)

    @property
    def too_large_to_parse(self) -> bool:
        return self.parse_peak_mb > MAX_PARSE_PEAK_MB

    @property
    def refusal(self) -> str | None:
        if not self.too_large_to_parse:
            return None
        return (
            f"too large to parse: estimated parse peak {_peak_text(self.parse_peak_mb)}, "
            f"longest line {self.longest_line:,} bytes"
        )


def shape_of(root: Path, path: str) -> FileShape:
    """The measured facts and fired triggers of one file, given the repository folder and the path."""
    return measure((root / path).read_bytes())


def refusal_of(root: Path, path: str) -> str | None:
    """Why the file must not be parsed, or None. A file small enough to be safe by size is not read."""
    file = root / path
    if file.stat().st_size <= PARSEABLE_UP_TO_BYTES:
        return None
    return measure(file.read_bytes()).refusal


def measure(content: bytes) -> FileShape:
    """Line lengths are counted in bytes, which over-counts multibyte text and so errs on the safe side."""
    lines = content.split(b"\n")
    lengths = [len(line) for line in lines]
    return FileShape(
        size_bytes=len(content),
        line_count=len(lines) - 1 if lines[-1] == b"" else len(lines),
        longest_line=max(lengths),
        squared_thousands=sum((length / 1000) ** 2 for length in lengths),
    )


def _peak_text(megabytes: float) -> str:
    return f"{megabytes:,.0f} MB" if megabytes < 1000 else f"{megabytes / 1000:,.0f} GB"
