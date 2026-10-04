"""When a cache entry was last confirmed: a run confirms an entry whenever it finds a real file whose
content key the entry matches, or reuses an answer. Stamps count whole UTC days, so a store writes at
most one stamp per entry per day and a same-day warm run writes none."""

from __future__ import annotations

import time
from dataclasses import dataclass

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class Confirmations:
    """How many entries a store holds, how many of them were last confirmed before a given day, and
    the oldest confirmation day, None for an empty store."""

    held: int
    unconfirmed: int
    oldest: int | None


def today() -> int:
    """The current UTC day, counted from the Unix epoch."""
    return day_of(time.time())


def day_of(timestamp: float) -> int:
    """The UTC day of a Unix timestamp, such as a file's modification time."""
    return int(timestamp // SECONDS_PER_DAY)
