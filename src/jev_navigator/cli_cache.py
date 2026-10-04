"""``jvn cache``: what JVN keeps on disk, and its housekeeping rules run now; and the housekeeping every
run does as it ends."""

from __future__ import annotations

import sys

from . import housekeeping
from .cache_root import cache_root
from .confirmation import today
from .data_root import runs_root
from .housekeeping import CacheStatus, Status, Sweep

CACHE_ACTIONS = ("status", "prune")


def run_cache_command(action: str) -> int:
    try:
        lines = (
            _status_lines(housekeeping.status()) if action == "status" else [_pruned(housekeeping.prune())]
        )
    except Exception as error:  # noqa: BLE001 - the command reports any failure and exits 1
        print(f"jvn cache {action}: {error}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


def tidy_after_run() -> bool:
    """Housekeeping once a run has ended, saved or failed; whether Ctrl-C interrupted it. A failure is
    a notice on stderr, never the run's failure. An interrupt ends the cleanup with one notice; what
    the sweep had not finished goes on in a later run."""
    try:
        housekeeping.tidy()
    except KeyboardInterrupt:
        print("jvn: housekeeping interrupted after the run ended; a later run finishes it", file=sys.stderr)
        return True
    except Exception as error:  # noqa: BLE001 - a failed prune must never fail the run it follows
        print(f"jvn: housekeeping skipped: {error}", file=sys.stderr)
    return False


def _status_lines(status: Status) -> list[str]:
    runs = status.runs
    return [
        f"cache: {cache_root()}",
        f"runs folder: {runs_root()}",
        f"using {_size(status.used)} of the budget {_size(status.budget)} "
        f"({housekeeping.DISK_BUDGET_VARIABLE})",
        _cache_line("facts", status.facts, "entry", "entries"),
        _cache_line("names", status.names, "file", "files"),
        _cache_line("answers", status.answers, "request", "requests"),
        f"runs: {_count(runs.count, 'run', 'runs')}, {_size(runs.bytes)}; {runs.expired:,} past retention "
        f"({housekeeping.FINISHED_RUN_DAYS} days finished, {housekeeping.RESUMABLE_RUN_DAYS} resumable), "
        f"{_size(runs.expired_bytes)}",
        f"trash: {_size(status.trash_bytes)} still to delete",
    ]


def _cache_line(name: str, cache: CacheStatus, singular: str, plural: str) -> str:
    oldest = "" if cache.oldest is None else f", oldest confirmed {_days_ago(cache.oldest)}"
    return (
        f"{name}: {_count(cache.held, singular, plural)}, {_size(cache.bytes)}, "
        f"{cache.unconfirmed:,} unconfirmed for {housekeeping.UNCONFIRMED_DAYS} days{oldest}; "
        f"{cache.retired:,} from other JVN versions "
        f"({cache.retired_unused:,} unused for {cache.retired_after_days} days), "
        f"{_size(cache.retired_bytes)}"
    )


def _pruned(sweep: Sweep) -> str:
    return (
        f"removed {_count(sweep.deleted_files, 'file', 'files')} ({_size(sweep.deleted_bytes)}); forgot "
        f"{_count(sweep.forgotten['names'], 'name table file', 'name table files')} and "
        f"{_count(sweep.forgotten['answers'], 'answered request', 'answered requests')}"
    )


def _days_ago(day: int) -> str:
    days = today() - day
    return "today" if days <= 0 else _count(days, "day ago", "days ago")


def _count(number: int, singular: str, plural: str) -> str:
    return f"{number:,} {singular if number == 1 else plural}"


def _size(size: int) -> str:
    for unit, factor in (("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if size >= factor:
            return f"{size / factor:.1f} {unit}"
    return f"{size} B"
