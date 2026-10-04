"""What JVN deletes from the disk, when, and in which order: the one owner of that policy.

Caches are the most valuable data JVN keeps, but only while they represent real files (André,
04.10.2026). Each store stamps its own entries and deletes them on request; this module decides:

1. Dead by rules: another JVN version's fact identity folder or name table goes once no JVN version
   used it for ``IDENTITY_UNUSED_DAYS``. An older layout of the default answer store holds paid-for
   answers, so it follows the answer rule: it goes once unused for ``UNCONFIRMED_DAYS``.
2. Dead by content: a fact entry or a name table file content no run confirmed against a real file
   for ``UNCONFIRMED_DAYS`` goes.
3. Answers in the default shared store no run reused for ``UNCONFIRMED_DAYS`` go, with their item
   answers and refusals. A store at a path the user names is never opened.
4. Run folders JVN named itself in ``runs_root()`` go after ``FINISHED_RUN_DAYS``, or
   ``RESUMABLE_RUN_DAYS`` while they hold ``resume.json``. A folder the user named is never touched.
5. Over the disk budget, run folders go first, oldest first; then other versions' facts and name
   tables, least recently used first; then fact entries and name table rows, least recently confirmed
   first. Answers go last: older layouts of the default store, then its least recently used answers.

Housekeeping deletes only inside ``cache_root()`` and ``runs_root()`` and never follows a link. A
folder it deletes first moves into a ``.trash`` folder in its root, so the decision is made at once
and the deleting is bounded. A CLI run sweeps at most once a day per cache root and deletes at most
``DELETIONS_PER_RUN`` files; a sweep that reaches the bound goes on in the next run.
"""

from __future__ import annotations

import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .cache_root import cache_root
from .confirmation import SECONDS_PER_DAY, Confirmations, today
from .data_root import data_root, run_folder_started, runs_root
from .index.fact_cache import FactCache
from .index.name_table import NameTable, table_path, user_name_tables
from .judgments.store import SqliteAnswerStore, default_shared_store, is_default_store_file
from .shared_database import database_base

IDENTITY_UNUSED_DAYS = 3
UNCONFIRMED_DAYS = 30
FINISHED_RUN_DAYS = 14
RESUMABLE_RUN_DAYS = 30
DEFAULT_DISK_BUDGET = 5_000_000_000
DISK_BUDGET_VARIABLE = "JEV_NAVIGATOR_DISK_BUDGET"
DELETIONS_PER_RUN = 2_000
TRASH = ".trash"
_SWEPT = ".swept"
_FORGET_STEP = 200
_UNITS = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
_BUDGET = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)?\s*", re.IGNORECASE)


@dataclass
class Sweep:
    """What one sweep removed, and whether it got through every rule within its bound."""

    deleted_files: int = 0
    deleted_bytes: int = 0
    forgotten: dict[str, int] = field(default_factory=lambda: {"names": 0, "answers": 0})
    finished: bool = True


@dataclass(frozen=True)
class CacheStatus:
    """One cache store: its current entries and the retired ones only other JVN versions could read."""

    bytes: int
    held: int
    unconfirmed: int
    oldest: int | None
    retired: int
    retired_unused: int
    retired_bytes: int
    retired_after_days: int


@dataclass(frozen=True)
class RunsStatus:
    count: int
    bytes: int
    expired: int
    expired_bytes: int


@dataclass(frozen=True)
class Status:
    facts: CacheStatus
    names: CacheStatus
    answers: CacheStatus
    runs: RunsStatus
    trash_bytes: int
    used: int
    budget: int


def tidy() -> Sweep | None:
    """The housekeeping a CLI run does as it ends; None when this cache root was swept today."""
    swept = cache_root() / _SWEPT
    if swept.exists() and time.time() - swept.stat().st_mtime < SECONDS_PER_DAY:
        return None
    return _sweep(_Allowance(DELETIONS_PER_RUN))


def prune() -> Sweep:
    """Every rule now, with no bound."""
    return _sweep(_Allowance(None))


def status() -> Status:
    """What each store holds and what each rule would remove; removes nothing."""
    now, held = time.time(), _Holdings.now()
    facts = _cache_status(
        _total(held.entries),
        _fact_confirmations(held.entries, now),
        held.retired_facts,
        IDENTITY_UNUSED_DAYS,
        now,
    )
    names, answers = (
        _cache_status(_store_bytes(store), _confirmations(store), retired, store.retired_after_days, now)
        for store, retired in ((_NAMES, held.retired_names), (_ANSWERS, held.retired_answers))
    )
    expired = [run for run in held.runs if run.expired(now)]
    runs = RunsStatus(
        len(held.runs),
        _total(run.item for run in held.runs),
        len(expired),
        _total(run.item for run in expired),
    )
    return Status(facts, names, answers, runs, held.trash, held.used(), disk_budget())


def disk_budget() -> int:
    """``DISK_BUDGET_VARIABLE`` in bytes, or with a decimal unit such as ``750MB``; else 5 GB."""
    text = os.environ.get(DISK_BUDGET_VARIABLE)
    if not text:
        return DEFAULT_DISK_BUDGET
    match = _BUDGET.fullmatch(text)
    if match is None:
        raise ValueError(f"{DISK_BUDGET_VARIABLE}={text!r} is not a size such as 5GB, 750MB or 1024")
    return int(float(match[1]) * _UNITS[(match[2] or "").upper()])


@dataclass(frozen=True)
class _Item:
    """Something housekeeping may delete: a folder, a file, or a database with its side files."""

    paths: tuple[Path, ...]
    bytes: int
    touched: float
    folder: bool = False


@dataclass(frozen=True)
class _Run:
    item: _Item
    started: float
    resumable: bool

    def expired(self, now: float) -> bool:
        days = RESUMABLE_RUN_DAYS if self.resumable else FINISHED_RUN_DAYS
        return now - self.started > days * SECONDS_PER_DAY


@dataclass(frozen=True)
class _Store:
    """A shared SQLite store housekeeping looks after: its current file, and its retired layouts, which
    go once unused for ``retired_after_days``."""

    name: str
    folder: Callable[[], Path]
    current: Callable[[], Path]
    belongs: Callable[[str], bool]
    open: Callable[[], NameTable | SqliteAnswerStore]
    retired_after_days: int

    def groups(self) -> dict[Path, list[Path]]:
        """Each database file in the store's folder with its side files, keyed by the database."""
        grouped: dict[Path, list[Path]] = {}
        for path in _children(self.folder()):
            if self.belongs(path.name):
                grouped.setdefault(database_base(path), []).append(path)
        return grouped

    def retired(self) -> list[_Item]:
        current = self.current()
        return [_group_item(paths) for base, paths in self.groups().items() if base != current]


def _is_table_file(name: str) -> bool:
    return ".sqlite" in name


def _open_default_store() -> SqliteAnswerStore:
    return SqliteAnswerStore(default_shared_store())


_NAMES = _Store("names", user_name_tables, table_path, _is_table_file, NameTable, IDENTITY_UNUSED_DAYS)
_ANSWERS = _Store(
    "answers", cache_root, default_shared_store, is_default_store_file, _open_default_store, UNCONFIRMED_DAYS
)


@dataclass(frozen=True)
class _Holdings:
    """One listing of everything housekeeping may delete, taken once per sweep or status."""

    entries: list[_Item]
    retired_facts: list[_Item]
    retired_names: list[_Item]
    retired_answers: list[_Item]
    runs: list[_Run]
    trash: int

    @classmethod
    def now(cls) -> _Holdings:
        cache = FactCache()
        return cls(
            _items(entry for folder in cache.current_folders() for entry in _children(folder)),
            _items(cache.retired()),
            _NAMES.retired(),
            _ANSWERS.retired(),
            _runs(),
            sum(_size(trash) for trash in _trashes()),
        )

    @property
    def retired(self) -> list[_Item]:
        return [*self.retired_facts, *self.retired_names, *self.retired_answers]

    def used(self) -> int:
        caches = _total(self.entries) + _total(self.retired) + _store_bytes(_NAMES) + _store_bytes(_ANSWERS)
        return caches + _total(run.item for run in self.runs) + self.trash


@dataclass
class _Allowance:
    """How many more files this sweep may delete; None is unbounded."""

    left: int | None
    sweep: Sweep = field(default_factory=Sweep)

    @property
    def exhausted(self) -> bool:
        return self.left is not None and self.left <= 0

    @property
    def limit(self) -> int:
        return self.left if self.left is not None else 2**62

    def spend(self, count: int) -> None:
        if self.left is not None:
            self.left -= count


def _sweep(allowance: _Allowance) -> Sweep:
    now = time.time()
    _empty_trashes(allowance)
    kept = _apply_rules(_Holdings.now(), now, allowance)
    _keep_to_budget(kept, allowance)
    _empty_trashes(allowance)
    allowance.sweep.finished = not allowance.exhausted and not _trashes()
    if allowance.sweep.finished:
        _mark_swept()
    return allowance.sweep


def _apply_rules(held: _Holdings, now: float, allowance: _Allowance) -> _Holdings:
    """Rules 1 to 4; returns what they left."""
    retired_facts = _kept(held.retired_facts, _older_than(IDENTITY_UNUSED_DAYS, now), allowance)
    retired_names, retired_answers = (
        _kept(items, _older_than(store.retired_after_days, now), allowance)
        for store, items in ((_NAMES, held.retired_names), (_ANSWERS, held.retired_answers))
    )
    entries = _kept(held.entries, _older_than(UNCONFIRMED_DAYS, now), allowance)
    for store in (_NAMES, _ANSWERS):
        _forget_unconfirmed(store, today() - UNCONFIRMED_DAYS, allowance)
    runs = [run for run in held.runs if not (run.expired(now) and _removed(run.item, allowance))]
    return _Holdings(entries, retired_facts, retired_names, retired_answers, runs, held.trash)


def _older_than(days: int, now: float) -> Callable[[_Item], bool]:
    """Whether an item was last used or confirmed more than ``days`` before ``now``."""
    return lambda item: now - item.touched > days * SECONDS_PER_DAY


def _kept(items: list[_Item], doomed: Callable[[_Item], bool], allowance: _Allowance) -> list[_Item]:
    return [item for item in items if not (doomed(item) and _removed(item, allowance))]


def _keep_to_budget(held: _Holdings, allowance: _Allowance) -> None:
    """Rule 5: everything ahead of the answers, then name rows, then older answer layouts, and the
    default store's answers last."""
    excess = held.used() - disk_budget()
    if excess <= 0:
        return
    excess = _evicted(_ahead_of_answers(held), excess, allowance)
    excess = _forgotten(_NAMES, excess, allowance)
    excess = _evicted(_least_recently_used(held.retired_answers), excess, allowance)
    _forgotten(_ANSWERS, excess, allowance)


def _ahead_of_answers(held: _Holdings) -> Iterator[_Item]:
    yield from (run.item for run in sorted(held.runs, key=lambda run: run.started))
    yield from _least_recently_used([*held.retired_facts, *held.retired_names])
    yield from _least_recently_used(held.entries)


def _least_recently_used(items: list[_Item]) -> list[_Item]:
    return sorted(items, key=lambda item: item.touched)


def _evicted(items: Iterable[_Item], excess: int, allowance: _Allowance) -> int:
    """Removes ``items`` in order while JVN is over its budget; returns the bytes still over it."""
    for item in items:
        if excess <= 0 or allowance.exhausted:
            break
        if _removed(item, allowance):
            excess -= item.bytes
    return excess


def _forgotten(store: _Store, excess: int, allowance: _Allowance) -> int:
    """Forgets ``store``'s least recently confirmed rows while JVN is over its budget; returns the bytes
    still over it."""
    while excess > 0 and not allowance.exhausted:
        before = _store_bytes(store)
        if not _forget_unconfirmed(store, today() + 1, allowance, step=_FORGET_STEP):
            break
        excess -= before - _store_bytes(store)
    return excess


def _removed(item: _Item, allowance: _Allowance) -> bool:
    """Removes ``item``: a folder moves into its root's trash at once, files are deleted within the
    allowance. Whether it is gone."""
    if item.folder:
        _discard(item.paths[0])
        return True
    for path in item.paths:
        if allowance.exhausted:
            return False
        _delete_file(path, allowance)
    return True


def _forget_unconfirmed(store: _Store, before: int, allowance: _Allowance, step: int | None = None) -> int:
    if allowance.exhausted or not store.current().exists():
        return 0
    limit = allowance.limit if step is None else min(step, allowance.limit)
    forgotten = store.open().forget_unconfirmed(before=before, limit=limit)
    allowance.spend(forgotten)
    allowance.sweep.forgotten[store.name] += forgotten
    return forgotten


def _discard(folder: Path) -> None:
    trash = (cache_root() if folder.is_relative_to(cache_root()) else data_root()) / TRASH
    trash.mkdir(exist_ok=True)
    try:
        folder.rename(trash / f"{uuid.uuid4().hex}-{folder.name}")
    except FileNotFoundError:
        return


def _trashes() -> list[Path]:
    return [trash for trash in (cache_root() / TRASH, data_root() / TRASH) if _children(trash)]


def _empty_trashes(allowance: _Allowance) -> None:
    for trash in _trashes():
        for top, folders, files in os.walk(trash, topdown=False):
            for name in files:
                if allowance.exhausted:
                    return
                _delete_file(Path(top) / name, allowance)
            for name in folders:
                _delete_folder(Path(top) / name)


def _delete_file(path: Path, allowance: _Allowance) -> None:
    try:
        size = path.lstat().st_blocks * 512
        path.unlink()
    except FileNotFoundError:
        return
    allowance.spend(1)
    allowance.sweep.deleted_files += 1
    allowance.sweep.deleted_bytes += size


def _delete_folder(path: Path) -> None:
    """An emptied folder, or a link to one, which is removed itself and never followed."""
    try:
        path.unlink() if path.is_symlink() else path.rmdir()
    except FileNotFoundError:
        return


def _mark_swept() -> None:
    swept = cache_root() / _SWEPT
    swept.parent.mkdir(parents=True, exist_ok=True)
    swept.touch()


def _confirmations(store: _Store) -> Confirmations:
    if not store.current().exists():
        return Confirmations(0, 0, None)
    return store.open().confirmations(before=today() - UNCONFIRMED_DAYS)


def _fact_confirmations(entries: list[_Item], now: float) -> Confirmations:
    oldest = min((item.touched for item in entries), default=None)
    unconfirmed = sum(map(_older_than(UNCONFIRMED_DAYS, now), entries))
    return Confirmations(
        len(entries), unconfirmed, None if oldest is None else int(oldest // SECONDS_PER_DAY)
    )


def _cache_status(
    size: int, confirmations: Confirmations, retired: list[_Item], retired_after_days: int, now: float
) -> CacheStatus:
    return CacheStatus(
        size,
        confirmations.held,
        confirmations.unconfirmed,
        confirmations.oldest,
        len(retired),
        sum(map(_older_than(retired_after_days, now), retired)),
        _total(retired),
        retired_after_days,
    )


def _store_bytes(store: _Store) -> int:
    return sum(_size(path) for path in store.groups().get(store.current(), []))


def _runs() -> list[_Run]:
    runs = []
    for folder in _children(runs_root()):
        started = run_folder_started(folder.name)
        item = _item(folder) if started is not None else None
        if item is not None and item.folder:
            runs.append(_Run(item, started.timestamp(), (folder / "resume.json").exists()))
    return runs


def _items(paths: Iterable[Path]) -> list[_Item]:
    return [item for path in paths if (item := _item(path)) is not None]


def _item(path: Path) -> _Item | None:
    """``path`` as one item, from a single ``lstat``: a link is a file, so removing it never touches
    what it points to. None when the path is gone."""
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    folder = stat.S_ISDIR(status.st_mode)
    size = _size(path) if folder else status.st_blocks * 512
    return _Item((path,), size, status.st_mtime, folder)


def _group_item(paths: list[Path]) -> _Item:
    statuses = [path.lstat() for path in paths]
    return _Item(
        tuple(paths), sum(status.st_blocks * 512 for status in statuses), max(s.st_mtime for s in statuses)
    )


def _size(path: Path) -> int:
    """Bytes on disk under ``path``, never following a link."""
    if not path.is_dir() or path.is_symlink():
        return path.lstat().st_blocks * 512
    return sum(
        (Path(top) / name).lstat().st_blocks * 512
        for top, folders, files in os.walk(path)
        for name in (*files, *folders)
    )


def _total(items: Iterable[_Item]) -> int:
    return sum(item.bytes for item in items)


def _children(folder: Path) -> list[Path]:
    try:
        return list(folder.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
