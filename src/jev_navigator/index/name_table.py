"""Every name a file's facts hold, and where it sits, kept on disk per file content.

A row ties a name to a definition, a call or a reference in one file content, identified by its git
blob id, so a new index maps its files to rows without reading them, and answers a name lookup with
no text search and no parse. Rows hold names and line numbers, never code: a receiver is a plain
chain of names or ``scope_scan.OPAQUE_RECEIVER``, as the facts hold it.

One table file serves one ``table_identity``: a change to the parser, to any language's rules or to
the code that turns facts into rows starts a new table. A file's rows are written in the same
transaction as its entry in ``files``, with other files up to a bound on rows, so two processes
writing at once leave the table whole and a reader never sees half a file. Each entry carries the
day a run last confirmed it, by covering a scope that holds the file; housekeeping forgets a file's
rows and entry together once it goes unconfirmed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from itertools import islice
from pathlib import Path

from ..cache_root import cache_root
from ..confirmation import Confirmations, today
from ..shared_database import open_shared_database, release_free_pages
from .fact_cache import facts_identity
from .scope_scan import FileFacts

SYMBOL = "symbol"
DECLARATION = "declaration"
CALL = "call"
REFERENCE = "reference"
DEFINITION_KINDS = (SYMBOL, DECLARATION)
_ANONYMOUS = "<anonymous>"
_logger = logging.getLogger(__name__)
_QUERY_CHUNK = 500
ROWS_PER_TRANSACTION = 10_000


@dataclass(frozen=True)
class FileEntry:
    """What the table knows about one file content beyond its names: whether the parser could only
    partly read it, and the lines its ERROR nodes span."""

    incomplete: bool
    unparsed_lines: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NameRow:
    """One place a name sits in one file content. ``position`` is the place's index among the file's
    facts of its kind, so rows come back in the facts' order."""

    blob: str
    kind: str
    position: int
    start: int
    end: int
    role: str | None = None
    receiver: str | None = None


class NameTable:
    def __init__(self, root: Path | None = None) -> None:
        self.path = table_path(root)
        self._lock = threading.Lock()
        self._db = open_shared_database(self.path, _SCHEMA)

    def entries(self, blobs: Collection[str]) -> dict[str, FileEntry]:
        """The entries of those ``blobs`` whose rows the table holds, each confirmed today. An entry
        last confirmed on an earlier day is stamped first and returned only when the stamp still
        found it, so a prune running meanwhile never leaves a returned entry without its rows. Only
        that stamp writes, so a same-day warm run takes no write lock."""
        found, stale = self._read_entries(blobs)
        for blob in stale - self._confirm(stale):
            del found[blob]
        return found

    def forget_unconfirmed(self, before: int, limit: int) -> int:
        """Deletes the rows and entries of at most ``limit`` file contents last confirmed before the
        day ``before``, least recently confirmed first; returns how many, and frees their space."""
        chosen = "select blob from files where confirmed < ? order by confirmed, blob limit ?"
        with self._lock:
            with self._db:
                self._db.execute(f"delete from names where blob in ({chosen})", (before, limit))
                forgotten = self._db.execute(f"delete from files where blob in ({chosen})", (before, limit))
            release_free_pages(self._db)
        return forgotten.rowcount

    def confirmations(self, before: int) -> Confirmations:
        with self._lock:
            held, unconfirmed, oldest = self._db.execute(
                "select count(*), count(*) filter (where confirmed < ?), min(confirmed) from files",
                (before,),
            ).fetchone()
        return Confirmations(held, unconfirmed, oldest)

    def _read_entries(self, blobs: Collection[str]) -> tuple[dict[str, FileEntry], set[str]]:
        found: dict[str, FileEntry] = {}
        stale: set[str] = set()
        current = today()
        with self._lock:
            for chunk in _chunks(sorted(blobs)):
                marks = ",".join("?" * len(chunk))
                query = (
                    f"select blob, incomplete, unparsed_lines, confirmed from files where blob in ({marks})"
                )
                for blob, incomplete, stretches, confirmed in self._db.execute(query, chunk):
                    found[blob] = FileEntry(bool(incomplete), tuple(map(tuple, json.loads(stretches))))
                    if confirmed < current:
                        stale.add(blob)
        return found, stale

    def _confirm(self, blobs: set[str]) -> set[str]:
        """Stamps ``blobs`` confirmed today; returns those the table still held. A stamp that fails is
        logged and never costs the lookup: the entries the read found are kept."""
        confirmed: set[str] = set()
        try:
            with self._lock, self._db:
                for chunk in _chunks(sorted(blobs)):
                    marks = ",".join("?" * len(chunk))
                    stamped = self._db.execute(
                        f"update files set confirmed = ? where blob in ({marks}) returning blob",
                        (today(), *chunk),
                    )
                    confirmed.update(blob for (blob,) in stamped.fetchall())
        except sqlite3.Error as error:
            _logger.warning("name table %s: %d entries not stamped: %s", self.path, len(blobs), error)
            return blobs
        return confirmed

    def add(self, facts_by_blob: Mapping[str, FileFacts]) -> None:
        """Writes the rows of each file content the table does not hold yet, about
        ``ROWS_PER_TRANSACTION`` rows per transaction. Each commit waits for a disk sync, so a
        transaction holds many contents; the write lock is held for the whole transaction, so the
        rows bound how long another process waits to write. A content is never split, and the table
        is a cache, so a crash loses at most one transaction, which the next run rebuilds. Contents
        already held take no write lock, so warm runs never wait on each other; a content another
        process wrote meanwhile is left as it is."""
        held = self.entries(facts_by_blob.keys())
        missing = ((blob, facts) for blob, facts in facts_by_blob.items() if blob not in held)
        with self._lock:
            for transaction in _transactions(missing):
                self._write(transaction)

    def _write(self, contents: list[tuple[str, FileFacts, list[tuple]]]) -> None:
        with self._db:
            self._db.execute("begin immediate")
            for blob, facts, rows in contents:
                if self._add_entry(blob, facts):
                    self._db.executemany("insert into names values (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def rows(self, name: str) -> tuple[NameRow, ...]:
        with self._lock:
            found = self._db.execute(
                "select blob, kind, position, start, end, role, receiver from names where name = ?",
                (name,),
            ).fetchall()
        return tuple(NameRow(*row) for row in found)

    def definitions(self, blob: str) -> tuple[tuple[str, NameRow], ...]:
        """Each named definition in one file content, with its name, in the facts' order: symbols,
        then declarations."""
        with self._lock:
            found = self._db.execute(
                "select name, blob, kind, position, start, end, role, receiver from names"
                " where blob = ? and kind in (?, ?) order by kind = ?, position",
                (blob, SYMBOL, DECLARATION, DECLARATION),
            ).fetchall()
        return tuple((name, NameRow(*row)) for name, *row in found)

    def _add_entry(self, blob: str, facts: FileFacts) -> bool:
        stretches = json.dumps([list(stretch) for stretch in facts.unparsed_lines])
        added = self._db.execute(
            "insert or ignore into files values (?, ?, ?, ?)",
            (blob, int(facts.incomplete), stretches, today()),
        )
        return added.rowcount == 1


def table_path(root: Path | None = None) -> Path:
    """The table file this JVN reads, in ``root`` or else in ``user_name_tables()``."""
    return (root or user_name_tables()) / f"{table_identity()}.sqlite"


def user_name_tables() -> Path:
    """The folder of the name tables every index on this machine shares, one per table identity."""
    return cache_root() / "names"


@cache
def table_identity() -> str:
    """The facts' identity and the source of this module, which decides what a row holds."""
    source = Path(__file__).read_bytes()
    return hashlib.sha256(facts_identity().encode() + b"\0" + source).hexdigest()


def git_blob_id(content: bytes) -> str:
    """The id git gives ``content`` as a blob, so a clean tracked file is identified from the index
    listing alone."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def _rows(blob: str, facts: FileFacts) -> Iterator[tuple]:
    structure = facts.structure
    for kind, spans in ((SYMBOL, structure.symbols), (DECLARATION, structure.declarations)):
        for position, span in enumerate(spans):
            if span.name != _ANONYMOUS:
                yield span.name, blob, kind, position, span.start, span.end, None, None
    for position, call in enumerate(facts.calls):
        yield call.name, blob, CALL, position, call.line, call.line, None, call.receiver
    for position, reference in enumerate(facts.references):
        line, role = reference.line, reference.role
        yield reference.name, blob, REFERENCE, position, line, line, role, reference.receiver


def _transactions(
    contents: Iterable[tuple[str, FileFacts]],
) -> Iterator[list[tuple[str, FileFacts, list[tuple]]]]:
    """The contents with their rows, grouped so a group's rows, one per content's entry included,
    stay within ``ROWS_PER_TRANSACTION``; a content larger than that is a group of its own."""
    group: list[tuple[str, FileFacts, list[tuple]]] = []
    group_rows = 0
    for blob, facts in contents:
        rows = list(_rows(blob, facts))
        if group and group_rows + len(rows) + 1 > ROWS_PER_TRANSACTION:
            yield group
            group, group_rows = [], 0
        group.append((blob, facts, rows))
        group_rows += len(rows) + 1
    if group:
        yield group


def _chunks(values: list[str]) -> Iterator[list[str]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, _QUERY_CHUNK)):
        yield chunk


_SCHEMA = """
create table files (
    blob text primary key,
    incomplete integer not null,
    unparsed_lines text not null,
    confirmed integer not null
) without rowid;
create index files_by_confirmation on files (confirmed, blob);
create table names (
    name text not null,
    blob text not null,
    kind text not null,
    position integer not null,
    start integer not null,
    end integer not null,
    role text,
    receiver text,
    primary key (name, blob, kind, position)
) without rowid;
create index definitions_by_blob on names (blob) where kind in ('symbol', 'declaration');
"""
