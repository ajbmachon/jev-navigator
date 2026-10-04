"""SQLite files that several JVN processes read and write at once."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing, suppress
from pathlib import Path

_SIDE_FILE_SUFFIXES = ("-wal", "-shm")


class UnusableDatabaseError(RuntimeError):
    """The file at a shared database path cannot serve as one."""


def open_shared_database(path: Path, schema: str, version: int = 0) -> sqlite3.Connection:
    """A connection to the file at ``path``, created first when it is missing. A new file is built
    whole under a temporary name, in WAL mode with ``schema`` and ``version``, then linked into
    place. Linking fails when another process got there first, so every process opens one finished
    file and none has to change its journal mode, which needs the file to itself. Every new file,
    name table and answer store alike, has incremental auto-vacuum, so ``release_free_pages`` can
    return the space of deleted rows. Opening stamps the file's modification time, so that time
    says when a JVN process of any version last used the file; a file that refuses the stamp still
    opens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _create(path, schema, version)
    database = sqlite3.connect(path, timeout=30, check_same_thread=False)
    _require_database(database, path)
    with suppress(OSError):
        os.utime(path)
    return database


def database_base(path: Path) -> Path:
    """The database file a SQLite side file (its write-ahead log or shared memory) belongs to."""
    for suffix in _SIDE_FILE_SUFFIXES:
        if path.name.endswith(suffix):
            return path.with_name(path.name.removesuffix(suffix))
    return path


def release_free_pages(database: sqlite3.Connection) -> None:
    """Returns the pages deleted rows left free to the disk: they move to the end of the file, and a
    checkpoint cuts them off and empties the write-ahead log, as far as no reader still needs it. The
    vacuum frees one page per step, and Python's ``execute`` before 3.12 runs it for one step only,
    so it runs as a script, which steps it to the end. A script commits an open transaction first."""
    database.executescript("pragma incremental_vacuum;")
    database.execute("pragma wal_checkpoint(truncate)").fetchall()


def _create(path: Path, schema: str, version: int) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(temporary)) as database:
            database.execute("pragma auto_vacuum = incremental")
            switch_to_wal(database, path)
            database.executescript(schema)
            database.execute(f"pragma user_version = {version}")
            database.commit()
        with suppress(FileExistsError):
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def switch_to_wal(database: sqlite3.Connection, path: Path) -> None:
    """SQLite answers a WAL request it cannot meet, on a file system without the locks WAL needs,
    with the mode it kept instead of an error."""
    mode = database.execute("pragma journal_mode=wal").fetchone()[0]
    if mode != "wal":
        raise UnusableDatabaseError(
            f"{path} cannot be shared between runs: SQLite kept it in {mode} mode instead of WAL, "
            "so its file system lacks the locks WAL needs; point it at a local disk"
        )


def _require_database(database: sqlite3.Connection, path: Path) -> None:
    try:
        database.execute("pragma schema_version").fetchone()
    except sqlite3.DatabaseError as error:
        database.close()
        raise UnusableDatabaseError(f"{path} is not a SQLite database: {error}") from error
