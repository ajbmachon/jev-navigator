"""The one reader of scope files: each file as the index first read it, checked against the disk."""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Callable, MutableMapping
from functools import lru_cache
from pathlib import Path

from .languages import split_lines

DISAPPEARED = "disappeared after inventory"
CHANGED = "changed on disk after the index first read it"


class SourceFiles:
    """Every read is checked against the first. A file that changed or disappeared is reported in
    ``unavailable``; its text stays the text it had when first read, kept compressed, so a slice
    always holds the code its SHA-256 names. Split lines are cached for ``line_cache_files`` files.
    ``standing_first_read(file, content)`` names the bytes that stand as a file's first read, given
    what the disk holds then (``None`` when the file is gone); bytes other than the disk's count as a
    change."""

    def __init__(
        self,
        root: Path,
        unavailable: MutableMapping[str, str],
        line_cache_files: int,
        standing_first_read: Callable[[str, bytes | None], bytes | None] = lambda file, content: content,
    ) -> None:
        self._root = root
        self._unavailable = unavailable
        self._standing_first_read = standing_first_read
        self._sha256: dict[str, str] = {}
        self._first_read: dict[str, bytes] = {}
        self._lines = lru_cache(maxsize=line_cache_files)(self._read_lines)

    def current(self, file: str) -> bytes | None:
        """The file's bytes while they still equal its first read; ``None`` once it changed or
        disappeared."""
        content = self._disk_bytes(file)
        if file not in self._sha256:
            self._remember_first_read(file, self._standing_first_read(file, content))
        if content is None:
            self._unavailable[file] = DISAPPEARED
            return None
        if self._sha256.get(file) != hashlib.sha256(content).hexdigest():
            self._unavailable[file] = CHANGED
            return None
        return content

    def first_read(self, file: str) -> bytes | None:
        """The bytes the index first read from ``file``; ``None`` when it was gone before any read."""
        content = self.current(file)
        if content is not None:
            return content
        stored = self._first_read.get(file)
        return zlib.decompress(stored) if stored is not None else None

    def lines(self, file: str) -> tuple[str, ...]:
        return self._lines(file)

    def sha256(self, file: str) -> str:
        """The SHA-256 of the bytes the index first read from ``file``."""
        if file not in self._sha256:
            self.current(file)
        return self._sha256.get(file, "")

    def _disk_bytes(self, file: str) -> bytes | None:
        try:
            return (self._root / file).read_bytes()
        except FileNotFoundError:
            return None

    def _remember_first_read(self, file: str, standing: bytes | None) -> None:
        if standing is not None:
            self._sha256[file] = hashlib.sha256(standing).hexdigest()
            self._first_read[file] = zlib.compress(standing)

    def _read_lines(self, file: str) -> tuple[str, ...]:
        content = self.first_read(file)
        return split_lines(content.decode(errors="replace")) if content is not None else ()
