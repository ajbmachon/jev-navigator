"""The spelling map: every real spelling of a word in scope, found through its word parts.

A file spells words: identifiers, config keys, and the words of its strings and comments, all read
from its whole text, so every language and format gives them the same way; an env template gives
only its keys. A file's name is a spelling too. A spelling splits into word parts at underscores,
hyphens, dots and dollar signs, at a lower-to-upper case change, and between letters and digits
(``createWebsite``: create, website; ``HTTPServer``: http, server). Each part is lowercased and made
singular (``websites``: website; ``policies``: policy). A spelling's keys are all its parts joined and
every run of up to ``RUN_PARTS`` neighbouring parts joined, so ``Website``, ``websites``,
``web_site``, ``WEB_SITE``, ``website.ts`` and ``createWebsite`` all hold the key ``website``.

``SpellingMap.names`` keys a term the same way, its last part also as a plural and a singular in
``-es``, so ``status`` meets ``statuses``, and answers every spelling holding that key, rarest first:
the one fewest files hold.

``SpellingTable`` keeps each file content's spellings and their lines on disk by git blob id, so one
content's words are read once, by whichever index reads it first. The table's identity is this
module's source, so a change to how words are read starts a new table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import weakref
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

from .. import memory_limit
from ..cache_root import cache_root
from ..confirmation import Confirmations, today
from ..shared_database import open_shared_database, release_free_pages
from .languages import split_lines

RUN_PARTS = 4
"""The most neighbouring parts one key joins, besides the key of all parts."""
_WORD = re.compile(r"[^\W\d][\w$]*(?:-[^\W\d][\w$]*)*|\$[\w$]+")
_ENV_KEY = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][\w.]*)\s*=")
_SEPARATORS = re.compile(r"[\W_]+")
_PART_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=\D)(?=\d)|(?<=\d)(?=\D)")
_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})
_KEPT_ENDINGS = ("ss", "us", "is")
_logger = logging.getLogger(__name__)
_ES_PLURAL_ENDINGS = ("sses", "xes", "zes", "ches", "shes")

Words = Mapping[str, tuple[int, ...]]
"""Each word a file content spells, with the lines it sits on."""


@dataclass(frozen=True)
class SpellingPlace:
    """A file holding a spelling, and its lines there; a file name's place has no lines: the whole file."""

    file: str
    lines: tuple[int, ...]


@dataclass(frozen=True)
class Spelling:
    """One real spelling: a word files spell, or a file name. ``files`` is how many scope files hold
    it, its rarity; ``places`` names them in file order, up to the caller's cap, and ``capped`` counts
    the files the cap left out."""

    word: str
    file_name: bool
    files: int
    places: tuple[SpellingPlace, ...]
    capped: int


def word_parts(word: str) -> tuple[str, ...]:
    """``word``'s parts, lowercased and singular: ``createWebsites`` gives create, website."""
    pieces = (piece for chunk in _SEPARATORS.split(word) for piece in _PART_BOUNDARY.split(chunk))
    return tuple(singular(piece.lower()) for piece in pieces if piece)


def singular(part: str) -> str:
    """A lowercased part without its plural ending: ``policies`` policy, ``classes`` class, ``boxes``
    box, ``websites`` website. A part of two letters, or ending in ``ss``, ``us`` or ``is``, keeps
    its form, so ``status`` and ``analysis`` stay whole."""
    if len(part) <= 2 or part.endswith(_KEPT_ENDINGS):
        return part
    if part.endswith("ies") and len(part) > 4:
        return f"{part[:-3]}y"
    if part.endswith(_ES_PLURAL_ENDINGS):
        return part[:-2]
    return part.removesuffix("s")


def spelling_keys(word: str) -> frozenset[str]:
    """The keys ``word`` holds: all its parts joined, and every run of up to ``RUN_PARTS`` of them."""
    parts = word_parts(word)
    runs = (
        "".join(parts[start:end])
        for start in range(len(parts))
        for end in range(start + 1, min(start + RUN_PARTS, len(parts)) + 1)
    )
    return frozenset((*runs, "".join(parts))) - {""}


def lookup_keys(term: str) -> frozenset[str]:
    """The keys a lookup of ``term`` reads: its parts joined, with its last part also read plural in
    ``-es`` and, ending in ``-es``, without it, since ``singular`` cannot tell ``statuses`` from
    ``cases``."""
    words = _WORD.findall(term)
    if not words:
        return frozenset()
    *head, last = words
    forms = [last, f"{last}es", *([last[:-2]] if last.lower().endswith("es") and len(last) > 4 else [])]
    leading = "".join(part for word in head for part in word_parts(word))
    return frozenset(leading + "".join(word_parts(form)) for form in forms) - {""}


def text_words(texts: Iterable[str]) -> tuple[str, ...]:
    """The words ``texts`` spell, each once, in order of first mention."""
    return tuple(dict.fromkeys(word for text in texts for word in _WORD.findall(text)))


def words_in(file: str, content: bytes) -> dict[str, tuple[int, ...]]:
    """Each word ``content`` spells and the lines it sits on; an env template spells only its keys."""
    lines: dict[str, list[int]] = {}
    keys_only = PurePosixPath(file).name in _ENV_TEMPLATES
    for number, text in enumerate(split_lines(content.decode(errors="replace")), start=1):
        found = _ENV_KEY.findall(text) if keys_only else _WORD.findall(text)
        for word in dict.fromkeys(found):
            lines.setdefault(word, []).append(number)
    return {word: tuple(numbers) for word, numbers in lines.items()}


def file_words(
    table: SpellingTable, blobs: Mapping[str, str], read: Callable[[str], bytes | None]
) -> dict[str, Words]:
    """The words of each file of ``blobs`` (file to blob id): from the table, or read from the file's
    bytes and written to it. A file ``read`` finds gone has none."""
    found = dict(table.words(set(blobs.values())))
    read_now: dict[str, Words] = {}
    for file, blob in blobs.items():
        if blob in found or blob in read_now:
            continue
        memory_limit.check()
        content = read(file)
        if content is not None:
            read_now[blob] = words_in(file, content)
    table.add(read_now)
    found |= read_now
    return {file: found[blob] for file, blob in blobs.items() if blob in found}


class SpellingMap:
    """The spellings of a set of files: the words each file spells, and each file's name."""

    def __init__(self, words_by_file: Mapping[str, Words]) -> None:
        self._words_by_file = words_by_file
        self._files_by_word: dict[str, list[str]] = {}
        for file, words in words_by_file.items():
            for word in words:
                self._files_by_word.setdefault(word, []).append(file)
        self._files_by_name: dict[str, list[str]] = {}
        for file in words_by_file:
            self._files_by_name.setdefault(PurePosixPath(file).name, []).append(file)
        self._words_by_key = _by_key(self._files_by_word, spelling_keys)
        self._names_by_key = _by_key(self._files_by_name, _file_name_keys)

    def names(self, term: str, max_files: int | None = None) -> tuple[Spelling, ...]:
        """Every spelling holding ``term``'s key (see ``lookup_keys``), rarest first, then by word;
        each names at most ``max_files`` files, all of them when None."""
        if max_files is not None and max_files < 1:
            raise ValueError("max_files must be at least 1")
        keys = lookup_keys(term)
        words = {word for key in keys for word in self._words_by_key.get(key, ())}
        names = {name for key in keys for name in self._names_by_key.get(key, ())}
        found = [
            *(self._spelling(word, max_files) for word in words),
            *(self._file_name(name, max_files) for name in names),
        ]
        return tuple(sorted(found, key=lambda spelling: (spelling.files, spelling.word, spelling.file_name)))

    def _spelling(self, word: str, max_files: int | None) -> Spelling:
        files = self._files_by_word[word]
        listed = files[:max_files]
        places = tuple(SpellingPlace(file, self._words_by_file[file][word]) for file in listed)
        return Spelling(word, False, len(files), places, len(files) - len(listed))

    def _file_name(self, name: str, max_files: int | None) -> Spelling:
        files = self._files_by_name[name]
        listed = tuple(SpellingPlace(file, ()) for file in files[:max_files])
        return Spelling(name, True, len(files), listed, len(files) - len(listed))


def _file_name_keys(name: str) -> frozenset[str]:
    """A file name's keys: its stem's (the name without its last suffix, ``website.test`` of
    ``website.test.ts``), and all its parts joined, so a lookup of ``website.ts`` meets it while a
    lookup of ``ts`` meets no file by its suffix alone."""
    stem, dot, _ = name.rpartition(".")
    return spelling_keys(stem if dot and stem else name) | {"".join(word_parts(name))}


def _by_key(spellings: Iterable[str], keys_of: Callable[[str], frozenset[str]]) -> dict[str, set[str]]:
    by_key: dict[str, set[str]] = {}
    for spelling in spellings:
        for key in keys_of(spelling):
            by_key.setdefault(key, set()).add(spelling)
    return by_key


class SpellingTable:
    """Each file content's words and their lines, keyed by git blob id, with the day a run last
    confirmed the content by reading a scope that holds it."""

    def __init__(self, root: Path | None = None) -> None:
        self.path = table_path(root)
        self._lock = threading.Lock()
        self._db = open_shared_database(self.path, _SCHEMA)
        weakref.finalize(self, self._db.close)

    def words(self, blobs: Collection[str]) -> dict[str, Words]:
        """The words of those ``blobs`` the table holds, each confirmed today. Only a content last
        confirmed on an earlier day is stamped, so a same-day warm run takes no write lock."""
        wanted = json.dumps(sorted(blobs))
        with self._lock:
            rows = self._db.execute(
                "select blob, words, confirmed from words where blob in (select value from json_each(?))",
                (wanted,),
            ).fetchall()
        stale = [blob for blob, _, confirmed in rows if confirmed < today()]
        if stale:
            self._confirm(stale)
        return {blob: _decoded(words) for blob, words, _ in rows}

    def add(self, words_by_blob: Mapping[str, Words]) -> None:
        """Writes the words of each content the table does not hold yet, in one transaction; a content
        another process wrote meanwhile is left as it is."""
        if not words_by_blob:
            return
        rows = [(blob, _encoded(words), today()) for blob, words in words_by_blob.items()]
        with self._lock, self._db:
            self._db.executemany("insert or ignore into words values (?, ?, ?)", rows)

    def forget_unconfirmed(self, before: int, limit: int) -> int:
        """Deletes the words of at most ``limit`` contents last confirmed before the day ``before``,
        least recently confirmed first; returns how many, and frees their space."""
        chosen = "select blob from words where confirmed < ? order by confirmed, blob limit ?"
        with self._lock:
            with self._db:
                forgotten = self._db.execute(f"delete from words where blob in ({chosen})", (before, limit))
            release_free_pages(self._db)
        return forgotten.rowcount

    def confirmations(self, before: int) -> Confirmations:
        with self._lock:
            held, unconfirmed, oldest = self._db.execute(
                "select count(*), count(*) filter (where confirmed < ?), min(confirmed) from words",
                (before,),
            ).fetchone()
        return Confirmations(held, unconfirmed, oldest)

    def _confirm(self, blobs: list[str]) -> None:
        """A stamp that fails is logged and never costs the lookup: the words the read found are kept."""
        try:
            with self._lock, self._db:
                self._db.execute(
                    "update words set confirmed = ? where blob in (select value from json_each(?))",
                    (today(), json.dumps(blobs)),
                )
        except sqlite3.Error as error:
            _logger.warning("spelling table %s: %d contents not stamped: %s", self.path, len(blobs), error)


def table_path(root: Path | None = None) -> Path:
    """The table file this JVN reads, in ``root`` or else in ``user_spelling_tables()``."""
    return (root or user_spelling_tables()) / f"{table_identity()}.sqlite"


def user_spelling_tables() -> Path:
    """The folder of the spelling tables every index on this machine shares, one per identity."""
    return cache_root() / "spellings"


@cache
def table_identity() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _encoded(words: Words) -> str:
    return json.dumps(words, separators=(",", ":"))


def _decoded(text: str) -> Words:
    return {word: tuple(lines) for word, lines in json.loads(text).items()}


_SCHEMA = """
create table words (
    blob text primary key,
    words text not null,
    confirmed integer not null
) without rowid;
create index words_by_confirmation on words (confirmed, blob);
"""
