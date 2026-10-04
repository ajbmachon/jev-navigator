"""Every Jev answer is kept: by request hash for replay, and by item content for reuse on the next scan.

A record holds hashes, question ids (each with its wording hash), raw answers, the served model, the
thresholds in force, and the source (file, line range, commit) of every code item it judged, so a
request can be rebuilt from the repository at that commit, exactly unless its items carried a field
that can quote code, which is withheld (see ``rebuild``). It holds no request text unless
``keep_requests`` is set, because a request carries code the library cannot know the owner of; set
it only for your own or open-source code.

With ``keep_requests`` a record keeps the request twice: ``request``, written with sorted keys for
reading, and the body as it was sent (``sent_body_base64``; ``sent_exact`` when the client's transport
captured the wire bytes, else the body as the library handed it over). Jev can answer the two orders
differently, so ask again only from ``record.sent_request()``, with a judge that has no store (one
with this store answers from it).

``SqliteAnswerStore`` is one store shared by every run on a machine, so a repeated run on unchanged
code asks nothing again. It never holds code, state or question text, in any mode: only hashes,
unit locations, batch member ids, the model, raw answers and timestamps. ``LayeredAnswerStore`` puts
a run's own pack in front of it, so the pack still holds every answer the run used.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ..cache_root import cache_root
from ..confirmation import Confirmations, today
from ..shared_database import open_shared_database, release_free_pages
from .answers import Answer, JevResponse, answer_from_json
from .relations import without_quoted_code

SHARED_STORE_VARIABLE = "JEV_NAVIGATOR_ANSWER_STORE"
SKELETON_ITEM_FIELDS = frozenset(
    {"file", "lines", "commit", "file_sha256", "reached_by", "span_key", "name", "place"}
)
"""The item fields a run pack keeps without ``keep_requests``: ids, locations, hashes and names. Every
other field, such as a Trace link line or a Find signature, can quote code and is withheld; a
``reached_by`` that quotes a key is rendered from the unit's location."""
INPUT_BUDGET_REFUSAL = "input_budget_refusal"
"""A store line recording that a route refused one exact request for its input size."""


@dataclass(frozen=True)
class AnswerRecord:
    request_sha256: str
    question_ids: tuple[str, ...]
    answers: Mapping[str, dict]
    model: str
    input_tokens: int | None
    thresholds: Mapping[str, float]
    item_keys: Mapping[str, str] = field(default_factory=dict)
    sources: Mapping[str, Mapping] = field(default_factory=dict)
    skeleton: Mapping = field(default_factory=dict)
    request: Mapping | None = None
    recorded_at: str = ""
    sent_body_base64: str | None = None
    sent_exact: bool = False
    batch: Mapping = field(default_factory=dict)

    def response(self) -> JevResponse:
        """The stored answers as a replayed response. Replaying sends nothing, so it reports no
        token count."""
        answers = {question_id: answer_from_json(raw) for question_id, raw in self.answers.items()}
        return JevResponse(answers, self.model, None, self.request_sha256, from_store=True)

    def sent_request(self) -> tuple[dict, dict]:
        """The state and questions as they were sent, every key in its sent order."""
        if self.sent_body_base64 is None:
            raise ValueError("this record keeps no request; store answers with keep_requests=True")
        body = json.loads(base64.b64decode(self.sent_body_base64))
        return body["state"], body["questions"]


@dataclass(frozen=True)
class StoredItemAnswer:
    """One item's stored answer and the request that produced it."""

    answer: Answer
    request_sha256: str
    model: str


class AnswerStore(Protocol):
    def by_request(self, request_sha256: str, served_model: str | None) -> AnswerRecord | None: ...

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None: ...

    def put(self, record: AnswerRecord) -> None: ...

    def refused(self, request_sha256: str, route: str, input_box: int) -> bool: ...

    def put_refusal(self, request_sha256: str, route: str, input_box: int) -> None: ...


class JsonlAnswerStore:
    """Append-only JSON lines. ``item_keys`` maps an item key (item content hash, shared-state hash,
    question id with its wording hash, batch membership hash) to the question id that answered it;
    lookups also match the served model recorded with the answer. Input-size refusals are kept as
    ``input_budget_refusal`` lines keyed by request hash, route and the input box in force."""

    def __init__(self, path: Path, *, keep_requests: bool = False) -> None:
        self.path = Path(path)
        self.keep_requests = keep_requests
        self._records: dict[str, dict[str, AnswerRecord]] = {}
        self._items: dict[str, dict[str, StoredItemAnswer]] = {}
        self._refusals: set[tuple[str, str, int]] = set()
        self._write_lock = threading.Lock()
        self._load()

    def by_request(self, request_sha256: str, served_model: str | None) -> AnswerRecord | None:
        """``served_model`` None accepts a record from any model (replay), the newest first."""
        by_model = self._records.get(request_sha256, {})
        if served_model is None:
            return next(reversed(by_model.values()), None)
        return by_model.get(served_model)

    def records(self) -> tuple[AnswerRecord, ...]:
        return tuple(record for by_model in self._records.values() for record in by_model.values())

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        """``served_model`` None accepts an answer from any model (replay)."""
        answers = self._items.get(item_key, {})
        if served_model is None:
            return next(reversed(answers.values()), None)
        return answers.get(served_model)

    def refused(self, request_sha256: str, route: str, input_box: int) -> bool:
        """Whether ``route`` refused this exact request for its input size before, under the same
        input box."""
        return (request_sha256, route, input_box) in self._refusals

    def put_refusal(self, request_sha256: str, route: str, input_box: int) -> None:
        line = {"kind": INPUT_BUDGET_REFUSAL, "request_sha256": request_sha256, "route": route}
        with self._write_lock:
            self._append({**line, "input_box": input_box})
            self._refusals.add((request_sha256, route, input_box))

    def put(self, record: AnswerRecord) -> None:
        stored = record if self.keep_requests else _without_request(record)
        stored = _stamped(stored)
        with self._write_lock:
            self._append(asdict(stored))
            self._index(stored)

    def _append(self, line: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as lines:
            lines.write(json.dumps(line, sort_keys=True) + "\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            kind = raw.get("kind")
            if kind == INPUT_BUDGET_REFUSAL:
                self._refusals.add((raw["request_sha256"], raw["route"], raw["input_box"]))
            elif kind != "llm_step":
                self._index(_record_from_json(raw))

    def _index(self, record: AnswerRecord) -> None:
        by_model = self._records.setdefault(record.request_sha256, {})
        by_model.pop(record.model, None)
        by_model[record.model] = record
        for item_key, question_id in record.item_keys.items():
            by_model = self._items.setdefault(item_key, {})
            by_model[record.model] = StoredItemAnswer(
                answer_from_json(record.answers[question_id]), record.request_sha256, record.model
            )


SHARED_STORE_VERSION = 2
"""The layout of ``SqliteAnswerStore``; a file in any other layout is refused, never migrated. The
default file names it, so JVN versions with different layouts never share one default file."""
_DEFAULT_STORE_FILE = re.compile(r"\.?answers(?:-v\d+)?\.sqlite(?:-wal|-shm|\.\w+)?")
_logger = logging.getLogger(__name__)


class UnsupportedAnswerStoreError(RuntimeError):
    """The shared store file was written in a layout this JVN does not read."""


class StoreInCacheFolderError(ValueError):
    """A store the user names lies inside JVN's cache folder, which housekeeping prunes."""


_SCHEMA = """
create table answers (request_sha256 text not null, model text not null, record text not null);
create index answers_by_request on answers (request_sha256, model);
create table item_answers (
    item_key text not null, model text not null, request_sha256 text not null, answer text not null
);
create index item_answers_by_key on item_answers (item_key, model);
create table refusals (request_sha256 text not null, route text not null, input_box integer not null);
create index refusals_by_request on refusals (request_sha256, route, input_box);
create table confirmations (request_sha256 text primary key, confirmed integer not null) without rowid;
create index confirmations_by_day on confirmations (confirmed, request_sha256);
"""
_CONFIRM = (
    "insert into confirmations values (?, ?)"
    " on conflict (request_sha256) do update set confirmed = excluded.confirmed"
)
_CONFIRMED_ON = "left join confirmations c on c.request_sha256 = {table}.request_sha256"


class SqliteAnswerStore:
    """One SQLite file shared by every run on a machine. Answers are inserted, never changed; the newest
    answer for a key wins a lookup. WAL mode lets several runs read and write the file at once. A
    record is stored without its request, sent bytes and skeleton, so no code, state or question
    text reaches this file whatever the caller keeps.

    Each request carries the day a run last stored or reused one of its answers or its refusal; a
    reuse on an earlier day restamps it, so a same-day replay writes nothing, and a stamp that fails
    is logged and never costs the reuse. ``forget_unconfirmed`` deletes a request's answers, item
    answers and refusals together. Housekeeping calls it only for ``default_shared_store()``, where a
    request unused for 30 days goes (André, 04.10.2026, replacing O29 for the default store); a store
    at a path the user names keeps every answer.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._db = open_shared_database(self.path, _SCHEMA, SHARED_STORE_VERSION)
        self._refuse_another_layout()

    def _refuse_another_layout(self) -> None:
        """A new file is created whole in the current layout (see ``open_shared_database``), so any
        other version is a file written by another JVN, never one still being created."""
        version = self._db.execute("pragma user_version").fetchone()[0]
        if version != SHARED_STORE_VERSION:
            self._db.close()
            raise UnsupportedAnswerStoreError(
                f"{self.path} holds answer store version {version}, and this JVN reads version "
                f"{SHARED_STORE_VERSION}; point --answer-store or {SHARED_STORE_VARIABLE} at a new file"
            )

    def by_request(self, request_sha256: str, served_model: str | None) -> AnswerRecord | None:
        """``served_model`` None accepts a record from any model (replay), the newest first."""
        query = (
            "select answers.record, answers.request_sha256, c.confirmed from answers "
            f"{_CONFIRMED_ON.format(table='answers')} where answers.request_sha256 = ?"
        )
        parameters: tuple = (request_sha256,)
        if served_model is not None:
            query, parameters = f"{query} and answers.model = ?", (request_sha256, served_model)
        row = self._reused(f"{query} order by answers.rowid desc limit 1", parameters)
        return _record_from_json(json.loads(row[0])) if row else None

    def records(self) -> tuple[AnswerRecord, ...]:
        with self._lock:
            rows = self._db.execute("select record from answers order by rowid").fetchall()
        return tuple(_record_from_json(json.loads(record)) for (record,) in rows)

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        """``served_model`` None accepts an answer from any model (replay)."""
        query = (
            "select item_answers.answer, item_answers.request_sha256, c.confirmed, item_answers.model "
            f"from item_answers {_CONFIRMED_ON.format(table='item_answers')} where item_answers.item_key = ?"
        )
        parameters: tuple = (item_key,)
        if served_model is not None:
            query, parameters = f"{query} and item_answers.model = ?", (item_key, served_model)
        row = self._reused(f"{query} order by item_answers.rowid desc limit 1", parameters)
        return StoredItemAnswer(answer_from_json(json.loads(row[0])), row[1], row[3]) if row else None

    def put(self, record: AnswerRecord) -> None:
        stored = _stamped(_code_free(record))
        items = [
            (item_key, stored.model, stored.request_sha256, json.dumps(stored.answers[question_id]))
            for item_key, question_id in stored.item_keys.items()
        ]
        with self._lock, self._db:
            self._db.execute(
                "insert into answers values (?, ?, ?)",
                (stored.request_sha256, stored.model, json.dumps(asdict(stored), sort_keys=True)),
            )
            self._db.executemany("insert into item_answers values (?, ?, ?, ?)", items)
            self._db.execute(_CONFIRM, (stored.request_sha256, today()))

    def refused(self, request_sha256: str, route: str, input_box: int) -> bool:
        query = (
            "select refusals.input_box, refusals.request_sha256, c.confirmed from refusals "
            f"{_CONFIRMED_ON.format(table='refusals')} "
            "where refusals.request_sha256 = ? and refusals.route = ? and refusals.input_box = ?"
        )
        return self._reused(query, (request_sha256, route, input_box)) is not None

    def put_refusal(self, request_sha256: str, route: str, input_box: int) -> None:
        with self._lock, self._db:
            self._db.execute("insert into refusals values (?, ?, ?)", (request_sha256, route, input_box))
            self._db.execute(_CONFIRM, (request_sha256, today()))

    def forget_unconfirmed(self, before: int, limit: int) -> int:
        """Deletes the answers, item answers and refusals of at most ``limit`` requests last used before
        the day ``before``, least recently used first; returns how many, and frees their space."""
        chosen = (
            "select request_sha256 from confirmations where confirmed < ? "
            "order by confirmed, request_sha256 limit ?"
        )
        with self._lock:
            with self._db:
                for table in ("answers", "item_answers", "refusals"):
                    self._db.execute(
                        f"delete from {table} where request_sha256 in ({chosen})", (before, limit)
                    )
                forgotten = self._db.execute(
                    f"delete from confirmations where request_sha256 in ({chosen})", (before, limit)
                )
            release_free_pages(self._db)
        return forgotten.rowcount

    def confirmations(self, before: int) -> Confirmations:
        with self._lock:
            held, unconfirmed, oldest = self._db.execute(
                "select count(*), count(*) filter (where confirmed < ?), min(confirmed) from confirmations",
                (before,),
            ).fetchone()
        return Confirmations(held, unconfirmed, oldest)

    def _reused(self, query: str, parameters: tuple) -> tuple | None:
        """The row ``query`` finds, whose second and third columns are its request and that request's
        confirmation day; a request last confirmed on an earlier day is stamped today."""
        with self._lock:
            row = self._db.execute(query, parameters).fetchone()
            if row is not None and (row[2] is None or row[2] < today()):
                self._stamp(row[1])
        return row

    def _stamp(self, request_sha256: str) -> None:
        try:
            with self._db:
                self._db.execute(_CONFIRM, (request_sha256, today()))
        except sqlite3.Error as error:
            _logger.warning("answer store %s: reuse of %s not stamped: %s", self.path, request_sha256, error)


class LayeredAnswerStore:
    """A run's own pack in front of the shared store. Lookups try the pack first; an answer found only
    in the shared store is copied into the pack, so the pack alone replays every answer the run used.
    Every new answer and refusal goes to both."""

    def __init__(self, run: JsonlAnswerStore, shared: SqliteAnswerStore) -> None:
        self.run = run
        self.shared = shared

    def by_request(self, request_sha256: str, served_model: str | None) -> AnswerRecord | None:
        found = self.run.by_request(request_sha256, served_model)
        if found is not None:
            return found
        shared = self.shared.by_request(request_sha256, served_model)
        if shared is not None:
            self.run.put(shared)
        return shared

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        found = self.run.by_item(item_key, served_model)
        if found is not None:
            return found
        shared = self.shared.by_item(item_key, served_model)
        if shared is not None and self.run.by_request(shared.request_sha256, shared.model) is None:
            self.run.put(self.shared.by_request(shared.request_sha256, shared.model))
        return shared

    def records(self) -> tuple[AnswerRecord, ...]:
        return self.run.records()

    def put(self, record: AnswerRecord) -> None:
        self.run.put(record)
        self.shared.put(record)

    def refused(self, request_sha256: str, route: str, input_box: int) -> bool:
        return self.run.refused(request_sha256, route, input_box) or self.shared.refused(
            request_sha256, route, input_box
        )

    def put_refusal(self, request_sha256: str, route: str, input_box: int) -> None:
        self.run.put_refusal(request_sha256, route, input_box)
        self.shared.put_refusal(request_sha256, route, input_box)


def _record_from_json(raw: dict) -> AnswerRecord:
    raw["question_ids"] = tuple(raw["question_ids"])
    return AnswerRecord(**raw)


def run_answer_store(pack: Path, shared: Path | None = None) -> LayeredAnswerStore:
    """A run's own pack at ``pack`` in front of the shared store at ``shared``, else at
    ``shared_store_path()``."""
    return LayeredAnswerStore(JsonlAnswerStore(pack), SqliteAnswerStore(shared or shared_store_path()))


def shared_store_path(named: str | None = None) -> Path:
    """The shared store's file: ``named`` (the ``--answer-store`` flag) when given, else
    ``SHARED_STORE_VARIABLE`` when set, else ``default_shared_store()``. A store the user names lies
    outside ``cache_root()``, because housekeeping prunes that folder and never touches a named store."""
    if named:
        return _named_store(named, "--answer-store")
    if variable := os.environ.get(SHARED_STORE_VARIABLE):
        return _named_store(variable, SHARED_STORE_VARIABLE)
    return default_shared_store()


def _named_store(named: str, source: str) -> Path:
    path, folder = Path(named).expanduser().resolve(), cache_root().resolve()
    if path.is_relative_to(folder):
        raise StoreInCacheFolderError(
            f"{source} names {path}, inside JVN's cache folder {folder}; JVN prunes that folder, so keep "
            "a store you name elsewhere"
        )
    return path


def default_shared_store() -> Path:
    return cache_root() / f"answers-v{SHARED_STORE_VERSION}.sqlite"


def is_default_store_file(name: str) -> bool:
    """Whether ``name`` is a file of a default store in any layout: the store, its write-ahead log and
    shared memory, or a store still being created."""
    return _DEFAULT_STORE_FILE.fullmatch(name) is not None


def _code_free(record: AnswerRecord) -> AnswerRecord:
    return replace(_without_request(record), skeleton={})


def _without_request(record: AnswerRecord) -> AnswerRecord:
    return AnswerRecord(
        **{
            **asdict(record),
            "request": None,
            "skeleton": _code_free_skeleton(record.skeleton),
            "sources": {asked: _code_free_fields(fields) for asked, fields in record.sources.items()},
            "sent_body_base64": None,
            "sent_exact": False,
            "question_ids": record.question_ids,
        }
    )


def _code_free_skeleton(skeleton: Mapping) -> dict:
    """The skeleton with each item cut to ``SKELETON_ITEM_FIELDS``, naming the fields withheld."""
    if not skeleton:
        return dict(skeleton)
    items = skeleton["items"]
    withheld = sorted({name for item in items for name in item if name not in SKELETON_ITEM_FIELDS})
    return {**skeleton, "items": [_code_free_fields(item) for item in items], "withheld_fields": withheld}


def _code_free_fields(fields: Mapping) -> dict:
    """A unit's fields cut to ``SKELETON_ITEM_FIELDS``, with a ``reached_by`` that quotes code
    rendered from the unit's location."""
    kept = {name: value for name, value in fields.items() if name in SKELETON_ITEM_FIELDS}
    if "reached_by" in kept:
        line = kept["lines"][0] if kept.get("lines") else 0
        kept["reached_by"] = without_quoted_code(kept["reached_by"], kept.get("file", ""), line)
    return kept


def _stamped(record: AnswerRecord) -> AnswerRecord:
    stamp = record.recorded_at or datetime.now(UTC).isoformat(timespec="seconds")
    return AnswerRecord(**{**asdict(record), "recorded_at": stamp, "question_ids": record.question_ids})
