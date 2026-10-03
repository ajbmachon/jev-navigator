"""Every Jev answer is kept: by request hash for replay, and by item content for reuse on the next scan.

A record holds hashes, question ids (each with its wording hash), raw answers, the served model, the
thresholds in force, and the source (file, line range, commit) of every code item it judged, so the
request can be rebuilt from the repository at that commit. It holds no request text unless
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
import os
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .answers import Answer, JevResponse, answer_from_json
from .relations import without_quoted_code

DEFAULT_SHARED_STORE = Path.home() / ".cache/jev-navigator/answers.sqlite"
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
    def by_request(self, request_sha256: str) -> AnswerRecord | None: ...

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
        self._records: dict[str, AnswerRecord] = {}
        self._items: dict[str, dict[str, StoredItemAnswer]] = {}
        self._refusals: set[tuple[str, str, int]] = set()
        self._write_lock = threading.Lock()
        self._load()

    def by_request(self, request_sha256: str) -> AnswerRecord | None:
        return self._records.get(request_sha256)

    def records(self) -> tuple[AnswerRecord, ...]:
        return tuple(self._records.values())

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
        self._records[record.request_sha256] = record
        for item_key, question_id in record.item_keys.items():
            by_model = self._items.setdefault(item_key, {})
            by_model[record.model] = StoredItemAnswer(
                answer_from_json(record.answers[question_id]), record.request_sha256, record.model
            )


SHARED_STORE_VERSION = 1
"""The layout of ``SqliteAnswerStore``; a file in any other layout is refused, never migrated."""


class UnsupportedAnswerStoreError(RuntimeError):
    """The shared store file was written in a layout this JVN does not read."""


_SCHEMA = """
create table if not exists answers (request_sha256 text not null, model text not null, record text not null);
create index if not exists answers_by_request on answers (request_sha256, model);
create table if not exists item_answers (
    item_key text not null, model text not null, request_sha256 text not null, answer text not null
);
create index if not exists item_answers_by_key on item_answers (item_key, model);
create table if not exists refusals (
    request_sha256 text not null, route text not null, input_box integer not null
);
create index if not exists refusals_by_request on refusals (request_sha256, route, input_box);
"""


class SqliteAnswerStore:
    """One SQLite file shared by every run on a machine. Rows are only ever inserted, never updated or
    deleted, so no answer is lost; the newest answer for a key wins a lookup. WAL mode lets several
    runs read and write the file at once. A record is stored without its request, sent bytes and
    skeleton, so no code, state or question text reaches this file whatever the caller keeps.

    Provisional (open question O29 for André): the location defaults to ``DEFAULT_SHARED_STORE``
    under the JVN cache root, overridable with ``SHARED_STORE_VARIABLE``, and answers never expire.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._db.execute("pragma journal_mode=wal")
        self._open_current_layout()

    def _open_current_layout(self) -> None:
        """Create the layout in a new file; refuse a file written in any other layout."""
        version = self._db.execute("pragma user_version").fetchone()[0]
        if version == 0 and not self._db.execute("select name from sqlite_master").fetchall():
            self._db.executescript(_SCHEMA)
            self._db.execute(f"pragma user_version = {SHARED_STORE_VERSION}")
            return
        if version != SHARED_STORE_VERSION:
            self._db.close()
            raise UnsupportedAnswerStoreError(
                f"{self.path} holds answer store version {version}, this JVN reads version "
                f"{SHARED_STORE_VERSION}; delete {self.path} and its -wal and -shm files, or point "
                "--answer-store at a new file"
            )

    def by_request(self, request_sha256: str) -> AnswerRecord | None:
        row = self._one(
            "select record from answers where request_sha256 = ? order by rowid desc limit 1",
            (request_sha256,),
        )
        return _record_from_json(json.loads(row[0])) if row else None

    def record_for(self, request_sha256: str, model: str) -> AnswerRecord | None:
        row = self._one(
            "select record from answers where request_sha256 = ? and model = ? order by rowid desc limit 1",
            (request_sha256, model),
        )
        return _record_from_json(json.loads(row[0])) if row else None

    def records(self) -> tuple[AnswerRecord, ...]:
        with self._lock:
            rows = self._db.execute("select record from answers order by rowid").fetchall()
        return tuple(_record_from_json(json.loads(record)) for (record,) in rows)

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        """``served_model`` None accepts an answer from any model (replay)."""
        if served_model is None:
            query, parameters = (
                "select answer, request_sha256, model from item_answers where item_key = ?",
                (item_key,),
            )
        else:
            query = "select answer, request_sha256, model from item_answers where item_key = ? and model = ?"
            parameters = (item_key, served_model)
        row = self._one(f"{query} order by rowid desc limit 1", parameters)
        return StoredItemAnswer(answer_from_json(json.loads(row[0])), row[1], row[2]) if row else None

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

    def refused(self, request_sha256: str, route: str, input_box: int) -> bool:
        query = "select 1 from refusals where request_sha256 = ? and route = ? and input_box = ?"
        return self._one(query, (request_sha256, route, input_box)) is not None

    def put_refusal(self, request_sha256: str, route: str, input_box: int) -> None:
        with self._lock, self._db:
            self._db.execute("insert into refusals values (?, ?, ?)", (request_sha256, route, input_box))

    def _one(self, query: str, parameters: tuple) -> tuple | None:
        with self._lock:
            return self._db.execute(query, parameters).fetchone()


class LayeredAnswerStore:
    """A run's own pack in front of the shared store. Lookups try the pack first; an answer found only
    in the shared store is copied into the pack, so the pack alone replays every answer the run used.
    Every new answer and refusal goes to both."""

    def __init__(self, run: JsonlAnswerStore, shared: SqliteAnswerStore) -> None:
        self.run = run
        self.shared = shared

    def by_request(self, request_sha256: str) -> AnswerRecord | None:
        found = self.run.by_request(request_sha256)
        if found is not None:
            return found
        shared = self.shared.by_request(request_sha256)
        if shared is not None:
            self.run.put(shared)
        return shared

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        found = self.run.by_item(item_key, served_model)
        if found is not None:
            return found
        shared = self.shared.by_item(item_key, served_model)
        if shared is not None and self.run.by_request(shared.request_sha256) is None:
            self.run.put(self.shared.record_for(shared.request_sha256, shared.model))
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


def shared_store_path(environment: Mapping[str, str] | None = None) -> Path:
    """The shared store's file: ``SHARED_STORE_VARIABLE`` when set, else ``DEFAULT_SHARED_STORE``."""
    environment = os.environ if environment is None else environment
    return Path(environment.get(SHARED_STORE_VARIABLE) or DEFAULT_SHARED_STORE)


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
