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
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .answers import Answer, JevResponse, answer_from_json


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

    def response(self) -> JevResponse:
        answers = {question_id: answer_from_json(raw) for question_id, raw in self.answers.items()}
        return JevResponse(answers, self.model, 0, self.request_sha256, from_store=True)

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


class AnswerStore(Protocol):
    def by_request(self, request_sha256: str) -> AnswerRecord | None: ...

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None: ...

    def put(self, record: AnswerRecord) -> None: ...


class JsonlAnswerStore:
    """Append-only JSON lines. ``item_keys`` maps an item key (item content hash, shared-state hash,
    question id with its wording hash) to the question id that answered it; lookups also match the
    served model recorded with the answer."""

    def __init__(self, path: Path, *, keep_requests: bool = False) -> None:
        self.path = Path(path)
        self.keep_requests = keep_requests
        self._records: dict[str, AnswerRecord] = {}
        self._items: dict[str, dict[str, StoredItemAnswer]] = {}
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

    def put(self, record: AnswerRecord) -> None:
        stored = record if self.keep_requests else _without_request(record)
        stored = _stamped(stored)
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
            if raw.get("kind") != "llm_step":
                self._index(_record_from_json(raw))

    def _index(self, record: AnswerRecord) -> None:
        self._records[record.request_sha256] = record
        for item_key, question_id in record.item_keys.items():
            by_model = self._items.setdefault(item_key, {})
            by_model[record.model] = StoredItemAnswer(
                answer_from_json(record.answers[question_id]), record.request_sha256
            )


def _record_from_json(raw: dict) -> AnswerRecord:
    raw["question_ids"] = tuple(raw["question_ids"])
    return AnswerRecord(**raw)


def _without_request(record: AnswerRecord) -> AnswerRecord:
    return AnswerRecord(
        **{
            **asdict(record),
            "request": None,
            "sent_body_base64": None,
            "sent_exact": False,
            "question_ids": record.question_ids,
        }
    )


def _stamped(record: AnswerRecord) -> AnswerRecord:
    stamp = record.recorded_at or datetime.now(UTC).isoformat(timespec="seconds")
    return AnswerRecord(**{**asdict(record), "recorded_at": stamp, "question_ids": record.question_ids})
