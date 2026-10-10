"""The agent search request: one closed, typed request a calling agent writes, and its parser.

The msgspec Structs below own the request's shape, limits and defaults. ``agent_search_schema`` exports
them as JSON Schema, which ``schemas/agent-search-v1.json`` holds for a tool description to carry;
``python -m jev_navigator.directives.agent_search_request`` prints it. Every object is closed: an
unknown field is refused with the JSON path that names it, so an agent can correct its own request.

A request holds hypotheses, each a mechanism in plain words plus the evidence points that would show it
and the points that would refute it, and the composition the search runs: terms, anchors, files, scope
and the relations to follow. Only point text is ever judged; a mechanism is returned as written.
Hypothesis and point ids are letters and digits, so a point's id ``h1.e1`` and its state key ``h1_e1``
map one to one.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Mapping
from typing import Annotated, Any, Literal

import msgspec
from msgspec import Meta, Struct

SCHEMA_ID = "https://jev-navigator/schemas/agent-search-v1.json"
MAX_HYPOTHESES = 3
MAX_EVIDENCE_POINTS = 4
MAX_REFUTING_POINTS = 2
DEFAULT_BUDGET_REQUESTS = 8

Id = Annotated[
    str,
    Meta(
        pattern=r"^[A-Za-z][A-Za-z0-9]{0,15}$",
        description="Letters and digits, starting with a letter, such as h1, e2 or r1.",
    ),
]
PointText = Annotated[
    str,
    Meta(
        min_length=1,
        max_length=600,
        description="One concrete piece of code in plain words, such as 'code that refuses an order "
        "over the item limit'.",
    ),
]
NonEmpty = Annotated[str, Meta(min_length=1)]
Follow = Literal["callers", "callees", "named_files"]


class Point(Struct, forbid_unknown_fields=True, frozen=True):
    """A point to search for: code the agent expects to find, judged one unit at a time."""

    id: Id
    point: PointText


class Hypothesis(Struct, forbid_unknown_fields=True, frozen=True):
    """One candidate explanation. The mechanism is returned as written and never judged; the
    evidence points would show it and the refuting points would contradict it."""

    id: Id
    mechanism: Annotated[str, Meta(min_length=1, max_length=2000)]
    evidence: Annotated[tuple[Point, ...], Meta(min_length=1, max_length=MAX_EVIDENCE_POINTS)]
    refuted_by: Annotated[tuple[Point, ...], Meta(max_length=MAX_REFUTING_POINTS)]

    def __post_init__(self) -> None:
        ids = [point.id for point in (*self.evidence, *self.refuted_by)]
        if repeated := sorted({id for id in ids if ids.count(id) > 1}):
            raise ValueError(f"hypothesis {self.id}: point ids must be unique; repeated {repeated}")


class Anchor(Struct, forbid_unknown_fields=True, frozen=True):
    """Lines to start from, relative to the repository root: ``line``, or ``line`` to ``end``."""

    file: NonEmpty
    line: Annotated[int, Meta(ge=1)]
    end: Annotated[int, Meta(ge=1)] | None = None

    def __post_init__(self) -> None:
        if self.end is not None and self.end < self.line:
            raise ValueError(f"anchor {self.file}:{self.line}: end {self.end} is before line {self.line}")


class SearchScope(Struct, forbid_unknown_fields=True, frozen=True):
    """The files the search may reach beyond the anchors: folders or globs to include (empty for
    all) and exclude, and whether test files count."""

    include: tuple[NonEmpty, ...]
    exclude: tuple[NonEmpty, ...]
    with_tests: bool


class AgentSearchRequest(Struct, forbid_unknown_fields=True, frozen=True):
    """One composed search: hypotheses with their points, and how to search for them."""

    hypotheses: Annotated[tuple[Hypothesis, ...], Meta(min_length=1, max_length=MAX_HYPOTHESES)]
    terms: Annotated[
        tuple[NonEmpty, ...],
        Meta(
            description="Identifiers, literals and config keys to seed and rank the search; never its scope."
        ),
    ]
    anchors: tuple[Anchor, ...]
    files: tuple[NonEmpty, ...]
    scope: SearchScope
    follow: Annotated[
        tuple[Follow, ...],
        Meta(description="Relations to expand through from the best units of a point still undecided."),
    ]
    budget_requests: Annotated[
        int,
        Meta(ge=1, le=64, description="Judge requests this search may send, role labels included."),
    ] = DEFAULT_BUDGET_REQUESTS

    def __post_init__(self) -> None:
        ids = [hypothesis.id for hypothesis in self.hypotheses]
        if repeated := sorted({id for id in ids if ids.count(id) > 1}):
            raise ValueError(f"hypothesis ids must be unique; repeated {repeated}")
        if len(set(self.follow)) != len(self.follow):
            raise ValueError(f"follow names a relation twice: {list(self.follow)}")


class InvalidAgentSearchRequestError(ValueError):
    """The request does not match ``AgentSearchRequest``; the message names the field and why."""


def parse_agent_search_request(
    raw: AgentSearchRequest | Mapping[str, Any] | str | bytes,
) -> AgentSearchRequest:
    """The request ``raw`` holds, from JSON text or an already decoded mapping."""
    if isinstance(raw, AgentSearchRequest):
        return raw
    try:
        if isinstance(raw, str | bytes):
            return msgspec.json.decode(raw, type=AgentSearchRequest)
        return msgspec.convert(raw, AgentSearchRequest)
    except (msgspec.ValidationError, msgspec.DecodeError) as error:
        raise InvalidAgentSearchRequestError(f"invalid agent search request: {error}") from error


def agent_search_schema() -> dict[str, Any]:
    """The request's JSON Schema, exported from the Structs that parse it. A description taken from a
    docstring is dedented, so every Python version exports the same text: before 3.13 a docstring keeps
    its indentation."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **_dedented(msgspec.json.schema(AgentSearchRequest)),
    }


def _dedented(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            key: inspect.cleandoc(value) if key == "description" else _dedented(value)
            for key, value in schema.items()
        }
    return [_dedented(value) for value in schema] if isinstance(schema, list) else schema


def schema_text() -> str:
    """The schema as ``schemas/agent-search-v1.json`` stores it."""
    return json.dumps(agent_search_schema(), indent=2) + "\n"


if __name__ == "__main__":
    sys.stdout.write(schema_text())
