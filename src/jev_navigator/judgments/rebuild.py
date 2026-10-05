"""Rebuild a stored batched request from the repository at its commit, and prove it is the same one.

Without ``keep_requests`` the store keeps no client code: the question wording, each item's ids,
locations, hashes and names, and hashes of the code and of the shared state. A field that can
quote code (a Trace link line, a Find signature) is withheld, so such a request no longer
rebuilds exactly; the mismatch then names the withheld fields first. ``rebuild_request`` re-reads
each item's code from an index at that commit, adds the shared state the caller supplies, masks it
as the judge did (a value hidden anywhere in the request is hidden everywhere), and compares the
request hash with the stored one; when they differ it says which part changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..index.code_index import CodeIndex
from ..index.spans import Span
from ..index.units import read_ranges
from .judge import CODE_FIELD
from .questions import content_hash, request_sha256
from .secrets import DEFAULT_MASKER, Masker, mask_request
from .store import AnswerRecord


@dataclass(frozen=True)
class RebuiltRequest:
    state: Mapping
    questions: Mapping
    request_sha256: str
    matches: bool
    differences: tuple[str, ...]


def rebuild_request(
    record: AnswerRecord,
    index_at_commit: CodeIndex,
    shared: Mapping,
    *,
    masker: Masker | None = DEFAULT_MASKER,
) -> RebuiltRequest:
    skeleton = record.skeleton
    if not skeleton:
        raise ValueError("this record has no skeleton; only batched checks can be rebuilt")
    places = skeleton.get("places") or [None] * len(skeleton["items"])
    items = [
        _item(index_at_commit, fields, place) for fields, place in zip(skeleton["items"], places, strict=True)
    ]
    state = {**shared, skeleton["list_name"]: items}
    questions = skeleton["questions"]
    if masker:
        state, questions, _ = mask_request(state, questions, masker)
    rebuilt_hash = request_sha256(state, questions)
    matches = rebuilt_hash == record.request_sha256
    differences = () if matches else (*_withheld(skeleton), *_differences(skeleton, state))
    return RebuiltRequest(state, questions, rebuilt_hash, matches, differences)


def _item(index: CodeIndex, fields: Mapping, place: Mapping | None) -> dict:
    """An item's fields with its code re-read: from its place's runs when the judge recorded one,
    else from its own file and lines."""
    if place is not None:
        return {**fields, CODE_FIELD: read_ranges(index, place["file"], place["runs"])}
    first, last = fields["lines"]
    code = index.read_slice(Span(fields["file"], first, last)).text
    return {**fields, CODE_FIELD: code}


def _withheld(skeleton: Mapping) -> tuple[str, ...]:
    withheld = skeleton.get("withheld_fields")
    return (f"item fields withheld from the store: {', '.join(withheld)}",) if withheld else ()


def _differences(skeleton: Mapping, state: Mapping) -> tuple[str, ...]:
    """``state`` is the rebuilt, masked state; its shared part is everything but the item list."""
    found = []
    masked_shared = {key: value for key, value in state.items() if key != skeleton["list_name"]}
    if content_hash(masked_shared) != skeleton["shared_sha256"]:
        found.append("shared state")
    items = state[skeleton["list_name"]]
    places = skeleton.get("places") or [None] * len(items)
    for slot, (item, place, stored_hash) in enumerate(
        zip(items, places, skeleton["item_code_sha256"], strict=True)
    ):
        if content_hash(item[CODE_FIELD]) != stored_hash:
            found.append(f"code of item {slot} ({_location(item, place)})")
    return tuple(found) or ("item fields or question wording",)


def _location(item: Mapping, place: Mapping | None) -> str:
    """Where an item's code was read from: its place's runs, or its own lines."""
    if place is None:
        return f"{item['file']} lines {item['lines'][0]}-{item['lines'][1]}"
    runs = ", ".join(f"{first}-{last}" for first, last in place["runs"])
    return f"{place['file']} lines {runs}"
