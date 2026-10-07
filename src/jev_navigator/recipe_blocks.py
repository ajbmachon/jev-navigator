"""Code-only blocks for small, caller-configured searches.

These collect evidence, not semantic verdicts. In particular a downward chain is
not proof that every call is a guard or runs before a handler. Dynamic middleware,
wrappers and schema bindings enter as explicit attachments from a wiring map.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .index.bindings import Binding
from .index.code_index import CodeIndex
from .index.languages import language_read
from .index.spans import Span, TextHit
from .index.units import LineAnchor
from .sources import Reach, Seeds, pointed_files


@dataclass(frozen=True)
class ChainAttachment:
    """A wiring-map edge, retaining its registration site and certainty.

    The caller supplies router middleware, wrappers, decorators and schema
    validation that static calls alone cannot bind. Unproven edges are returned
    as unknown, never walked as established wiring.
    """

    entry: Span
    applied: Span
    registration: LineAnchor
    relation: str
    proven: bool = False


@dataclass(frozen=True)
class ChainLink:
    source: Span
    target: Span | None
    at: LineAnchor
    relation: str
    proven: bool
    reason: str


@dataclass(frozen=True)
class GuardChain:
    """Downward evidence with every unresolved edge and depth cut visible."""

    units: tuple[Span, ...]
    links: tuple[ChainLink, ...]
    unknown: tuple[ChainLink, ...]
    depth_cut: tuple[Span, ...]


def guard_chain(
    index: CodeIndex,
    entry: Span,
    *,
    attachments: Sequence[ChainAttachment] = (),
    depth: int | None = None,
) -> GuardChain:
    """Walk calls and passed callbacks downward from an entry, never its callers.

    Decorator expressions travel with their function. Only resolved targets are
    expanded; unknown receivers and external calls remain explicit gaps. A depth
    is an optional caller policy, with the unexpanded boundary counted.
    """
    if depth is not None and depth < 0:
        raise ValueError("depth must be non-negative")
    seen = {entry.key: entry}
    pending = deque([(entry, 0)])
    links: list[ChainLink] = []
    cuts: list[Span] = []
    while pending:
        current, distance = pending.popleft()
        if depth is not None and distance >= depth:
            cuts.append(current)
            continue
        first = index.decorator_starts_in(current.file).get(current, current.start)
        view = Span(current.file, first, current.end, current.name)
        outgoing = [
            _chain_link(current, edge.binding.target, current.file, edge.line, "call", edge.binding)
            for edge in index.callee_edges(view)
        ]
        outgoing.extend(
            _chain_link(
                current,
                ref.binding.target if ref.binding else None,
                ref.file,
                ref.line,
                ref.role,
                ref.binding,
            )
            for ref in index.references_in(view)
            if ref.role in {"argument", "decorator"}
        )
        outgoing.extend(
            ChainLink(
                current,
                item.applied,
                item.registration,
                item.relation,
                item.proven,
                "caller-supplied wiring" if item.proven else "unconfirmed wiring",
            )
            for item in attachments
            if item.entry.key == current.key
        )
        for link in outgoing:
            links.append(link)
            target = link.target
            if link.proven and target is not None and target.key not in seen:
                seen[target.key] = target
                pending.append((target, distance + 1))
    return GuardChain(
        tuple(seen.values()), tuple(links), tuple(link for link in links if not link.proven), tuple(cuts)
    )


def _chain_link(
    source: Span, target: Span | None, file: str, line: int, relation: str, binding: Binding | None
) -> ChainLink:
    return ChainLink(
        source,
        target,
        LineAnchor(file, line),
        relation,
        binding is not None and binding.proven,
        binding.reason if binding else "no binding",
    )


@dataclass(frozen=True)
class OwnerDefinitionSource:
    """Definitions bound at actual uses in the seed owners, not all same-named definitions.

    A name explicitly qualified as ``module.member`` keeps its receiver. Failed
    bindings contribute no guessed definition; the chain block exposes such gaps
    when a caller needs a coverage record.
    """

    name: ClassVar[str] = "owner_definition"
    label: ClassVar[str] = "owner-qualified definitions"

    def reach(self, index: CodeIndex, seeds: Seeds) -> Iterable[Reach]:
        requested = frozenset(name.rsplit(".", 1)[-1] for name in seeds.names)
        for file in pointed_files(index, seeds):
            if not language_read(file):
                continue
            owners = [unit for unit in seeds.units if unit.path == file]
            for name in seeds.names:
                leaf = name.rsplit(".", 1)[-1]
                receiver = name.rsplit(".", 1)[0] if "." in name else None
                for hit in index.search_text(leaf, whole_word=True):
                    if hit.file != file or (
                        owners
                        and not any(start <= hit.line <= end for unit in owners for start, end in unit.ranges)
                    ):
                        continue
                    binding = index.binding_of(file, hit.line, leaf, receiver)
                    if binding.target is not None:
                        target = binding.target
                        yield Reach(
                            LineAnchor(target.file, target.start),
                            self.name,
                            f"{file}:{hit.line}:{name}",
                            1,
                            frozenset({name}),
                        )
            # Parsed uses also retain receivers and aliases absent from prose.
            for unit in owners:
                for start, end in unit.ranges:
                    view = Span(file, start, end)
                    bindings = [(edge.name, edge.line, edge.binding) for edge in index.callee_edges(view)]
                    bindings.extend((ref.name, ref.line, ref.binding) for ref in index.references_in(view))
                    for name, line, binding in bindings:
                        if name in requested and binding is not None and binding.target is not None:
                            target = binding.target
                            yield Reach(
                                LineAnchor(target.file, target.start),
                                self.name,
                                f"{file}:{line}:{name}",
                                1,
                                frozenset({name}),
                            )


OWNER_DEFINITIONS = OwnerDefinitionSource()


def value_source(
    index: CodeIndex, key: str, *, aliases: Sequence[str] = (), files: Sequence[str] = ()
) -> tuple[Reach, ...]:
    """Units naming a setting: definitions, defaults, env reads, overrides and docs.

    Matching is lexical, not a claim about the value's role. An explicit alias
    from a wiring map connects a public environment key to an internal setting.
    A scope restricts the search, and is never inferred from a missing result.
    """
    if not key:
        raise ValueError("key must not be empty")
    allowed = frozenset(files) if files else None
    terms = tuple(dict.fromkeys((key, *aliases)))
    index.search_texts(terms)
    return tuple(
        Reach(LineAnchor(hit.file, hit.line), "value_source", term, 1, frozenset({term}))
        for term in terms
        for hit in index.search_text(term, whole_word=True)
        if allowed is None or hit.file in allowed
    )


class PresenceStatus(StrEnum):
    PRESENT = "present"
    ABSENT = "absent among checked places"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PresencePlace:
    """A file or an exact inclusive span where a literal is expected."""

    file: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class PlacePresence:
    place: PresencePlace
    status: PresenceStatus
    matches: tuple[TextHit, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class PresenceCheck:
    literal: str
    places: tuple[PlacePresence, ...]

    @property
    def checked(self) -> int:
        return sum(place.status is not PresenceStatus.UNKNOWN for place in self.places)

    @property
    def status(self) -> PresenceStatus:
        if any(place.status is PresenceStatus.PRESENT for place in self.places):
            return PresenceStatus.PRESENT
        if not self.places or any(place.status is PresenceStatus.UNKNOWN for place in self.places):
            return PresenceStatus.UNKNOWN
        return PresenceStatus.ABSENT


def presence_check(index: CodeIndex, literal: str, places: Sequence[PresencePlace]) -> PresenceCheck:
    """Check a literal only in the enumerated places. No semantic absence proof.

    Excluded, vanished, unreadable and invalid spans stay unknown. Repeated places
    count once. An empty expected-place list can never establish absence.
    """
    if not literal:
        raise ValueError("literal must not be empty")
    results = []
    for place in dict.fromkeys(places):
        reason = ""
        if place.file not in index.files:
            reason = index.not_indexed_files.get(place.file, "outside the index scope")
        elif not language_read(place.file):
            reason = index.text_files_left_out([place.file]).get(place.file, "")
        lines = () if reason else index.lines(place.file)
        reason = reason or index.unavailable_files.get(place.file, "")
        start = 1 if place.start is None else place.start
        end = len(lines) if place.end is None else place.end
        whole_empty_file = not lines and place.start is None and place.end is None
        if not reason and not whole_empty_file and (start < 1 or end < start or end > len(lines)):
            reason = "invalid or unavailable line range"
        if reason:
            results.append(PlacePresence(place, PresenceStatus.UNKNOWN, reason=reason))
            continue
        hits = tuple(
            TextHit(place.file, number, lines[number - 1])
            for number in range(start, end + 1)
            if literal in lines[number - 1]
        )
        results.append(PlacePresence(place, PresenceStatus.PRESENT if hits else PresenceStatus.ABSENT, hits))
    return PresenceCheck(literal, tuple(results))
