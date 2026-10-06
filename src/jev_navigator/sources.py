"""Sources: where a search's candidates come from, as code facts without a model.

A source takes seeds and the index and reaches places. A place is a file (every unit listed in it),
an anchor (the unit holding a line, or the lines a range names) or a definition span the index knows
(opened whole). Each place it reaches carries its provenance: the source, the seed it came from, a
distance, the request names it was reached by, and for a link between two pieces of code the
relation in words and how sure the index is of it. A source makes no Jev call.

A source never builds units and never scores them. The search resolves places into units with its
own room and reading, so the units, the anchors that named none and each seed's counts have one
owner. The frontier measures every unit's code features the same way whichever source reached it,
so two sources reaching one unit never score it differently; only the distance is a source's own.
A workflow is a composition: the sources that start it, the sources a unit that clears a target's
bar expands through, the frontier's policy and shares, and Jev judging in queue order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from . import operations
from .index.bindings import Binding
from .index.code_index import CodeIndex
from .index.languages import language_read
from .index.scope import is_lockfile
from .index.spans import Span, TextHit
from .index.units import Anchor, LineAnchor


@dataclass(frozen=True)
class Seeds:
    """What sources start from. ``names`` and ``texts`` come from the request (the texts are the
    targets' descriptions), ``files`` and ``anchors`` from the caller, and ``spans`` are code a search
    already holds: the spans of a unit that cleared a target's bar (``units.unit_spans``), or the code
    find opened, which may be a function, a window of lines or a stretch chosen by position."""

    names: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    anchors: tuple[Anchor, ...] = ()
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True)
class Reach:
    """One place a source reached. ``at`` is a file (every unit listed in it), a line (the unit
    holding it), a range of lines chosen by position, or a definition span the index knows, opened
    whole. ``source`` is the source's name and ``seed`` what it reached the place from: a name, a path,
    an anchor as ``file:line`` or a span's key. ``distance`` is how far the place lies from what the
    caller pointed at, 0 for an anchor; the frontier subtracts it from a unit's value and keeps the
    smallest when several sources reach one unit. ``names`` are the request names the source reached
    the place by; each name's rarity counts them. ``relation`` says in words how the place relates to
    its seed, such as "calls place", and ``binding`` how sure the index is that a call or use reaches
    the seed, for a link the index resolved by name."""

    at: str | Anchor | Span
    source: str
    seed: str
    distance: int
    names: frozenset[str] = frozenset()
    relation: str = ""
    binding: Binding | None = None


class Source(Protocol):
    """A primitive that reaches places from seeds, without a model. ``name`` is how a result's
    provenance names it, and ``label`` how a coverage record counts the units it reached that were left
    unjudged, such as "callers"."""

    @property
    def name(self) -> str: ...

    @property
    def label(self) -> str: ...

    def reach(self, index: CodeIndex, seeds: Seeds) -> Iterable[Reach]:
        """Every place reachable from ``seeds``, in the order the source ranks them."""
        ...


@dataclass(frozen=True)
class AnchorSource:
    """The units the caller's anchors name, at distance 0."""

    name: ClassVar[str] = "anchor"
    label: ClassVar[str] = "from anchors"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [Reach(anchor, self.name, anchor_text(anchor), 0) for anchor in seeds.anchors]


@dataclass(frozen=True)
class FileSource:
    """Every unit of the caller's files: at distance 1 for a file near the anchors (an anchor's own
    file, or a file it imports where the index reads that language), 2 for any other."""

    name: ClassVar[str] = "file"
    label: ClassVar[str] = "from files"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        near = near_files(index, seeds.anchors)
        return [Reach(file, self.name, file, 1 if file in near else 2) for file in dict.fromkeys(seeds.files)]


@dataclass(frozen=True)
class NameSource:
    """The units holding each line a request name occurs on, at distance 3, the rarest name first, so
    a common word never decides which hits of a rare name are seen. With ``text_files`` only the hits
    in files JVN does not parse are kept, never a lockfile's, so a common word never floods a text
    search with a lockfile's pieces."""

    text_files: bool = False
    name: ClassVar[str] = "name"
    label: ClassVar[str] = "from name hits"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        hits = {name: self._hits(index, name) for name in dict.fromkeys(seeds.names)}
        return [
            Reach(LineAnchor(hit.file, hit.line), self.name, name, 3, frozenset({name}))
            for name in sorted(hits, key=lambda name: (len(hits[name]), name))
            for hit in hits[name]
        ]

    def _hits(self, index: CodeIndex, name: str) -> tuple[TextHit, ...]:
        hits = index.search_text(name)
        if not self.text_files:
            return hits
        return tuple(hit for hit in hits if not language_read(hit.file) and not is_lockfile(hit.file))


@dataclass(frozen=True)
class CallerSource:
    """The functions calling each named seed span, where the index reads the calls, at distance 1."""

    name: ClassVar[str] = "caller"
    label: ClassVar[str] = "callers"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(caller.file, caller.start), self.name, span.key, 1)
            for span in seeds.spans
            if span.is_named
            for caller, _ in operations.caller_functions(index, span)
        ]


@dataclass(frozen=True)
class CalleeSource:
    """The functions each seed span calls, where the index reads the calls, at distance 1."""

    name: ClassVar[str] = "callee"
    label: ClassVar[str] = "callees"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(callee.file, callee.start), self.name, span.key, 1)
            for span in seeds.spans
            for callee, _ in operations.callee_functions(index, span)
        ]


@dataclass(frozen=True)
class DefinitionSource:
    """The units defining each request name (functions, classes, constants, assignments, types and
    enums, as ``CodeIndex.find_definition`` finds them), at distance 1."""

    name: ClassVar[str] = "definition"
    label: ClassVar[str] = "definitions"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(span.file, span.start), self.name, name, 1, frozenset({name}))
            for name in dict.fromkeys(seeds.names)
            for span in index.find_definition(name)
        ]


@dataclass(frozen=True)
class ReferenceSource:
    """The units using each request name other than by calling it (passed on, stored, assigned, used as
    a decorator, exported or returned, as ``CodeIndex.find_references`` finds them), at distance 2."""

    name: ClassVar[str] = "reference"
    label: ClassVar[str] = "references"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(reference.file, reference.line), self.name, name, 2, frozenset({name}))
            for name in dict.fromkeys(seeds.names)
            for reference in index.find_references(name)
        ]


@dataclass(frozen=True)
class ImportSource:
    """Every unit of the files that the anchors' files and the seed spans' files import, where the
    index reads the language, at distance 1."""

    name: ClassVar[str] = "import"
    label: ClassVar[str] = "imported files"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(imported, self.name, file, 1)
            for file in pointed_files(index, seeds)
            if language_read(file)
            for imported in index.imports(file)
        ]


@dataclass(frozen=True)
class ImporterSource:
    """Every unit of the files that import the anchors' files or the seed spans' files, at distance 1."""

    name: ClassVar[str] = "importer"
    label: ClassVar[str] = "importing files"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(importer, self.name, file, 1)
            for file in pointed_files(index, seeds)
            for importer in index.dependents(file)
        ]


@dataclass(frozen=True)
class NamedFileSource:
    """Every unit of the scope files the targets' descriptions or the anchors' files name by path or
    run as a module (``operations.files_named_by``), at distance 1: the files JVN parses, or with
    ``text_files`` the rest."""

    text_files: bool = False
    name: ClassVar[str] = "named_file"
    label: ClassVar[str] = "named files"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        named = operations.files_named_by(index, seeds.texts, [anchor.file for anchor in seeds.anchors])
        files = named.text if self.text_files else named.code
        return [Reach(file, self.name, named.named_by[file], 1) for file in files]


@dataclass(frozen=True)
class ModelSource:
    """The Prisma schema's model and view blocks each seed span queries through Prisma Client
    (``operations.queried_models``), at distance 1."""

    name: ClassVar[str] = "model"
    label: ClassVar[str] = "queried models"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(file, block.start), self.name, span.key, 1)
            for span in seeds.spans
            for file, block in operations.queried_models(index, index.read_slice(span).text)
        ]


@dataclass(frozen=True)
class ClientCallSource:
    """The code that queries a Prisma model or view block each seed span overlaps, found by the text of
    its Prisma Client calls (``operations.client_calls``), at distance 1."""

    name: ClassVar[str] = "client_call"
    label: ClassVar[str] = "client calls"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(hit.file, hit.line), self.name, span.key, 1)
            for span in seeds.spans
            for hit, _ in operations.client_calls(index, span)
        ]


ANCHORS = AnchorSource()
FILES = FileSource()
NAMES = NameSource()
TEXT_NAMES = NameSource(text_files=True)
CALLERS = CallerSource()
CALLEES = CalleeSource()
DEFINITIONS = DefinitionSource()
REFERENCES = ReferenceSource()
IMPORTS = ImportSource()
IMPORTERS = ImporterSource()
NAMED_FILES = NamedFileSource()
TEXT_NAMED_FILES = NamedFileSource(text_files=True)
MODELS = ModelSource()
CLIENT_CALLS = ClientCallSource()


def anchor_text(anchor: Anchor) -> str:
    """``file:line`` for a line, ``file:start-end`` for a range."""
    if isinstance(anchor, LineAnchor):
        return f"{anchor.file}:{anchor.line}"
    return f"{anchor.file}:{anchor.start}-{anchor.end}"


def near_files(index: CodeIndex, anchors: Sequence[Anchor]) -> frozenset[str]:
    """The anchors' files in scope and the files they import, where the index reads the language."""
    scope = frozenset(index.files)
    anchored = {anchor.file for anchor in anchors if anchor.file in scope}
    imported = {
        file
        for anchored_file in anchored
        if language_read(anchored_file)
        for file in index.imports(anchored_file)
    }
    return frozenset(anchored | imported)


def pointed_files(index: CodeIndex, seeds: Seeds) -> tuple[str, ...]:
    """The scope files the anchors and the seed spans sit in, in that order, each once."""
    scope = frozenset(index.files)
    pointed = [*(anchor.file for anchor in seeds.anchors), *(span.file for span in seeds.spans)]
    return tuple(file for file in dict.fromkeys(pointed) if file in scope)
