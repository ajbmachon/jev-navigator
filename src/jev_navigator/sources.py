"""Sources: where a search's candidates come from, as code facts without a model.

A source takes seeds and the index and reaches places. A place is a file, meaning every unit listed
in it, or an anchor, meaning the unit holding it. Each place it reaches carries its provenance: the
source, the seed it came from, a distance, and the request names it was reached by. A source makes
no Jev call.

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
from .index.code_index import CodeIndex
from .index.languages import language_read
from .index.scope import is_lockfile
from .index.spans import Span, TextHit
from .index.spellings import Spelling, text_words
from .index.units import Anchor, LineAnchor, Unit, UnitKind, read_ranges

SPELLING_FILES = 20
"""The most files a spelling the spelling source follows may be held by: one more common points
nowhere in particular."""


@dataclass(frozen=True)
class Seeds:
    """What sources start from. ``names`` and ``texts`` come from the request (the texts are the
    targets' descriptions), ``files`` and ``anchors`` from the caller, and ``units`` are units a search
    already judged, such as one that cleared a target's bar."""

    names: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    anchors: tuple[Anchor, ...] = ()
    units: tuple[Unit, ...] = ()


@dataclass(frozen=True)
class Reach:
    """One place a source reached. ``at`` is a file (every unit listed in it) or an anchor (the unit
    holding it). ``source`` is the source's name and ``seed`` what it reached the place from: a name,
    a path, an anchor as ``file:line`` or a unit id. ``distance`` is how far the place lies from what
    the caller pointed at, 0 for an anchor; the frontier subtracts it from a unit's value and keeps the
    smallest when several sources reach one unit. ``names`` are the request names the source reached
    the place by; each name's rarity counts them."""

    at: str | Anchor
    source: str
    seed: str
    distance: int
    names: frozenset[str] = frozenset()


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
class SpellingSource:
    """The places each real spelling of a request name or of a word of the targets' descriptions sits
    (``CodeIndex.names``), at distance 3, the rarest spelling first, each reached by the name or word
    it spells: ``website`` reaches the lines spelling ``createWebsite`` and ``WEBSITE_ID`` and every
    unit of ``website.ts``. A spelling held by more than ``max_files`` files is too common to point
    anywhere and is left out; None follows every one. With ``text_files`` only the files JVN does not
    parse are kept, never a lockfile."""

    max_files: int | None = SPELLING_FILES
    text_files: bool = False
    name: ClassVar[str] = "spelling"
    label: ClassVar[str] = "spellings"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        terms = dict.fromkeys((*seeds.names, *text_words(seeds.texts)))
        spelled = sorted(
            (
                (term, spelling)
                for term in terms
                for spelling in index.names(term)
                if self.max_files is None or spelling.files <= self.max_files
            ),
            key=lambda found: (found[1].files, found[1].word, found[0]),
        )
        return [reach for term, spelling in spelled for reach in self._places(term, spelling)]

    def _places(self, term: str, spelling: Spelling) -> list[Reach]:
        kept = [place for place in spelling.places if self._reads(place.file)]
        if spelling.file_name:
            return [Reach(place.file, self.name, term, 3, frozenset({term})) for place in kept]
        return [
            Reach(LineAnchor(place.file, line), self.name, term, 3, frozenset({term}))
            for place in kept
            for line in place.lines
        ]

    def _reads(self, file: str) -> bool:
        if self.text_files:
            return not language_read(file) and not is_lockfile(file)
        return bool(language_read(file))


@dataclass(frozen=True)
class CallerSource:
    """The functions calling each seed unit that is a named function or method, where the index reads
    the calls, at distance 1."""

    name: ClassVar[str] = "caller"
    label: ClassVar[str] = "callers"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(caller.file, caller.start), self.name, unit.id, 1)
            for unit in seeds.units
            if (function := function_span(index, unit)) is not None and function.is_named
            for caller, _ in operations.caller_functions(index, function)
        ]


@dataclass(frozen=True)
class CalleeSource:
    """The functions each seed unit that is a function or method calls, where the index reads the
    calls, at distance 1."""

    name: ClassVar[str] = "callee"
    label: ClassVar[str] = "callees"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(callee.file, callee.start), self.name, unit.id, 1)
            for unit in seeds.units
            if (function := function_span(index, unit)) is not None
            for callee, _ in operations.callee_functions(index, function)
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
    """Every unit of the files that the anchors' files and the seed units' files import, where the
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
    """Every unit of the files that import the anchors' files or the seed units' files, at distance 1."""

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
    """The Prisma schema's model and view blocks each seed unit queries through Prisma Client
    (``operations.queried_models``), at distance 1."""

    name: ClassVar[str] = "model"
    label: ClassVar[str] = "queried models"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(file, block.start), self.name, unit.id, 1)
            for unit in seeds.units
            for file, block in operations.queried_models(index, read_ranges(index, unit.path, unit.ranges))
        ]


@dataclass(frozen=True)
class ClientCallSource:
    """The code that queries each seed unit that is a Prisma model or view block, found by the text of
    its Prisma Client calls (``operations.client_calls``), at distance 1."""

    name: ClassVar[str] = "client_call"
    label: ClassVar[str] = "client calls"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(hit.file, hit.line), self.name, unit.id, 1)
            for unit in seeds.units
            if unit.kind is UnitKind.SCHEMA_BLOCK
            for hit, _ in operations.client_calls(index, Span(unit.path, unit.start, unit.end))
        ]


ANCHORS = AnchorSource()
FILES = FileSource()
NAMES = NameSource()
TEXT_NAMES = NameSource(text_files=True)
SPELLINGS = SpellingSource()
TEXT_SPELLINGS = SpellingSource(text_files=True)
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
    """The scope files the anchors and the seed units sit in, in that order, each once."""
    scope = frozenset(index.files)
    pointed = [*(anchor.file for anchor in seeds.anchors), *(unit.path for unit in seeds.units)]
    return tuple(file for file in dict.fromkeys(pointed) if file in scope)


def function_span(index: CodeIndex, unit: Unit) -> Span | None:
    """The function or method span a unit is, or None for any other unit."""
    if unit.kind not in (UnitKind.FUNCTION, UnitKind.METHOD):
        return None
    return next((span for span in index.functions_in(unit.path) if span.key == unit.id), None)
