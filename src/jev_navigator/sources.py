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

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar

from . import operations
from .index.bindings import Binding, falls_inside
from .index.code_index import CodeIndex
from .index.languages import language_read
from .index.scope import is_lockfile, is_test_file
from .index.spans import CallEdge, Span, TextHit
from .index.units import Anchor, LineAnchor, RangeAnchor
from .judgments.relations import key_mention

ADJACENT_LINES = 40
FILE_HEAD_LINES = 40
MAX_KEY_HITS = 30
PASSED_ON_ROLES = frozenset(
    {"argument", "decorator", "collection", "assignment", "export", "return", "receiver", "type", "base"}
)
_ENVIRONMENT_READ = re.compile(
    r"""(?:environ(?:\.get)?\(?\[?|getenv\(|process\.env\.)\s*["']?([A-Z][A-Z0-9_]{2,})"""
)
_QUOTED_KEY = re.compile(r"""["'`]([A-Za-z_][\w.:/\-]{5,79})["'`]""")
_KEY_SHAPE = re.compile(r"[._:/-]")
_Item = TypeVar("_Item")


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
    """The functions holding a call to each named seed span, where the index binds the call to that
    definition or leaves it unbound (``names_exactly``), at distance 1. Each definition is its own
    node here, so a call bound to a method is not a call to the class holding it; ``CallSiteSource``
    is the view-shaped sibling find uses."""

    name: ClassVar[str] = "caller"
    label: ClassVar[str] = "callers"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                LineAnchor(caller.file, caller.start),
                self.name,
                span.key,
                1,
                relation=f"calls {span.name}",
                binding=binding,
            )
            for span in seeds.spans
            if span.is_named
            for caller, binding in operations.caller_functions(index, span)
        ]


@dataclass(frozen=True)
class CallSiteSource:
    """The code holding each call to a named seed span's name that may reach the lines the span shows
    (``falls_inside``: a resolved target overlaps them), test files last, at distance 1. Unlike
    ``CallerSource`` it keeps calls outside every function, and calls bound to any definition the
    span overlaps rather than only to the span's own, so a window inside a class still has callers."""

    name: ClassVar[str] = "call_site"
    label: ClassVar[str] = "call sites"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                LineAnchor(site.file, site.line),
                self.name,
                span.key,
                1,
                relation=f"calls {span.name}",
                binding=site.binding,
            )
            for span in seeds.spans
            if span.is_named
            for site in _tests_last(
                (site for site in index.find_callers(span.name) if falls_inside(site.binding, span)),
                lambda site: site.file,
            )
        ]


@dataclass(frozen=True)
class CalleeSource:
    """The definitions each seed span calls, where the index reads the calls: a call's resolved target,
    else every definition of the called name (``operations.link_targets``), at distance 1. Calls the
    index proves come first, then calls whose targets are not all in test files, then the names
    called from the fewest places, as the rarer name says more."""

    name: ClassVar[str] = "callee"
    label: ClassVar[str] = "callees"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                target,
                self.name,
                span.key,
                1,
                relation=f"called by {_span_label(span)}",
                binding=edge.binding,
            )
            for span in seeds.spans
            for edge in sorted(index.callee_edges(span), key=lambda edge: _callee_rank(index, edge))
            for target in operations.link_targets(index, edge.binding, edge.name)
        ]


@dataclass(frozen=True)
class DefinitionSource:
    """The units defining each request name (functions, classes, constants, assignments, types and
    enums, as ``CodeIndex.find_definition`` finds them), at distance 1."""

    name: ClassVar[str] = "definition"
    label: ClassVar[str] = "definitions"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(LineAnchor(span.file, span.start), self.name, name, 1, frozenset({name}), f"defines {name}")
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
            Reach(
                LineAnchor(reference.file, reference.line),
                self.name,
                name,
                2,
                frozenset({name}),
                relation=f"refers to {name} as {reference.role}",
                binding=reference.binding,
            )
            for name in dict.fromkeys(seeds.names)
            for reference in index.find_references(name)
        ]


@dataclass(frozen=True)
class ReferrerSource:
    """The code using a named seed span's name other than by calling it, where the use may reach the
    lines the span shows (``falls_inside``), at distance 1. ``ReferenceSource`` starts from the
    request's names instead and keeps every use."""

    name: ClassVar[str] = "referrer"
    label: ClassVar[str] = "referring code"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                LineAnchor(reference.file, reference.line),
                self.name,
                span.key,
                1,
                relation=f"refers to {span.name} as {reference.role}",
                binding=reference.binding,
            )
            for span in seeds.spans
            if span.is_named
            for reference in index.find_references(span.name)
            if falls_inside(reference.binding, span)
        ]


@dataclass(frozen=True)
class PassedOnSource:
    """The definitions of the names each seed span passes on without calling them (``PASSED_ON_ROLES``:
    as an argument, decorator, collection item, assignment, export, return value, receiver, type or
    base class): a resolved target, else every definition of the name, at distance 1."""

    name: ClassVar[str] = "passed_on"
    label: ClassVar[str] = "passed on"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                target,
                self.name,
                span.key,
                1,
                relation=f"passed on by {_span_label(span)} as {reference.role}",
                binding=reference.binding,
            )
            for span in seeds.spans
            for reference in index.references_in(span)
            if reference.role in PASSED_ON_ROLES
            for target in operations.link_targets(index, reference.binding, reference.name)
        ]


@dataclass(frozen=True)
class ImportSource:
    """Every unit of the files that the anchors' files and the seed spans' files import, where the
    index reads the language, at distance 1."""

    name: ClassVar[str] = "import"
    label: ClassVar[str] = "imported files"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(imported, self.name, file, 1, relation=f"imported by {file}")
            for file in pointed_files(index, seeds)
            if language_read(file)
            for imported in index.imports(file)
        ]


@dataclass(frozen=True)
class ImportedCodeSource:
    """What each seed span imports, re-exports or requires from files in scope, at distance 1. Code
    outside every function and class stands for its module, so all of its file's import statements
    count; a function or class counts only its own lines, as callees already follow the calls it
    makes. A name taken by name reaches its definition in the module the import resolves to; a whole
    module, or a name that module only passes on from elsewhere, reaches the module's start.
    ``ImportSource`` reaches every unit of every file a file imports instead."""

    name: ClassVar[str] = "imported_code"
    label: ClassVar[str] = "imported code"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [reach for span in seeds.spans for reach in self._imported_by(index, span)]

    def _imported_by(self, index: CodeIndex, span: Span) -> list[Reach]:
        module_level = not any(symbol.contains(span.start) for symbol in index.symbols_in(span.file))
        text = "\n".join(index.lines(span.file)) if module_level else index.read_slice(span).text
        importer = span.file if module_level else _span_label(span)
        reaches = []
        for fact, names in index.imports_in(span.file, text):
            relation = f"imported by {importer}" + ("" if fact.proven else f", candidate: {fact.reason}")
            definitions = sorted(
                (
                    found
                    for name in names or ()
                    for found in index.find_definition(name)
                    if found.file == fact.path
                ),
                key=lambda found: found.start,
            )
            reaches += [Reach(found, self.name, span.key, 1, relation=relation) for found in definitions]
            if names is None or not names <= {definition.name for definition in definitions}:
                head = file_head(index, fact.path)
                reaches.append(Reach(head, self.name, span.key, 1, relation=f"start of a module {relation}"))
        return reaches


@dataclass(frozen=True)
class ImporterSource:
    """Every unit of the files that import the anchors' files or the seed spans' files, at distance 1."""

    name: ClassVar[str] = "importer"
    label: ClassVar[str] = "importing files"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(importer, self.name, file, 1, relation=f"imports {file}")
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
    """The Prisma schema's model and view blocks each seed span queries through Prisma Client, found by
    the text of the calls (``operations.queried_models``), each block whole, at distance 1."""

    name: ClassVar[str] = "model"
    label: ClassVar[str] = "queried models"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                Span(file, block.start, block.end, block.name),
                self.name,
                span.key,
                1,
                relation=(
                    f"{block.keyword} {block.name}, which {_span_label(span)} queries by the text "
                    f"`{block.client_call_text}`"
                ),
            )
            for span in seeds.spans
            for file, block in operations.queried_models(index, index.read_slice(span).text)
        ]


@dataclass(frozen=True)
class ClientCallSource:
    """The code that queries a Prisma model or view block each seed span overlaps, found by the text of
    its Prisma Client calls (``operations.client_calls``), test files last, at distance 1."""

    name: ClassVar[str] = "client_call"
    label: ClassVar[str] = "client calls"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                LineAnchor(hit.file, hit.line),
                self.name,
                span.key,
                1,
                relation=f"queries {block.keyword} {block.name} by the text `{block.client_call_text}`",
            )
            for span in seeds.spans
            for hit, block in _tests_last(operations.client_calls(index, span), lambda call: call[0].file)
        ]


@dataclass(frozen=True)
class SameFileSource:
    """The other functions of each seed span's file, or the other blocks of a Prisma schema, nearest to
    the span first, at distance 1; a function nested in another is part of that function. An anonymous
    span first reaches its nearest named container, else its nearest container: a callback in a
    test's callback reaches that test."""

    name: ClassVar[str] = "same_file"
    label: ClassVar[str] = "same file"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(other, self.name, span.key, 1, relation=f"in the same file as {_span_label(span)}")
            for span in seeds.spans
            for other in _same_file_spans(index, span)
        ]


@dataclass(frozen=True)
class KeyMentionSource:
    """The lines elsewhere that mention an environment variable a seed span reads or a key it quotes,
    the rarest key first, at distance 2. A quoted key has at least six characters and a key's shape: a
    dot, underscore, colon, slash or dash. A key found on more than ``MAX_KEY_HITS`` lines is too
    common to point anywhere and is skipped."""

    name: ClassVar[str] = "key_mention"
    label: ClassVar[str] = "key mentions"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [reach for span in seeds.spans for reach in self._mentions(index, span)]

    def _mentions(self, index: CodeIndex, span: Span) -> list[Reach]:
        hits_by_key = {
            key: _lines_mentioning(index, span, key) for key in _keys_in(index.read_slice(span).text)
        }
        usable = [(key, hits) for key, hits in hits_by_key.items() if 0 < len(hits) <= MAX_KEY_HITS]
        return [
            Reach(LineAnchor(hit.file, hit.line), self.name, span.key, 2, relation=key_mention(key))
            for key, hits in sorted(usable, key=lambda item: len(item[1]))
            for hit in hits
        ]


@dataclass(frozen=True)
class CoChangeSource:
    """The start of the two files most often committed together with each seed span's file, at
    distance 2."""

    name: ClassVar[str] = "co_changed"
    label: ClassVar[str] = "files committed together"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                file_head(index, other),
                self.name,
                file,
                2,
                relation=f"start of a file committed with {file} {commits} times",
            )
            for file in dict.fromkeys(span.file for span in seeds.spans)
            for other, commits in index.co_changed_files(file, limit=2)
        ]


@dataclass(frozen=True)
class LinesBeforeSource:
    """Up to ``ADJACENT_LINES`` lines just before each seed span, at distance 1."""

    name: ClassVar[str] = "lines_before"
    label: ClassVar[str] = "lines before"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [
            Reach(
                RangeAnchor(span.file, max(1, span.start - ADJACENT_LINES), span.start - 1),
                self.name,
                span.key,
                1,
                relation=f"the lines before {span.key}",
            )
            for span in seeds.spans
            if span.start > 1
        ]


@dataclass(frozen=True)
class LinesAfterSource:
    """Up to ``ADJACENT_LINES`` lines just after each seed span, at distance 1."""

    name: ClassVar[str] = "lines_after"
    label: ClassVar[str] = "lines after"

    def reach(self, index: CodeIndex, seeds: Seeds) -> list[Reach]:
        return [self._after(index, span) for span in seeds.spans if span.end < len(index.lines(span.file))]

    def _after(self, index: CodeIndex, span: Span) -> Reach:
        end = min(len(index.lines(span.file)), span.end + ADJACENT_LINES)
        return Reach(
            RangeAnchor(span.file, span.end + 1, end),
            self.name,
            span.key,
            1,
            relation=f"the lines after {span.key}",
        )


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
CALL_SITES = CallSiteSource()
REFERRERS = ReferrerSource()
PASSED_ON = PassedOnSource()
IMPORTED_CODE = ImportedCodeSource()
SAME_FILE = SameFileSource()
KEY_MENTIONS = KeyMentionSource()
CO_CHANGED = CoChangeSource()
LINES_BEFORE = LinesBeforeSource()
LINES_AFTER = LinesAfterSource()


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


def file_head(index: CodeIndex, file: str) -> RangeAnchor:
    """The first ``FILE_HEAD_LINES`` lines of ``file``, or all of a shorter file."""
    return RangeAnchor(file, 1, min(len(index.lines(file)), FILE_HEAD_LINES))


def _tests_last(items: Iterable[_Item], file_of: Callable[[_Item], str]) -> list[_Item]:
    return sorted(items, key=lambda item: is_test_file(file_of(item)))


def _span_label(span: Span) -> str:
    return span.name if span.is_named else span.key


def _callee_rank(index: CodeIndex, edge: CallEdge) -> tuple[bool, bool, int]:
    """A callee with no definition reaches nothing, so its call sites are never counted."""
    targets = operations.link_targets(index, edge.binding, edge.name)
    only_tests = bool(targets) and all(is_test_file(target.file) for target in targets)
    return not edge.binding.proven, only_tests, index.call_site_count(edge.name) if targets else 0


def _same_file_spans(index: CodeIndex, span: Span) -> list[Span]:
    functions = (*index.functions_in(span.file), *operations.schema_block_spans(index, span.file))
    outermost = [other for other in functions if not any(_encloses(outer, other) for outer in functions)]
    others = [other for other in outermost if not other.overlaps(span)]
    nearest_first = sorted(others, key=lambda other: (_lines_between(other, span), other.start))
    container = None if span.is_named else _nearest_container(index, span)
    return ([container] if container is not None else []) + nearest_first


def _nearest_container(index: CodeIndex, span: Span) -> Span | None:
    """The smallest named symbol holding ``span``, else the smallest symbol holding it."""
    containers = [
        other for other in index.symbols_in(span.file) if other != span and other.contains(span.start)
    ]
    named = [other for other in containers if other.is_named]
    return min(named or containers, key=Span.size, default=None)


def _encloses(outer: Span, inner: Span) -> bool:
    same_lines = (outer.start, outer.end) == (inner.start, inner.end)
    return not same_lines and outer.start <= inner.start and inner.end <= outer.end


def _lines_between(other: Span, span: Span) -> int:
    return span.start - other.end if other.end < span.start else other.start - span.end


def _keys_in(code: str) -> list[str]:
    quoted = [key for key in _QUOTED_KEY.findall(code) if _KEY_SHAPE.search(key)]
    return list(dict.fromkeys(_ENVIRONMENT_READ.findall(code) + quoted))


def _lines_mentioning(index: CodeIndex, span: Span, key: str) -> list[TextHit]:
    return [
        hit
        for hit in index.search_text(key, MAX_KEY_HITS + 1, whole_word=True)
        if not (hit.file == span.file and span.contains(hit.line))
    ]
