"""Mechanical code lookups over a narrowed set of files: no model, only ast-grep, ripgrep and git.

Every lookup stays inside the scope the index was built with. ``max_files`` is an explicit caller
policy only; the index has no default refusal and never drops files from a requested scope.
"""

from __future__ import annotations

import re
import tempfile
import threading
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TypeVar

from . import listing, tools
from .bindings import Binding, BindingResolver, CallFacts, binding_from_facts
from .fact_cache import FactCache
from .imports import (
    ImportFact,
    imported_modules,
    imported_names,
    module_imports,
    reexported_names,
    resolve_import,
)
from .languages import (
    declares_type,
    declares_value,
    language_of,
)
from .memo import memoized
from .name_table import CALL, DEFINITION_KINDS, REFERENCE, FileEntry, NameRow, NameTable, git_blob_id
from .packages import Packages
from .scope_scan import CallMatch, FileFacts, FileStructure, ReferenceMatch, Unparsed, scan_facts
from .source_files import DISAPPEARED, SourceFiles
from .spans import CallEdge, CallSite, CodeSlice, Reference, Span, TextHit
from .tsconfig import ScriptPaths, nearest_script_paths

DEFAULT_WINDOW_RADIUS = 10
MAX_TEXT_HITS = 20
# The bytes kept on either side of a text hit, so a hit in a one-line bundle never holds the line.
TEXT_HIT_CONTEXT_BYTES = 200
CO_CHANGE_COMMITS = 200
LINE_CACHE_FILES = 512
_COMMIT_MARK = "@@commit@@"
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})
_WORD = re.compile(r"[\w$]+")
ScanObserver = Callable[[str, str, int], None]
_Result = TypeVar("_Result")
CoChange = tuple[str, int]


_NO_STRUCTURE = FileStructure((), (), ())
_KIND_ORDER = {kind: rank for rank, kind in enumerate((*DEFINITION_KINDS, CALL, REFERENCE))}


class RevisionMismatchError(ValueError):
    """A directive asked for a revision the index does not hold."""


class ScopeTooWideError(ValueError):
    """The index was asked to cover more files than its limit."""


class UnsafePathError(ValueError):
    """A scope path is a symbolic link or resolves outside the index root, so reading it could leave
    the root."""


class CodeIndex:
    def __init__(
        self,
        root: Path,
        files: Iterable[str],
        *,
        max_files: int | None = None,
        commit: str = "",
        changed_files: Iterable[str] = (),
        git_root: Path | None = None,
        binding_resolver: BindingResolver | None = None,
        scan_observer: ScanObserver | None = None,
        fact_cache_dir: Path | None = None,
        blob_ids: Mapping[str, str] | None = None,
        not_indexed: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.git_root = Path(git_root) if git_root is not None else self.root
        self.binding_resolver = binding_resolver
        self.scan_observer = scan_observer
        self._snapshot: tempfile.TemporaryDirectory | None = None
        self.commit = commit
        self._changed = frozenset(changed_files)
        self.files = tuple(sorted(set(files)))
        if max_files is not None and len(self.files) > max_files:
            raise ScopeTooWideError(
                f"{len(self.files)} files is wider than the limit of {max_files}; narrow the scope"
            )
        _require_inside(self.root, self.files)
        self._scope = frozenset(self.files)
        self._code_files = tuple(path for path in self.files if language_of(path))
        self._unavailable: dict[str, str] = {}
        self._refused: dict[str, str] = {}
        self._not_indexed = dict(not_indexed or {})
        self._sources = SourceFiles(
            self.root, self._unavailable, LINE_CACHE_FILES, _held_weakly(self._standing_first_read)
        )
        self._unparsed = Unparsed()
        self._facts: dict[str, FileFacts] = {}
        self._facts_lock = threading.RLock()
        self._fact_cache = FactCache(fact_cache_dir)
        self._listed_blobs = {
            file: blob for file, blob in (blob_ids or {}).items() if file not in self._changed
        }
        self._blobs: dict[str, str] = {}
        self._name_table = NameTable()
        self._unwritten: dict[str, FileFacts] = {}
        self._entries: dict[str, FileEntry] | None = None
        self._files_by_blob: dict[str, tuple[str, ...]] = {}
        self._file_order: dict[str, int] = {}
        self._reached: set[str] = set()
        self._incomplete_in_table: frozenset[str] = frozenset()

    @classmethod
    def from_git(
        cls,
        root: Path,
        prefixes: Sequence[str] = (),
        *,
        max_files: int | None = None,
        binding_resolver: BindingResolver | None = None,
        scan_observer: ScanObserver | None = None,
        fact_cache_dir: Path | None = None,
    ) -> CodeIndex:
        """The regular files git tracks under ``prefixes`` (every one when none are given), read from the
        checkout at its commit; for a repository root with an explicit path list. Symbolic links and
        submodules are left out: a link can point outside the scope, or at a directory. Every file under
        the root that is not tracked, and a requested path with no file, is named in
        ``not_indexed_files``; ``from_directory`` indexes untracked files too."""
        root = Path(root)
        blobs = _regular_blobs(tools.git(["ls-files", "--stage", "-z", "--", *prefixes], root))
        commit = tools.git(["rev-parse", "HEAD"], root).strip()
        return cls(
            root,
            list(blobs),
            max_files=max_files,
            commit=commit,
            changed_files=_changed_under(root, prefixes),
            binding_resolver=binding_resolver,
            scan_observer=scan_observer,
            fact_cache_dir=fact_cache_dir,
            blob_ids=blobs,
            not_indexed=listing.left_out_of_tracked(root, prefixes, blobs),
        )

    @classmethod
    def from_directory(
        cls,
        root: Path,
        prefixes: Sequence[str] = (),
        *,
        max_files: int | None = None,
        exclude_paths: Sequence[Path] = (),
        binding_resolver: BindingResolver | None = None,
        scan_observer: ScanObserver | None = None,
        fact_cache_dir: Path | None = None,
    ) -> CodeIndex:
        """Index current files, tracked by git or not, minus ignored ones (see ``listing.working_files``),
        with Git metadata when available.

        Modified and untracked files are included; ignored ones are named in ``not_indexed_files``. In a
        Git worktree, slices carry the current HEAD plus ``+worktree`` when their file differs; outside
        Git they carry no commit. Every slice also carries its current file SHA-256, so either case
        identifies the inspected bytes.
        """
        root = Path(root)
        excluded = tuple(path.resolve() for path in exclude_paths)
        listed = listing.working_files(root, prefixes)
        files = tuple(
            file
            for file in listed.files
            if not any((root / file).resolve().is_relative_to(path) for path in excluded)
        )
        commit, changed, blobs = _working_git_metadata(root, prefixes)
        return cls(
            root,
            files,
            max_files=max_files,
            commit=commit,
            changed_files=changed,
            binding_resolver=binding_resolver,
            scan_observer=scan_observer,
            fact_cache_dir=fact_cache_dir,
            blob_ids=blobs,
            not_indexed=listed.not_indexed,
        )

    @classmethod
    def at_commit(
        cls,
        repository: Path,
        commit: str,
        prefixes: Sequence[str] = (),
        *,
        max_files: int | None = None,
    ) -> CodeIndex:
        """The regular files under ``prefixes`` as they were at ``commit``, read from git objects into a
        private temporary directory; the checkout is never touched. History lookups still run in
        ``repository``. Symbolic links and submodules are left out, as in ``from_git``. The commit's
        tsconfig files come along (outside the scope), so path aliases resolve."""
        repository = Path(repository)
        sha = tools.git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repository).strip()
        blobs = _regular_blobs(tools.git(["ls-tree", "-r", "-z", sha, "--", *prefixes], repository))
        listed = list(blobs)
        if max_files is not None and len(listed) > max_files:
            raise ScopeTooWideError(
                f"{len(listed)} files is wider than the limit of {max_files}; narrow the scope"
            )
        snapshot = tempfile.TemporaryDirectory(prefix=f"jev-navigator-{sha[:8]}-")
        tools.export_blobs(repository, _blobs_to_export(repository, sha, listed), Path(snapshot.name))
        index = cls(
            snapshot.name, listed, max_files=max_files, commit=sha, git_root=repository, blob_ids=blobs
        )
        index._snapshot = snapshot
        return index

    def require_commit(self, commit: str) -> None:
        """Raises unless this index reads exactly ``commit``: a working-tree index with uncommitted
        changes in scope never counts. Use ``CodeIndex.at_commit`` for a historical revision."""
        if not self.commit or not self.commit.startswith(commit) or len(commit) < 7:
            raise RevisionMismatchError(
                f"the index reads {self.commit or 'an unknown revision'}, not {commit}"
            )
        if self._changed:
            raise RevisionMismatchError(
                f"the working tree has uncommitted changes in {sorted(self._changed)[:3]}"
            )

    @property
    def unparsed_files(self) -> frozenset[str]:
        """Files whose grammar reports ERROR nodes, after ensuring every available file has facts."""
        available = self._available_files(self._code_files)
        self._ensure_facts(available)
        self._reached.update(available)
        return self._unparsed.files

    @property
    def observed_unparsed_files(self) -> frozenset[str]:
        """Files found only partly parsed among those navigation actually reached.

        Unlike ``unparsed_files``, this receipt never starts another repository-wide scan. Pair it
        with ``parser_scans_pending`` before making any claim about the whole scope.
        """
        return self._known_unparsed() & self._reached

    def _known_unparsed(self) -> frozenset[str]:
        """Every file known to be only partly parsed: scanned so far, or recorded so in the table."""
        return self._unparsed.files | self._incomplete_in_table

    @property
    def parsed_files(self) -> frozenset[str]:
        """Files navigation has reached so far, through their facts or their name rows, never one it
        refused to parse; reading it never starts a scan. Covering the scope for the name table
        reaches no file. A file that changed or vanished after it was reached still counts, since its
        facts and rows come from the bytes first read, and it is listed in ``unavailable_files`` too."""
        return frozenset(self._reached - self._refused.keys())

    @property
    def parser_scans_completed(self) -> tuple[str, ...]:
        return ("facts",) if not self.parser_scans_pending else ()

    @property
    def parser_scans_pending(self) -> tuple[str, ...]:
        """The fact scan is pending until navigation has reached every available code file, through its
        facts, its name rows or the parser's refusal of it; covering the scope for the table reaches none."""
        available = set(self._available_files(self._code_files))
        return () if available <= self._reached else ("facts",)

    @property
    def unavailable_files(self) -> dict[str, str]:
        """Files this index has no facts for, each with the reason: an inventory entry that disappeared or
        changed after the index first read it, or a file the parser refused (see ``refused_files``)."""
        return {**self._unavailable, **self._refused}

    @property
    def refused_files(self) -> dict[str, str]:
        """Readable files the fact scan never parsed, each with the reason, such as too large to parse.
        Their text stays searchable; what they define is unknown, never absent."""
        return dict(self._refused)

    @property
    def not_indexed_files(self) -> dict[str, str]:
        """Files and folders under the root that were never part of the scope, each with the reason, such
        as ignored; a folder ends in ``/``."""
        return dict(self._not_indexed)

    @property
    def available_files(self) -> tuple[str, ...]:
        return self._available_files(self.files)

    def _run_scan(self, name: str, scan: Callable[[], object], file_count: int):
        if self.scan_observer is not None:
            self.scan_observer(name, "started", file_count)
        try:
            result = scan()
        except BaseException:
            if self.scan_observer is not None:
                self.scan_observer(name, "failed", file_count)
            raise
        if self.scan_observer is not None:
            self.scan_observer(name, "completed", file_count)
        return result

    def enclosing_symbol(self, file: str, line: int) -> Span | None:
        containing = [span for span in self.functions_in(file) if span.contains(line)]
        return min(containing, key=Span.size, default=None)

    def functions_in(self, file: str) -> tuple[Span, ...]:
        return self._file_structure(file).functions

    def functions_in_files(self, files: Sequence[str]) -> tuple[Span, ...]:
        """Enumerate functions with one batched fact scan for the not-yet-cached source files."""
        files = tuple(dict.fromkeys(files))
        for file in files:
            self._require_in_scope(file)
        available = self._available_files(files)
        self._ensure_facts(tuple(file for file in available if language_of(file)))
        self._reached.update(available)
        return tuple(span for file in available for span in self.functions_in(file))

    def facts_in_files(self, files: Sequence[str]) -> dict[str, FileFacts]:
        """The facts of ``files`` (scope code files), read with one batched fact scan for those not
        yet cached; a file the index cannot read is left out."""
        for file in files:
            self._require_in_scope(file)
        self._ensure_facts(files)
        self._reached.update(files)
        return {file: self._facts[file] for file in files if file in self._facts}

    def definitions_in(self, file: str) -> tuple[Span, ...]:
        """The named functions, classes and declarations in ``file``, from the name table, so a
        warm index answers without loading the file's facts."""
        self._require_in_scope(file)
        if file not in self._cover_scope() or file in self._unavailable:
            return ()
        rows = self._name_table.definitions(self._blob_of(file) or "")
        return tuple(dict.fromkeys(Span(file, row.start, row.end, name) for name, row in rows))

    def symbols_in(self, file: str) -> tuple[Span, ...]:
        """Functions and classes."""
        return self._file_structure(file).symbols

    def top_level_symbols(self, file: str) -> tuple[Span, ...]:
        """The functions and classes no other function or class of the file contains, in file order."""
        return tuple(sorted(_outermost(self.symbols_in(file)), key=lambda span: (span.start, -span.end)))

    def declarations_in(self, file: str) -> tuple[Span, ...]:
        """Module-level constants, assignments, types, interfaces and enums."""
        return self._file_structure(file).declarations

    def find_definition(self, name: str) -> tuple[Span, ...]:
        """Functions, classes, and module-level constants, assignments, types, interfaces and enums."""
        return self._definitions_by_name(name)

    def find_callers(self, name: str) -> tuple[CallSite, ...]:
        """Calls to ``name`` found by name in the syntax tree, each with its binding status. When one
        line holds both ``x.name(...)`` and ``name(...)``, the plain call stands for that line."""
        sites: dict[tuple[str, int], str | None] = {}
        for call in self._calls_with_name(name):
            key = (call.file, call.line)
            if call.name == name and (key not in sites or call.receiver is None):
                sites[key] = call.receiver
        self._load_facts_for_bindings(name, (file for file, _ in sites))
        return tuple(
            CallSite(
                file, line, self.enclosing_symbol(file, line), self.binding_of(file, line, name, receiver)
            )
            for (file, line), receiver in sorted(sites.items())
        )

    def call_site_count(self, name: str) -> int:
        """How many call sites in scope call ``name``; a name called from fewer places is more specific."""
        return len(self._calls_with_name(name))

    def find_callees(self, function: Span) -> tuple[str, ...]:
        """Names called inside ``function``; see ``callee_edges`` for their bindings."""
        return tuple(dict.fromkeys(edge.name for edge in self.callee_edges(function)))

    def callee_edges(self, function: Span) -> tuple[CallEdge, ...]:
        self._require_in_scope(function.file)
        calls = [
            call
            for call in self._facts_in(function.file).calls
            if function.start <= call.line <= function.end
        ]
        edges: dict[str, CallEdge] = {}
        for call in calls:
            if call.name not in edges:
                binding = self.binding_of(call.file, call.line, call.name, call.receiver)
                edges[call.name] = CallEdge(call.name, call.line, binding)
        return tuple(edges.values())

    def find_references(self, name: str) -> tuple[Reference, ...]:
        """Uses of ``name`` that are not calls: arguments, collection entries, assignments,
        decorators, exports, returns, method receivers, types and conditions, each with its role,
        holder and binding. Code reached this way (a callback, a registry entry, a parameter typed
        with a class) has no call edge to follow. A member passed as an argument (``self.handler``)
        is bound like a method call on an unknown receiver, never proven by a same-named function."""
        matches = self._references_with_name(name)
        self._load_facts_for_bindings(name, (match.file for match in matches))
        return self._references(matches)

    def references_in(self, function: Span) -> tuple[Reference, ...]:
        """Names ``function`` passes on without calling them, limited to names defined in scope."""
        self._require_in_scope(function.file)
        inside = (
            match
            for match in self._facts_in(function.file).references
            if function.start <= match.line <= function.end
        )
        return tuple(ref for ref in self._references(inside) if self.find_definition(ref.name))

    def binding_of(
        self, file: str, line: int, name: str, receiver: str | None, role: str | None = None
    ) -> Binding:
        """Computed once per site and cached for the life of the index. ``role`` is a reference's
        role, None for a call, and decides which definitions count: a type names a class or a
        declaration a type can name, an export any definition, and a call or any other use (an
        argument, receiver, condition, decorator...) a function, class or declaration a value can
        name, such as a module constant holding a callable."""
        return self._compute_binding(file, line, name, receiver, role)

    def _references(self, matches: Iterable[ReferenceMatch]) -> tuple[Reference, ...]:
        matches = sorted(set(matches))
        return tuple(
            Reference(
                match.name,
                match.file,
                match.line,
                match.role,
                self.enclosing_symbol(match.file, match.line),
                self.binding_of(match.file, match.line, match.name, match.receiver, match.role),
            )
            for match in matches
        )

    @memoized
    def _compute_binding(
        self, file: str, line: int, name: str, receiver: str | None, role: str | None
    ) -> Binding:
        if self.binding_resolver is not None:
            injected = self.binding_resolver.resolve_call(file, line, name, receiver)
            if injected is not None:
                return injected
        definitions, top_level = self._nameable_definitions(name, role)
        facts = CallFacts(
            file,
            name,
            receiver,
            definitions,
            top_level,
            self._imported_from(file, name),
            self._files_hiding(name),
        )
        return binding_from_facts(facts)

    def _load_facts_for_bindings(self, name: str, use_files: Iterable[str]) -> None:
        """Loads, in one scan, the facts that binding the uses of ``name`` reads: the files the uses
        sit in and the files that define the name. With a warm table and an empty fact cache they
        would otherwise load one file per scan. A name nothing uses loads nothing."""
        uses = tuple(use_files)
        if not uses:
            return
        definition_files = (span.file for span in self.find_definition(name))
        self._ensure_facts(tuple(dict.fromkeys((*uses, *definition_files))))

    @memoized
    def _nameable_definitions(self, name: str, role: str | None) -> tuple[tuple[Span, ...], tuple[Span, ...]]:
        """The definitions of ``name`` a use in ``role`` can name, and those of them at top level.
        Neither depends on where the use sits, so every use of a name shares them."""
        definitions = tuple(span for span in self.find_definition(name) if self._can_name(role, span))
        return definitions, tuple(span for span in definitions if span in self._top_level_spans(span.file))

    def _files_hiding(self, name: str) -> frozenset[str]:
        """Where a definition of ``name`` could sit unseen: a file gone from the disk, or an unparsed or
        refused file whose unread lines mention the name. A definition names what it defines, so
        lines that never mention the name cannot hold one. Every file that mentions it has already
        been scanned to look for its definitions, so the answer does not depend on scan order."""
        unread = (*self._known_unparsed(), *self._refused)
        mentioning = (file for file in unread if name in self._read_unread_names(file))
        return frozenset(mentioning) | self._unavailable.keys()

    @memoized
    def _read_unread_names(self, file: str) -> frozenset[str]:
        """The words on the lines of ``file`` that its ERROR nodes span; the whole file's words while
        its facts are still being recorded, or when the parser refused it."""
        lines = self._lines_of(file)
        known = self._facts.get(file) or (self._entries or {}).get(file)
        stretches = known.unparsed_lines if known is not None else ((1, len(lines)),)
        text = "\n".join(line for start, end in stretches for line in lines[start - 1 : end])
        return frozenset(_WORD.findall(text))

    def _file_structure(self, file: str) -> FileStructure:
        self._require_in_scope(file)
        if not language_of(file):
            return _NO_STRUCTURE
        return self._facts_in(file).structure

    @memoized
    def _definitions_by_name(self, name: str) -> tuple[Span, ...]:
        definitions = {
            Span(file, row.start, row.end, name): None
            for file, row in self._readable_places(name, DEFINITION_KINDS)
        }
        return tuple(definitions)

    def _can_name(self, role: str | None, span: Span) -> bool:
        """Whether a use in ``role`` can name the definition ``span``; ``binding_of`` gives the rule."""
        if role == "export":
            return True
        structure = self._facts_in(span.file).structure
        if span in structure.symbols:
            return role != "type" or span not in structure.functions
        first_line = self.read_slice(Span(span.file, span.start, span.start)).text
        return declares_type(first_line) if role == "type" else declares_value(first_line)

    @memoized
    def _calls_with_name(self, name: str) -> tuple[CallMatch, ...]:
        return tuple(
            CallMatch(file, row.start, name, row.receiver)
            for file, row in self._readable_places(name, (CALL,))
        )

    @memoized
    def _references_with_name(self, name: str) -> tuple[ReferenceMatch, ...]:
        return tuple(
            ReferenceMatch(file, row.start, row.role or "", name, row.receiver)
            for file, row in self._readable_places(name, (REFERENCE,))
        )

    def _readable_places(self, name: str, kinds: Sequence[str]) -> Iterator[tuple[str, NameRow]]:
        """The places of ``name`` of ``kinds`` in files still readable: a file that disappeared or
        changed since the scope was covered answers nothing."""
        for file, row in self._places_named(name):
            if row.kind in kinds and file not in self._unavailable:
                yield file, row

    @memoized
    def _places_named(self, name: str) -> tuple[tuple[str, NameRow], ...]:
        """Every place in scope ``name`` sits, as (file, row), in file order and then in the order of
        the file's facts: definitions first, symbols before declarations."""
        self._cover_scope()
        places = [
            (file, row)
            for row in self._name_table.rows(name)
            for file in self._files_by_blob.get(row.blob, ())
        ]
        self._reached.update(file for file, _ in places)
        return tuple(sorted(places, key=self._place_order))

    def _place_order(self, place: tuple[str, NameRow]) -> tuple[int, int, int]:
        file, row = place
        return self._file_order[file], _KIND_ORDER[row.kind], row.position

    def _cover_scope(self) -> dict[str, FileEntry]:
        """Every available code file's table entry, writing the rows of files the table lacks from
        their cached or freshly parsed facts. Runs once per index; a file that cannot be read or is
        refused by the parse guard has no entry."""
        with self._facts_lock:
            if self._entries is not None:
                return self._entries
            files = self._available_files(self._code_files)
            blobs = self._blobs_of(files)
            held = self._name_table.entries(set(blobs.values()))
            self._ensure_facts([file for file, blob in blobs.items() if blob not in held])
            blobs = self._blobs_of(files)
            held = self._name_table.entries(set(blobs.values()))
            self._entries = {file: held[blob] for file, blob in blobs.items() if blob in held}
            by_blob: dict[str, list[str]] = {}
            for file in self._entries:
                by_blob.setdefault(blobs[file], []).append(file)
            self._files_by_blob = {blob: tuple(files) for blob, files in by_blob.items()}
            self._file_order = {file: position for position, file in enumerate(self._entries)}
            self._incomplete_in_table = frozenset(
                file for file, entry in self._entries.items() if entry.incomplete
            )
            return self._entries

    def _blobs_of(self, files: Iterable[str]) -> dict[str, str]:
        return {file: blob for file in files if (blob := self._blob_of(file)) is not None}

    def _blob_of(self, file: str) -> str | None:
        """The git blob id of ``file``'s content: from the Git listing for a clean tracked file,
        otherwise hashed from the bytes the index read."""
        if file in self._listed_blobs:
            return self._listed_blobs[file]
        if file not in self._blobs:
            content = self._read_bytes(file)
            if content is None:
                return None
            self._blobs[file] = git_blob_id(content)
        return self._blobs[file]

    def _remember_facts(self, file: str, facts: FileFacts) -> None:
        """Keeps ``file``'s facts in memory and queues its rows for the name table. A file the parse
        guard refused never gets here, so it has no rows."""
        self._facts[file] = facts
        blob = self._blob_of(file)
        if blob is not None:
            self._unwritten[blob] = facts

    def _write_names(self) -> None:
        if self._unwritten:
            self._name_table.add(self._unwritten)
            self._unwritten = {}

    def _facts_in(self, file: str) -> FileFacts:
        self._require_in_scope(file)
        self._reached.add(file)
        if (known := self._facts.get(file)) is not None:
            return known
        self._ensure_facts((file,))
        return self._facts.get(file, FileFacts(_NO_STRUCTURE, (), ()))

    def _ensure_facts(self, files: Sequence[str]) -> None:
        with self._facts_lock:
            contents = self._load_cached_facts(files)
            if contents:
                self._parse(contents)
            self._write_names()

    def _parse(self, contents: Mapping[str, bytes]) -> None:
        """Parses the files whose bytes are ``contents``; the caller holds the facts lock."""
        to_scan = tuple(contents)
        scanned = self._run_scan("facts", lambda: self._scan_available_facts(to_scan), len(to_scan))
        for file, facts in scanned.items():
            if self._read_bytes(file) is None:
                continue
            if facts.refusal is not None:
                self._refused[file] = facts.refusal
                continue
            self._remember_facts(file, facts)
            self._fact_cache.save(file, contents[file], facts)

    def _load_cached_facts(self, files: Sequence[str]) -> dict[str, bytes]:
        """Remembers the persisted facts of ``files``; returns the bytes of those still to parse.

        The caller holds the facts lock."""
        to_parse: dict[str, bytes] = {}
        for file in files:
            if language_of(file) is None or file in self._facts or file in self._refused:
                continue
            content = self._read_bytes(file)
            if content is None:
                continue
            cached = self._fact_cache.load(file, content)
            if cached is None:
                to_parse[file] = content
                continue
            self._remember_facts(file, cached)
            if cached.incomplete:
                self._unparsed.add("facts", (file,))
        return to_parse

    def _scan_available_facts(self, files: Sequence[str]) -> dict[str, FileFacts]:
        return self._on_available(
            tuple(files), lambda remaining: scan_facts(remaining, self.root, self._unparsed)
        )

    def _on_available(self, files: tuple[str, ...], run: Callable[[tuple[str, ...]], _Result]) -> _Result:
        """``run(files)``. Only when the tool fails are the files checked; it then runs again over
        those still present, and a failure with every file present is the tool's own."""
        while True:
            try:
                return run(files)
            except tools.ToolFailedError:
                available = self._available_files(files)
                if available == files:
                    raise
                files = available

    @memoized
    def _top_level_spans(self, file: str) -> frozenset[Span]:
        """Symbols and declarations of ``file`` that no class or other function contains. A function
        starting on a declaration's first line is the value it declares, not its container."""
        symbols = self.symbols_in(file)
        top_symbols = _outermost(symbols)
        top_declarations = (
            span
            for span in self.declarations_in(file)
            if not any(other.start < span.start <= other.end for other in symbols)
        )
        return frozenset((*top_symbols, *top_declarations))

    def _imported_from(self, file: str, name: str) -> tuple[ImportFact, ...]:
        specifier = self._read_imported_names(file).get(name)
        if specifier is None:
            return ()
        resolved = resolve_import(
            specifier, file, self._scope, self._script_paths(file), self._read_packages()
        )
        if resolved is None:
            return ()
        found = {resolved.path: resolved}
        pending = [resolved]
        seen = {(resolved.path, resolved.proven)}
        while pending:
            exporter = pending.pop()
            source = "\n".join(self._lines_of(exporter.path))
            for names, target_specifier in reexported_names(source, exporter.path):
                if names is not None and name not in names:
                    continue
                target = resolve_import(
                    target_specifier,
                    exporter.path,
                    self._scope,
                    self._script_paths(exporter.path),
                    self._read_packages(),
                )
                if target is None:
                    continue
                inherited = ImportFact(
                    target.path,
                    exporter.proven and target.proven,
                    target.reason if exporter.proven else exporter.reason,
                )
                identity = (inherited.path, inherited.proven)
                if identity in seen:
                    continue
                seen.add(identity)
                exported = self._facts_in(inherited.path).export_names
                if inherited.path in self._refused or name in exported:
                    prior = found.get(inherited.path)
                    if prior is None or inherited.proven:
                        found[inherited.path] = inherited
                pending.append(inherited)
        return tuple(found.values())

    @memoized
    def _read_imported_names(self, file: str) -> dict[str, str]:
        return imported_names("\n".join(self._lines_of(file)), file)

    def read_slice(self, span: Span, origin: str = "") -> CodeSlice:
        lines = self._lines_of(span.file)
        return CodeSlice(
            span,
            "\n".join(lines[span.start - 1 : span.end]),
            origin,
            self._revision_of(span.file),
            self._file_sha256(span.file),
        )

    def read_window(
        self, file: str, line: int, radius: int = DEFAULT_WINDOW_RADIUS, origin: str = ""
    ) -> CodeSlice:
        span = Span(file, max(1, line - radius), min(len(self._lines_of(file)), line + radius))
        return self.read_slice(span, origin)

    def search_text(
        self, text: str, max_hits: int = MAX_TEXT_HITS, *, whole_word: bool = False
    ) -> tuple[TextHit, ...]:
        """Lines holding ``text``, searched once per text for the life of the index. A hit's text is
        the line up to ``TEXT_HIT_CONTEXT_BYTES`` around its first match; ``whole_word`` keeps only
        matches no word character touches."""
        return self._search_text(text, max_hits, whole_word)

    @memoized
    def _search_text(self, text: str, max_hits: int, whole_word: bool) -> tuple[TextHit, ...]:
        found = self._on_available(
            self._available_files(self.files),
            lambda files: tools.ripgrep_fixed(
                text, files, self.root, max_hits, TEXT_HIT_CONTEXT_BYTES, whole_word=whole_word
            ),
        )
        hits = sorted(hit for hit in found if hit.file in self._scope)
        return tuple(hits[:max_hits])

    def imports(self, file: str) -> tuple[str, ...]:
        source = "\n".join(self._lines_of(file))
        script_paths = self._script_paths(file)
        packages = self._read_packages()
        resolved = (
            resolve_import(specifier, file, self._scope, script_paths, packages)
            for specifier in imported_modules(source, file)
        )
        return tuple(dict.fromkeys(fact.path for fact in resolved if fact))

    def imports_in(self, file: str, text: str) -> tuple[tuple[ImportFact, frozenset[str] | None], ...]:
        """The scope files ``text``, lines of ``file``, imports from, in source order, each with the
        names it takes by name or None for the whole module."""
        script_paths = self._script_paths(file)
        packages = self._read_packages()
        found: dict[str, tuple[ImportFact, frozenset[str] | None]] = {}
        for specifier, names in module_imports(text, file):
            fact = resolve_import(specifier, file, self._scope, script_paths, packages)
            if fact is None or fact.path == file:
                continue
            if fact.path in found:
                prior = found[fact.path][1]
                names = None if prior is None or names is None else prior | names
                fact = found[fact.path][0]
            found[fact.path] = (fact, names)
        return tuple(found.values())

    def dependents(self, file: str) -> tuple[str, ...]:
        self._require_in_scope(file)
        return tuple(path for path in self._code_files if path != file and file in self.imports(path))

    def co_changed_files(self, file: str, limit: int = 5) -> tuple[CoChange, ...]:
        """Scope files most often committed together with ``file``, with their shared-commit counts;
        the history is read once per file for the life of the index."""
        self._require_in_scope(file)
        return self._read_co_changes(file)[:limit]

    @memoized
    def _read_co_changes(self, file: str) -> tuple[CoChange, ...]:
        if not self.commit:
            return ()
        log = tools.git(
            [
                "-c",
                "core.quotePath=false",
                "log",
                *([self.commit] if self.commit else []),
                f"-n{CO_CHANGE_COMMITS}",
                "--full-diff",
                "--name-only",
                f"--format=format:{_COMMIT_MARK}",
                "--",
                file,
            ],
            self.git_root,
        )
        counts = Counter(
            path for commit in _commits(log) if file in commit for path in commit if path != file
        )
        in_scope = [(path, count) for path, count in counts.most_common() if path in self._scope]
        return tuple(sorted(in_scope, key=lambda item: (-item[1], item[0])))

    def _revision_of(self, file: str) -> str:
        if not self.commit:
            return ""
        return f"{self.commit}+worktree" if file in self._changed else self.commit

    def lines(self, file: str) -> tuple[str, ...]:
        return self._lines_of(file)

    def _script_paths(self, file: str) -> ScriptPaths | None:
        """The path aliases of the configs nearest to a script file, read once per directory."""
        return None if file.endswith(".py") else self._read_script_paths(str(PurePosixPath(file).parent))

    @memoized
    def _read_script_paths(self, directory: str) -> ScriptPaths | None:
        return nearest_script_paths(self.root, directory)

    @memoized
    def _read_packages(self) -> Packages:
        """The package.json files of the folders holding scope files, read once, on first use."""
        return Packages(self.root, self.files)

    def _lines_of(self, file: str) -> tuple[str, ...]:
        """The file's lines as the index first read it."""
        self._require_in_scope(file)
        return self._sources.lines(file)

    def _file_sha256(self, file: str) -> str:
        self._require_in_scope(file)
        return self._sources.sha256(file)

    def _read_bytes(self, file: str) -> bytes | None:
        """The file's bytes while they equal its first read, so facts and table rows are only built
        from them."""
        return self._sources.current(file)

    def _standing_first_read(self, file: str, content: bytes | None) -> bytes | None:
        """The bytes that stand as ``file``'s first read, given what the disk holds now (``None`` when
        the file is gone). Once table rows of its listed blob were answered for the file, that blob is
        what the index read, whatever the disk holds now. Otherwise the disk's bytes stand, and bytes
        that differ from the listed blob (a checkout that converts line endings, or an edit since the
        listing) are keyed by their own hash."""
        listed = self._listed_blobs.get(file)
        if listed is None:
            return content
        actual = git_blob_id(content) if content is not None else None
        if actual == listed:
            return content
        if self._entries is not None and file in self._entries:
            return tools.git_blob(self.root, listed)
        if actual is not None:
            del self._listed_blobs[file]
            self._blobs[file] = actual
        return content

    def _available_files(self, files: Sequence[str]) -> tuple[str, ...]:
        """The ``files`` still readable as the index first read them: neither reported unavailable
        before nor gone from disk now."""
        available = []
        for file in files:
            if file in self._unavailable:
                continue
            try:
                if (self.root / file).is_file():
                    available.append(file)
                else:
                    self._unavailable[file] = DISAPPEARED
            except FileNotFoundError:
                self._unavailable[file] = DISAPPEARED
        return tuple(available)

    def _require_in_scope(self, file: str) -> None:
        if file not in self._scope:
            raise ValueError(f"{file} is outside the index scope")


def _held_weakly(method: Callable[..., _Result]) -> Callable[..., _Result]:
    """``method``, called through a weak reference to its object, so whatever holds the returned
    function never keeps that object alive."""
    weak = weakref.WeakMethod(method)
    return lambda *args: weak()(*args)


def _blobs_to_export(repository: Path, commit: str, listed: Sequence[str]) -> dict[str, str]:
    """Object ids of the listed files and of every script config (tsconfig, jsconfig, package.json)
    in ``commit``, keyed by path."""
    tree = _regular_blobs(tools.git(["ls-tree", "-r", "-z", commit], repository))
    return {path: tree[path] for path in [*listed, *_script_configs(tree)]}


def _script_configs(paths: Iterable[str]) -> list[str]:
    return [
        path
        for path in paths
        if (
            (PurePosixPath(path).name.startswith(("tsconfig", "jsconfig")) and path.endswith(".json"))
            or PurePosixPath(path).name == "package.json"
        )
        and "node_modules/" not in path
    ]


def _require_inside(root: Path, files: Iterable[str]) -> None:
    """Every reader (the parser, ripgrep, plain reads) opens scope files by path, so a path that is a
    link or leads out of the root is refused before any of them runs."""
    resolved_root = root.resolve()
    for file in files:
        path = root / file
        if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
            raise UnsafePathError(f"{file} is a symbolic link or lies outside {root}")


def _changed_paths(status: str) -> list[str]:
    """Paths from ``git status --porcelain -z``; a rename or copy record is followed by its old path."""
    records = status.split("\0")
    changed = []
    skip_next = False
    for record in records:
        if skip_next:
            skip_next = False
            continue
        if len(record) > 3:
            changed.append(record[3:])
            skip_next = record[0] in "RC"
    return changed


def _working_git_metadata(root: Path, prefixes: Sequence[str]) -> tuple[str, list[str], dict[str, str]]:
    """The HEAD commit, the changed and untracked paths, and the index blob id of each tracked file;
    all empty outside Git, and no revision before the first commit. A repository git refuses raises
    rather than reading as a plain folder."""
    if not tools.inside_git_worktree(root):
        return "", [], {}
    commit = tools.head_commit(root)
    staged = tools.git(["ls-files", "--stage", "-z", "--", *prefixes], root)
    return commit, _changed_under(root, prefixes), _regular_blobs(staged)


def _changed_under(root: Path, prefixes: Sequence[str]) -> list[str]:
    """The changed and untracked paths under ``root``, relative to it. git status names paths from the
    top of the repository, which is not ``root`` for a folder inside it."""
    status = tools.git(["status", "--porcelain", "-z", "--untracked-files=all", "--", *prefixes], root)
    prefix = tools.git(["rev-parse", "--show-prefix"], root).rstrip("\n")
    return [path.removeprefix(prefix) for path in _changed_paths(status) if path.startswith(prefix)]


def _regular_blobs(listing: str) -> dict[str, str]:
    """The blob id of each path in ``git ls-files --stage -z`` or ``git ls-tree -r -z`` output whose
    mode is a regular file, in listing order. NUL separation keeps names with non-ASCII characters
    exactly as they are on disk."""
    blobs: dict[str, str] = {}
    for line in listing.split("\0"):
        details, _, path = line.partition("\t")
        fields = details.split(" ")
        if fields[0] in _REGULAR_FILE_MODES:
            blobs.setdefault(path, fields[1] if fields[1] != "blob" else fields[2])
    return blobs


def _commits(log: str) -> list[set[str]]:
    blocks = log.split(_COMMIT_MARK)
    return [{line.strip() for line in block.split("\n") if line.strip()} for block in blocks if block.strip()]


def _outermost(symbols: Sequence[Span]) -> list[Span]:
    return [
        span for span in symbols if not any(other != span and other.contains(span.start) for other in symbols)
    ]
