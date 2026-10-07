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

from .. import memory_limit
from . import listing, tools
from .bindings import (
    Binding,
    BindingResolver,
    CallFacts,
    binding_from_facts,
    binding_in_namespace,
    local_binding,
    unparsed_binding,
)
from .fact_cache import FactCache
from .imports import (
    ImportedName,
    ImportFact,
    imported_modules,
    imported_names,
    module_imports,
    reexported_names,
    resolve_import,
)
from .languages import export_words, is_schema_file, language_of
from .memo import memoized
from .name_table import CALL, DEFINITION_KINDS, REFERENCE, FileEntry, NameRow, NameTable, git_blob_id
from .packages import Packages
from .prisma_schema import SchemaBlock, schema_blocks
from .scope import text_files_left_out
from .scope_scan import (
    CallMatch,
    FileFacts,
    FileStructure,
    LocalName,
    ModuleAlias,
    ReferenceMatch,
    Unparsed,
    first_identifier,
    scan_facts,
)
from .source_files import DISAPPEARED, SourceFiles
from .spans import CallEdge, CallSite, CodeSlice, Reference, Span, TextHit
from .text_blocks import TextBlock, text_blocks
from .text_search import TextMatcher
from .tsconfig import ScriptPaths, nearest_script_paths

DEFAULT_WINDOW_RADIUS = 10
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


_NO_STRUCTURE = FileStructure()
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
        self.schema_files = tuple(path for path in self.files if is_schema_file(path))
        self._unavailable: dict[str, str] = {}
        self._refused: dict[str, str] = {}
        self._not_indexed = dict(not_indexed or {})
        self._text_left_out: dict[str, str] = {}
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
        tsconfig files come along (outside the scope), so path aliases resolve. ``close()``, or the end
        of a ``with`` block over the index, removes the directory; it is removed at once if the index
        cannot be built."""
        repository = Path(repository)
        sha = tools.git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repository).strip()
        blobs = _regular_blobs(tools.git(["ls-tree", "-r", "-z", sha, "--", *prefixes], repository))
        listed = list(blobs)
        if max_files is not None and len(listed) > max_files:
            raise ScopeTooWideError(
                f"{len(listed)} files is wider than the limit of {max_files}; narrow the scope"
            )
        snapshot = tempfile.TemporaryDirectory(prefix=f"jev-navigator-{sha[:8]}-")
        try:
            tools.export_blobs(repository, _blobs_to_export(repository, sha, listed), Path(snapshot.name))
            index = cls(
                snapshot.name, listed, max_files=max_files, commit=sha, git_root=repository, blob_ids=blobs
            )
        except BaseException:
            snapshot.cleanup()
            raise
        index._snapshot = snapshot
        return index

    def close(self) -> None:
        """Removes the private directory ``at_commit`` read the revision into; nothing for an index over a
        checkout. The index reads no file after it."""
        if self._snapshot is not None:
            self._snapshot.cleanup()
            self._snapshot = None

    def __enter__(self) -> CodeIndex:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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
        """Files navigation has reached so far, through their facts, their name rows or a schema's
        blocks, never one it refused to parse; reading it never starts a scan. Covering the scope for
        the name table reaches no file. A file that changed or vanished after it was reached still
        counts, since its facts and rows come from the bytes first read, and it is listed in
        ``unavailable_files`` too."""
        return frozenset(self._reached - self._refused.keys())

    @property
    def parser_scans_completed(self) -> tuple[str, ...]:
        return ("facts",) if not self.parser_scans_pending else ()

    @property
    def parser_scans_pending(self) -> tuple[str, ...]:
        """The fact scan is pending until navigation has reached every available code file, through its
        facts, its name rows or the parser's refusal of it, and every Prisma schema, through its blocks;
        covering the scope for the table reaches none."""
        available = set(self._available_files((*self._code_files, *self.schema_files)))
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
        return _innermost(self.functions_in(file), line)

    def known_enclosing_symbol(self, file: str, line: int) -> Span | None:
        """``enclosing_symbol`` from the facts already in memory, or None while there are none: it
        never parses or loads a file, so a caller that only labels a place starts no work."""
        facts = self._facts.get(file)
        return None if facts is None else _innermost(facts.structure.functions, line)

    def functions_in(self, file: str) -> tuple[Span, ...]:
        return self._file_structure(file).functions

    def decorator_starts_in(self, file: str) -> dict[Span, int]:
        """Each function of ``file`` whose decorators sit before it, and the line of its first
        decorator. The function's span still starts at its own first line, below its decorators."""
        structure = self._file_structure(file)
        starts = {(start, end): line for start, end, line in structure.decorated}
        return {
            span: starts[(span.start, span.end)]
            for span in structure.functions
            if (span.start, span.end) in starts
        }

    def stubs_in(self, file: str) -> tuple[Span, ...]:
        """The functions of ``file`` whose body only declares a shape: ``...``, ``pass``, a docstring
        or ``raise NotImplementedError``."""
        structure = self._file_structure(file)
        stubs = set(structure.stubs)
        return tuple(span for span in structure.functions if (span.start, span.end) in stubs)

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

    def read_language(self, file: str) -> str | None:
        """The language ``file``'s facts were read as, so every later scan of the file reads it with
        the same grammar; None when no grammar read it."""
        facts = self.facts_in_files([file]).get(file)
        return facts.language if facts is not None else None

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

    def module_names(self, file: str) -> tuple[str, ...]:
        """The names a reader finds ``file``'s code by, best first: the functions and classes the
        module names (see ``FileStructure.module_symbols``) or assigns to its CommonJS exports and
        each module-level constant a call builds a function for (see ``ConstantFunction``), then
        each function of an object a module-level variable holds, as `api.list`, each function of an
        object a module-level call or `new` is passed, and each function or class of a namespace,
        each group in file order. A file holding none of these has no names."""
        structure = self._file_structure(file)
        symbols = set(structure.symbols)
        own = _named((*structure.module_symbols, *structure.commonjs_exports))
        own += [(function.span, function.constant) for function in structure.constant_functions]
        held = [(member.span, f"{member.owner}.{member.span.name}") for member in structure.object_members]
        held += [(span, span.name) for span in structure.argument_members]
        held += [
            (member.span, member.span.name)
            for member in structure.namespace_members
            if member.span in symbols
        ]
        named_held = [(span, name) for span, name in held if span.is_named]
        return tuple(dict.fromkeys((*_in_file_order(own), *_in_file_order(named_held))))

    def constant_function_names(self, file: str) -> dict[Span, str]:
        """The name each function a module-level constant's call holds goes by (see
        ``ConstantFunction``): the constant's, then the keys around it, `userRouter.list`. Functions on
        one line share a span, which goes by the first one's name. Functions that would share one
        name are told apart by their first line, `pair.<anonymous:4>`."""
        names: dict[Span, str] = {}
        for function in self._file_structure(file).constant_functions:
            names.setdefault(function.span, ".".join((function.constant, *function.keys)))
        counts = Counter(names.values())
        return {
            span: f"{name}.<anonymous:{span.start}>" if counts[name] > 1 else name
            for span, name in names.items()
        }

    @memoized
    def schema_blocks_in(self, file: str) -> tuple[SchemaBlock, ...]:
        """The model, view, enum and type blocks of a Prisma schema in scope, which reach the schema;
        none for any other file."""
        self._require_in_scope(file)
        if not is_schema_file(file):
            return ()
        blocks = schema_blocks(self._lines_of(file))
        self._reached.add(file)
        return blocks

    @memoized
    def text_blocks_in(self, file: str) -> tuple[TextBlock, ...]:
        """The blocks a text search reads ``file`` in, by its format (``text_blocks``)."""
        return text_blocks(file, self._lines_of(file))

    def text_files_left_out(self, files: Sequence[str]) -> dict[str, str]:
        """Which of ``files``, none of them a file JVN parses, a text search leaves out instead of
        reading them as plain text, each with the reason (``scope.text_files_left_out``), decided once
        per file for the life of the index. A file gone from the disk is named in ``unavailable_files``
        instead."""
        undecided = [file for file in self._available_files(files) if file not in self._text_left_out]
        if undecided:
            left_out = text_files_left_out(self.root, undecided)
            self._text_left_out.update({file: left_out.get(file, "") for file in undecided})
        return {file: reason for file in files if (reason := self._text_left_out.get(file))}

    def declarations_in(self, file: str) -> tuple[Span, ...]:
        """Constants, assignments, types, interfaces and enums at module level or directly in a
        TypeScript namespace."""
        return self._file_structure(file).declarations

    def find_definition(self, name: str) -> tuple[Span, ...]:
        """Functions, classes, and the constants, assignments, types, interfaces and enums of
        ``declarations_in``."""
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
        return tuple(
            CallEdge(call.name, call.line, self.binding_of(call.file, call.line, call.name, call.receiver))
            for call in calls
        )

    def find_references(self, name: str) -> tuple[Reference, ...]:
        """Uses of ``name`` that are not calls: arguments, collection entries, assignments,
        decorators, exports, returns, method receivers, types, base classes and conditions, each
        with its role, holder and binding. Code reached this way (a callback, a registry entry, a
        parameter typed with a class, a subclass) has no call edge to follow. A member passed as an
        argument (``self.handler``) is bound like a method call on an unknown receiver, never proven
        by a same-named function."""
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
        if not self._binds_locally(file, line, name, receiver, role):
            enclosing = self._binding_beyond_the_function(file, line, name, receiver, role)
            if enclosing is not None:
                return enclosing
        else:
            own_module = self._binding_through_local_module(file, line, name, receiver, role)
            if own_module is not None:
                return own_module
            if receiver is None:
                return self._binding_through_local_member(file, line, name, role) or local_binding(name)
        return binding_from_facts(self._call_facts(file, name, receiver, role))

    def _binding_beyond_the_function(
        self, file: str, line: int, name: str, receiver: str | None, role: str | None
    ) -> Binding | None:
        """A use no function binds for itself names, in this order, a member of a namespace around
        it, then what its module imports under that name; a method call names the module its
        receiver holds."""
        if receiver is not None:
            return self._binding_through_module(file, name, receiver, role)
        in_namespace = self._binding_in_namespace(file, line, name, role)
        return in_namespace if in_namespace is not None else self._binding_through_import(file, name, role)

    def _call_facts(self, file: str, name: str, receiver: str | None, role: str | None) -> CallFacts:
        definitions, module_scope = self._nameable_definitions(name, role)
        return CallFacts(
            file,
            name,
            receiver,
            definitions,
            module_scope,
            (),
            (),
            self._files_hiding(name),
        )

    def _binding_in_namespace(self, file: str, line: int, name: str, role: str | None) -> Binding | None:
        """A use inside a TypeScript namespace names a member of the innermost namespace around it that
        defines ``name`` before anything outside, imports included; None when no namespace around
        the line does. Lines the index could not parse inside that namespace which mention ``name``
        may hide a closer definition, which leaves the use unknown; lost lines outside it cannot."""
        members = [
            member
            for member in self._file_structure(file).namespace_members
            if member.first <= line <= member.last
            and member.span.name == name
            and self._can_name(role, member.span)
        ]
        if not members:
            return None
        first, last = _innermost_lines((member.first, member.last) for member in members)
        if self._unread_lines_mention(file, name, first, last):
            return unparsed_binding(name, (file,))
        innermost = [member.span for member in members if (member.first, member.last) == (first, last)]
        return binding_in_namespace(name, line, (first, last), innermost)

    def _unread_lines_mention(self, file: str, name: str, first: int, last: int) -> bool:
        """Whether lines ``first`` to ``last`` of ``file`` that its ERROR nodes span mention ``name``."""
        lines = self._lines_of(file)
        return any(
            name in _WORD.findall(text)
            for start, end in self._facts_in(file).unparsed_lines
            for text in lines[max(start, first) - 1 : min(end, last)]
        )

    def _binds_locally(self, file: str, line: int, name: str, receiver: str | None, role: str | None) -> bool:
        """Whether a function holding ``line`` binds the name the use looks up first for its own body:
        ``stop`` in ``stop()``, ``db`` in ``db.query()``. That name then holds a local value, never a
        definition, import or module alias of its module (only the module a function's own require
        binds, see ``_binding_through_local_module``); a method on it is still looked up by its own
        name. A type is looked up among types, which no local value replaces, and an export names
        module-level code. A function counts from its first line, so a call on that line before the
        function starts counts as inside it."""
        if role in ("type", "export"):
            return False
        looked_up = name if receiver is None else first_identifier(receiver)
        return any(
            local.first <= line <= local.last for local in self._read_local_bindings(file).get(looked_up, ())
        )

    def _binding_through_local_module(
        self, file: str, line: int, name: str, receiver: str | None, role: str | None
    ) -> Binding | None:
        """The binding of a use of a name that the innermost function around ``line`` binds exactly
        once, and to a whole module of the scope (``const db = require('./db')``, see ``LocalName``):
        ``db()`` calls the module's default export, ``db.query()`` the ``query`` it exports. None
        when that function binds the name more than once or to anything else, or when the receiver is
        longer than the name."""
        module = self._local_module(file, line, name if receiver is None else receiver)
        if module is None:
            return None
        return self._binding_through_exporters(file, module, "default" if receiver is None else name, role)

    def _binding_through_local_member(
        self, file: str, line: int, name: str, role: str | None
    ) -> Binding | None:
        """The binding of ``name()`` when the innermost function around ``line`` binds ``name`` once, to
        a member of a plain name (``const { insert } = client``, ``const utc = client.toUtc``, see
        ``LocalName.member``): the call is ``client.insert()``. None when it binds it otherwise."""
        held = self._held(file, line, name)
        if held is None or not held.member:
            return None
        holder, member = held.member.split(".", 1)
        return self.binding_of(file, line, member, holder, role)

    def _local_module(self, file: str, line: int, name: str) -> str | None:
        """The module ``name`` holds at ``line`` (see ``_held``), when it holds a whole module."""
        held = self._held(file, line, name)
        return None if held is None else held.module or None

    def _held(self, file: str, line: int, name: str) -> LocalName | None:
        """The binding of ``name`` at ``line`` when the innermost function there that binds it binds it
        once, to a module or a member, and ``line`` lies from that binding to the end of its block."""
        holding = [
            local
            for local in self._read_local_bindings(file).get(name, ())
            if local.first <= line <= local.last
        ]
        if not holding:
            return None
        innermost = _innermost_lines((local.first, local.last) for local in holding)
        own = [local for local in holding if (local.first, local.last) == innermost]
        if len(own) != 1 or not own[0].line <= line <= own[0].block_end:
            return None
        return own[0]

    @memoized
    def _read_local_bindings(self, file: str) -> dict[str, tuple[LocalName, ...]]:
        bindings: dict[str, list[LocalName]] = {}
        for local in self._file_structure(file).local_names:
            bindings.setdefault(local.name, []).append(local)
        return {name: tuple(found) for name, found in bindings.items()}

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
        """The definitions of ``name`` a use in ``role`` can name, and those of them their module's
        scope names. Neither depends on where the use sits, so every use of a name shares them."""
        definitions = tuple(span for span in self.find_definition(name) if self._can_name(role, span))
        return definitions, tuple(span for span in definitions if span in self._module_scope_spans(span.file))

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
        return span in (structure.type_declarations if role == "type" else structure.value_declarations)

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
        """The file counts as reached once its facts are known: a parse that fails, for any reason,
        reaches nothing."""
        self._require_in_scope(file)
        facts = self._facts.get(file)
        if facts is None:
            self._ensure_facts((file,))
            facts = self._facts.get(file, FileFacts(_NO_STRUCTURE, (), ()))
        self._reached.add(file)
        return facts

    def _ensure_facts(self, files: Sequence[str]) -> None:
        with self._facts_lock:
            contents = self._load_cached_facts(files)
            if contents:
                self._parse(contents)
            self._write_names()

    def _parse(self, contents: Mapping[str, bytes]) -> None:
        """Parses the files whose bytes are ``contents``; the caller holds the facts lock."""
        scanned = self._run_scan("facts", lambda: self._scan_available_facts(contents), len(contents))
        for file, facts in scanned.items():
            if self._read_bytes(file) is None:
                continue
            if facts.refusal is not None:
                self._refused[file] = facts.refusal
                continue
            self._remember_facts(file, facts)
            self._fact_cache.save(file, contents[file], facts)

    def _load_cached_facts(self, files: Sequence[str]) -> dict[str, bytes]:
        """Remembers persisted facts; returns the bytes of files still to parse. The host owns
        loaded facts' memory; the shared JVN slot is reserved before loading them.

        The caller holds the facts lock."""
        to_parse: dict[str, bytes] = {}
        for file in files:
            if language_of(file) is None or file in self._facts or file in self._refused:
                continue
            memory_limit.check()
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

    def _scan_available_facts(self, contents: Mapping[str, bytes]) -> dict[str, FileFacts]:
        return self._on_available(
            tuple(contents),
            lambda remaining: scan_facts(
                {file: contents[file] for file in remaining}, self.root, self._unparsed
            ),
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
    def _module_scope_spans(self, file: str) -> frozenset[Span]:
        """Symbols and declarations that ``file``'s module scope names; the scan decides it for
        symbols from the syntax tree."""
        return frozenset((*self._file_structure(file).module_symbols, *self._module_declarations(file)))

    def _module_declarations(self, file: str) -> tuple[Span, ...]:
        """The declarations no class, function or namespace contains. A function starting on a
        declaration's first line is the value it declares, not its container."""
        structure = self._file_structure(file)
        members = {member.span for member in structure.namespace_members}
        return tuple(
            span
            for span in structure.declarations
            if span not in members
            and not any(other.start < span.start <= other.end for other in structure.symbols)
        )

    def _binding_through_module(
        self, file: str, name: str, receiver: str, role: str | None
    ) -> Binding | None:
        """The binding of ``receiver.name`` when ``receiver`` holds a whole module of the scope
        (``import * as receiver``, ``const receiver = require(...)``, ``from pkg import receiver``),
        decided like an import of ``name`` from that module (see ``_binding_through_exporters``).
        None when ``receiver`` holds no module of the scope, or when its from-import may take the
        package's own name instead (see ``_package_binds_otherwise``)."""
        alias = self._module_alias(file, receiver)
        if alias is None:
            return self._binding_through_default_object(file, name, receiver, role)
        if alias.from_import and self._package_binds_otherwise(file, alias.specifier):
            return None
        return self._binding_through_exporters(file, alias.specifier, name, role)

    def _binding_through_default_object(
        self, file: str, name: str, receiver: str, role: str | None
    ) -> Binding | None:
        """The binding of ``receiver.name`` when ``receiver`` is a default import of a module whose
        default export object holds ``name`` (``export default { insert, utc: toUtc }``, see
        ``FileFacts.default_members``): that module's own definition the member holds. None when
        ``receiver`` is no default import of a module of the scope, or its default object has no such
        member."""
        imported = self._read_imported_names(file).get(receiver)
        if imported is None or imported.exported is not None:
            return None
        module = resolve_import(
            imported.specifier, file, self._scope, self._script_paths(file), self._read_packages()
        )
        if module is None:
            return None
        own = dict(self._facts_in(module.path).default_members).get(name)
        if own is None:
            return None
        definitions = tuple(
            span
            for span in self._module_scope_spans(module.path)
            if span.name == own and self._can_name(role, span)
        )
        hiding = {module.path} if self._hides(module.path, own) else set()
        return binding_from_facts(
            CallFacts(file, name, None, definitions, (), definitions, (module,), hiding)
        )

    def _package_binds_otherwise(self, file: str, specifier: str) -> bool:
        """Whether ``from package import module`` may give something other than the module
        ``specifier`` names: Python takes the package's own name first, so the package's
        ``__init__`` must not define that name, import anything else under it, star-import, or have
        lines the index could not read that mention it."""
        module = self._module_path(file, specifier)
        if module is None:
            return False
        path = PurePosixPath(module)
        holder, member = (
            (path.parent.parent, path.parent.name) if path.name == "__init__.py" else (path.parent, path.stem)
        )
        init = str(holder / "__init__.py")
        if init not in self._scope:
            return False
        bound = [alias for alias in self._facts_in(init).module_aliases if alias.name in (member, "*")]
        imports_otherwise = any(
            alias.name == "*" or self._module_path(init, alias.specifier) != module for alias in bound
        )
        return imports_otherwise or self._binds_otherwise(init, member) or init in self._files_hiding(member)

    def _module_path(self, file: str, specifier: str) -> str | None:
        resolved = resolve_import(
            specifier, file, self._scope, self._script_paths(file), self._read_packages()
        )
        return None if resolved is None else resolved.path

    def _binding_through_import(self, file: str, name: str, role: str | None) -> Binding | None:
        """The binding of ``name()`` when ``file`` imports ``name``: by that name or under another
        (``import { stop as name }``, ``const { stop: name } = require(...)``, ``from m import stop as
        name``), or as a default import, which takes the module's default export. Decided from the module the
        import names (see ``_binding_through_exporters``). None when nothing imports ``name``, when
        ``file`` defines ``name`` itself, or when the import names no module of the scope."""
        imported = self._read_imported_names(file).get(name)
        if imported is None or self._defines(file, name, role):
            return None
        return self._binding_through_exporters(file, imported.specifier, imported.exported or "default", role)

    def _binding_through_exporters(
        self, file: str, specifier: str, name: str, role: str | None
    ) -> Binding | None:
        """The binding of ``name`` imported from the module ``specifier`` names, read from that
        module's own facts and those of the modules it re-exports ``name`` from, never from a search
        of the scope: one definition proves the target, several leave a candidate, an exporter that
        may hide it (see ``_hides``) leaves it unknown, and no definition leaves a candidate naming
        the modules. None when ``specifier`` names no module of the scope."""
        exporters = self._read_exporters(file, specifier, name)
        if not exporters:
            return None
        definitions = tuple(
            span
            for exporter in exporters
            for span in self._read_importable_definitions(exporter.path, name, role)
        )
        hiding = {exporter.path for exporter in exporters if self._hides(exporter.path, name)}
        return binding_from_facts(
            CallFacts(file, name, None, definitions, (), definitions, exporters, hiding)
        )

    def _hides(self, exporter: str, name: str) -> bool:
        """Whether ``exporter`` may export ``name`` where the index cannot see it: the parser refused
        the file, so nothing it exports was read, or its unparsed lines say a word the export of
        ``name`` is written with (see ``export_words``) or the name of a definition it exports as
        ``name`` (see ``_own_names``), or it vanished."""
        if self._refused_parse(exporter):
            return True
        looked_up = {*export_words(name), *self._own_names(exporter, name)}
        return any(exporter in self._files_hiding(each) for each in looked_up)

    def _refused_parse(self, file: str) -> bool:
        """Whether the parser refused ``file`` (see ``refused_files``), asking for its facts first, so
        the answer never depends on what was read before."""
        self._facts_in(file)
        return file in self._refused

    def _defines(self, file: str, name: str, role: str | None) -> bool:
        """Whether ``file``'s module scope defines ``name`` as a definition ``role`` can name."""
        return any(
            span.name == name and self._can_name(role, span) for span in self._module_scope_spans(file)
        )

    @memoized
    def _read_importable_definitions(self, file: str, name: str, role: str | None) -> tuple[Span, ...]:
        """The definitions another module imports from ``file`` as ``name``, symbols before
        declarations as ``find_definition`` orders them; every importer shares them. A Python module
        exports its whole module scope under its own names. A script module exports the module-scope
        definition an export statement or list names, its default export or a CommonJS export of that
        name, under its own name or the one an export list gives it (`export { inner as outer }`), and
        the functions and classes it assigns to CommonJS exports."""
        structure = self._file_structure(file)
        own_names = self._own_names(file, name)
        exported = {span for span in self._module_scope_spans(file) if span.name in own_names}
        exported |= {span for span in structure.commonjs_exports if span.name == name}
        return tuple(
            span
            for span in (*structure.symbols, *structure.declarations)
            if span in exported and self._can_name(role, span)
        )

    def _exports(self, file: str, name: str) -> bool:
        """Whether ``file`` exports a definition of its own as ``name``: one in its module scope for a
        Python module, one its export statements name for a script module."""
        if language_of(file) == "python":
            return any(span.name == name for span in self._module_scope_spans(file))
        return name in self._read_export_names(file)

    def _own_names(self, file: str, name: str) -> frozenset[str]:
        """The names of the definitions ``file`` exports as ``name``: the same name in a Python
        module, and in a script module the ones ``_read_export_names`` gives."""
        return (
            frozenset((name,))
            if language_of(file) == "python"
            else self._read_export_names(file).get(name, frozenset())
        )

    @memoized
    def _read_export_names(self, file: str) -> dict[str, frozenset[str]]:
        """Each name script module ``file`` exports from its own definitions, ``default`` for its
        default export, with the names of the definitions it may export under it: a module that
        assigns `module.exports` twice has two."""
        facts = self._facts_in(file)
        renamed: dict[str, set[str]] = {}
        for exported, own in facts.renamed_exports:
            renamed.setdefault(exported, set()).add(own)
        own_names = {name: frozenset((name,)) for name in (*facts.export_names, *facts.exported_values)}
        return own_names | {exported: frozenset(owns) for exported, owns in renamed.items()}

    @memoized
    def _read_exporters(self, file: str, specifier: str, name: str) -> tuple[ImportFact, ...]:
        """The module ``file``'s import of ``specifier`` resolves to, then each module it re-exports
        ``name`` from that exports it or may hide it (see ``_hides``), with the evidence for each."""
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
            for names, target_specifier in self._read_reexports(exporter.path):
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
                if self._hides(inherited.path, name) or self._exports(inherited.path, name):
                    prior = found.get(inherited.path)
                    if prior is None or inherited.proven:
                        found[inherited.path] = inherited
                pending.append(inherited)
        return tuple(found.values())

    @memoized
    def _read_reexports(self, file: str) -> tuple[tuple[frozenset[str] | None, str], ...]:
        return reexported_names("\n".join(self._lines_of(file)), file)

    @memoized
    def _read_imported_names(self, file: str) -> dict[str, ImportedName]:
        return imported_names("\n".join(self._lines_of(file)), file)

    def _module_alias(self, file: str, name: str) -> ModuleAlias | None:
        """The alias that binds ``name`` when module-level code binds it to one whole module and in
        no other way (see ``ModuleAlias`` and ``_binds_otherwise``); None when it binds it to none,
        to two, or otherwise too."""
        aliases = {alias for alias in self._facts_in(file).module_aliases if alias.name == name}
        if len(aliases) != 1 or self._binds_otherwise(file, first_identifier(name)):
            return None
        return aliases.pop()

    def _binds_otherwise(self, file: str, name: str) -> bool:
        """Whether code of ``file`` binds its module-level ``name`` other than by an import: a function
        or class the module names, or a binding such as `name = make()` (see ``module_bindings``)."""
        return name in self._facts_in(file).module_bindings or any(
            span.name == name for span in self._file_structure(file).module_symbols
        )

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
        self, text: str, max_hits: int | None = None, *, whole_word: bool = False
    ) -> tuple[TextHit, ...]:
        """Every line holding ``text``, in file and line order, or only the first ``max_hits`` when a
        caller bounds them, searched once per text, bound and word rule for the life of the index. A
        hit's text is the line up to ``TEXT_HIT_CONTEXT_BYTES`` around its first match; ``whole_word``
        keeps only matches no word character touches."""
        return self._search_text(text, max_hits, whole_word)

    def search_texts(self, texts: Iterable[str]) -> dict[str, tuple[TextHit, ...]]:
        """Exact hits for many terms in one repository scan, cached like search_text.

        Ripgrep returns only line identities. The literal matcher retains every overlapping term
        on those lines and gives each its own bounded context, so batching never drops provenance.
        """
        texts = tuple(dict.fromkeys(texts))
        cache = self.__dict__.setdefault("_memoized__search_text", {})
        missing = [text for text in texts if (text, None, False) not in cache]
        ordinary = [text for text in missing if text and "\n" not in text and "\r" not in text]
        for text in missing:
            if text not in ordinary:
                self.search_text(text)
        if ordinary:
            matches = self._on_available(
                self._available_files(self.files),
                lambda files: tools.ripgrep_term_lines(ordinary, files, self.root),
            )
            matcher = TextMatcher(ordinary)
            hits: dict[str, list[TextHit]] = {text: [] for text in ordinary}
            for line in sorted(matches):
                source = self.lines(line.file)[line.line - 1]
                seen = set()
                for term, position in matcher.matches(source):
                    if term not in seen:
                        seen.add(term)
                        context = source[
                            max(0, position - TEXT_HIT_CONTEXT_BYTES) : position
                            + len(term)
                            + TEXT_HIT_CONTEXT_BYTES
                        ]
                        hits[term].append(TextHit(line.file, line.line, context))
            cache.update(((text, None, False), tuple(hits[text])) for text in ordinary)
        return {text: cache[(text, None, False)] for text in texts}

    @memoized
    def _search_text(self, text: str, max_hits: int | None, whole_word: bool) -> tuple[TextHit, ...]:
        found = self._on_available(
            self._available_files(self.files),
            lambda files: tools.ripgrep_fixed(
                text, files, self.root, max_hits, TEXT_HIT_CONTEXT_BYTES, whole_word=whole_word
            ),
        )
        hits = sorted(hit for hit in found if hit.file in self._scope)
        return tuple(hits if max_hits is None else hits[:max_hits])

    @memoized
    def imports(self, file: str) -> tuple[str, ...]:
        """The scope files ``file`` imports, read once per index from the file as the index first read it."""
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
        """The scope files that import ``file``, in scope order."""
        self._require_in_scope(file)
        return self._read_dependents().get(file, ())

    @memoized
    def _read_dependents(self) -> dict[str, tuple[str, ...]]:
        """Each scope file's importers, from one read of every code file's imports per index."""
        importers: dict[str, list[str]] = {}
        for path in self._code_files:
            for imported in self.imports(path):
                if imported != path:
                    importers.setdefault(imported, []).append(path)
        return {imported: tuple(paths) for imported, paths in importers.items()}

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


def _named(spans: Iterable[Span]) -> list[tuple[Span, str]]:
    return [(span, span.name) for span in spans if span.is_named]


def _in_file_order(spans_and_names: Iterable[tuple[Span, str]]) -> tuple[str, ...]:
    """The names by their spans' order, outer first where spans start together, each name once."""
    ordered = sorted(spans_and_names, key=lambda entry: (entry[0].start, -entry[0].end))
    return tuple(dict.fromkeys(name for _, name in ordered))


def _commits(log: str) -> list[set[str]]:
    blocks = log.split(_COMMIT_MARK)
    return [{line.strip() for line in block.split("\n") if line.strip()} for block in blocks if block.strip()]


def _innermost_lines(lines: Iterable[tuple[int, int]]) -> tuple[int, int]:
    """Of the nested scopes around one line, as first and last lines, the innermost: the one that
    starts last, and of those starting together the one that ends first."""
    return max(lines, key=lambda scope: (scope[0], -scope[1]))


def _innermost(functions: Iterable[Span], line: int) -> Span | None:
    return min((span for span in functions if span.contains(line)), key=Span.size, default=None)
