"""Mechanical code lookups over a narrowed set of files: no model, only ast-grep, ripgrep and git.

Every lookup stays inside the scope the index was built with. ``max_files`` is an explicit caller
policy only; the index has no default refusal and never drops files from a requested scope.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from functools import cache, lru_cache
from pathlib import Path, PurePosixPath
from typing import TypeVar

from . import tools
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
    split_lines,
)
from .packages import Packages
from .scope_scan import FileFacts, FileStructure, ReferenceMatch, Unparsed, scan_facts
from .spans import CallEdge, CallSite, CodeSlice, Reference, Span, TextHit
from .tsconfig import ScriptPaths, nearest_script_paths

DEFAULT_WINDOW_RADIUS = 10
MAX_TEXT_HITS = 20
CO_CHANGE_COMMITS = 200
LINE_CACHE_FILES = 512
_COMMIT_MARK = "@@commit@@"
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})
_WORD = re.compile(r"[\w$]+")
ScanObserver = Callable[[str, str, int], None]
_Result = TypeVar("_Result")
CoChange = tuple[str, int]


_NO_STRUCTURE = FileStructure((), (), ())


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
        self._lines_of = lru_cache(maxsize=LINE_CACHE_FILES)(self._read_lines)
        self._sha256: dict[str, str] = {}
        self._script_paths_in = cache(self._read_script_paths)
        self._packages = cache(self._read_packages)
        self._unparsed = Unparsed()
        self._unavailable: dict[str, str] = {}
        self._facts: dict[str, FileFacts] = {}
        self._fact_files_by_name: dict[str, set[str]] = {}
        self._facts_lock = threading.RLock()
        self._fact_cache = FactCache(fact_cache_dir)
        self._files_for_name = cache(self._candidate_files)
        self._discovered: dict[str, frozenset[str]] = {}
        self._text_hits = cache(self._search_text)
        self._co_changes = cache(self._read_co_changes)
        self._calls_named = cache(self._calls_with_name)
        self._references_named = cache(self._references_with_name)
        self._definitions = cache(self._definitions_by_name)
        self._top_level_in = cache(self._top_level_spans)
        self._names_imported = cache(self._read_imported_names)
        self._binding = cache(self._compute_binding)
        self._unread_names = cache(self._read_unread_names)

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
        """The tracked regular files under ``prefixes`` (every one when none are given). Symbolic links
        and submodules are left out: a link can point outside the scope, or at a directory."""
        root = Path(root)
        listed = _regular_files(tools.git(["ls-files", "--stage", "-z", "--", *prefixes], root))
        commit = tools.git(["rev-parse", "HEAD"], root).strip()
        changed = _changed_paths(tools.git(["status", "--porcelain", "-z", "--", *prefixes], root))
        return cls(
            root,
            listed,
            max_files=max_files,
            commit=commit,
            changed_files=changed,
            binding_resolver=binding_resolver,
            scan_observer=scan_observer,
            fact_cache_dir=fact_cache_dir,
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
        """Index current files using ripgrep's ignore policy, with Git metadata when available.

        Modified and untracked files are included. In a Git worktree, slices carry the current HEAD
        plus ``+worktree`` when their file differs; outside Git they carry no commit. Every slice
        also carries its current file SHA-256, so either case identifies the inspected bytes.
        """
        root = Path(root)
        excluded = tuple(path.resolve() for path in exclude_paths)
        files = tuple(
            file
            for file in tools.listed_files(root, prefixes)
            if not any((root / file).resolve().is_relative_to(path) for path in excluded)
        )
        commit, changed = _working_git_metadata(root, prefixes)
        return cls(
            root,
            files,
            max_files=max_files,
            commit=commit,
            changed_files=changed,
            binding_resolver=binding_resolver,
            scan_observer=scan_observer,
            fact_cache_dir=fact_cache_dir,
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
        listed = _regular_files(tools.git(["ls-tree", "-r", "-z", sha, "--", *prefixes], repository))
        if max_files is not None and len(listed) > max_files:
            raise ScopeTooWideError(
                f"{len(listed)} files is wider than the limit of {max_files}; narrow the scope"
            )
        snapshot = tempfile.TemporaryDirectory(prefix=f"jev-navigator-{sha[:8]}-")
        tools.export_blobs(repository, _blobs_to_export(repository, sha, listed), Path(snapshot.name))
        index = cls(snapshot.name, listed, max_files=max_files, commit=sha, git_root=repository)
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
        self._ensure_facts(self._available_files(self._code_files))
        return self._unparsed.files

    @property
    def observed_unparsed_files(self) -> frozenset[str]:
        """Files found unparsed by scans that navigation actually needed.

        Unlike ``unparsed_files``, this receipt never starts another repository-wide scan. Pair it
        with ``parser_scans_pending`` before making any claim about the whole scope.
        """
        return self._unparsed.files

    @property
    def parsed_files(self) -> frozenset[str]:
        """Files navigation has parsed so far; reading it never starts a scan."""
        return frozenset(self._facts)

    @property
    def parser_scans_completed(self) -> tuple[str, ...]:
        return ("facts",) if not self.parser_scans_pending else ()

    @property
    def parser_scans_pending(self) -> tuple[str, ...]:
        available = set(self._available_files(self._code_files))
        return () if available <= self._facts.keys() else ("facts",)

    @property
    def unavailable_files(self) -> dict[str, str]:
        """Files this index cannot read facts from, each with the reason: an inventory entry that disappeared
        after the index was created, or a file too large to parse safely."""
        return dict(self._unavailable)

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
        return tuple(span for file in available for span in self.functions_in(file))

    def symbols_in(self, file: str) -> tuple[Span, ...]:
        """Functions and classes."""
        return self._file_structure(file).symbols

    def declarations_in(self, file: str) -> tuple[Span, ...]:
        """Module-level constants, assignments, types, interfaces and enums."""
        return self._file_structure(file).declarations

    def find_definition(self, name: str) -> tuple[Span, ...]:
        """Functions, classes, and module-level constants, assignments, types, interfaces and enums."""
        return self._definitions(name)

    def find_callers(self, name: str) -> tuple[CallSite, ...]:
        """Calls to ``name`` found by name in the syntax tree, each with its binding status. When one
        line holds both ``x.name(...)`` and ``name(...)``, the plain call stands for that line."""
        sites: dict[tuple[str, int], str | None] = {}
        for call in self._calls_named(name):
            key = (call.file, call.line)
            if call.name == name and (key not in sites or call.receiver is None):
                sites[key] = call.receiver
        return tuple(
            CallSite(
                file, line, self.enclosing_symbol(file, line), self.binding_of(file, line, name, receiver)
            )
            for (file, line), receiver in sorted(sites.items())
        )

    def call_site_count(self, name: str) -> int:
        """How many call sites in scope call ``name``; a name called from fewer places is more specific."""
        return len(self._calls_named(name))

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
        self.prefetch_names(call.name for call in calls)
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
        return self._references(self._references_named(name))

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
        return self._binding(file, line, name, receiver, role)

    def _references(self, matches: Iterable[ReferenceMatch]) -> tuple[Reference, ...]:
        matches = sorted(set(matches))
        self.prefetch_names(match.name for match in matches)
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

    def _compute_binding(
        self, file: str, line: int, name: str, receiver: str | None, role: str | None
    ) -> Binding:
        if self.binding_resolver is not None:
            injected = self.binding_resolver.resolve_call(file, line, name, receiver)
            if injected is not None:
                return injected
        definitions = tuple(span for span in self.find_definition(name) if self._can_name(role, span))
        facts = CallFacts(
            file,
            name,
            receiver,
            definitions,
            tuple(span for span in definitions if span in self._top_level_in(span.file)),
            self._imported_from(file, name),
            self._files_hiding(name),
        )
        return binding_from_facts(facts)

    def _files_hiding(self, name: str) -> frozenset[str]:
        """Where a definition of ``name`` could sit unseen: a file gone from the disk, or an unparsed
        file whose lines under an ERROR node mention the name. A definition names what it defines, so
        lines that never mention the name cannot hold one. Every file that mentions it has already
        been scanned to look for its definitions, so the answer does not depend on scan order."""
        unparsed = (file for file in self.observed_unparsed_files if name in self._unread_names(file))
        return frozenset(unparsed) | self.unavailable_files.keys()

    def _read_unread_names(self, file: str) -> frozenset[str]:
        """The words on the lines of ``file`` that its ERROR nodes span; the whole file's words while
        its facts are still being recorded."""
        lines = self._lines_of(file)
        facts = self._facts.get(file)
        stretches = facts.unparsed_lines if facts is not None else ((1, len(lines)),)
        text = "\n".join(line for start, end in stretches for line in lines[start - 1 : end])
        return frozenset(_WORD.findall(text))

    def _file_structure(self, file: str) -> FileStructure:
        self._require_in_scope(file)
        if not language_of(file):
            return _NO_STRUCTURE
        return self._facts_in(file).structure

    def _definitions_by_name(self, name: str) -> tuple[Span, ...]:
        files = self._files_for_name(name)
        self._ensure_facts(files)
        definitions = {
            span: None
            for file in files
            for span in (
                *self._facts_in(file).structure.symbols,
                *self._facts_in(file).structure.declarations,
            )
            if span.name == name
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

    def _calls_with_name(self, name: str):
        files = self._files_for_name(name)
        self._ensure_facts(files)
        return tuple(call for file in files for call in self._facts_in(file).calls if call.name == name)

    def _references_with_name(self, name: str):
        files = self._files_for_name(name)
        self._ensure_facts(files)
        return tuple(
            reference
            for file in files
            for reference in self._facts_in(file).references
            if reference.name == name
        )

    def prefetch_names(self, names: Iterable[str]) -> None:
        """Finds the unparsed files that may mention each of ``names`` with one ripgrep, so the
        lookups of these names that follow start no search of their own."""
        self._discover(names)

    def _candidate_files(self, name: str) -> tuple[str, ...]:
        # Parsed facts already answer name membership. Only unparsed inventory needs text discovery.
        self._discover((name,))
        with self._facts_lock:
            candidates = self._fact_files_by_name.get(name, set()) | self._discovered[name]
        return tuple(file for file in self._code_files if file in candidates)

    def _discover(self, names: Iterable[str]) -> None:
        with self._facts_lock:
            new = [name for name in dict.fromkeys(names) if name not in self._discovered]
            remaining = tuple(
                file for file in self._code_files if file not in self._facts and file not in self._unavailable
            )
        if not new:
            return
        mentioning = self._on_available(remaining, lambda files: tools.ripgrep_files(new, files, self.root))
        found = self._files_by_name(new, mentioning)
        with self._facts_lock:
            for name in new:
                self._discovered.setdefault(name, found[name])

    def _files_by_name(self, names: Sequence[str], files: Sequence[str]) -> dict[str, frozenset[str]]:
        """Which of ``files`` (each known to hold one of ``names``) holds each name, reading each file
        once."""
        if len(names) == 1:
            return {names[0]: frozenset(files)}
        holding: dict[str, set[str]] = {name: set() for name in names}
        patterns = [(name, name.encode()) for name in names]
        for file in files:
            content = self._read_bytes(file) or b""
            for name, pattern in patterns:
                if pattern in content:
                    holding[name].add(file)
        return {name: frozenset(found) for name, found in holding.items()}

    def _remember_facts(self, file: str, facts: FileFacts) -> None:
        self._facts[file] = facts
        names = {
            item.name
            for item in (
                *facts.structure.symbols,
                *facts.structure.declarations,
                *facts.calls,
                *facts.references,
            )
        }
        for name in names:
            self._fact_files_by_name.setdefault(name, set()).add(file)

    def _facts_in(self, file: str) -> FileFacts:
        self._require_in_scope(file)
        self._ensure_facts((file,))
        return self._facts.get(file, FileFacts(_NO_STRUCTURE, (), ()))

    def _ensure_facts(self, files: Sequence[str]) -> None:
        with self._facts_lock:
            contents = self._load_cached_facts(files)
            if not contents:
                return
            to_scan = tuple(contents)
            scanned = self._run_scan(
                "facts",
                lambda: self._scan_available_facts(to_scan),
                len(to_scan),
            )
            for file, facts in scanned.items():
                if self._read_bytes(file) is None:
                    continue
                self._remember_facts(file, facts)
                if facts.refusal is None:
                    self._fact_cache.save(file, contents[file], facts)
                else:
                    self._unavailable[file] = facts.refusal

    def _load_cached_facts(self, files: Sequence[str]) -> dict[str, bytes]:
        """Remembers the persisted facts of ``files``; returns the bytes of those still to parse.

        The caller holds the facts lock."""
        to_parse: dict[str, bytes] = {}
        for file in files:
            if language_of(file) is None or file in self._facts or file in self._unavailable:
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

    def _top_level_spans(self, file: str) -> frozenset[Span]:
        """Symbols and declarations of ``file`` that no class or other function contains. A function
        starting on a declaration's first line is the value it declares, not its container."""
        symbols = self.symbols_in(file)
        top_symbols = (
            span
            for span in symbols
            if not any(other != span and other.contains(span.start) for other in symbols)
        )
        top_declarations = (
            span
            for span in self.declarations_in(file)
            if not any(other.start < span.start <= other.end for other in symbols)
        )
        return frozenset((*top_symbols, *top_declarations))

    def _imported_from(self, file: str, name: str) -> tuple[ImportFact, ...]:
        specifier = self._names_imported(file).get(name)
        if specifier is None:
            return ()
        resolved = resolve_import(specifier, file, self._scope, self._script_paths(file), self._packages())
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
                    self._packages(),
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
                if name in self._facts_in(inherited.path).export_names:
                    prior = found.get(inherited.path)
                    if prior is None or inherited.proven:
                        found[inherited.path] = inherited
                pending.append(inherited)
        return tuple(found.values())

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

    def search_text(self, text: str, max_hits: int = MAX_TEXT_HITS) -> tuple[TextHit, ...]:
        """Lines holding ``text``, searched once per text for the life of the index."""
        return self._text_hits(text, max_hits)

    def _search_text(self, text: str, max_hits: int) -> tuple[TextHit, ...]:
        readable = tuple(file for file in self.files if file not in self._unavailable)
        found = self._on_available(
            readable, lambda files: tools.ripgrep_fixed(text, files, self.root, max_hits)
        )
        hits = sorted(hit for hit in found if hit.file in self._scope)
        return tuple(hits[:max_hits])

    def imports(self, file: str) -> tuple[str, ...]:
        source = "\n".join(self._lines_of(file))
        script_paths = self._script_paths(file)
        packages = self._packages()
        resolved = (
            resolve_import(specifier, file, self._scope, script_paths, packages)
            for specifier in imported_modules(source, file)
        )
        return tuple(dict.fromkeys(fact.path for fact in resolved if fact))

    def imports_in(self, file: str, text: str) -> tuple[tuple[ImportFact, frozenset[str] | None], ...]:
        """The scope files ``text``, lines of ``file``, imports from, in source order, each with the
        names it takes by name or None for the whole module."""
        script_paths = self._script_paths(file)
        packages = self._packages()
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
        return self._co_changes(file)[:limit]

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
        return None if file.endswith(".py") else self._script_paths_in(str(PurePosixPath(file).parent))

    def _read_script_paths(self, directory: str) -> ScriptPaths | None:
        return nearest_script_paths(self.root, directory)

    def _read_packages(self) -> Packages:
        """The package.json files of the folders holding scope files, read once, on first use."""
        return Packages(self.root, self.files)

    def _read_lines(self, file: str) -> tuple[str, ...]:
        self._require_in_scope(file)
        content = self._read_bytes(file)
        return split_lines(content.decode(errors="replace")) if content is not None else ()

    def _file_sha256(self, file: str) -> str:
        """The SHA-256 of the bytes the index first read from ``file``."""
        self._require_in_scope(file)
        if file not in self._sha256:
            self._read_bytes(file)
        return self._sha256.get(file, "")

    def _read_bytes(self, file: str) -> bytes | None:
        """The file's bytes, every read checked against the first: once a file changes on disk, its
        facts and lines no longer agree, so it is reported unavailable instead of read."""
        try:
            content = (self.root / file).read_bytes()
        except FileNotFoundError:
            self._unavailable[file] = "disappeared after inventory"
            return None
        digest = hashlib.sha256(content).hexdigest()
        if self._sha256.setdefault(file, digest) != digest:
            self._unavailable[file] = "changed on disk after the index first read it"
            return None
        return content

    def _available_files(self, files: Sequence[str]) -> tuple[str, ...]:
        available = []
        for file in files:
            try:
                if (self.root / file).is_file():
                    available.append(file)
                else:
                    self._unavailable[file] = "disappeared after inventory"
            except FileNotFoundError:
                self._unavailable[file] = "disappeared after inventory"
        return tuple(available)

    def _require_in_scope(self, file: str) -> None:
        if file not in self._scope:
            raise ValueError(f"{file} is outside the index scope")


def _blobs_to_export(repository: Path, commit: str, listed: Sequence[str]) -> dict[str, str]:
    """Object ids of the listed files and of every script config (tsconfig, jsconfig, package.json)
    in ``commit``, keyed by path."""
    tree = _tree_blobs(tools.git(["ls-tree", "-r", "-z", commit], repository))
    return {path: tree[path] for path in [*listed, *_script_configs(tree)]}


def _tree_blobs(listing: str) -> dict[str, str]:
    """Object ids of the regular files in ``git ls-tree -r -z`` output, keyed by path."""
    blobs = {}
    for line in listing.split("\0"):
        details, _, path = line.partition("\t")
        mode, _, object_id = details.partition(" blob ")
        if mode in _REGULAR_FILE_MODES:
            blobs[path] = object_id
    return blobs


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


def _working_git_metadata(root: Path, prefixes: Sequence[str]) -> tuple[str, list[str]]:
    try:
        commit = tools.git(["rev-parse", "HEAD"], root).strip()
        status = tools.git(["status", "--porcelain", "-z", "--untracked-files=all", "--", *prefixes], root)
    except tools.ToolFailedError:
        return "", []
    return commit, _changed_paths(status)


def _regular_files(listing: str) -> list[str]:
    """Paths from ``git ls-files --stage -z`` or ``git ls-tree -r -z`` output whose mode is a regular
    file. NUL separation keeps names with non-ASCII characters exactly as they are on disk."""
    files = []
    for line in listing.split("\0"):
        details, _, path = line.partition("\t")
        if details.split(" ", 1)[0] in _REGULAR_FILE_MODES:
            files.append(path)
    return list(dict.fromkeys(files))


def _commits(log: str) -> list[set[str]]:
    blocks = log.split(_COMMIT_MARK)
    return [{line.strip() for line in block.split("\n") if line.strip()} for block in blocks if block.strip()]
