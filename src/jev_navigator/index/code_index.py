"""Mechanical code lookups over a narrowed set of files: no model, only ast-grep, ripgrep and git.

Every lookup stays inside the scope the index was built with. ``max_files`` is an explicit caller
policy only; the index has no default refusal and never drops files from a requested scope.
"""

from __future__ import annotations

import hashlib
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from pathlib import Path, PurePosixPath

from . import tools
from .bindings import Binding, BindingResolver, CallFacts, binding_from_facts
from .fact_cache import FactCache
from .imports import (
    ImportFact,
    imported_modules,
    imported_names,
    reexported_names,
    resolve_import,
)
from .languages import (
    language_of,
)
from .packages import Packages
from .scope_scan import FileFacts, FileStructure, ReferenceMatch, Unparsed, scan_facts
from .spans import CallEdge, CallSite, CodeSlice, Reference, Span, TextHit
from .tsconfig import ScriptPaths, nearest_script_paths

DEFAULT_WINDOW_RADIUS = 10
MAX_TEXT_HITS = 20
CO_CHANGE_COMMITS = 200
_COMMIT_MARK = "@@commit@@"
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})
ScanObserver = Callable[[str, str, int], None]


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
        self._lines_of = cache(self._read_lines)
        self._file_sha256 = cache(self._read_file_sha256)
        self._script_paths_in = cache(self._read_script_paths)
        self._packages = cache(self._read_packages)
        self._unparsed = Unparsed()
        self._unavailable: dict[str, str] = {}
        self._facts: dict[str, FileFacts] = {}
        self._facts_lock = threading.RLock()
        self._fact_cache = FactCache(fact_cache_dir)
        self._files_for_name = cache(self._candidate_files)
        self._calls_named = cache(self._calls_with_name)
        self._references_named = cache(self._references_with_name)
        self._definitions = cache(self._definitions_by_name)
        self._top_level_in = cache(self._top_level_spans)
        self._names_imported = cache(self._read_imported_names)
        self._binding = cache(self._compute_binding)

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
    def parser_scans_completed(self) -> tuple[str, ...]:
        return ("facts",) if not self.parser_scans_pending else ()

    @property
    def parser_scans_pending(self) -> tuple[str, ...]:
        available = set(self._available_files(self._code_files))
        return () if available <= self._facts.keys() else ("facts",)

    @property
    def unavailable_files(self) -> dict[str, str]:
        """Inventory entries that disappeared after this working-directory index was created."""
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
        edges: dict[str, CallEdge] = {}
        for call in self._facts_in(function.file).calls:
            inside = call.file == function.file and function.start <= call.line <= function.end
            if inside and call.name not in edges:
                binding = self.binding_of(call.file, call.line, call.name, call.receiver)
                edges[call.name] = CallEdge(call.name, call.line, binding)
        return tuple(edges.values())

    def find_references(self, name: str) -> tuple[Reference, ...]:
        """Uses of ``name`` that are not calls: arguments, collection entries, assignments,
        decorators, exports, returns, method receivers, types and conditions, each with its role,
        holder and binding. Code reached this way (a callback, a registry entry, a parameter typed
        with a class) has no call edge to follow."""
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

    def binding_of(self, file: str, line: int, name: str, receiver: str | None) -> Binding:
        """Computed once per call site and cached for the life of the index."""
        return self._binding(file, line, name, receiver)

    def _references(self, matches: Iterable[ReferenceMatch]) -> tuple[Reference, ...]:
        return tuple(
            Reference(
                match.name,
                match.file,
                match.line,
                match.role,
                self.enclosing_symbol(match.file, match.line),
                self.binding_of(match.file, match.line, match.name, None),
            )
            for match in sorted(set(matches))
        )

    def _compute_binding(self, file: str, line: int, name: str, receiver: str | None) -> Binding:
        if self.binding_resolver is not None:
            injected = self.binding_resolver.resolve_call(file, line, name, receiver)
            if injected is not None:
                return injected
        definitions = tuple(span for span in self.find_definition(name) if self._is_callable(span))
        facts = CallFacts(
            file,
            name,
            receiver,
            definitions,
            tuple(span for span in definitions if span in self._top_level_in(file)),
            self._imported_from(file, name),
            self.observed_unparsed_files | self.unavailable_files.keys(),
        )
        return binding_from_facts(facts)

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

    def _is_callable(self, span: Span) -> bool:
        return span in self._facts_in(span.file).structure.symbols

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

    def _candidate_files(self, name: str) -> tuple[str, ...]:
        return tools.ripgrep_files(name, self._available_files(self._code_files), self.root)

    def _facts_in(self, file: str) -> FileFacts:
        self._require_in_scope(file)
        self._ensure_facts((file,))
        return self._facts.get(file, FileFacts(_NO_STRUCTURE, (), ()))

    def _ensure_facts(self, files: Sequence[str]) -> None:
        with self._facts_lock:
            missing = [
                file
                for file in self._available_files(files)
                if language_of(file) is not None and file not in self._facts
            ]
            if not missing:
                return
            to_scan = []
            contents: dict[str, bytes] = {}
            for file in missing:
                try:
                    content = (self.root / file).read_bytes()
                except FileNotFoundError:
                    self._unavailable[file] = "disappeared after inventory"
                    continue
                contents[file] = content
                cached = self._fact_cache.load(file, content)
                if cached is None:
                    to_scan.append(file)
                else:
                    self._facts[file] = cached
                    if cached.incomplete:
                        self._unparsed.add("facts", (file,))
            if not to_scan:
                return
            scanned = self._run_scan(
                "facts",
                lambda: self._scan_available_facts(to_scan),
                len(to_scan),
            )
            self._facts.update(scanned)
            for file, facts in scanned.items():
                self._fact_cache.save(file, contents[file], facts)

    def _scan_available_facts(self, files: Sequence[str]) -> dict[str, FileFacts]:
        remaining = tuple(files)
        while remaining:
            try:
                return scan_facts(remaining, self.root, self._lines_of, self._unparsed)
            except tools.ToolFailedError:
                available = self._available_files(remaining)
                if available == remaining:
                    raise
                remaining = available
        return {}

    def _top_level_spans(self, file: str) -> frozenset[Span]:
        """Symbols of ``file`` that no class or other function contains."""
        symbols = self.symbols_in(file)
        return frozenset(
            span
            for span in symbols
            if not any(other != span and other.contains(span.start) for other in symbols)
        )

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
        found = tools.ripgrep_fixed(text, self._available_files(self.files), self.root, max_hits)
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

    def dependents(self, file: str) -> tuple[str, ...]:
        self._require_in_scope(file)
        return tuple(path for path in self._code_files if path != file and file in self.imports(path))

    def co_changed_files(self, file: str, limit: int = 5) -> tuple[tuple[str, int], ...]:
        """Scope files most often committed together with ``file``, with their shared-commit counts."""
        self._require_in_scope(file)
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
        return tuple(sorted(in_scope, key=lambda item: (-item[1], item[0]))[:limit])

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
        try:
            return _split_lines((self.root / file).read_text(errors="replace"))
        except FileNotFoundError:
            self._unavailable[file] = "disappeared after inventory"
            return ()

    def _read_file_sha256(self, file: str) -> str:
        self._require_in_scope(file)
        try:
            return hashlib.sha256((self.root / file).read_bytes()).hexdigest()
        except FileNotFoundError:
            self._unavailable[file] = "disappeared after inventory"
            return ""

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


def _split_lines(text: str) -> tuple[str, ...]:
    """Lines split at newlines only, as the parser counts them; ``str.splitlines`` also splits at form
    feeds and other separators, which would shift every line number after them."""
    lines = text.replace("\r", "").split("\n")
    return tuple(lines[:-1] if lines and lines[-1] == "" else lines)


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
