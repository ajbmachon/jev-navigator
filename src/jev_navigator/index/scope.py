"""Which files a search covers, decided from paths, git and file heads, never by parsing.

A scope starts from the directory's file listing (``tools.listed_files``) and keeps only files JVN
parses, plus markup when docs are asked for. Tests, generated code, vendored code and docs are left
out unless asked for:

- tests: ``is_test_file``, a test folder or a test file name;
- generated: a true ``linguist-generated`` attribute, or a comment line holding ``@generated`` or
  ``do not edit``, in any case, among a file's first lines;
- vendored: a true ``linguist-vendored`` attribute, or a ``vendor``, ``third_party`` or
  ``node_modules`` folder;
- docs: a ``docs`` folder, or a markup file.

A false linguist attribute keeps a file its path or header would leave out. ``include`` and
``exclude`` entries are folders or files when they hold no ``*``, ``?`` or ``[``; otherwise they are
globs over the whole path, where ``**`` crosses folders and a glob without ``/`` matches the file
name at any depth unless a leading ``/`` anchors it at the root. A scope with more files than
``max_files`` is refused with its file counts per folder and language, labelled so each label, as an
``include`` entry under the same other filters, keeps exactly the files it counts.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath

from . import tools
from .languages import LANGUAGE_BY_SUFFIX, language_of

HEADER_LINES = 10
HEADER_BYTES = 4096
GENERATED_MARKERS = (b"@generated", b"do not edit")
COMMENT_STARTS = (b"#", b"//", b"/*", b"*", b"<!--")
VENDORED_FOLDERS = frozenset({"vendor", "third_party", "node_modules"})
DOCS_FOLDER = "docs"
MARKUP_SUFFIXES = frozenset({".md", ".mdx", ".markdown", ".rst", ".adoc", ".asciidoc"})
MARKUP = "markup"
KNOWN_LANGUAGES = frozenset(LANGUAGE_BY_SUFFIX.values())
TEST_FOLDERS = frozenset({"test", "tests", "__tests__", "spec"})
_TEST_FILE_NAME = re.compile(r"^test_|_test\.|\.test\.|\.spec\.|^conftest\.py$")
_GLOB_CHARACTERS = frozenset("*?[")
_GENERATED = "linguist-generated"
_VENDORED = "linguist-vendored"


class InvalidScopeError(ValueError):
    """A scope field cannot be used; ``path`` names the field the way a request error does."""

    def __init__(self, path: str, problem: str) -> None:
        super().__init__(f"{path}: {problem}")
        self.path = path
        self.problem = problem


@dataclass(frozen=True, kw_only=True)
class Scope:
    """The request's ``scope`` object, built from the canonical request, whose schema owns the cap
    and the switches' values; an empty filter or a missing ``changed_since`` means not filtered."""

    repo: Path
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    with_tests: bool
    with_generated: bool
    with_vendored: bool
    with_docs: bool
    changed_since: str | None = None
    max_files: int


@dataclass(frozen=True)
class ResolvedScope:
    """The files a search covers, and every filter that chose them."""

    root: Path
    files: tuple[str, ...]
    filters: Mapping[str, object]


@dataclass(frozen=True)
class ScopeRefusal:
    """A scope over its cap, counted without parsing: ``files`` in scope against ``cap``, by folder
    one level below the folder all of them share (``/dir/*`` for files directly in it) and by
    language."""

    files: int
    cap: int
    filters: Mapping[str, object]
    counts_by_folder: Mapping[str, int]
    counts_by_language: Mapping[str, int]


def resolve_scope(scope: Scope) -> ResolvedScope | ScopeRefusal:
    root = _checked_root(scope)
    _check_languages(scope)
    changed_since_commit = _resolved_ref(root, scope.changed_since)
    files = [path for path in tools.listed_files(root) if _kept_by_path(scope, path)]
    if changed_since_commit is not None:
        files = _changed(root, changed_since_commit, files)
    files = _kept_by_attributes_and_header(root, scope, files)
    filters = _filters(scope, changed_since_commit)
    if len(files) > scope.max_files:
        return ScopeRefusal(
            len(files), scope.max_files, filters, counts_by_folder(files), counts_by_language(files)
        )
    return ResolvedScope(root, tuple(files), filters)


def is_test_file(path: str) -> bool:
    """A file in a test folder at any depth, or named like a test (``test_*``, ``*_test.*``,
    ``*.test.*``, ``*.spec.*``, ``conftest.py``)."""
    folders, _, name = path.rpartition("/")
    return not TEST_FOLDERS.isdisjoint(folders.split("/")) or bool(_TEST_FILE_NAME.search(name))


def counts_by_folder(files: Sequence[str]) -> dict[str, int]:
    """File counts one level below the deepest folder every file shares."""
    shared = _shared_folder(files)
    counts: Counter[str] = Counter()
    for file in files:
        below = PurePosixPath(file).parts[len(shared) :]
        if len(below) > 1:
            counts["/".join((*shared, below[0])) + "/"] += 1
        else:
            counts["/" + "/".join((*shared, "*"))] += 1
    return dict(counts)


def counts_by_language(files: Iterable[str]) -> dict[str, int]:
    return dict(Counter(language_of(file) or MARKUP for file in files))


def _checked_root(scope: Scope) -> Path:
    root = Path(scope.repo)
    if not root.is_dir():
        raise InvalidScopeError("/scope/repo", f"{root} is not a directory")
    return root


def _check_languages(scope: Scope) -> None:
    unknown = sorted(set(scope.languages) - KNOWN_LANGUAGES)
    if unknown:
        raise InvalidScopeError("/scope/languages", f"unknown {unknown}; known: {sorted(KNOWN_LANGUAGES)}")


def _resolved_ref(root: Path, ref: str | None) -> str | None:
    """The commit ``ref`` names. ``--end-of-options`` reads a ref that begins with a dash, such as
    a tag ``-v1``, as a name, never as a git option."""
    if ref is None:
        return None
    try:
        return tools.git(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], root).strip()
    except tools.ToolFailedError as error:
        raise InvalidScopeError("/scope/changed_since", f"{ref!r} names no commit here: {error}") from error


def _kept_by_path(scope: Scope, path: str) -> bool:
    if not _parsed_or_wanted_markup(scope, path):
        return False
    if scope.include and not any(_matches(entry, path) for entry in scope.include):
        return False
    if any(_matches(entry, path) for entry in scope.exclude):
        return False
    if scope.languages and language_of(path) not in scope.languages and not _is_markup(path):
        return False
    if not scope.with_tests and is_test_file(path):
        return False
    return scope.with_docs or not _is_docs(path)


def _parsed_or_wanted_markup(scope: Scope, path: str) -> bool:
    return language_of(path) is not None or (scope.with_docs and _is_markup(path))


def _is_markup(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in MARKUP_SUFFIXES


def _is_docs(path: str) -> bool:
    return DOCS_FOLDER in PurePosixPath(path).parts[:-1] or _is_markup(path)


def _in_vendored_folder(path: str) -> bool:
    return not VENDORED_FOLDERS.isdisjoint(PurePosixPath(path).parts[:-1])


def _changed(root: Path, commit: str, files: Sequence[str]) -> list[str]:
    """The files that differ from ``commit`` in the working tree, untracked files included."""
    differing = tools.git(
        ["diff", "--name-only", "-z", "--relative", "--no-renames", "--diff-filter=d", commit, "--"], root
    )
    untracked = tools.git(["ls-files", "-z", "-o", "--exclude-standard"], root)
    changed = {path for path in (differing + untracked).split("\0") if path}
    return [path for path in files if path in changed]


@dataclass(frozen=True)
class _Linguist:
    """A file's linguist attributes: True when set or ``true``, False when unset or ``false``, None
    when unspecified."""

    generated: bool | None = None
    vendored: bool | None = None


def _kept_by_attributes_and_header(root: Path, scope: Scope, files: Sequence[str]) -> list[str]:
    if scope.with_generated and scope.with_vendored:
        return list(files)
    attributes = _linguist_attributes(root, files)
    return [
        path
        for path in files
        if (scope.with_vendored or not _is_vendored(path, attributes.get(path, _Linguist())))
        and (scope.with_generated or not _is_generated(root, path, attributes.get(path, _Linguist())))
    ]


def _is_vendored(path: str, linguist: _Linguist) -> bool:
    return linguist.vendored if linguist.vendored is not None else _in_vendored_folder(path)


def _is_generated(root: Path, path: str, linguist: _Linguist) -> bool:
    return linguist.generated if linguist.generated is not None else _has_generated_header(root, path)


def _linguist_attributes(root: Path, files: Sequence[str]) -> dict[str, _Linguist]:
    """Read with one ``git check-attr`` over standard input; outside git there are none."""
    if not files or not tools.inside_git_worktree(root):
        return {}
    output = tools.git(["check-attr", "-z", "--stdin", _GENERATED, _VENDORED], root, stdin="\0".join(files))
    fields = output.split("\0")
    values: dict[str, dict[str, bool | None]] = {}
    for path, attribute, value in zip(fields[0::3], fields[1::3], fields[2::3], strict=False):
        values.setdefault(path, {})[attribute] = _attribute_value(value)
    return {path: _Linguist(found.get(_GENERATED), found.get(_VENDORED)) for path, found in values.items()}


def _attribute_value(value: str) -> bool | None:
    if value in ("set", "true"):
        return True
    if value in ("unset", "false"):
        return False
    return None


def _has_generated_header(root: Path, path: str) -> bool:
    with (root / path).open("rb") as file:
        head = file.read(HEADER_BYTES)
    return any(_marks_generated(line) for line in head.split(b"\n")[:HEADER_LINES])


def _marks_generated(line: bytes) -> bool:
    comment = line.strip().lower()
    return comment.startswith(COMMENT_STARTS) and any(marker in comment for marker in GENERATED_MARKERS)


def _matches(entry: str, path: str) -> bool:
    if _GLOB_CHARACTERS.isdisjoint(entry):
        folder = entry.strip("/")
        return path == folder or path.startswith(f"{folder}/")
    return bool(_glob(entry).fullmatch(path))


@cache
def _glob(pattern: str) -> re.Pattern[str]:
    anchored = pattern.startswith("/")
    pattern = pattern.strip("/")
    if not anchored and "/" not in pattern:
        pattern = f"**/{pattern}"
    expression = []
    position = 0
    while position < len(pattern):
        token, position = _glob_token(pattern, position)
        expression.append(token)
    return re.compile("".join(expression))


def _glob_token(pattern: str, position: int) -> tuple[str, int]:
    if pattern.startswith("**/", position):
        return "(?:.*/)?", position + 3
    if pattern.startswith("**", position):
        return ".*", position + 2
    character = pattern[position]
    if character == "*":
        return "[^/]*", position + 1
    if character == "?":
        return "[^/]", position + 1
    if character == "[" and (end := pattern.find("]", position + 2)) != -1:
        members = pattern[position + 1 : end].replace("\\", "\\\\")
        return f"[{'^' + members[1:] if members.startswith('!') else members}]", end + 1
    return re.escape(character), position + 1


def _shared_folder(files: Sequence[str]) -> tuple[str, ...]:
    return tuple(os.path.commonprefix([PurePosixPath(file).parts[:-1] for file in files]))


def _filters(scope: Scope, changed_since_commit: str | None) -> dict[str, object]:
    return {
        "include": list(scope.include),
        "exclude": list(scope.exclude),
        "languages": list(scope.languages),
        "with_tests": scope.with_tests,
        "with_generated": scope.with_generated,
        "with_vendored": scope.with_vendored,
        "with_docs": scope.with_docs,
        "changed_since": scope.changed_since,
        "changed_since_commit": changed_since_commit,
        "max_files": scope.max_files,
    }
