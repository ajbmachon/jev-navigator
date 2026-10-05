"""The files under an index root that the index covers, and every one it leaves out, with the reason.

A left-out folder that holds no indexed file is named once, ending in ``/``, so an ignored
``node_modules/`` is one entry, not thousands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from . import tools

IGNORED = "ignored"
SEPARATE_REPOSITORY = "a separate git repository"
SYMBOLIC_LINK = "a symbolic link"
NOT_TRACKED = "not tracked by git"
NO_FILE = "no file at this path"
_UNTRACKED = ("ls-files", "-z", "-o", "--exclude-standard")
_RIPGREP_FILES = ("--files", "--hidden", "--null", "--glob", "!.git", "--glob", "!.git/**")


@dataclass(frozen=True)
class Listing:
    files: tuple[str, ...]
    not_indexed: Mapping[str, str] = field(default_factory=dict)


def working_files(cwd: Path, prefixes: Sequence[str] = ()) -> Listing:
    """Every regular, non-symlink file under ``cwd``, tracked by git or not, minus ignored ones,
    including hidden paths. In a Git worktree git's ignore rules apply, with the lines of an enclosing
    repository's .gitignore that match ``cwd``; a folder outside Git uses ripgrep's ignore policy."""
    if tools.inside_git_worktree(cwd):
        return _git_working_files(cwd, prefixes)
    return _plain_folder_files(cwd, prefixes)


def left_out_of_tracked(cwd: Path, prefixes: Sequence[str], tracked: Iterable[str]) -> dict[str, str]:
    """What a listing of the ``tracked`` files under ``cwd`` leaves out: untracked and ignored files,
    and a requested path with no file."""
    tracked = tuple(tracked)
    untracked, left_out = _regular_files(cwd, _git_entries(cwd, _UNTRACKED, prefixes))
    left_out |= dict.fromkeys(_collapsed(untracked, tracked), NOT_TRACKED)
    left_out |= _ignored_by_git(cwd, prefixes, tracked)
    left_out |= {prefix: NO_FILE for prefix in prefixes if not (cwd / prefix).exists()}
    return left_out


def _git_working_files(cwd: Path, prefixes: Sequence[str]) -> Listing:
    files, left_out = _regular_files(cwd, _git_entries(cwd, (*_UNTRACKED, "-c"), prefixes))
    return Listing(files, left_out | _ignored_by_git(cwd, prefixes, files))


def _plain_folder_files(cwd: Path, prefixes: Sequence[str]) -> Listing:
    listed = _ripgrep_entries(cwd, prefixes)
    files, left_out = _regular_files(cwd, listed)
    every = _ripgrep_entries(cwd, prefixes, "--no-ignore")
    ignored = _collapsed(set(every) - set(listed), files)
    links = [link for link in _symbolic_links(cwd, prefixes) if not _inside_any(link, ignored)]
    return Listing(files, left_out | dict.fromkeys(ignored, IGNORED) | dict.fromkeys(links, SYMBOLIC_LINK))


def _symbolic_links(cwd: Path, prefixes: Sequence[str]) -> list[str]:
    """ripgrep lists no symbolic link, so a plain folder's links come from a walk that never follows
    one: about 0.3 seconds over 105,000 files."""
    starts = prefixes or (".",)
    command = ["find", *starts, "-name", ".git", "-prune", "-o", "-type", "l", "-print0"]
    return _entries(tools.run_command(command, cwd))


def _inside_any(path: str, folders: Iterable[str]) -> bool:
    return any(folder.endswith("/") and path.startswith(folder) for folder in folders)


def _ignored_by_git(cwd: Path, prefixes: Sequence[str], kept: Iterable[str]) -> dict[str, str]:
    """``--directory`` stops git at a wholly ignored folder instead of walking it: in a repository with
    104,000 ignored files that takes 0.1 seconds instead of 15. ``--no-empty-directory`` would hide
    every ignored entry under a folder the enclosing repository does not track, so it is not given."""
    ignored = _git_entries(cwd, (*_UNTRACKED, "-i", "--directory"), prefixes)
    return dict.fromkeys(_collapsed(ignored, kept), IGNORED)


def _git_entries(cwd: Path, arguments: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    return _entries(tools.git([*arguments, "--", *prefixes], cwd))


def _ripgrep_entries(cwd: Path, prefixes: Sequence[str], *options: str) -> list[str]:
    return _entries(tools.run_command([*tools.RIPGREP_SAFE, *_RIPGREP_FILES, *options, *prefixes], cwd))


def _entries(output: str) -> list[str]:
    return [raw.removeprefix("./") for raw in output.split("\0") if raw]


def _regular_files(cwd: Path, paths: Iterable[str]) -> tuple[tuple[str, ...], dict[str, str]]:
    """The regular files among ``paths``, and the rest with their reason. A tracked file that is gone
    from the disk is neither: there is nothing under the root to name."""
    files: list[str] = []
    left_out: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        candidate = cwd / path
        if candidate.is_symlink():
            left_out[path] = SYMBOLIC_LINK
        elif candidate.is_dir():
            left_out[f"{path.rstrip('/')}/"] = SEPARATE_REPOSITORY
        elif candidate.is_file():
            files.append(path)
    return tuple(sorted(files)), left_out


def _collapsed(left_out: Iterable[str], kept: Iterable[str]) -> set[str]:
    """Each left-out file, or its outermost folder that holds no kept file."""
    holding = {parent for file in kept for parent in PurePosixPath(file).parents}
    named = set()
    for file in left_out:
        folders = reversed(PurePosixPath(file).parents[:-1])
        outermost = next((folder for folder in folders if folder not in holding), None)
        named.add(file if outermost is None else f"{outermost}/")
    return named
