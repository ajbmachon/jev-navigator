"""TypeScript path aliases (``compilerOptions.paths`` and ``baseUrl``) from the nearest tsconfig.json,
or jsconfig.json in a folder without one, and the build layout (``outDir``, ``declarationDir`` and
``rootDir``) of a package's tsconfig.json.

The config is read from disk under the index root, following relative ``extends`` chains. Comments
and trailing commas are allowed, as TypeScript allows them. Package ``extends`` (``@tsconfig/...``)
are not followed. The mapped targets come back as root-relative paths without a suffix; the import
resolver then tries the usual suffixes and ``index`` files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CONFIG_NAMES = ("tsconfig.json", "jsconfig.json")
_MAX_EXTENDS = 8
_COMMENT_OR_STRING = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


@dataclass(frozen=True)
class ScriptPaths:
    """``base`` is the root-relative directory that alias targets and bare specifiers resolve from."""

    base: str
    paths: tuple[tuple[str, tuple[str, ...]], ...]
    has_base_url: bool

    def candidates(self, specifier: str) -> list[str]:
        """Root-relative paths the specifier may name, in the order TypeScript tries them: the
        targets of the one pattern TypeScript picks (an exact match, else the wildcard with the
        longest prefix before ``*``), then the specifier under ``baseUrl``."""
        mapped = []
        patterns = dict(self.paths)
        chosen = best_pattern(patterns, specifier)
        if chosen is not None:
            pattern, wildcard = chosen
            mapped = [_join(self.base, target.replace("*", wildcard)) for target in patterns[pattern]]
        if self.has_base_url:
            mapped.append(_join(self.base, specifier))
        return mapped


def best_pattern(patterns: Iterable[str], specifier: str) -> tuple[str, str] | None:
    """The pattern TypeScript picks for ``specifier`` (Node picks package.json keys the same way),
    with what its ``*`` stands for: an exact match, else the wildcard with the longest prefix."""
    matching = [(pattern, star) for pattern in patterns if (star := _match(pattern, specifier)) is not None]
    exact = [entry for entry in matching if "*" not in entry[0]]
    longest_prefix = max(matching, key=lambda entry: len(entry[0].split("*", 1)[0]), default=None)
    return exact[0] if exact else longest_prefix


def nearest_script_paths(root: Path, directory: str) -> ScriptPaths | None:
    """The alias settings of the config nearest to ``directory`` (root-relative), or None.
    None also when that config, a config it extends or its base directory is a symbolic link or lies
    outside the root: the aliases then stay unknown and bindings through them stay unproven."""
    root = root.resolve()
    current = PurePosixPath(directory)
    while True:
        for name in CONFIG_NAMES:
            config = root / current / name
            if config.is_symlink():
                return None
            if config.is_file():
                return _script_paths(root, config)
        if str(current) in ("", "."):
            return None
        current = current.parent


def build_layout(root: Path, directory: str) -> list[tuple[str, str]]:
    """``(output folder, source folder)`` pairs, root-relative, from the tsconfig.json in
    ``directory``: its ``outDir`` and ``declarationDir`` mirror its ``rootDir``."""
    root = root.resolve()
    config = root / directory / CONFIG_NAMES[0]
    chain = _extends_chain(root, config) if config.is_file() and not config.is_symlink() else None
    folders: dict[str, str | None] = {}
    for path in reversed(chain or []):
        compiler = _read(path).get("compilerOptions", {})
        for key in ("outDir", "declarationDir", "rootDir"):
            if isinstance(compiler.get(key), str):
                resolved = (path.parent / compiler[key]).resolve()
                folders[key] = (
                    normalised(resolved.relative_to(root).as_posix())
                    if resolved.is_relative_to(root)
                    else None
                )
    source = folders.get("rootDir")
    return [
        (output, source)
        for key in ("outDir", "declarationDir")
        if (output := folders.get(key)) and source is not None
    ]


def _script_paths(root: Path, config: Path) -> ScriptPaths | None:
    """A child config overrides its parents. ``baseUrl`` is relative to the config that sets it;
    without one, ``paths`` targets are relative to the config that sets ``paths``."""
    chain = _extends_chain(root, config)
    if chain is None:
        return None
    base_url_dir: Path | None = None
    paths_dir: Path | None = None
    paths: dict = {}
    for path in reversed(chain):
        compiler = _read(path).get("compilerOptions", {})
        if isinstance(compiler.get("baseUrl"), str):
            base_url_dir = path.parent / compiler["baseUrl"]
        if isinstance(compiler.get("paths"), dict):
            paths_dir, paths = path.parent, compiler["paths"]
    base_dir = base_url_dir or paths_dir
    if base_dir is None or not base_dir.resolve().is_relative_to(root):
        return None
    base = normalised(base_dir.resolve().relative_to(root).as_posix())
    patterns = tuple(
        (pattern, tuple(target for target in targets if isinstance(target, str)))
        for pattern, targets in paths.items()
        if isinstance(targets, list)
    )
    return ScriptPaths(base, patterns, base_url_dir is not None)


def _extends_chain(root: Path, config: Path) -> list[Path] | None:
    """The config and the relative configs it extends, or None when one of them is a symbolic link
    or lies outside the root."""
    chain = [config]
    while len(chain) < _MAX_EXTENDS:
        parent = _read(chain[-1]).get("extends")
        if not isinstance(parent, str) or not parent.startswith("."):
            break
        target = chain[-1].parent / (parent if parent.endswith(".json") else f"{parent}.json")
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            return None
        if not target.is_file():
            break
        chain.append(target.resolve())
    return chain


def _read(config: Path) -> dict:
    try:
        text = config.read_text(errors="replace")
    except OSError:
        return {}
    without_comments = _COMMENT_OR_STRING.sub(_keep_strings, text)
    try:
        loaded = json.loads(_TRAILING_COMMA.sub(r"\1", without_comments))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _keep_strings(match: re.Match) -> str:
    return match.group(0) if match.group(0).startswith('"') else ""


def _match(pattern: str, specifier: str) -> str | None:
    """The part of ``specifier`` that the pattern's ``*`` stands for, or None when it does not match."""
    if "*" not in pattern:
        return "" if pattern == specifier else None
    prefix, suffix = pattern.split("*", 1)
    long_enough = len(specifier) >= len(prefix) + len(suffix)
    if not (long_enough and specifier.startswith(prefix) and specifier.endswith(suffix)):
        return None
    return specifier[len(prefix) : len(specifier) - len(suffix)]


def _join(base: str, target: str) -> str:
    return normalised(f"{base}/{target}" if base else target)


def normalised(path: str) -> str:
    """A root-relative POSIX path with ``.`` and ``..`` segments folded away."""
    parts: list[str] = []
    for part in PurePosixPath(path).parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)
