"""TypeScript and JavaScript project settings from the tsconfig/jsconfig files nearest a folder: path
aliases (``compilerOptions.paths`` and ``baseUrl``) and build folders (``outDir``, ``declarationDir``
and ``rootDir``).

The nearest folder holding a config decides, and every config in it is read, tsconfig.json first:
a tsconfig.json that only lists project references still finds the aliases in its tsconfig.app.json.
Configs are read from disk under the index root, following ``extends`` (a file or a list of files)
through relative paths and the repository's own packages; comments and trailing commas are allowed,
as TypeScript allows them. Package ``extends`` from outside the repository (``@tsconfig/...``) are
not followed. The mapped targets come back as root-relative paths without a suffix; the import
resolver then tries the usual suffixes and ``index`` files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CONFIG_NAMES = ("tsconfig.json", "jsconfig.json")
_CONFIG_FILE = re.compile(r"^(?:tsconfig|jsconfig)(?:\.[\w.-]+)?\.json$")
_MAX_EXTENDS = 8
_COMMENT_OR_STRING = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

# A package specifier in ``extends``, and the root-relative config that extends it, to the
# root-relative config file the specifier names, or None.
PackageConfig = Callable[[str, str], "str | None"]


class _RefusedConfigError(Exception):
    """A config in the chain is a symbolic link or lies outside the root."""


@dataclass(frozen=True)
class AliasTable:
    """One config's aliases. ``base`` is the root-relative directory that alias targets and bare
    specifiers resolve from."""

    base: str
    paths: tuple[tuple[str, tuple[str, ...]], ...]
    has_base_url: bool

    def candidates(self, specifier: str) -> list[str]:
        """Root-relative paths the specifier may name, in the order TypeScript tries them: the
        targets of the one pattern TypeScript picks (an exact match, else the wildcard with the
        longest prefix before ``*``), then the specifier under ``baseUrl``."""
        mapped = []
        chosen = self._chosen_pattern(specifier)
        if chosen is not None:
            pattern, targets = chosen
            wildcard = _match(pattern, specifier) or ""
            mapped = [_join(self.base, target.replace("*", wildcard)) for target in targets]
        if self.has_base_url:
            mapped.append(_join(self.base, specifier))
        return mapped

    def _chosen_pattern(self, specifier: str) -> tuple[str, tuple[str, ...]] | None:
        exact = [entry for entry in self.paths if "*" not in entry[0] and entry[0] == specifier]
        if exact:
            return exact[0]
        matching = [
            entry for entry in self.paths if "*" in entry[0] and _match(entry[0], specifier) is not None
        ]
        return max(matching, key=lambda entry: len(entry[0].split("*", 1)[0]), default=None)


@dataclass(frozen=True)
class ScriptPaths:
    """The alias tables of the configs in the nearest config folder, tsconfig.json first."""

    tables: tuple[AliasTable, ...]

    def candidates(self, specifier: str) -> list[str]:
        return [path for table in self.tables for path in table.candidates(specifier)]


def nearest_script_paths(
    root: Path, directory: str, package_config: PackageConfig | None = None
) -> ScriptPaths | None:
    """The aliases of the configs nearest to ``directory`` (root-relative), or None. A config that
    is a symbolic link, extends one, or whose base directory lies outside the root adds no aliases:
    they stay unknown and bindings through them stay unproven."""
    root = root.resolve()
    current = PurePosixPath(directory)
    while True:
        configs = _configs_in(root / current)
        if configs:
            tables = tuple(
                table for config in configs if (table := _alias_table(root, config, package_config))
            )
            return ScriptPaths(tables) if tables else None
        if str(current) in ("", "."):
            return None
        current = current.parent


def build_layouts(
    root: Path, directory: str, package_config: PackageConfig | None = None
) -> tuple[tuple[str, str | None], ...]:
    """``(output folder, rootDir or None)`` pairs, root-relative, from the configs in ``directory``
    itself: where its build writes files (``outDir``, ``declarationDir``) and the source folder they
    mirror."""
    root = root.resolve()
    layouts: dict[tuple[str, str | None], None] = {}
    for config in _configs_in(root / directory):
        try:
            order = _base_first(root, config, package_config)
        except _RefusedConfigError:
            continue
        folders: dict[str, str | None] = {}
        for path in order:
            compiler = _read(path).get("compilerOptions", {})
            for key in ("outDir", "declarationDir", "rootDir"):
                if isinstance(compiler.get(key), str):
                    folders[key] = _inside(root, path.parent / compiler[key])
        for key in ("outDir", "declarationDir"):
            if folders.get(key):
                layouts[(folders[key], folders.get("rootDir"))] = None
    return tuple(layouts)


def _configs_in(folder: Path) -> list[Path]:
    """The folder's configs, tsconfig.json and jsconfig.json first, then the rest by name. Symbolic
    links are listed, so their folder still decides, and ``_base_first`` refuses to read them."""
    if not folder.is_dir():
        return []
    found = [
        path
        for path in folder.iterdir()
        if _CONFIG_FILE.match(path.name) and (path.is_file() or path.is_symlink())
    ]
    return sorted(found, key=lambda path: (_rank(path.name), path.name))


def _rank(name: str) -> int:
    return CONFIG_NAMES.index(name) if name in CONFIG_NAMES else len(CONFIG_NAMES)


def _alias_table(root: Path, config: Path, package_config: PackageConfig | None) -> AliasTable | None:
    """A child config overrides its parents. ``baseUrl`` is relative to the config that sets it;
    without one, ``paths`` targets are relative to the config that sets ``paths``."""
    try:
        order = _base_first(root, config, package_config)
    except _RefusedConfigError:
        return None
    base_url_dir: Path | None = None
    paths_dir: Path | None = None
    paths: dict = {}
    for path in order:
        compiler = _read(path).get("compilerOptions", {})
        if isinstance(compiler.get("baseUrl"), str):
            base_url_dir = path.parent / compiler["baseUrl"]
        if isinstance(compiler.get("paths"), dict):
            paths_dir, paths = path.parent, compiler["paths"]
    base_dir = base_url_dir or paths_dir
    if base_dir is None:
        return None
    base = _inside(root, base_dir)
    if base is None:
        return None
    patterns = tuple(
        (pattern, tuple(target for target in targets if isinstance(target, str)))
        for pattern, targets in paths.items()
        if isinstance(targets, list)
    )
    return AliasTable(base, patterns, base_url_dir is not None)


def _base_first(root: Path, config: Path, package_config: PackageConfig | None, depth: int = 0) -> list[Path]:
    """The configs ``config`` extends, each base before the config extending it, then ``config``.
    A missing or outside-repository package base is skipped; raises ``_RefusedConfigError`` when one is a
    symbolic link or lies outside the root."""
    if config.is_symlink() or not config.resolve().is_relative_to(root):
        raise _RefusedConfigError(str(config))
    parents = _read(config).get("extends")
    targets = [parents] if isinstance(parents, str) else parents if isinstance(parents, list) else []
    order: list[Path] = []
    for target in targets:
        if depth >= _MAX_EXTENDS or not isinstance(target, str):
            break
        base = _extends_target(root, config, target, package_config)
        if base is not None:
            order += _base_first(root, base, package_config, depth + 1)
    return [*order, config.resolve()]


def _extends_target(
    root: Path, config: Path, target: str, package_config: PackageConfig | None
) -> Path | None:
    if target.startswith("."):
        base = config.parent / (target if target.endswith(".json") else f"{target}.json")
    elif package_config is not None and (
        named := package_config(target, config.resolve().relative_to(root).as_posix())
    ):
        base = root / named
    else:
        return None
    if base.is_symlink() or not base.resolve().is_relative_to(root):
        raise _RefusedConfigError(str(base))
    return base if base.is_file() else None


def _inside(root: Path, path: Path) -> str | None:
    resolved = path.resolve()
    return normalised(resolved.relative_to(root).as_posix()) if resolved.is_relative_to(root) else None


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
