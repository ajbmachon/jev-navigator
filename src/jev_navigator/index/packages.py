"""package.json facts for script imports: the ``imports`` map a package resolves ``#`` specifiers
through, and the repository's own packages by name, with their ``exports`` map and entry fields.

Manifests are read from the folders that hold scope files, up to the index root, whatever tool
manages the workspace. A manifest that is a symbolic link is not read.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path, PurePosixPath

from .tsconfig import best_pattern, build_layout, normalised

# Conditions in the order a code navigator wants them: an explicit "source", then "types", whose
# declarations mirror the source file by file, then runtime conditions, which may point every
# subpath at one bundle. Other conditions follow in written order.
CONDITIONS = ("source", "types", "import", "module", "development", "browser", "node", "default", "require")
ENTRY_FIELDS = ("source", "types", "typings", "module", "main")
_SCRIPT_EXTENSION = re.compile(r"(\.d)?\.[cm]?[jt]sx?$")


@dataclass(frozen=True)
class Manifest:
    """One package.json; ``directory`` is root-relative, and targets come back relative to it."""

    directory: str
    fields: dict

    def import_targets(self, specifier: str) -> list[str]:
        """Targets of a ``#`` specifier, best first; a target without ``./`` names a package."""
        imports = self.fields.get("imports")
        return _lookup(imports, specifier) if isinstance(imports, dict) else []

    def export_targets(self, subpath: str) -> list[str]:
        """Targets of the package itself (``subpath`` empty) or of one subpath, best first. Without
        an ``exports`` map, the entry fields, or the subpath inside the package."""
        key = f"./{subpath}" if subpath else "."
        exports = self.fields.get("exports")
        if isinstance(exports, dict) and any(k.startswith(".") for k in exports):
            return _lookup(exports, key)
        if exports is not None:
            return [] if subpath else _pick(exports)
        entries = [value for field in ENTRY_FIELDS if isinstance(value := self.fields.get(field), str)]
        return [key] if subpath else [*entries, "./index"]


class Packages:
    def __init__(self, root: Path, files: Iterable[str]) -> None:
        self.root = Path(root)
        self._layout = cache(partial(build_layout, self.root))
        folders = {_folder(str(parent)) for file in files for parent in PurePosixPath(file).parents}
        self._manifests = {
            folder: Manifest(folder, fields) for folder in sorted(folders) if (fields := self._read(folder))
        }
        self._named: dict[str, list[Manifest]] = defaultdict(list)
        for manifest in self._manifests.values():
            if isinstance(name := manifest.fields.get("name"), str):
                self._named[name].append(manifest)

    def scope_of(self, file: str) -> Manifest | None:
        """The nearest package.json above ``file``: the one Node reads its ``#`` imports from."""
        return next(
            (m for p in PurePosixPath(file).parents if (m := self._manifests.get(_folder(str(p))))), None
        )

    def named(self, name: str, near: str) -> Manifest | None:
        """The package called ``name`` closest to the file ``near``: one it lies in, else the one
        sharing the most leading folders with it. None when no package or several tie."""
        claimants = self._named.get(name, [])
        closeness = [_closeness(manifest.directory, near) for manifest in claimants]
        closest = max(closeness, default=None)
        best = [manifest for manifest, c in zip(claimants, closeness, strict=True) if c == closest]
        return best[0] if len(best) == 1 else None

    def target_bases(self, directory: str, target: str) -> list[str]:
        """Root-relative bases for a target the package in ``directory`` declares: the target, then,
        for build output the repository does not contain, the source it is built from. That is the
        path under ``rootDir`` when the package's tsconfig puts the target in its ``outDir`` or
        ``declarationDir``; else the target with its leading folders dropped one at a time, under the
        package's ``src`` folder and the package."""
        declared = _join(directory, target)
        stem = _SCRIPT_EXTENSION.sub("", declared)
        parts = _SCRIPT_EXTENSION.sub("", normalised(target)).split("/")
        mapped = [
            _join(source, stem[len(output) + 1 :])
            for output, source in self._layout(directory)
            if stem.startswith(f"{output}/")
        ]
        mirrored = [
            _join(home, "/".join(parts[skip:]))
            for skip in range(1, len(parts))
            for home in (_join(directory, "src"), directory)
        ]
        return [declared, *mapped, *mirrored]

    def _read(self, folder: str) -> dict | None:
        path = self.root / folder / "package.json"
        if path.is_symlink() or not path.is_file():
            return None
        try:
            fields = json.loads(path.read_text(errors="replace"))
        except (OSError, ValueError):
            return None
        return fields if isinstance(fields, dict) else None


def package_name(specifier: str) -> tuple[str, str]:
    """``(name, subpath)`` of a bare specifier: ``@scope/pkg/a/b`` is ``("@scope/pkg", "a/b")``."""
    parts = specifier.split("/")
    cut = 2 if specifier.startswith("@") else 1
    return "/".join(parts[:cut]), "/".join(parts[cut:])


def _lookup(table: dict, key: str) -> list[str]:
    chosen = best_pattern(table, key)
    return [target.replace("*", chosen[1]) for target in _pick(table[chosen[0]])] if chosen else []


def _pick(value: object) -> list[str]:
    """Every target of an ``exports`` or ``imports`` entry, best first: a string, a fallback list in
    order, or its conditions in ``CONDITIONS`` order and then the rest; nested."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [target for item in value for target in _pick(item)]
    if isinstance(value, dict):
        ranked = [c for c in CONDITIONS if c in value] + [c for c in value if c not in CONDITIONS]
        return [target for condition in ranked for target in _pick(value[condition])]
    return []


def _closeness(folder: str, path: str) -> tuple[bool, int]:
    """Whether ``path`` lies in ``folder``, and how many leading folders the two share."""
    mine, theirs = PurePosixPath(folder).parts, PurePosixPath(path).parent.parts
    shared = 0
    while shared < min(len(mine), len(theirs)) and mine[shared] == theirs[shared]:
        shared += 1
    return shared == len(mine), shared


def _folder(path: str) -> str:
    return "" if path == "." else path


def _join(base: str, target: str) -> str:
    return normalised(f"{base}/{target}" if base else target)
