"""package.json facts for script imports: each package's ``imports`` map (``#src/*``), and the
repository's own packages by name, with their ``exports`` map and entry fields.

Manifests are read from disk under the index root, in the folders that hold scope files, so a package
whose files are in scope is found wherever its package.json sits and whatever tool manages the
workspace. A manifest that is a symbolic link is not read, and nothing above the root is.

When several manifests claim a name (templates, fixtures, nested workspaces), a package importing its
own name gets itself, as in Node; otherwise the claimant sharing the longest path with the importer
wins, since a workspace links its own members. A tie resolves to neither: the import then stays
unresolved rather than guessed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .tsconfig import build_layouts, normalised

MANIFEST_NAME = "package.json"
# Conditions in the order a code navigator wants them: an explicit "source", then "types", whose
# declarations mirror the source file by file (TypeScript reads it first too), then the runtime
# conditions, which may point every subpath at one bundle.
CONDITIONS = ("source", "types", "import", "module", "development", "browser", "node", "default", "require")
ENTRY_FIELDS = ("source", "types", "typings", "module", "main")
# Folders that hold build output. A declared target inside one that is not in the scope maps back to
# the source it is built from.
BUILD_FOLDERS = frozenset({"dist", "build", "out", "lib", "esm", "cjs", "es", "types", "dts"})
_DECLARATION = (".d.ts", ".d.mts", ".d.cts")


@dataclass(frozen=True)
class Manifest:
    """One package.json: ``directory`` is root-relative; targets come back relative to it."""

    directory: str
    name: str | None
    imports: dict
    exports: object
    entries: tuple[str, ...]

    def import_targets(self, specifier: str) -> list[str]:
        """Targets of a ``#`` specifier, best first; a target without ``./`` names a package."""
        return _lookup(self.imports, specifier)

    def export_targets(self, subpath: str) -> list[str]:
        """Targets for ``name`` (``subpath`` empty) or ``name/subpath``, best first. Without an
        ``exports`` map, the entry fields for the package itself, else the subpath inside it."""
        key = f"./{subpath}" if subpath else "."
        exports = self.exports
        if isinstance(exports, dict) and any(str(k).startswith(".") for k in exports):
            return _lookup(exports, key)
        if exports is not None:
            return _pick(exports) if not subpath else []
        return [f"./{subpath}"] if subpath else [*self.entries, "./index"]


class Packages:
    def __init__(self, root: Path, files: Iterable[str]) -> None:
        self.root = Path(root)
        folders = {""}
        for file in files:
            folders.update(_folder(str(parent)) for parent in PurePosixPath(file).parents)
        self._manifests = {folder: m for folder in sorted(folders) if (m := self._read(folder))}
        claimed: dict[str, list[str]] = defaultdict(list)
        for folder, manifest in self._manifests.items():
            if manifest.name:
                claimed[manifest.name].append(folder)
        self._claimants = {name: tuple(folders) for name, folders in claimed.items()}

    def named(self, name: str, near: str) -> Manifest | None:
        """The repository's package called ``name`` as seen from the file ``near``, or None when no
        manifest claims it or the closest claimants are tied."""
        own = self.scope_of(near)
        if own is not None and own.name == name:
            return own
        folders = self._claimants.get(name, ())
        if len(folders) == 1:
            return self._manifests[folders[0]]
        shared = {folder: _shared_depth(folder, near) for folder in folders}
        closest = [folder for folder, depth in shared.items() if depth == max(shared.values(), default=0)]
        return self._manifests[closest[0]] if len(closest) == 1 else None

    def scope_of(self, file: str) -> Manifest | None:
        """The nearest package.json above ``file``: the one Node reads its ``#`` imports from."""
        for parent in PurePosixPath(file).parents:
            if manifest := self._manifests.get(_folder(str(parent))):
                return manifest
        return None

    def at(self, folder: str) -> Manifest | None:
        return self._manifests.get(folder)

    def config_file(self, specifier: str, near: str) -> str | None:
        """The root-relative config a package ``extends`` names (``@repo/tsconfig/base.json``, or
        the package's own tsconfig.json) from the config ``near``, or None when no package of the
        repository has it."""
        name, subpath = package_name(specifier)
        manifest = self.named(name, near)
        if manifest is None:
            return None
        declared = manifest.export_targets(subpath) if subpath and isinstance(manifest.exports, dict) else []
        for target in [*declared, f"./{subpath}" if subpath else "./tsconfig.json"]:
            base = _join(manifest.directory, target)
            for candidate in (base, f"{base}.json"):
                if (self.root / candidate).is_file():
                    return candidate
        return None

    def sources_of(self, manifest: Manifest, target: str) -> list[str]:
        """Root-relative bases a build-output target is built from, most specific first: the
        package tsconfig's outDir or declarationDir mapped onto its rootDir, then the target with its
        build folders stripped, under the package's src/, the package, and each folder above it (a
        rootDir above the package mirrors the path from there: dist/types/pkg/src/x.d.ts)."""
        relative = normalised(target)
        for suffix in _DECLARATION:
            if relative.endswith(suffix):
                relative = relative[: -len(suffix)]
        full = _join(manifest.directory, relative)
        bases = []
        for out_dir, root_dir in build_layouts(self.root, manifest.directory, self.config_file):
            if root_dir is not None and full.startswith(f"{out_dir}/"):
                bases.append(_join(root_dir, full[len(out_dir) + 1 :]))
        parts = relative.split("/")
        first = next((i for i, part in enumerate(parts) if part in BUILD_FOLDERS), None)
        if first is None:
            return bases
        above = [_folder(str(p)) for p in PurePosixPath(manifest.directory or ".").parents]
        homes = [_join(manifest.directory, "src"), manifest.directory, *above]
        for skip in range(1, 4):
            rest = "/".join(parts[first + skip :])
            if not rest:
                break
            bases += [_join(home, rest) for home in homes]
        return bases

    def _read(self, folder: str) -> Manifest | None:
        path = self.root / folder / MANIFEST_NAME
        if path.is_symlink() or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(errors="replace"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        name = data.get("name")
        imports = data.get("imports")
        entries = tuple(data[f] for f in ENTRY_FIELDS if isinstance(data.get(f), str))
        return Manifest(
            folder,
            name if isinstance(name, str) else None,
            imports if isinstance(imports, dict) else {},
            data.get("exports"),
            entries,
        )


def package_name(specifier: str) -> tuple[str, str]:
    """``(name, subpath)`` of a bare specifier: ``@scope/pkg/a/b`` is ``("@scope/pkg", "a/b")``."""
    parts = specifier.split("/")
    cut = 2 if specifier.startswith("@") else 1
    return "/".join(parts[:cut]), "/".join(parts[cut:])


def _pick(value: object) -> list[str]:
    """Every target of an exports or imports entry, best first: a string, a fallback list in order,
    or its conditions in ``CONDITIONS`` order and then the rest in written order; nested."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [target for item in value for target in _pick(item)]
    if isinstance(value, dict):
        ranked = [c for c in CONDITIONS if c in value] + [c for c in value if c not in CONDITIONS]
        return [target for condition in ranked for target in _pick(value[condition])]
    return []


def _lookup(table: dict, key: str) -> list[str]:
    """Node's rule: the exact key, else the ``*`` pattern with the longest prefix before the star."""
    if key in table and "*" not in key:
        return _pick(table[key])
    matching = [
        (pattern, star) for pattern in table if "*" in pattern and (star := _star(pattern, key)) is not None
    ]
    if not matching:
        return []
    pattern, star = max(matching, key=lambda item: (len(item[0].split("*", 1)[0]), len(item[0])))
    return [target.replace("*", star) for target in _pick(table[pattern])]


def _star(pattern: str, key: str) -> str | None:
    """What the pattern's ``*`` stands for in ``key`` (at least one character), or None."""
    head, tail = pattern.split("*", 1)
    if len(key) > len(head) + len(tail) and key.startswith(head) and key.endswith(tail):
        return key[len(head) : len(key) - len(tail)]
    return None


def _shared_depth(folder: str, file: str) -> int:
    """How many leading folders ``folder`` and the folder holding ``file`` have in common."""
    depth = 0
    for mine, theirs in zip(PurePosixPath(folder).parts, PurePosixPath(file).parent.parts, strict=False):
        if mine != theirs:
            break
        depth += 1
    return depth


def _folder(path: str) -> str:
    return "" if path == "." else path


def _join(base: str, target: str) -> str:
    return normalised(f"{base}/{target}" if base else target)
