"""Import statements, including ones that span several lines, resolved to files inside the index's
scope.

Script specifiers resolve in TypeScript's order from what the repository declares: a relative path,
the nearest config's path aliases, then the ``imports`` map of the importer's package.json for a
``#`` specifier or a repository package by name. A declared target the repository does not contain
is build output and resolves to the source it is built from."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from .packages import Packages, package_name
from .tsconfig import ScriptPaths, normalised

_PYTHON_FROM = re.compile(r"^[ \t]*from\s+(\.*[\w.]*)\s+import\s+(\([^)]*\)|[^\n]*)", re.M)
_PYTHON_IMPORT = re.compile(
    r"^[ \t]*import[ \t]+([\w.]+(?:[ \t]+as[ \t]+\w+)?(?:[ \t]*,[ \t]*[\w.]+(?:[ \t]+as[ \t]+\w+)?)*)", re.M
)
_PYTHON_IMPORTED_MODULE = re.compile(r"([\w.]+)(?:[ \t]+as[ \t]+(\w+))?")
_SCRIPT_FROM = re.compile(
    r"""^[ \t]*(import|export)\s+(?:type\s+)?"""
    r"""((?:(?!\n[ \t]*(?:import|export)\b)[\w$*\s{},])*?)\s*from\s*['"]([^'"]+)['"]""",
    re.M,
)
_SCRIPT_COMMENT_OR_STRING = re.compile(
    r""""(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|//[^\n]*|/\*.*?\*/""", re.S
)
_SCRIPT_BARE = re.compile(r"""(?:\brequire\(\s*|\bimport\s*\(\s*|^[ \t]*import\s+)['"]([^'"]+)['"]""", re.M)
_SCRIPT_SUFFIXES = (".ts", ".tsx", ".d.ts", ".js", ".mjs", ".cjs", ".jsx")
# ESM TypeScript imports a module by the name it compiles to, so `./x.js` names `x.ts` when it exists.
_SOURCES_OF_OUTPUT = {
    ".js": (".ts", ".tsx", ".d.ts"),
    ".jsx": (".tsx",),
    ".mjs": (".mts", ".d.mts"),
    ".cjs": (".cts", ".d.cts"),
}
_PYTHON_ROOTS = ("", "src/")


@dataclass(frozen=True)
class ImportedName:
    """A name imported by name: the module specifier, and the name the module exports it under
    (``stop`` for ``import { stop as halt }``). None for a default import, which names no export."""

    specifier: str
    exported: str | None


@dataclass(frozen=True)
class ImportFact:
    """A discoverable repository path and whether its import mapping proves that path."""

    path: str
    proven: bool
    reason: str


def imported_modules(source: str, path: str) -> list[str]:
    """The module specifiers a file imports, in source order, each once."""
    if path.endswith(".py"):
        found = [(match.start(), match.group(1)) for match in _PYTHON_FROM.finditer(source)]
        found += [(position, module) for position, module, _ in _python_imports(source)]
    else:
        code = _without_script_comments(source)
        found = [(match.start(), match.group(3)) for match in _SCRIPT_FROM.finditer(code)]
        found += [(match.start(), match.group(1)) for match in _SCRIPT_BARE.finditer(code)]
    return list(dict.fromkeys(specifier for _, specifier in sorted(found)))


def module_imports(source: str, path: str) -> tuple[tuple[str, frozenset[str] | None], ...]:
    """Each module specifier the source imports, re-exports or requires, in source order, with the
    names it takes by name as that module exports them; ``None`` when it takes the whole module (a
    namespace or default import, ``export *``, ``require``, a dynamic or bare import, ``import m``)."""
    if path.endswith(".py"):
        found = [
            (match.start(), match.group(1), _python_names(match.group(2)))
            for match in _PYTHON_FROM.finditer(source)
        ]
        found += [(position, module, None) for position, module, _ in _python_imports(source)]
    else:
        code = _without_script_comments(source)
        found = [
            (match.start(), match.group(3), _script_names(match.group(1), match.group(2)))
            for match in _SCRIPT_FROM.finditer(code)
        ]
        found += [(match.start(), match.group(1), None) for match in _SCRIPT_BARE.finditer(code)]
    taken: dict[str, frozenset[str] | None] = {}
    for _, specifier, names in sorted(found, key=lambda entry: entry[0]):
        if names == frozenset():
            continue
        before = taken.get(specifier, frozenset())
        taken[specifier] = None if before is None or names is None else before | names
    return tuple(taken.items())


def _python_imports(source: str) -> list[tuple[int, str, str | None]]:
    """Each module an ``import`` statement names, with its position and its ``as`` name, if any:
    ``import json, app.billing as billing`` imports ``json`` and ``app.billing``."""
    return [
        (statement.start(1) + module.start(), module.group(1), module.group(2))
        for statement in _PYTHON_IMPORT.finditer(source)
        for module in _PYTHON_IMPORTED_MODULE.finditer(statement.group(1))
    ]


def _python_names(clause: str) -> frozenset[str] | None:
    parts = [
        part.strip() for part in _PYTHON_COMMENT.sub("", clause).strip("()\n ").split(",") if part.strip()
    ]
    if "*" in parts:
        return None
    return frozenset(_exported(part) for part in parts)


def _script_names(keyword: str, clause: str) -> frozenset[str] | None:
    """The names a script import or re-export takes, as its module exports them."""
    if "*" in clause or (keyword == "import" and _SCRIPT_DEFAULT_NAME.match(clause)):
        return None
    names = frozenset(
        _exported(part)
        for braces in _SCRIPT_BRACES.findall(clause)
        for part in braces.split(",")
        if part.strip()
    )
    return None if "default" in names else names


def resolve_import(
    specifier: str,
    importer: str,
    scope: frozenset[str],
    script_paths: ScriptPaths | None = None,
    packages: Packages | None = None,
) -> ImportFact | None:
    """The scope file a specifier suggests, with its evidence, or None when none is in scope.
    ``script_paths`` are the importer's config aliases and ``packages`` the repository's
    package.json files, both used for non-relative script specifiers."""
    if importer.endswith(".py"):
        path = _resolve_python(specifier, importer, scope)
        return ImportFact(path, True, "Python import") if path else None
    return _resolve_script(specifier, importer, scope, script_paths, packages)


def _resolve_python(specifier: str, importer: str, scope: frozenset[str]) -> str | None:
    dots = len(specifier) - len(specifier.lstrip("."))
    module_path = specifier.lstrip(".").replace(".", "/")
    if dots:
        base = PurePosixPath(importer).parents[dots - 1]
        roots = [f"{base}/" if str(base) != "." else ""]
    else:
        roots = list(_PYTHON_ROOTS)
    for root in roots:
        for candidate in (f"{root}{module_path}.py", f"{root}{module_path}/__init__.py"):
            if candidate in scope:
                return candidate
    return None


def _resolve_script(
    specifier: str,
    importer: str,
    scope: frozenset[str],
    script_paths: ScriptPaths | None,
    packages: Packages | None,
    seen: frozenset[str] = frozenset(),
) -> ImportFact | None:
    if specifier.startswith("."):
        path = _scope_file([normalised(f"{PurePosixPath(importer).parent}/{specifier}")], scope)
        return ImportFact(path, True, "relative import") if path else None
    found = _scope_file(script_paths.candidates(specifier) if script_paths else [], scope)
    if found is not None:
        return ImportFact(found, True, "script config path mapping")
    if packages is None or specifier in seen:
        return None
    seen = seen | {specifier}
    if specifier.startswith("#"):
        manifest = packages.scope_of(importer)
        targets = manifest.import_targets(specifier) if manifest else []
        certain = manifest is not None and packages.nearest_on_disk(importer, manifest)
    else:
        name, subpath = package_name(specifier)
        manifest = packages.named(name, importer)
        targets = manifest.export_targets(subpath) if manifest else []
        certain = packages.links(name, importer)
    if manifest is None:
        return None
    mapping = f"repository package.json mapping for {specifier}"
    uncertain = "" if certain else ", a package the importer does not link by workspace: or self-reference"
    # Discovery keeps main's order: each target, then the source it is built from.
    chosen: ImportFact | None = None
    for target in targets:
        if target.startswith("."):
            path = _scope_file(packages.target_bases(manifest.directory, target), scope)
            if path is not None:
                chosen = ImportFact(path, False, mapping)
                break
            continue
        redirected = _resolve_script(target, importer, scope, script_paths, packages, seen)
        if redirected is not None:
            chosen = ImportFact(redirected.path, redirected.proven and not target.startswith("#"), mapping)
            break
    if chosen is None:
        return None
    # Proof: every declared target that exists names the same file, and it is not a declaration file.
    declared = set()
    for target in targets:
        if target.startswith("."):
            if (
                path := _scope_file(packages.target_bases(manifest.directory, target)[:1], scope)
            ) is not None:
                declared.add(path)
        elif (
            redirected := _resolve_script(target, importer, scope, script_paths, packages, seen)
        ) is not None:
            declared.add(
                redirected.path if redirected.proven and not target.startswith("#") else "<unproven>"
            )
    agreed = declared == {chosen.path}
    proven = certain and agreed and not chosen.path.endswith((".d.ts", ".d.mts", ".d.cts"))
    if proven:
        return ImportFact(chosen.path, True, mapping)
    why = uncertain or (
        ", declared targets differ by condition"
        if len(declared) > 1
        else ", a declaration file"
        if agreed
        else ", source inferred from build output"
    )
    return ImportFact(chosen.path, False, mapping + why)


def _scope_file(bases: list[str], scope: frozenset[str]) -> str | None:
    """The first scope file a base names under TypeScript's suffix rules."""
    return next((file for base in bases for file in _script_files(base) if file in scope), None)


def _script_files(base: str) -> list[str]:
    extension = PurePosixPath(base).suffix
    stem = base[: -len(extension)] if extension else base
    return [
        *(stem + source for source in _SOURCES_OF_OUTPUT.get(extension, ())),
        base,
        *(base + suffix for suffix in _SCRIPT_SUFFIXES),
        *(f"{base}/index{suffix}" for suffix in _SCRIPT_SUFFIXES),
    ]


_PYTHON_COMMENT = re.compile(r"#[^\n]*")
_SCRIPT_DEFAULT_NAME = re.compile(r"^\s*([\w$]+)\s*(?:,|$)")
# `const { verify, sign: signToken } = require('./jwt')`, which imports `verify` and `signToken`, but
# not `require('./jwt').verify` or `require('./jwt')(options)`.
_SCRIPT_REQUIRED_NAMES = re.compile(
    r"""\b(?:const|let|var)\s*\{([^}]*)\}\s*=\s*require\(\s*['"]([^'"]+)['"]\s*\)(?!\s*[.(\[])"""
)
_SCRIPT_BRACES = re.compile(r"\{([^}]*)\}")


def imported_names(source: str, path: str) -> dict[str, ImportedName]:
    """Local name to what it imports, for names imported by name (``from m import a as b``, also
    parenthesised over several lines; ``import { a as b } from "m"`` and ``import a from "m"``, also
    over several lines; ``const { a, b: c } = require("m")``). Type-only names are included;
    namespace imports are not."""
    if path.endswith(".py"):
        return {
            _local(part): ImportedName(match.group(1), _exported(part))
            for match in _PYTHON_FROM.finditer(source)
            for part in _PYTHON_COMMENT.sub("", match.group(2)).strip("()\n ").split(",")
            if part.strip() and part.strip() != "*"
        }
    code = _without_script_comments(source)
    names = {
        local: ImportedName(match.group(2), exported)
        for match in _SCRIPT_REQUIRED_NAMES.finditer(code)
        for exported, local in _destructured_pairs(match.group(1))
    }
    for match in _SCRIPT_FROM.finditer(code):
        keyword, clause, specifier = match.groups()
        if keyword != "import":
            continue
        default = _SCRIPT_DEFAULT_NAME.match(clause)
        if default:
            names[default.group(1)] = ImportedName(specifier, None)
        for braces in _SCRIPT_BRACES.findall(clause):
            names.update(
                {
                    _local(part): _script_imported(specifier, _exported(part))
                    for part in braces.split(",")
                    if part.strip()
                }
            )
    return names


def _script_imported(specifier: str, exported: str) -> ImportedName:
    """``import { default as entry }`` is a default import: it names no export."""
    return ImportedName(specifier, None if exported == "default" else exported)


def reexported_names(source: str, path: str) -> tuple[tuple[frozenset[str] | None, str], ...]:
    """Names re-exported from each script module; ``None`` means an ``export *`` wildcard."""
    if path.endswith(".py"):
        return ()
    exports = []
    for match in _SCRIPT_FROM.finditer(_without_script_comments(source)):
        keyword, clause, specifier = match.groups()
        if keyword != "export":
            continue
        stripped = clause.strip()
        if stripped == "*":
            exports.append((None, specifier))
            continue
        names = frozenset(
            _local(part)
            for braces in _SCRIPT_BRACES.findall(clause)
            for part in braces.split(",")
            if part.strip()
        )
        if names:
            exports.append((names, specifier))
    return tuple(exports)


def _without_script_comments(source: str) -> str:
    """The source with ``//`` and ``/* */`` comments removed; string literals are kept whole, so a
    ``//`` inside a string is not taken for a comment."""
    return _SCRIPT_COMMENT_OR_STRING.sub(_keep_literal, source)


def _keep_literal(match: re.Match) -> str:
    text = match.group(0)
    return "" if text.startswith("/") else text


def _destructured_pairs(pattern: str) -> list[tuple[str, str]]:
    """The property each name of an object pattern takes, and the local name it binds: ``(a, a)`` and
    ``(b, c)`` in ``a, b: c = 1, ...rest``, but not the rest element."""
    pairs = []
    for part in pattern.split(","):
        key, _, local = part.split("=")[0].partition(":")
        key, local = key.strip(), (local or key).strip()
        if key and not key.startswith("..."):
            pairs.append((key, local))
    return pairs


def _local(part: str) -> str:
    return part.split(" as ")[-1].strip().removeprefix("type ").strip()


def _exported(part: str) -> str:
    """The name an import part takes as its module exports it: ``a`` in ``a as b`` or ``type a as b``."""
    return part.strip().removeprefix("type ").split(" as ")[0].strip()
