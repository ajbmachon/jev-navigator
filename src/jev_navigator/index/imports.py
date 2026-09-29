"""Import statements, including ones that span several lines, resolved to files inside the index's
scope.

TypeScript and JavaScript specifiers resolve the way TypeScript resolves them, from what the
repository declares: relative paths with TypeScript's suffix rules (``./x.js`` names ``x.ts``), the
path aliases of the nearest tsconfig/jsconfig files, the nearest package.json ``imports`` map
(``#src/*``), and the repository's own packages by name through their ``exports`` map or entry
fields. A declared target in build output that is not in the scope (``dist/index.js``) resolves to
the source it is built from. A specifier none of these explain stays unresolved."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from .packages import Packages, package_name
from .tsconfig import ScriptPaths, normalised

_PYTHON_FROM = re.compile(r"^[ \t]*from\s+(\.*[\w.]*)\s+import\s+(\([^)]*\)|[^\n]*)", re.M)
_PYTHON_IMPORT = re.compile(r"^[ \t]*import\s+([\w.]+)", re.M)
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
# `#x` may map to another package, which may map on again; a longer chain is left unresolved.
_MAX_REDIRECTS = 4
_PYTHON_ROOTS = ("", "src/")


def imported_modules(source: str, path: str) -> list[str]:
    """The module specifiers a file imports, in source order, each once."""
    if path.endswith(".py"):
        found = [(match.start(), match.group(1)) for match in _PYTHON_FROM.finditer(source)]
        found += [(match.start(), match.group(1)) for match in _PYTHON_IMPORT.finditer(source)]
    else:
        code = _without_script_comments(source)
        found = [(match.start(), match.group(3)) for match in _SCRIPT_FROM.finditer(code)]
        found += [(match.start(), match.group(1)) for match in _SCRIPT_BARE.finditer(code)]
    return list(dict.fromkeys(specifier for _, specifier in sorted(found)))


def resolve_import(
    specifier: str,
    importer: str,
    scope: frozenset[str],
    script_paths: ScriptPaths | None = None,
    packages: Packages | None = None,
) -> str | None:
    """The scope file a specifier names, or None for packages and files outside the scope.
    ``script_paths`` are the importer's tsconfig aliases and ``packages`` the repository's
    package.json files, both used for non-relative script specifiers."""
    if importer.endswith(".py"):
        return _resolve_python(specifier, importer, scope)
    return _resolve_script(specifier.split("?", 1)[0], importer, scope, script_paths, packages)


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
    redirects: int = 0,
) -> str | None:
    """TypeScript's order: a relative path; else the tsconfig aliases, then a ``#`` import of the
    importer's package or a repository package by name."""
    if specifier.startswith("."):
        return _scope_file([normalised(f"{PurePosixPath(importer).parent}/{specifier}")], scope, packages)
    found = _scope_file(script_paths.candidates(specifier) if script_paths else [], scope, packages)
    if found is not None or packages is None or redirects >= _MAX_REDIRECTS:
        return found
    if specifier.startswith("#"):
        manifest = packages.scope_of(importer)
        targets = manifest.import_targets(specifier) if manifest else []
    else:
        name, subpath = package_name(specifier)
        manifest = packages.named(name, importer)
        targets = manifest.export_targets(subpath) if manifest else []
    for target in targets:
        if not target.startswith("."):
            found = _resolve_script(target, importer, scope, script_paths, packages, redirects + 1)
        else:
            base = normalised(f"{manifest.directory}/{target}" if manifest.directory else target)
            found = _scope_file([base], scope, packages) or _scope_file(
                packages.sources_of(manifest, target), scope, packages
            )
        if found is not None:
            return found
    return None


def _scope_file(bases: list[str], scope: frozenset[str], packages: Packages | None) -> str | None:
    """The first scope file a base names under TypeScript's suffix rules, or through the entry
    fields of a package.json in a base that is a folder."""
    for base in bases:
        found = next((file for file in _script_files(base) if file in scope), None)
        if found is None and packages is not None and (manifest := packages.at(base)) is not None:
            entries = [normalised(f"{base}/{entry}") for entry in manifest.entries]
            found = next((file for entry in entries for file in _script_files(entry) if file in scope), None)
        if found is not None:
            return found
    return None


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
_SCRIPT_BRACES = re.compile(r"\{([^}]*)\}")


def imported_names(source: str, path: str) -> dict[str, str]:
    """Local name to module specifier, for names imported by name (``from m import a as b``, also
    parenthesised over several lines; ``import { a as b } from "m"`` and ``import a from "m"``, also
    over several lines). Type-only names are included; namespace imports are not."""
    if path.endswith(".py"):
        return {
            _local(part): match.group(1)
            for match in _PYTHON_FROM.finditer(source)
            for part in _PYTHON_COMMENT.sub("", match.group(2)).strip("()\n ").split(",")
            if part.strip() and part.strip() != "*"
        }
    names: dict[str, str] = {}
    for match in _SCRIPT_FROM.finditer(_without_script_comments(source)):
        keyword, clause, specifier = match.groups()
        if keyword != "import":
            continue
        default = _SCRIPT_DEFAULT_NAME.match(clause)
        if default:
            names[default.group(1)] = specifier
        for braces in _SCRIPT_BRACES.findall(clause):
            names.update({_local(part): specifier for part in braces.split(",") if part.strip()})
    return names


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


def _local(part: str) -> str:
    return part.split(" as ")[-1].strip().removeprefix("type ").strip()
