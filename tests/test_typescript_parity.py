"""Opt-in: script imports resolve to the files TypeScript's own resolver picks.

Set ``JVN_TYPESCRIPT`` to an installed ``typescript`` package folder (``.../node_modules/typescript``)
and have ``node`` on PATH; otherwise these tests are skipped. The repository's packages are linked into
node_modules the way an install links them, so TypeScript finds them by name. Build output that is not
checked in is left out: TypeScript cannot resolve it without a build, and ``test_imports`` covers it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from git_repos import write_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.imports import imported_modules

TYPESCRIPT = os.environ.get("JVN_TYPESCRIPT")
pytestmark = pytest.mark.skipif(
    not TYPESCRIPT or shutil.which("node") is None,
    reason="set JVN_TYPESCRIPT to a typescript package folder and put node on PATH",
)

# Resolves each [importer, specifier] under the importer's nearest tsconfig.json (TypeScript follows
# its extends), and prints the root-relative file, "<external>", or null.
ORACLE = r"""
const [tsPath, root, pairsJson] = process.argv.slice(1);
const ts = require(tsPath);
const fs = require("fs");
const path = require("path");
const fallback = { moduleResolution: ts.ModuleResolutionKind.Bundler, module: ts.ModuleKind.ESNext };
const host = { ...ts.sys, onUnRecoverableConfigFileDiagnostic: () => {} };
function optionsFor(file) {
  for (let dir = path.dirname(path.join(root, file)); dir.startsWith(root); dir = path.dirname(dir)) {
    const config = path.join(dir, "tsconfig.json");
    if (fs.existsSync(config)) return ts.getParsedCommandLineOfConfigFile(config, {}, host).options;
    if (dir === root) break;
  }
  return fallback;
}
const answers = JSON.parse(pairsJson).map(([file, specifier]) => {
  const options = { ...optionsFor(file), allowJs: true };
  const found = ts.resolveModuleName(specifier, path.join(root, file), options, ts.sys)
    .resolvedModule?.resolvedFileName;
  if (!found) return null;
  const relative = path.relative(root, fs.realpathSync(found)).split(path.sep).join("/");
  return relative.startsWith("..") || relative.includes("node_modules/") ? "<external>" : relative;
});
process.stdout.write(JSON.stringify(answers));
"""

MONOREPO = {
    "packages/tsconfig/package.json": '{"name": "@acme/tsconfig"}',
    "packages/tsconfig/base.json": '{"compilerOptions": {"module": "esnext", "moduleResolution": "bundler"}}',
    "packages/shared/package.json": (
        '{"name": "@acme/shared", "exports": {".": {"types": "./src/index.ts", "default": "./dist/index.js"},'
        ' "./*": {"types": "./src/*.ts", "default": "./dist/*.js"}}}'
    ),
    "packages/shared/src/index.ts": 'export { cents } from "./money.js";\n',
    "packages/shared/src/money.ts": "export const cents = (value: number) => value * 100;\n",
    "apps/web/package.json": '{"name": "web", "imports": {"#lib/*": ["./src/lib/*.tsx", "./src/lib/*.ts"]}}',
    "apps/web/tsconfig.json": (
        '{"extends": "@acme/tsconfig/base.json",'
        ' "compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["./src/*"]}}}'
    ),
    "apps/web/src/lib/format.ts": "export const format = (value: number) => `${value}`;\n",
    "apps/web/src/lib/view.tsx": "export const View = 1;\n",
    "apps/web/src/page.ts": (
        'import { format } from "./lib/format.js";\n'
        'import { View } from "#lib/view";\n'
        'import { format as again } from "#lib/format";\n'
        'import { format as aliased } from "@/lib/format";\n'
        'import { cents } from "@acme/shared";\n'
        'import { cents as money } from "@acme/shared/money";\n'
    ),
    "scripts/util.mts": "export const util = 1;\n",
    "scripts/job.mts": 'import { util } from "./util.mjs";\n',
}


def test_every_import_resolves_to_the_file_typescript_picks(tmp_path: Path) -> None:
    # Arrange
    write_files(tmp_path, MONOREPO)
    for name, folder in {"@acme/shared": "packages/shared", "@acme/tsconfig": "packages/tsconfig"}.items():
        link = tmp_path / "node_modules" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(tmp_path / folder, target_is_directory=True)
    index = CodeIndex(tmp_path, list(MONOREPO))
    importers = [file for file in MONOREPO if not file.endswith(".json")]
    pairs = [(file, spec) for file in importers for spec in imported_modules(MONOREPO[file], file)]

    # Act
    answers = json.loads(
        subprocess.run(
            [
                "node",
                "-e",
                ORACLE,
                str(Path(TYPESCRIPT or "").resolve()),
                str(tmp_path.resolve()),
                json.dumps(pairs),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    typescript = {
        file: tuple(dict.fromkeys(a for (f, _), a in zip(pairs, answers, strict=True) if f == file))
        for file in importers
    }

    # Assert
    assert all(answer in index.files for answer in answers), answers  # every import is TypeScript-resolvable
    assert {file: index.imports(file) for file in importers} == typescript
