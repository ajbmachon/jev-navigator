from __future__ import annotations

import json
from pathlib import Path

import pytest
from git_repos import commit_all, write_files

from jev_navigator import operations
from jev_navigator.directives.places import neighbours
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.imports import (
    ImportedName,
    imported_modules,
    imported_names,
    module_imports,
    reexported_names,
)
from jev_navigator.index.spans import Span

ROOT_TSCONFIG = """\
{
  // Aliases for the web app; comments and trailing commas are allowed here.
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@config": ["src/config/index.ts"],
    },
  },
}
"""


def indexed(root: Path, files: dict[str, str]) -> CodeIndex:
    """Writes ``files`` and indexes every one except the JSON configs."""
    write_files(root, files)
    return CodeIndex(root, [name for name in files if not name.endswith(".json")])


def test_an_import_spanning_several_lines_resolves_its_names(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/app/format.ts": "export function formatPrice(cents: number) {\n  return cents / 100;\n}\n",
            "src/app/page.ts": (
                "import {\n  formatPrice,\n  type Price,\n} from './format';\n\n"
                "export function show() {\n  return formatPrice(1);\n}\n"
            ),
        },
    )

    # Act
    imports = index.imports("src/app/page.ts")
    call = index.find_callers("formatPrice")[0]

    # Assert
    assert imports == ("src/app/format.ts",)
    assert call.binding.status == "resolved"
    assert call.binding.target == Span("src/app/format.ts", 1, 3, "formatPrice")


def test_script_reexports_name_the_source_module_and_exported_names() -> None:
    # Arrange
    source = """\
export * from "./orders";
export { refund, createOrder as placeOrder } from "./commands";
import { ignored } from "./ignored";
"""

    # Act
    exports = reexported_names(source, "src/services/index.ts")

    # Assert
    assert exports == (
        (None, "./orders"),
        (frozenset({"refund", "placeOrder"}), "./commands"),
    )
    assert reexported_names("from .orders import create_order", "app/__init__.py") == ()


def test_destructuring_a_require_imports_each_local_name_under_its_exported_name() -> None:
    # Arrange
    source = """\
const { verify, sign: signToken, decode = fallback, ...rest } = require('./jwt');
const { app } = require("./app")(options);
"""

    # Act
    names = imported_names(source, "src/main.js")

    # Assert
    assert names == {
        "verify": ImportedName("./jwt", "verify"),
        "signToken": ImportedName("./jwt", "sign"),
        "decode": ImportedName("./jwt", "decode"),
    }


def test_the_export_surface_is_the_ast_grep_statement_nodes(tmp_path: Path) -> None:
    """The surface names real statements only, and only the module's own definitions: a private
    definition, a default export, a re-export and a template-literal body contribute nothing. An
    export list entry under another name and the default export record the definition each
    exports."""
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/service.ts": (
                "function privateRun() {}\n"
                "export function run() {}\n"
                "export const READY = true;\n"
                "const local = true;\n"
                "export { local as publicLocal };\n"
                "export default function defaultRun() {}\n"
                'export * from "./one";\n'
                "export { refund, createOrder as placeOrder } from './commands';\n"
                "const tpl = `export function inTemplate() {}`;\n"
            ),
        },
    )

    # Act
    facts = index._facts_in("src/service.ts")

    # Assert
    assert facts.export_names == ("READY", "publicLocal", "run")
    assert facts.renamed_exports == (("default", "defaultRun"), ("publicLocal", "local"))


def test_a_destructured_export_is_part_of_the_export_surface(tmp_path: Path) -> None:
    """`export const { verify, sign: signToken } = jwt` exports `verify` and `signToken`, so an import
    through a barrel that re-exports the module finds `verify`. A property key, a default value and a
    computed key export nothing."""
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/jwt/tools.ts": (
                "const jwt = make();\n"
                "export const { verify, sign: signToken, decode = fallback, [key]: other } = jwt;\n"
            ),
            "src/jwt/index.ts": 'export * from "./tools";\n',
            "src/page.ts": (
                'import { verify } from "./jwt";\nexport function check() {\n  return verify();\n}\n'
            ),
        },
    )

    # Act
    names = index._facts_in("src/jwt/tools.ts").export_names
    binding = index.find_callers("verify")[0].binding

    # Assert
    assert names == ("decode", "other", "signToken", "verify")
    assert (binding.status, binding.target) == ("resolved", Span("src/jwt/tools.ts", 2, 2, "verify"))


def test_a_template_literal_body_is_not_part_of_the_export_surface(tmp_path: Path) -> None:
    """The regression: a template literal whose text looks like export statements names nothing,
    so the privately defined `run` behind the barrel stays a candidate."""
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/one.ts": (
                "const tpl = `export function run() {}`;\nfunction run() { return 1; }\n"
            ),
            "src/services/index.ts": 'export * from "./one";\n',
            "src/page.ts": 'import { run } from "./services";\nrun();\n',
        },
    )

    # Act
    call = index.find_callers("run")[0]

    # Assert
    assert call.binding.status == "candidate"
    assert call.binding.target is None


def test_a_public_export_beside_a_template_literal_stays_proven(tmp_path: Path) -> None:
    """The positive control: the real `export function` is still the surface, even when the same
    file's template literal repeats its shape."""
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/one.ts": (
                "const tpl = `export function inner() {}`;\nexport function run() { return 1; }\n"
            ),
            "src/services/index.ts": 'export * from "./one";\n',
            "src/page.ts": 'import { run } from "./services";\nrun();\n',
        },
    )

    # Act
    call = index.find_callers("run")[0]

    # Assert
    assert call.binding.status == "resolved"
    assert call.binding.target == Span("src/services/one.ts", 2, 2, "run")


def test_a_function_inside_an_exported_arrow_does_not_name_the_export(tmp_path: Path) -> None:
    """The regression: the export's own name is the surface, not the first function in its body,
    so a call to `Page` through the barrel is proven to reach it."""
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/views/page.ts": (
                "export const Page = () => {\n  function helper() { return 1; }\n  return helper();\n};\n"
            ),
            "src/views/index.ts": 'export * from "./page";\n',
            "src/app.ts": 'import { Page } from "./views";\nPage();\n',
        },
    )

    # Act
    call = index.find_callers("Page")[0]

    # Assert
    assert call.binding.status == "resolved"
    assert call.binding.target == Span("src/views/page.ts", 1, 4, "Page")


def test_a_call_imported_through_a_barrel_has_a_proven_target(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/orders.ts": "export function createOrder() { return 1; }\n",
            "src/services/index.ts": 'export * from "./orders";\n',
            "src/routes.ts": (
                'import { createOrder } from "./services";\n'
                "export function postOrder() { return createOrder(); }\n"
            ),
        },
    )

    # Act
    call = index.find_callers("createOrder")[0]

    # Assert
    assert call.binding.status == "resolved"
    assert call.binding.target == Span("src/services/orders.ts", 1, 1, "createOrder")


def test_two_wildcard_reexports_with_the_same_name_stay_ambiguous(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/one.ts": "export function run() { return 1; }\n",
            "src/services/two.ts": "export function run() { return 2; }\n",
            "src/services/index.ts": 'export * from "./one";\nexport * from "./two";\n',
            "src/page.ts": 'import { run } from "./services";\nrun();\n',
        },
    )

    # Act
    call = index.find_callers("run")[0]

    # Assert
    assert call.binding.status == "candidate"
    assert call.binding.target is None


def test_a_private_name_behind_a_wildcard_barrel_is_not_a_proven_import(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/one.ts": "function run() { return 1; }\n",
            "src/services/index.ts": 'export * from "./one";\n',
            "src/page.ts": 'import { run } from "./services";\nrun();\n',
        },
    )

    # Act
    call = index.find_callers("run")[0]

    # Assert
    assert call.binding.status == "candidate"
    assert call.binding.target is None


def test_a_path_alias_from_the_nearest_tsconfig_resolves_to_a_scope_file(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "tsconfig.json": ROOT_TSCONFIG,
            "src/lib/utils.ts": "export function cn(...parts: string[]) {\n  return parts.join(' ');\n}\n",
            "src/config/index.ts": "export const settings = {};\n",
            "src/components/button.ts": (
                'import { cn } from "@/lib/utils";\nimport { settings } from "@config";\n\n'
                "export function button() {\n  return cn('a', 'b');\n}\n"
            ),
        },
    )

    # Act
    imports = index.imports("src/components/button.ts")
    call = index.find_callers("cn")[0]

    # Assert
    assert imports == ("src/lib/utils.ts", "src/config/index.ts")
    assert call.binding.status == "resolved"


def test_the_nearest_config_wins_and_extends_inherits_paths(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "tsconfig.json": ROOT_TSCONFIG,
            "src/lib/utils.ts": "export const shared = 1;\n",
            "packages/web/tsconfig.json": '{"compilerOptions": {"paths": {"@/*": ["./app/*"]}}}',
            "packages/web/app/util.ts": "export const local = 1;\n",
            "packages/web/app/page.ts": 'import { local } from "@/util";\n',
            "packages/api/tsconfig.json": '{"extends": "../../tsconfig.json"}',
            "packages/api/handler.ts": 'import { shared } from "@/lib/utils";\n',
            "packages/docs/jsconfig.json": '{"compilerOptions": {"paths": {"@/*": ["./lib/*"]}}}',
            "packages/docs/lib/util.js": "export const local = 1;\n",
            "packages/docs/page.js": 'import { local } from "@/util";\n',
        },
    )

    # Act
    web = index.imports("packages/web/app/page.ts")
    api = index.imports("packages/api/handler.ts")
    docs = index.imports("packages/docs/page.js")

    # Assert
    assert web == ("packages/web/app/util.ts",)
    assert api == ("src/lib/utils.ts",)
    assert docs == ("packages/docs/lib/util.js",)


def test_a_parenthesised_python_import_over_several_lines_lists_every_name() -> None:
    # Arrange
    source = "from app.jobs import (\n    send_invoice,  # the monthly run\n    refund as give_back,\n)\n"

    # Act
    names = imported_names(source, "app/routes.py")

    # Assert
    assert names == {
        "send_invoice": ImportedName("app.jobs", "send_invoice"),
        "give_back": ImportedName("app.jobs", "refund"),
    }


def test_a_python_import_of_several_modules_imports_each_in_source_order() -> None:
    # Arrange
    source = "import json, app.billing as billing, app.mail\nfrom app.jobs import run\n"

    # Act
    modules = imported_modules(source, "app/routes.py")
    taken = module_imports(source, "app/routes.py")

    # Assert
    assert modules == ["json", "app.billing", "app.mail", "app.jobs"]
    assert taken == (
        ("json", None),
        ("app.billing", None),
        ("app.mail", None),
        ("app.jobs", frozenset({"run"})),
    )


def test_an_index_at_an_old_commit_still_reads_configs_outside_its_scope(tmp_path: Path) -> None:
    # Arrange
    write_files(
        tmp_path,
        {
            "tsconfig.json": ROOT_TSCONFIG,
            "package.json": '{"imports": {"#lib/*": "./src/lib/*.ts"}}',
            "src/lib/utils.ts": "export const shared = 1;\n",
            "src/lib/money.ts": "export const cents = 1;\n",
            "src/app/page.ts": 'import { shared } from "@/lib/utils";\nimport { cents } from "#lib/money";\n',
        },
    )
    commit_all(tmp_path)

    # Act
    index = CodeIndex.at_commit(tmp_path, "HEAD", prefixes=("src/",))

    # Assert
    assert index.imports("src/app/page.ts") == ("src/lib/utils.ts", "src/lib/money.ts")
    assert "tsconfig.json" not in index.files
    assert "package.json" not in index.files


def test_a_multi_line_import_with_comments_inside_keeps_its_module_and_names() -> None:
    # Arrange
    source = "import {\n  a, // the first\n  /* the second */ b,\n} from './x';\n"

    # Act
    modules = imported_modules(source, "src/p.ts")
    names = imported_names(source, "src/p.ts")

    # Assert
    assert modules == ["./x"]
    assert names == {"a": ImportedName("./x", "a"), "b": ImportedName("./x", "b")}


def test_an_import_after_a_statement_without_semicolon_keeps_its_names() -> None:
    # Arrange
    source = "export default Foo\nimport { a } from './a'\n"

    # Act
    names = imported_names(source, "src/p.ts")

    # Assert
    assert names == {"a": ImportedName("./a", "a")}


def test_a_script_import_keeps_the_name_its_module_exports() -> None:
    """`import { stop as halt }` takes `stop`, also as a type; a default import takes no exported
    name, so there is none to follow."""
    # Arrange
    source = (
        "import main, { stop as halt, type Kind as K, default as entry } from './x';\n"
        "import { start } from './y';\n"
    )

    # Act
    names = imported_names(source, "src/p.ts")

    # Assert
    assert names == {
        "main": ImportedName("./x", None),
        "halt": ImportedName("./x", "stop"),
        "K": ImportedName("./x", "Kind"),
        "entry": ImportedName("./x", None),
        "start": ImportedName("./y", "start"),
    }


def test_the_alias_with_the_longest_prefix_wins_like_typescript(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "tsconfig.json": (
                '{"compilerOptions": {"baseUrl": ".", "paths": '
                '{"@/*": ["src/*"], "@/components/*": ["src/ui/*"]}}}'
            ),
            "src/components/button.ts": "export const wrong = 1;\n",
            "src/ui/button.ts": "export const right = 1;\n",
            "src/page.ts": 'import { right } from "@/components/button";\n',
        },
    )

    # Act
    imports = index.imports("src/page.ts")

    # Assert
    assert imports == ("src/ui/button.ts",)


def test_an_exact_alias_wins_over_a_wildcard_like_typescript(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "tsconfig.json": (
                '{"compilerOptions": {"paths": {"@*": ["./lib/*"], "@config": ["./cfg/main.ts"]}}}'
            ),
            "lib/config.ts": "export const wrong = 1;\n",
            "cfg/main.ts": "export const right = 1;\n",
            "page.ts": 'import { right } from "@config";\n',
        },
    )

    # Act
    imports = index.imports("page.ts")

    # Assert
    assert imports == ("cfg/main.ts",)


def test_a_config_whose_base_lies_above_the_index_root_leaves_aliases_unknown(tmp_path: Path) -> None:
    # Arrange
    write_files(
        tmp_path,
        {
            "tsconfig.base.json": (
                '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["packages/web/src/*"]}}}'
            ),
            "packages/web/tsconfig.json": '{"extends": "../../tsconfig.base.json"}',
            "packages/web/src/util.ts": "export function u() {\n  return 1;\n}\n",
            "packages/web/src/page.ts": (
                'import { u } from "@/util";\n\nexport function p() {\n  return u();\n}\n'
            ),
        },
    )
    index = CodeIndex(tmp_path / "packages/web", ["src/util.ts", "src/page.ts"])

    # Act
    imports = index.imports("src/page.ts")
    call = index.find_callers("u")[0]

    # Assert
    assert imports == ()
    assert call.binding.status == "candidate"


def test_configs_outside_the_root_or_behind_a_symbolic_link_are_not_read(tmp_path: Path) -> None:
    # Arrange
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.json").write_text(
        '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["../repo/src/*"]}}}'
    )
    (outside / "tsconfig.json").write_text(
        '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["../repo/src/*"]}}}'
    )
    root = tmp_path / "repo"
    index = indexed(
        root,
        {
            "tsconfig.json": '{"extends": "../outside/evil.json"}',
            "src/util.ts": "export const u = 1;\n",
            "src/page.ts": 'import { u } from "@/util";\n',
        },
    )
    (root / "linked").mkdir()
    (root / "linked/tsconfig.json").symlink_to(outside / "tsconfig.json")
    (root / "linked/page.ts").write_text('import { u } from "@/util";\n')
    linked = CodeIndex(root, ["src/util.ts", "src/page.ts", "linked/page.ts"])

    # Act
    through_extends = index.imports("src/page.ts")
    through_link = linked.imports("linked/page.ts")

    # Assert
    assert through_extends == ()
    assert through_link == ()


def test_an_esm_specifier_names_the_typescript_source_it_compiles_from(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/format.ts": "export function formatPrice(cents: number) {\n  return cents / 100;\n}\n",
            "src/view.tsx": "export const View = 1;\n",
            "src/job.mts": "export const job = 1;\n",
            "src/legacy.cts": "export const legacy = 1;\n",
            "src/page.ts": (
                'import { formatPrice } from "./format.js";\nimport { View } from "./view.jsx";\n'
                'import { job } from "./job.mjs";\nimport { legacy } from "./legacy.cjs";\n\n'
                "export function show() {\n  return formatPrice(1);\n}\n"
            ),
        },
    )

    # Act
    imports = index.imports("src/page.ts")
    call = index.find_callers("formatPrice")[0]

    # Assert
    assert imports == ("src/format.ts", "src/view.tsx", "src/job.mts", "src/legacy.cts")
    assert call.binding.status == "resolved"
    assert call.binding.target == Span("src/format.ts", 1, 3, "formatPrice")


def test_hash_imports_follow_the_nearest_package_json_through_fallbacks_and_conditions(
    tmp_path: Path,
) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "app/package.json": (
                '{"imports": {"#src/*": ["./src/*.tsx", "./src/*.ts"],'
                ' "#config": {"types": "./src/config.ts", "default": "./dist/config.js"}}}'
            ),
            "app/src/lib/util.ts": "export const util = 1;\n",
            "app/src/config.ts": "export const config = 1;\n",
            "app/src/page.ts": 'import { util } from "#src/lib/util";\nimport { config } from "#config";\n',
            "app/tools/package.json": '{"name": "tools"}',
            "app/tools/run.ts": 'import { util } from "#src/lib/util";\n',
        },
    )

    # Act
    page = index.imports("app/src/page.ts")
    nested = index.imports("app/tools/run.ts")

    # Assert
    assert page == ("app/src/lib/util.ts", "app/src/config.ts")
    assert nested == ()  # Node reads `#` imports from the nearest package.json only


@pytest.mark.parametrize(
    ("package", "specifier", "expected"),
    [
        (
            {
                "packages/shared/package.json": (
                    '{"name": "@acme/shared", "exports": {"./*": {"types": "./dist/types/*.d.ts",'
                    ' "development": "./dist/dev/index.js", "default": "./dist/index.js"}}}'
                ),
            },
            "@acme/shared/format",
            "packages/shared/src/format.ts",
        ),
        (
            {"packages/shared/package.json": '{"name": "@acme/shared", "main": "./bundle/index.mjs"}'},
            "@acme/shared",
            "packages/shared/src/index.ts",
        ),
        (
            {
                "packages/shared/package.json": '{"name": "@acme/shared", "exports": "./build/index.js"}',
                "packages/shared/tsconfig.json": '{"compilerOptions": {"outDir": "build", "rootDir": "lib"}}',
                "packages/shared/lib/index.ts": "export const index = 1;\n",
            },
            "@acme/shared",
            "packages/shared/lib/index.ts",
        ),
    ],
    ids=["exports-types-before-bundle", "main-in-any-build-folder", "outdir-onto-rootdir"],
)
def test_a_repository_package_resolves_by_name_to_its_source(
    tmp_path: Path, package: dict[str, str], specifier: str, expected: str
) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            **package,
            "packages/shared/src/index.ts": "export const index = 1;\n",
            "packages/shared/src/format.ts": "export const format = 1;\n",
            "apps/web/page.ts": f'import {{ index }} from "{specifier}";\n',
        },
    )

    # Act
    imports = index.imports("apps/web/page.ts")

    # Assert
    assert imports == (expected,)


def test_a_name_several_packages_claim_resolves_within_the_importers_workspace(tmp_path: Path) -> None:
    # Arrange
    ui = '{"name": "@ws/ui", "exports": {"./*": "./src/*.ts"}}'
    index = indexed(
        tmp_path,
        {
            "templates/next/packages/ui/package.json": ui,
            "templates/next/packages/ui/src/button.ts": "export const button = 1;\n",
            "templates/next/apps/web/page.ts": 'import { button } from "@ws/ui/button";\n',
            "templates/vite/packages/ui/package.json": ui,
            "templates/vite/packages/ui/src/button.ts": "export const button = 2;\n",
            "templates/vite/apps/web/page.ts": 'import { button } from "@ws/ui/button";\n',
            "scripts/check.ts": 'import { button } from "@ws/ui/button";\n',
        },
    )

    # Act
    next_page = index.imports("templates/next/apps/web/page.ts")
    vite_page = index.imports("templates/vite/apps/web/page.ts")
    outside = index.imports("scripts/check.ts")

    # Assert
    assert next_page == ("templates/next/packages/ui/src/button.ts",)
    assert vite_page == ("templates/vite/packages/ui/src/button.ts",)
    assert outside == ()  # two claimants equally far: neither is guessed


def test_package_json_outside_the_root_or_behind_a_symbolic_link_is_not_read(tmp_path: Path) -> None:
    # Arrange
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "package.json").write_text('{"name": "@acme/shared", "exports": "./src/index.ts"}')
    write_files(
        tmp_path / "repo",
        {
            # Read from inside packages/web by mistake, this would map #src/util onto its src/util.ts.
            "package.json": '{"imports": {"#src/*": "./src/*.ts"}}',
            "packages/shared/src/index.ts": "export const shared = 1;\n",
            "packages/web/src/util.ts": "export const util = 1;\n",
            "packages/web/src/page.ts": (
                'import { shared } from "@acme/shared";\nimport { util } from "#src/util";\n'
            ),
        },
    )
    (tmp_path / "repo/packages/shared/package.json").symlink_to(outside / "package.json")
    linked = CodeIndex(tmp_path / "repo", ["packages/shared/src/index.ts", "packages/web/src/page.ts"])
    above = CodeIndex(tmp_path / "repo/packages/web", ["src/util.ts", "src/page.ts"])

    # Act
    through_link = linked.imports("packages/web/src/page.ts")
    from_above = above.imports("src/page.ts")

    # Assert
    assert through_link == ()
    assert from_above == ()


def test_guessed_package_source_stays_discoverable_without_proving_a_call(tmp_path: Path) -> None:
    index = indexed(
        tmp_path,
        {
            "packages/shared/package.json": '{"name": "@acme/shared", "main": "./bundle/index.mjs"}',
            "packages/shared/src/index.ts": "export function handle() { return 1; }\n",
            "other/handle.ts": "export function handle() { return 2; }\n",
            "apps/web/page.ts": (
                'import { handle } from "@acme/shared";\nexport function run() { return handle(); }\n'
            ),
        },
    )
    run = index.find_definition("run")[0]
    edge = index.callee_edges(run)[0]

    # The original package mapping and its dependents remain useful for discovery.
    assert index.imports("apps/web/page.ts") == ("packages/shared/src/index.ts",)
    assert index.dependents("packages/shared/src/index.ts") == ("apps/web/page.ts",)
    assert edge.binding.status == "candidate"
    assert edge.binding.target is None
    assert "package.json mapping for @acme/shared" in edge.binding.reason

    offered = neighbours(index, index.read_slice(run))
    handle_files = {place.open().span.file for place in offered if place.open().span.name == "handle"}
    assert handle_files == {"packages/shared/src/index.ts", "other/handle.ts"}
    links = [link for link in operations.trace_graph(index, [run]).links if link.name == "handle"]
    assert {link.target.file for link in links if link.target} == handle_files
    assert all(link.binding.status == "candidate" for link in links)


def test_duplicate_package_proximity_is_a_candidate_without_losing_its_path(tmp_path: Path) -> None:
    ui = '{"name": "@ws/ui", "exports": {"./button": "./src/button.ts"}}'
    index = indexed(
        tmp_path,
        {
            "templates/next/packages/ui/package.json": ui,
            "templates/next/packages/ui/src/button.ts": "export function button() { return 1; }\n",
            "templates/next/apps/web/page.ts": (
                'import { button } from "@ws/ui/button";\nexport function render() { return button(); }\n'
            ),
            "templates/vite/packages/ui/package.json": ui,
            "templates/vite/packages/ui/src/button.ts": "export function button() { return 2; }\n",
        },
    )

    assert index.imports("templates/next/apps/web/page.ts") == ("templates/next/packages/ui/src/button.ts",)
    edge = index.callee_edges(index.find_definition("render")[0])[0]
    assert edge.binding.status == "candidate"
    assert edge.binding.target is None
    assert "@ws/ui/button" in edge.binding.reason


def test_package_uncertainty_propagates_through_a_relative_barrel(tmp_path: Path) -> None:
    index = indexed(
        tmp_path,
        {
            "package.json": '{"name": "@ws/ui", "exports": {"./button": "./dist/button.js"}}',
            "src/button.ts": "export function button() { return 1; }\n",
            "src/services/index.ts": 'export { button } from "@ws/ui/button";\n',
            "src/page.ts": (
                'import { button } from "./services";\nexport function render() { return button(); }\n'
            ),
        },
    )

    assert index.imports("src/page.ts") == ("src/services/index.ts",)
    edge = index.callee_edges(index.find_definition("render")[0])[0]
    assert edge.binding.status == "candidate"
    assert edge.binding.target is None
    assert "package.json mapping for @ws/ui/button" in edge.binding.reason


def test_package_redirects_continue_past_four_links_and_stop_cycles(tmp_path: Path) -> None:
    redirects = {f"#hop{i}": f"#hop{i + 1}" for i in range(6)}
    redirects["#hop6"] = "./src/answer.ts"
    redirects["#cycle"] = "#cycle"
    index = indexed(
        tmp_path,
        {
            "package.json": json.dumps({"imports": redirects}),
            "src/answer.ts": "export function answer() { return 42; }\n",
            "src/page.ts": (
                'import { answer } from "#hop0";\nimport "#cycle";\n'
                "export function run() { return answer(); }\n"
            ),
        },
    )

    assert index.imports("src/page.ts") == ("src/answer.ts",)
    edge = index.callee_edges(index.find_definition("run")[0])[0]
    assert edge.binding.status == "candidate"
    assert edge.binding.target is None


SHARED_PACKAGE = '{"name": "@acme/shared", "exports": {"./*": "./src/*.ts"}}'


@pytest.mark.parametrize(
    ("files", "specifier", "status"),
    [
        ({"apps/web/package.json": '{"imports": {"#lib/*": "./src/lib/*.ts"}}'}, "#lib/money", "resolved"),
        (
            {
                "apps/web/package.json": '{"dependencies": {"@acme/shared": "workspace:*"}}',
                "packages/shared/package.json": SHARED_PACKAGE,
            },
            "@acme/shared/money",
            "resolved",
        ),
        (
            {
                "apps/web/package.json": '{"dependencies": {"@acme/shared": "workspace:^1.2.0"}}',
                "packages/shared/package.json": SHARED_PACKAGE,
            },
            "@acme/shared/money",
            "resolved",
        ),
        (
            {"apps/web/package.json": '{"name": "web", "exports": {"./lib/*": "./src/lib/*.ts"}}'},
            "web/lib/money",
            "resolved",
        ),
        (
            {
                "apps/web/package.json": '{"dependencies": {"@acme/shared": "^1.0.0"}}',
                "packages/shared/package.json": SHARED_PACKAGE,
            },
            "@acme/shared/money",
            "candidate",
        ),
        (
            {
                "apps/web/package.json": '{"dependencies": {"@acme/shared": "workspace:@acme/other@*"}}',
                "packages/shared/package.json": SHARED_PACKAGE,
            },
            "@acme/shared/money",
            "candidate",
        ),
        ({"apps/web/package.json": '{"name": "web", "exports": null}'}, "web/src/lib/money", "candidate"),
        (
            {
                "apps/web/package.json": (
                    '{"imports": {"#lib/*": {"import": "./src/lib/*.ts", "require": "./src/cjs/*.ts"}}}'
                ),
                "apps/web/src/cjs/money.ts": "export function cents() { return 3; }\n",
            },
            "#lib/money",
            "candidate",
        ),
        (
            {
                "apps/web/package.json": '{"imports": {"#lib/*": "./src/lib/*.ts"}}',
                "apps/web/src/package.json": "{",
            },
            "#lib/money",
            "candidate",
        ),
    ],
    ids=[
        "hash-import",
        "workspace-dependency",
        "workspace-range",
        "self-reference",
        "version-range-may-be-installed",
        "workspace-alias-links-another-package",
        "null-exports-disable-self-reference",
        "targets-differ-by-condition",
        "nearer-unreadable-package-json",
    ],
)
def test_a_declared_package_mapping_proves_a_call_only_when_the_package_and_file_are_certain(
    tmp_path: Path, files: dict[str, str], specifier: str, status: str
) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            **files,
            "apps/web/src/lib/money.ts": "export function cents() { return 1; }\n",
            "packages/shared/src/money.ts": "export function cents() { return 2; }\n",
            "apps/web/src/page.ts": (
                f'import {{ cents }} from "{specifier}";\nexport function total() {{ return cents(); }}\n'
            ),
        },
    )

    # Act
    edge = index.callee_edges(index.find_definition("total")[0])[0]

    # Assert
    assert edge.binding.status == status
    assert (edge.binding.target is not None) == (status == "resolved")
