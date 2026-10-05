from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from git_repos import commit_files, read_files, write_files

from jev_navigator.comments import find_comments
from jev_navigator.directives.find_code import Outcome, find_code
from jev_navigator.directives.places import neighbours_and_omissions, place_for_line
from jev_navigator.index import listing, scope_scan, tools
from jev_navigator.index.bindings import Binding
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.languages import FLOW_LANGUAGE, FLOW_SGCONFIG, has_flow_pragma, language_of
from jev_navigator.index.scope_scan import OPAQUE_RECEIVER, FileFacts, FileStructure, Unparsed, scan_facts
from jev_navigator.index.spans import Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient


@pytest.fixture
def ast_grep_runs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None, list[str]]]:
    """One entry per ast-grep invocation: its first rule id, the sgconfig passed (None for a plain
    scan), and the files scanned."""
    runs: list[tuple[str, str | None, list[str]]] = []
    original_rules = tools.ast_grep_rules

    def counted_rules(rules: str, files, cwd, config=None, **keywords):
        runs.append((rules.split("\n", 1)[0], config, list(files)))
        return original_rules(rules, files, cwd, config=config, **keywords)

    monkeypatch.setattr(tools, "ast_grep_rules", counted_rules)
    return runs


class CountingResolver:
    """Answers nothing, so the index decides; counts how often the index asks."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, int, str]] = []

    def resolve_call(self, file: str, line: int, name: str, receiver: str | None) -> Binding | None:
        self.asked.append((file, line, name))
        return None


def committed(root: Path, files: dict[str, str]) -> CodeIndex:
    commit_files(root, files)
    return CodeIndex.from_git(root, fact_cache_dir=root.parent / f"{root.name}-fact-cache")


# The evidence bundle's flow_adapter.js: a real Flow-typed storage adapter (the eval fixture).
FLOW_ADAPTER = """\
// @flow
import type { StorageAdapter, QueryOptions } from './StorageAdapter';

const toPostgresValue = (value: any, options: ?QueryOptions): any => {
  const fields: Array<string>[] = [];
  return value;
};

export class PostgresAdapter implements StorageAdapter {
  _client: any;

  constructor({ uri }: { uri: string }) {
    this._client = connect(uri);
  }

  async createObject(className: string, object: Object, options: ?QueryOptions): Promise<void> {
    await this._client.none('INSERT INTO $1:name', [className, toPostgresValue(object, options)]);
  }

  find(className: string, query: Object): Promise<Array<Object>> {
    return this._client.any('SELECT * FROM $1:name', [className, query]);
  }
}

function connect(uri: string): any {
  return { uri };
}
"""

# The evidence bundle's RootTag.js (react-native v0.71.3): its `export opaque type` is a Flow
# construct no shipped grammar recovers, so the file must stay honestly incomplete.
ROOT_TAG = """\
/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @flow strict
 * @format
 */

import * as React from 'react';

export opaque type RootTag = number;

export const RootTagContext: React$Context<RootTag> =
  React.createContext<RootTag>(0);

if (__DEV__) {
  RootTagContext.displayName = 'RootTagContext';
}

/**
 * Intended to only be used by `AppContainer`.
 */
export function createRootTag(rootTag: number | RootTag): RootTag {
  return rootTag;
}
"""

ROOT_TAG_CALLER = """\
import { createRootTag } from './RootTag';

export function show(rootTag) {
  return createRootTag(rootTag);
}
"""

MEMORY_ADAPTER = """\
export class MemoryAdapter {
  constructor() {
    this.rows = [];
  }

  createObject(className, object) {
    this.rows.push({ className, object });
  }
}
"""

# The evidence bundle's auth.js: a plain-JS control with calls and a plain function.
AUTH_MIDDLEWARE = """\
export function authenticate(req, res, next) {
  if (!req.headers.authorization) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  return next();
}
"""

ADAPTER_CALLER = """\
import { PostgresAdapter } from './postgres';

export function store(uri) {
  const adapter = new PostgresAdapter({ uri });
  return adapter.find('users', {});
}
"""


def an_adapter_scope(tmp_path: Path) -> CodeIndex:
    """A Flow-typed file the tsx grammar recovers, plain-JS siblings, and a caller."""
    return committed(
        tmp_path,
        {
            "src/adapters/postgres.js": FLOW_ADAPTER,
            "src/adapters/memory.js": MEMORY_ADAPTER,
            "src/adapters/index.js": ADAPTER_CALLER,
        },
    )


def a_root_tag_scope(tmp_path: Path) -> CodeIndex:
    """A real Flow file whose opaque type no shipped grammar recovers, and its caller."""
    return committed(
        tmp_path,
        {"src/native/RootTag.js": ROOT_TAG, "src/native/show.js": ROOT_TAG_CALLER},
    )


def test_each_file_is_parsed_once_and_a_new_index_reuses_its_facts(
    sample_index: CodeIndex, ast_grep_runs
) -> None:
    # Arrange
    opened = [
        sample_index.read_slice(span) for file in sample_index.files for span in sample_index.symbols_in(file)
    ]

    # Act
    for code in opened:
        neighbours_and_omissions(sample_index, code)
    for name in ("validate_order", "check_limits", "handleOrder"):
        sample_index.find_callers(name)
        sample_index.find_references(name)

    cold_runs = len(ast_grep_runs)
    warm = CodeIndex.from_git(sample_index.root, fact_cache_dir=sample_index.root.parent / "fact-cache")
    for file in warm.files:
        warm.symbols_in(file)

    # Assert
    assert len(opened) >= 6
    assert cold_runs == len(sample_index._code_files)
    assert len(ast_grep_runs) == cold_runs


def test_each_call_site_is_bound_once_however_often_it_is_looked_up(sample_repo: Path) -> None:
    # Arrange
    resolver = CountingResolver()
    index = CodeIndex.from_git(sample_repo, binding_resolver=resolver)

    # Act
    first = index.find_callers("check_limits")
    index.find_callers("check_limits")
    index.callee_edges(index.find_definition("validate_order")[0])

    # Assert
    assert len(first) == 1
    assert resolver.asked.count(("app/validation.py", 7, "check_limits")) == 1
    assert len(resolver.asked) == len(set(resolver.asked))


def test_an_external_parser_failure_is_not_relabelled_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "module.py").write_text("def run():\n    return 1\n")

    def fail_parser(*arguments, **options):
        raise tools.ToolFailedError("ast-grep failed for a real tool reason")

    monkeypatch.setattr(tools, "ast_grep_rules", fail_parser)
    index = CodeIndex(tmp_path, ["module.py"], fact_cache_dir=tmp_path / "cache")

    with pytest.raises(tools.ToolFailedError, match="real tool reason"):
        index.functions_in("module.py")


def test_a_file_whose_parse_was_killed_is_not_counted_as_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: once the index exists, every ast-grep it starts is killed partway, as the machine's
    # memory watchdog or a user's `kill` would stop it.
    (tmp_path / "module.py").write_text("def run():\n    return 1\n")
    index = CodeIndex(tmp_path, ["module.py"], fact_cache_dir=tmp_path / "cache")
    killed = tmp_path / "killed-bin"
    killed.mkdir()
    (killed / tools.AST_GREP).write_text("#!/bin/sh\nkill -9 $$\n")
    (killed / tools.AST_GREP).chmod(0o755)
    working_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", f"{killed}{os.pathsep}{working_path}")

    # Act
    with pytest.raises(tools.ToolFailedError, match="exited -9"):
        index.functions_in("module.py")

    # Assert: the file is still unread, so no run counts it or calls the fact scan complete, and a
    # working parser reads it later.
    assert "module.py" not in index.parsed_files
    assert index.parser_scans_pending == ("facts",)
    monkeypatch.setenv("PATH", working_path)
    assert [span.name for span in index.functions_in("module.py")] == ["run"]
    assert "module.py" in index.parsed_files


def test_scan_facts_skips_unsupported_files_and_still_parses_supported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# notes\n")
    (tmp_path / "module.py").write_text("def greet(): return 1\n")

    empty = FileFacts(FileStructure(), (), ())

    unsupported = scan_facts(read_files(tmp_path, ["notes.md"]), tmp_path, Unparsed())
    mixed = scan_facts(read_files(tmp_path, ["module.py", "notes.md"]), tmp_path, Unparsed())

    assert unsupported == {"notes.md": empty}
    assert mixed["module.py"].structure.functions == (Span("module.py", 1, 1, "greet"),)
    assert mixed["notes.md"] == empty


def _many(template: str, count: int = 300) -> str:
    return "".join(template.format(n=n) for n in range(count))


def _wide_script_function() -> str:
    """A function with many parameters whose loop and catch bodies bind many locals."""
    parameters = _many("  p{n},\n")
    body = _many("    const l{n} = p{n};\n")
    return (
        f"function wide(\n{parameters}) {{\n  for (const item of items) {{\n{body}  }}\n"
        f"  try {{ run(); }} catch (error) {{\n{body}  }}\n}}\n"
    )


PANEL = """\
import { useState } from 'react';

export interface PanelProps { title: string; rows: Row[] }
export type Row = { id: string };
export enum Mode { Open, Closed }

export const Panel = ({ title, rows }: PanelProps) => {
  const [mode, setMode] = useState<Mode>(Mode.Open);
  const visible = rows.filter((row) => row.id !== '' && mode === Mode.Open);
  return <section onClick={() => setMode(Mode.Closed)}>{title}{visible.length}</section>;
};

export default class Store {
  #rows = new Map<string, Row>();
  add(row: Row) { this.#rows.set(row.id, row); return this; }
}
"""


def test_the_scan_builds_the_same_facts_as_from_every_field_the_parser_prints(
    sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scan decodes only the match fields its facts are built from. Built from every field of
    the parser's output instead, every file's facts are the same, so no field the facts read is
    skipped: Python, TypeScript, TSX, plain JavaScript, Flow and a file the grammar cannot recover."""
    write_files(
        sample_repo,
        {
            "web/panel.tsx": PANEL,
            "js/postgres.js": FLOW_ADAPTER,
            "js/RootTag.js": ROOT_TAG,
            "js/memory.js": MEMORY_ADAPTER,
            "js/auth.js": AUTH_MIDDLEWARE,
            "js/index.js": ADAPTER_CALLER,
        },
    )
    files = [file for file in listing.working_files(sample_repo).files if language_of(file)]

    narrow = scan_facts(read_files(sample_repo, files), sample_repo, Unparsed())
    monkeypatch.setattr(scope_scan, "decode_match", json.loads)
    whole = scan_facts(read_files(sample_repo, files), sample_repo, Unparsed())

    assert {language_of(file) for file in files} == {"python", "typescript", "tsx", "javascript"}
    assert any(facts.incomplete for facts in whole.values())
    assert narrow == whole


@pytest.mark.parametrize(
    ("file", "source"),
    [
        (
            "module.ts",
            _many("const a{n} = {n}, b{n} = {n};\nexport type T{n} = string;\n")
            + _many("const r{n} = require('./r{n}');\nimport * as ns{n} from './ns{n}';\n")
            + "const api = {\n"
            + _many("  m{n}() {{ return {n}; }},\n")
            + "};\n"
            + _wide_script_function(),
        ),
        (
            "module.js",
            _many("const {{ c{n} }} = settings;\nfoo.p{n} = function () {{ return {n}; }};\n")
            + "module.exports = {\n"
            + _many("  e{n}() {{ return {n}; }},\n  s{n},\n  p{n}: p{n},\n")
            + "};\n"
            + _wide_script_function(),
        ),
        (
            "module.py",
            _many("first{n}, second{n} = {n}, {n}\napp.debug{n} = True\n")
            + "def wide(\n"
            + _many("    p{n},\n")
            + "):\n    for item in items:\n"
            + _many("        l{n} = p{n}\n"),
        ),
    ],
    ids=["typescript", "javascript", "python"],
)
def test_no_fact_rule_prints_more_than_the_node_it_matched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: str, source: str
) -> None:
    """ast-grep prints every node a rule's relations match. A relation to a large ancestor, such as
    the program, a module statement or an object literal, printed that ancestor once per match, so
    the parser's output and memory grew with matches times file size: 1.2 MB of ordinary code
    peaked over 2.5 GB. A match prints its own node three times (its text, its lines and its primary
    label), each as JSON. The matches are recorded as printed, every field decoded, because the
    scan's own decoder skips the related nodes the whole file was printed in."""
    # Arrange
    (tmp_path / file).write_text(source)
    printed: list[dict] = []
    original_rules = tools.ast_grep_rules

    def recorded_rules(*arguments, **options):
        for match in original_rules(*arguments, **options | {"decode": json.loads}):
            printed.append(match)
            yield match

    monkeypatch.setattr(tools, "ast_grep_rules", recorded_rules)

    # Act
    scan_facts(read_files(tmp_path, [file]), tmp_path, Unparsed())

    # Assert
    oversized = {
        match["ruleId"]
        for match in printed
        if len(json.dumps(match)) > 2_000 + 3 * len(json.dumps(match["text"]))
    }
    assert printed
    assert oversized == set()


def test_a_plain_call_wins_over_a_method_call_of_the_same_name_on_one_line(tmp_path: Path) -> None:
    # Arrange
    index = committed(
        tmp_path,
        {"app/orders.py": "def save(order):\n    return helpers.save(order), save(order)\n"},
    )

    # Act
    site = index.find_callers("save")[0]

    # Assert
    assert site.binding.status == "resolved"


def test_calls_on_one_line_keep_their_source_order_on_every_scan(tmp_path: Path) -> None:
    """ast-grep runs its rules in parallel, so the matches of `new Date(...)` and of `merge(...)`
    arrive in either order. Calls on one line are ordered as the source writes them, on every scan.
    Of two calls starting at one place, such as `new Foo(a)` and `new Foo(a).bar(...)`, the outer
    comes first, as one rule already orders `foo.bar().baz()`."""
    # Arrange
    source = (
        "function f(a) {\n"
        "  const d = new Date(merge(a), now());\n"
        "  x = new Foo(a).bar(now());\n"
        "  const t = new Date(merge(a), now()).getTime();\n"
        "  return foo.bar().baz();\n"
        "}\n"
    )
    (tmp_path / "order.js").write_text(source)
    expected = {
        2: ("Date", "merge", "now"),
        3: ("bar", "Foo", "now"),
        4: ("getTime", "Date", "merge", "now"),
        5: ("baz", "bar"),
    }

    # Act
    orders = {
        tuple(
            (call.line, call.name)
            for call in scan_facts(read_files(tmp_path, ["order.js"]), tmp_path, Unparsed())["order.js"].calls
        )
        for _ in range(50)
    }

    # Assert
    assert orders == {tuple((line, name) for line, names in expected.items() for name in names)}


def test_the_flow_pragma_is_taken_from_leading_comments_not_from_strings_or_the_body() -> None:
    """Only comments before the first line of code count: `@flow` inside a string, after code, or in
    a lookalike word must not route a plain JavaScript file to the Flow grammar, and a byte-order
    mark or shebang may precede the pragma."""
    # Act and assert
    assert has_flow_pragma(["// @flow", "const a = 1;"])
    assert has_flow_pragma(["\ufeff#!/usr/bin/env node", "/* @flow strict */", "const a = 1;"])
    assert has_flow_pragma(["/**", " * @flow", " */", "const a = 1;"])
    assert has_flow_pragma(["/* @flow */ const a = 1;"])
    assert not has_flow_pragma(['const note = "// @flow";', "export function later() {}"])
    assert not has_flow_pragma(["const a = 1;", "// @flow"])
    assert not has_flow_pragma(["// @flowish", "const a = 1;"])
    assert not has_flow_pragma(["const a = 1;"])


def test_the_flow_pragma_survives_a_leading_comment_longer_than_a_hundred_lines() -> None:
    """Leading comments have no length cap: a `@flow` pragma past the hundredth leading line still
    routes, whether it hides in one block comment or in a run of line comments."""
    # Arrange
    padded_block = ["/* banner"] + [f" * padding line {i}" for i in range(120)] + [" * @flow", " */"]
    padded_lines = [f"// padding line {i}" for i in range(120)] + ["// @flow"]

    # Act and assert
    assert has_flow_pragma([*padded_block, "const a = 1;"])
    assert has_flow_pragma([*padded_lines, "const a = 1;"])


def test_the_pragma_scan_stops_at_real_code_even_when_the_head_is_long() -> None:
    """Stopping at the first line of code — not at an arbitrary line count — keeps later `@flow`
    occurrences false-positive free no matter how long the leading comment was: `@flow` in a string,
    in a trailing comment, or after code never routes."""
    # Arrange
    long_head = [f"// padding line {i}" for i in range(150)]

    # Act and assert
    assert not has_flow_pragma([*long_head, 'const note = "@flow";'])
    assert not has_flow_pragma([*long_head, "const a = 1; // @flow"])
    assert not has_flow_pragma([*long_head, "const a = 1;", "// @flow"])
    assert not has_flow_pragma([*long_head, "const a = 1; /* @flow */"])


def test_a_method_on_a_one_line_class_is_named_and_counted_itself(tmp_path: Path) -> None:
    """`class Box { v() { return 1; } }` on one line must give the class its own class span and the
    method its own function span, not one function named after the class that swallows both."""
    # Arrange
    index = committed(tmp_path, {"src/box.ts": "class Box { v() { return 1; } }\n"})

    # Act and assert
    assert [(span.name, span.start, span.end) for span in index.functions_in("src/box.ts")] == [("v", 1, 1)]
    assert {span.name for span in index.symbols_in("src/box.ts")} == {"Box", "v"}


def test_a_symbol_is_named_by_the_syntax_tree_and_a_callback_stays_anonymous(tmp_path: Path) -> None:
    """An expression is named by the declarator, field, key or assignment holding it, seen through
    parentheses and casts, before its own name; a callback passed to a call has no name. Every
    grammar is scanned together, so a node kind one grammar lacks would fail the whole scan."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/routes.ts": (
                "export const load = (async () => fetchPage()) satisfies PageLoad;\n"
                "export const GET = (() => respond()) as Handler;\n"
                "const handler = function inner() { return 1; };\n"
                'it("saves the order", () => { save(); });\n'
                "orders.save = () => 1;\n"
                'const routes = {\n  "risk.triage": () => 1,\n  plain: () => 2,\n};\n'
                "abstract class Shape { #area() { return 0; } }\n"
                "const Model = class {};\n"
                "function* pages() {}\n"
            ),
            "src/view.tsx": (
                'export const View = (() => <p />) satisfies Page;\ndescribe("view", () => {});\n'
            ),
            "src/legacy.js": (
                "const run = (function () {});\n"
                "class Job { start = () => 1; }\n"
                "const each = function* () {};\n"
            ),
            "app/jobs.py": "class Job:\n    def run(self):\n        return 1\n",
        },
    )

    # Act
    named = {
        file: [(span.start, span.name) for span in index.symbols_in(file)]
        for file in ("src/routes.ts", "src/view.tsx", "src/legacy.js", "app/jobs.py")
    }

    # Assert
    assert named == {
        "src/routes.ts": [
            (1, "load"),
            (2, "GET"),
            (3, "handler"),
            (4, "<anonymous>"),
            (5, "save"),
            (7, "<anonymous>"),
            (8, "plain"),
            (10, "Shape"),
            (10, "area"),
            (11, "Model"),
            (12, "pages"),
        ],
        "src/view.tsx": [(1, "View"), (2, "<anonymous>")],
        "src/legacy.js": [(1, "run"), (2, "Job"), (2, "start"), (3, "each")],
        "app/jobs.py": [(1, "Job"), (2, "run")],
    }
    assert index.unparsed_files == set()


def test_a_declaration_that_starts_mid_line_is_named_from_its_own_column(tmp_path: Path) -> None:
    """A declaration after other code on its line is named from where it starts, not from the line's
    first word: `if (ready) { start(); } const late = 1;` declares `late`, never `if`."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/late.ts": (
                "if (ready) { start(); } const late = () => 1;\n"
                'start("ü"); export type Id = string;\n'
                "start(); export const LIMIT = 3;\n"
            ),
            "app/settings.py": "DEBUG = False; TIMEOUT = 30\n",
        },
    )

    # Act
    declared = {
        file: [(span.start, span.name) for span in index.declarations_in(file)]
        for file in ("src/late.ts", "app/settings.py")
    }

    # Assert
    assert declared == {
        "src/late.ts": [(1, "late"), (2, "Id"), (3, "LIMIT")],
        "app/settings.py": [(1, "DEBUG"), (1, "TIMEOUT")],
    }


def test_a_declaration_names_every_name_it_binds(tmp_path: Path) -> None:
    """`const a = 1, b = 2` binds `a` and `b`, a destructuring binds each name it pulls out, and
    `first, second = 1, 2` binds both. A name declared inside the value is not the declaration's,
    and an attribute or item assignment binds no name at all."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/values.ts": (
                "export const a = 1, b = 2;\n"
                "export const { c, d: e, [key]: k, ...rest } = source;\n"
                "const [f, [g], h = fallback] = list;\n"
                "export type Id = string;\n"
                "const make = function () { const inner = 1; return inner; };\n"
            ),
            "src/values.js": "const a = 1, b = 2;\nconst { c, d: e } = settings;\n",
            "app/settings.py": (
                "first, second = 1, 2\n*head, last = [1, 2]\napp.debug = True\nconfig['x'] = 1\n"
                "TIMEOUT = RETRIES = 3\n"
            ),
        },
    )

    # Act
    declared = {
        file: [(span.start, span.name) for span in index.declarations_in(file)]
        for file in ("src/values.ts", "src/values.js", "app/settings.py")
    }

    # Assert
    assert declared == {
        "src/values.ts": [
            (1, "a"),
            (1, "b"),
            (2, "c"),
            (2, "e"),
            (2, "k"),
            (2, "rest"),
            (3, "f"),
            (3, "g"),
            (3, "h"),
            (4, "Id"),
            (5, "make"),
        ],
        "src/values.js": [(1, "a"), (1, "b"), (2, "c"), (2, "e")],
        "app/settings.py": [
            (1, "first"),
            (1, "second"),
            (2, "head"),
            (2, "last"),
            (5, "RETRIES"),
            (5, "TIMEOUT"),
        ],
    }


def test_module_level_var_and_ambient_declarations_define_their_names(tmp_path: Path) -> None:
    """A module-level `var` and TypeScript's `declare const`, `declare let`, `declare var` and
    `declare function` define names, so a call to one finds a definition instead of none. A `var`
    inside a function is the function's own."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "lib/app.js": (
                "var app = exports = module.exports = {};\nvar a = 1, b = 2;\n"
                "function inner() {\n  var hidden = 1;\n  return hidden;\n}\n"
            ),
            "types/env.d.ts": (
                "declare const VERSION: string;\nexport declare let mode: number, level: number;\n"
                "declare var legacy: number;\ndeclare function boot(): void;\n"
                "export declare function stop(code: number): void;\n"
            ),
            "src/main.ts": "export function main() {\n  return boot();\n}\n",
        },
    )

    # Act
    declared = {
        file: [(span.start, span.name) for span in index.declarations_in(file)]
        for file in ("lib/app.js", "types/env.d.ts")
    }
    boot = index.find_callers("boot")[0].binding

    # Assert
    assert declared == {
        "lib/app.js": [(1, "app"), (2, "a"), (2, "b")],
        "types/env.d.ts": [
            (1, "VERSION"),
            (2, "level"),
            (2, "mode"),
            (3, "legacy"),
            (4, "boot"),
            (5, "stop"),
        ],
    }
    assert (boot.status.value, boot.reason) == (
        "candidate",
        "name match only; 1 definitions in scope and no import of a module in scope names it",
    )


def test_whether_a_type_or_a_value_names_a_declaration_follows_its_own_kind(tmp_path: Path) -> None:
    """A type alias is named only by a type and a constant only by a value, also when the declaration
    starts after other code on its line."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/late.ts": (
                "if (ready) { start(); } const late = 1;\n"
                "start(); type Shape = { side: number };\n"
                "export function use(value: late): Shape {\n  late();\n  return Shape();\n}\n"
            )
        },
    )
    use = index.find_definition("use")[0]

    # Act
    by_types = {ref.name: ref.binding.status.value for ref in index.references_in(use) if ref.role == "type"}
    by_calls = {edge.name: edge.binding.status.value for edge in index.callee_edges(use)}

    # Assert
    assert by_types == {"late": "unresolved", "Shape": "resolved"}
    assert by_calls == {"late": "resolved", "Shape": "unresolved"}


def test_a_function_given_as_a_default_value_is_named_by_the_name_it_defaults(tmp_path: Path) -> None:
    """`onError = () => {}` in a parameter list is the function a call `onError()` may reach, so it
    is named `onError`, also as a destructured default. On a one-line function it shares the
    function's lines, and the calls on that line stay the function's own."""
    # Arrange
    script = (
        "export function upload(file, onError = () => {}) {\n  return onError;\n}\n"
        "export function save({ onDone = () => 1 } = {}, [first = () => 2] = []) {}\n"
        "export function retry(again = () => 1) { return attempt(); }\n"
    )
    index = committed(tmp_path, {"src/upload.ts": script, "src/upload.js": script})

    # Act
    named = {
        file: [(span.start, span.name) for span in index.functions_in(file)]
        for file in ("src/upload.ts", "src/upload.js")
    }
    attempt_callers = {site.file: site.caller for site in index.find_callers("attempt")}

    # Assert
    expected = [
        (1, "upload"),
        (1, "onError"),
        (4, "save"),
        (4, "onDone"),
        (4, "first"),
        (5, "retry"),
        (5, "again"),
    ]
    assert named == {"src/upload.ts": expected, "src/upload.js": expected}
    assert {file: caller.name for file, caller in attempt_callers.items()} == {
        "src/upload.ts": "retry",
        "src/upload.js": "retry",
    }


def test_a_module_level_function_sharing_its_line_and_name_with_a_held_method_stays_top_level(
    tmp_path: Path,
) -> None:
    """Spans are lines and a name, so `handler` and the object's method `handler` on one line are one
    span. The module-level function among them makes it a module-level definition."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/oneline.js": (
                "function handler() { return 1; } const table = { handler() { return 2; } }; handler();\n"
            )
        },
    )

    # Act
    binding = index.find_callers("handler")[0].binding

    # Assert
    assert (binding.status.value, binding.target) == ("resolved", Span("src/oneline.js", 1, 1, "handler"))


def test_symbols_sharing_a_line_are_top_level_only_when_nothing_holds_them(tmp_path: Path) -> None:
    """Symbols on one line each hold the other's first line, so lines cannot say which is top level;
    the syntax tree can. `retry` and the one-line class `Box` stay provable from their file, while
    `retry`'s default, an object literal's method and a function inside a callback stay candidates."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/jobs.ts": (
                "export function retry(again = () => 1) { return again(); }\n"
                "class Box { v() { return 1; } }\n"
                "const pair = { a() { return 1; }, b() { return 2; } };\n"
                'describe("jobs", () => { function helper() { return 1; } });\n'
                "export function run() {\n  retry();\n  new Box();\n  a();\n  helper();\n}\n"
            )
        },
    )

    # Act
    statuses = {
        name: index.find_callers(name)[0].binding.status.value
        for name in ("retry", "Box", "again", "a", "helper")
    }

    # Assert
    assert statuses == {
        "retry": "resolved",
        "Box": "resolved",
        "again": "candidate",
        "a": "candidate",
        "helper": "candidate",
    }


def test_an_object_literals_functions_are_its_properties_not_names_in_scope(tmp_path: Path) -> None:
    """`const api = { fetch() {} }` defines `api.fetch`, never a name `fetch`: neither a bare call in
    its file nor `import { fetch }` from another file proves it."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/api.ts": (
                "export const api = {\n  fetch() { return 1; },\n  stop: () => 2,\n};\n"
                "export function local() {\n  return fetch() + stop();\n}\n"
            ),
            "src/use.ts": "import { fetch } from './api';\nexport function go() {\n  return fetch();\n}\n",
        },
    )

    # Act
    statuses = {
        (site.file, name): site.binding.status.value
        for name in ("fetch", "stop")
        for site in index.find_callers(name)
    }

    # Assert
    assert statuses == {
        ("src/api.ts", "fetch"): "candidate",
        ("src/api.ts", "stop"): "candidate",
        ("src/use.ts", "fetch"): "candidate",
    }


def test_a_default_in_a_module_level_destructuring_is_never_the_proven_target(tmp_path: Path) -> None:
    """`const { onError = () => {} } = options` gives `onError` one possible value: the options may
    hold another function. The default keeps its name, but `onError()` is never proven to call it."""
    # Arrange
    script = "const {\n  onError = () => {\n    return 'default';\n  },\n} = options;\nonError();\n"
    index = committed(tmp_path, {"src/config.js": script, "src/config.ts": script})

    # Act
    named = {
        file: [span.name for span in index.functions_in(file)] for file in ("src/config.js", "src/config.ts")
    }
    targets = {file: index.binding_of(file, 6, "onError", None).target for file in named}

    # Assert
    assert named == {"src/config.js": ["onError"], "src/config.ts": ["onError"]}
    assert all(target != Span(file, 2, 4, "onError") for file, target in targets.items()), targets


def test_a_namespace_member_is_no_module_level_definition(tmp_path: Path) -> None:
    """A TypeScript namespace's members are its own: `config` inside namespace B is never proven to be
    namespace A's, and a module-level `read()` never reaches a namespace's `read`."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/spaces.ts": (
                "namespace A { export const config = 1; export function read() { return 1; } }\n"
                "namespace B {\n  export const config = 2;\n  export function show() { return config; }\n}\n"
                "read();\n"
            ),
            "src/cfg.ts": "export const config = 3;\n",
            "src/main.ts": (
                "import { config } from './cfg';\nnamespace A {\n  export const config = 1;\n}\n"
                "export function show() {\n  return config;\n}\n"
            ),
        },
    )

    # Act
    config = index.binding_of("src/spaces.ts", 4, "config", None, "return")
    read = index.binding_of("src/spaces.ts", 6, "read", None)
    imported = index.binding_of("src/main.ts", 6, "config", None, "return")

    # Assert
    assert config.target != Span("src/spaces.ts", 1, 1, "config"), config
    assert read.status.value != "resolved", read
    assert (imported.status.value, imported.target) == ("resolved", Span("src/cfg.ts", 1, 1, "config"))


def test_a_namespace_member_is_a_definition_inside_its_own_namespace(tmp_path: Path) -> None:
    """Inside namespace B, `config` names B's own member, never A's. A namespace's functions and its
    unexported constants bind from inside it, ahead of a module-level definition, and the innermost
    namespace holding the use wins. A use sharing the namespace's first or last line may sit outside
    it, so it stays open, and a method of an object the namespace holds is the object's, not a
    member."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/spaces.ts": (
                "namespace A { export const config = 1; }\n"
                "namespace B {\n  export const config = 2;\n  export function read() { return config; }\n}\n"
            ),
            "src/utils.ts": (
                "export namespace Utils {\n  export function helper() { return 1; }\n  const limit = 3;\n"
                "  export function main() { return helper() + limit; }\n}\n"
            ),
            "src/nested.ts": (
                "namespace Outer {\n  const depth = 1;\n  namespace Inner {\n    const depth = 2;\n"
                "    export function inner() { return depth; }\n  }\n"
                "  export function outer() { return depth; }\n}\nconst depth = 0;\n"
            ),
            "src/oneline.ts": "namespace A { export const config = 1; } config;\n",
            "src/table.ts": (
                "namespace T {\n  const table = { handler() { return 2; } };\n"
                "  export function run() { return handler(); }\n}\n"
            ),
        },
    )
    uses = {
        "config": ("src/spaces.ts", 4, "config", "return"),
        "helper": ("src/utils.ts", 4, "helper", None),
        "limit": ("src/utils.ts", 4, "limit", "return"),
        "inner depth": ("src/nested.ts", 5, "depth", "return"),
        "outer depth": ("src/nested.ts", 7, "depth", "return"),
        "config after a one-line namespace": ("src/oneline.ts", 1, "config", "return"),
        "a method of a namespace's value": ("src/table.ts", 3, "handler", None),
    }

    # Act
    bindings = {
        use: index.binding_of(file, line, name, None, role) for use, (file, line, name, role) in uses.items()
    }

    # Assert
    assert {use: (binding.status.value, binding.target) for use, binding in bindings.items()} == {
        "config": ("resolved", Span("src/spaces.ts", 3, 3, "config")),
        "helper": ("resolved", Span("src/utils.ts", 2, 2, "helper")),
        "limit": ("resolved", Span("src/utils.ts", 3, 3, "limit")),
        "inner depth": ("resolved", Span("src/nested.ts", 4, 4, "depth")),
        "outer depth": ("resolved", Span("src/nested.ts", 2, 2, "depth")),
        "config after a one-line namespace": ("candidate", None),
        "a method of a namespace's value": ("candidate", None),
    }


def test_a_var_in_a_namespace_is_a_member_of_that_namespace(tmp_path: Path) -> None:
    """A `var` directly in a namespace body is that namespace's member like a `const`: inside the
    namespace its name binds to it, and at module level it is no definition."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/legacy.ts": (
                "namespace Legacy {\n  var count = 1;\n  export function read() { return count; }\n}\n"
                "export function outside() { return count; }\n"
            ),
        },
    )

    # Act
    inside = index.binding_of("src/legacy.ts", 3, "count", None, "return")
    outside = index.binding_of("src/legacy.ts", 5, "count", None, "return")

    # Assert
    assert (inside.status.value, inside.target) == ("resolved", Span("src/legacy.ts", 2, 2, "count"))
    assert (outside.status.value, outside.target) == ("candidate", None), outside


def test_a_function_expressions_own_name_names_nothing_in_its_module(tmp_path: Path) -> None:
    """A named function or class expression binds its own name only inside itself, so `handler()`
    after `run(function handler() {})` reaches no definition of the module, on one line or several.
    An expression a declaration holds is named by the declaration: `f()` after `const f = function
    g() {}` still resolves, and `g()` finds no definition. One an assignment names, `later = function
    inner() {}`, defines `later` beside its `let`, which leaves `later()` open."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "callback.js": "run(function handler() { return 1; }); handler();\n",
            "lines.js": (
                "run(function retry() {\n  return 1;\n});\nregister(class Job {});\nretry();\nnew Job();\n"
            ),
            "held.js": "const f = function g() { return 1; };\nf();\ng();\n",
            "assigned.js": "let later;\nlater = function inner() { return 1; };\nlater();\n",
        },
    )
    sites = {
        "one-line callback": ("callback.js", 1, "handler"),
        "callback on its own lines": ("lines.js", 5, "retry"),
        "class expression": ("lines.js", 6, "Job"),
        "the declaration's name": ("held.js", 2, "f"),
        "the expression's own name": ("held.js", 3, "g"),
        "an expression an assignment names": ("assigned.js", 3, "later"),
    }

    # Act
    bindings = {site: index.binding_of(file, line, name, None) for site, (file, line, name) in sites.items()}

    # Assert
    assert {
        site: (binding.status.value, binding.target and binding.target.key)
        for site, binding in bindings.items()
    } == {
        "one-line callback": ("candidate", None),
        "callback on its own lines": ("candidate", None),
        "class expression": ("candidate", None),
        "the declaration's name": ("resolved", "held.js:1-1"),
        "the expression's own name": ("unresolved", None),
        "an expression an assignment names": ("candidate", None),
    }


def test_a_namespace_member_comes_before_an_import_and_after_a_functions_own_name(tmp_path: Path) -> None:
    """Inside a namespace its own `config` hides the module's import of `config`, while a parameter
    `config` hides the member. A function or constant a namespace exports is no export of its
    module, so an import of that name never reaches the module's private function of the same
    name."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/cfg.ts": "export const config = 3;\n",
            "src/spaces.ts": (
                "import { config } from './cfg';\nnamespace B {\n  const config = 2;\n"
                "  export function read() { return config; }\n"
                "  export function given(config) { return config; }\n}\n"
            ),
            "src/utils.ts": (
                "function helper() { return 0; }\nfunction run() { return 0; }\n"
                "export namespace Utils {\n  export function helper() { return 1; }\n"
                "  export const run = () => 1;\n}\n"
            ),
            "src/use.ts": "import { helper, run } from './utils';\nhelper();\nrun();\n",
        },
    )

    # Act
    member = index.binding_of("src/spaces.ts", 4, "config", None, "return")
    parameter = index.binding_of("src/spaces.ts", 5, "config", None, "return")
    imported = [
        index.binding_of("src/use.ts", line, name, None) for line, name in ((2, "helper"), (3, "run"))
    ]

    # Assert
    assert (member.status.value, member.target) == ("resolved", Span("src/spaces.ts", 3, 3, "config"))
    assert (parameter.status.value, parameter.target) == ("candidate", None), parameter
    assert [(binding.status.value, binding.target) for binding in imported] == [("candidate", None)] * 2


def test_a_namespace_member_is_never_passed_over_for_an_import_because_of_unparsed_lines(
    tmp_path: Path,
) -> None:
    """Lines the parser lost outside the namespace cannot hide a closer definition, so the member
    still wins over the module's import of its name. Lost lines inside the namespace that mention
    the name could hide one, so the use is unknown there, and never the import's."""
    # Arrange
    namespace = (
        "namespace B {\n  export function config() { return 2; }\n"
        "  export function f() {\n    return config();\n  }\n"
    )
    index = committed(
        tmp_path,
        {
            "src/cfg.ts": "export function config() { return 1; }\n",
            "src/outside.ts": (
                "import { config } from './cfg';\n" + namespace + "}\nconst broken = config(((;\n"
            ),
            "src/inside.ts": (
                "import { config } from './cfg';\n" + namespace + "  const broken = config(((;\n}\n"
            ),
        },
    )

    # Act
    outside = index.binding_of("src/outside.ts", 5, "config", None)
    inside = index.binding_of("src/inside.ts", 5, "config", None)

    # Assert
    assert (outside.status.value, outside.target) == ("resolved", Span("src/outside.ts", 3, 3, "config"))
    assert (inside.status.value, inside.target) == ("unknown", None), inside


def test_several_definitions_of_a_name_in_one_file_make_a_candidate(tmp_path: Path) -> None:
    """Two module-level definitions of one name leave the call open; a declaration and the function
    it holds are one definition, also over several lines and through an import."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "app/util.py": (
                "try:\n    import fast\n\n    def pick():\n        return fast.pick()\n"
                "except ImportError:\n\n    def pick():\n        return 1\n\n\n"
                "def use():\n    return pick()\n"
            ),
            "src/x.ts": "export const load =\n  () => 2;\nconst handler =\n  () => 1;\nhandler();\n",
            "src/use.ts": "import { load } from './x';\nload();\n",
        },
    )

    # Act
    pick = index.binding_of("app/util.py", 13, "pick", None)
    handler = index.binding_of("src/x.ts", 5, "handler", None)
    load = index.binding_of("src/use.ts", 2, "load", None)

    # Assert
    assert (pick.status.value, pick.target) == ("candidate", None)
    assert (handler.status.value, handler.target) == ("resolved", Span("src/x.ts", 4, 4, "handler"))
    assert (load.status.value, load.target) == ("resolved", Span("src/x.ts", 2, 2, "load"))


def test_a_callback_on_exactly_a_named_functions_lines_is_that_function(tmp_path: Path) -> None:
    """A callback spanning exactly a named function's lines is the same place at line granularity,
    so it stays part of that function: the function stays top level, a same-file call to it stays
    proven, and a call inside the callback is still that function's call."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/util.ts": (
                "export const ids = (xs: { id: number }[]) => xs.map((x) => x.id);\n"
                "export function total(xs: number[]) { return xs.reduce((a, b) => a + b, 0); }\n"
                "export const loadUser = (id: string) => request(id).then((r) => r.json());\n"
                "export const wait = (ms: number) => new Promise((done) => {\n"
                "  setTimeout(done, ms);\n"
                "});\n"
                "function request(id: string) { return fetch(id); }\n"
                "export function run() {\n"
                "  return ids([]).length + total([]);\n"
                "}\n"
            ),
        },
    )

    # Act
    symbols = [(span.start, span.end, span.name) for span in index.symbols_in("src/util.ts")]
    bindings = {name: index.find_callers(name)[0].binding for name in ("ids", "total")}
    request_caller = index.find_callers("request")[0].caller

    # Assert
    assert symbols == [
        (1, 1, "ids"),
        (2, 2, "total"),
        (3, 3, "loadUser"),
        (4, 6, "wait"),
        (7, 7, "request"),
        (8, 10, "run"),
    ]
    assert {name: (binding.status.value, binding.reason) for name, binding in bindings.items()} == {
        "ids": ("resolved", "defined in the same file"),
        "total": ("resolved", "defined in the same file"),
    }
    assert request_caller is not None and request_caller.name == "loadUser"


def test_a_flow_typed_class_keeps_its_methods(tmp_path: Path) -> None:
    """eval: `@flow` methods are recovered, not merely reported as omitted. The deciding spans the
    navigation needs (the class, its constructor and methods, module functions) resolve, the file
    parses without grammar errors, and the caller's binding degrades no further than a candidate."""
    # Arrange
    index = an_adapter_scope(tmp_path)

    # Act
    names = {span.name for span in index.symbols_in("src/adapters/postgres.js")}

    # Assert
    assert index.unparsed_files == set()
    assert {"PostgresAdapter", "constructor", "createObject", "find", "toPostgresValue", "connect"} <= names
    assert index.find_definition("createObject")
    site = index.find_callers("find")[0]
    assert (site.file, site.line) == ("src/adapters/index.js", 5)
    assert site.binding.status == "candidate"


def test_a_call_through_a_whole_module_import_reads_only_that_module(tmp_path: Path, ast_grep_runs) -> None:
    """`jwt.verify()` after `import * as jwt from './jwt'` or `const jwt = require('./jwt')` calls the
    `verify` that module defines or re-exports. It is read from that module's own facts, so another
    file defining a `verify` is never parsed to bind it."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/jwt.ts": "export function verify(token: string) {\n  return token;\n}\n",
            "src/index.ts": "export * from './jwt';\n",
            "src/unrelated.ts": "export function verify() {\n  return 0;\n}\n",
            "src/esm.ts": (
                "import * as jwt from './jwt';\nimport * as utils from './index';\n\n"
                "export function direct(token: string) {\n  return jwt.verify(token);\n}\n\n"
                "export function reexported(token: string) {\n  return utils.verify(token);\n}\n"
            ),
            "src/cjs.js": (
                "const jwt = require('./jwt');\n\nfunction check(token) {\n  return jwt.verify(token);\n}\n"
            ),
        },
    )
    callers = [
        next(span for span in index.functions_in(file) if span.name == name)
        for file, name in (("src/esm.ts", "direct"), ("src/esm.ts", "reexported"), ("src/cjs.js", "check"))
    ]

    # Act
    bindings = [edge.binding for caller in callers for edge in index.callee_edges(caller)]
    scanned = {file for _, _, files in ast_grep_runs for file in files}

    # Assert
    verify = Span("src/jwt.ts", 1, 3, "verify")
    assert [(binding.status.value, binding.target) for binding in bindings] == [("resolved", verify)] * 3
    assert "src/unrelated.ts" not in scanned


def test_a_module_alias_is_read_from_module_level_code_only(tmp_path: Path) -> None:
    """A name holds a whole module when module-level code binds it to `require('m')` or `import * as`.
    A require inside a function, a member read off a require, a call of one, a require of a computed
    name, a re-export, a template string and a comment hold none."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/main.ts": (
                "import * as jwt from './jwt';\nimport { sign } from './jwt';\n"
                "export * as tools from './tools';\n"
                "const db = require('./db');\nvar legacy = require(\"./legacy\");\n"
                "const verify = require('./jwt').verify;\nconst app = require('./app')(options);\n"
                "const dynamic = require(name);\n"
                "const template = `const fake = require('./fake');`;\n// const old = require('./old');\n"
                "function local() {\n  const store = require('./store');\n  return store;\n}\n"
            ),
        },
    )

    # Act
    aliases = index._facts_in("src/main.ts").module_aliases

    # Assert
    assert dict(aliases) == {"jwt": "./jwt", "db": "./db", "legacy": "./legacy"}


def test_a_python_module_alias_is_read_from_module_level_imports_only(tmp_path: Path) -> None:
    """`import a.b as n` binds `n` to `a.b`, and `import a.b` makes `a` and `a.b` reach the modules
    of those names; one statement may import several modules, and a module-level `try` counts. A
    name imported from a module, an import inside a function or class, and a comment hold none."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "app/main.py": (
                "import app.jobs as jobs\nimport app.mail  # sends receipts\n"
                "import json, app.billing as billing\nfrom app import tools\n# import app.old as old\n"
                "try:\n    import ujson as fast\nexcept ImportError:\n    pass\n\n\n"
                "def f():\n    import app.local as local\n    return local\n\n\n"
                "class K:\n    import app.inner as inner\n"
            ),
        },
    )

    # Act
    aliases = index._facts_in("app/main.py").module_aliases

    # Assert
    assert dict(aliases) == {
        "jobs": "app.jobs",
        "app": "app",
        "app.mail": "app.mail",
        "json": "json",
        "billing": "app.billing",
        "fast": "ujson",
    }


def test_a_call_through_a_python_module_import_reads_only_that_module(tmp_path: Path, ast_grep_runs) -> None:
    """`jobs.run()` after `import app.jobs as jobs`, and `app.jobs.run()` after `import app.jobs`,
    call the `run` that module defines, read from its own facts like a script module import. A
    parameter named `jobs` replaces the import inside its function, and a name a `try` binds to one
    module and its `except` to another holds neither."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/jobs.py": "def run(task):\n    return task\n",
            "app/unrelated.py": "def run():\n    return 0\n",
            "app/worker.py": (
                "import app.jobs as jobs\nimport app.jobs\n\n\n"
                "def aliased(task):\n    return jobs.run(task)\n\n\n"
                "def dotted(task):\n    return app.jobs.run(task)\n\n\n"
                "def injected(jobs, task):\n    return jobs.run(task)\n"
            ),
            "app/fast.py": "def loads(text):\n    return text\n",
            "app/slow.py": "def loads(text):\n    return text\n",
            "app/reader.py": (
                "try:\n    import app.fast as codec\nexcept ImportError:\n    import app.slow as codec\n\n\n"
                "def read(text):\n    return codec.loads(text)\n"
            ),
        },
    )
    caller = {span.name: span for span in index.functions_in("app/worker.py")}

    # Act
    through_imports = [
        edge.binding for name in ("aliased", "dotted") for edge in index.callee_edges(caller[name])
    ]
    scanned = {file for _, _, files in ast_grep_runs for file in files}
    injected = index.callee_edges(caller["injected"])[0].binding
    either = index.binding_of("app/reader.py", 8, "loads", "codec")

    # Assert
    run = Span("app/jobs.py", 1, 2, "run")
    assert [(binding.status.value, binding.target) for binding in through_imports] == [("resolved", run)] * 2
    assert "app/unrelated.py" not in scanned
    assert (injected.status.value, injected.target) == ("candidate", None)
    assert (either.status.value, either.target) == ("candidate", None)


def test_a_call_through_a_module_alias_binds_only_where_no_local_name_replaces_it(tmp_path: Path) -> None:
    """`db.query()` binds to db.js's `query` where `db` is the module-level alias; a parameter `db`, a
    `const store = require(...)` inside a function, a name module-level code binds to two modules, or
    an alias that only a template string spells, leave the call a candidate."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "db.js": "exports.query = function () { return 1; };\n",
            "fake.js": "exports.query = function () { return 2; };\n",
            "handler.js": (
                "const db = require('./db');\nfunction useReal() { return db.query(); }\n"
                "function useInjected(db) { return db.query(); }\n"
            ),
            "two.js": (
                "function a() { const store = require('./db'); return store.query(); }\n"
                "function b() { const store = require('./fake'); return store.query(); }\n"
            ),
            "gen.js": (
                "const template = `const api = require('./db');`;\n"
                "function emit(api) { return api.query(); }\n"
            ),
            "twice.js": (
                "var store = require('./db');\nvar store = require('./fake');\n"
                "function use() { return store.query(); }\n"
            ),
        },
    )
    sites = {
        "handler.js": ((2, "db"), (3, "db")),
        "two.js": ((1, "store"), (2, "store")),
        "gen.js": ((2, "api"),),
        "twice.js": ((3, "store"),),
    }

    # Act
    bindings = {
        (file, line): index.binding_of(file, line, "query", receiver).status.value
        for file, found in sites.items()
        for line, receiver in found
    }
    real = index.binding_of("handler.js", 2, "query", "db")

    # Assert
    assert (real.status.value, real.target) == ("resolved", Span("db.js", 1, 1, "query"))
    assert bindings == {
        ("handler.js", 2): "resolved",
        ("handler.js", 3): "candidate",
        ("two.js", 1): "candidate",
        ("two.js", 2): "candidate",
        ("gen.js", 2): "candidate",
        ("twice.js", 3): "candidate",
    }


def test_an_import_alias_replaced_by_a_local_name_binds_nothing_through_the_import(tmp_path: Path) -> None:
    """A parameter or local variable with an alias's name replaces the import inside its function:
    `halt()` there is not the module's `stop`, in TypeScript or Python, nor `cls()` after
    `cls = self.client_class` and a fallback import `as cls`. Such a local value is a candidate whose
    target is not resolved, never an absent definition."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/x.ts": "export function stop(code: number) {\n  return code;\n}\n",
            "src/param.ts": (
                "import { stop as halt } from './x';\n\n"
                "export function quit(halt: () => number) {\n  return halt();\n}\n"
            ),
            "src/local.ts": (
                "import { stop as halt } from './x';\n\nexport function quit() {\n  const halt = () => 0;\n"
                "  return halt();\n}\n"
            ),
            "app/__init__.py": "",
            "app/jobs.py": "def refund(order):\n    return order\n\n\nclass Client:\n    pass\n",
            "app/routes.py": (
                "from app.jobs import refund as give_back\n\n\ndef undo(give_back):\n    return give_back()\n"
            ),
            "app/factory.py": (
                "def make(self):\n    cls = self.client_class\n    if cls is None:\n"
                "        from app.jobs import Client as cls\n    return cls()\n"
            ),
        },
    )
    sites = (("src/param.ts", 4, "halt"), ("src/local.ts", 5, "halt"), ("app/routes.py", 5, "give_back"))

    # Act
    targets = {file: index.binding_of(file, line, name, None).target for file, line, name in sites}
    cls = index.binding_of("app/factory.py", 5, "cls", None)

    # Assert
    assert Span("src/x.ts", 1, 3, "stop") not in targets.values(), targets
    assert targets["app/routes.py"] != Span("app/jobs.py", 1, 2, "refund")
    assert (cls.status.value, cls.reason) == (
        "candidate",
        "cls is bound inside the calling function; its value is not resolved",
    )


def test_a_local_name_replaces_an_import_for_values_and_never_for_types(tmp_path: Path) -> None:
    """A parameter `stop` replaces `import { stop }` inside its own function only: a call there names
    the local value, a call in the next function the import. A class body's name reaches none of its
    methods. A type is looked up among types, so `cfg: Config` beside a parameter `Config` still names
    the interface."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/x.ts": "export function stop(code: number) {\n  return code;\n}\n",
            "src/main.ts": (
                "import { stop } from './x';\n\n"
                "export function quit(stop: () => number) {\n  return stop();\n}\n\n"
                "export function close() {\n  return stop(0);\n}\n"
            ),
            "src/same.ts": (
                "export interface Config {\n  depth: number;\n}\n"
                "export function make(Config: number) {\n"
                "  const cfg: Config = { depth: Config };\n  return cfg;\n}\n"
            ),
            "app/__init__.py": "",
            "app/jobs.py": "def refund(order):\n    return order\n",
            "app/orders.py": (
                "from app.jobs import refund as give_back\n\n\ndef build():\n    class Orders:\n"
                "        give_back = None\n\n"
                "        def undo(self, order):\n            return give_back(order)\n"
            ),
        },
    )

    # Act
    shadowed = index.binding_of("src/main.ts", 4, "stop", None)
    imported = index.binding_of("src/main.ts", 8, "stop", None)
    typed = index.binding_of("src/same.ts", 5, "Config", None, "type")
    method = index.binding_of("app/orders.py", 9, "give_back", None)

    # Assert
    assert (shadowed.status.value, shadowed.target) == ("candidate", None)
    assert (method.status.value, method.target) == ("resolved", Span("app/jobs.py", 1, 2, "refund"))
    assert (imported.status.value, imported.target) == ("resolved", Span("src/x.ts", 1, 3, "stop"))
    assert (typed.status.value, typed.target) == ("resolved", Span("src/same.ts", 1, 3, "Config"))


def test_an_export_list_entry_under_another_name_exports_its_own_definition(tmp_path: Path) -> None:
    """`export { inner as outer }` exports `inner` under the name `outer`: an import of `outer`
    reaches `inner`, never the module's private `outer`, and an import of `inner` reaches nothing.
    A re-export's names are another module's, so `export { b } from` never exports the module's
    private `b`."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/ren.ts": (
                "function inner() { return 1; }\nfunction outer() { return 2; }\nexport { inner as outer };\n"
            ),
            "src/use_outer.ts": "import { outer } from './ren';\nouter();\n",
            "src/use_inner.ts": "import { inner } from './ren';\ninner();\n",
            "src/barrel.ts": "function b() { return 0; }\nexport { b } from './missing';\n",
            "src/use_barrel.ts": "import { b } from './barrel';\nb();\n",
        },
    )

    # Act
    outer = index.binding_of("src/use_outer.ts", 2, "outer", None)
    inner = index.binding_of("src/use_inner.ts", 2, "inner", None)
    reexported = index.binding_of("src/use_barrel.ts", 2, "b", None)

    # Assert
    assert (outer.status.value, outer.target) == ("resolved", Span("src/ren.ts", 1, 1, "inner"))
    assert (inner.status.value, inner.target) == ("candidate", None), inner
    assert (reexported.status.value, reexported.target) == ("candidate", None), reexported


def test_a_commonjs_export_under_another_name_exports_its_own_definition(tmp_path: Path) -> None:
    """`exports.parse = urlParse` and `module.exports = { a: b }` export `urlParse` and `b` under the
    names `parse` and `a`: an import of `a` reaches `b`, never the module's private `a`."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "url.js": "function urlParse() { return 1; }\nexports.parse = urlParse;\n",
            "m.js": "function a() { return 0; }\nfunction b() { return 1; }\nmodule.exports = { a: b };\n",
            "app.js": (
                "const { parse } = require('./url');\nconst { a } = require('./m');\n"
                "const m = require('./m');\nparse();\na();\nm.a();\n"
            ),
        },
    )
    sites = {"parse": (4, "parse", None), "a": (5, "a", None), "m.a": (6, "a", "m")}

    # Act
    bindings = {
        site: index.binding_of("app.js", line, name, receiver)
        for site, (line, name, receiver) in sites.items()
    }

    # Assert
    assert {
        site: (binding.status.value, binding.target and binding.target.key)
        for site, binding in bindings.items()
    } == {
        "parse": ("resolved", "url.js:1-1"),
        "a": ("resolved", "m.js:2-2"),
        "m.a": ("resolved", "m.js:2-2"),
    }


def test_a_default_export_is_imported_only_as_the_default(tmp_path: Path) -> None:
    """A default import takes the module's default export under any local name, and `{ default as
    entry }` is one too. The default export's own name is no named export: `import { make }`, a
    member `defaults.make()` of the whole module, and `const { solo } = require(...)` of a module that
    assigns `module.exports = solo` reach nothing, and an import named like a private function never
    reaches it. A re-export passes the default on, and a module that assigns `module.exports` twice
    leaves it open."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "defaults.ts": "export default function make() {\n  return 1;\n}\n",
            "jobs.ts": "function run() { return 0; }\nexport default function start() { return run(); }\n",
            "single.js": "function solo() {\n  return 4;\n}\nmodule.exports = solo;\n",
            "main.ts": (
                "import begin from './defaults';\nimport { default as entry } from './defaults';\n"
                "import { make } from './defaults';\nimport * as defaults from './defaults';\n"
                "import run from './jobs';\n"
                "begin();\nentry();\nmake();\ndefaults.make();\nrun();\n"
            ),
            "app.js": "const { solo } = require('./single');\nsolo();\n",
            "barrel.ts": "export { default } from './defaults';\n",
            "through.ts": "import made from './barrel';\nmade();\n",
            "twice.js": "function a() {}\nfunction b() {}\nmodule.exports = a;\nmodule.exports = b;\n",
            "esm.js": "import either from './twice';\neither();\n",
        },
    )
    sites = {
        "a default import under another name": ("main.ts", 6, "begin", None),
        "default as entry": ("main.ts", 7, "entry", None),
        "the default's own name by name": ("main.ts", 8, "make", None),
        "the default's own name on the module": ("main.ts", 9, "make", "defaults"),
        "a default import named like a private function": ("main.ts", 10, "run", None),
        "a destructured require of module.exports": ("app.js", 2, "solo", None),
        "a default passed on by a re-export": ("through.ts", 2, "made", None),
        "a default assigned twice": ("esm.js", 2, "either", None),
    }

    # Act
    bindings = {
        site: index.binding_of(file, line, name, receiver)
        for site, (file, line, name, receiver) in sites.items()
    }

    # Assert
    assert {
        site: (binding.status.value, binding.target and binding.target.key)
        for site, binding in bindings.items()
    } == {
        "a default import under another name": ("resolved", "defaults.ts:1-3"),
        "default as entry": ("resolved", "defaults.ts:1-3"),
        "the default's own name by name": ("candidate", None),
        "the default's own name on the module": ("candidate", None),
        "a default import named like a private function": ("resolved", "jobs.ts:2-2"),
        "a destructured require of module.exports": ("candidate", None),
        "a default passed on by a re-export": ("resolved", "defaults.ts:1-3"),
        "a default assigned twice": ("candidate", None),
    }


def test_a_default_whose_definition_may_sit_in_unparsed_lines_stays_unknown(tmp_path: Path) -> None:
    """A syntax error neither the JavaScript nor the flow grammar reads hides the declaration of the
    function `export default loadAdapter` names. A default import of it is unknown, since the lines
    the parser lost mention the definition's own name. (parse-server's AdapterLoader.js once hid it
    with Flow types, which the flow reading now reads.)"""
    # Arrange
    index = committed(
        tmp_path,
        {
            "loader.js": (
                "export function +* loadAdapter(adapter) {\n  return adapter;\n}\n"
                "export default loadAdapter;\n"
            ),
            "use.js": "import load from './loader';\nload();\n",
        },
    )

    # Act
    binding = index.binding_of("use.js", 2, "load", None)

    # Assert
    assert binding.status.value == "unknown", binding


def test_only_what_a_script_module_exports_is_importable(tmp_path: Path) -> None:
    """A script module's own functions are importable only where it exports them: by an `export`
    statement or list, as its default export, or as a CommonJS export (`exports.x = x`, a function
    assigned to `exports.x`, a member of `module.exports = {...}`). A module exporting `new Logger()`
    exports no `log`, and an unexported helper stays the module's own."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "logger.js": (
                "function log(message) { return message; }\nclass Logger {\n"
                "  log(message) { return log(message); }\n}\nmodule.exports = new Logger();\n"
            ),
            "named.js": (
                "function log(message) { return message; }\nfunction query() { return 1; }\n"
                "function walk() { return 2; }\nmodule.exports = { log, walk: walk };\n"
                "exports.query = query;\n"
            ),
            "app.js": (
                "const logger = require('./logger');\nconst { log } = require('./logger');\n"
                "const named = require('./named');\nlogger.log('hello');\nlog('hello');\n"
                "named.log('x');\nnamed.query();\nnamed.walk();\n"
            ),
            "service.ts": (
                "function helper() {\n  return 1;\n}\nexport function run() {\n  return helper();\n}\n"
            ),
            "defaults.ts": "export default function make() {\n  return 1;\n}\n",
            "aliased.ts": "function build() {\n  return 2;\n}\nexport default build;\n",
            "listed.ts": "function listed() {\n  return 3;\n}\nexport { listed };\n",
            "single.js": "function solo() {\n  return 4;\n}\nmodule.exports = solo;\n",
            "made.js": "export default function made() {\n  return 5;\n}\n",
            "esm.js": "import made from './made';\nmade();\n",
            "main.ts": (
                "import * as service from './service';\nimport { helper } from './service';\n"
                "import make from './defaults';\nimport build from './aliased';\n"
                "import { listed } from './listed';\nimport solo from './single';\n"
                "service.helper();\nhelper();\nmake();\nbuild();\nlisted();\nsolo();\n"
            ),
        },
    )
    sites = {
        ("app.js", 4): ("log", "logger"),
        ("app.js", 5): ("log", None),
        ("app.js", 6): ("log", "named"),
        ("app.js", 7): ("query", "named"),
        ("app.js", 8): ("walk", "named"),
        ("main.ts", 7): ("helper", "service"),
        ("main.ts", 8): ("helper", None),
        ("main.ts", 9): ("make", None),
        ("main.ts", 10): ("build", None),
        ("main.ts", 11): ("listed", None),
        ("main.ts", 12): ("solo", None),
        ("esm.js", 2): ("made", None),
    }

    # Act
    bindings = {site: index.binding_of(*site, name, receiver) for site, (name, receiver) in sites.items()}

    # Assert
    assert {
        site: (binding.status.value, binding.target and binding.target.key)
        for site, binding in bindings.items()
    } == {
        ("app.js", 4): ("candidate", None),
        ("app.js", 5): ("candidate", None),
        ("app.js", 6): ("resolved", "named.js:1-1"),
        ("app.js", 7): ("resolved", "named.js:2-2"),
        ("app.js", 8): ("resolved", "named.js:3-3"),
        ("main.ts", 7): ("candidate", None),
        ("main.ts", 8): ("candidate", None),
        ("main.ts", 9): ("resolved", "defaults.ts:1-3"),
        ("main.ts", 10): ("resolved", "aliased.ts:1-3"),
        ("main.ts", 11): ("resolved", "listed.ts:1-3"),
        ("main.ts", 12): ("resolved", "single.js:1-3"),
        ("esm.js", 2): ("resolved", "made.js:1-3"),
    }


def test_a_member_read_through_an_import_is_decided_like_a_named_import(tmp_path: Path) -> None:
    """`lib.make()` through a module alias and `stroll()` through a renamed import are decided like
    `make()` after `import { make }`: two modules re-exporting `make` leave it a candidate, an
    exporting module whose unparsed lines mention the name leaves it unknown, and a vanished module
    is named as the one that could hold it. A module the import names but that exports no such name
    leaves a candidate that says so, never an absent definition."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "lib/a.ts": "export function make() { return 'a'; }\n",
            "lib/b.ts": "export function make() { return 'b'; }\n",
            "lib/index.ts": "export * from './a';\nexport * from './b';\n",
            "use_ns.ts": "import * as lib from './lib/index';\nlib.make();\n",
            "broken.js": (
                "exports.run = function () { return 'recovered'; };\n"
                "exports.other = function ( { run( ;; ) ) ) => {\n}\n"
            ),
            "use_broken.js": "const broken = require('./broken');\nbroken.run();\n",
            "lone.js": "exports.solo = function () {};\n",
            "gone.js": "exports.solo = function () {};\n",
            "use_lone.js": "const lone = require('./lone');\nlone.solo();\n",
            "tools.js": "exports.run = function () {};\n",
            "use_tools.js": "const tools = require('./tools');\ntools.walk();\n",
            "use_tools.ts": (
                "import { walk } from './tools';\nimport { walk as stroll } from './tools';\n"
                "walk();\nstroll();\n"
            ),
        },
    )
    (tmp_path / "lone.js").unlink()
    (tmp_path / "gone.js").unlink()
    sites = {
        ("use_ns.ts", 2): ("make", "lib"),
        ("use_broken.js", 2): ("run", "broken"),
        ("use_lone.js", 2): ("solo", "lone"),
        ("use_tools.js", 2): ("walk", "tools"),
        ("use_tools.ts", 3): ("walk", None),
        ("use_tools.ts", 4): ("stroll", None),
    }

    # Act
    bindings = {site: index.binding_of(*site, name, receiver) for site, (name, receiver) in sites.items()}

    # Assert
    assert {site: (binding.status.value, binding.target) for site, binding in bindings.items()} == {
        ("use_ns.ts", 2): ("candidate", None),
        ("use_broken.js", 2): ("unknown", None),
        ("use_lone.js", 2): ("unknown", None),
        ("use_tools.js", 2): ("candidate", None),
        ("use_tools.ts", 3): ("candidate", None),
        ("use_tools.ts", 4): ("candidate", None),
    }
    assert bindings[("use_ns.ts", 2)].reason == "import suggests multiple definitions: lib/a.ts, lib/b.ts"
    assert bindings[("use_lone.js", 2)].reason == "solo may be defined in files not parsed: lone.js"
    assert {
        bindings[site].reason for site in (("use_tools.js", 2), ("use_tools.ts", 3), ("use_tools.ts", 4))
    } == {
        "the import names tools.js, where the index finds no definition exported as walk; a name that "
        "module imports and passes on is not followed"
    }


def test_a_name_a_python_module_imports_and_passes_on_is_not_claimed_unexported(tmp_path: Path) -> None:
    """A Python module exports the names it imports at module level too, so `from pkg.api import
    compute` where api.py imports `compute` from pkg.core is no proof that api.py exports none: the
    candidate says the index found no definition there and does not follow the import."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/core.py": "def compute():\n    return 1\n",
            "pkg/api.py": "from pkg.core import compute\n\n\ndef compute_local():\n    return 2\n",
            "use_api.py": "from pkg.api import compute\ncompute()\n",
        },
    )

    # Act
    binding = index.binding_of("use_api.py", 2, "compute", None)

    # Assert
    assert (binding.status.value, binding.reason) == (
        "candidate",
        "the import names pkg/api.py, where the index finds no definition exported as compute; a name "
        "that module imports and passes on is not followed",
    )


def test_a_name_imported_under_an_alias_binds_to_the_exported_definition(
    tmp_path: Path, ast_grep_runs
) -> None:
    """`halt()` after `import { stop as halt }`, `const { stop: halt } = require()` or, in Python,
    `from app.jobs import refund as give_back`, calls the name the module exports. It is read from
    that module's own facts, so another file defining either name is never parsed to bind it."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/x.ts": "export function stop(code: number) {\n  return code;\n}\n",
            "src/unrelated.ts": "export function stop() {\n  return 0;\n}\nexport function halt() {}\n",
            "src/esm.ts": (
                "import { stop as halt } from './x';\n\nexport function quit() {\n  return halt(1);\n}\n"
            ),
            "src/cjs.js": (
                "const { stop: halt } = require('./x');\n\nfunction quit() {\n  return halt(1);\n}\n"
            ),
            "app/__init__.py": "",
            "app/jobs.py": "def refund(order):\n    return order\n",
            "app/unrelated.py": "def refund():\n    return 0\n\n\ndef give_back():\n    return 0\n",
            "app/routes.py": (
                "from app.jobs import refund as give_back\n\n\n"
                "def undo(order):\n    return give_back(order)\n"
            ),
        },
    )
    callers = [
        next(span for span in index.functions_in(file) if span.name == name)
        for file, name in (("src/esm.ts", "quit"), ("src/cjs.js", "quit"), ("app/routes.py", "undo"))
    ]

    # Act
    bindings = [edge.binding for caller in callers for edge in index.callee_edges(caller)]
    scanned = {file for _, _, files in ast_grep_runs for file in files}

    # Assert
    assert [(binding.status.value, binding.target) for binding in bindings] == [
        ("resolved", Span("src/x.ts", 1, 3, "stop")),
        ("resolved", Span("src/x.ts", 1, 3, "stop")),
        ("resolved", Span("app/jobs.py", 1, 2, "refund")),
    ]
    assert scanned.isdisjoint({"src/unrelated.ts", "app/unrelated.py"})


def test_a_definition_in_the_same_file_wins_over_the_alias_it_replaces(tmp_path: Path) -> None:
    """Python may define a name again after importing it under that alias; the call reaches the
    file's own definition, as it does for a name imported without an alias."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/jobs.py": "def refund(order):\n    return order\n",
            "app/routes.py": (
                "from app.jobs import refund as give_back\n\n\n"
                "def give_back(order):\n    return None\n\n\n"
                "def undo(order):\n    return give_back(order)\n"
            ),
        },
    )

    # Act
    [site] = index.find_callers("give_back")

    # Assert
    assert (site.binding.status.value, site.binding.target) == (
        "resolved",
        Span("app/routes.py", 4, 5, "give_back"),
    )


def test_the_flow_partition_is_scanned_on_its_own(tmp_path: Path, ast_grep_runs) -> None:
    """The `languageGlobs` config is global per invocation, so `@flow` files are scanned in their own
    invocation and plain JavaScript keeps the JavaScript grammar byte for byte."""
    # Arrange
    index = an_adapter_scope(tmp_path)

    # Act
    assert index.find_definition("createObject")  # the structure scan
    index.find_references("connect")  # the calls and references scans

    # Assert
    flow_runs = [files for _, config, files in ast_grep_runs if config]
    assert flow_runs and all(files == ["src/adapters/postgres.js"] for files in flow_runs)
    assert all("src/adapters/memory.js" not in files for _, config, files in ast_grep_runs if config)


def test_plain_javascript_is_unchanged_whether_or_not_flow_files_share_the_scope(tmp_path: Path) -> None:
    # Arrange
    alone = committed(tmp_path / "alone", {"src/memory.js": MEMORY_ADAPTER, "src/auth.js": AUTH_MIDDLEWARE})
    mixed = committed(
        tmp_path / "mixed",
        {
            "src/memory.js": MEMORY_ADAPTER,
            "src/auth.js": AUTH_MIDDLEWARE,
            "src/postgres.js": FLOW_ADAPTER,
        },
    )

    # Act and assert
    for file in ("src/memory.js", "src/auth.js"):
        assert mixed.functions_in(file) == alone.functions_in(file)
        assert mixed.symbols_in(file) == alone.symbols_in(file)
        assert mixed.declarations_in(file) == alone.declarations_in(file)
    for name in ("authenticate", "createObject"):
        span = alone.find_definition(name)[0]
        assert mixed.find_callees(span) == alone.find_callees(span)
        assert mixed.find_references(name) == alone.find_references(name)
    assert alone.unparsed_files == set()


def test_the_export_surface_facts_come_from_the_parser_nodes(tmp_path: Path) -> None:
    """Declaration name nodes and the module's own specifier nodes carry the surface: a default, a
    wildcard, a multi-line re-export list (another module's names), two constants in one statement
    and a template-literal body are each handled by the parser, not by source-text scanning."""
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/service.ts": (
                "export function run() {}\n"
                "export default function defaultRun() {}\n"
                'export * from "./one";\n'
                "export {\n"
                "  refund,\n"
                "  createOrder as placeOrder,\n"
                "} from './commands';\n"
                "const tpl = `export function inTemplate() {}`;\n"
                "export const first = 1, second = 2;\n"
            ),
        },
    )

    # Act
    facts = index._facts_in("src/service.ts")

    # Assert
    assert facts.export_names == ("first", "run", "second")
    assert facts.incomplete is False


def test_script_constructors_are_calls_with_their_existing_binding(tmp_path: Path) -> None:
    # Arrange
    index = committed(
        tmp_path,
        {
            "src/memory.ts": "export class MemoryAdapter {}\n",
            "src/build.ts": (
                'import { MemoryAdapter } from "./memory";\n'
                "export function build() { return new MemoryAdapter(); }\n"
            ),
        },
    )
    build = index.find_definition("build")[0]

    # Act
    edge = next(edge for edge in index.callee_edges(build) if edge.name == "MemoryAdapter")

    # Assert
    assert edge.binding.status == "resolved"
    assert edge.binding.target == Span("src/memory.ts", 1, 1, "MemoryAdapter")


@pytest.mark.parametrize(
    ("files", "name", "file"),
    [
        (
            {
                "app/plugins.py": "def order_created(order):\n    return order\n",
                "app/orders.py": (
                    "def place_order(manager, order, call_event):\n"
                    "    call_event(manager.order_created, order)\n"
                ),
            },
            "order_created",
            "app/orders.py",
        ),
        (
            {
                "src/plugins.ts": "export function orderCreated(order) { return order; }\n",
                "src/orders.ts": (
                    "export function placeOrder(manager, order, callEvent) {\n"
                    "  callEvent(manager.orderCreated, order);\n"
                    "}\n"
                ),
            },
            "orderCreated",
            "src/orders.ts",
        ),
    ],
)
def test_a_bound_member_passed_as_an_argument_uses_the_member_name(
    tmp_path: Path, files: dict[str, str], name: str, file: str
) -> None:
    # Arrange
    index = committed(tmp_path, files)

    # Act
    references = index.find_references(name)

    # Assert
    assert [(reference.file, reference.role) for reference in references] == [(file, "argument")]


def test_an_unsupported_flow_construct_keeps_its_file_incomplete(tmp_path: Path) -> None:
    """`export opaque type` degrades to a grammar ERROR: the file must stay honestly unparsed while
    the methods recovery did keep still count."""
    # Arrange
    index = a_root_tag_scope(tmp_path)

    # Act
    names = {span.name for span in index.functions_in("src/native/RootTag.js")}

    # Assert
    assert "createRootTag" in names
    assert index.unparsed_files == {"src/native/RootTag.js"}


# Parse Server's shape: Babel strips Flow types from every file, so most of its typed files carry no
# `@flow` pragma (src/Config.js, src/triggers.js and src/LiveQuery/Subscription.js at 8.6.95).
UNMARKED_FLOW = {
    "src/Config.js": (
        "import AppCache from './cache';\n"
        "import type { Subscription } from './Subscription';\n"
        "\n"
        "export class Config {\n"
        "  static get(applicationId: string, mount: string) {\n"
        "    return AppCache.get(applicationId);\n"
        "  }\n"
        "}\n"
    ),
    "src/triggers.js": (
        "export function triggerExists(className: string, type: string, applicationId: string): boolean {\n"
        "  return getTrigger(className, type, applicationId) != undefined;\n"
        "}\n"
        "\n"
        "export function getTrigger(className, type, applicationId) {\n"
        "  return null;\n"
        "}\n"
    ),
    "src/Subscription.js": (
        "export type QueryData = { [attr: string]: any };\n"
        "\n"
        "export class Subscription {\n"
        "  // It is query condition eg query.where\n"
        "  query: QueryData;\n"
        "  className: string;\n"
        "\n"
        "  hasSubscribingClient(clientId: number): boolean {\n"
        "    return true;\n"
        "  }\n"
        "}\n"
    ),
}


def test_javascript_with_flow_types_and_no_pragma_parses_fully(tmp_path: Path) -> None:
    # Arrange
    index = committed(tmp_path, UNMARKED_FLOW)

    # Act
    names = {file: {span.name for span in index.symbols_in(file)} for file in UNMARKED_FLOW}

    # Assert
    assert index.unparsed_files == set()
    assert names == {
        "src/Config.js": {"Config", "get"},
        "src/triggers.js": {"triggerExists", "getTrigger"},
        "src/Subscription.js": {"Subscription", "hasSubscribingClient"},
    }
    assert [site.file for site in index.find_callers("getTrigger")] == ["src/triggers.js"]


def test_javascript_the_flow_reading_reads_worse_keeps_its_own_facts(tmp_path: Path) -> None:
    # Arrange: a syntax error no grammar reads, and a variable named `as`, which minified JavaScript
    # uses and the flow reading cannot read
    source = (
        "export function count(n) {\n  let as = n;\n  return as;\n}\n\n"
        "export function broken() {\n  return 1 +* 2;\n}\n"
    )
    index = committed(tmp_path, {"src/broken.js": source})

    # Act
    [facts] = index.facts_in_files(["src/broken.js"]).values()

    # Assert
    assert index.unparsed_files == {"src/broken.js"}
    assert facts.unparsed_lines == ((7, 7),)
    assert [span.name for span in index.functions_in("src/broken.js")] == ["count", "broken"]


def test_javascript_both_readings_leave_equally_unread_stays_javascript(tmp_path: Path) -> None:
    # Arrange: a syntax error each grammar leaves one line unread at
    source = {"src/broken.js": "export function broken() {\n  return 1 +* 2;\n}\n"}
    write_files(tmp_path, source)

    # Act
    facts = scan_facts(read_files(tmp_path, source), tmp_path, Unparsed())

    # Assert
    assert (facts["src/broken.js"].language, facts["src/broken.js"].unparsed_lines) == (
        "javascript",
        ((2, 2),),
    )


def test_comments_in_flow_typed_javascript_without_the_pragma_are_read_as_flow(
    tmp_path: Path, ast_grep_runs
) -> None:
    # Arrange: on Parse Server's src/GraphQL/ParseGraphQLSchema.js the JavaScript grammar swallowed
    # lines 65 to 500 and 7 of its 24 comments; the comment scan reads a file as its facts were read
    index = committed(tmp_path, UNMARKED_FLOW)

    # Act
    found = find_comments(index, ["src/Subscription.js"])

    # Assert
    comment_runs = [config for rule, config, _ in ast_grep_runs if rule == "id: comment"]
    assert comment_runs == [FLOW_SGCONFIG]
    assert [block.span.start for block in found.kept] == [4]
    assert found.refused_files == {}


def test_whether_a_file_is_flow_is_judged_by_its_first_read_bytes(tmp_path: Path) -> None:
    # Arrange: the index first read the pragma; the disk now holds plain JavaScript
    (tmp_path / "typed.js").write_text("export function typed(value) {\n  return value;\n}\n")
    first_read = {"typed.js": b"// @flow\nexport function typed(value) {\n  return value;\n}\n"}

    # Act
    facts = scan_facts(first_read, tmp_path, Unparsed())

    # Assert
    assert facts["typed.js"].language == FLOW_LANGUAGE


BROKEN_FLOW = "// @flow\nexport class Broken {\n  find(a: string:\n"


@pytest.mark.parametrize(
    ("files", "name", "status", "hiding"),
    [
        pytest.param(
            {"src/native/RootTag.js": ROOT_TAG, "src/native/show.js": ROOT_TAG_CALLER},
            "createRootTag",
            "resolved",
            None,
            id="a-definition-recovered-outside-the-unread-lines-binds",
        ),
        pytest.param(
            {
                "src/broken.js": BROKEN_FLOW,
                "src/use.js": "import { find } from './broken';\n\n"
                "export function use() {\n  return find('a');\n}\n",
            },
            "find",
            "unknown",
            "src/broken.js",
            id="a-name-on-an-unread-line-stays-unknown",
        ),
        pytest.param(
            {
                "src/broken.js": BROKEN_FLOW,
                "src/use.js": "import { find as locate } from './broken';\n\n"
                "export function use() {\n  return locate('a');\n}\n",
            },
            "locate",
            "unknown",
            "src/broken.js",
            id="an-alias-of-a-name-on-an-unread-line-stays-unknown",
        ),
        pytest.param(
            {"src/broken.js": BROKEN_FLOW, "src/use.js": "export function use() {\n  return missing();\n}\n"},
            "missing",
            "unresolved",
            None,
            id="a-name-no-unread-line-mentions-is-unresolved",
        ),
    ],
)
def test_a_partly_recovered_file_leaves_unknown_only_the_names_its_unread_lines_mention(
    tmp_path: Path, files: dict[str, str], name: str, status: str, hiding: str | None
) -> None:
    """Code the grammar swallowed is unknown, not absent, but a definition names what it defines:
    lines that never mention a name cannot hold its definition."""
    # Arrange
    index = committed(tmp_path, files)
    assert len(index.unparsed_files) == 1

    # Act
    [site] = index.find_callers(name)

    # Assert
    assert site.binding.status == status, site.binding.reason
    if hiding is not None:
        assert hiding in site.binding.reason


def test_a_malformed_flow_file_still_counts_as_unparsed(tmp_path: Path) -> None:
    # Arrange
    index = committed(tmp_path, {"src/broken.js": "// @flow\nexport class Broken {\n  find(a: string:\n"})

    # Act and assert
    assert index.unparsed_files == {"src/broken.js"}


def test_a_search_over_a_scope_with_grammar_errors_never_reports_nothing_left(tmp_path: Path) -> None:
    # Arrange
    index = a_root_tag_scope(tmp_path)
    judge = Judge(ScriptedJevClient(nouls=lambda question_id, question, state: 0.05))
    start = [place_for_line(index, "src/native/show.js", 4, "start")]

    # Act
    result = find_code(index, judge, "where the root tag is made", start)

    # Assert
    assert result.outcome == Outcome.SCOPE_INCOMPLETE
    assert result.unparsed_files == {"src/native/RootTag.js"}
    assert result.history.steps[-1].judgments["unparsed_files"] == ["src/native/RootTag.js"]


def test_an_empty_search_that_never_reached_every_file_says_so_without_parsing_the_rest(
    tmp_path: Path, ast_grep_runs
) -> None:
    # Arrange
    index = committed(
        tmp_path,
        {
            "app/start.py": "def start():\n    return 1\n",
            "app/billing.py": "def bill():\n    return 2\n",
            "app/mail.py": "def send():\n    return 3\n",
        },
    )
    judge = Judge(ScriptedJevClient(nouls=lambda question_id, question, state: 0.05))
    start = [place_for_line(index, "app/start.py", 2, "start")]

    # Act
    result = find_code(index, judge, "where an order is shipped", start, moves={})

    # Assert
    parsed = {file for _rule, _config, files in ast_grep_runs for file in files}
    assert parsed == {"app/start.py"}
    assert result.outcome == Outcome.SCOPE_INCOMPLETE
    assert result.parser_scans_pending == ("facts",)
    assert (result.files_judged, result.files_read_only, result.files_never_reached) == (1, 0, 2)


def test_an_empty_search_that_parsed_every_file_reports_nothing_left(tmp_path: Path) -> None:
    # Arrange
    index = committed(tmp_path, {"app/start.py": "def start():\n    return 1\n"})
    judge = Judge(ScriptedJevClient(nouls=lambda question_id, question, state: 0.05))
    start = [place_for_line(index, "app/start.py", 2, "start")]

    # Act
    result = find_code(index, judge, "where an order is shipped", start, moves={})

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert result.parser_scans_pending == ()
    assert (result.files_judged, result.files_read, result.code_files) == (1, 1, 1)


def test_reading_the_parsed_files_receipt_parses_nothing(tmp_path: Path, ast_grep_runs) -> None:
    # Arrange
    index = committed(
        tmp_path, {"app/a.py": "def a():\n    return 1\n", "app/b.py": "def b():\n    return 2\n"}
    )
    index.functions_in("app/a.py")
    runs_before = len(ast_grep_runs)

    # Act
    parsed = index.parsed_files

    # Assert
    assert parsed == {"app/a.py"}
    assert len(ast_grep_runs) == runs_before


RECEIVERS = {
    "app/clients.ts": "export function load(client, cfg, store) {\n"
    '  client("api-key-literal").fetch(1);\n'
    '  cfg["token-literal"].get(2);\n'
    "  `template-${store}`.trim();\n"
    "  this.store.save(3);\n"
    "  store?.rows.push(4);\n"
    "  run(this.handler, cfg.read);\n"
    "}\n",
    "app/model.py": "class Model:\n    def save(self):\n        self.items.append(1)\n"
    '        super().save()\n        open("secret-path").read()\n',
}


def test_a_receiver_is_kept_only_as_a_plain_chain_of_names(tmp_path: Path) -> None:
    # Arrange
    commit_files(tmp_path, RECEIVERS)

    # Act
    facts = scan_facts(read_files(tmp_path, sorted(RECEIVERS)), tmp_path, Unparsed())

    # Assert
    calls = {(call.name, call.receiver) for fact in facts.values() for call in fact.calls}
    references = {(ref.name, ref.receiver) for fact in facts.values() for ref in fact.references}
    assert {
        ("fetch", OPAQUE_RECEIVER),
        ("get", OPAQUE_RECEIVER),
        ("trim", OPAQUE_RECEIVER),
        ("save", "this.store"),
        ("push", "store.rows"),
        ("append", "self.items"),
        ("read", OPAQUE_RECEIVER),
    } <= calls
    assert {("handler", "this"), ("read", "cfg")} <= references
    receivers = " ".join(str(call.receiver) for fact in facts.values() for call in fact.calls)
    assert "literal" not in receivers and "secret" not in receivers and "template" not in receivers


def test_a_parsed_file_that_vanished_still_counts_as_read_and_is_listed_unavailable(tmp_path: Path) -> None:
    # Arrange: its facts come from the bytes read before it vanished, which slices keep
    index = committed(
        tmp_path, {"app/a.py": "def a():\n    return 1\n", "app/b.py": "def b():\n    return 2\n"}
    )
    index.functions_in("app/a.py")
    (tmp_path / "app" / "a.py").unlink()

    # Act
    pending = index.parser_scans_pending

    # Assert
    assert pending == ("facts",)
    assert "app/a.py" in index.parsed_files
    assert "app/a.py" in index.unavailable_files
