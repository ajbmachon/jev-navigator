from __future__ import annotations

import json
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.find_code import Outcome, find_code
from jev_navigator.directives.places import neighbours_and_omissions, place_for_line
from jev_navigator.index import tools
from jev_navigator.index.bindings import Binding
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.languages import has_flow_pragma
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

    def counted_rules(rules: str, files, cwd, config=None, *, refused):
        runs.append((rules.split("\n", 1)[0], config, list(files)))
        return original_rules(rules, files, cwd, config=config, refused=refused)

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

    def fail_parser(rules, files, cwd, config=None, *, refused):
        del rules, files, cwd, config, refused
        raise tools.ToolFailedError("ast-grep failed for a real tool reason")

    monkeypatch.setattr(tools, "ast_grep_rules", fail_parser)
    index = CodeIndex(tmp_path, ["module.py"], fact_cache_dir=tmp_path / "cache")

    with pytest.raises(tools.ToolFailedError, match="real tool reason"):
        index.functions_in("module.py")


def test_scan_facts_skips_unsupported_files_and_still_parses_supported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# notes\n")
    (tmp_path / "module.py").write_text("def greet(): return 1\n")

    empty = FileFacts(FileStructure((), (), ()), (), ())

    unsupported = scan_facts(["notes.md"], tmp_path, Unparsed())
    mixed = scan_facts(["module.py", "notes.md"], tmp_path, Unparsed())

    assert unsupported == {"notes.md": empty}
    assert mixed["module.py"].structure.functions == (Span("module.py", 1, 1, "greet"),)
    assert mixed["notes.md"] == empty


@pytest.mark.parametrize(
    ("file", "declaration"),
    [("module.ts", "const value{n} = {n};\n"), ("module.py", "value{n} = {n}\n")],
)
def test_a_module_declaration_is_printed_without_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: str, declaration: str
) -> None:
    """ast-grep prints every node a rule's relations matched. A rule asking whether a declaration sits
    in the program printed the whole file once per declaration, so the parser's output, and its
    memory, grew with declarations times file size: 1.2 MB of ordinary code peaked over 2.5 GB."""
    # Arrange
    source = "".join(declaration.format(n=n) for n in range(400))
    (tmp_path / file).write_text(source)
    printed: list[dict] = []
    original_rules = tools.ast_grep_rules

    def recorded_rules(*arguments, **options):
        for match in original_rules(*arguments, **options):
            printed.append(match)
            yield match

    monkeypatch.setattr(tools, "ast_grep_rules", recorded_rules)

    # Act
    facts = scan_facts([file], tmp_path, Unparsed())

    # Assert
    declarations = [match for match in printed if match["ruleId"] == "declaration"]
    assert len(facts[file].structure.declarations) == len(declarations) == 400
    assert all(len(json.dumps(match)) < len(source) for match in declarations)


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
            for call in scan_facts(["order.js"], tmp_path, Unparsed())["order.js"].calls
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
    """Declaration name nodes and specifier nodes carry the surface: default, wildcard, a multi-line
    list, two constants in one statement and a template-literal body are each handled by the
    parser, not by source-text scanning."""
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
    assert facts.export_names == ("first", "placeOrder", "refund", "run", "second")
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
    facts = scan_facts(sorted(RECEIVERS), tmp_path, Unparsed())

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
