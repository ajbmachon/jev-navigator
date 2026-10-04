from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from git_repos import commit_all, git, write_files

from jev_navigator.index.code_index import CodeIndex, ScopeTooWideError, UnsafePathError
from jev_navigator.index.spans import Span, TextHit


def test_parsed_fact_lookups_do_not_launch_repeated_text_searches(tmp_path, monkeypatch):
    from jev_navigator.index import tools

    (tmp_path / "owner.py").write_text("def target(value):\n    return value\n")
    (tmp_path / "caller.py").write_text(
        "from owner import target\ndef caller(value):\n    callback = target\n    return target(value)\n"
    )
    cache = tmp_path / "cache"
    for _ in range(2):  # The same contract holds for fresh parser facts and persisted facts.
        index = CodeIndex(tmp_path, ("owner.py", "caller.py"), fact_cache_dir=cache)
        index.functions_in_files(index.files)
        searches = []
        run = tools.run_command

        def observe(arguments, *args, searches=searches, run=run, **kwargs):
            if arguments[0] == tools.RIPGREP:
                searches.append(arguments)
            return run(arguments, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(tools, "run_command", observe)
            assert [span.file for span in index.find_definition("target")] == ["owner.py"]
            assert [(site.file, site.line) for site in index.find_callers("target")] == [("caller.py", 4)]
            assert any(ref.file == "caller.py" for ref in index.find_references("target"))
            assert index.find_definition("absent") == ()
        assert searches == [], "Already parsed names must not spawn new repository searches"


def test_functions_in_lists_python_and_typescript_functions_with_names(sample_index: CodeIndex) -> None:
    # Act
    python_names = [span.name for span in sample_index.functions_in("app/validation.py")]
    script_names = [span.name for span in sample_index.functions_in("web/handlers.ts")]

    # Assert
    assert python_names == ["validate_order", "check_limits", "noop"]
    assert script_names == ["handleOrder", "parseOrder"]


def test_enclosing_symbol_returns_the_innermost_function(sample_index: CodeIndex) -> None:
    # Act
    span = sample_index.enclosing_symbol("app/orders.py", 6)

    # Assert
    assert span == Span("app/orders.py", 5, 7, "place")


def test_enclosing_symbol_is_none_at_module_level(sample_index: CodeIndex) -> None:
    assert sample_index.enclosing_symbol("app/orders.py", 1) is None


def test_find_definition_finds_the_function_across_files(sample_index: CodeIndex) -> None:
    # Act
    definitions = sample_index.find_definition("check_limits")

    # Assert
    assert definitions == (Span("app/validation.py", 10, 12, "check_limits"),)


def test_find_callers_binds_plain_and_method_calls_to_their_enclosing_function(
    sample_index: CodeIndex,
) -> None:
    # Act
    callers = sample_index.find_callers("validate_order")
    script_callers = sample_index.find_callers("handleOrder")

    # Assert
    assert [(site.file, site.line, site.caller.name) for site in callers] == [("app/orders.py", 6, "place")]
    assert [(site.file, site.caller.name if site.caller else None) for site in script_callers] == [
        ("web/routes.ts", "<anonymous>")
    ]


def test_find_callers_ignores_the_definition_itself(sample_index: CodeIndex) -> None:
    assert sample_index.find_callers("check_limits")[0].caller.name == "validate_order"


def test_find_callees_lists_called_names_inside_a_function(sample_index: CodeIndex) -> None:
    # Arrange
    validate = sample_index.find_definition("validate_order")[0]

    # Act
    callees = sample_index.find_callees(validate)

    # Assert
    assert callees == ("ValueError", "check_limits")


def test_read_slice_and_window_return_exact_lines(sample_index: CodeIndex) -> None:
    # Arrange
    check = sample_index.find_definition("check_limits")[0]

    # Act
    whole_function = sample_index.read_slice(check)
    window = sample_index.read_window("app/validation.py", 11, radius=1)

    # Assert
    assert whole_function.text.splitlines()[0] == "def check_limits(order):"
    assert window.span == Span("app/validation.py", 10, 12)
    assert window.text.splitlines()[1].strip() == 'limit = read_setting("orders.max_items")'


def test_search_text_finds_string_keys_in_scope_files(sample_index: CodeIndex) -> None:
    # Act
    hits = sample_index.search_text("orders.max_items")

    # Assert
    assert [(hit.file, hit.line) for hit in hits] == [("app/validation.py", 1), ("app/validation.py", 11)]


def test_imports_and_dependents_resolve_to_scope_files(sample_index: CodeIndex) -> None:
    # Act
    python_imports = sample_index.imports("app/orders.py")
    script_dependents = sample_index.dependents("web/handlers.ts")

    # Assert
    assert python_imports == ("app/validation.py",)
    assert script_dependents == ("web/routes.ts",)


def test_co_changed_files_counts_commits_shared_with_the_file(sample_index: CodeIndex) -> None:
    # Act
    co_changed = sample_index.co_changed_files("app/orders.py")

    # Assert
    assert co_changed[0] == ("app/validation.py", 2)


def test_scope_wider_than_the_limit_is_refused(sample_repo: Path) -> None:
    with pytest.raises(ScopeTooWideError):
        CodeIndex.from_git(sample_repo, max_files=2)


def test_explicit_unbounded_scope_keeps_every_tracked_file(sample_repo: Path) -> None:
    index = CodeIndex.from_git(sample_repo, max_files=None)

    assert len(index.files) > 2


def test_paths_outside_the_scope_are_refused(sample_repo: Path) -> None:
    # Arrange
    index = CodeIndex.from_git(sample_repo, prefixes=("web/",))

    # Act and Assert
    assert index.find_definition("validate_order") == ()
    with pytest.raises(ValueError, match="outside the index scope"):
        index.read_window("app/orders.py", 1)


def test_every_slice_records_its_source_file_lines_commit_and_how_it_was_reached(
    sample_repo: Path,
) -> None:
    # Arrange
    (sample_repo / "web/handlers.ts").write_text("// changed in the worktree\n")
    index = CodeIndex.from_git(sample_repo)
    head = git(sample_repo, "rev-parse", "HEAD").strip()

    # Act
    committed = index.read_slice(index.find_definition("check_limits")[0], origin="callee of validate_order")
    edited = index.read_window("web/handlers.ts", 1, radius=0)

    # Assert
    assert committed.source() == {
        "file": "app/validation.py",
        "lines": [10, 12],
        "commit": head,
        "file_sha256": hashlib.sha256((sample_repo / "app/validation.py").read_bytes()).hexdigest(),
        "reached_by": "callee of validate_order",
    }
    assert edited.commit == f"{head}+worktree"


def test_find_definition_covers_classes_constants_and_module_assignments(sample_index: CodeIndex) -> None:
    # Act and Assert
    assert sample_index.find_definition("OrderService") == (Span("app/orders.py", 4, 7, "OrderService"),)
    assert sample_index.find_definition("LIMITS_KEY") == (Span("app/validation.py", 1, 1, "LIMITS_KEY"),)
    assert sample_index.find_definition("parseOrder") == (Span("web/handlers.ts", 6, 6, "parseOrder"),)


def test_at_commit_reads_the_old_version_without_touching_the_checkout(sample_repo: Path) -> None:
    # Arrange
    first_commit = git(sample_repo, "rev-list", "--max-parents=0", "HEAD").strip()
    checkout_before = (sample_repo / "app/validation.py").read_text()

    # Act
    old = CodeIndex.at_commit(sample_repo, first_commit, prefixes=("app/",))

    # Assert
    assert "web/routes.ts" not in old.files and "app/settings.py" not in old.files
    assert [span.name for span in old.functions_in("app/validation.py")] == ["validate_order", "check_limits"]
    assert old.read_slice(old.find_definition("check_limits")[0]).commit == first_commit
    assert set(old.co_changed_files("app/orders.py")) == {("app/__init__.py", 1), ("app/validation.py", 1)}
    assert (sample_repo / "app/validation.py").read_text() == checkout_before


def test_call_bindings_say_whether_the_target_is_proven(sample_index: CodeIndex) -> None:
    # Act
    imported = sample_index.find_callers("validate_order")[0]
    same_file = sample_index.find_callers("check_limits")[0]
    script_import = sample_index.find_callers("handleOrder")[0]

    # Assert
    assert (imported.binding.status, imported.binding.target.file) == ("resolved", "app/validation.py")
    assert same_file.binding.status == "resolved" and "same file" in same_file.binding.reason
    assert script_import.binding.status == "resolved"


def test_method_calls_and_unknown_names_are_not_claimed_as_resolved(sample_index: CodeIndex) -> None:
    # Arrange
    place = sample_index.find_definition("place")[0]

    # Act
    edges = {edge.name: edge.binding for edge in sample_index.callee_edges(place)}

    # Assert
    assert edges["validate_order"].status == "resolved"
    assert edges["save"].status == "unresolved" and "no definition" in edges["save"].reason
    assert sample_index.find_callers("read_setting")[0].binding.status == "unresolved"


def test_a_name_defined_elsewhere_without_an_import_is_only_a_candidate(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "a.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "b.py").write_text("def use():\n    return helper()\n")
    (tmp_path / "c.py").write_text(
        "class Box:\n    def helper(self):\n        return 2\n\n\ndef run(box):\n    return box.helper()\n"
    )
    index = CodeIndex(tmp_path, ["a.py", "b.py", "c.py"])

    # Act
    bindings = {site.file: site.binding for site in index.find_callers("helper")}

    # Assert
    assert bindings["b.py"].status == "candidate" and "no import" in bindings["b.py"].reason
    assert bindings["c.py"].status == "candidate" and "receiver" in bindings["c.py"].reason


def test_an_injected_binding_resolver_wins(sample_index: CodeIndex, sample_repo: Path) -> None:
    # Arrange
    from jev_navigator.index.bindings import Binding

    class CodeRelations:
        def resolve_call(self, file: str, line: int, name: str, receiver: str | None) -> Binding | None:
            return Binding("resolved", "code relation", Span("app/orders.py", 5, 7, "place"))

    index = CodeIndex.from_git(sample_repo, binding_resolver=CodeRelations())

    # Act
    site = index.find_callers("check_limits")[0]

    # Assert
    assert site.binding.reason == "code relation"


REGISTRY_PY = """\
from app.jobs import send_invoice


def notify():
    return send_invoice


@send_invoice
def decorated():
    pass


def register_jobs(scheduler):
    scheduler.add(send_invoice)
    scheduler.add(job=send_invoice)
    table = {"invoice": send_invoice}
    queue = [send_invoice]
    current = send_invoice
    send_invoice()
"""

REGISTRY_TS = """\
import { sendInvoice } from "./jobs";
export { sendInvoice };
export default sendInvoice;
const table = { invoice: sendInvoice, sendInvoice };
let current = sendInvoice;
current = sendInvoice;
register(sendInvoice);
sendInvoice();
"""

JOBS_PY = """\
def send_invoice(order):
    return order
"""


@pytest.fixture
def registry_index(tmp_path: Path) -> CodeIndex:
    (tmp_path / "app").mkdir()
    (tmp_path / "app/registry.py").write_text(REGISTRY_PY)
    (tmp_path / "app/jobs.py").write_text(JOBS_PY)
    (tmp_path / "app/registry.ts").write_text(REGISTRY_TS)
    return CodeIndex(tmp_path, ["app/registry.py", "app/jobs.py", "app/registry.ts"])


def test_find_references_lists_non_call_usages_with_their_role(registry_index: CodeIndex) -> None:
    # Act
    python = [(ref.line, ref.role) for ref in registry_index.find_references("send_invoice")]
    script = [(ref.line, ref.role) for ref in registry_index.find_references("sendInvoice")]

    # Assert
    assert python == [
        (5, "return"),
        (8, "decorator"),
        (14, "argument"),
        (15, "argument"),
        (16, "collection"),
        (17, "collection"),
        (18, "assignment"),
    ]
    assert script == [
        (2, "export"),
        (3, "export"),
        (4, "collection"),
        (5, "assignment"),
        (6, "assignment"),
        (7, "argument"),
    ]


def test_references_carry_the_enclosing_function_and_a_binding(registry_index: CodeIndex) -> None:
    # Act
    reference = registry_index.find_references("send_invoice")[2]

    # Assert
    assert reference.holder.name == "register_jobs"
    assert reference.binding.status == "resolved"
    assert reference.binding.target == Span("app/jobs.py", 1, 2, "send_invoice")


def test_references_in_lists_the_names_a_function_passes_on_without_calling(
    registry_index: CodeIndex,
) -> None:
    # Arrange
    register_jobs = registry_index.find_definition("register_jobs")[0]

    # Act
    references = registry_index.references_in(register_jobs)

    # Assert
    assert {(ref.name, ref.role) for ref in references} >= {
        ("send_invoice", "argument"),
        ("send_invoice", "collection"),
        ("send_invoice", "assignment"),
    }
    assert "scheduler" not in {ref.name for ref in references}


PASSED_MEMBER_PY = """\
def handler(event):
    return event


class Client:
    def send(self, bus):
        bus.on(self.handler)
        bus.on(handler)
        bus.on(handler, self.handler)
"""

PASSED_MEMBER_TS = """\
function handler(event) {
  return event;
}

class Client {
  send(bus) {
    bus.on(this.handler);
    bus.on(handler);
    bus.on(handler, this.handler);
  }
}
"""


@pytest.mark.parametrize(
    ("file", "source", "function"),
    [
        ("client.py", PASSED_MEMBER_PY, Span("client.py", 1, 2, "handler")),
        ("client.ts", PASSED_MEMBER_TS, Span("client.ts", 1, 3, "handler")),
    ],
    ids=["python", "typescript"],
)
def test_a_passed_member_is_a_candidate_while_a_passed_function_is_resolved(
    tmp_path: Path, file: str, source: str, function: Span
) -> None:
    """`bus.on(self.handler)` passes an attribute of `self`, not the function `handler` defined in the
    same file: it is bound like a method call on an unknown receiver. A line passing both stands as the
    plain name, as a plain call does for callers. Persisted facts keep the receiver too."""
    # Arrange
    (tmp_path / file).write_text(source)
    cache = tmp_path / "cache"

    for _ in range(2):  # Fresh parser facts, then the persisted ones.
        # Act
        index = CodeIndex(tmp_path, [file], fact_cache_dir=cache)
        references = index.find_references("handler")

        # Assert
        assert [(ref.line, ref.binding.status, ref.binding.target) for ref in references] == [
            (7, "candidate", None),
            (8, "resolved", function),
            (9, "resolved", function),
        ]


PASSED_MEMBER_CONSTANT_PY = """\
TIMEOUT = 5


class Client:
    def send(self, bus):
        bus.wait(self.TIMEOUT)
        bus.wait(TIMEOUT)
"""

PASSED_MEMBER_CONSTANT_TS = """\
const TIMEOUT = 5;

class Client {
  send(bus) {
    bus.wait(this.TIMEOUT);
    bus.wait(TIMEOUT);
  }
}
"""


@pytest.mark.parametrize(
    ("file", "source", "constant", "member_line"),
    [
        ("client.py", PASSED_MEMBER_CONSTANT_PY, Span("client.py", 1, 1, "TIMEOUT"), 6),
        ("client.ts", PASSED_MEMBER_CONSTANT_TS, Span("client.ts", 1, 1, "TIMEOUT"), 5),
    ],
    ids=["python", "typescript"],
)
def test_a_passed_member_is_a_candidate_while_a_passed_constant_is_resolved(
    tmp_path: Path, file: str, source: str, constant: Span, member_line: int
) -> None:
    """`bus.wait(self.TIMEOUT)` passes an attribute of `self`, not the module constant `TIMEOUT`: an
    argument can name a constant, but a member argument is bound like a method call on an unknown
    receiver. The bare `TIMEOUT` on the next line is proven by the same-file definition."""
    # Arrange
    (tmp_path / file).write_text(source)

    # Act
    references = CodeIndex(tmp_path, [file]).find_references("TIMEOUT")

    # Assert
    assert [(ref.line, ref.binding.status, ref.binding.target) for ref in references] == [
        (member_line, "candidate", None),
        (member_line + 1, "resolved", constant),
    ]


USES_PY = """\
from app.rules import ALLOWED, PATTERN, Store


class Checker:
    def check(self, store: Store, name) -> Store | None:
        PATTERN.match(name)
        if ALLOWED:
            return store
        assert name in ALLOWED
        blocked = not ALLOWED
        total = ALLOWED + 1
        return self.store
"""

USES_TS = """\
import { ALLOWED, PATTERN, Store } from "./rules";

interface Store { save(): void }

export function check(store: Store, names: Array<Store>): Store {
  PATTERN.test(store.name);
  if (ALLOWED) {}
  const same = ALLOWED === names;
  const blocked = !ALLOWED;
  const total = ALLOWED + 1;
  return ALLOWED ? store : store;
}
"""


@pytest.fixture
def uses_index(tmp_path: Path) -> CodeIndex:
    (tmp_path / "app").mkdir()
    (tmp_path / "app/uses.py").write_text(USES_PY)
    (tmp_path / "app/uses.ts").write_text(USES_TS)
    return CodeIndex(tmp_path, ["app/uses.py", "app/uses.ts"])


def test_a_name_used_as_a_receiver_a_type_or_in_a_condition_is_a_reference(uses_index: CodeIndex) -> None:
    # Act
    python = {
        name: [(ref.line, ref.role) for ref in uses_index.find_references(name) if ref.file == "app/uses.py"]
        for name in ("PATTERN", "ALLOWED", "Store")
    }
    script = {
        name: [(ref.line, ref.role) for ref in uses_index.find_references(name) if ref.file == "app/uses.ts"]
        for name in ("PATTERN", "ALLOWED", "Store")
    }

    # Assert
    assert python == {
        "PATTERN": [(6, "receiver")],
        "ALLOWED": [(7, "condition"), (9, "condition"), (10, "condition")],
        "Store": [(5, "type")],
    }
    assert script == {
        "PATTERN": [(6, "receiver")],
        "ALLOWED": [(7, "condition"), (8, "condition"), (9, "condition"), (11, "condition")],
        "Store": [(5, "type")],
    }


def test_self_and_arithmetic_operands_are_not_references(uses_index: CodeIndex) -> None:
    # Act
    references = uses_index.find_references("self") + uses_index.find_references("ALLOWED")

    # Assert
    assert [ref for ref in references if ref.name == "self"] == []
    assert 11 not in {ref.line for ref in references if ref.file == "app/uses.py"}
    assert 10 not in {ref.line for ref in references if ref.file == "app/uses.ts"}


def test_a_function_passes_on_the_names_on_its_first_line(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "transport.ts").write_text(
        "export interface Answer { body: string }\n\n"
        "export function send(url: string): Promise<Answer> {\n  return fetch(url);\n}\n"
    )
    index = CodeIndex(tmp_path, ["transport.ts"])

    # Act
    references = index.references_in(index.find_definition("send")[0])

    # Assert
    assert [(ref.name, ref.line, ref.role) for ref in references] == [("Answer", 3, "type")]


@pytest.mark.parametrize(
    ("files", "holder", "declaration"),
    [
        pytest.param(
            {
                "hmr.ts": "interface PropagationBoundary {\n  boundary: string\n}\n\n"
                "export function propagateUpdate(boundaries: PropagationBoundary[]): boolean {\n"
                "  return boundaries.length > 0\n}\n"
            },
            "propagateUpdate",
            Span("hmr.ts", 1, 3, "PropagationBoundary"),
            id="interface",
        ),
        pytest.param(
            {
                "modes.ts": 'export const Mode = { Full: "full" } as const;\n'
                "export type Mode = (typeof Mode)[keyof typeof Mode];\n\n"
                "export function reload(mode: Mode) {\n  return mode;\n}\n"
            },
            "reload",
            Span("modes.ts", 2, 2, "Mode"),
            id="type-alias-named-like-a-constant",
        ),
        pytest.param(
            {
                "items.py": 'from typing import TypeVar\n\nItem = TypeVar("Item")\n\n\n'
                "def first(items: list[Item]):\n    return items[0]\n"
            },
            "first",
            Span("items.py", 3, 3, "Item"),
            id="python-type-alias",
        ),
    ],
)
def test_a_type_reference_binds_to_the_declaration_it_names(
    tmp_path: Path, files: dict[str, str], holder: str, declaration: Span
) -> None:
    # Arrange
    write_files(tmp_path, files)
    index = CodeIndex(tmp_path, list(files))

    # Act
    references = index.references_in(index.find_definition(holder)[0])

    # Assert
    assert [(ref.name, ref.role, ref.binding.status, ref.binding.target) for ref in references] == [
        (declaration.name, "type", "resolved", declaration)
    ]


@pytest.mark.parametrize(
    ("files", "name", "expected"),
    [
        pytest.param(
            {
                "redaction.py": 'import re\n\nSECRET_PATTERN = re.compile(r"key=\\w+")\n\n\n'
                'def redact(text):\n    return SECRET_PATTERN.sub("key=[hidden]", text)\n'
            },
            "SECRET_PATTERN",
            [(7, "receiver", Span("redaction.py", 3, 3, "SECRET_PATTERN"))],
            id="receiver",
        ),
        pytest.param(
            {
                "limits.ts": "export const MAX_ITEMS = 50;\n\n"
                "export function clamp(items: string[]) {\n"
                "  if (items.length > MAX_ITEMS) {\n    return items.slice(0, MAX_ITEMS);\n  }\n"
                "  return items;\n}\n"
            },
            "MAX_ITEMS",
            [
                (4, "condition", Span("limits.ts", 1, 1, "MAX_ITEMS")),
                (5, "argument", Span("limits.ts", 1, 1, "MAX_ITEMS")),
            ],
            id="condition-and-argument",
        ),
        pytest.param(
            {
                "modes.ts": 'export const Mode = { Full: "full" } as const;\n'
                "export type Mode = (typeof Mode)[keyof typeof Mode];\n\n"
                "export function isFull(mode: Mode) {\n  return mode === Mode.Full;\n}\n"
            },
            "Mode",
            [(4, "type", Span("modes.ts", 2, 2, "Mode")), (5, "receiver", Span("modes.ts", 1, 1, "Mode"))],
            id="value-and-type-named-alike",
        ),
        pytest.param(
            {
                "cache.py": "import functools\n\ncached = functools.lru_cache(maxsize=None)\n\n\n"
                "@cached\ndef load(path):\n    return path\n"
            },
            "cached",
            [(6, "decorator", Span("cache.py", 3, 3, "cached"))],
            id="decorator",
        ),
        pytest.param(
            {"options.ts": "interface Options {\n  strict: boolean\n}\n\nexport { Options };\n"},
            "Options",
            [(5, "export", Span("options.ts", 1, 3, "Options"))],
            id="export-of-an-interface",
        ),
    ],
)
def test_a_non_call_reference_binds_to_the_declaration_it_names(
    tmp_path: Path, files: dict[str, str], name: str, expected: list[tuple[int, str, Span]]
) -> None:
    # Arrange
    write_files(tmp_path, files)
    index = CodeIndex(tmp_path, list(files))

    # Act
    references = index.find_references(name)

    # Assert
    assert [(ref.line, ref.role, ref.binding.status, ref.binding.target) for ref in references] == [
        (line, role, "resolved", declaration) for line, role, declaration in expected
    ]


@pytest.mark.parametrize(
    ("files", "name", "site", "declaration"),
    [
        pytest.param(
            {
                "dispatch.py": 'from handlers import make_handler\n\nhandle = make_handler("orders")\n\n\n'
                "def dispatch(event):\n    return handle(event)\n"
            },
            "handle",
            ("dispatch.py", 7),
            Span("dispatch.py", 3, 3, "handle"),
            id="same-file",
        ),
        pytest.param(
            {
                "client.ts": "export const request = createClient({ retries: 3 });\n",
                "orders.ts": 'import { request } from "./client";\n\n'
                'export function loadOrders() {\n  return request("/orders");\n}\n',
            },
            "request",
            ("orders.ts", 4),
            Span("client.ts", 1, 1, "request"),
            id="imported",
        ),
        pytest.param(
            {
                "css.ts": "function createCssContext() {\n"
                '  const Style = () => "style";\n  return { Style };\n}\n\n'
                "export const Style = createCssContext().Style;\n",
                "page.ts": 'import { Style } from "./css";\n\n'
                "export function page() {\n  return Style();\n}\n",
            },
            "Style",
            ("page.ts", 4),
            Span("css.ts", 6, 6, "Style"),
            id="imported-past-a-nested-function",
        ),
        pytest.param(
            {
                "compose.ts": "export const compose = <T>(value: T): T => {\n  return value;\n};\n",
                "app.ts": 'import { compose } from "./compose";\n\n'
                "export function run() {\n  return compose(1);\n}\n",
            },
            "compose",
            ("app.ts", 4),
            Span("compose.ts", 1, 3, "compose"),
            id="imported-generic-arrow",
        ),
    ],
)
def test_a_call_binds_to_the_module_constant_it_names(
    tmp_path: Path, files: dict[str, str], name: str, site: tuple[str, int], declaration: Span
) -> None:
    # Arrange
    write_files(tmp_path, files)
    index = CodeIndex(tmp_path, list(files))

    # Act
    callers = index.find_callers(name)

    # Assert
    assert [(call.file, call.line, call.binding.status, call.binding.target) for call in callers] == [
        (*site, "resolved", declaration)
    ]


def test_a_one_line_function_calls_what_its_first_line_calls(sample_index: CodeIndex) -> None:
    # Arrange
    parse_order = sample_index.find_definition("parseOrder")[0]

    # Act
    callees = sample_index.find_callees(parse_order)

    # Assert
    assert parse_order.start == parse_order.end
    assert callees == ("parse",)


def test_tracked_symbolic_links_stay_out_of_the_scope(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "skills/real").mkdir(parents=True)
    (tmp_path / "skills/real/tool.py").write_text('KEY = "shared.key"\n')
    (tmp_path / "app").mkdir()
    (tmp_path / "app/main.py").write_text('SETTING = "shared.key"\n')
    (tmp_path / "app/linked_dir").symlink_to("../skills/real", target_is_directory=True)
    (tmp_path / "app/linked_file.py").symlink_to("main.py")
    (tmp_path / "app/outside.py").symlink_to("/etc/hosts")
    commit_all(tmp_path)

    # Act
    working = CodeIndex.from_git(tmp_path, prefixes=("app/",))
    historical = CodeIndex.at_commit(tmp_path, "HEAD", prefixes=("app/",))

    # Assert
    assert working.files == historical.files == ("app/main.py",)
    assert [hit.file for hit in working.search_text("shared.key")] == ["app/main.py"]
    assert [hit.file for hit in historical.search_text("shared.key")] == ["app/main.py"]


def test_file_names_with_non_ascii_characters_enter_the_scope_as_they_are_on_disk(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "app").mkdir()
    (tmp_path / "app/größe.py").write_text("def groesse():\n    return 1\n")
    commit_all(tmp_path)
    (tmp_path / "app/größe.py").write_text("def groesse():\n    return 2\n")

    # Act
    working = CodeIndex.from_git(tmp_path, prefixes=("app/",))
    historical = CodeIndex.at_commit(tmp_path, "HEAD", prefixes=("app/",))

    # Assert
    assert working.files == historical.files == ("app/größe.py",)
    assert working.read_slice(working.find_definition("groesse")[0]).text.endswith("return 2")
    assert working._changed == frozenset({"app/größe.py"})


def test_an_index_at_a_commit_reads_a_file_whose_name_holds_a_newline(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "app").mkdir()
    (tmp_path / "app/line\nbreak.py").write_text("def split_name():\n    return 1\n")
    (tmp_path / "app/plain.py").write_text("def plain():\n    return 2\n")
    commit_all(tmp_path)

    # Act
    historical = CodeIndex.at_commit(tmp_path, "HEAD", prefixes=("app/",))

    # Assert
    assert historical.files == ("app/line\nbreak.py", "app/plain.py")
    assert (historical.root / "app/line\nbreak.py").read_text().endswith("return 1\n")
    assert (historical.root / "app/plain.py").read_text().endswith("return 2\n")


def test_search_text_reads_a_line_that_is_not_utf8_as_the_index_does(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "labels.py").write_bytes(b'LABEL = "caf\xe9"\nLIMIT = 5\n')
    index = CodeIndex(tmp_path, ["labels.py"])

    # Act
    hits = index.search_text("LABEL")

    # Assert
    assert hits == (TextHit("labels.py", 1, index.lines("labels.py")[0]),)
    assert hits[0].text == 'LABEL = "caf\ufffd"'


def test_working_directory_inventory_includes_outer_changes_and_excludes_nested_repositories(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "tracked.py").write_text("VALUE = 1\n")
    git(tmp_path, "add", "tracked.py")
    git(tmp_path, "commit", "-q", "-m", "tracked")
    (tmp_path / "tracked.py").write_text("VALUE = 2\n")
    (tmp_path / "untracked.py").write_text("UNTRACKED = True\n")
    (tmp_path / ".gitignore").write_text("ignored.py\n")
    (tmp_path / "ignored.py").write_text("IGNORED = True\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    git(nested, "init", "-q", "-b", "main")
    (nested / "duplicate.py").write_text("DUPLICATE = True\n")

    index = CodeIndex.from_directory(tmp_path)

    assert index.files == (".gitignore", "tracked.py", "untracked.py")


def test_non_git_directory_inventory_includes_untracked_files(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n")

    assert CodeIndex.from_directory(tmp_path).files == ("module.py",)


def test_search_text_reads_a_line_holding_a_unicode_line_separator(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "messages.js").write_text('const MESSAGE = "first\u2028second";\n')
    index = CodeIndex(tmp_path, ["messages.js"])

    # Act
    hits = index.search_text("MESSAGE")

    # Assert
    assert [(hit.file, hit.line) for hit in hits] == [("messages.js", 1)]


def test_co_changed_files_count_files_with_non_ascii_names(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "app").mkdir()
    (tmp_path / "app/größe.py").write_text("SIZE = 1\n")
    (tmp_path / "app/maß.py").write_text("MEASURE = 1\n")
    commit_all(tmp_path)
    index = CodeIndex.from_git(tmp_path, prefixes=("app/",))

    # Act
    co_changed = index.co_changed_files("app/größe.py")

    # Assert
    assert co_changed == (("app/maß.py", 1),)


def test_line_numbers_follow_newlines_only_like_the_parser(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "app.py").write_text('BANNER = "a\fb"\r\n\r\ndef second():\n    return 2\n')
    index = CodeIndex(tmp_path, ["app.py"])

    # Act
    second = index.find_definition("second")[0]

    # Assert
    assert (second.start, second.end) == (3, 4)
    assert index.read_slice(second).text == "def second():\n    return 2"


@pytest.mark.parametrize("scope_path", ["link.py", "linked_dir/inside.py", "../outside.py"])
def test_a_scope_path_that_leaves_the_root_is_refused(tmp_path: Path, scope_path: str) -> None:
    # Arrange
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "inside.py").write_text("OUTSIDE = True\n")
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.py").write_text("OUTSIDE = True\n")
    (root / "link.py").symlink_to(elsewhere / "inside.py")
    (root / "linked_dir").symlink_to(elsewhere, target_is_directory=True)

    # Act and assert
    with pytest.raises(UnsafePathError, match=scope_path.replace(".", r"\.")):
        CodeIndex(root, [scope_path])


def test_top_level_symbols_are_the_functions_and_classes_no_other_symbol_contains(
    sample_index: CodeIndex,
) -> None:
    top_level = [span.name for span in sample_index.top_level_symbols("app/orders.py")]

    assert top_level == ["OrderService", "cancel"]
    assert "place" in [span.name for span in sample_index.symbols_in("app/orders.py")]


def test_a_rendered_component_is_a_call_and_a_platform_element_is_not(tmp_path: Path) -> None:
    (tmp_path / "notices.tsx").write_text("export function LoadFailed() {\n  return <p>Not loaded</p>;\n}\n")
    (tmp_path / "basket.tsx").write_text(
        'import { LoadFailed } from "./notices";\n'
        "export function Basket() {\n"
        "  return <div><LoadFailed /></div>;\n"
        "}\n"
        "export function Page() {\n"
        "  return <ui.Frame><LoadFailed>x</LoadFailed></ui.Frame>;\n"
        "}\n"
    )
    index = CodeIndex(tmp_path, ("notices.tsx", "basket.tsx"), fact_cache_dir=tmp_path / "cache")

    calls = [(site.file, site.line, site.caller.name) for site in index.find_callers("LoadFailed")]

    assert calls == [("basket.tsx", 3, "Basket"), ("basket.tsx", 6, "Page")]
    assert [site.line for site in index.find_callers("Frame")] == [6]
    assert index.find_callers("div") == ()
