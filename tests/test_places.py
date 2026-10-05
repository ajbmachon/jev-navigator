"""The places the search can open and the neighbours each move lists for opened code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.places import (
    MAX_DEFINITION_LINES,
    MAX_KEY_HITS,
    MOVES,
    Move,
    Place,
    function_place,
    neighbours,
    neighbours_and_omissions,
    place_for_line,
    range_place,
    restored_signature,
    window_place,
)
from jev_navigator.index.bindings import Binding, BindingStatus
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import CodeSlice, Span
from jev_navigator.judgments.relations import key_mention


def committed_index(root: Path, files: Mapping[str, str]) -> CodeIndex:
    commit_files(root, files)
    return CodeIndex(root, list(files))


def neighbour_signatures(index: CodeIndex, name: str) -> dict[str, str]:
    opened = index.read_slice(index.find_definition(name)[0])
    return {place.key: place.signature for place in neighbours(index, opened)}


def test_code_reached_only_through_a_reference_is_offered_both_ways(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "jobs.py": "def send_invoice(order):\n    return order\n",
            "registry.py": "from jobs import send_invoice\n\n\ndef register_jobs(scheduler):\n"
            '    scheduler.add({"invoice": send_invoice})\n',
        },
    )
    send_invoice, register_jobs = (
        index.find_definition("send_invoice")[0],
        index.find_definition("register_jobs")[0],
    )

    # Act
    from_registry = neighbour_signatures(index, "register_jobs")
    from_job = neighbour_signatures(index, "send_invoice")

    # Assert
    assert "passed on by register_jobs as collection" in from_registry[send_invoice.key]
    assert "refers to send_invoice as collection" in from_job[register_jobs.key]


def test_the_other_functions_of_the_opened_file_are_offered_whole(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {"orders.py": "def place(order):\n    return order\n\n\ndef refund(order):\n    return None\n"},
    )
    refund = index.find_definition("refund")[0]

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    assert "in the same file as place" in offered[refund.key]


def test_the_lines_before_an_opened_slice_are_offered(tmp_path: Path) -> None:
    # Arrange
    header = "".join(f"SETTING_{number} = {number}\n" for number in range(30))
    index = committed_index(tmp_path, {"orders.py": header + "\n\ndef place(order):\n    return order\n"})

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    before = [signature for signature in offered.values() if "the lines before" in signature]
    assert len(before) == 1


def test_windows_chosen_by_position_are_labelled_by_their_range_and_first_code_line(
    tmp_path: Path,
) -> None:
    # Arrange
    header = "import os\n\n\n" + "".join(f"SETTING_{number} = {number}\n" for number in range(30))
    after = "\n\nLIMIT = 5\n" + "".join(f"OTHER_{number} = {number}\n" for number in range(50))
    index = committed_index(tmp_path, {"orders.py": header + "def place(order):\n    return order\n" + after})
    opened = index.find_definition("place")[0]

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    before = next(signature for signature in offered.values() if "the lines before" in signature)
    after_place = next(signature for signature in offered.values() if "the lines after" in signature)
    assert before == f"orders.py:1-{opened.start - 1} `import os` (the lines before {opened.key})"
    last = opened.end + 40
    assert after_place == f"orders.py:{opened.end + 1}-{last} `LIMIT = 5` (the lines after {opened.key})"


def test_the_lines_after_a_place_end_at_the_last_line_even_with_a_form_feed(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"orders.py": 'def place(order):\n    return "a\fb"\n\nLIMIT = 5\n'})

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    after_place = next(signature for signature in offered.values() if "the lines after" in signature)
    assert after_place.startswith("orders.py:3-4 `LIMIT = 5`")


def test_a_window_around_a_line_outside_any_definition_is_labelled_by_that_line(tmp_path: Path) -> None:
    # Arrange
    imports = "".join(f"import module_{number}\n" for number in range(14))
    index = committed_index(tmp_path, {"routes.py": imports + 'register("x-order-limit")\n' + "\n" * 20})

    # Act
    place = place_for_line(index, "routes.py", 15, 'mentions "x-order-limit"')

    # Assert
    assert place.signature == 'routes.py:5-25 line 15 `register("x-order-limit")` (mentions "x-order-limit")'


def test_quoted_keys_are_searched_across_the_scope_including_docs_and_config(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": 'def place(order):\n    return order.get("max_items_per_order")\n',
            "config.yaml": "limits:\n  max_items_per_order: 20\n",
            "README.md": "# Orders\n\nSet `max_items_per_order` to cap orders.\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    mentions = {
        key.split(":")[0]
        for key, signature in offered.items()
        if key_mention("max_items_per_order") in signature
    }
    assert mentions == {"config.yaml", "README.md"}


def test_a_quoted_key_matches_whole_names_only(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": 'def place(order):\n    return order.get("invoice_day")\n',
            "billing.py": "def send_invoice_day_reminder():\n    pass\n",
            "config.yaml": "invoice_day: 5\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    mentions = {
        key.split(":")[0] for key, signature in offered.items() if key_mention("invoice_day") in signature
    }
    assert mentions == {"config.yaml"}


def test_quoted_words_that_are_not_key_shaped_are_not_searched(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": 'def place(order):\n    return order.get("status"), open(order, "read")\n',
            "config.yaml": "status: open\nmode: read\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    assert [signature for signature in offered.values() if "mentions" in signature] == []


def test_rarer_keys_come_first_and_a_key_found_everywhere_is_skipped(tmp_path: Path) -> None:
    # Arrange
    everywhere = {f"notes/{number}.md": "see `app.common`\n" for number in range(MAX_KEY_HITS + 1)}
    index = committed_index(
        tmp_path,
        {
            **everywhere,
            "orders.py": 'def place(order):\n    return order["app.common"], order["app.twice"], '
            'order["app.once"]\n',
            "a.yaml": "app.twice: 1\n",
            "b.yaml": "app.twice: 2\n",
            "c.yaml": "app.once: 3\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    mentioned = [signature.split("mentions ")[1] for signature in offered.values() if "mentions" in signature]
    assert mentioned == ["`app.once`)", "`app.twice`)", "`app.twice`)"]


def test_the_caller_chooses_which_moves_list_neighbours(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": (
                "def place(order):\n    return refund(order)\n\n\ndef refund(order):\n    return None\n"
            )
        },
    )
    opened = index.read_slice(index.find_definition("place")[0])

    def always_the_config(index: CodeIndex, code: CodeSlice) -> list[Place]:
        return [window_place(index, "orders.py", 1, "a custom move")]

    # Act
    same_file_only = neighbours(index, opened, moves={"same_file": MOVES["same_file"]})
    custom = neighbours(index, opened, moves={"custom": always_the_config})

    # Assert
    assert [place.signature.split(" (")[-1] for place in same_file_only] == ["in the same file as place)"]
    assert [place.signature.endswith("(a custom move)") for place in custom] == [True]


def test_the_default_moves_cannot_be_changed_by_a_caller() -> None:
    # Act and assert
    with pytest.raises(TypeError):
        MOVES["always_open_this"] = lambda index, opened: []  # type: ignore[index]


def offered_from(index: CodeIndex, file: str, line: int, per_kind: int = 8) -> list[Place]:
    opened = place_for_line(index, file, line, "start").open()
    return neighbours(index, opened, per_kind)


BANNER = "/*!\n * library\n * MIT Licensed\n */\n\n'use strict';\n\n"


@pytest.mark.parametrize(
    ("files", "opened_at", "offered", "not_offered"),
    [
        pytest.param(
            {
                "index.js": BANNER + "module.exports = require('./lib/app');\n",
                "lib/app.js": BANNER
                + "var proto = {};\n\nmodule.exports = function createApplication() {};\n",
            },
            ("index.js", 8),
            "`var proto = {};` (start of a module imported by index.js)",
            None,
            id="a-required-module-opens-at-its-first-line-of-code",
        ),
        pytest.param(
            {
                "src/jwt/index.ts": "export { verify } from './jwt'\n\n"
                "declare module '..' {\n  interface Variables {}\n}\n",
                "src/jwt/jwt.ts": "export const verify = (token: string) => token\n"
                "export const sign = (payload: string) => payload\n",
            },
            ("src/jwt/index.ts", 4),
            "src/jwt/jwt.ts:1 `export const verify = (token: string) => token` "
            "(imported by src/jwt/index.ts)",
            "export const sign",
            id="a-re-exported-name-opens-its-definition-and-only-it",
        ),
        pytest.param(
            {
                "flask/__init__.py": '"""The package."""\n\nfrom .app import Flask as Flask\n',
                "flask/app.py": "class Flask:\n    def run(self):\n        return self\n",
            },
            ("flask/__init__.py", 3),
            "flask/app.py:1 `class Flask:` (imported by flask/__init__.py)",
            None,
            id="a-python-import-opens-the-imported-class",
        ),
        pytest.param(
            {
                "package.json": '{"imports": {"#money": "./src/money.js"}}',
                "src/money.js": "export function cents() {\n  return 4;\n}\n",
                "src/money.d.ts": "export declare function cents(): number;\n",
                "src/index.js": 'export { cents } from "#money";\n',
            },
            ("src/index.js", 1),
            "candidate: repository package.json mapping for #money, a declaration file",
            None,
            id="a-declaration-file-beside-its-javascript-is-only-a-candidate",
        ),
        pytest.param(
            {
                "app.js": "var helper = require('./helper');\n\nfunction handle() {\n  return 1;\n}\n",
                "helper.js": "module.exports = function helper() {};\n",
            },
            ("app.js", 4),
            None,
            "imported by",
            id="a-function-leaves-its-files-imports-to-module-level-code",
        ),
    ],
)
def test_module_level_code_offers_what_its_file_imports(
    tmp_path: Path,
    files: dict[str, str],
    opened_at: tuple[str, int],
    offered: str | None,
    not_offered: str | None,
) -> None:
    # Arrange
    index = committed_index(tmp_path, files)

    # Act
    signatures = [place.signature for place in offered_from(index, *opened_at)]

    # Assert
    if offered is not None:
        assert any(offered in signature for signature in signatures), signatures
    if not_offered is not None:
        assert not any(not_offered in signature for signature in signatures), signatures


def test_a_line_on_a_class_opens_the_class_so_code_naming_it_is_offered(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "store.py": "from typing import Protocol\n\n\nclass Store(Protocol):\n"
            "    def save(self, record) -> None: ...\n",
            "phases.py": "from store import Store\n\n\ndef export_phase(store: Store, records):\n"
            "    for record in records:\n        store.save(record)\n",
        },
    )

    # Act
    place = place_for_line(index, "store.py", 4, "start")
    offered = {candidate.key: candidate.signature for candidate in offered_from(index, "store.py", 4)}

    # Assert
    assert place.open().span.name == "Store"
    assert "refers to Store as type" in offered["phases.py:4-6"]


def test_a_line_on_a_module_constant_opens_the_constant_so_code_naming_it_is_offered(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "limits.py": 'BLOCKED_TOOLS = frozenset(\n    {"shell", "write"}\n)\n',
            "limits_test.py": "from limits import BLOCKED_TOOLS\n\n\n"
            "def test_blocked_tools_match_the_runtime():\n"
            '    runtime = {"shell", "write"}\n    assert BLOCKED_TOOLS == runtime\n',
        },
    )

    # Act
    place = place_for_line(index, "limits.py", 2, "start")
    offered = {candidate.key: candidate.signature for candidate in offered_from(index, "limits.py", 2)}

    # Assert
    assert place.open().span == index.find_definition("BLOCKED_TOOLS")[0]
    assert "refers to BLOCKED_TOOLS as condition" in offered["limits_test.py:4-6"]


def long_class_index(root: Path) -> CodeIndex:
    attributes = "".join(f"    FIELD_{number} = {number}\n" for number in range(MAX_DEFINITION_LINES))
    body = attributes.replace("    FIELD_60 = 60\n", '    FIELD_60 = build_field("sixty")\n')
    return committed_index(
        root,
        {
            "mutations.py": "from fields import build_field\n\n\nclass CreateOrder:\n" + body,
            "fields.py": "def build_field(name):\n    return name\n",
            "schema.py": "from mutations import CreateOrder\n\nMUTATIONS = [CreateOrder]\n",
        },
    )


def test_a_line_in_a_class_too_long_to_open_whole_opens_a_window_named_after_the_class(
    tmp_path: Path,
) -> None:
    # Arrange
    index = long_class_index(tmp_path)

    # Act
    opened = place_for_line(index, "mutations.py", 65, "start").open()

    # Assert
    assert opened.span == Span("mutations.py", 55, 75, "CreateOrder")


def test_moves_start_from_a_window_inside_a_long_class(tmp_path: Path) -> None:
    # Arrange
    index = long_class_index(tmp_path)

    # Act
    offered = {place.key: place.signature for place in offered_from(index, "mutations.py", 65)}

    # Assert
    assert "called by CreateOrder" in offered["fields.py:1-2"]
    assert "refers to CreateOrder as collection" in offered["schema.py:3-3"]


def test_uses_proven_to_reach_another_definition_of_the_name_are_not_offered(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "archive.py": "def handler(event):\n    return event\n",
            "jobs.py": "def handler(event):\n    return None\n",
            "wiring.py": "from jobs import handler\n\n\ndef run(event):\n    return handler(event)\n\n\n"
            "def wire(bus):\n    bus.on(handler)\n",
        },
    )
    handlers = {span.file: span for span in index.find_definition("handler")}
    moves = {name: MOVES[name] for name in ("callers", "referenced_by")}

    # Act
    from_archive = neighbours(index, index.read_slice(handlers["archive.py"]), moves=moves)
    from_jobs = neighbours(index, index.read_slice(handlers["jobs.py"]), moves=moves)

    # Assert
    assert from_archive == []
    assert [place.signature.split("` ")[1] for place in from_jobs] == [
        "(calls handler)",
        "(refers to handler as argument)",
    ]


def test_callees_called_from_few_places_come_first(tmp_path: Path) -> None:
    # Arrange
    helpers = "".join(f"def helper_{number}(value):\n    return value\n\n\n" for number in range(9))
    helper_calls = "".join(f"    helper_{number}(event)\n" for number in range(9))
    index = committed_index(
        tmp_path,
        {
            "helpers.py": helpers,
            "events.py": "def save_event(event):\n    return event\n",
            "handler.py": "def handle(event):\n" + helper_calls + "    save_event(event)\n\n\n"
            "def audit(event):\n" + helper_calls,
        },
    )

    # Act
    offered = neighbour_signatures(index, "handle")

    # Assert
    callees = [signature.split("`")[1] for signature in offered.values() if "called by handle" in signature]
    assert callees[0] == "def save_event(event):"
    assert len(callees) == 10


def test_a_callee_with_no_definition_is_never_counted_when_callees_are_ranked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: handle calls one defined helper and one name defined nowhere in scope
    index = committed_index(
        tmp_path,
        {
            "helpers.py": "def save_event(event):\n    return event\n",
            "handler.py": "def handle(event):\n    undefined_logger(event)\n    return save_event(event)\n",
        },
    )
    counted: list[str] = []
    real_count = index.call_site_count

    def recorded_count(name: str) -> int:
        counted.append(name)
        return real_count(name)

    monkeypatch.setattr(index, "call_site_count", recorded_count)

    # Act
    offered = neighbour_signatures(index, "handle")

    # Assert: it yields no place, so counting its call sites is wasted work
    assert any("def save_event(event):" in signature for signature in offered.values())
    assert "undefined_logger" not in counted


def test_an_anonymous_handler_offers_proven_callees_before_test_only_candidates(
    tmp_path: Path,
) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "src/orders.ts": "export function createOrder() { return 1; }\n",
            "src/routes.ts": (
                'import { createOrder } from "./orders";\n'
                "router.post('/orders', (request) => {\n"
                "  error(request);\n"
                "  return createOrder();\n"
                "});\n"
            ),
            "tests/cache.test.ts": "export function error(value) { return value; }\n",
        },
    )
    handler = next(span for span in index.functions_in("src/routes.ts") if span.name == "<anonymous>")
    opened = index.read_slice(handler)

    # Act
    offered = neighbours(index, opened, per_kind=1, moves={"callees": MOVES["callees"]})

    # Assert
    assert [place.key for place in offered] == ["src/orders.ts:1-1"]
    assert "called by src/routes.ts:" in offered[0].signature


def test_a_module_registration_window_offers_the_registered_function(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "src/auth.ts": "export function authenticate(request) { return request; }\n",
            "src/app.ts": ('import { authenticate } from "./auth";\napp.use(authenticate);\n'),
        },
    )

    # Act
    offered = offered_from(index, "src/app.ts", 2)

    # Assert
    authentication = next(place for place in offered if place.key == "src/auth.ts:1-1")
    assert "passed on by src/app.ts:1-2 as argument" in authentication.signature
    assert "candidate" not in authentication.signature


def test_a_condition_is_not_described_as_passing_a_name_on(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "cache.py": "CACHE_READY = True\n\n\ndef read():\n    if CACHE_READY:\n        return 1\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "read")

    # Assert
    assert not any("passed on by read as condition" in signature for signature in offered.values())


def test_an_anonymous_callback_offers_its_named_containing_function(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.ts": (
                "export function createOrder(database) {\n"
                "  return database.transaction(async (transaction) => {\n"
                "    return transaction.save();\n"
                "  });\n"
                "}\n"
            )
        },
    )
    callback = next(span for span in index.functions_in("orders.ts") if span.name == "<anonymous>")

    # Act
    offered = neighbours(
        index,
        index.read_slice(callback),
        moves={"same_file": MOVES["same_file"]},
    )

    # Assert
    assert [place.key for place in offered] == ["orders.ts:1-5"]
    assert "in the same file as orders.ts:" in offered[0].signature


def test_a_callback_inside_a_test_callback_offers_that_test_first(tmp_path: Path) -> None:
    """Test callbacks are anonymous, so a callback nested in one has no named container: the
    same-file move offers its nearest container, the test it belongs to, before the other tests."""
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.test.ts": (
                'it("rejects an empty cart", () => {\n'
                "  expect(() => placeOrder([])).toThrow();\n"
                "});\n"
                'it("accepts one item", () => {\n'
                "  expect(placeOrder([1])).toBe(1);\n"
                "});\n"
            )
        },
    )
    inner = next(span for span in index.functions_in("orders.test.ts") if span.start == span.end == 2)

    # Act
    offered = neighbours(index, index.read_slice(inner), moves={"same_file": MOVES["same_file"]})

    # Assert
    assert [place.key for place in offered] == ["orders.test.ts:1-3", "orders.test.ts:4-6"]


def test_a_constant_used_as_a_method_receiver_is_passed_on(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "redaction.py": 'import re\n\nSECRET_PATTERN = re.compile(r"key=\\w+")\n\n\n'
            'def redact(text):\n    return SECRET_PATTERN.sub("key=[hidden]", text)\n'
        },
    )

    # Act
    offered = neighbour_signatures(index, "redact")

    # Assert
    assert offered["redaction.py:3-3"].endswith("(passed on by redact as receiver)")


def numbered_functions(count: int) -> str:
    return "".join(
        f"def step_{number}(value):\n    return value + {number}\n\n\n" for number in range(1, count + 1)
    )


def test_the_other_functions_of_the_file_are_offered_nearest_first(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"steps.py": numbered_functions(12)})

    # Act
    offered = [
        place.signature for place in neighbours(index, index.read_slice(index.find_definition("step_10")[0]))
    ]

    # Assert
    same_file = [signature.split("`")[1] for signature in offered if "in the same file" in signature]
    assert same_file == [f"def step_{number}(value):" for number in (9, 11, 8, 12, 7, 6, 5, 4, 3, 2, 1)]


def test_a_nested_function_is_offered_only_as_part_of_its_function(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "steps.py": "def outer(value):\n    def inner():\n        return value\n\n"
            "    return inner()\n\n\ndef other(value):\n    return value\n"
        },
    )

    # Act
    offered = neighbour_signatures(index, "other")

    # Assert
    same_file = [signature for signature in offered.values() if "in the same file" in signature]
    assert [signature.split("`")[1] for signature in same_file] == ["def outer(value):"]


@pytest.mark.parametrize(
    "test_path",
    ["tests/helpers.py", "app/test_orders.py", "app/orders_test.py", "app/conftest.py", "spec/orders.py"],
)
def test_callers_in_test_files_come_after_the_other_callers(tmp_path: Path, test_path: str) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": "def place_order(order):\n    return order\n",
            test_path: "from orders import place_order\n\n\ndef check_order():\n    place_order({})\n",
            "zz/checkout.py": "from orders import place_order\n\n\n"
            "def checkout(order):\n    place_order(order)\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place_order")

    # Assert
    callers = [key.split(":")[0] for key, signature in offered.items() if "calls place_order" in signature]
    assert callers == ["zz/checkout.py", test_path]


def test_the_lines_before_and_after_stay_off_the_opened_code_in_a_short_file(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path, {"orders.py": "import os\n\ndef place(order):\n    return order\n\nLIMIT = 5\n"}
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    positional = {key for key, signature in offered.items() if "the lines" in signature}
    assert positional == {"orders.py:1-2", "orders.py:5-6"}


def a_move_offering(*places: Place) -> Move:
    return lambda index, opened: list(places)


@pytest.mark.parametrize("file", ["orders.py", "v~2/orders.py", "v:5~2/orders.py"])
def test_a_restored_place_rebuilds_the_signature_its_builder_gave(tmp_path: Path, file: str) -> None:
    # Arrange: a place from each builder, in a file whose path may hold a tilde, and a colon and a
    # number before it, as a window key does; the function and the window carry a name-match
    # binding, which their signatures mark.
    source = "import os\n\n\ndef place(order):\n    limit = os.environ['LIMIT']\n    return check(order)\n"
    index = committed_index(tmp_path, {file: source})
    name_match = Binding(BindingStatus.CANDIDATE, "same name in another file")
    places = [
        function_place(index, index.find_definition("place")[0], "calls check", binding=name_match),
        window_place(index, file, 5, "reads LIMIT", radius=2, binding=name_match),
        range_place(index, file, 1, 2, "the start of a co-changed file"),
    ]

    # Act
    rebuilt = [
        restored_signature(index, place.key, place.kind, place.open().span, place.relation, place.binding)
        for place in places
    ]

    # Assert
    assert rebuilt == [place.signature for place in places]
    assert all("`" in signature for signature in rebuilt)


def test_places_that_open_the_same_lines_are_offered_once(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path, {"routes.py": "".join(f"ROUTE_{number} = {number}\n" for number in range(30))}
    )
    opened = index.read_slice(Span("routes.py", 25, 30))
    window = window_place(index, "routes.py", 11, "mentions a key")
    same_lines = range_place(index, "routes.py", 1, 21, "the start of a co-changed file")

    # Act
    offered = neighbours(
        index, opened, moves={"keys": a_move_offering(window), "files": a_move_offering(same_lines)}
    )

    # Assert
    assert [place.key for place in offered] == [window.key]


def test_a_place_wholly_inside_the_opened_code_is_not_offered(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path, {"routes.py": "".join(f"ROUTE_{number} = {number}\n" for number in range(30))}
    )
    opened = index.read_slice(Span("routes.py", 1, 20))
    inside = range_place(index, "routes.py", 3, 5, "inside")
    overlapping = range_place(index, "routes.py", 15, 25, "overlapping")

    # Act
    offered = neighbours(index, opened, moves={"lines": a_move_offering(inside, overlapping)})

    # Assert
    assert [place.key for place in offered] == [overlapping.key]


def test_identical_code_in_two_files_stays_two_places(tmp_path: Path) -> None:
    # Arrange
    refund = "def refund(order):\n    return None\n"
    index = committed_index(
        tmp_path,
        {
            "orders.py": "def place(order):\n    return refund(order)\n",
            "billing.py": refund,
            "legacy_billing.py": refund,
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    assert {"billing.py:1-2", "legacy_billing.py:1-2"} <= set(offered)


def test_a_place_kept_by_an_earlier_move_does_not_use_a_later_moves_cap(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path, {"routes.py": "".join(f"ROUTE_{number} = {number}\n" for number in range(30))}
    )
    opened = index.read_slice(Span("routes.py", 1, 1))
    shared = range_place(index, "routes.py", 5, 6, "shared")
    only_later = range_place(index, "routes.py", 8, 9, "only later")

    # Act
    kept, omitted = neighbours_and_omissions(
        index, opened, 1, {"first": a_move_offering(shared), "later": a_move_offering(shared, only_later)}
    )

    # Assert
    assert [place.key for place in kept] == [shared.key, only_later.key]
    assert omitted == []


def test_document_window_navigation_retains_text_edges_without_syntax_scanning(tmp_path):
    index = committed_index(
        tmp_path,
        {
            "README.md": 'The setting is "policy.limit".\n',
            "policy.py": 'def check(settings):\n    return settings["policy.limit"]\n',
        },
    )
    opened = index.read_window("README.md", 1)
    offered = neighbours(index, opened)
    assert any(place.open().span.file == "policy.py" for place in offered)
    assert index.references_in(opened.span) == ()
    assert index.callee_edges(opened.span) == ()
    assert index.functions_in("README.md") == ()
    assert index.read_window("README.md", 1).text == 'The setting is "policy.limit".'
