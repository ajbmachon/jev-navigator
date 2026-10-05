"""Units: the functions, methods and top-level code a search judges, and the lines that name them.

Every test runs the real ast-grep parser over a real git repository, so each record below is what
the index reports for that source, not what this module assumes it reports. Each index gets its own
fact-cache directory, so nothing is read from facts another test cached.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pytest
from git_repos import commit_all, git, write_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.index.units import (
    Item,
    LineAnchor,
    RangeAnchor,
    Unit,
    UnitKind,
    UnresolvedAnchor,
    best_piece,
    items_to_judge,
    list_units,
    read_ranges,
    resolve_anchors,
    unit_score,
)
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.questions import serialized_chars

JEV_BOX = JEV_INPUT_LIMITS.box_chars
SMALL_BOX = 3_000

ROUTES = """\
from app.handlers import create, read

ROUTES = [
    ("/orders", create),
    ("/orders/<id>", read),
]


def create(request):
    return save(request)


class Settings:
    timeout = 30
    retries = {"orders": 3}

    def describe(self):
        return "settings"


app.register(ROUTES)
"""

ONLY_IMPORTS = """\
# Helpers the routes use.
from app.handlers import create
from app.models import (
    Order,
    Item,
)
import json


def first(items):
    return items[0]
"""

SCRIPT_IMPORTS = """\
/**
 * Wires the order handlers.
 */
import {
  handleOrder,
  parseOrder,
} from "./handlers";
import type { Order } from "./types";
// the router
const express = require("express");

export function wire(app) {
  app.post("/orders", handleOrder);
}
"""

BASKET = """\
class Basket:
    def add(self, item):
        return item

    def total(self):
        def helper(x):
            return x * 2
        return helper(2)
"""

ONE_LINE_NESTED = "function pick(items) { return items.map((item) => item.id); }\n"
ONE_LINE_METHODS = "export const pair = { first() { return 1; }, second() { return 2; } };\n"
PANEL = """\
"use client";
// A panel that opens.
import { useState } from "react";

export function Panel() {
  const [open] = useState(false);
  return open;
}
"""
VITE_CONFIG = """\
import { defineConfig } from "vite";

export default defineConfig(() => {
  return { base: "/" };
}
);
"""
MOUNT = """\
import express from "express";
express.Router().use("/api", require("./api"));
"""

TEST_FILE = """\
def test_create():
    assert create
"""
HELPERS = "def first():\n    return 1\n\n\ndef second():\n    return 2\n"
LAYERS = """\
def outer():
    def middle():
        def inner():
            return 1
        return inner()
    return middle()
"""

TABLE = (
    "TABLE = {\n"
    + "".join(f'    "k_{number}": {number},\n' for number in range(10, 100))
    + "}\n\n\ndef lookup(key):\n    return TABLE[key]\n\n\nEXTRA = [\n"
    + "".join(f'    "e_{number}",\n' for number in range(10, 40))
    + "]\n"
)
TABLE_BOX = 1_500


def _steps(count: int) -> str:
    return "".join(f"    step_{number} = order.value_{number}\n" for number in range(100, 100 + count))


BIG = "def big(order):\n" + _steps(147) + "    return order\n"
IMAGE = "def image(order):\n" + _steps(64) + '    data = "' + "A" * 5_000 + '"\n    return data\n'
LONG_BUT_SMALL = "def long_but_small(order):\n" + _steps(298) + "    return order\n"

SHOP = {
    "app/routes.py": ROUTES,
    "app/only_imports.py": ONLY_IMPORTS,
    "app/basket.py": BASKET,
    "app/big.py": BIG,
    "app/image.py": IMAGE,
    "app/long.py": LONG_BUT_SMALL,
    "app/table.py": TABLE,
    "app/helpers.py": HELPERS,
    "app/layers.py": LAYERS,
    "web/wire.ts": SCRIPT_IMPORTS,
    "web/pick.ts": ONE_LINE_NESTED,
    "web/pair.ts": ONE_LINE_METHODS,
    "web/mount.js": MOUNT,
    "web/panel.tsx": PANEL,
    "web/vite.config.ts": VITE_CONFIG,
    "tests/test_routes.py": TEST_FILE,
    "README.md": "# Shop\n",
    "config.json": '{"retries": 3}\n',
}


@pytest.fixture
def shop(tmp_path: Path) -> CodeIndex:
    root = tmp_path / "shop"
    write_files(root, SHOP)
    commit_all(root)
    return CodeIndex.from_git(root, fact_cache_dir=tmp_path / "facts")


def _units_by_id(index: CodeIndex, files: tuple[str, ...], box: int = JEV_BOX) -> dict:
    return {unit.id: unit for unit in list_units(index, files, box_chars=box).units}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _lines(source: str, start: int, end: int) -> str:
    return "\n".join(source.split("\n")[start - 1 : end])


def test_unit_record_carries_identity_kind_symbol_and_content_hash(sample_index: CodeIndex) -> None:
    # Act
    units = _units_by_id(sample_index, ("app/orders.py",))

    # Assert
    place = units["app/orders.py:5-7"]
    assert (place.path, place.start, place.end, place.ranges) == ("app/orders.py", 5, 7, ((5, 7),))
    assert (place.kind, place.symbol, place.language, place.test) == (
        UnitKind.METHOD,
        "OrderService.place",
        "python",
        False,
    )
    assert place.content_sha256 == _sha256(
        "    def place(self, order):\n        validate_order(order)\n        return self.store.save(order)"
    )
    assert place.revision == sample_index.commit
    assert place.pieces == ()
    assert place.nested_in is None
    cancel = units["app/orders.py:10-13"]
    assert (cancel.kind, cancel.symbol) == (UnitKind.FUNCTION, "cancel")


def test_an_unchanged_function_keeps_its_hash_across_commits(sample_repo: Path, tmp_path: Path) -> None:
    git(sample_repo, "checkout", "-q", "HEAD~1")
    before_index = CodeIndex.from_git(sample_repo, fact_cache_dir=tmp_path / "facts-before")
    before = _units_by_id(before_index, ("app/orders.py", "app/validation.py"))
    git(sample_repo, "checkout", "-q", "main")
    after_index = CodeIndex.from_git(sample_repo, fact_cache_dir=tmp_path / "facts-after")

    # Act
    after = _units_by_id(after_index, ("app/orders.py", "app/validation.py"))

    # Assert: orders.py changed between the commits, but not inside `place`, so only the file's
    # hash moved; `noop` changed its body, so its own hash moved.
    place = Span("app/orders.py", 5, 7)
    assert before_index.read_slice(place).file_sha256 != after_index.read_slice(place).file_sha256
    assert before["app/orders.py:5-7"].content_sha256 == after["app/orders.py:5-7"].content_sha256
    assert before["app/validation.py:4-7"].content_sha256 == after["app/validation.py:4-7"].content_sha256
    assert before["app/validation.py:15-16"].content_sha256 != after["app/validation.py:15-16"].content_sha256
    assert before["app/validation.py:15-16"].revision != after["app/validation.py:15-16"].revision


def test_a_function_larger_than_the_box_is_cut_into_pieces_of_at_most_60_lines(shop: CodeIndex) -> None:
    # Act
    big = _units_by_id(shop, ("app/big.py",), SMALL_BOX)["app/big.py:1-149"]

    # Assert: the pieces cover the function exactly, in order, with no overlap.
    assert [(piece.index, piece.start, piece.end) for piece in big.pieces] == [
        (0, 1, 60),
        (1, 61, 120),
        (2, 121, 149),
    ]
    assert all(piece.end - piece.start + 1 <= 60 for piece in big.pieces)
    assert [piece.content_sha256 for piece in big.pieces] == [
        _sha256(_lines(BIG, 1, 60)),
        _sha256(_lines(BIG, 61, 120)),
        _sha256(_lines(BIG, 121, 149)),
    ]
    assert big.content_sha256 == _sha256(_lines(BIG, 1, 149))
    assert not any(piece.too_large_to_judge for piece in big.pieces)
    assert [big.piece_id(piece) for piece in big.pieces] == [
        "app/big.py:1-149#p0",
        "app/big.py:1-149#p1",
        "app/big.py:1-149#p2",
    ]
    second = items_to_judge(big)[1]
    assert second == Item("app/big.py:1-149#p1", "app/big.py", ((61, 120),))
    assert read_ranges(shop, second.file, second.ranges) == _lines(BIG, 61, 120)


def test_a_piece_over_the_box_is_named_too_large_to_judge_and_left_out_of_the_judged_pieces(
    shop: CodeIndex,
) -> None:
    # Act
    image = _units_by_id(shop, ("app/image.py",), SMALL_BOX)["app/image.py:1-67"]

    # Assert: the piece holding the 5,000-character line keeps its range and its size.
    first, second = image.pieces
    assert (first.start, first.end, first.too_large_to_judge) == (1, 60, False)
    assert (second.start, second.end, second.too_large_to_judge) == (61, 67, True)
    assert second.chars > 5_000
    assert image.judged_pieces == (first,)
    assert image.too_large_pieces == (second,)
    assert items_to_judge(image) == (Item("app/image.py:1-67#p0", "app/image.py", ((1, 60),)),)


def test_a_unit_over_the_box_with_at_most_60_lines_is_one_piece_too_large_to_judge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    write_files(root, {"app/logo.py": 'def logo():\n    return "' + "B" * 4_000 + '"\n'})
    commit_all(root)
    index = CodeIndex.from_git(root, fact_cache_dir=tmp_path / "facts")

    # Act
    logo = _units_by_id(index, ("app/logo.py",), SMALL_BOX)["app/logo.py:1-2"]

    # Assert
    assert [(piece.start, piece.end, piece.too_large_to_judge) for piece in logo.pieces] == [(1, 2, True)]
    assert logo.judged_pieces == () and items_to_judge(logo) == ()


def test_a_unit_is_measured_in_escaped_json_like_every_request(tmp_path: Path) -> None:
    # Arrange: 1,600 Chinese characters are about 1,600 characters as text but 9,600 escaped.
    greeting = 'def greet():\n    return "' + "你好" * 800 + '"\n'
    root = tmp_path / "repo"
    write_files(root, {"app/greet.py": greeting})
    commit_all(root)
    index = CodeIndex.from_git(root, fact_cache_dir=tmp_path / "facts")

    # Act
    greet = _units_by_id(index, ("app/greet.py",), SMALL_BOX)["app/greet.py:1-2"]

    # Assert: the unit is over the box as a request spells it, so it is named too large to judge.
    (piece,) = greet.pieces
    assert piece.chars == serialized_chars(greeting.rstrip("\n")) > SMALL_BOX
    assert piece.too_large_to_judge


def test_a_long_function_that_fits_the_box_stays_one_unit(shop: CodeIndex) -> None:
    # Act
    long_unit = _units_by_id(shop, ("app/long.py",))["app/long.py:1-300"]

    # Assert: 300 lines, about 9,600 characters, far under the 76,800-character box.
    assert long_unit.pieces == ()
    assert items_to_judge(long_unit) == (Item("app/long.py:1-300", "app/long.py", ((1, 300),)),)
    assert read_ranges(shop, long_unit.path, long_unit.ranges) == _lines(LONG_BUT_SMALL, 1, 300)


def test_a_cut_function_scores_by_its_best_piece_and_keeps_piece_ranges(shop: CodeIndex) -> None:
    units = _units_by_id(shop, ("app/big.py", "app/routes.py"), SMALL_BOX)
    big = units["app/big.py:1-149"]
    create = units["app/routes.py:9-10"]
    first, second, third = big.pieces
    scores = {big.piece_id(first): 0.2, big.piece_id(second): 0.9, big.piece_id(third): 0.4, create.id: 0.7}

    # Act
    score = unit_score(big, scores)
    best = best_piece(big, scores)

    # Assert: the unit takes its best piece's score and names that piece's lines as the place to read.
    assert score == 0.9
    assert (best.start, best.end) == (61, 120)
    assert [(piece.start, piece.end) for piece in big.pieces] == [(1, 60), (61, 120), (121, 149)]
    assert unit_score(create, scores) == 0.7 and best_piece(create, scores) is None
    assert unit_score(big, {}) is None


def test_on_a_tie_the_earliest_piece_is_the_place_to_read(shop: CodeIndex) -> None:
    # Arrange: the second and third pieces share the best score
    big = _units_by_id(shop, ("app/big.py",), SMALL_BOX)["app/big.py:1-149"]
    first, second, third = big.pieces
    scores = {big.piece_id(first): 0.2, big.piece_id(second): 0.8, big.piece_id(third): 0.8}

    # Act
    best = best_piece(big, scores)

    # Assert
    assert (best.start, best.end) == (61, 120)


def test_round_zero_lists_outermost_units_and_every_line_of_code_is_in_one(shop: CodeIndex) -> None:
    source_files = tuple(file for file in shop.files if file.endswith((".py", ".ts", ".tsx", ".js")))

    # Act
    listed = list_units(shop, source_files, box_chars=JEV_BOX).units

    # Assert: a nested function's text is already inside its holder's, so it is not listed, and
    # every function the parser finds lies inside a listed unit.
    assert all(unit.nested_in is None for unit in listed)
    assert "app/basket.py:6-7" not in {unit.id for unit in listed}
    for function in (span for file in source_files for span in shop.functions_in(file)):
        assert any(
            unit.path == function.file and unit.start <= function.start and function.end <= unit.end
            for unit in listed
        ), function


def test_methods_are_qualified_by_their_class_and_hold_their_nested_functions(shop: CodeIndex) -> None:
    # Act
    units = _units_by_id(shop, ("app/basket.py",))

    # Assert: `helper`, nested in `Basket.total`, is not listed; its lines are in its holder's unit.
    assert [(unit.symbol, unit.kind, unit.nested_in, unit.ranges) for unit in units.values()] == [
        ("Basket.add", UnitKind.METHOD, None, ((2, 3),)),
        ("Basket.total", UnitKind.METHOD, None, ((5, 8),)),
        ("<top level>", UnitKind.TOP_LEVEL, None, ((1, 1),)),
    ]


def test_unit_ids_are_unique_when_two_functions_share_their_lines(shop: CodeIndex) -> None:
    # Act
    listed = list_units(shop, shop.files, box_chars=JEV_BOX).units

    # Assert: two methods on one line, and a function with its arrow callback on one line, have the
    # same text, so each line is one unit, named by its first named function.
    ids = [unit.id for unit in listed]
    assert len(ids) == len(set(ids))
    assert [(unit.id, unit.symbol) for unit in listed if unit.path == "web/pair.ts"] == [
        ("web/pair.ts:1-1", "first")
    ]
    assert [unit.symbol for unit in listed if unit.path == "web/pick.ts"] == ["pick"]


def test_a_test_file_marks_its_units_as_tests(shop: CodeIndex) -> None:
    # Act
    units = _units_by_id(shop, ("tests/test_routes.py", "app/routes.py"))

    # Assert
    assert units["tests/test_routes.py:1-2"].test is True
    assert units["app/routes.py:9-10"].test is False


def test_files_the_parser_cannot_read_are_named_with_their_reason(shop: CodeIndex) -> None:
    # Act
    listing = list_units(shop, ("README.md", "config.json", "app/routes.py"), box_chars=JEV_BOX)

    # Assert
    assert {unit.path for unit in listing.units} == {"app/routes.py"}
    assert listing.unlisted == {
        "README.md": "language not supported",
        "config.json": "language not supported",
    }


def test_a_files_top_level_code_is_one_unit_without_function_bodies(shop: CodeIndex) -> None:
    # Act
    top = _units_by_id(shop, ("app/routes.py",))["app/routes.py:top"]

    # Assert: the route table, the class body outside its method and the registration call, in
    # order, each run of lines with its own range; no line of `create` or `describe`.
    assert (top.kind, top.symbol, top.nested_in) == (UnitKind.TOP_LEVEL, "<top level>", None)
    assert top.ranges == ((1, 6), (13, 15), (21, 21))
    assert (top.start, top.end) == (1, 21)
    assert items_to_judge(top) == (Item("app/routes.py:top", "app/routes.py", ((1, 6), (13, 15), (21, 21))),)
    text = read_ranges(shop, top.path, top.ranges)
    assert read_ranges(shop, top.path, json.loads(json.dumps(top.ranges))) == text  # as a request's lines
    assert text == "\n".join((_lines(ROUTES, 1, 6), _lines(ROUTES, 13, 15), _lines(ROUTES, 21, 21)))
    assert "return save(request)" not in text and 'return "settings"' not in text
    assert top.content_sha256 == _sha256(text)


def test_a_file_of_only_imports_has_no_top_level_unit(shop: CodeIndex) -> None:
    # Act
    units = _units_by_id(shop, ("app/only_imports.py", "web/wire.ts"))

    # Assert: comments, blank lines and imports, also over several lines, are not top-level code.
    assert sorted(units) == ["app/only_imports.py:10-11", "web/wire.ts:12-14"]


def test_a_directive_and_a_line_of_closing_brackets_are_not_top_level_code(shop: CodeIndex) -> None:
    # Act
    units = _units_by_id(shop, ("web/panel.tsx", "web/vite.config.ts"))

    # Assert: "use client" above imports, and the `);` left after a callback, list no top-level unit.
    assert sorted(units) == ["web/panel.tsx:5-8", "web/vite.config.ts:3-5"]


def test_a_require_inside_other_code_is_top_level_code(shop: CodeIndex) -> None:
    # Act
    units = _units_by_id(shop, ("web/mount.js",))

    # Assert: only a whole `require` statement counts as an import; mounting a router is code.
    assert [(unit.id, unit.ranges) for unit in units.values()] == [("web/mount.js:top", ((1, 2),))]


def test_a_file_gone_after_the_inventory_is_named_not_listed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_files(
        root, {"app/kept.py": "def kept():\n    return 1\n", "app/gone.py": "def gone():\n    return 2\n"}
    )
    index = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "facts")
    (root / "app/gone.py").unlink()

    # Act
    listing = list_units(index, index.files, box_chars=JEV_BOX)

    # Assert
    assert [unit.id for unit in listing.units] == ["app/kept.py:1-2"]
    assert listing.unlisted == {"app/gone.py": "disappeared after inventory"}


def test_a_file_the_index_never_held_is_named_with_the_index_reason_not_raised(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_files(root, {"app/kept.py": "def kept():\n    return 1\n"})
    commit_all(root)
    index = CodeIndex.from_git(root, ["app/kept.py", "app/removed.py"], fact_cache_dir=tmp_path / "facts")

    # Act
    listing = list_units(index, ["app/kept.py", "app/removed.py", "app/never_asked.py"], box_chars=JEV_BOX)

    # Assert: the index's own reason where it has one, else that the file is outside its scope.
    assert [unit.id for unit in listing.units] == ["app/kept.py:1-2"]
    assert listing.unlisted == {
        "app/removed.py": "no file at this path",
        "app/never_asked.py": "not in the index scope",
    }


def test_top_level_code_over_the_box_is_cut_into_pieces(shop: CodeIndex) -> None:
    # Act
    top = _units_by_id(shop, ("app/table.py",), TABLE_BOX)["app/table.py:top"]

    # Assert: no piece spans the gap the function leaves between the two runs.
    assert top.ranges == ((1, 92), (99, 130))
    assert [(piece.index, piece.start, piece.end) for piece in top.pieces] == [
        (0, 1, 60),
        (1, 61, 92),
        (2, 99, 130),
    ]
    assert not any(piece.too_large_to_judge for piece in top.pieces)


FLASK_VIEWS = """\
from app import app
import functools


@app.route(
    "/orders",
    methods=["POST"],
)
@functools.cache
def create_order():
    return save()


class Views:
    @staticmethod
    # Lists every order.
    def list_orders():
        return []
"""

NEST_CONTROLLER = """\
import { Controller, Get, HttpCode, Post } from "@nestjs/common";

@Controller("orders")
export class OrdersController {
  @Get(":id")
  find(id: string) {
    return id;
  }

  @Post()
  // Created orders answer 201.
  @HttpCode(201)
  create() {
    return 1;
  }
}
"""


@pytest.fixture
def decorated(tmp_path: Path) -> CodeIndex:
    root = tmp_path / "decorated"
    write_files(root, {"app/views.py": FLASK_VIEWS, "web/orders.controller.ts": NEST_CONTROLLER})
    commit_all(root)
    return CodeIndex.from_git(root, fact_cache_dir=tmp_path / "facts")


def _code_lines(source: str) -> list[int]:
    return [number for number, line in enumerate(source.split("\n"), 1) if line.strip()]


def _code_lines_in(units: Iterable[Unit], source: str) -> list[int]:
    """Each line of code in ``units``, once per unit holding it, so a line in two units shows twice."""
    code = set(_code_lines(source))
    lines = (line for unit in units for start, end in unit.ranges for line in range(start, end + 1))
    return sorted(line for line in lines if line in code)


def test_a_python_function_unit_includes_its_decorators(decorated: CodeIndex) -> None:
    # Act
    units = _units_by_id(decorated, ("app/views.py",))

    # Assert: the route travels with its handler, so Jev judges the handler with its route, and the
    # decorator lines leave the top-level code. The id and the index's span still start at `def`.
    assert sorted(units) == ["app/views.py:10-11", "app/views.py:17-18", "app/views.py:top"]
    assert units["app/views.py:10-11"].ranges == ((5, 11),)
    assert read_ranges(decorated, "app/views.py", units["app/views.py:10-11"].ranges).startswith(
        '@app.route(\n    "/orders",'
    )
    assert (units["app/views.py:17-18"].symbol, units["app/views.py:17-18"].ranges) == (
        "Views.list_orders",
        ((15, 18),),
    )
    assert units["app/views.py:top"].ranges == ((1, 2), (14, 14))
    assert _code_lines_in(units.values(), FLASK_VIEWS) == _code_lines(FLASK_VIEWS)
    assert [(span.start, span.end) for span in decorated.functions_in("app/views.py")] == [(10, 11), (17, 18)]


def test_a_typescript_method_unit_includes_its_decorators(decorated: CodeIndex) -> None:
    # Act
    units = _units_by_id(decorated, ("web/orders.controller.ts",))

    # Assert: each method takes its own decorators, past a comment between them; the class's
    # decorator stays with the class head in the top-level code.
    assert sorted(units) == [
        "web/orders.controller.ts:13-15",
        "web/orders.controller.ts:6-8",
        "web/orders.controller.ts:top",
    ]
    assert units["web/orders.controller.ts:6-8"].ranges == ((5, 8),)
    assert units["web/orders.controller.ts:13-15"].ranges == ((10, 15),)
    assert units["web/orders.controller.ts:6-8"].symbol == "OrdersController.find"
    assert units["web/orders.controller.ts:top"].ranges == ((1, 4), (16, 16))
    assert _code_lines_in(units.values(), NEST_CONTROLLER) == _code_lines(NEST_CONTROLLER)
    assert [(span.start, span.end) for span in decorated.functions_in("web/orders.controller.ts")] == [
        (6, 8),
        (13, 15),
    ]


CLIENT_PROTOCOL = """\
from typing import Protocol


class JevClient(Protocol):
    def ask(self, state, questions) -> dict: ...

    def close(self) -> None:
        '''Releases the connection.'''

    def flush(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def retry(self) -> None:
        \"\"\"Tries again.\"\"\"
        raise NotImplementedError("retry is not supported")


def call(client: JevClient):
    \"\"\"Asks once.\"\"\"
    return client.ask({}, {})
"""


@pytest.fixture
def protocol(tmp_path: Path) -> CodeIndex:
    root = tmp_path / "protocol"
    write_files(root, {"app/client.py": CLIENT_PROTOCOL})
    commit_all(root)
    return CodeIndex.from_git(root, fact_cache_dir=tmp_path / "facts")


def test_a_stub_joins_its_files_top_level_code(protocol: CodeIndex) -> None:
    # Act
    units = _units_by_id(protocol, ("app/client.py",))

    # Assert: methods whose body is only `...`, a docstring, `pass` or `raise NotImplementedError`,
    # alone or together, declare a shape and do nothing, so the Protocol is judged whole in the
    # top-level code, decorators included. A docstring before real code is no stub. The index still
    # knows every stub as a function.
    assert sorted(units) == ["app/client.py:22-24", "app/client.py:top"]
    assert units["app/client.py:top"].ranges == ((1, 19),)
    assert _code_lines_in(units.values(), CLIENT_PROTOCOL) == _code_lines(CLIENT_PROTOCOL)
    assert [span.name for span in protocol.functions_in("app/client.py")] == [
        "ask",
        "close",
        "flush",
        "reset",
        "retry",
        "call",
    ]


def test_a_line_on_a_decorator_names_the_function_it_decorates(decorated: CodeIndex) -> None:
    # Act
    resolved = resolve_anchors(
        decorated,
        (LineAnchor("app/views.py", 5), LineAnchor("web/orders.controller.ts", 12)),
        box_chars=JEV_BOX,
    )

    # Assert: `@app.route(` and NestJS's `@HttpCode(201)` sit before their functions' own lines
    assert [unit.id for unit in resolved.units] == ["app/views.py:10-11", "web/orders.controller.ts:13-15"]
    assert resolved.unresolved == ()


def test_a_line_in_a_stub_names_the_top_level_code_it_joins(protocol: CodeIndex) -> None:
    # Act
    resolved = resolve_anchors(
        protocol, (LineAnchor("app/client.py", 6), LineAnchor("app/client.py", 16)), box_chars=JEV_BOX
    )

    # Assert: both stubs are in the one top-level unit, named once
    assert [unit.id for unit in resolved.units] == ["app/client.py:top"]


def test_a_line_between_units_names_the_top_level_code_or_is_reported(shop: CodeIndex) -> None:
    # Act: line 93 of table.py is blank between TABLE and `lookup`; line 3 of helpers.py is blank
    # between its two functions, and helpers.py has no top-level code
    resolved = resolve_anchors(
        shop, (LineAnchor("app/table.py", 93), LineAnchor("app/helpers.py", 3)), box_chars=JEV_BOX
    )

    # Assert
    assert [unit.id for unit in resolved.units] == ["app/table.py:top"]
    assert resolved.unresolved == (
        UnresolvedAnchor(
            LineAnchor("app/helpers.py", 3), "line 3 of app/helpers.py: blank and outside every function"
        ),
    )


def test_a_range_across_two_functions_names_both_and_each_once(shop: CodeIndex) -> None:
    # Act: lines 93 and 94 of table.py are blank, between TABLE and `lookup` on lines 95 and 96
    resolved = resolve_anchors(
        shop,
        (
            RangeAnchor("app/helpers.py", 1, 6),
            LineAnchor("app/helpers.py", 6),
            RangeAnchor("app/table.py", 93, 96),
        ),
        box_chars=JEV_BOX,
    )

    # Assert: a range's blank lines name no unit
    assert [unit.id for unit in resolved.units] == [
        "app/helpers.py:1-2",
        "app/helpers.py:5-6",
        "app/table.py:95-96",
    ]


def test_a_range_names_the_function_holding_a_nested_one_and_a_line_the_innermost(shop: CodeIndex) -> None:
    # Act: `helper` is nested in `Basket.total` (lines 5 to 8), on lines 6 and 7
    whole_method = resolve_anchors(shop, (RangeAnchor("app/basket.py", 5, 8),), box_chars=JEV_BOX)
    inner = resolve_anchors(shop, (LineAnchor("app/basket.py", 7),), box_chars=JEV_BOX)

    # Assert: a line inside the nested function names it, a unit the listing leaves out, and its
    # nested_in names the listed method that holds its text
    assert [unit.id for unit in whole_method.units] == ["app/basket.py:5-8"]
    assert [(unit.symbol, unit.nested_in) for unit in inner.units] == [
        ("Basket.total.helper", "app/basket.py:5-8")
    ]


def test_listed_only_names_the_outermost_function_holding_a_nested_one(shop: CodeIndex) -> None:
    # Arrange: `inner` sits in `middle`, which sits in `outer`; `helper` sits in `Basket.total`
    anchors = (LineAnchor("app/layers.py", 4), RangeAnchor("app/basket.py", 6, 7))

    # Act
    innermost = resolve_anchors(shop, anchors, box_chars=JEV_BOX)
    listed = resolve_anchors(shop, anchors, box_chars=JEV_BOX, listed_only=True)

    # Assert
    assert [unit.symbol for unit in innermost.units] == ["outer.middle.inner", "Basket.total.helper"]
    assert [unit.symbol for unit in listed.units] == ["outer", "Basket.total"]


def test_listed_only_reports_a_line_in_top_level_code_the_listing_leaves_out(shop: CodeIndex) -> None:
    # Arrange: only_imports.py's top-level code is a comment and imports, on lines 1 to 7
    anchors = (LineAnchor("app/only_imports.py", 3), RangeAnchor("app/only_imports.py", 1, 7))

    # Act
    innermost = resolve_anchors(shop, anchors, box_chars=JEV_BOX)
    listed = resolve_anchors(shop, anchors, box_chars=JEV_BOX, listed_only=True)

    # Assert
    assert [unit.id for unit in innermost.units] == ["app/only_imports.py:top"]
    assert listed.units == ()
    assert [item.problem for item in listed.unresolved] == [
        "line 3 of app/only_imports.py: top-level code of only imports, comments, directives and "
        "brackets, which a listing leaves out",
        "lines 1 to 7 of app/only_imports.py: top-level code of only imports, comments, directives and "
        "brackets, which a listing leaves out",
    ]


def test_listed_only_names_only_units_the_listing_lists(shop: CodeIndex) -> None:
    # Arrange: a line and a range on every line of each file
    files = ("app/basket.py", "app/layers.py", "app/only_imports.py", "app/routes.py", "web/pick.ts")
    anchors = [
        anchor
        for file in files
        for count in [len(shop.lines(file))]
        for anchor in (*(LineAnchor(file, line) for line in range(1, count + 1)), RangeAnchor(file, 1, count))
    ]
    listed_ids = {unit.id for unit in list_units(shop, files, box_chars=JEV_BOX).units}

    # Act
    resolved = resolve_anchors(shop, anchors, box_chars=JEV_BOX, listed_only=True)

    # Assert: every listed unit is named, and nothing else
    assert {unit.id for unit in resolved.units} == listed_ids


def test_an_anchor_outside_the_scope_or_its_file_is_reported_before_any_parse(
    sample_repo: Path, tmp_path: Path
) -> None:
    # Arrange
    scans = []
    index = CodeIndex.from_git(
        sample_repo, fact_cache_dir=tmp_path / "facts", scan_observer=lambda *event: scans.append(event)
    )
    anchors = (
        LineAnchor("app/missing.py", 3),
        LineAnchor("app/orders.py", 99),
        RangeAnchor("app/orders.py", 7, 5),
    )

    # Act
    resolved = resolve_anchors(index, anchors, box_chars=JEV_BOX)

    # Assert
    assert resolved.units == ()
    assert [item.problem for item in resolved.unresolved] == [
        "app/missing.py is not in scope",
        "line 99 is outside app/orders.py, which has 14 lines",
        "the range 7-5 ends before it starts",
    ]
    assert scans == []
