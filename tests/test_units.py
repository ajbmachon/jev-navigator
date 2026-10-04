"""Units: the functions, methods and top-level code a search judges, and the anchors that name them.

Every test runs the real ast-grep parser over a real git repository, so each record below is what
the index reports for that source, not what this module assumes it reports. Each index gets its own
fact-cache directory, so nothing is read from facts another test cached.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from git_repos import commit_all, git, write_files

from jev_navigator.index.bindings import BindingStatus
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.index.units import (
    Edge,
    EdgeKind,
    EstablishedBy,
    Item,
    LineAnchor,
    Origin,
    RangeAnchor,
    SymbolAnchor,
    UnitKind,
    best_piece,
    items_to_judge,
    list_units,
    read_ranges,
    read_units,
    resolve_anchors,
    unit_score,
    write_units,
)
from jev_navigator.judgments.client import JEV_INPUT_BOX_CHARS
from jev_navigator.judgments.questions import serialized_chars

JEV_BOX = JEV_INPUT_BOX_CHARS
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
    assert place.reached_by == Origin.SCOPE
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


def test_an_anonymous_callback_is_named_by_its_holder(sample_index: CodeIndex) -> None:
    # Act
    resolved = resolve_anchors(sample_index, (LineAnchor("web/routes.ts", 4),), box_chars=JEV_BOX)

    # Assert
    [callback] = resolved.units
    assert (callback.id, callback.symbol) == ("web/routes.ts:4-4", "registerRoutes.<anonymous:4>")
    assert callback.kind == UnitKind.FUNCTION
    assert callback.nested_in == "web/routes.ts:3-5"


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


def test_methods_and_nested_functions_are_qualified_by_every_holder(shop: CodeIndex) -> None:
    # Act
    units = _units_by_id(shop, ("app/basket.py",))
    helper = resolve_anchors(shop, (LineAnchor("app/basket.py", 7),), box_chars=JEV_BOX).units

    # Assert
    assert [(unit.symbol, unit.kind, unit.nested_in) for unit in units.values()] == [
        ("Basket.add", UnitKind.METHOD, None),
        ("Basket.total", UnitKind.METHOD, None),
        ("<top level>", UnitKind.TOP_LEVEL, None),
    ]
    assert [(unit.symbol, unit.kind, unit.nested_in) for unit in helper] == [
        ("Basket.total.helper", UnitKind.FUNCTION, "app/basket.py:5-8")
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

    anchored = resolve_anchors(shop, (LineAnchor("README.md", 1),), box_chars=JEV_BOX)

    # Assert
    assert {unit.path for unit in listing.units} == {"app/routes.py"}
    assert [item.problem for item in anchored.unresolved] == ["language not supported"]
    assert listing.unlisted == {
        "README.md": "language not supported",
        "config.json": "language not supported",
    }


def test_unit_stream_round_trips_as_json_lines(shop: CodeIndex) -> None:
    listed = list_units(shop, shop.files, box_chars=SMALL_BOX).units
    anchored = resolve_anchors(shop, (LineAnchor("app/routes.py", 21),), box_chars=SMALL_BOX).units
    edge = Edge(
        "app/routes.py:9-10",
        "app/basket.py:2-3",
        EdgeKind.CALLS,
        EstablishedBy.PARSER,
        BindingStatus.CANDIDATE,
    )
    reached = replace(listed[0], reached_by=edge)
    units = (*listed, *anchored, reached)
    stream = io.StringIO()

    # Act
    write_units(units, stream)
    stream.seek(0)
    read_back = read_units(stream)

    # Assert
    assert read_back == list(units)
    lines = stream.getvalue().splitlines()
    assert len(lines) == len(units)
    assert json.loads(lines[-1])["reached_by"] == {
        "from": "app/routes.py:9-10",
        "to": "app/basket.py:2-3",
        "kind": "calls",
        "established_by": "parser",
        "binding": "candidate",
    }
    assert any(json.loads(line)["pieces"] for line in lines)


def test_symbol_and_line_anchors_resolve_to_the_same_unit(sample_index: CodeIndex) -> None:
    anchors = (
        SymbolAnchor("check_limits"),
        SymbolAnchor("check_limits", "app/validation.py"),
        LineAnchor("app/validation.py", 12),
        RangeAnchor("app/validation.py", 10, 12),
    )

    # Act
    resolved = [resolve_anchors(sample_index, (anchor,), box_chars=JEV_BOX) for anchor in anchors]

    # Assert
    assert {tuple(unit.id for unit in result.units) for result in resolved} == {("app/validation.py:10-12",)}
    assert all(result.units[0].reached_by == Origin.ANCHOR for result in resolved)
    assert all(result.unresolved == () for result in resolved)


def test_a_qualified_symbol_names_one_method(sample_index: CodeIndex) -> None:
    # Act
    qualified = resolve_anchors(sample_index, (SymbolAnchor("OrderService.place"),), box_chars=JEV_BOX)
    plain = resolve_anchors(sample_index, (SymbolAnchor("place"),), box_chars=JEV_BOX)

    # Assert
    assert [unit.id for unit in qualified.units] == ["app/orders.py:5-7"]
    assert [unit.id for unit in plain.units] == ["app/orders.py:5-7"]


def test_a_class_symbol_resolves_to_its_top_level_code_and_its_methods(shop: CodeIndex) -> None:
    # Act
    resolved = resolve_anchors(shop, (SymbolAnchor("Settings"),), box_chars=JEV_BOX)

    # Assert
    assert [unit.id for unit in resolved.units] == ["app/routes.py:top", "app/routes.py:17-18"]


def test_a_line_inside_a_nested_function_names_the_nested_function(shop: CodeIndex) -> None:
    # Act
    inner = resolve_anchors(shop, (LineAnchor("app/basket.py", 7),), box_chars=JEV_BOX)
    outer = resolve_anchors(shop, (LineAnchor("app/basket.py", 8),), box_chars=JEV_BOX)

    # Assert
    assert [unit.symbol for unit in inner.units] == ["Basket.total.helper"]
    assert [unit.symbol for unit in outer.units] == ["Basket.total"]


def test_a_range_anchor_resolves_to_each_outermost_unit_it_touches(shop: CodeIndex) -> None:
    # Act
    whole_method = resolve_anchors(shop, (RangeAnchor("app/basket.py", 5, 8),), box_chars=JEV_BOX)
    across = resolve_anchors(shop, (RangeAnchor("app/routes.py", 8, 14),), box_chars=JEV_BOX)

    # Assert: the nested helper sits inside the cited method, so the method stands for it; a range
    # from a function into a class body touches the function and the top-level code, never the blank
    # lines between them.
    assert [unit.id for unit in whole_method.units] == ["app/basket.py:5-8"]
    assert [unit.id for unit in across.units] == ["app/routes.py:9-10", "app/routes.py:top"]


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


def test_an_anchor_outside_every_function_resolves_to_the_top_level_unit(shop: CodeIndex) -> None:
    # Act
    resolved = resolve_anchors(shop, (LineAnchor("app/routes.py", 21),), box_chars=JEV_BOX)

    # Assert
    assert [(unit.id, unit.kind, unit.reached_by) for unit in resolved.units] == [
        ("app/routes.py:top", UnitKind.TOP_LEVEL, Origin.ANCHOR)
    ]


def test_an_anchor_on_an_import_names_the_top_level_code_the_listing_leaves_out(shop: CodeIndex) -> None:
    # Act
    resolved = resolve_anchors(shop, (LineAnchor("app/only_imports.py", 3),), box_chars=JEV_BOX)

    # Assert: the caller pointed at it, so it is judged even though round 0 would not list it.
    assert [(unit.id, unit.ranges) for unit in resolved.units] == [("app/only_imports.py:top", ((1, 7),))]


def test_an_unknown_symbol_anchor_is_reported_not_guessed(sample_index: CodeIndex) -> None:
    # Act
    resolved = resolve_anchors(sample_index, (SymbolAnchor("resolve_provider"),), box_chars=JEV_BOX)

    # Assert
    assert resolved.units == ()
    assert [(item.anchor, item.problem) for item in resolved.unresolved] == [
        (
            SymbolAnchor("resolve_provider"),
            "no function, class or declaration named resolve_provider in scope",
        )
    ]


def test_an_anchor_outside_the_scope_or_the_file_is_reported_before_any_parse(
    sample_repo: Path, tmp_path: Path
) -> None:
    scans = []
    index = CodeIndex.from_git(
        sample_repo, fact_cache_dir=tmp_path / "facts", scan_observer=lambda *event: scans.append(event)
    )
    anchors = (
        LineAnchor("app/missing.py", 3),
        LineAnchor("app/orders.py", 99),
        RangeAnchor("app/orders.py", 7, 5),
        SymbolAnchor("cancel", "app/missing.py"),
    )

    # Act
    resolved = resolve_anchors(index, anchors, box_chars=JEV_BOX)

    # Assert
    assert resolved.units == ()
    assert [item.problem for item in resolved.unresolved] == [
        "app/missing.py is not in scope",
        "line 99 is outside app/orders.py, which has 14 lines",
        "the range 7-5 ends before it starts",
        "app/missing.py is not in scope",
    ]
    assert scans == []
