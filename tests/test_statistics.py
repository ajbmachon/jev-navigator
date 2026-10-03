"""Structural measurements taken from the index's own parser: sizes, ranges, counts, coverage.

Every test here runs the real ast-grep parser over real temporary Python and JavaScript/TypeScript
source, so the numbers below are what the parser reports, not what this module assumes it reports.
Each index is built with its own fact-cache directory, so a measurement is taken from a fresh parse
instead of from facts another test cached.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import write_files

from jev_navigator.directives import statistics as stats
from jev_navigator.index.code_index import CodeIndex

BASKET = """\
class Basket:
    def add(self, item):
        return sorted(
            [item],
            reverse=True,
        )

    def total(self):
        def helper(x):
            return x * 2
        return helper(2)


def standalone(items):
    return sorted(items)
"""

CART = """\
export class Cart {
  add(item) {
    return item;
  }
  total() {
    return this.items.reduce((a, b) => a + b, 0);
  }
}

export const sum = (xs) => xs.reduce((a, b) => a + b, 0);

function scale(xs) {
  const twice = (x) => x * 2;
  return xs.map(twice);
}
"""

TIES = """\
function pickOne(items) {
  return items.filter((item) => item.ok);
}

function pickTwo(items) {
  return items.filter((item) => item.ok);
}
"""

BROKEN = """\
function ok(a) {
  return a;
}

function broken(: {
  return a;
}
"""

ONE_LINE_CLASS = "class Box { v() { return 1; } }\n"

SPLIT_CLASS = "class Box {\n  v() { return 1; }\n}\n"

NOTES = "# notes\n\nnothing here is code\n"


def _index(tmp_path: Path, files: dict[str, str], *, observe=None) -> CodeIndex:
    """An index over ``files`` as they lie on disk, parsed from scratch for this test."""
    root = tmp_path / "repo"
    write_files(root, files)
    return CodeIndex.from_directory(
        root, fact_cache_dir=tmp_path / "facts", scan_observer=observe or (lambda *_: None)
    )


def _span(symbol) -> tuple[int, int]:
    return (symbol.span.start, symbol.span.end)


def _shown(ranking: stats.Largest) -> list[str]:
    return [f"{symbol.span.file}:{symbol.span.name}:{symbol.size}" for symbol in ranking.largest]


def test_function_sizes_are_the_inclusive_lines_they_fill(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET, "web/cart.ts": CART})

    # Act
    measured = stats.symbol_spans(index, "app/basket.py") + stats.symbol_spans(index, "web/cart.ts")
    by_name = {(symbol.span.name, symbol.span.file): symbol for symbol in measured}

    # Assert
    assert _span(by_name[("add", "app/basket.py")]) == (2, 6)  # a method, measured with its body
    assert by_name[("helper", "app/basket.py")].size == 2  # a function nested in a method
    assert by_name[("standalone", "app/basket.py")].size == 2
    assert by_name[("sum", "web/cart.ts")].size == 1  # a one-line arrow function
    assert _span(by_name[("scale", "web/cart.ts")]) == (12, 15)
    assert all(symbol.size == symbol.span.end - symbol.span.start + 1 for symbol in measured)
    assert all(symbol.size == symbol.span.size() for symbol in measured)


def test_methods_and_nested_functions_count_as_functions_and_name_their_holder(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET})

    # Act
    counts = stats.count_symbols(index)
    basket = counts.per_file["app/basket.py"]

    # Assert: the index calls a method and a nested def functions too, so they are counted as such.
    assert basket.counts == {"function": 4, "class": 1}
    assert counts.total == {"function": 4, "class": 1}
    assert [(symbol.span.name, symbol.holder.name) for symbol in basket.held] == [
        ("add", "Basket"),
        ("total", "Basket"),
        ("helper", "total"),
    ]
    assert [symbol.span.name for symbol in basket.functions if not symbol.held] == ["standalone"]


def test_classes_are_measured_apart_from_functions(tmp_path: Path) -> None:
    index = _index(tmp_path, {"web/cart.ts": CART})

    # Act
    cart = stats.count_symbols(index).per_file["web/cart.ts"]
    ranking = stats.largest_functions(index)

    # Assert
    assert [symbol.span.name for symbol in cart.classes] == ["Cart"]
    assert _span(cart.classes[0]) == (1, 8) and cart.classes[0].size == 8
    assert "Cart" not in {symbol.span.name for symbol in cart.functions}
    assert all(symbol.kind == "function" for symbol in ranking.measured)


def test_the_widest_function_is_shown_with_its_inclusive_line_range(tmp_path: Path) -> None:
    index = _index(tmp_path, {"web/cart.ts": CART})

    # Act
    ranking = stats.largest_functions(index)

    # Assert: methods and nested functions are counted but kept out of the ranking until the caller
    # asks for them, so a method is never ranked above the top-level functions it sits in.
    assert ranking.size == 4
    assert [(symbol.span.name, symbol.size) for symbol in ranking.biggest] == [("scale", 4)]
    assert _span(ranking.biggest[0]) == (12, 15)
    assert [(symbol.span.name, symbol.size) for symbol in ranking.largest] == [
        ("scale", 4),
        ("add", 3),
        ("total", 3),
        ("<anonymous>", 1),
        ("<anonymous>", 1),
        ("sum", 1),
        ("twice", 1),
    ]
    assert ranking.largest == ranking.measured
    assert ranking.caveat == "" and ranking.complete
    assert stats.count_symbols(index, ("web/cart.ts",)).total == {"function": 7, "class": 1}


def test_functions_of_equal_size_are_all_kept(tmp_path: Path) -> None:
    index = _index(tmp_path, {"web/ties.ts": TIES, "web/cart.ts": CART})

    # Act
    ties_only = stats.largest_functions(index, ("web/ties.ts",))
    whole_scope = stats.largest_functions(index)

    # Assert: two 3-line module-level functions tie, and the tie is shown whole rather than cut to
    # one name, whether the tie is inside one file or spread across two.
    assert ties_only.size == 3
    assert [(symbol.span.name, symbol.size) for symbol in ties_only.biggest] == [
        ("pickOne", 3),
        ("pickTwo", 3),
    ]
    assert [symbol.span.file for symbol in whole_scope.biggest] == ["web/cart.ts"]
    assert [(symbol.span.file, symbol.span.name) for symbol in whole_scope.biggest] == [
        ("web/cart.ts", "scale")
    ]


def test_held_functions_join_the_ranking_unless_the_caller_declines(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET})

    # Act
    every = stats.largest_functions(index)
    outside = stats.largest_functions(index, held=False)

    # Assert: a method is a function to the index, so the default ranking measures and ranks it too;
    # ``held=False`` keeps the ranking to the symbols nothing else holds.
    assert [(symbol.span.name, symbol.size) for symbol in every.biggest] == [("add", 5)]
    assert "helper" in {symbol.span.name for symbol in every.measured}
    assert [(symbol.span.name, symbol.size) for symbol in outside.biggest] == [("standalone", 2)]
    assert "add" not in {symbol.span.name for symbol in outside.measured}


def test_per_file_counts_add_up_to_the_scope_total(tmp_path: Path) -> None:
    index = _index(
        tmp_path,
        {
            "app/basket.py": BASKET,
            "web/cart.ts": CART,
            "web/ties.ts": TIES,
            "web/broken.ts": BROKEN,
            "web/oneline.ts": ONE_LINE_CLASS,
        },
    )

    # Act
    counts = stats.count_symbols(index)

    # Assert
    assert counts.per_file["web/ties.ts"].counts == {"function": 4, "class": 0}
    assert counts.per_file["web/oneline.ts"].counts == {"function": 1, "class": 1}
    assert counts.total == {
        "function": sum(measured.counts["function"] for measured in counts.per_file.values()),
        "class": sum(measured.counts["class"] for measured in counts.per_file.values()),
    }
    assert stats.count_symbols(index, ()).count_of("web/cart.ts") is None


def test_nested_lines_are_covered_once_when_counting_covered_lines(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET})

    # Act
    counts = stats.count_symbols(index)

    # Assert: `class Basket` fills lines 1-11, and its method `add` and the function nested in
    # `total` sit inside those lines, so the file covers 13 lines of code, not one per symbol.
    assert counts.per_file["app/basket.py"].symbol_lines == 13
    assert counts.symbol_lines == 13


def test_a_file_that_only_partly_parsed_reports_a_floor(tmp_path: Path) -> None:
    index = _index(tmp_path, {"web/broken.ts": BROKEN})

    # Act
    counts = stats.count_symbols(index)
    broken = counts.per_file["web/broken.ts"]

    # Assert: what the grammar swallowed is unknown, not absent, so the count is a floor and the
    # coverage says so instead of claiming a complete count.
    assert broken.counts == {"function": 1, "class": 0}
    assert broken.parsed is False and broken.symbol_lines == 3
    assert counts.coverage.unparsed == ("web/broken.ts",)
    assert counts.coverage.measured == ("web/broken.ts",)
    assert counts.complete is False and "parsed only partially" in counts.caveat


def test_a_class_and_its_only_method_on_one_line_come_back_as_one_symbol(tmp_path: Path) -> None:
    collapsed = _index(tmp_path / "one", {"web/oneline.ts": ONE_LINE_CLASS})
    written_out = _index(tmp_path / "two", {"web/split.ts": SPLIT_CLASS})

    # Act
    one_line = stats.count_symbols(collapsed).per_file["web/oneline.ts"]
    three_lines = stats.count_symbols(written_out).per_file["web/split.ts"]

    # Assert: the parser names the class and its method apart even on one shared line, so both are
    # counted; the method is not held, because its holder fills no more lines than it does.
    assert [(symbol.span.name, symbol.held) for symbol in one_line.functions] == [("v", False)]
    assert [(symbol.span.name, symbol.held) for symbol in one_line.classes] == [("Box", False)]
    assert one_line.counts == {"function": 1, "class": 1}
    assert [(symbol.span.name, symbol.held) for symbol in three_lines.functions] == [("v", True)]
    assert [(symbol.span.name, symbol.held) for symbol in three_lines.classes] == [("Box", False)]
    assert three_lines.counts == {"function": 1, "class": 1}
    assert stats.scan_coverage(written_out).complete and stats.scan_coverage(written_out).statement == ""


def test_a_one_line_symbol_covers_its_own_line_once(tmp_path: Path) -> None:
    index = _index(
        tmp_path, {"web/tiny.ts": "function tiny() { return 1; }\n", "web/oneline.ts": ONE_LINE_CLASS}
    )

    # Act
    counts = stats.count_symbols(index)

    # Assert: a symbol written on a single line still covers that line — a one-line function and a
    # one-line class with its method each cover exactly one line, counted once despite nesting.
    assert counts.per_file["web/tiny.ts"].symbol_lines == 1
    assert counts.per_file["web/oneline.ts"].symbol_lines == 1
    assert counts.symbol_lines == 2


def test_an_empty_scope_measures_nothing_and_says_so(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET})

    # Act
    counts = stats.count_symbols(index, ())

    # Assert
    assert counts.total == {"function": 0, "class": 0}
    assert counts.per_file == {}
    assert counts.complete is False and counts.caveat == "no scope was given, so no file was measured"


def test_a_scope_of_documents_measures_nothing_and_says_so(tmp_path: Path) -> None:
    index = _index(tmp_path, {"notes.md": NOTES, "web/cart.ts": CART})

    # Act
    documents = stats.scan_coverage(index, ("notes.md",))
    mixed = stats.scan_coverage(index, ("notes.md", "web/cart.ts"))

    # Assert: a document has no grammar to parse, which is named as a gap and never read as proof
    # that the scope holds no code.
    assert documents.measured == () and documents.skipped == ("notes.md",)
    assert documents.complete is False and "is code the parser can read" in documents.statement
    assert mixed.measured == ("web/cart.ts",) and mixed.skipped == ("notes.md",)
    assert mixed.complete and mixed.statement == ""


def test_a_path_that_was_never_inventoried_is_a_gap_not_an_empty_result(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET, "web/gone.ts": "function gone() { return 1; }\n"})
    (tmp_path / "repo" / "web" / "gone.ts").unlink()  # inventoried, then taken off the disk

    # Act
    missing = stats.scan_coverage(index, ("web/never.ts",))
    mixed = stats.scan_coverage(index, ("app/basket.py", "web/gone.ts", "web/never.ts"))

    # Assert: an unreadable file is a gap, never an empty result, whether it was inventoried and
    # disappeared or was never in the index at all.
    assert missing.unmeasured == ("web/never.ts",) and missing.measured == ()
    assert missing.complete is False and "never scanned" in missing.statement
    assert mixed.measured == ("app/basket.py",)
    assert mixed.unmeasured == ("web/gone.ts", "web/never.ts")
    assert mixed.complete is False and "never scanned" in mixed.statement


def test_a_scope_the_caller_names_is_measured_as_it_was_given(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET, "web/cart.ts": CART, "notes.md": NOTES})

    # Act
    one_file = stats.count_symbols(index, ("web/cart.ts", "web/cart.ts"))

    # Assert: a repeated path is measured once, and nothing outside the named scope is measured.
    assert stats.scope_of(index, ("web/cart.ts", "web/cart.ts")) == ("web/cart.ts",)
    assert list(one_file.per_file) == ["web/cart.ts"]
    assert one_file.total == {"function": 7, "class": 1}
    assert one_file.complete and one_file.caveat == ""


def test_a_limit_shortens_the_shown_ranking_never_the_measurement(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/basket.py": BASKET, "web/cart.ts": CART})

    # Act
    capped = stats.count_symbols(index, largest_of=3)

    # Assert: three of the thirteen measured symbols are shown, all thirteen were still measured, and
    # the ranking admits that what it left out could hold a narrower symbol.
    assert len(capped.largest.largest) == 3 and len(capped.largest.measured) == 13
    assert [symbol.size for symbol in capped.largest.largest] == [11, 8, 5]
    assert capped.largest.truncated and capped.largest.complete is False
    assert "only 3 of 13 measured symbols are shown" in capped.largest.caveat
    assert capped.complete and capped.caveat == ""  # the count itself saw the whole scope


def test_the_whole_scope_is_measured_in_one_parser_pass(tmp_path: Path) -> None:
    seen: list[tuple[str, str, int]] = []
    index = _index(
        tmp_path,
        {
            "app/basket.py": BASKET,
            "web/cart.ts": CART,
            "web/ties.ts": TIES,
            "web/broken.ts": BROKEN,
            "web/oneline.ts": ONE_LINE_CLASS,
            "notes.md": NOTES,
        },
        observe=lambda *event: seen.append(event),
    )

    # Act
    counts = stats.count_symbols(index)

    # Assert: the scope goes to the index's own fact scan once, before any count is taken from it,
    # and not once per symbol or once per file.
    assert seen == [("facts", "started", 5), ("facts", "completed", 5)]
    assert index.parser_scans_completed == ("facts",) and index.parser_scans_pending == ()
    assert len(counts.largest.measured) == 20 and len(counts.largest.largest) == 20


def test_source_behind_a_measurement_is_read_through_the_index(tmp_path: Path) -> None:
    index = _index(tmp_path, {"web/cart.ts": CART})

    # Act
    biggest = stats.largest_functions(index, ("web/cart.ts",)).biggest[0]
    read = stats.read_source(index, biggest)

    # Assert
    assert (read.span.start, read.span.end) == (12, 15)
    assert read.source()["lines"] == [12, 15] and read.source()["reached_by"] == "statistics"
    assert read.text.splitlines() == [
        "function scale(xs) {",
        "  const twice = (x) => x * 2;",
        "  return xs.map(twice);",
        "}",
    ]


def test_an_empty_file_that_was_read_is_not_reported_as_a_gap(tmp_path: Path) -> None:
    index = _index(tmp_path, {"app/__init__.py": "", "app/basket.py": BASKET})

    # Act
    coverage = stats.scan_coverage(index)

    # Assert: nothing was found in that file because it holds nothing, which is not a gap.
    assert coverage.measured == ("app/__init__.py", "app/basket.py")
    assert coverage.complete and coverage.statement == ""


def test_the_shared_sample_repository_is_measured_by_the_same_parser(sample_index: CodeIndex) -> None:
    # Act
    counts = stats.count_symbols(sample_index)
    ranking = stats.largest_functions(sample_index)

    # Assert: a method, an arrow function and a module-level function are all functions here, a
    # class is not, and a file read in full with nothing in it is still a file that was read.
    assert counts.per_file["app/orders.py"].counts == {"function": 2, "class": 1}
    assert [symbol.span.name for symbol in counts.per_file["app/orders.py"].held] == ["place"]
    assert counts.per_file["app/comments.py"].counts == {"function": 3, "class": 1}
    assert counts.per_file["app/validation.py"].counts == {"function": 3, "class": 0}
    assert counts.per_file["web/handlers.ts"].counts == {"function": 2, "class": 0}
    assert [symbol.span.name for symbol in counts.per_file["web/routes.ts"].held] == ["<anonymous>"]
    empty = {"function": 0, "class": 0}
    assert counts.per_file["app/__init__.py"].counts == counts.per_file["app/settings.py"].counts == empty
    assert counts.total == {"function": 12, "class": 2} and counts.symbol_lines == 38
    assert counts.complete and counts.caveat == ""

    # Assert: nothing written inside another symbol and no class crowds the ranking, so a Python
    # method, a Python def and a TypeScript arrow of the same length are ranked beside each other.
    assert [symbol.span.name for symbol in ranking.biggest] == ["total"]
    assert ranking.size == 7
    assert len(ranking.measured) == counts.total["function"]  # every function is ranked, held or not
    assert stats.read_source(sample_index, ranking.biggest[0]).text.startswith("def total")
    assert [symbol.span.name for symbol in ranking.largest] == [
        "total",
        "cancel",
        "validate_order",
        "handleOrder",
        "place",
        "check_limits",
        "registerRoutes",
        "charge",
        "normalise",
        "noop",
        "parseOrder",
        "<anonymous>",
    ]
    assert all(symbol.kind == "function" for symbol in ranking.measured)
    assert ranking.complete and ranking.caveat == ""


@pytest.mark.parametrize("kind,expected", [("function", 1), ("class", 1)])
def test_counting_one_kind_does_not_fabricate_zero_for_an_unrequested_kind(tmp_path, kind, expected):
    index = _index(tmp_path, {"sample.py": "class Box:\n    def value(self):\n        return 1\n"})
    result = stats.count_symbols(index, kinds=(kind,))
    assert result.total == {kind: expected}
    assert result.per_file["sample.py"].counts == {kind: expected}
