"""Statistics packs written from an index the caller already has.

Every test runs the real ast-grep parser over real temporary Python and TypeScript source, so a pack
is checked against what the parser reports for the same files, not against what this module assumes
it reports. Each index gets its own fact-cache directory, so a pack is measured from a fresh parse
unless the test is about reusing an index that has already scanned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from git_repos import write_files

from jev_navigator.cli_statistics import STATISTICS_OPERATIONS, create_statistics_pack
from jev_navigator.directives import statistics as stats
from jev_navigator.index.code_index import CodeIndex

BASKET = """\
class Basket:
    def add(self, item):
        return sorted(
            [item],
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

NOTES = """\
# Notes

Nothing here is code, so nothing here holds a function to count.
"""

# Every file here parses, so a ranking taken over all of them is a complete measurement: 14 functions
# and 2 classes, 31 symbol lines, and 7 symbols wide enough to be compared to a whole file.
CLEAN = {"app/basket.py": BASKET, "web/cart.ts": CART, "web/ties.ts": TIES}

# `broken.ts` only partly parses and `oneline.ts` puts a method on the same line as its class, so a
# pack written over all of them has to say what it could not see as well as what it measured.
HARD = {
    "app/basket.py": BASKET,
    "web/cart.ts": CART,
    "web/ties.ts": TIES,
    "web/broken.ts": BROKEN,
    "web/oneline.ts": ONE_LINE_CLASS,
    "docs/notes.md": NOTES,
}


def _write(tmp_path: Path, files: dict[str, str]) -> Path:
    """Write ``files`` into a throwaway repository and name the directory holding them."""
    root = tmp_path / "repo"
    write_files(root, files)
    return root


def _listed(pack: dict) -> list[dict]:
    """Every symbol the pack lists in its line-range table, in the order the pack lists them."""
    return [symbol for symbols in pack["ranges"]["per_file"].values() for symbol in symbols]


def test_a_pack_writes_both_files_and_measures_the_whole_scope_in_one_parser_pass(
    tmp_path: Path,
) -> None:
    # Arrange
    events: list[tuple[str, str, int]] = []
    root = _write(tmp_path, {"app/basket.py": BASKET, "web/cart.ts": CART, "docs/notes.md": NOTES})
    index = CodeIndex.from_directory(
        root, fact_cache_dir=tmp_path / "facts", scan_observer=lambda *event: events.append(event)
    )

    # Act
    pack = create_statistics_pack(root, (), tmp_path / "pack", index=index)

    # Assert: one scan of the whole scope, whatever the number of operations asks for afterwards
    assert len([event for event in events if event[1] == "started"]) == 1
    assert pack["operations"] == list(STATISTICS_OPERATIONS)
    assert pack["options"] == {
        "kinds": ["function", "class"],
        "held": False,
        "limit": None,
        "scope": None,
        "quote_source": False,
        "min_lines": None,
        "max_lines": None,
    }
    assert pack["scope"]["measured"] == ["app/basket.py", "web/cart.ts"]
    assert pack["scope"]["skipped"] == ["docs/notes.md"]
    assert pack["scope"]["unmeasured"] == [] and pack["scope"]["unparsed"] == []
    assert pack["coverage"] == {"complete": True, "statement": "", "unavailable": {}, "not_indexed": {}}
    assert pack["limits"][0].startswith("Symbols come from the index's own parser")

    # Assert: what was measured stays whole in the document, while the count of it is only a summary
    assert pack["counts"]["totals"] == {"function": 10, "class": 2}
    assert pack["counts"]["symbol_lines"] == 25
    assert pack["largest"]["measured"] == 5 and pack["largest"]["shown"] == 5
    assert pack["largest"]["complete"] is True and pack["largest"]["caveat"] == ""
    assert pack["ranges"]["listed"] == 5

    # Assert: what came back as JSON is what was returned, and the returned manifest is what ran
    assert (tmp_path / "pack" / "statistics.json").read_text().endswith("}\n")
    assert json.loads((tmp_path / "pack" / "statistics.json").read_text()) == json.loads(json.dumps(pack))

    # Assert: the document keeps the ranking and the full listing beside it, in that order
    document = (tmp_path / "pack" / "statistics.md").read_text()
    assert document.startswith("# Structural statistics\n")
    assert "Not fully covered" not in document
    assert "Skipped outside parser coverage: `docs/notes.md`" in document
    assert document.index("## Largest symbols") < document.index("## Line ranges")
    assert "Scope: whole directory, measured: `app/basket.py`, `web/cart.ts`" in document
    assert "all measured symbols" in document and "`web/cart.ts:1-8`" in document


def test_a_reused_index_is_measured_from_its_cached_facts_and_never_scanned_again(
    tmp_path: Path,
) -> None:
    # Arrange
    events: list[tuple[str, str, int]] = []
    root = _write(tmp_path, HARD)
    index = CodeIndex.from_directory(
        root, fact_cache_dir=tmp_path / "facts", scan_observer=lambda *event: events.append(event)
    )
    first = create_statistics_pack(root, (), tmp_path / "first", index=index)
    scans_before = len([event for event in events if event[1] == "started"])

    # Act
    second = create_statistics_pack(
        root, (), tmp_path / "second", operations=("count", "largest"), index=index
    )

    # Assert: the facts the first pack already read are read again, never scanned a second time
    assert scans_before == 1 and len([event for event in events if event[1] == "started"]) == 1
    assert second["operations"] == ["count", "largest"]
    assert second["counts"]["totals"] == first["counts"]["totals"] == {"function": 16, "class": 3}
    assert second["counts"]["symbol_lines"] == first["counts"]["symbol_lines"] == 35
    assert second["largest"]["ranking"] == first["largest"]["ranking"]

    # Assert: a pack without the range operation has no line-range section, and says nothing about one
    document = (tmp_path / "second" / "statistics.md").read_text()
    assert "## Line ranges" not in document
    assert "## Counts" in document and "## Largest symbols" in document
    assert "## Line ranges" in (tmp_path / "first" / "statistics.md").read_text()


def test_the_pack_carries_the_measurement_the_owner_took_without_repeating_it(
    tmp_path: Path,
) -> None:
    # Arrange
    root = _write(tmp_path, HARD)
    index = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "facts")
    pack = create_statistics_pack(root, (), tmp_path / "pack", limit=3, index=index)
    fresh = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "fresh")
    taken = stats.count_symbols(fresh, None, kinds=("function", "class"), largest_of=3, held=False)

    # Assert: the counts are the owner's, unchanged, and the ranking came out of that same call
    assert pack["counts"]["totals"] == taken.total == {"function": 16, "class": 3}
    assert pack["counts"]["symbol_lines"] == taken.symbol_lines == 35
    assert {
        file: (record["counts"], record["symbol_lines"], record["parsed"])
        for file, record in pack["counts"]["per_file"].items()
    } == {
        file: (measured.counts, measured.symbol_lines, measured.parsed)
        for file, measured in taken.per_file.items()
    }
    assert pack["largest"]["measured"] == len(taken.largest.measured) == 10
    assert pack["largest"]["shown"] == 3 < pack["largest"]["measured"]
    assert [(record["file"], record["name"], record["size"]) for record in pack["largest"]["ranking"]] == [
        (symbol.span.file, symbol.span.name, symbol.size) for symbol in taken.largest.largest
    ]
    assert pack["largest"]["caveat"] == taken.caveat == pack["coverage"]["statement"]

    # Assert: the same three measurements taken by a caller who names the operations instead of
    # reusing the count's own ranking still describe the same repository.
    separate = stats.largest_functions(fresh, None, kinds=("function", "class"), limit=3, held=False)
    assert [(symbol.span.file, symbol.span.name) for symbol in separate.largest] == [
        (record["file"], record["name"]) for record in pack["largest"]["ranking"]
    ]
    assert len(separate.measured) == pack["largest"]["measured"]

    # Assert: every measured symbol keeps its range in the pack, so a top-N cut hides nothing
    assert pack["ranges"]["listed"] == pack["largest"]["measured"] == 10
    assert {(record["file"], record["name"], tuple(record["lines"])) for record in _listed(pack)} == {
        (symbol.span.file, symbol.span.name, (symbol.span.start, symbol.span.end))
        for symbol in taken.largest.measured
    }

    # Assert: what the parser could not read stays a gap, and a one-line class keeps its method
    assert pack["scope"]["unparsed"] == ["web/broken.ts"]
    assert pack["counts"]["per_file"]["web/broken.ts"]["parsed"] is False
    assert pack["counts"]["per_file"]["web/broken.ts"]["counts"] == {"function": 1, "class": 0}
    assert pack["coverage"]["complete"] is False and pack["largest"]["complete"] is False
    assert [
        (record["name"], record["kind"], record["lines"])
        for record in _listed(pack)
        if record["file"] == "web/oneline.ts"
    ] == [("v", "function", [1, 1]), ("Box", "class", [1, 1])]
    document = (tmp_path / "pack" / "statistics.md").read_text()
    assert "Not fully covered" in document and "unknown, not absent" in document
    assert "`web/broken.ts`" in document
    assert "| `web/oneline.ts` | `v` | function | 1-1 |" in document
    assert "| `web/oneline.ts` | `Box` | class | 1-1 |" in document


def test_a_limit_shortens_the_ranking_shown_never_the_count(tmp_path: Path) -> None:
    # Arrange
    root = _write(tmp_path, CLEAN)
    index = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "facts")

    # Act
    whole = create_statistics_pack(root, (), tmp_path / "whole", index=index)
    top = create_statistics_pack(root, (), tmp_path / "top", limit=2, index=index)

    # Assert: the same files measured twice, so only what is shown may differ
    assert top["counts"]["totals"] == whole["counts"]["totals"] == {"function": 14, "class": 2}
    assert top["counts"]["symbol_lines"] == whole["counts"]["symbol_lines"] == 31
    assert top["options"]["limit"] == 2 and whole["options"]["limit"] is None
    assert top["largest"]["measured"] == whole["largest"]["measured"] == 7
    assert top["largest"]["shown"] == 2 < top["largest"]["measured"]
    assert whole["largest"]["shown"] == whole["largest"]["measured"] == 7
    assert top["largest"]["truncated"] is True and whole["largest"]["truncated"] is False
    assert top["largest"]["complete"] is False and whole["largest"]["complete"] is True
    assert top["ranges"]["listed"] == top["largest"]["measured"] == 7
    assert top["ranges"]["per_file"]["web/ties.ts"]

    # Assert: the shorter ranking says how short it is, in both files, and lists what it left out
    assert "only 2 of 7 measured symbols are shown" in top["largest"]["caveat"]
    assert whole["largest"]["caveat"] == ""
    assert "only the 2 widest" in (tmp_path / "top" / "statistics.md").read_text()
    assert "all measured symbols" in (tmp_path / "whole" / "statistics.md").read_text()
    assert "Measured 7 symbols and shows only 2" in (tmp_path / "top" / "statistics.md").read_text()
    assert "could hold a narrower symbol of its own" in (tmp_path / "top" / "statistics.md").read_text()


def test_held_functions_are_measured_beside_the_symbol_holding_them(tmp_path: Path) -> None:
    # Arrange
    root = _write(tmp_path, CLEAN)
    index = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "facts")

    # Act
    files = create_statistics_pack(root, (), tmp_path / "files", index=index)
    every = create_statistics_pack(root, (), tmp_path / "every", held=True, index=index)

    # Assert: top-level ranking excludes methods and nested functions, not whole files
    assert files["options"]["held"] is False and every["options"]["held"] is True
    assert all(record["held"] is False for record in files["largest"]["ranking"])
    assert files["largest"]["measured"] == 7 and every["largest"]["measured"] == 16

    # Assert: comparing symbol against symbol measures every symbol, holder and held together, so a
    # container is never ranked above a symbol inside it while hiding that symbol from the pack.
    assert sum(every["counts"]["totals"].values()) == every["largest"]["measured"] == 16
    assert {(record["file"], record["name"]) for record in files["largest"]["ranking"]} == {
        ("app/basket.py", "Basket"),
        ("app/basket.py", "standalone"),
        ("web/cart.ts", "Cart"),
        ("web/cart.ts", "sum"),
        ("web/cart.ts", "scale"),
        ("web/ties.ts", "pickOne"),
        ("web/ties.ts", "pickTwo"),
    }
    assert {
        (record["file"], record["name"])
        for record in every["largest"]["ranking"]
        if "basket" in record["file"]
    } == {
        ("app/basket.py", "Basket"),
        ("app/basket.py", "add"),
        ("app/basket.py", "total"),
        ("app/basket.py", "helper"),
        ("app/basket.py", "standalone"),
    }
    assert {record["holder"] for record in every["largest"]["ranking"] if record["held"]} == {
        "app/basket.py:1-10 Basket",
        "app/basket.py:7-10 total",
        "web/cart.ts:1-8 Cart",
        "web/cart.ts:5-7 total",
        "web/cart.ts:12-15 scale",
        "web/ties.ts:1-3 pickOne",
        "web/ties.ts:5-7 pickTwo",
    }
    assert (
        "out of the ranking, leaving only top-level symbols"
        in (tmp_path / "files" / "statistics.md").read_text()
    )
    assert "beside the symbols holding them" in (tmp_path / "every" / "statistics.md").read_text()


def test_a_file_that_could_not_be_measured_stays_a_gap_and_not_a_zero(tmp_path: Path) -> None:
    # Arrange
    root = _write(tmp_path, {"ops/broken.ts": BROKEN, "ops/gone.ts": TIES, "docs/notes.md": NOTES})
    index = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "facts")
    (root / "ops" / "gone.ts").unlink()

    # Act
    pack = create_statistics_pack(
        root,
        (),
        tmp_path / "pack",
        scope=("ops/broken.ts", "ops/gone.ts", "docs/notes.md"),
        index=index,
    )
    document = (tmp_path / "pack" / "statistics.md").read_text()

    # Assert: a file that is not code is skipped, a file that is gone is a gap, and only a file the
    # parser read is measured. A pack that mixed them would report a clean scan of an empty repository
    # for a directory it never looked inside.
    assert pack["scope"]["measured"] == ["ops/broken.ts"]
    assert pack["scope"]["unmeasured"] == ["ops/gone.ts"]
    assert pack["scope"]["skipped"] == ["docs/notes.md"]
    assert pack["scope"]["unparsed"] == ["ops/broken.ts"]
    assert pack["coverage"]["complete"] is False
    assert "were never scanned" in pack["coverage"]["statement"]
    assert "unknown, not absent" in pack["coverage"]["statement"]
    assert "`ops/gone.ts`" in document and "`ops/broken.ts`" in document
    assert "never scanned" in document and "partly parsed" in document
    assert "unknown, not absent" in document and "`scope.unmeasured`" in document
    # A file the parser only partly read keeps the count it gave, marked as a floor and not a zero
    assert pack["counts"]["per_file"]["ops/broken.ts"]["counts"] == {"function": 1, "class": 0}
    assert pack["counts"]["per_file"]["ops/broken.ts"]["symbol_lines"] == 3
    assert "so the count is a floor" in document
    assert pack["largest"]["caveat"] == pack["coverage"]["statement"]


def test_the_manifest_names_every_file_left_out_and_the_report_counts_them(tmp_path: Path) -> None:
    # Arrange: TypeScript built in place, its output ignored beside the sources it came from
    built = {f"src/module{n}.js{suffix}": "built\n" for n in range(400) for suffix in ("", ".map")}
    root = _write(
        tmp_path, {".ignore": "*.js\n*.js.map\ndist/\n", "src/ties.ts": TIES, "dist/a.ts": TIES, **built}
    )

    # Act
    pack = create_statistics_pack(root, (), tmp_path / "pack", fact_cache_dir=tmp_path / "facts")
    report = (tmp_path / "pack" / "statistics.md").read_text()

    # Assert
    assert pack["coverage"]["not_indexed"] == {"dist/": "ignored", **dict.fromkeys(built, "ignored")}
    assert "| ignored | `dist/` | 1 |" in report
    assert "| ignored | `src/` | 800 |" in report
    assert "`coverage.not_indexed` in `statistics.json`" in report
    assert "module7.js" not in report


def test_an_empty_scope_measures_nothing_and_fills_no_gap(
    tmp_path: Path,
) -> None:
    # Arrange
    root = _write(tmp_path, HARD)
    index = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "facts")

    # Act
    pack = create_statistics_pack(root, (), tmp_path / "pack", scope=(), index=index)
    document = (tmp_path / "pack" / "statistics.md").read_text()

    # Assert
    assert pack["scope"] == {
        "prefixes": [],
        "measured": [],
        "skipped": [],
        "unmeasured": [],
        "unparsed": [],
    }
    assert pack["coverage"]["complete"] is False
    assert pack["coverage"]["statement"] == "no scope was given, so no file was measured"
    assert pack["counts"]["totals"] == {"function": 0, "class": 0}
    assert pack["counts"]["per_file"] == {} and pack["ranges"]["per_file"] == {}
    assert pack["largest"]["ranking"] == [] and pack["largest"]["measured"] == 0
    assert pack["largest"]["caveat"] == pack["coverage"]["statement"]

    # Assert: the sections still run, because the operations asked for them, and they report nothing
    assert "## Counts" in document and "## Largest symbols" in document and "## Line ranges" in document
    assert "no scope was given, so no file was measured" in document
    assert "basket.py" not in document and "cart.ts" not in document


def test_the_options_are_recorded_with_the_measurement_that_shaped_them(tmp_path: Path) -> None:
    # Arrange
    root = _write(tmp_path, HARD)
    index = CodeIndex.from_directory(root, fact_cache_dir=tmp_path / "facts")

    # Act
    pack = create_statistics_pack(
        root,
        ("app/",),
        tmp_path / "pack",
        kinds=("class",),
        quote_source=True,
        scope=("app/basket.py",),
        index=index,
    )

    # Assert
    assert pack["options"] == {
        "kinds": ["class"],
        "held": False,
        "limit": None,
        "scope": ["app/basket.py"],
        "quote_source": True,
        "min_lines": None,
        "max_lines": None,
    }
    assert pack["scope"]["prefixes"] == ["app/"]
    assert pack["scope"]["measured"] == ["app/basket.py"] and pack["scope"]["skipped"] == []
    assert pack["counts"]["totals"] == {"class": 1}
    assert pack["counts"]["per_file"]["app/basket.py"]["counts"] == {"class": 1}
    assert "web/cart.ts" not in (tmp_path / "pack" / "statistics.json").read_text()

    # Assert: what was quoted is the file's own text for that range, line for line
    quoted = _listed(pack)
    assert quoted and all(record["source"] for record in quoted)
    basket = (root / "app" / "basket.py").read_text().splitlines()
    for record in quoted:
        start, end = record["lines"]
        assert record["source"].splitlines() == basket[start - 1 : end]
    document = (tmp_path / "pack" / "statistics.md").read_text()
    assert "Kinds measured: class" in document and "quoted in this pack: yes" in document


def test_a_pack_refuses_an_occupied_output_and_an_option_that_measures_nothing(
    tmp_path: Path,
) -> None:
    # Arrange
    root = _write(tmp_path, HARD)
    occupied = tmp_path / "occupied"
    occupied.mkdir(parents=True)
    (occupied / "evidence.json").write_text("{}\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="needs somewhere to write"):
        create_statistics_pack(root, (), None)
    with pytest.raises(ValueError, match="is a file"):
        create_statistics_pack(root, (), root / "web" / "broken.ts")
    with pytest.raises(ValueError, match="not empty"):
        create_statistics_pack(root, (), occupied)
    with pytest.raises(ValueError, match="nothing called catalog"):
        create_statistics_pack(root, (), tmp_path / "catalog", operations=("catalog",))
    with pytest.raises(ValueError, match="nothing to measure called method"):
        create_statistics_pack(root, (), tmp_path / "method", kinds=("method",))
    with pytest.raises(ValueError, match="needs a positive number"):
        create_statistics_pack(root, (), tmp_path / "zero", limit=0)

    # Assert: a refused call leaves no half-written pack behind
    assert not (tmp_path / "catalog").exists()
    assert not (tmp_path / "method").exists()
    assert not (tmp_path / "zero").exists()
