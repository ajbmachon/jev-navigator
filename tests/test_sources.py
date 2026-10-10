from __future__ import annotations

from pathlib import Path

import pytest
from conftest import WEBSITE_QUERIES, labelled
from git_repos import commit_files
from shop_search import shop_index

from jev_navigator.directives.find_all import find_all
from jev_navigator.directives.frontier import VALUE
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import LineAnchor, RangeAnchor, Unit, list_units, read_ranges
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.judge import Judge
from jev_navigator.selection.scent import ScentIndex, ScentIndexTooLargeError, scent_document, unit_scent
from jev_navigator.sources import (
    ANCHORS,
    CALLEES,
    CALLERS,
    CLIENT_CALLS,
    DEFINITIONS,
    FILE_WORDS,
    FILES,
    IMPORTERS,
    IMPORTS,
    MODELS,
    NAMED_FILES,
    NAMES,
    REFERENCES,
    TEXT_NAMED_FILES,
    TEXT_NAMES,
    Reach,
    ScentSource,
    Seeds,
)
from jev_navigator.testing import ScriptedJevClient

BOX = JEV_INPUT_LIMITS.box_chars
SCHEMA = "prisma/schema.prisma"


def units_by_symbol(index, *files: str) -> dict[str, Unit]:
    return {unit.symbol: unit for unit in list_units(index, files, box_chars=BOX).units}


def places(reaches: list[Reach]) -> list[tuple]:
    return [(reach.at, reach.seed, reach.distance) for reach in reaches]


def test_the_anchor_source_reaches_each_anchor_at_distance_zero(tmp_path: Path) -> None:
    # Arrange
    index = shop_index(tmp_path)
    anchors = (LineAnchor("orders/service.py", 5), RangeAnchor("orders/limits.py", 4, 6))

    # Act
    reaches = ANCHORS.reach(index, Seeds(anchors=anchors))

    # Assert
    assert places(reaches) == [
        (anchors[0], "orders/service.py:5", 0),
        (anchors[1], "orders/limits.py:4-6", 0),
    ]


def test_the_file_source_puts_the_anchors_files_and_their_imports_nearer_than_other_files(
    tmp_path: Path,
) -> None:
    # Arrange: service.py holds the anchor and imports limits.py; invoice.py is neither
    index = shop_index(tmp_path)
    seeds = Seeds(
        files=("billing/invoice.py", "orders/limits.py", "orders/service.py"),
        anchors=(LineAnchor("orders/service.py", 5),),
    )

    # Act
    reaches = FILES.reach(index, seeds)

    # Assert
    assert [(reach.at, reach.distance) for reach in reaches] == [
        ("billing/invoice.py", 2),
        ("orders/limits.py", 1),
        ("orders/service.py", 1),
    ]


def test_the_name_source_reaches_every_hit_of_the_rarest_name_first_by_that_name(tmp_path: Path) -> None:
    # Arrange: MAX_ITEMS is on five lines and check_limit on six
    index = shop_index(tmp_path)

    # Act
    reaches = NAMES.reach(index, Seeds(names=("check_limit", "MAX_ITEMS")))

    # Assert
    assert [reach.seed for reach in reaches] == ["MAX_ITEMS"] * 5 + ["check_limit"] * 6
    assert all(reach.names == {reach.seed} and reach.distance == 3 for reach in reaches)
    assert reaches[0].at == LineAnchor("orders/api.ts", 2)


def test_a_text_name_source_keeps_only_hits_in_text_files_that_are_not_lockfiles(tmp_path: Path) -> None:
    # Arrange
    files = {
        "app.py": "RETRY_LIMIT = 3\n",
        "config/retry.yaml": "retry:\n  RETRY_LIMIT: 3\n",
        "package-lock.json": '{"RETRY_LIMIT": 3}\n',
    }
    index = shop_index(tmp_path, files)

    # Act
    in_code = NAMES.reach(index, Seeds(names=("RETRY_LIMIT",)))
    in_text = TEXT_NAMES.reach(index, Seeds(names=("RETRY_LIMIT",)))

    # Assert
    assert {reach.at.file for reach in in_code} == set(files)
    assert [reach.at for reach in in_text] == [LineAnchor("config/retry.yaml", 2)]


def test_the_caller_and_callee_sources_reach_the_functions_on_either_side_of_a_seed_unit(
    tmp_path: Path,
) -> None:
    # Arrange: place_order and a test call check_limit
    index = shop_index(tmp_path)
    check_limit = units_by_symbol(index, "orders/limits.py")["check_limit"]
    place_order = units_by_symbol(index, "orders/service.py")["place_order"]

    # Act
    callers = CALLERS.reach(index, Seeds(units=(check_limit,)))
    callees = CALLEES.reach(index, Seeds(units=(place_order,)))

    # Assert
    assert places(callers) == [
        (LineAnchor("orders/service.py", 4), check_limit.id, 1),
        (LineAnchor("tests/test_limits.py", 4), check_limit.id, 1),
    ]
    assert places(callees) == [(LineAnchor("orders/limits.py", 4), place_order.id, 1)]


def test_a_unit_that_is_not_a_function_reaches_no_callers_or_callees(tmp_path: Path) -> None:
    # Arrange: limits.py's top-level code is the MAX_ITEMS constant
    index = shop_index(tmp_path)
    top_level = units_by_symbol(index, "orders/limits.py")["<top level>"]

    # Act
    reached = [
        *CALLERS.reach(index, Seeds(units=(top_level,))),
        *CALLEES.reach(index, Seeds(units=(top_level,))),
    ]

    # Assert
    assert reached == []


def test_the_definition_and_reference_sources_reach_a_names_definition_and_its_uses(tmp_path: Path) -> None:
    # Arrange
    index = shop_index(tmp_path)
    seeds = Seeds(names=("check_limit", "MAX_ITEMS"))

    # Act
    definitions = DEFINITIONS.reach(index, seeds)
    references = REFERENCES.reach(index, seeds)

    # Assert: the comparison with MAX_ITEMS is a use; calls are the caller source's
    assert places(definitions) == [
        (LineAnchor("orders/limits.py", 4), "check_limit", 1),
        (LineAnchor("orders/limits.py", 1), "MAX_ITEMS", 1),
    ]
    assert places(references) == [(LineAnchor("orders/limits.py", 5), "MAX_ITEMS", 2)]
    assert [reach.names for reach in definitions] == [{"check_limit"}, {"MAX_ITEMS"}]


def test_the_import_sources_follow_imports_out_of_and_into_the_anchors_files(tmp_path: Path) -> None:
    # Arrange
    index = shop_index(tmp_path)

    # Act
    imported = IMPORTS.reach(index, Seeds(anchors=(LineAnchor("orders/service.py", 5),)))
    importing = IMPORTERS.reach(index, Seeds(anchors=(LineAnchor("orders/limits.py", 4),)))

    # Assert
    assert places(imported) == [("orders/limits.py", "orders/service.py", 1)]
    assert places(importing) == [
        ("orders/service.py", "orders/limits.py", 1),
        ("tests/test_limits.py", "orders/limits.py", 1),
    ]


def test_the_named_file_sources_split_the_files_a_description_names_into_code_and_text(
    tmp_path: Path,
) -> None:
    # Arrange
    files = {"jobs/sweep.py": "def sweep():\n    return 1\n", "deploy/ci.yml": "jobs: [sweep]\n"}
    index = shop_index(tmp_path, files)
    seeds = Seeds(texts=("the sweep in jobs/sweep.py that ci.yml schedules",))

    # Act
    code = NAMED_FILES.reach(index, seeds)
    text = TEXT_NAMED_FILES.reach(index, seeds)

    # Assert
    assert places(code) == [("jobs/sweep.py", "jobs/sweep.py", 1)]
    assert places(text) == [("deploy/ci.yml", "ci.yml", 1)]


def test_the_model_sources_link_prisma_queries_and_the_models_they_query(
    tmp_path: Path, umami_schema: str
) -> None:
    # Arrange: website.ts queries model Website, lines 98 to 131 of the schema
    index = shop_index(tmp_path, {SCHEMA: umami_schema, "src/website.ts": WEBSITE_QUERIES})
    update_website = units_by_symbol(index, "src/website.ts")["updateWebsite"]
    website = units_by_symbol(index, SCHEMA)["model Website"]

    # Act
    queried = MODELS.reach(index, Seeds(units=(update_website,)))
    calls = CLIENT_CALLS.reach(index, Seeds(units=(website,)))

    # Assert
    assert places(queried) == [(LineAnchor(SCHEMA, 98), update_website.id, 1)]
    assert places(calls) == [
        (LineAnchor("src/website.ts", 4), website.id, 1),
        (LineAnchor("src/website.ts", 8), website.id, 1),
    ]


def test_find_all_judges_only_the_units_the_sources_a_caller_composes_reach(tmp_path: Path) -> None:
    # Arrange: the definitions of two names, with no files and no name hits
    index = shop_index(tmp_path)
    judge = Judge(labelled({}), items_per_request=4)

    # Act
    result = find_all(
        index, judge, {"limit": "the item limit"}, names=["check_limit", "MAX_ITEMS"], sources=(DEFINITIONS,)
    )

    # Assert
    assert {(unit.path, unit.symbol) for unit in result.units} == {
        ("orders/limits.py", "check_limit"),
        ("orders/limits.py", "<top level>"),
    }
    assert set(result.entered_by.values()) == {DEFINITIONS.name}
    assert result.sources == (DEFINITIONS, CALLERS, CALLEES)
    assert result.stopped_by == "scope_examined"


def test_under_a_ranked_policy_a_unit_two_sources_reach_counts_at_the_nearer_source(tmp_path: Path) -> None:
    # Arrange: check_limit's definition line is also one of its name's hits, the name source listed first
    index = shop_index(tmp_path)
    judge = Judge(labelled({}), items_per_request=4)

    # Act
    result = find_all(
        index,
        judge,
        {"limit": "the item limit"},
        names=["check_limit"],
        sources=(NAMES, DEFINITIONS),
        policy=VALUE,
    )

    # Assert
    check_limit = next(unit for unit in result.units if unit.symbol == "check_limit")
    place_order = next(unit for unit in result.units if unit.symbol == "place_order")
    assert result.entered_by[check_limit.id] == DEFINITIONS.name
    assert result.features["limit"][check_limit.id].distance == 1
    assert (result.entered_by[place_order.id], result.features["limit"][place_order.id].distance) == (
        NAMES.name,
        3,
    )


def test_word_matched_files_keep_source_priority_under_a_request_cap(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "a/limit.py": "def alpha():\n    return 1\n",
            "z/feature_limit.py": "def beta():\n    return 2\n",
            "scope.py": "def scoped():\n    return 3\n",
        },
    )
    client = ScriptedJevClient()
    result = find_all(
        index,
        Judge(client, max_calls=1, items_per_request=1),
        {"p": "feature limit"},
        files=["scope.py"],
        sources=(FILE_WORDS, FILES),
    )
    assert result.stopped_by == "budget"
    assert [score.unit.path for score in result.scores("p")] == ["z/feature_limit.py"]
    assert client.requests[0][0]["items"][0]["file"] == "z/feature_limit.py"


def test_owner_qualified_callees_with_the_same_name_reach_both_bound_owners(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "app.py": (
                "import worker\nimport guard\n\ndef run():\n    worker.decide()\n    return guard.decide()\n"
            ),
            "worker.py": "def decide():\n    return 1\n",
            "guard.py": "def decide():\n    return 2\n",
            "unrelated.py": "def decide():\n    return 3\n",
        },
    )
    unit = units_by_symbol(index, "app.py")["run"]
    assert {reach.at for reach in CALLEES.reach(index, Seeds(units=(unit,)))} == {
        LineAnchor("worker.py", 1),
        LineAnchor("guard.py", 1),
    }


SCENTED = {
    "shop/limits.py": (
        "MAX_ITEMS = 4\n\n\n"
        "def check_item_limit(order):\n"
        "    # a refund never passes through here\n"
        "    if len(order.items) > MAX_ITEMS:\n"
        '        raise ValueError("order over the item limit")\n\n\n'
        "def item_count(order):\n    return len(order.items)\n"
    ),
    "shop/refunds.py": "def issue_refund(order):\n    return -order.total\n",
    "shop/quota.py": "def enforce_quota(order):\n    return order.quota\n",
    "shop/render.py": "def render_page(page):\n    return page.html\n",
}
LIMIT_POINT = "code that refuses an order over the item limit"


def test_the_scent_source_reaches_the_best_bm25_units_in_order_after_name_hits(tmp_path: Path) -> None:
    # Arrange: the expected order, scored unit by unit from each unit's own code
    commit_files(tmp_path, SCENTED)
    index = CodeIndex.from_git(tmp_path)
    seeds = Seeds(names=("MAX_ITEMS",), texts=(LIMIT_POINT,))
    units = list_units(index, index.files, box_chars=BOX).units
    scores = ScentIndex(
        scent_document(unit.id, unit.path, unit.symbol, read_ranges(index, unit.path, unit.ranges))
        for unit in units
    ).scores(f"{LIMIT_POINT} MAX_ITEMS")
    matched = [unit for unit in units if scores[unit.id] > 0]
    best = sorted(matched, key=lambda unit: (-scores[unit.id], unit.id))

    # Act
    reaches = ScentSource(limit=3).reach(index, seeds)

    # Assert
    assert places(reaches) == [
        (LineAnchor(unit.path, unit.start), f"bm25 {scores[unit.id]:.3f}", 4 + rank)
        for rank, unit in enumerate(best[:3])
    ]
    assert {reach.source for reach in reaches} == {"scent"} and not any(reach.names for reach in reaches)
    assert reaches[0].at == LineAnchor("shop/limits.py", 4), "check_item_limit matches the most words"
    assert "shop/render.py" not in {reach.at.file for reach in ScentSource(limit=99).reach(index, seeds)}


def test_the_scent_index_is_built_once_per_code_index_and_refuses_more_units_than_its_bound(
    tmp_path: Path,
) -> None:
    commit_files(tmp_path, SCENTED)
    index = CodeIndex.from_git(tmp_path)

    bound = "lists 6 units, more than the scent index's bound of 5"
    with pytest.raises(ScentIndexTooLargeError, match=bound):
        ScentSource(max_units=5).reach(index, Seeds(texts=(LIMIT_POINT,)))

    assert unit_scent(index, max_units=6) is unit_scent(index, max_units=6)


def test_the_scent_source_ranks_only_inside_the_seeds_scope_before_cutting_its_limit(tmp_path: Path) -> None:
    # Arrange: a test file that matches the point best, outside a scope that leaves tests out
    commit_files(
        tmp_path,
        {
            **SCENTED,
            "tests/test_limits.py": (
                "def test_refuses_an_order_over_the_item_limit(order_over_item_limit):\n"
                "    refuse_order_over_item_limit(order_over_item_limit)\n"
            ),
        },
    )
    index = CodeIndex.from_git(tmp_path)
    unscoped = Seeds(texts=(LIMIT_POINT,))

    everywhere = ScentSource(limit=99).reach(index, unscoped)

    # Act
    in_scope = ScentSource(limit=2).reach(
        index, Seeds(texts=(LIMIT_POINT,), in_scope=lambda path: not path.startswith("tests/"))
    )

    # Assert
    assert everywhere[0].at.file == "tests/test_limits.py"
    kept = [reach.at for reach in everywhere if reach.at.file != "tests/test_limits.py"]
    assert [(reach.at, reach.distance) for reach in in_scope] == [(kept[0], 4), (kept[1], 5)]
