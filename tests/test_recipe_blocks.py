from pathlib import Path

import pytest
from shop_search import shop_index

from jev_navigator.index.units import LineAnchor
from jev_navigator.recipe_blocks import (
    OWNER_DEFINITIONS,
    ChainAttachment,
    PresencePlace,
    PresenceStatus,
    guard_chain,
    presence_check,
    value_source,
)
from jev_navigator.recipe_configuration import SearchRecipe
from jev_navigator.sources import ANCHORS, CALLEES, FILES, Seeds

BOX = 76_800


def test_chain_walks_downward_through_imports_decorators_and_explicit_router_wiring(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "routes.py": "from guards import decorate, validate\n\n@decorate\ndef handler(x):\n"
            "    validate(x)\n    return x\n\ndef caller(x):\n    return handler(x)\n",
            "guards.py": "def decorate(fn):\n    return fn\n\ndef validate(x):\n"
            "    return check(x)\n\ndef check(x):\n    return bool(x)\n\n"
            "def middleware(x):\n    return validate(x)\n",
            "unrelated.py": "def validate(x):\n    return False\n",
        },
    )
    entry = index.find_definition("handler")[0]
    middleware = index.find_definition("middleware")[0]
    attachment = ChainAttachment(entry, middleware, LineAnchor("routes.py", 1), "router middleware", True)

    chain = guard_chain(index, entry, attachments=(attachment,))

    assert {span.name for span in chain.units} == {"handler", "decorate", "validate", "check", "middleware"}
    assert all(span.file != "unrelated.py" for span in chain.units)
    assert "caller" not in {span.name for span in chain.units}
    assert any(link.relation == "router middleware" and link.at.line == 1 for link in chain.links)


def test_chain_preserves_unknown_receivers_and_depth_cuts(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "app.py": "def entry(x):\n    guard(x)\n    unknown.check(x)\n\ndef guard(x):\n"
            "    return check(x)\n\ndef check(x):\n    return bool(x)\n",
        },
    )
    entry = index.find_definition("entry")[0]

    chain = guard_chain(index, entry, depth=1)

    assert {span.name for span in chain.units} == {"entry", "guard"}
    assert [span.name for span in chain.depth_cut] == ["guard"]
    assert any("receiver" in link.reason for link in chain.unknown)


def test_named_recipe_resolves_an_import_alias_without_admitting_same_named_owners(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "app.py": "from right import load as read\n\ndef entry():\n    return read()\n",
            "right.py": "def load():\n    return 1\n",
            "wrong.py": "def load():\n    return 2\n",
        },
    )
    recipe = SearchRecipe("named", 1, (OWNER_DEFINITIONS,))

    result = recipe.gather(index, Seeds(names=("read",), files=("app.py",)), box_chars=BOX)

    assert [(unit.path, unit.symbol) for unit in result.units] == [("right.py", "load")]
    assert all(
        reach.source == "owner_definition" for reaches in result.reached_by.values() for reach in reaches
    )


def test_value_sources_include_default_environment_override_and_docs(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "settings.py": 'import os\nTIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))\n',
            "overrides.py": "from settings import TIMEOUT\nTIMEOUT = 5\n",
            "docs/settings.md": "# Settings\nREQUEST_TIMEOUT defaults to 30 seconds.\n",
            "other.py": 'NOT_REQUEST_TIMEOUT = "irrelevant"\n',
        },
    )

    reaches = value_source(index, "REQUEST_TIMEOUT", aliases=("TIMEOUT",))

    assert {reach.at.file for reach in reaches} == {"settings.py", "overrides.py", "docs/settings.md"}
    scoped = value_source(index, "REQUEST_TIMEOUT", files=("docs/settings.md",))
    assert {reach.at.file for reach in scoped} == {"docs/settings.md"}


def test_presence_reports_literal_absence_only_over_valid_checked_places(tmp_path: Path):
    index = shop_index(tmp_path, {"app.py": "before = 1\nauthorize()\n", "readme.md": "Nothing here.\n"})

    partial = presence_check(
        index,
        "authorize",
        (
            PresencePlace("app.py", 1, 1),
            PresencePlace("readme.md"),
            PresencePlace("withheld.py"),
        ),
    )
    absent = presence_check(index, "authorize", (PresencePlace("app.py", 1, 1), PresencePlace("readme.md")))
    present = presence_check(index, "authorize", (PresencePlace("app.py"), PresencePlace("withheld.py")))

    assert partial.status is PresenceStatus.UNKNOWN and partial.checked == 2
    assert absent.status is PresenceStatus.ABSENT and absent.checked == 2
    assert present.status is PresenceStatus.PRESENT and present.checked == 1
    assert [(hit.file, hit.line) for hit in present.places[0].matches] == [("app.py", 2)]
    assert presence_check(index, "authorize", ()).status is PresenceStatus.UNKNOWN
    assert presence_check(index, "authorize", (PresencePlace("app.py", 1, 99),)).checked == 0


def test_presence_marks_a_file_removed_after_inventory_unknown(tmp_path: Path):
    index = shop_index(tmp_path, {"gone.md": "authorize\n"})
    (tmp_path / "gone.md").unlink()

    result = presence_check(index, "authorize", (PresencePlace("gone.md"),))

    assert result.status is PresenceStatus.UNKNOWN and result.checked == 0


def test_recipe_deduplicates_sources_and_expands_only_confirmed_units(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "app.py": "def entry():\n    return helper()\n\ndef helper():\n    return 1\n",
            "notes.md": "# Notes\nentry is public.\n",
        },
    )
    recipe = SearchRecipe("local", 1, (ANCHORS, FILES), (CALLEES,))
    seeds = Seeds(anchors=(LineAnchor("app.py", 2),), files=("app.py", "notes.md"))

    gathered = recipe.gather(index, seeds, box_chars=BOX)
    entry = next(unit for unit in gathered.units if unit.symbol == "entry")
    followup = recipe.continue_from(index, (entry,), box_chars=BOX)

    assert len(gathered.units) == 3
    assert len(gathered.reached_by[entry.id]) == 2
    assert [unit.symbol for unit in followup.units] == ["helper"]
    assert recipe.continue_from(index, (), box_chars=BOX).units == ()
    with pytest.raises(ValueError, match="reorder every candidate"):
        recipe.gather(index, seeds, box_chars=BOX, rank=lambda units: units[:1])
    # A caller's ranker changes order while preserving identity and evidence.
    ranked = recipe.gather(index, seeds, box_chars=BOX, rank=reversed)
    assert ranked.units == tuple(reversed(gathered.units))


def test_unproven_wiring_is_not_walked(tmp_path: Path):
    index = shop_index(tmp_path, {"app.py": "def entry():\n    pass\n\ndef wrapper():\n    pass\n"})
    entry, wrapper = index.find_definition("entry")[0], index.find_definition("wrapper")[0]

    chain = guard_chain(
        index, entry, attachments=(ChainAttachment(entry, wrapper, LineAnchor("app.py", 1), "wrapper"),)
    )

    assert chain.units == (entry,)
    assert len(chain.unknown) == 1 and chain.unknown[0].target == wrapper
