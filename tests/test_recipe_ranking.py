import runpy
from pathlib import Path

from shop_search import shop_index

from jev_navigator.index.units import LineAnchor
from jev_navigator.recipe_configuration import SearchRecipe
from jev_navigator.sources import ANCHORS, FILES, Seeds

RecipeRanker = runpy.run_path(Path(__file__).parents[1] / "evaluations/pack_recipes/ranking.py")[
    "RecipeRanker"
]


def test_recipe_uses_shared_combined_ranker_without_changing_evidence(tmp_path):
    index = shop_index(
        tmp_path,
        {"app.py": "def unrelated():\n    return 1\n\ndef validate_credentials():\n    return False\n"},
    )
    recipe = SearchRecipe("local", 2, (FILES,))
    seeds = Seeds(files=("app.py",))
    original = recipe.gather(index, seeds, box_chars=76_800)
    cited = SearchRecipe("cited", 1, (ANCHORS,)).gather(
        index, Seeds(anchors=(LineAnchor("app.py", 1),)), box_chars=76_800
    )
    ranked = recipe.gather(
        index, seeds, box_chars=76_800, rank=RecipeRanker(index, "validate credentials", cited.units)
    )

    assert ranked.units[0].symbol == "validate_credentials"
    assert {unit.id: unit for unit in ranked.units} == {unit.id: unit for unit in original.units}
    assert ranked.reached_by == original.reached_by
