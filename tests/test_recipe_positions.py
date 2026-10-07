import runpy
from pathlib import Path

from shop_search import shop_index

from jev_navigator.index.units import Reading, items_to_judge, list_units

request_rank = runpy.run_path(Path(__file__).parents[1] / "evaluations/pack_recipes/positions.py")[
    "request_rank"
]


def test_deciding_line_position_counts_split_pieces_before_next_unit(tmp_path):
    body = "".join(f"    value_{i} = {i}\n" for i in range(1080))
    index = shop_index(tmp_path, {"app.py": f"def large():\n{body}\ndef small():\n    return 1\n"})
    units = list_units(index, ["app.py"], box_chars=2000, reading=Reading.MIXED).units
    large, small = units
    items = items_to_judge(large)

    assert len(items) > 16
    assert request_rank(units, "app.py", items[16].ranges[0][0]) == 2
    assert request_rank(units, "app.py", small.start) == (len(items) // 16) + 1
    assert request_rank(units, "absent.py", 1) is None
