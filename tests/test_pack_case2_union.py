"""Canonical union admission and the actual paid-shape body/Item contract."""

import hashlib
import importlib
import json
from dataclasses import asdict
from pathlib import Path

from shop_search import shop_index

from jev_navigator.index.units import items_to_judge, read_ranges
from jev_navigator.search_plan import Approach, FileUnits, SearchPlan, execute_plan


def test_union_deduplicates_source_geometry_and_keeps_outside_plan_bodies_in_the_paid_reader(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/pack_case2"))
    union = importlib.import_module("union_prepare")
    shape = importlib.import_module("paid_shape")
    index = shop_index(
        tmp_path / "repo",
        {"guard.py": "def guard():\n    return 7\n", "outside.py": "def outside():\n    return 9\n"},
    )
    result = execute_plan(
        index,
        SearchPlan((Approach(1, FileUnits("outside.py"), ()), Approach(2, FileUnits("guard.py"), ()))),
        box_chars=70000,
    )
    plan = [
        {
            "unit": asdict(candidate.unit),
            "approach_ranks": [approach.rank for approach in candidate.approaches],
            "items": [
                {**asdict(item), "code": read_ranges(index, item.file, item.ranges)}
                for item in items_to_judge(candidate.unit)
            ],
        }
        for candidate in result.candidates
    ]
    guard = next(row for row in plan if row["unit"]["path"] == "guard.py")
    identity = union.canonical_id(guard["unit"])
    # A census request's masked spelling has a different ID for the same actual source.
    census = {
        identity: {
            "unit": {**guard["unit"], "id": identity},
            "item_unit": {**guard["unit"], "id": "recorded-mask-alias"},
            "approach_ranks": [],
            "scent": 0.9,
            "walk": 0.7,
            "features": [0.9, 0.7, 0, 0, 0, 0],
            "combined_score": 1.6,
            "ranking_ordinal": 1,
            "rank_source": "case1_combined_cv",
            "source_flags": {"plan": False, "ranking": True},
            "origin": {"ranking": {"request_ids": ["recorded-mask-alias"]}},
        }
    }
    candidates, orders = union.merge_plan(census, plan)
    assert len(candidates) == 2
    assert orders["ranking_only"] == [identity]
    assert orders["union"][0] == identity  # Frozen Case1 wins the ordinal tie.
    assert candidates[identity]["source_flags"] == {"plan": True, "ranking": True}
    outside = candidates[orders["union"][1]]
    assert outside["source_flags"] == {"plan": True, "ranking": False}
    assert outside["scent"] is None and outside["walk"] is None
    assert outside["rank_source"] == "plan_ordinal_fallback"
    path = tmp_path / "union-candidates.jsonl"
    emitted = [union.actual_record(candidates[key], index) for key in orders["union"]]
    path.write_text("".join(json.dumps(record) + "\n" for record in emitted))
    read = list(shape.entries(path))
    assert [body["code"] for body, _ in read] == [
        "def guard():\n    return 7",
        "def outside():\n    return 9",
    ]
    assert [(item.file, item.ranges) for _, item in read] == [
        ("guard.py", ((1, 2),)),
        ("outside.py", ((1, 2),)),
    ]
    assert union.reaches(emitted, {"file": "outside.py", "line": 2})
    assert not union.reaches(emitted, {"file": "outside.py", "line": 3, "floor": True})


def test_historical_segment_binding_hash_names_the_emitted_body(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/pack_case2"))
    union = importlib.import_module("union_prepare")
    index = shop_index(tmp_path / "repo", {"a.py": "def guard():\n    return 7\n"})
    unit = (
        execute_plan(index, SearchPlan((Approach(1, FileUnits("a.py"), ()),)), box_chars=70000)
        .candidates[0]
        .unit
    )
    # The retained census binder narrows source ranges without changing the holder hash.
    segment = {**asdict(unit), "ranges": [[2, 2]]}
    record = union.actual_record({"unit": segment, "item_unit": segment, "origin": {}}, index)
    assert record["items"][0]["code"] == "    return 7"
    assert record["items"][0]["ranges"] == ((2, 2),)
    assert record["unit"]["content_sha256"] == hashlib.sha256(b"    return 7").hexdigest()
    assert record["origin"]["binding_content_sha256"] == unit.content_sha256
