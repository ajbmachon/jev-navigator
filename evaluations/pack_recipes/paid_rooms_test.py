"""Integration checks against the archived Engine and find-eval dependencies.

Run explicitly with the same archived PYTHONPATH as the paid trial. These fixtures
exercise actual packet fitting and the original allocator without a Jev connection.
"""

from dataclasses import replace
from subprocess import run

import pytest
from paid_rooms import measure_case, measurement_budget


@pytest.fixture
def case(tmp_path):
    from enginepy.workflows.document_analysis.evidence_pack import (
        EvidencePack,
        PackCost,
        PackRequest,
        PackUnit,
        RankedUnit,
    )
    from enginepy.workflows.document_analysis.skeptic_packet import PacketRegion
    from jev_navigator.judgments.profiles import ROLES_V2

    lines = ["small_a = 1", "gap = 2", "small_b = 3"]
    lines += [f"value_{n} = '{'x' * 90}'" for n in range(350)]
    (tmp_path / "app.py").write_text("\n".join(lines) + "\n")
    run(["git", "init", "-q", str(tmp_path)], check=True)
    run(["git", "-C", str(tmp_path), "add", "app.py"], check=True)

    # Real source admission; parsing isn't needed for this post-judging boundary.
    from enginepy.workflows.document_analysis.code_relations import CodeRelations

    relations = CodeRelations(str(tmp_path), ({}, {}))
    floor = (PacketRegion("cited", "anchor", "app.py", 2, 2, "2: gap = 2"),)
    request = PackRequest("finding", {"p0": "values"}, ("app.py",), (), (), floor)
    candidates = [
        {"id": "small", "path": "app.py", "ranges": [[1, 1], [3, 3]], "symbol": "small", "pieces": []},
        {"id": "large", "path": "app.py", "ranges": [[4, 353]], "symbol": "large", "pieces": []},
    ]
    selected, items = [], []
    for candidate in candidates:
        runs = tuple(map(tuple, candidate["ranges"]))
        raw = "\n".join(lines[n - 1] for a, b in runs for n in range(a, b + 1))
        numbered = "\n".join(f"{n}: {lines[n - 1]}" for a, b in runs for n in range(a, b + 1))
        ranked = RankedUnit(candidate["id"], "app.py", runs, runs, candidate["symbol"], 0.9, 1)
        selected.append(PackUnit(ranked, ("p0 at P=0.900",), numbered))
        items.append({"file": "app.py", "code": raw})
    pack = EvidencePack("finding", floor, tuple(selected), ("judged 2 units",), PackCost(1, 123, ("budget",)))
    questions = {
        f"{check.name}#{slot}": check.to_question(f"items[{slot}]")
        for slot in range(2)
        for check in ROLES_V2.questions("p0")
    }
    group = {
        "key": "actual-body",
        "state": {"items": items, "targets": request.points},
        "questions": questions,
        "response": {
            "answers": {key: {"type": "noul", "noul": 0.9} for key in questions},
            "usage": {"input_tokens": 123},
            "model": "jev-1.13.0",
        },
    }
    labels = [{"file": "app.py", "first_line": n, "last_line": n} for n in (1, 2, 3, 353)]
    record = {"case": "example", "recipe": "pack-local", "primary": True, "units": candidates}
    return record, pack, relations, request, [group], labels


def test_actual_bodies_and_original_allocators_produce_free_room_curve(case):
    record, pack, relations, request, groups, labels = case
    result = measure_case(*case)
    assert result["provider_calls"] == 0
    assert result["judged_reach"]["references"] == 3
    assert [label["judged"] for label in result["judged_reach"]["labels"]] == [True, False, True, True]
    assert result["judged_reach"]["unbound"] == []
    assert result["judged_reach"]["complete_pairs"] == 2
    for room in ("7200", "20000", "36000"):
        curve = result["rooms"][room]
        for path in ("lab", "native"):
            assert [label["delivered"] for label in curve[path]["labels"]] == (
                [True, True, True, False] if room == "7200" else [True, True, True, True]
            )
            assert all(window["text"] for window in curve[path]["windows"])
        assert curve["native"]["chars"] <= int(room) * 4
        assert curve["budget"]["floor_chars"] == int(room) * 2
        assert curve["budget"]["header_chars"] == curve["budget"]["facts_chars"] == int(room) // 5
    # The source extent spans the gap; only its floor prints that missing source line.
    no_floor = replace(request, floor=())
    without_floor = measure_case(record, replace(pack, floor=()), relations, no_floor, groups, labels)
    for curve in without_floor["rooms"].values():
        assert not curve["native"]["labels"][1]["delivered"]
        assert not curve["lab"]["labels"][1]["delivered"]
    assert request.budget.tokens == pack.budget.tokens == 36_000
    assert result["native_before_fitting"]["references"] == 4
    assert result["rooms"]["7200"]["native"]["fitting_lost_references"] == 1
    assert result["rooms"]["36000"]["native"]["fitting_lost_references"] == 0


def test_response_completion_order_does_not_change_lab_tie_selection(case):
    from jev_navigator.judgments.profiles import ROLES_V2

    *inputs, groups, labels = case
    original = groups[0]
    separated = []
    for slot, item in enumerate(original["state"]["items"]):
        questions = {f"{check.name}#0": check.to_question("items[0]") for check in ROLES_V2.questions("p0")}
        separated.append(
            {
                **original,
                "key": f"body-{slot}",
                "state": {**original["state"], "items": [item]},
                "questions": questions,
                "response": {
                    **original["response"],
                    "answers": {key: {"type": "noul", "noul": 0.9} for key in questions},
                },
            }
        )
    in_order = measure_case(*inputs, separated, labels)
    reversed_order = measure_case(*inputs, list(reversed(separated)), labels)
    for room in in_order["rooms"]:
        assert in_order["rooms"][room]["lab"]["selected"] == reversed_order["rooms"][room]["lab"]["selected"]
        assert in_order["rooms"][room]["lab"]["labels"] == reversed_order["rooms"][room]["lab"]["labels"]


def test_partial_answers_do_not_become_complete_lab_judgments(case):
    *inputs, groups, labels = case
    group = groups[0]
    group["response"]["answers"].pop(next(iter(group["questions"])))
    result = measure_case(*inputs, groups, labels)
    assert result["judged_reach"]["complete_pairs"] == 1
    assert len(result["judged_reach"]["bindings"]) == 2
    assert result["rooms"]["36000"]["lab"]["observed_pairs"] == 1


def test_unbound_request_body_is_reported_without_inventing_span(case):
    record, pack, relations, request, groups, labels = case
    groups[0]["state"]["items"][0]["code"] = "not in source"
    result = measure_case(record, pack, relations, request, groups, labels)
    assert result["judged_reach"]["complete_pairs"] == 1
    assert result["judged_reach"]["unbound"][0]["binding"] == "unbound body"
    assert result["judged_reach"]["references"] == 1


def test_historical_evaluation_budget_does_not_change_production_minimum():
    from enginepy.workflows.document_analysis.skeptic_packet import PacketBudget

    with pytest.raises(ValueError, match="at least 20,000"):
        PacketBudget(7_200)
    budget = measurement_budget(7_200)
    assert budget.chars == 28_800
    assert budget.floor_chars == 14_400
    assert budget.header_chars == budget.facts_chars == 1_440
    assert isinstance(budget, PacketBudget)


@pytest.mark.parametrize("shown_file,expected_references", [("[MASKED].py", 3), ("missing/[MASKED].py", 0)])
def test_masked_request_paths_bind_uniquely_or_remain_unknown(case, shown_file, expected_references):
    for item in case[4][0]["state"]["items"]:
        item["file"] = shown_file
    result = measure_case(*case)
    assert result["judged_reach"]["references"] == expected_references
    assert len(result["judged_reach"]["unbound"]) == (0 if expected_references else 2)


def test_masked_path_binding_does_not_read_excluded_candidates(case):
    from pathlib import Path

    from enginepy.workflows.document_analysis.code_relations import CodeRelations

    record, pack, relations, request, groups, labels = case
    root = Path(relations._repo)
    (root / "vendor").mkdir()
    (root / "vendor/app.py").write_text((root / "app.py").read_text())
    run(["git", "-C", str(root), "add", "vendor/app.py"], check=True)
    record["units"].append({**record["units"][0], "id": "excluded", "path": "vendor/app.py"})
    relations = CodeRelations(str(root), ({}, {}), withheld=frozenset({"vendor/app.py"}))
    for item in groups[0]["state"]["items"]:
        item["file"] = "[MASKED].py"
    result = measure_case(record, pack, relations, request, groups, labels)
    assert result["judged_reach"]["references"] == 3
    assert result["judged_reach"]["unbound"] == []


def test_native_listed_holder_body_binds_beyond_frozen_nested_anchor(case):
    from pathlib import Path

    record, pack, relations, request, groups, _labels = case
    body = (
        "def outer():\n    class Runtime:\n        def run(self):\n            return 1\n    return Runtime()"
    )
    (Path(relations._repo) / "app.py").write_text(body + "\n")
    record["units"] = [
        {"id": "nested", "path": "app.py", "ranges": [[3, 4]], "symbol": "Runtime.run", "pieces": []}
    ]
    group = groups[0]
    group["state"]["items"] = [{"file": "app.py", "code": body}]
    group["questions"] = {key: value for key, value in group["questions"].items() if key.endswith("#0")}
    group["response"]["answers"] = {
        key: value for key, value in group["response"]["answers"].items() if key.endswith("#0")
    }
    labels = [{"file": "app.py", "first_line": n, "last_line": n} for n in (1, 5)]
    result = measure_case(record, pack, relations, request, groups, labels)
    assert result["judged_reach"]["unbound"] == []
    assert result["judged_reach"]["references"] == 2
    source = result["judged_reach"]["bindings"][0]["source"]
    assert source["place"] == "app.py:1-5"
    assert tuple(map(tuple, source["extent"])) == tuple(map(tuple, source["runs"])) == ((1, 5),)
    assert source["name"] == "outer"
    assert source["source_anchor"] == {"file": "app.py", "start": 3, "end": 4}
    assert all(label["delivered"] for label in result["rooms"]["36000"]["lab"]["labels"])
