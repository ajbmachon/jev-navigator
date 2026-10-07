import importlib.util
import json
from pathlib import Path

import pytest
from shop_search import shop_index

from jev_navigator.llm_step import ReplyParseError
from jev_navigator.search_plan import (
    Approach,
    FileUnits,
    SearchPlan,
    TermProvenance,
    decode_plan,
    execute_plan,
)


def recipe(name):
    path = Path(__file__).parents[1] / "examples/pack_case2" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_execution_receipts_round_trip_real_typed_provenance_and_unit_identity(tmp_path):
    index = shop_index(tmp_path / "repo", {"a.py": "def alpha():\n    return 7\n"})
    approach = Approach(1, FileUnits("a.py"), (TermProvenance("a.py", "supplied outline"),))
    result = execute_plan(index, SearchPlan((approach,)), box_chars=70000)
    path = tmp_path / "execution.json"
    recipe("receipts").write_json(path, result)
    saved = json.loads(path.read_text())
    assert saved["candidates"][0]["unit"]["id"] == "a.py:1-2"
    assert saved["outcomes"][0]["approach"]["call"] == {"operation": "file_units", "path": "a.py"}
    assert saved["outcomes"][0]["approach"]["provenance"] == [
        {"term": "a.py", "source": "supplied outline", "transformation": "copied"}
    ]


def test_planner_dry_run_has_explicit_input_fields_and_checks_actual_output_schema():
    planner = recipe("planner")
    context = planner.PlannerInput("Check the limit", (), "limits.py\n", (), {"roles": "unknown"})
    contract = planner.PlannerContract()
    prompt = contract.render(context)
    assert "Check the limit" in prompt and "limits.py" in prompt
    valid = '{"approaches":[{"rank":1,"call":{"operation":"file_units","path":"limits.py"},"provenance":[]}]}'
    assert contract.parse(valid, context).approaches[0].call.path == "limits.py"
    with pytest.raises(ReplyParseError):
        contract.parse('{"approaches":[{"rank":1,"call":{"operation":"shell"},"provenance":[]}]}', context)


def test_shell_trace_separates_newlines_and_redirections_before_real_plan_execution(tmp_path):
    trace = recipe("trace_translate")
    command = "rg -n 'alpha|beta' src 2>/dev/null\nnl -ba src/a.py | sed -n '1,2p'\nrg -n omega other"
    proposal = {
        "operation": "find_text",
        "arguments": {"pattern": "alpha|beta", "scopes": ["src", "2>/dev/null", "nl", "other"]},
    }
    corrected, note = trace.repair(proposal, command)
    assert note == "shell extraction corrected"
    plan = decode_plan(
        json.dumps(
            {
                "approaches": [
                    {
                        "rank": 1,
                        "call": {**corrected["arguments"], "operation": "find_text", "pattern_kind": "regex"},
                        "provenance": [],
                    }
                ]
            }
        )
    )
    index = shop_index(
        tmp_path, {"src/a.py": "def alpha():\n    return 7\n", "other/b.py": "def omega():\n    return 9\n"}
    )
    broken = decode_plan(
        json.dumps(
            {
                "approaches": [
                    {
                        "rank": 1,
                        "call": {**proposal["arguments"], "operation": "find_text", "pattern_kind": "regex"},
                        "provenance": [],
                    }
                ]
            }
        )
    )
    assert not execute_plan(index, broken, box_chars=70000).candidates
    result = execute_plan(index, plan, box_chars=70000)
    assert [candidate.unit.id for candidate in result.candidates] == ["src/a.py:1-2"]
    assert not result.outcomes[0].problems
    unknown = {"operation": "find_text", "arguments": {"pattern": "alpha", "scopes": ["missing"]}}
    assert trace.repair(unknown, "rg alpha missing 2>/dev/null")[0] == unknown


def test_shell_trace_preserves_quotes_and_does_not_invent_ambiguous_arguments():
    trace = recipe("trace_translate")
    assert trace.commands("rg 'alpha|beta' src && rg \"a>b\" other 2>&1") == [
        ["rg", "alpha|beta", "src"],
        ["rg", "a>b", "other"],
    ]
    proposal = {"operation": "find_text", "arguments": {"pattern": "alpha", "scopes": ["original"]}}
    assert trace.repair(proposal, "rg alpha src\nrg alpha other")[0] == proposal
    with pytest.raises(ValueError):
        trace.commands("rg alpha >")


def test_repeated_rounds_keep_all_approaches_on_one_canonical_unit(tmp_path):
    index = shop_index(tmp_path, {"a.py": "def alpha():\n    return 7\n"})
    first, second = Approach(1, FileUnits("a.py"), ()), Approach(2, FileUnits("a.py"), ())
    candidates = {}
    receipts = recipe("receipts")
    receipts.merge_candidates(candidates, execute_plan(index, SearchPlan((first,)), box_chars=70000))
    receipts.merge_candidates(candidates, execute_plan(index, SearchPlan((second,)), box_chars=70000))
    assert len(candidates) == 1
    assert candidates["a.py:1-2"].approaches == (first, second)


def test_shortlist_stream_uses_real_judge_with_16_item_groups_and_final_tail(tmp_path):
    from jev_navigator.index.units import items_to_judge, read_ranges
    from jev_navigator.judgments.judge import Judge
    from jev_navigator.judgments.questions import Check
    from jev_navigator.testing import ScriptedJevClient

    index = shop_index(tmp_path, {"a.py": "\n".join(f"def f{i}():\n    return {i}\n" for i in range(35))})
    candidates = execute_plan(
        index, SearchPlan((Approach(1, FileUnits("a.py"), ()),)), box_chars=70000
    ).candidates
    client = ScriptedJevClient()

    def entries():
        position = 0
        for candidate in candidates:
            for item in items_to_judge(candidate.unit):
                if position == 16:
                    assert len(client.requests) == 1
                yield ({"file": item.file, "code": read_ranges(index, item.file, item.ranges)}, item)
                position += 1

    answers = list(
        recipe("request_batches").judge_in_order(
            Judge(client, items_per_request=16), [Check("read", "Read `{item}.code`.")], entries(), {}, []
        )
    )
    assert len(answers) == 35
    assert [len(state["items"]) for state, _ in client.requests] == [16, 16, 3]
    assert [item["code"].splitlines()[0] for state, _ in client.requests for item in state["items"]] == [
        f"def f{i}():" for i in range(35)
    ]
