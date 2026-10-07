from pathlib import Path

import msgspec
import pytest
from shop_search import shop_index

from jev_navigator.directives.find_all import find_all, find_all_text
from jev_navigator.index.units import read_ranges
from jev_navigator.judgments.judge import Judge
from jev_navigator.search_plan import (
    Approach,
    Callees,
    Callers,
    ConfigKey,
    Definitions,
    FileUnits,
    FindText,
    MatchingFiles,
    NamedImports,
    PlanSource,
    References,
    SearchPlan,
    TermProvenance,
    decode_plan,
    execute_plan,
)
from jev_navigator.search_plan import (
    TestsOf as LookupTests,
)
from jev_navigator.testing import ScriptedJevClient


def approach(rank, call):
    return Approach(rank, call, (TermProvenance("check", "input"),))


def test_plan_merges_real_code_and_text_and_feeds_existing_search(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "entry.py": "from limit import check\n\ndef run():\n    return check()\n",
            "limit.py": "def check():\n    return 'LIMIT'\n",
            "config.toml": "LIMIT = 7\n",
            "docs/policy.md": "# Limit\nThe LIMIT is configured in config.toml.\n",
        },
    )
    plan = SearchPlan(
        (
            approach(4, FileUnits("limit.py")),
            approach(2, FindText("LIMIT")),
            approach(1, MatchingFiles(("*.toml",))),
            approach(3, Definitions("check", "limit.py")),
        )
    )
    result = execute_plan(index, plan, box_chars=70000)
    assert [c.unit.path for c in result.candidates] == ["config.toml", "docs/policy.md", "limit.py"]
    check = result.candidates[-1]
    assert [a.rank for a in check.approaches] == [2, 3, 4]
    assert "return 'LIMIT'" in read_ranges(index, check.unit.path, check.unit.ranges)
    client = ScriptedJevClient()
    found = find_all(
        index, Judge(client), {"p": "the configured limit"}, sources=(PlanSource(result),), hops=()
    )
    text_found = find_all_text(
        index, Judge(client), {"p": "the configured limit"}, sources=(PlanSource(result),), hops=()
    )
    assert {u.id for r in (found, text_found) for u in r.units} == {c.unit.id for c in result.candidates}
    assert len(client.requests) == 2
    assert [len(r[0]["items"]) for r in client.requests] == [1, 2]


def test_owner_bound_relations_imports_and_tests_do_not_follow_same_named_other_owner(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "entry.py": "from limit import check\n\ndef run():\n    return check()\n",
            "limit.py": "def check():\n    return 'LIMIT'\n",
            "other.py": "def check():\n    return 'OTHER'\n\ndef unrelated():\n    return check()\n",
            "tests/test_limit.py": (
                "from limit import check\n\ndef test_limit():\n    assert check() == 'LIMIT'\n"
            ),
            "config.toml": "LIMIT = 7\n",
        },
    )
    calls = (
        Callers("check", "limit.py"),
        Callees("entry.py", 3),
        NamedImports("entry.py"),
        ConfigKey("LIMIT", ("config.toml",)),
        References("LIMIT", kind="literal"),
        LookupTests("limit.py", "check"),
    )
    result = execute_plan(
        index, SearchPlan(tuple(approach(i, c) for i, c in enumerate(calls, 1))), box_chars=70000
    )
    by_rank = {o.approach.rank: o for o in result.outcomes}
    assert {r.at.file for r in by_rank[1].reaches} == {"entry.py", "tests/test_limit.py"}
    assert {r.at.file for r in by_rank[2].reaches} == {"limit.py"}
    assert {r.at for r in by_rank[3].reaches} == {"limit.py"}
    assert {c.unit.path for c in result.candidates} == {
        "entry.py",
        "tests/test_limit.py",
        "limit.py",
        "config.toml",
    }


def test_invalid_arguments_are_reported_with_no_replacement_and_valid_approaches_survive(tmp_path: Path):
    index = shop_index(tmp_path, {"a.py": "def alpha():\n    return 7\n"})
    calls = (
        FileUnits("missing/a.py"),
        Definitions("typo", "a.py"),
        FindText("(", pattern_kind="regex"),
        FindText("alpha", ("missing",)),
        FileUnits("a.py"),
    )
    result = execute_plan(
        index, SearchPlan(tuple(approach(i, c) for i, c in enumerate(calls, 1))), box_chars=70000
    )
    assert len(result.candidates) == 1
    assert [a.rank for a in result.candidates[0].approaches] == [5]
    assert all(o.problems and not o.reaches for o in result.outcomes[:4])
    assert not result.outcomes[4].problems


def test_regex_and_scope_globs_use_real_lines_and_report_excluded_text(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "app/a.py": "def alpha():\n    return 'TOKEN'\n",
            "other/a.py": "def beta():\n    return 'TOKEN'\n",
            "notes.md": "# Note\nTOKEN and token\n",
            ".env": "TOKEN=private\n",
        },
    )
    plan = SearchPlan(
        (approach(1, FindText("^def alpha|token", ("app", "notes.md", ".env"), "regex", True)),)
    )
    result = execute_plan(index, plan, box_chars=70000)
    assert {c.unit.path for c in result.candidates} == {"app/a.py", "notes.md"}
    assert any("env file" in p for p in result.outcomes[0].problems)


@pytest.mark.parametrize(
    "payload",
    [
        '{"approaches":[{"rank":1,"call":{"operation":"shell","command":"cat x"},"provenance":[]}]}',
        '{"approaches":[{"rank":11,"call":{"operation":"file_units","path":"a"},"provenance":[]}]}',
        '{"approaches":[{"rank":1,"call":{"operation":"file_units","path":"a","invented":1},"provenance":[]}]}',
    ],
)
def test_typed_plan_rejects_noncontract_operations_and_arguments(payload):
    with pytest.raises(msgspec.ValidationError):
        decode_plan(payload)


def test_direct_constructor_obeys_same_ten_approach_contract_and_unique_ranks():
    with pytest.raises(msgspec.ValidationError):
        execute_plan(None, SearchPlan(tuple(approach(i, FileUnits("a")) for i in range(1, 12))), box_chars=7)
    with pytest.raises(ValueError, match="unique"):
        decode_plan(
            msgspec.json.encode(SearchPlan((approach(1, FileUnits("a")), approach(1, FileUnits("b")))))
        )
