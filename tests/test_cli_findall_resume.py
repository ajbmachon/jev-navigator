"""Saved enumeration resumes through the real index, Judge store and CLI pack boundary."""

import json
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.cli import create_evidence_pack
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.testing import ScriptedJevClient

HELPERS = "".join(f"def helper_{number}(value):\n    return value + {number}\n\n" for number in range(20))


def enumerated(client: ScriptedJevClient) -> list[str]:
    """Each item a Find All request asked, by its content."""
    return [
        json.dumps(item, sort_keys=True)
        for state, _ in client.requests
        if "targets" in state
        for item in state["items"]
    ]


def test_findall_reopens_remaining_units_without_repeating_completed_judgments(tmp_path: Path):
    # Arrange: 23 units, so Find All needs two requests and a cap of three calls stops after the first
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "entry.py": "from policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "policy.py": "def admit(item):\n    return len(item) <= 3\n",
            "other.py": "def fits(item):\n    return len(item) <= 3\n",
            "helpers.py": HELPERS,
        },
    )

    def response(_id, _question, state):
        if "items" in state:
            code = state["items"][int(_id.rsplit("#", 1)[1])]["code"]
        else:
            code = state["slice"]["code"]
        return 0.96 if "len(item)" in code else 0.04

    first_client = ScriptedJevClient(nouls=response)
    first = tmp_path / "first"
    initial = create_evidence_pack(
        repository,
        (),
        "item limit",
        ("entry.py:3",),
        first,
        SearchBudget(max_calls=3, beam_width=1),
        first_client,
        workflow="findall",
    )
    assert initial["search"]["outcome"] == "budget"
    assert initial["search"]["not_judged"]
    second_client = ScriptedJevClient(nouls=response)
    resumed = create_evidence_pack(
        repository,
        (),
        "item limit",
        ("entry.py:3",),
        tmp_path / "second",
        SearchBudget(max_calls=1, beam_width=1),
        second_client,
        workflow="findall",
        resume_from=first,
    )
    assert resumed["search"]["outcome"] == "scope_examined"
    assert resumed["search"]["coverage"] == "units_examined"
    assert {x["name"] for x in resumed["search"]["found"]} == {"admit", "fits"}
    assert resumed["search"]["calls"] == 4
    assert resumed["search"]["calls_this_invocation"] == 1
    assert resumed["seed_search"] == initial["seed_search"]
    assert len(second_client.requests) == 1
    assert not set(enumerated(first_client)) & set(enumerated(second_client))
    assert len(enumerated(first_client)) + len(enumerated(second_client)) == 23

    checkpoint = json.loads((first / "resume.json").read_text())
    checkpoint["check_id"] = "previous-containment-question"
    (first / "resume.json").write_text(json.dumps(checkpoint))
    stale_client = ScriptedJevClient(nouls=response)
    with pytest.raises(ValueError, match="question changed"):
        create_evidence_pack(
            repository,
            (),
            "item limit",
            ("entry.py:3",),
            tmp_path / "stale",
            SearchBudget(max_calls=1, beam_width=1),
            stale_client,
            workflow="findall",
            resume_from=first,
        )
    assert stale_client.requests == []


@pytest.mark.parametrize("initial_calls", [0, 1, 2])
def test_findall_budget_stop_during_seed_search_continues_the_same_work(tmp_path, initial_calls):
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "entry.py": "from policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "policy.py": "def admit(item):\n    return len(item) <= 3\n",
            "other.py": "def fits(item):\n    return len(item) <= 3\n",
        },
    )
    provider = ScriptedJevClient(default_noul=0.96)
    previous = None
    manifest = None
    for turn, allowance in enumerate([initial_calls, 1, 1, 1, 1]):
        output = tmp_path / f"pack-{turn}"
        manifest = create_evidence_pack(
            repository,
            (),
            "item limit",
            ("entry.py:3",),
            output,
            SearchBudget(max_calls=allowance, beam_width=1),
            provider,
            workflow="findall",
            resume_from=previous,
        )
        if manifest["search"]["outcome"] == "scope_examined":
            break
        assert manifest["search"]["coverage"] == "partial"
        previous = output
    assert manifest["search"]["coverage"] == "units_examined"
    assert {x["name"] for x in manifest["search"]["found"]} == {"handle", "admit", "fits"}
    assert manifest["search"]["calls"] == len(provider.requests) == 3
