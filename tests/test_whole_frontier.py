from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from shop_search import shop_index

from jev_navigator.composition import FrontierConfiguration
from jev_navigator.index.units import RangeAnchor
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.secrets import SecretMasker
from jev_navigator.sources import (
    ANCHORS,
    CALLEES,
    FILE_WORDS,
    FILES,
    IMPORTS,
    LITERALS,
    NAMED_FILES,
    TEXT_NAMED_FILES,
)
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient


def test_budget_keeps_the_entire_reached_population_and_preserves_file_first_batches(tmp_path: Path) -> None:
    matched = "\n".join(f"def check_{i}():\n    return {i}\n" for i in range(35))
    index = shop_index(
        tmp_path,
        {
            "z/limits.py": matched,
            "a/unrelated.py": "def other():\n    return 99\n",
            "docs/limits.md": "# Limits\nEvery answer matters.\n",
        },
    )
    client = ScriptedJevClient()
    config = FrontierConfiguration(max_calls=1, sources=(FILE_WORDS, FILES), hops=())
    result = asyncio.run(config.search(index, Judge(client), {"p": "check limits"}, files=["a/unrelated.py"]))
    assert result.stopped_by == "budget"
    assert len(result.units) == 37
    assert len(result.judged["p"]) == 16
    assert len(result.not_judged) == 21
    [state, _] = client.requests[0]
    assert state["items"][0]["file"] == "docs/limits.md"
    assert all(item["file"] == "z/limits.py" for item in state["items"][1:])
    client = ScriptedJevClient()
    result = asyncio.run(
        FrontierConfiguration(max_calls=3, sources=config.sources, hops=()).search(
            index,
            Judge(client),
            {"p": "check limits"},
            files=["a/unrelated.py"],
        )
    )
    assert result.stopped_by == "scope_examined"
    assert len(result.judged["p"]) == 37
    # Concurrent sends can arrive in any order. Their batch membership remains file first.
    assert sorted(len(state["items"]) for state, _ in client.requests) == [5, 16, 16]
    [last_batch] = [state["items"] for state, _ in client.requests if len(state["items"]) == 5]
    assert last_batch[-1]["file"] == "a/unrelated.py"


def test_follow_imported_owner_callee_and_literal_even_after_a_no_answer(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "entry.py": "from worker import execute\n\ndef entry():\n    return execute()\n",
            "worker.py": "from helper import decide\n\ndef execute():\n    return decide()\n",
            "helper.py": "def decide():\n    return 'FEATURE_LIMIT'\n",
            "other.py": "def decide():\n    return 'WRONG_OWNER'\n",
            "config.toml": "FEATURE_LIMIT = 7\n",
        },
    )
    result = asyncio.run(
        FrontierConfiguration(
            sources=(FILES,),
            hops=(CALLEES, IMPORTS, LITERALS),
        ).search(index, Judge(ScriptedJevClient(default_noul=0.0)), {"p": "entry"}, files=["entry.py"])
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {score.unit.path for score in result.scores("p")} == {
        "entry.py",
        "worker.py",
        "helper.py",
        "config.toml",
    }
    assert not result.not_judged


def test_follow_named_file_and_its_text_to_a_second_named_file(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "app.py": "def run():\n    return 'docs/policy.md'\n",
            "docs/policy.md": "# Policy\nRead config/limits.toml for the limit.\n",
            "config/limits.toml": "maximum = 7\n",
        },
    )
    config = FrontierConfiguration(max_calls=1, sources=(FILES,), hops=(NAMED_FILES, TEXT_NAMED_FILES))
    first = asyncio.run(config.search(index, Judge(ScriptedJevClient()), {"p": "policy"}, files=["app.py"]))
    assert first.stopped_by == "budget"
    assert {unit.path for unit in first.units} == {"app.py", "docs/policy.md"}
    result = asyncio.run(
        FrontierConfiguration(sources=config.sources, hops=config.hops).search(
            index,
            Judge(ScriptedJevClient()),
            {"p": "policy"},
            files=["app.py"],
            completed=first.judged,
        )
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {unit.path for unit in result.units} == {"app.py", "docs/policy.md", "config/limits.toml"}
    assert len(result.scores("p")) == 3


def test_zero_budget_discovers_files_without_calling_the_client(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path, {"app.py": "def run():\n    return 1\n", "policy.md": "# Policy\nKeep it.\n"}
    )
    client = ScriptedJevClient()
    result = asyncio.run(
        FrontierConfiguration(max_calls=0, sources=(FILE_WORDS, FILES, ANCHORS)).search(
            index,
            Judge(client),
            {"p": "policy"},
            files=["app.py"],
        )
    )
    assert len(result.units) == 2
    assert len(result.not_judged) == 2
    assert result.calls == 0
    assert not client.requests


def test_matching_more_file_words_ranks_before_alphabetical_file_listing(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "a/limit.py": "def alpha():\n    return 1\n",
            "z/feature_limit.py": "def beta():\n    return 2\n",
        },
    )
    client = ScriptedJevClient()
    result = asyncio.run(
        FrontierConfiguration(sources=(FILE_WORDS,), hops=()).search(
            index, Judge(client), {"p": "feature limit"}
        )
    )
    assert result.stopped_by == "scope_examined"
    assert [item["file"] for item in client.requests[0][0]["items"]] == ["z/feature_limit.py", "a/limit.py"]


def test_cached_builtin_masking_keeps_path_rules_and_cross_item_secret_copies(tmp_path: Path) -> None:
    secret = "abc123TOKEN987secretVALUE456"
    index = shop_index(
        tmp_path,
        {
            "app.py": f"def run():\n    api_key = '{secret}'\n    return api_key\n",
            "config.yaml": f"message: {secret}\ntoken: config-only-value\n",
        },
    )
    cached, uncached = ScriptedJevClient(), ScriptedJevClient()
    config = FrontierConfiguration(sources=(FILES,), hops=())
    for judge in (Judge(cached), Judge(uncached, masker=SecretMasker())):
        result = asyncio.run(config.search(index, judge, {"p": "configuration"}, files=index.files))
        assert result.stopped_by == "scope_examined", result.failure
    assert cached.requests == uncached.requests
    code = str(cached.requests)
    assert secret not in code
    assert "config-only-value" not in code
    assert "[MASKED]" in code


def test_default_frontier_follows_fresh_identifier_references_and_incoming_calls(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "entry.py": "def entry():\n    callback = support_task\n    return callback\n",
            "support.py": "def support_task():\n    return 'FEATURE_LIMIT'\n",
            "consumer.py": "from entry import entry\n\ndef consume():\n    return entry()\n",
            "config.toml": "FEATURE_LIMIT = 7\n",
            "unrelated.py": "def other():\n    return 99\n",
        },
    )
    result = asyncio.run(
        FrontierConfiguration().search(
            index, Judge(ScriptedJevClient(default_noul=0.0)), {"p": "entry"}, files=["entry.py"]
        )
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {unit.path for unit in result.units} == {"entry.py", "support.py", "consumer.py", "config.toml"}
    assert not result.not_judged


def test_default_lists_target_word_files_before_paths_read_from_supplied_code(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "entry.py": "def entry():\n    return 'a/other.py'\n",
            "a/other.py": "def other():\n    return 1\n",
            "z/limit.py": "def maximum():\n    return 2\n",
        },
    )
    client = ScriptedJevClient()
    result = asyncio.run(
        FrontierConfiguration().search(index, Judge(client), {"p": "limit"}, files=["entry.py"])
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert client.requests[0][0]["items"][0]["file"] == "z/limit.py"


def test_literal_hops_search_fresh_values_once_while_following_a_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = shop_index(
        tmp_path,
        {
            "app.py": "def run():\n    return 'FEATURE_KEY'\n",
            "config.py": "FEATURE_KEY = 'OTHER_KEY'\n",
            "other.py": "OTHER_KEY = 'FEATURE_KEY'\n",
        },
    )
    searched = []
    search = index.search_texts

    def recorded(terms):
        terms = tuple(terms)
        searched.extend(terms)
        return search(terms)

    monkeypatch.setattr(index, "search_texts", recorded)
    result = asyncio.run(
        FrontierConfiguration(sources=(FILES,), hops=(LITERALS,)).search(
            index, Judge(ScriptedJevClient()), {"p": "configuration"}, files=["app.py"]
        )
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {unit.path for unit in result.units} == {"app.py", "config.py", "other.py"}
    assert searched == ["FEATURE_KEY", "OTHER_KEY"]


def test_frontier_uses_judge_concurrency_with_sixteen_items_per_request(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path, {"app.py": "\n".join(f"def task_{i}():\n    return {i}\n" for i in range(64))}
    )

    class TrackingClient(AsyncScriptedJevClient):
        active = 0
        peak = 0

        async def send(self, state, questions):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(0)
                return await super().send(state, questions)
            finally:
                self.active -= 1

    client = TrackingClient()
    result = asyncio.run(
        FrontierConfiguration(sources=(FILES,), hops=()).search(
            index, Judge(client, max_concurrency=2), {"p": "tasks"}, files=["app.py"]
        )
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert len(result.judged["p"]) == 64
    assert client.peak == 2
    assert [len(state["items"]) for state, _ in client.requests] == [16, 16, 16, 16]


def test_already_delivered_code_expands_without_paying_to_judge_it_again(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {"app.py": "def run():\n    return 'FEATURE_LIMIT'\n", "config.toml": "FEATURE_LIMIT = 7\n"},
    )
    client = ScriptedJevClient()
    result = asyncio.run(
        FrontierConfiguration(sources=(FILES,), hops=(LITERALS,)).search(
            index,
            Judge(client),
            {"p": "configuration"},
            files=["app.py"],
            delivered=[RangeAnchor("app.py", 1, 2)],
        )
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {unit.path for unit in result.units} == {"app.py", "config.toml"}
    assert [item["file"] for state, _ in client.requests for item in state["items"]] == ["config.toml"]


def test_same_callee_name_on_different_owners_follows_both_definitions(tmp_path: Path) -> None:
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
    result = asyncio.run(
        FrontierConfiguration(sources=(FILES,), hops=(CALLEES,)).search(
            index, Judge(ScriptedJevClient()), {"p": "run"}, files=["app.py"]
        )
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {unit.path for unit in result.units} == {"app.py", "worker.py", "guard.py"}
