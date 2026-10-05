"""A client's input limits: each route declares its own, the routed client packs to the tightest,
and a size refusal is remembered under the limits of the route that refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import BudgetedClient
from git_repos import commit_files
from system_one_stand_in import stand_in

from jev_navigator.adapters.routes import (
    DREX_CONCURRENCY,
    DREX_INPUT_LIMITS,
    JEV_CONCURRENCY,
    Route,
    RoutedJevClient,
    SystemOneClient,
    routes_from_env,
    system_one_client,
)
from jev_navigator.directives.find_code import SearchBudget, find_code
from jev_navigator.directives.places import place_for_line
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import JEV_INPUT_LIMITS, InputBudgetExceededError, InputLimits
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.thresholds import Thresholds

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
SHARED = {"doc": {"sentence": "s"}}
KEY = {
    "TYPESAFE_API_KEY": "local-test-key",
    "SYSTEM_ONE_DREX_API_KEY": "local-drex-key",
    "SYSTEM_ONE_DECIDER_API_KEY": "local-decider-key",
}
ONE_ROUND = SearchBudget(max_calls=4, beam_width=1, max_depth=1)


def _items(count: int, code_chars: int) -> list[dict]:
    return [{"file": f"f{index}.py", "code": f"v{index} = '{'x' * code_chars}'"} for index in range(count)]


def test_a_judge_packs_batches_to_the_box_its_client_declares() -> None:
    # Arrange: a Drex-sized client; two items fit its box together, as all four fit Jev's
    client = BudgetedClient(JEV_INPUT_LIMITS.box_chars, input_box=DREX_INPUT_LIMITS.box_chars)
    client.input_limits = DREX_INPUT_LIMITS

    # Act
    Judge(client).check_each(DESCRIBES, _items(4, code_chars=DREX_INPUT_LIMITS.box_chars * 2 // 5), SHARED)

    # Assert
    assert client.refusals == 0
    assert sorted(len(state["items"]) for state, _ in client.requests) == [2, 2]


def test_the_routed_client_declares_the_tightest_limits_of_its_routes() -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")
    environment = {**KEY, "SYSTEM_ONE_ROUTES": "drex,jev", "SYSTEM_ONE_DREX": "1", "SYSTEM_ONE_JEV": "1"}

    # Act
    routed = system_one_client(environment)

    # Assert
    try:
        assert [route.client.input_limits for route in routed.routes] == [DREX_INPUT_LIMITS, JEV_INPUT_LIMITS]
        assert routed.input_limits == InputLimits(DREX_INPUT_LIMITS.box_chars, JEV_INPUT_LIMITS.request_chars)
    finally:
        routed.close()


def test_a_custom_route_without_its_input_tokens_is_refused_naming_the_setting() -> None:
    # Arrange
    environment = {
        **KEY,
        "SYSTEM_ONE_ROUTES": "decider",
        "SYSTEM_ONE_DECIDER_ENDPOINT": "http://127.0.0.1:9",
        "SYSTEM_ONE_DECIDER_MODEL": "decider-4b",
    }

    # Act and Assert
    with pytest.raises(ValueError, match="SYSTEM_ONE_DECIDER_INPUT_TOKENS"):
        routes_from_env(environment)


def test_a_custom_route_takes_its_box_from_its_input_tokens() -> None:
    # Arrange
    pytest.importorskip("typesafe_sdk")
    environment = {
        **KEY,
        "SYSTEM_ONE_ROUTES": "decider",
        "SYSTEM_ONE_DECIDER_ENDPOINT": "http://127.0.0.1:9",
        "SYSTEM_ONE_DECIDER_MODEL": "decider-4b",
        "SYSTEM_ONE_DECIDER_INPUT_TOKENS": "4096",
        "SYSTEM_ONE_DECIDER_CONCURRENCY": "4",
    }

    # Act
    [route] = routes_from_env(environment)

    # Assert
    try:
        assert route.client.input_limits == InputLimits.from_tokens(4_096)
    finally:
        route.client.close()


def test_a_size_refusal_is_recorded_under_the_limits_of_the_route_that_refused(tmp_path: Path) -> None:
    # Arrange: the primary is down, so the fallback answers, and it refuses the request for its size
    pytest.importorskip("typesafe_sdk")
    with stand_in("jev-test", refuses_size=True) as jev:
        down = SystemOneClient(
            model="drex-test",
            api_key="local-test-key",
            base_url="http://127.0.0.1:9",
            input_limits=DREX_INPUT_LIMITS,
            max_concurrency=DREX_CONCURRENCY,
        )
        fallback = SystemOneClient(
            model="jev-test",
            api_key="local-test-key",
            base_url=jev.url,
            input_limits=JEV_INPUT_LIMITS,
            max_concurrency=JEV_CONCURRENCY,
        )
        routed = RoutedJevClient((Route("drex", down), Route("jev", fallback)))
        judge = Judge(routed, store=JsonlAnswerStore(tmp_path / "answers.jsonl"))

        # Act
        try:
            with pytest.raises(InputBudgetExceededError):
                judge.ask(
                    {"code": "x"}, {"q": {"type": "noul", "instructions": "y?"}}, thresholds=Thresholds()
                )
        finally:
            routed.close()

    # Assert
    [refusal] = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert (refusal["route"], refusal["input_box"]) == ("jev-test", JEV_INPUT_LIMITS.box_chars)


def test_a_find_opening_is_split_before_sending_when_it_exceeds_the_clients_box(tmp_path: Path) -> None:
    # Arrange: an opening that fits Jev's box but not Drex's, asked of a Drex-sized client
    helpers = [f"def helper_{index}():\n    return '{'h' * 500}'\n" for index in range(30)]
    calls = "".join(f"    helper_{index}()\n" for index in range(30))
    body = "".join(f"    x{index} = '{'x' * 400}'\n" for index in range(30))
    index = _committed_index(
        tmp_path, {"app.py": "def entry():\n" + body + calls + "\n" + "\n".join(helpers)}
    )
    client = BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=DREX_INPUT_LIMITS.box_chars)
    client.input_limits = DREX_INPUT_LIMITS

    # Act
    find_code(
        index, Judge(client), "the target", [place_for_line(index, "app.py", 1, "start")], budget=ONE_ROUND
    )

    # Assert
    assert client.requests
    assert client.refusals == 0


def _committed_index(root: Path, files: dict[str, str]) -> CodeIndex:
    commit_files(root, files)
    return CodeIndex(root, list(files))


def test_a_custom_route_without_its_concurrency_is_refused_naming_the_setting() -> None:
    # Arrange
    environment = {
        **KEY,
        "SYSTEM_ONE_ROUTES": "decider",
        "SYSTEM_ONE_DECIDER_ENDPOINT": "http://127.0.0.1:9",
        "SYSTEM_ONE_DECIDER_MODEL": "decider-4b",
        "SYSTEM_ONE_DECIDER_INPUT_TOKENS": "4096",
    }

    # Act and Assert
    with pytest.raises(ValueError, match="SYSTEM_ONE_DECIDER_CONCURRENCY"):
        routes_from_env(environment)
