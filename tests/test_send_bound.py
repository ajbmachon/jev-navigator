"""One judge, with every scope and every caller that shares it, keeps at most ``max_concurrency``
requests in flight, and a search whose openings split into nested batches still finishes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jev_navigator.judgments.judge import DEFAULT_MAX_CONCURRENCY

SEARCH = Path(__file__).with_name("send_bound_search.py")
SEARCH_DEADLINE_SECONDS = 60


def _searched(tmp_path: Path, mode: str, width: int, *, refuse_lists: bool = False) -> dict:
    """What the provider saw in one search of ``width`` places, run in its own process, so a judge
    that deadlocks fails here at the deadline and the process is killed instead of hanging pytest."""
    lists = "refuse-lists" if refuse_lists else "answer-lists"
    command = [sys.executable, str(SEARCH), mode, str(width), lists, str(tmp_path)]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=SEARCH_DEADLINE_SECONDS, check=False
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the search did not finish within {SEARCH_DEADLINE_SECONDS} seconds")
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_beam_of_forty_never_has_more_than_the_judges_bound_in_flight(tmp_path: Path, mode: str) -> None:
    # Act
    seen = _searched(tmp_path, mode, 40)

    # Assert
    assert seen["steps"] == 40
    assert seen["requests"] == 40
    assert seen["peak"] == DEFAULT_MAX_CONCURRENCY


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_openings_split_into_nested_batches_stay_within_the_bound_and_finish(
    tmp_path: Path, mode: str
) -> None:
    # Act
    seen = _searched(tmp_path, mode, 20, refuse_lists=True)

    # Assert
    assert seen["steps"] == 20
    assert seen["neighbour_batches"] >= 2 * 20
    assert seen["peak"] <= DEFAULT_MAX_CONCURRENCY
