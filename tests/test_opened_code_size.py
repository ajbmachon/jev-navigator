"""How much of an opened place a Find request shows: all of it when its requests fit the box of the
client's input limits, and otherwise the longest start that fits, with a visible cut note."""

from __future__ import annotations

import sys
from pathlib import Path

from conftest import BudgetedClient
from git_repos import commit_files
from search_deadline import searched_in_child

from jev_navigator.adapters.routes import DREX_INPUT_LIMITS
from jev_navigator.directives.find_code import Outcome, SearchBudget, find_code
from jev_navigator.directives.places import MOVES, place_for_line
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import serialized_chars
from jev_navigator.testing import ScriptedJevClient

NEIGHBOUR_CAP_SEARCH = Path(__file__).with_name("neighbour_cap_search.py")
ONE_OPENING = SearchBudget(max_steps=1, beam_width=1)
CALLEES = {"callees": MOVES["callees"]}
LINE_IN_A_REQUEST = 40
"""The characters one line of ``_function_of``'s body takes in a request, its newline escaped."""
AUDIT = "def audit(order):\n" + "".join(f"    log('{letter * 220}')\n" for letter in "abcdefgh")


def _function_of(name: str, characters: int) -> str:
    """A function of more than ``characters`` characters that calls ``audit`` first, then lines of
    equal width, then defines and calls its own tail."""
    lines = [f"    total_{line:05d} = order.amount * {line:05d}\n" for line in range(characters)]
    body = "".join(lines[: characters // len(lines[0]) + 1])
    tail = f"    def {name}_tail(order):\n        return order\n\n    return {name}_tail(order)\n"
    return f"def {name}(order):\n    audit(order)\n{body}{tail}"


def _index(tmp_path: Path, files: dict[str, str]) -> CodeIndex:
    commit_files(tmp_path / "repository", {"app/audit.py": AUDIT, **files})
    return CodeIndex(tmp_path / "repository", ["app/audit.py", *files], fact_cache_dir=tmp_path / "facts")


def _box_measure(state: dict, questions: dict) -> int:
    """What a box bounds: the state plus the longest question, in serialized characters."""
    return serialized_chars(state) + max(serialized_chars(question) for question in questions.values())


def _opened(index: CodeIndex, client, file: str, line: int = 1):
    start = place_for_line(index, file, line, "start")
    result = find_code(index, Judge(client), "the order total", [start], moves=CALLEES, budget=ONE_OPENING)
    return start, result


def test_a_function_of_20000_characters_goes_to_jev_whole(tmp_path: Path) -> None:
    # Arrange
    source = "from app.audit import audit\n\n\n" + _function_of("place", 20_000)
    index = _index(tmp_path, {"app/orders.py": source})

    # Act
    start, result = _opened(index, ScriptedJevClient(default_noul=0.1), "app/orders.py", line=4)

    # Assert
    (opened,) = result.starts
    assert len(start.open().text) > 20_000
    assert opened.code.text == start.open().text
    assert opened.code.span == start.open().span


def test_a_function_over_the_clients_box_shows_the_longest_start_that_fits_and_says_so(
    tmp_path: Path,
) -> None:
    # Arrange: a client with Drex's box of 19,660 characters, under Jev's 76,800. The function's
    # request about `audit` alone is the largest one, so it decides the cut.
    index = _index(tmp_path, {"app/orders.py": _function_of("place", 30_000)})
    client = BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=DREX_INPUT_LIMITS.box_chars)
    client.input_limits = DREX_INPUT_LIMITS

    # Act
    start, result = _opened(index, client, "app/orders.py")

    # Assert
    (opened,) = result.starts
    shown_lines = opened.code.span.end - opened.code.span.start + 1
    all_lines = len(start.open().text.split("\n"))
    tightest = max(_box_measure(state, questions) for state, questions in client.requests)
    assert client.refusals == 0
    assert client.requests[0][0]["slice"]["lines"] == f"{opened.code.span.start}-{opened.code.span.end}"
    assert 0 <= DREX_INPUT_LIMITS.box_chars - tightest < LINE_IN_A_REQUEST
    assert 12_000 < len(opened.code.text) < DREX_INPUT_LIMITS.box_chars
    assert opened.code.text.endswith(
        f"[cut after {shown_lines} of {all_lines} lines to fit the request size limit]"
    )
    assert start.open().text.startswith(opened.code.text.rpartition("\n")[0])


def test_a_function_shown_whole_leaves_room_for_each_neighbour_alone(tmp_path: Path) -> None:
    # Arrange: functions from 16,000 to 19,600 characters under Drex's box. Each calls `audit`, whose
    # preview of 1,900 characters travels with the function when the opening is split, and its own
    # tail, a neighbour only when the request leaves its lines out.
    sizes = range(16_000, 19_601, 200)
    index = _index(tmp_path, {f"app/place_{size}.py": _function_of(f"place_{size}", size) for size in sizes})

    for size in sizes:
        client = BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=DREX_INPUT_LIMITS.box_chars)
        client.input_limits = DREX_INPUT_LIMITS

        # Act
        _, result = _opened(index, client, f"app/place_{size}.py")

        # Assert
        assert (size, result.outcome, client.refusals) == (size, Outcome.BUDGET, 0)
        (opened,) = result.starts
        offered = [entry.place_key for entry in result.not_inspected]
        cut = "[cut after" in opened.code.text
        assert "app/audit.py:1-9" in offered
        assert (size, len(offered)) == (size, 2 if cut else 1)


def test_a_cut_under_a_per_kind_cap_settles_and_every_request_fits(tmp_path: Path) -> None:
    # Act: `jvn find --neighbours-per-kind 1` under Drex's box, in its own process
    # (neighbour_cap_search.py): cutting the function brings its tail on as a callee in place of `audit`
    seen = searched_in_child([sys.executable, str(NEIGHBOUR_CAP_SEARCH), str(tmp_path)])

    # Assert: the search ended, nothing was refused, and its one request shows the cut beside the tail
    # alone, while the cap set `audit` aside
    search = seen["search"]
    (start,) = search["starts"]
    first, last = start["source"]["lines"]
    (asked,) = seen["asked"]
    set_aside = {entry["signature"]: entry["reason"] for entry in search["not_inspected"]}
    assert seen["refusals"] == 0
    assert (first, last) == (1, last) and last < seen["function_lines"]
    assert asked["lines"] == f"1-{last}"
    assert (
        asked["last_line"]
        == f"[cut after {last} of {seen['function_lines']} lines to fit the request size limit]"
    )
    (neighbour,) = asked["neighbours"]
    assert neighbour.startswith(f"app/orders.py:{seen['tail_line']} `def place_tail(order):`")
    assert set_aside["app/audit.py:1 audit"] == "capped"
