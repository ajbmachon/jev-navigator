"""How much of an opened place a Find request shows: all of it when its requests fit the box of the
client's input limits, and otherwise the longest start that fits, with a visible cut note."""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import BudgetedClient
from git_repos import commit_files
from neighbour_cap_search import PLACE
from search_deadline import searched_in_child
from test_find_code import find_with

from jev_navigator.adapters.routes import DREX_INPUT_LIMITS
from jev_navigator.directives.find_code import (
    FOUND,
    Outcome,
    SearchBudget,
    SearchQuestions,
    find_code,
    shown_for_target,
)
from jev_navigator.directives.places import MOVES, place_for_line
from jev_navigator.directives.shown import shown_slice
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import Span
from jev_navigator.judgments.client import JEV_INPUT_LIMITS
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import serialized_chars
from jev_navigator.judgments.secrets import DEFAULT_MASKER, MASK
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


def _calling_nothing(characters: int) -> str:
    """A function of more than ``characters`` characters with no neighbour to list."""
    lines = [f"    total_{line:05d} = order.amount * {line:05d}\n" for line in range(characters)]
    return "def place(order):\n" + "".join(lines[: characters // len(lines[0]) + 1]) + "    return order\n"


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


def test_an_opening_whose_first_cut_already_fits_beside_its_neighbour_settles(tmp_path: Path) -> None:
    # Act: `jvn find` in its own process (neighbour_cap_search.py), so a neighbour loop that never
    # settles fails at the suite's deadline; it runs first, before any in-process search could hang
    seen = searched_in_child([sys.executable, str(NEIGHBOUR_CAP_SEARCH), str(tmp_path), "common"])

    # Assert: one request shows the whole function beside its callee and the lines before it
    (asked,) = seen["asked"]
    assert asked["lines"] == "4-6"
    assert [neighbour.split(" ", 1)[0] for neighbour in asked["neighbours"]] == [
        "app/audit.py:1",
        "app/orders.py:1-3",
    ]
    assert seen["refusals"] == 0


def test_the_start_a_cut_returns_is_one_its_measure_accepted_even_when_the_measure_is_not_monotone(
    tmp_path: Path,
) -> None:
    # Arrange: a measure that accepts 1 to 3 and 6 shown lines of 8, as a masked measure can when a
    # cut leaves a secret's key line out and its copies unmasked
    index = _index(tmp_path, {"app/eight.py": "".join(f"line_{number} = {number}\n" for number in range(8))})
    code = index.read_slice(Span("app/eight.py", 1, 8))
    accepted = {1, 2, 3, 6}
    measured = []

    def fits(shown) -> bool:
        measured.append(shown.span.end)
        return shown.span.end in accepted

    # Act
    shown = shown_slice(code, fits)

    # Assert
    assert shown.span.end in accepted
    assert shown.span.end in measured


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
    seen = searched_in_child([sys.executable, str(NEIGHBOUR_CAP_SEARCH), str(tmp_path), "per-kind-cap"])
    function_lines = PLACE.count("\n")
    tail_line = PLACE.split("\n").index("    def place_tail(order):") + 1

    # Assert: the search ended, nothing was refused, and its one request shows the cut beside the tail
    # alone, while the cap set `audit` aside
    search = seen["search"]
    (start,) = search["starts"]
    first, last = start["source"]["lines"]
    (asked,) = seen["asked"]
    set_aside = {entry["signature"]: entry["reason"] for entry in search["not_inspected"]}
    assert seen["refusals"] == 0
    assert (first, last) == (1, last) and last < function_lines
    assert asked["lines"] == f"1-{last}"
    assert asked["last_line"] == f"[cut after {last} of {function_lines} lines to fit the request size limit]"
    (neighbour,) = asked["neighbours"]
    assert neighbour.startswith(f"app/orders.py:{tail_line} `def place_tail(order):`")
    assert set_aside["app/audit.py:1 audit"] == "capped"


LONGER_FOUND = replace(FOUND, instructions=FOUND.instructions + " Read every line before you answer." * 20)


@pytest.mark.parametrize("found", [FOUND, LONGER_FOUND], ids=["default wording", "longer wording"])
@pytest.mark.parametrize("characters", range(17_600, 20_400, 400))
def test_find_shows_what_shown_for_target_shows(tmp_path: Path, found, characters: int) -> None:
    # Arrange: a function with no neighbour, so the first cut alone decides what Find shows, under
    # Drex's box of 19,660 characters: whole below it, cut above it
    index = _index(tmp_path, {"app/orders.py": _calling_nothing(characters)})
    client = BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=DREX_INPUT_LIMITS.box_chars)
    client.input_limits = DREX_INPUT_LIMITS
    start = place_for_line(index, "app/orders.py", 1, "start")

    # Act
    result = find_code(
        index,
        Judge(client),
        "the order total",
        [start],
        questions=SearchQuestions(found=found),
        moves=CALLEES,
        budget=ONE_OPENING,
    )
    shown = shown_for_target(start.open(), "the order total", DREX_INPUT_LIMITS, found=found)

    # Assert
    (opened,) = result.starts
    assert client.refusals == 0
    assert (opened.code.span, opened.code.text) == (shown.span, shown.text)


class ShortSecretMasker:
    """The rule #99 adds to the masker: the quoted value of every assignment to a name holding
    PASSWORD is masked however short it is, so a value shorter than the 8-character mask makes the
    request longer."""

    _VALUE = re.compile(r'PASSWORD\w* = "([^"]+)"')

    def mask(self, text: str, path: str | None = None) -> str:
        return self._VALUE.sub(lambda match: match[0].replace(match[1], MASK), text)

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return [match[1] for match in self._VALUE.finditer(text)]


def _numbered_secret(line: int) -> str:
    """A 6-character value under a secret-named key, 2 characters longer once masked."""
    return f'    DB_PASSWORD_{line:05d} = "k{line:05d}"\n'


def _hunter2(_line: int) -> str:
    """jvn-verifier's measured case: 7 characters under a suffixed secret key, 1 longer once masked."""
    return '    DB_PASSWORD_PROD = "hunter2"\n'


def _holding_short_secrets(characters: int, calls_audit: bool, secret_line=_numbered_secret) -> str:
    """A function of more than ``characters`` characters whose lines give secret-named keys short
    values."""
    lines = [secret_line(line) for line in range(characters)]
    body = "".join(lines[: characters // len(lines[0]) + 1])
    return "def place(order):\n" + ("    audit(order)\n" if calls_audit else "") + body + "    return order\n"


def _drex_client() -> BudgetedClient:
    client = BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=DREX_INPUT_LIMITS.box_chars)
    client.input_limits = DREX_INPUT_LIMITS
    return client


@pytest.mark.parametrize("characters", range(16_000, 19_600, 600))
def test_the_first_cut_measures_the_request_as_masked(tmp_path: Path, characters: int) -> None:
    # Arrange: a function with no neighbour, whose masked request is longer than its text
    index = _index(tmp_path, {"app/orders.py": _holding_short_secrets(characters, calls_audit=False)})
    client = _drex_client()
    start = place_for_line(index, "app/orders.py", 1, "start")
    judge = Judge(client, masker=ShortSecretMasker())

    # Act
    result = find_code(index, judge, "the order total", [start], moves=CALLEES, budget=ONE_OPENING)
    masked = shown_for_target(start.open(), "the order total", DREX_INPUT_LIMITS, masker=ShortSecretMasker())
    unmasked = shown_for_target(start.open(), "the order total", DREX_INPUT_LIMITS, masker=None)

    # Assert: Find shows the masked measure's cut, never longer than the unmasked one, and the
    # provider refuses nothing
    (opened,) = result.starts
    assert client.refusals == 0
    assert (opened.code.span, opened.code.text) == (masked.span, masked.text)
    assert masked.span.end <= unmasked.span.end


def test_a_function_whose_masked_request_is_over_the_box_is_cut_shorter_than_its_text_allows(
    tmp_path: Path,
) -> None:
    # Arrange: the largest function whose unmasked request fits Drex's box whole
    index = _index(tmp_path, {"app/orders.py": _holding_short_secrets(17_000, calls_audit=False)})
    start = place_for_line(index, "app/orders.py", 1, "start")
    code = start.open()

    # Act
    unmasked = shown_for_target(code, "the order total", DREX_INPUT_LIMITS, masker=None)
    masked = shown_for_target(code, "the order total", DREX_INPUT_LIMITS, masker=ShortSecretMasker())

    # Assert
    assert unmasked.span == code.span
    assert masked.span.end < code.span.end


@pytest.mark.parametrize(
    ("masker", "secret_line"),
    [(ShortSecretMasker(), _numbered_secret), (DEFAULT_MASKER, _hunter2)],
    ids=["#99's rule", "the judge's default masker"],
)
@pytest.mark.parametrize("calls_audit", [False, True], ids=["no neighbour", "a neighbour"])
@pytest.mark.parametrize("characters", range(16_000, 19_600, 600))
def test_every_request_of_an_opening_with_short_secrets_fits_once_masked(
    tmp_path: Path, characters: int, calls_audit: bool, masker, secret_line
) -> None:
    # Arrange: with `audit` called, the opening asks about a neighbour as well. The default masker
    # leaves "hunter2" alone until #99 lands; from then on this case measures the real masker.
    source = _holding_short_secrets(characters, calls_audit=calls_audit, secret_line=secret_line)
    index = _index(tmp_path, {"app/orders.py": source})
    client = _drex_client()
    start = place_for_line(index, "app/orders.py", 1, "start")
    judge = Judge(client, masker=masker)

    # Act
    result = find_code(index, judge, "the order total", [start], moves=CALLEES, budget=ONE_OPENING)

    # Assert
    assert result.outcome is not Outcome.FAILED
    assert client.refusals == 0
    assert len(client.requests) >= 1


def test_a_neighbour_that_fits_only_unmasked_does_not_fit_alone(tmp_path: Path) -> None:
    # Arrange: neighbours of more and more short secrets; the last one that fits unmasked
    check = FOUND
    shared = {"target": {"description": "the order total"}}
    plain, masking = Judge(_drex_client(), masker=None), Judge(_drex_client(), masker=ShortSecretMasker())

    def neighbour(lines: int) -> dict:
        code = "".join(f'DB_PASSWORD_{line:05d} = "k{line:05d}"\n' for line in range(lines))
        return {"file": "app/settings.py", "lines": [1, lines], "code": code}

    largest = max(
        lines for lines in range(400, 700) if plain.fits_alone(check, neighbour(lines), shared, "candidates")
    )

    # Act
    fits_masked = masking.fits_alone(check, neighbour(largest), shared, "candidates")

    # Assert
    assert not fits_masked


def _many_neighbours_with_short_secrets(tmp_path: Path) -> CodeIndex:
    """``place`` calls 60 helpers, each of 12 lines giving secret-named keys short values, so the
    opening's whole request lies near Drex's box at some neighbour cap."""
    helpers = "".join(
        f"def h{number}(order):\n"
        + "".join(
            f'    DB_PASSWORD_{number:02d}_{line:02d} = "k{number:02d}{line:02d}"\n' for line in range(12)
        )
        + "    return order\n\n\n"
        for number in range(60)
    )
    calls = "".join(f"    h{number}(order)\n" for number in range(60))
    names = ", ".join(f"h{number}" for number in range(60))
    place = f"from app.helpers import {names}\n\n\ndef place(order):\n{calls}    return order\n"
    commit_files(tmp_path / "repository", {"app/helpers.py": helpers, "app/orders.py": place})
    return CodeIndex(
        tmp_path / "repository", ["app/helpers.py", "app/orders.py"], fact_cache_dir=tmp_path / "facts"
    )


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_an_opening_whose_whole_request_fits_only_unmasked_is_split_before_anything_is_refused(
    tmp_path: Path, entry: str
) -> None:
    # Arrange: the whole request and the priority hint each fit unmasked at some caps and not masked
    index = _many_neighbours_with_short_secrets(tmp_path)
    start = place_for_line(index, "app/orders.py", 4, "start")

    # Act
    refused = {}
    for cap in range(14, 46):
        client = _drex_client()
        budget = SearchBudget(max_steps=1, beam_width=1, neighbours_per_kind=cap)
        judge = Judge(client, masker=ShortSecretMasker())
        find_with(entry)(index, judge, "the order total", [start], moves=CALLEES, budget=budget)
        refused[cap] = client.refusals

    # Assert
    assert {cap: count for cap, count in refused.items() if count} == {}
