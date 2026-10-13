"""The existence question over supplied pieces: one request, the pieces as file and code only, and
every eviction named."""

from __future__ import annotations

import json
from pathlib import Path

from git_repos import commit_all, write_files

from jev_navigator.directives.existence import ask_existence, existence_fits
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import items_to_judge, list_units, read_ranges
from jev_navigator.judgments.client import InputLimits
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.role_labels import LabelPiece
from jev_navigator.testing import ScriptedJevClient

FILES = {
    "app/limits.py": (
        "def check(order):\n"
        + "    order.seen = order.seen + 1  # filler\n" * 40
        + "    if len(order.items) > 4:\n        raise ValueError()\n"
    ),
    "app/orders.py": "def place(order):\n    check(order)\n    return order\n",
}


def pieces(tmp_path: Path) -> list[LabelPiece]:
    write_files(tmp_path, FILES)
    commit_all(tmp_path)
    index = CodeIndex.from_git(tmp_path)
    units = list_units(index, index.files, box_chars=76_800).units
    return [
        LabelPiece(place, read_ranges(index, place.file, place.ranges))
        for unit in units
        for place in items_to_judge(unit)
    ]


def test_every_point_is_asked_once_over_the_same_pieces_in_one_request(tmp_path: Path) -> None:
    # Arrange
    shown = pieces(tmp_path)
    client = ScriptedJevClient(nouls={"exists_limit": 0.9, "exists_refund": 0.1})
    targets = {"limit": "code that refuses a large order", "refund": "code that refunds"}

    # Act
    answers = ask_existence(Judge(client, masker=None, scanner=None), targets, shown)

    # Assert
    [(state, questions)] = client.requests
    assert state == {
        "targets": targets,
        "fetched": [{"file": piece.place.file, "code": piece.code} for piece in shown],
    }
    assert [question["instructions"] for question in questions.values()] == [
        "Does `fetched` contain the code that settles a part of `targets.limit`?",
        "Does `fetched` contain the code that settles a part of `targets.refund`?",
    ]
    assert {key: answer.probability for key, answer in answers.items()} == {"limit": 0.9, "refund": 0.1}
    assert all(answer.shown == tuple(piece.place.id for piece in shown) for answer in answers.values())
    assert all(answer.evicted == () and answer.answered_by.from_store is False for answer in answers.values())
    assert json.dumps(state).count("filler") == 40, "a piece is shown whole, never cut"


def test_pieces_that_do_not_fit_are_evicted_and_named_and_fits_says_so_beforehand(tmp_path: Path) -> None:
    shown = pieces(tmp_path)
    client = ScriptedJevClient(nouls={"exists_limit": 0.4})
    client.input_limits = InputLimits(box_chars=1_500)
    judge = Judge(client, masker=None, scanner=None)
    targets = {"limit": "code that refuses a large order"}

    assert existence_fits(judge, targets, shown[1:])
    assert not existence_fits(judge, targets, shown)
    answer = ask_existence(judge, targets, shown)["limit"]
    assert answer.evicted == (shown[0].place.id,)
    assert json.loads(json.dumps(client.requests[0][0]))["fetched"][0]["code"] == "[evicted]"
