"""A run's answers.jsonl keeps no code by default: ids, locations, hashes and names only."""

from __future__ import annotations

from pathlib import Path

from git_repos import commit_files

from jev_navigator.directives.find_code import SearchBudget, find_code
from jev_navigator.directives.places import MOVES, place_for_line
from jev_navigator.directives.trace import trace_workflow
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.rebuild import rebuild_request
from jev_navigator.judgments.relations import key_mention, without_quoted_code
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.thresholds import Thresholds
from jev_navigator.testing import ScriptedJevClient

HUB = """def handle(order, first_line_marker=None):
    return send(order, "call_site_marker")


def send(order, tag):
    return tag
"""
DESCRIBES_SLICE = Check(
    name="found",
    instructions="Does `slice.code` set a value?",
    yes=Criterion("It sets one."),
    no=Criterion("It does not."),
)
NEIGHBOUR = Check(
    name="could_contain_target",
    instructions="Does `{item}.code`, under `{item}.signature`, implement `target.description`?",
    yes=Criterion("It implements it."),
    no=Criterion("It does not."),
)


def _trace_into(pack: JsonlAnswerStore, root: Path) -> None:
    (root / "hub.py").write_text(HUB)
    index = CodeIndex.from_directory(root)
    start = index.find_definition("handle")[0]
    trace_workflow(index, Judge(ScriptedJevClient(), store=pack), "How does an order become a tag?", [start])


def test_a_trace_pack_keeps_neither_a_first_line_nor_a_call_site_line(tmp_path: Path) -> None:
    # Arrange
    pack = tmp_path / "answers.jsonl"

    # Act
    _trace_into(JsonlAnswerStore(pack), tmp_path)

    # Assert
    stored = pack.read_text()
    assert "hub.py" in stored and "handle" in stored
    assert "first_line_marker" not in stored
    assert "call_site_marker" not in stored


def test_a_find_neighbour_pack_keeps_no_signature(tmp_path: Path) -> None:
    # Arrange
    pack = tmp_path / "answers.jsonl"
    neighbour = {
        "file": "hub.py",
        "lines": [1, 2],
        "place": "hub.py:1#handle",
        "signature": "def handle(order, first_line_marker=None):",
        "code": "def handle(order, first_line_marker=None):\n    return order",
    }

    # Act
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(pack)).check_each(
        NEIGHBOUR, [neighbour], {"target": {"description": "sends an order"}}, list_name="candidates"
    )

    # Assert
    stored = pack.read_text()
    assert "hub.py:1#handle" in stored
    assert "first_line_marker" not in stored


def test_a_pack_that_keeps_requests_keeps_the_whole_skeleton(tmp_path: Path) -> None:
    # Arrange
    pack = tmp_path / "answers.jsonl"

    # Act
    _trace_into(JsonlAnswerStore(pack, keep_requests=True), tmp_path)

    # Assert
    assert "call_site_marker" in pack.read_text()


def test_rebuilding_a_request_whose_code_fields_were_withheld_names_them(tmp_path: Path) -> None:
    # Arrange
    pack = tmp_path / "answers.jsonl"
    _trace_into(JsonlAnswerStore(pack), tmp_path)
    record = JsonlAnswerStore(pack).records()[0]

    # Act
    rebuilt = rebuild_request(record, CodeIndex.from_directory(tmp_path), {})

    # Assert
    assert not rebuilt.matches
    assert any("links" in difference for difference in rebuilt.differences)


def test_a_find_pack_names_a_key_mention_without_quoting_the_key(tmp_path: Path) -> None:
    # Arrange: the opened function quotes a dictionary key; another file mentions the same key
    commit_files(
        tmp_path,
        {
            "app/limits.py": 'def item_limit(settings):\n    return settings["dict_key_marker.limit"]\n',
            "app/defaults.py": 'DEFAULTS = {\n    "dict_key_marker.limit": 4,\n}\n',
        },
    )
    index = CodeIndex.from_directory(tmp_path)
    start = place_for_line(index, "app/limits.py", 2, "start")
    pack = tmp_path / "answers.jsonl"
    judge = Judge(ScriptedJevClient(default_noul=0.6), store=JsonlAnswerStore(pack))

    # Act
    find_code(
        index,
        judge,
        "the default item limit",
        [start],
        budget=SearchBudget(max_steps=3),
        moves={"keys_mentioned": MOVES["keys_mentioned"]},
    )

    # Assert
    stored = pack.read_text()
    assert "mentions a key (app/defaults.py:" in stored
    assert "dict_key_marker" not in stored


def test_a_pack_that_keeps_requests_keeps_the_key_mention_verbatim(tmp_path: Path) -> None:
    # Arrange
    commit_files(
        tmp_path,
        {
            "app/limits.py": 'def item_limit(settings):\n    return settings["dict_key_marker.limit"]\n',
            "app/defaults.py": 'DEFAULTS = {\n    "dict_key_marker.limit": 4,\n}\n',
        },
    )
    index = CodeIndex.from_directory(tmp_path)
    start = place_for_line(index, "app/limits.py", 2, "start")
    pack = tmp_path / "answers.jsonl"
    judge = Judge(ScriptedJevClient(default_noul=0.6), store=JsonlAnswerStore(pack, keep_requests=True))

    # Act
    find_code(
        index,
        judge,
        "the default item limit",
        [start],
        budget=SearchBudget(max_steps=3),
        moves={"keys_mentioned": MOVES["keys_mentioned"]},
    )

    # Assert
    assert key_mention("dict_key_marker.limit") in pack.read_text()


def test_a_key_holding_a_backtick_is_dropped_whole_from_the_relation() -> None:
    # Arrange
    relation = key_mention("a`b dict_key_marker` tail")

    # Act
    rendered = without_quoted_code(relation, "app/defaults.py", 2)

    # Assert
    assert rendered == "mentions a key (app/defaults.py:2)"


def test_a_pack_drops_a_backtick_key_whole_from_the_sources(tmp_path: Path) -> None:
    # Arrange
    pack = tmp_path / "answers.jsonl"
    source = {
        "file": "app/defaults.py",
        "lines": [2, 2],
        "reached_by": key_mention("a`b dict_key_marker` tail"),
    }
    question = DESCRIBES_SLICE.to_question()

    # Act
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(pack)).ask(
        {"slice": {"code": "x = 1"}}, {"found": question}, thresholds=Thresholds(), sources={"found": source}
    )

    # Assert
    stored = pack.read_text()
    assert "mentions a key (app/defaults.py:2)" in stored
    assert "dict_key_marker" not in stored and "tail" not in stored
