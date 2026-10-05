from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import BudgetedClient
from git_repos import commit_files

from jev_navigator.directives.find_all import match_check
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import items_to_judge, list_units, read_ranges
from jev_navigator.judgments.answers import ChoiceAnswer, JevResponse, NoulAnswer
from jev_navigator.judgments.client import (
    JEV_INPUT_LIMITS,
    JEV_REQUEST_TOKEN_LIMIT,
    JEV_STATE_TOKEN_LIMIT,
    InputBudgetExceededError,
    InputLimits,
    MissingAnswerError,
    ReplayOnlyClient,
    UnansweredQuestionError,
    chars_for_tokens,
)
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import (
    CallCapReachedError,
    CallOffer,
    Judge,
)
from jev_navigator.judgments.questions import Check, Criterion, Pick, content_hash, serialized_chars
from jev_navigator.judgments.secrets import SecretInRequestError, SecretMasker
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.thresholds import NoulVerdict, Thresholds
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient


def _asked_item(question_id: str, state: dict) -> str:
    """The code of the item a batched question asks about, so scripts answer by content, not slot."""
    return state["items"][int(question_id.split("#")[1])]["code"]


DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)

WHOLE_STATE_CHECK = Check(
    name="whole",
    instructions="Does `doc.sentence` name a value?",
    yes=Criterion("The sentence names a value."),
    no=Criterion("The sentence names no value."),
)


@pytest.mark.parametrize(
    ("probability", "verdict"),
    [
        (0.80, NoulVerdict.YES),
        (0.7999, NoulVerdict.UNSURE),
        (0.20, NoulVerdict.NO),
        (0.2001, NoulVerdict.UNSURE),
    ],
)
def test_noul_band_edges_are_inclusive_and_the_middle_stays_unsure(probability: float, verdict) -> None:
    assert Thresholds().noul_verdict(probability) == verdict


def test_choice_confidence_of_exactly_the_minimum_counts_as_confident() -> None:
    # Arrange
    thresholds = Thresholds()

    # Act and Assert
    assert thresholds.choice_min_confidence == 0.70
    assert thresholds.choice_is_confident(0.70)
    assert not thresholds.choice_is_confident(0.6999)


def test_choice_confidence_comes_from_the_distribution_not_the_winner() -> None:
    # Act
    answer = ChoiceAnswer.from_probabilities({"a": 0.6, "b": 0.2, "c": 0.2})

    # Assert
    assert answer.choice == "a"
    assert answer.confidence == pytest.approx(0.4)


def test_thresholds_precedence_is_defaults_then_environment_then_directive_then_call() -> None:
    # Arrange
    environment = {"JEV_NAVIGATOR_CHOICE_MIN_CONFIDENCE": "0.5", "JEV_NAVIGATOR_NOUL_YES_AT": "0.9"}
    judge = Judge(ScriptedJevClient(), thresholds=Thresholds.from_env(environment))

    # Act
    effective = judge.effective({"noul_yes_at": 0.85, "noul_no_at": 0.1}, {"noul_no_at": 0.05})

    # Assert
    assert effective == Thresholds(choice_min_confidence=0.5, noul_yes_at=0.85, noul_no_at=0.05)


@pytest.mark.parametrize(
    "overrides", [{"noul_no_at": 0.8}, {"noul_yes_at": 1.2}, {"choice_min_confidence": -0.1}]
)
def test_invalid_thresholds_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        Thresholds().updated(overrides)


def test_check_each_batches_items_into_one_request_with_three_way_verdicts() -> None:
    # Arrange
    by_code = {"def a(): ...": 0.95, "def b(): ...": 0.5, "def c(): ...": 0.05}
    client = ScriptedJevClient(
        nouls=lambda question_id, _question, state: by_code[_asked_item(question_id, state)]
    )
    judge = Judge(client)
    items = [{"code": code} for code in by_code]

    # Act
    results = judge.check_each(DESCRIBES, items, {"doc": {"sentence": "a validates orders"}})

    # Assert
    assert [result.verdict for result in results] == [NoulVerdict.YES, NoulVerdict.UNSURE, NoulVerdict.NO]
    assert len(client.requests) == 1
    state, questions = client.requests[0]
    assert state["doc"] == {"sentence": "a validates orders"}
    assert list(questions.values())[1]["instructions"] == (
        "Is `items[1].code` the implementation that `doc.sentence` describes?"
    )


def test_a_check_without_criteria_sends_only_its_instructions_and_is_answered() -> None:
    # Arrange
    client = ScriptedJevClient(nouls={"plain#0": 0.95})
    plain = Check(name="plain", instructions="Does `{item}.code` validate orders?")

    # Act
    [result] = Judge(client).check_each(plain, [{"code": "def a(): ..."}])

    # Assert
    _, questions = client.requests[0]
    assert list(questions.values()) == [
        {"type": "noul", "instructions": "Does `items[0].code` validate orders?"}
    ]
    assert result.verdict == NoulVerdict.YES


@pytest.mark.parametrize("given", ["yes", "no"])
def test_a_check_with_only_one_criterion_is_refused(given: str) -> None:
    # Arrange
    one_side = {given: Criterion("The code validates orders.")}

    # Act and assert
    with pytest.raises(ValueError, match="both"):
        Check(name="half", instructions="Does `{item}.code` validate orders?", **one_side)


ADMITS = Check("admits", "Does `{item}.code` decide whether an order is admitted?")
ORDERS = (
    "LIMIT = 3\n\n\ndef admit(items):\n    return len(items) <= LIMIT\n\n\n"
    "def refuse(items):\n    return not admit(items)\n\n\nSTRICT = True\n"
)


def _judged_places(tmp_path: Path) -> tuple[CodeIndex, list, list[dict]]:
    """Every unit of a committed file as Find judges it: its place, and an entry of file and code."""
    repo = tmp_path / "repo"
    repo.mkdir()
    commit_files(repo, {"orders.py": ORDERS})
    index = CodeIndex.from_git(repo)
    units = list_units(index, ["orders.py"], box_chars=JEV_INPUT_LIMITS.box_chars).units
    places = [item for unit in units for item in items_to_judge(unit)]
    entries = [{"file": place.file, "code": read_ranges(index, place.file, place.ranges)} for place in places]
    return index, places, entries


def test_an_item_judged_at_a_place_sends_only_its_own_fields_and_its_answer_names_the_place(
    tmp_path: Path,
) -> None:
    # Arrange
    index, places, entries = _judged_places(tmp_path)
    client = ScriptedJevClient(default_noul=0.9)

    # Act
    results = Judge(client).check_each(ADMITS, entries, places=places)

    # Assert
    state, _ = client.requests[0]
    assert [sorted(item) for item in state["items"]] == [["code", "file"]] * len(places)
    assert sorted(result.place.id for result in results) == sorted(place.id for place in places)
    assert all(
        result.item["code"] == read_ranges(index, result.place.file, result.place.ranges)
        for result in results
    )


def test_items_judged_at_places_are_sent_in_line_order_whatever_order_they_came_in(tmp_path: Path) -> None:
    # Arrange
    _, places, entries = _judged_places(tmp_path)
    by_content = sorted(entries, key=content_hash)
    in_line_order = sorted(zip(places, entries, strict=True), key=lambda pair: pair[0].ranges[0][0])
    by_line = [entry for _, entry in in_line_order]
    assert by_content != by_line, "the content order must differ from the line order for this test to tell"
    client = ScriptedJevClient(default_noul=0.9)

    # Act
    Judge(client).check_each(ADMITS, list(reversed(entries)), places=list(reversed(places)))

    # Assert
    state, _ = client.requests[0]
    assert state["items"] == by_line


def test_items_judged_before_with_the_same_batch_mates_are_answered_from_the_store(tmp_path: Path) -> None:
    # Arrange
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    items = [{"code": "x = 1"}, {"code": "y = 2"}]
    Judge(ScriptedJevClient(nouls={"describes": 0.9}), store=store).check_each(
        DESCRIBES, items, {"doc": {"sentence": "s"}}
    )
    second_client = ScriptedJevClient(nouls={"describes": 0.1})
    second = Judge(
        second_client, store=JsonlAnswerStore(tmp_path / "answers.jsonl"), served_model="jev-scripted"
    )

    # Act
    results = second.check_each(DESCRIBES, items, {"doc": {"sentence": "s"}})

    # Assert
    assert [(result.probability, result.from_store) for result in results] == [(0.9, True), (0.9, True)]
    assert second_client.requests == []


def test_independent_checks_share_a_request_and_reuse_only_the_matching_cached_answers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answers.jsonl"
    shared = {"doc": {"sentence": "sets a value"}}
    items = [{"code": "x = 1"}, {"code": "y = 2"}]
    changes = Check(
        "changes",
        "Does `{item}.code` change a value?",
        Criterion("It changes a value."),
        Criterion("It does not change a value."),
    )
    Judge(ScriptedJevClient(nouls={"describes": 0.9}), store=JsonlAnswerStore(path)).check_each(
        DESCRIBES, items, shared
    )
    changes_by_code = {"x = 1": 0.5, "y = 2": 0.95}
    client = ScriptedJevClient(
        nouls=lambda question_id, _question, state: changes_by_code[_asked_item(question_id, state)]
    )
    judge = Judge(client, store=JsonlAnswerStore(path), served_model="jev-scripted")

    results = judge.check_every([DESCRIBES, changes], items, shared)

    assert len(client.requests) == 1
    state, questions = client.requests[0]
    assert sorted(item["code"] for item in state["items"]) == ["x = 1", "y = 2"]
    assert set(questions) == {f"{changes.question_id}#0", f"{changes.question_id}#1"}
    assert [(r.item, r.probability, r.from_store) for r in results["describes"]] == [
        (items[0], 0.9, True),
        (items[1], 0.9, True),
    ]
    assert [r.verdict for r in results["changes"]] == [NoulVerdict.UNSURE, NoulVerdict.YES]

    replay_client = ScriptedJevClient(default_noul=0.01)
    replay = Judge(replay_client, store=JsonlAnswerStore(path), served_model="jev-scripted")
    reordered = replay.check_every([changes, DESCRIBES], items[::-1], shared)
    assert replay_client.requests == []
    for check in (changes, DESCRIBES):
        assert [(r.item, r.probability, r.request_sha256) for r in reordered[check.name]] == [
            (r.item, r.probability, r.request_sha256) for r in reversed(results[check.name])
        ]
        assert all(r.from_store for r in reordered[check.name])


def test_the_same_item_against_different_shared_state_is_judged_again(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    Judge(ScriptedJevClient(nouls={"describes": 0.9}), store=JsonlAnswerStore(path)).check_each(
        DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "x is set to one"}}
    )
    client = ScriptedJevClient(nouls={"describes": 0.1})

    # Act
    results = Judge(client, store=JsonlAnswerStore(path), served_model="jev-scripted").check_each(
        DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "x is never set"}}
    )

    # Assert
    assert (results[0].probability, results[0].from_store) == (0.1, False)
    assert len(client.requests) == 1


def test_answers_from_another_served_model_are_not_reused(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    shared = {"doc": {"sentence": "s"}}
    Judge(
        ScriptedJevClient(nouls={"describes": 0.9}, model="jev-1.12.0"), store=JsonlAnswerStore(path)
    ).check_each(DESCRIBES, [{"code": "x = 1"}], shared)
    upgraded = ScriptedJevClient(nouls={"describes": 0.1}, model="jev-1.13.0")

    # Act
    results = Judge(upgraded, store=JsonlAnswerStore(path), served_model="jev-1.13.0").check_each(
        DESCRIBES, [{"code": "x = 1"}], shared
    )

    # Assert
    assert results[0].from_store is False and len(upgraded.requests) == 1


def test_an_unknown_served_model_is_a_miss_until_the_first_live_answer(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    shared = {"doc": {"sentence": "s"}}
    Judge(ScriptedJevClient(nouls={"describes": 0.9}), store=JsonlAnswerStore(path)).check_each(
        DESCRIBES, [{"code": "x = 1"}], shared
    )
    client = ScriptedJevClient(nouls={"describes": 0.1})
    judge = Judge(client, store=JsonlAnswerStore(path))

    # Act
    first = judge.check_each(DESCRIBES, [{"code": "x = 1"}], shared)
    second = judge.check_each(DESCRIBES, [{"code": "x = 1"}], shared)

    # Assert
    assert first[0].from_store is False
    assert second[0].from_store is True


def test_stored_answers_replay_without_calls_under_new_thresholds(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    Judge(ScriptedJevClient(nouls={"describes": 0.7}), store=JsonlAnswerStore(path)).check_each(
        DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "s"}}
    )
    replay = Judge(ReplayOnlyClient(), store=JsonlAnswerStore(path), thresholds=Thresholds(noul_yes_at=0.65))

    # Act
    results = replay.check_each(DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "s"}})

    # Assert
    assert results[0].verdict == NoulVerdict.YES
    assert replay.calls == 0


def test_replay_without_a_stored_answer_fails_loudly(tmp_path: Path) -> None:
    judge = Judge(ReplayOnlyClient(), store=JsonlAnswerStore(tmp_path / "answers.jsonl"))
    with pytest.raises(MissingAnswerError):
        judge.check_each(DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "s"}})


def test_store_keeps_answers_and_thresholds_but_no_request_text_by_default(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"

    # Act
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(path)).check_each(
        DESCRIBES, [{"code": "customer_code()"}], {"doc": {"sentence": "s"}}
    )

    # Assert
    stored = path.read_text()
    record = json.loads(stored.splitlines()[0])
    assert "customer_code" not in stored
    assert record["request"] is None and record["sent_body_base64"] is None
    assert '"noul_yes_at": 0.8' in stored


def test_records_written_before_the_sent_body_was_kept_still_load(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(path)).check_each(
        DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "s"}}
    )
    older = {
        key: value
        for key, value in json.loads(path.read_text()).items()
        if key not in {"sent_body_base64", "sent_exact"}
    }
    path.write_text(json.dumps(older) + "\n")

    # Act
    record = JsonlAnswerStore(path).records()[0]

    # Assert
    assert record.sent_exact is False
    with pytest.raises(ValueError, match="keep_requests"):
        record.sent_request()


def test_secrets_are_masked_before_any_request_leaves() -> None:
    # Arrange
    client = ScriptedJevClient()
    key_block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    items = [{"code": f'TOKEN = "ghp_{"a1" * 18}"\n{key_block}\npassword = "hunter2hunter2"'}]

    # Act
    Judge(client).check_each(DESCRIBES, items, {"doc": {"sentence": "s"}})

    # Assert
    sent = str(client.requests[0][0])
    assert "ghp_" not in sent and "MIIEow" not in sent and "hunter2" not in sent
    assert "PRIVATE KEY" not in sent


PLANTED_VALUE = "order-hook-4f7a1c"


def test_a_value_masked_in_one_place_is_masked_everywhere_in_the_request() -> None:
    # Arrange
    client = ScriptedJevClient()
    state = {
        "slice": {"code": f'WEBHOOK_TOKEN = "{PLANTED_VALUE}"'},
        "candidates": [{"signature": f'hooks.py:9 `send(order)` (mentions "{PLANTED_VALUE}")'}],
    }
    pick = Pick("open_first", "Which entry of `candidates` most likely sends the order?")

    # Act
    Judge(client).pick(pick, {"0": f'hooks.py:9 (mentions "{PLANTED_VALUE}")', "1": "other.py:3"}, state)

    # Assert
    sent_state, sent_questions = client.requests[0]
    sent = json.dumps([sent_state, sent_questions])
    assert PLANTED_VALUE not in sent
    assert sent_state["slice"]["code"] == 'WEBHOOK_TOKEN = "[MASKED]"'
    assert '(mentions "[MASKED]")' in sent_state["candidates"][0]["signature"]


def test_a_value_masked_in_one_item_is_masked_in_the_other_items_of_its_batch() -> None:
    # Arrange
    client = ScriptedJevClient()
    items = [{"code": f'WEBHOOK_TOKEN = "{PLANTED_VALUE}"'}, {"code": f'post("{PLANTED_VALUE}", order)'}]

    # Act
    Judge(client).check_each(DESCRIBES, items, {"doc": {"sentence": "s"}})

    # Assert
    sent_items = client.requests[0][0]["items"]
    assert [item["code"] for item in sent_items] == ['WEBHOOK_TOKEN = "[MASKED]"', 'post("[MASKED]", order)']


class CountingMasker(SecretMasker):
    """The built-in masker, counting how often each text is masked."""

    def __init__(self) -> None:
        self.masked: Counter[str] = Counter()

    def mask(self, text: str, path: str | None = None) -> str:
        self.masked[text] += 1
        return super().mask(text, path)


def test_each_item_is_masked_once_per_judging_call_across_several_batches() -> None:
    # Arrange
    masker = CountingMasker()
    client = ScriptedJevClient()
    items = [{"code": f"def part{index}():\n    return {index}"} for index in range(4)]

    # Act
    Judge(client, masker=masker, items_per_request=1).check_each(DESCRIBES, items, {"doc": {"sentence": "s"}})

    # Assert
    assert len(client.requests) > 1
    assert [masker.masked[item["code"]] for item in items] == [1, 1, 1, 1]


def test_the_final_scan_refuses_a_secret_that_masking_cannot_reach_on_the_batch_path() -> None:
    # Arrange
    client = ScriptedJevClient()
    items = [{"code": "x = 1"}, {"code": "y = 2", f"ghp_{'c3' * 18}": "key text is never masked"}]

    # Act and assert
    with pytest.raises(SecretInRequestError):
        Judge(client).check_each(DESCRIBES, items, {"doc": {"sentence": "s"}})
    assert client.requests == []


def test_the_final_scan_refuses_a_masked_value_that_is_also_a_state_key() -> None:
    # Arrange
    client = ScriptedJevClient()
    state = {"slice": {"code": f'WEBHOOK_TOKEN = "{PLANTED_VALUE}"'}, PLANTED_VALUE: {"code": "x = 1"}}

    # Act and assert
    with pytest.raises(SecretInRequestError):
        Judge(client).ask(state, {"q": DESCRIBES.to_question()}, thresholds=Thresholds())
    assert client.requests == []


def test_the_final_scan_refuses_when_a_host_turns_masking_off() -> None:
    # Arrange
    client = ScriptedJevClient()
    judge = Judge(client, masker=None)

    # Act and Assert
    with pytest.raises(SecretInRequestError):
        judge.check_each(DESCRIBES, [{"code": f'key = "ghp_{"b2" * 18}"'}], {})
    assert client.requests == []


def test_high_entropy_values_in_assignments_are_masked_and_ordinary_strings_are_not() -> None:
    # Act
    masked = SecretMasker().mask(
        'seed = "Zq8vN3xL0pR7tY2wK5mB9cH4"\nlabel = "orders.max_items.default.value"'
    )

    # Assert
    assert "Zq8vN3" not in masked
    assert "orders.max_items.default.value" in masked


def test_options_that_contain_a_secret_are_never_offered() -> None:
    # Arrange
    client = ScriptedJevClient()
    pick = Pick(
        "which_key", "Which of these string keys most likely names the setting `claim.text` mentions?"
    )
    options = {"orders.max_items": "3 hits", f"ghp_{'c3' * 18}": "1 hit"}

    # Act
    result = Judge(client).pick(pick, options, {"claim": {"text": "the limit is never read"}})

    # Assert
    assert result.choice == "orders.max_items"
    assert list(client.requests[0][1].values())[0]["criteria"] == {"orders.max_items": "3 hits"}


def test_choose_call_asks_every_argument_in_one_request_and_reports_low_confidence() -> None:
    # Arrange
    route = Pick("route", "Which lookup most likely reaches the check that `claim.text` says is missing?")
    offers = [
        CallOffer(
            "find_callers",
            "the functions that call a name",
            Pick("caller_of", "Whose callers?"),
            {"validate_order": "app/validation.py:4"},
        ),
        CallOffer(
            "search_text",
            "the lines containing a string key",
            Pick("key", "Which key?"),
            {"orders.max_items": "2 hits"},
        ),
    ]
    client = ScriptedJevClient(choices={"route": {"find_callers": 0.55, "search_text": 0.45}})

    # Act
    decision = Judge(client).choose_call(route, offers, {"claim": {"text": "no limit check"}})

    # Assert
    assert (decision.operation, decision.argument) == ("find_callers", "validate_order")
    assert not decision.confident
    assert len(client.requests) == 1 and len(client.requests[0][1]) == 3


def test_store_records_where_every_judged_item_came_from(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    items = [{"file": "app/orders.py", "lines": [5, 7], "commit": "abc123", "code": "def place(): ..."}]

    # Act
    Judge(ScriptedJevClient(), store=JsonlAnswerStore(path)).check_each(
        DESCRIBES, items, {"doc": {"sentence": "s"}}
    )

    # Assert
    record = json.loads(path.read_text())
    assert list(record["sources"].values()) == [
        {"file": "app/orders.py", "lines": [5, 7], "commit": "abc123"}
    ]
    assert "def place" not in path.read_text()


def test_ask_all_sends_checks_picks_and_scores_over_one_state_in_one_request() -> None:
    # Arrange
    from jev_navigator.judgments.questions import Rate

    accurate = Check(
        "accurate",
        "Does `comment.text` match what `slice.code` does?",
        Criterion("The comment states what the code does."),
        Criterion("The comment says otherwise."),
    )
    kind = Pick("kind", "What kind of comment is `comment.text`?")
    usefulness = Rate(
        "useful",
        "How much does `comment.text` add beyond `slice.code`?",
        ("Repeats the code.", "Adds a little context.", "Explains a reason the code cannot show."),
    )
    client = ScriptedJevClient(
        nouls={"accurate": 0.3},
        choices={"kind": {"why": 0.8, "what": 0.2}},
        scores={"useful": [0.1, 0.2, 0.7]},
    )
    state = {"comment": {"text": "retry because the API drops the first call"}, "slice": {"code": "retry()"}}

    # Act
    answers = Judge(client).ask_all(
        state, checks=[accurate], picks=[(kind, {"why": "", "what": ""})], scores=[usefulness]
    )

    # Assert
    assert len(client.requests) == 1
    assert answers.checks["accurate"].probability == 0.3
    assert answers.checks["accurate"].verdict == NoulVerdict.UNSURE
    assert answers.picks["kind"].probabilities == {"why": 0.8, "what": 0.2}
    assert answers.scores["useful"].score == pytest.approx(1.6)
    assert answers.scores["useful"].probabilities == {"0": 0.1, "1": 0.2, "2": 0.7}


def test_score_answers_survive_the_store_round_trip(tmp_path: Path) -> None:
    # Arrange
    from jev_navigator.judgments.questions import Rate

    rate = Rate("useful", "How useful is `x`?", ("not", "somewhat", "very"))
    path = tmp_path / "answers.jsonl"
    Judge(ScriptedJevClient(scores={"useful": [0.0, 0.5, 0.5]}), store=JsonlAnswerStore(path)).ask_all(
        {"x": 1}, scores=[rate]
    )

    # Act
    replayed = Judge(ReplayOnlyClient(), store=JsonlAnswerStore(path)).ask_all({"x": 1}, scores=[rate])

    # Assert
    assert replayed.scores["useful"].score == pytest.approx(1.5)


def test_a_whole_request_answered_by_another_model_is_asked_again(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    state = {"items": [{"code": "def increment(x): return x + 1"}], "doc": {"sentence": "Add one."}}
    questions = {"describes": DESCRIBES.to_question("items[0]")}
    Judge(ScriptedJevClient(default_noul=0.95, model="model-a"), store=JsonlAnswerStore(path)).ask(
        state, questions, thresholds=Thresholds()
    )
    second = ScriptedJevClient(default_noul=0.05, model="model-b")
    judge = Judge(second, store=JsonlAnswerStore(path), served_model="model-b")

    # Act
    answer = judge.ask(state, questions, thresholds=Thresholds())

    # Assert
    assert answer.model == "model-b" and len(second.requests) == 1
    assert answer.request_sha256 == JsonlAnswerStore(path).records()[0].request_sha256


def test_a_judge_refuses_to_send_past_its_global_call_cap() -> None:
    # Arrange
    client = ScriptedJevClient(nouls={"describes": 0.9})
    judge = Judge(client, max_calls=1)
    judge.check_each(DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "s"}})

    # Act / Assert
    with pytest.raises(CallCapReachedError):
        judge.check_each(DESCRIBES, [{"code": "y = 2"}], {"doc": {"sentence": "s"}})
    assert len(client.requests) == 1


def test_a_scoped_judge_counts_its_own_calls_and_adds_them_to_its_parent() -> None:
    # Arrange
    judge = Judge(ScriptedJevClient(nouls={"describes": 0.9}))
    first, second = judge.scope(), judge.scope()

    # Act
    first.check_each(DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "s"}})
    second.check_each(DESCRIBES, [{"code": "y = 2"}], {"doc": {"sentence": "s"}})
    second.check_each(DESCRIBES, [{"code": "z = 3"}], {"doc": {"sentence": "s"}})

    # Assert
    assert (first.calls, second.calls, judge.calls) == (1, 2, 3)
    assert judge.input_total.reported == 300
    assert judge.input_total.not_reported == 0


class _UsageClient:
    """Answers each call with the next scripted input-token value; None is a response without usage."""

    model = "jev-scripted"

    def __init__(self, *tokens: int | None) -> None:
        self.tokens = list(tokens)

    def ask(self, state, questions):
        answers = {question_id: NoulAnswer(0.9) for question_id in questions}
        return JevResponse(answers, self.model, self.tokens.pop(0))


def test_a_response_without_usage_is_reported_as_not_reported_and_never_added_as_zero() -> None:
    # Arrange
    judge = Judge(_UsageClient(10, None, 0))

    # Act
    answered = [
        judge.ask({"s": index}, {"q": {"type": "noul"}}, thresholds=Thresholds()) for index in range(3)
    ]

    # Assert
    assert [response.input_tokens for response in answered] == [10, None, 0]
    assert judge.input_total.reported == 10
    assert judge.input_total.not_reported == 1
    assert judge.unanswered_requests == 0


def test_a_reported_zero_counts_as_reported_not_missing() -> None:
    # Arrange
    judge = Judge(_UsageClient(10, 0))

    # Act
    for index in range(2):
        judge.ask({"s": index}, {"q": {"type": "noul"}}, thresholds=Thresholds())

    # Assert
    assert (judge.input_total.reported, judge.input_total.not_reported) == (10, 0)


def test_a_scoped_judge_adds_unreported_responses_to_its_parent() -> None:
    judge = Judge(_UsageClient(None))
    scoped = judge.scope()

    scoped.ask({"s": 1}, {"q": {"type": "noul"}}, thresholds=Thresholds())

    assert (scoped.input_total.not_reported, judge.input_total.not_reported) == (1, 1)


class _DropsAnAnswer(ScriptedJevClient):
    """A provider whose response leaves out the answer to the question about the second item."""

    def _answer_all(self, state: Mapping, questions: Mapping) -> JevResponse:
        answered = super()._answer_all(state, questions)
        kept = {
            question_id: answer
            for question_id, answer in answered.answers.items()
            if not question_id.endswith("#1")
        }
        return replace(answered, answers=kept)


def test_a_response_missing_an_asked_answer_is_refused_and_leaves_the_run_pack_readable(
    tmp_path: Path,
) -> None:
    # Arrange
    path = tmp_path / "answers.jsonl"
    client = _DropsAnAnswer(default_noul=0.9)
    judge = Judge(client, store=JsonlAnswerStore(path))

    # Act
    with pytest.raises(UnansweredQuestionError) as refused:
        judge.check_each(DESCRIBES, [{"code": "x = 1"}, {"code": "y = 2"}], {"doc": {"sentence": "s"}})

    # Assert: the refusal names the unanswered question, the paid response still counts, and the
    # pack a later run opens holds nothing it cannot read
    unanswered = list(client.requests[0][1])[1]
    assert unanswered in str(refused.value)
    assert judge.input_total.reported == 100
    assert JsonlAnswerStore(path).records() == ()


def test_a_store_replay_is_marked_replayed_and_reports_no_token_count(tmp_path: Path) -> None:
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    state, questions = {"slice": {"code": "x = 1"}}, {"q": {"type": "noul", "instructions": "Is it?"}}
    Judge(ScriptedJevClient(), store=store).ask(state, questions, thresholds=Thresholds())
    replaying = Judge(ScriptedJevClient(), store=store, served_model="jev-scripted")

    replayed = replaying.ask(state, questions, thresholds=Thresholds())

    assert replayed.from_store is True
    assert replayed.input_tokens is None
    assert replaying.input_total.responses == 0
    assert replaying.calls == 0


def test_every_result_carries_the_hash_of_the_masked_request_that_answered_it(tmp_path: Path) -> None:
    # Arrange
    from jev_navigator.judgments.questions import Rate, request_sha256

    client = ScriptedJevClient(nouls={"describes": 0.9})
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    judge = Judge(client, store=store)
    secret_state = {"doc": {"sentence": f'TOKEN = "ghp_{"a1" * 18}"'}}
    kind = Pick("kind", "What kind of comment is `doc.sentence`?")
    offers = [CallOffer("search_text", "lines with a key", Pick("key", "Which key?"), {"k": "1 hit"})]
    usefulness = Rate("useful", "How much does `doc.sentence` add?", ("Nothing.", "Something."))

    # Act
    checked = judge.check_each(DESCRIBES, [{"code": "x = 1"}], secret_state)
    replayed = judge.check_each(DESCRIBES, [{"code": "x = 1"}], secret_state)
    everything = judge.ask_all(
        secret_state, checks=[WHOLE_STATE_CHECK], picks=[(kind, {"why": ""})], scores=[usefulness]
    )
    picked = judge.pick(kind, {"why": "", "what": ""}, secret_state)
    decision = judge.choose_call(Pick("route", "Which lookup?"), offers, secret_state)

    # Assert
    sent_hashes = [request_sha256(state, questions) for state, questions in client.requests]
    assert "ghp_" not in str(client.requests)
    assert checked[0].request_sha256 == sent_hashes[0]
    assert replayed[0].from_store and replayed[0].request_sha256 == sent_hashes[0]
    assert everything.request_sha256 == sent_hashes[1]
    assert everything.checks["whole"].request_sha256 == sent_hashes[1]
    assert everything.picks["kind"].request_sha256 == sent_hashes[1]
    assert everything.scores["useful"].request_sha256 == sent_hashes[1]
    assert picked.request_sha256 == sent_hashes[2]
    assert decision.request_sha256 == sent_hashes[3]
    assert decision.route.request_sha256 == sent_hashes[3]


def test_independent_checks_cannot_silently_share_a_result_name() -> None:
    client = ScriptedJevClient(default_noul=0.9)
    judge = Judge(client)
    other = replace(DESCRIBES, instructions="Does `{item}.code` write a database row?")

    with pytest.raises(ValueError, match="unique names"):
        judge.check_every([DESCRIBES, other], [{"code": "return 1"}])

    assert client.requests == []


def _hub_item(chars: int, links: int = 0) -> dict:
    """The sanitized shape of the failing trace item: a tiny span whose link context dominates."""
    item = {
        "file": "src/hub.py",
        "lines": [45, 47],
        "commit": "53bc622413712cb0f63ac0f9954bf1971b69f485",
        "span_key": "src/hub.py:45-47",
        "code": "answers = {question_id: answer_from_json(raw) for question_id, raw in self.answers.items()}",
    }
    if links:
        item["links"] = [
            f"call answer_from_json: self#response -> src/other.py:{index}#answer_from_json | "
            f"at src/hub.py:46 {'x' * chars}"
            for index in range(links)
        ]
    return item


def _padding_item(label: str, chars: int) -> dict:
    return {"file": f"{label}.py", "lines": [1, 2], "code": f"def {label}():\n    {'y' * chars}"}


def test_items_that_would_overflow_one_request_are_packed_so_no_request_exceeds_the_input_budget() -> None:
    client = BudgetedClient(JEV_INPUT_LIMITS.request_chars)
    judge = Judge(client)
    items = [_padding_item(f"part{index}", 28_000) for index in range(4)]

    results = judge.check_every([DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts")

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 4
    assert client.refusals == 0, "a request the measurement already rejects must not be paid for"
    assert len(client.requests) == 2
    judged_files: list[str] = []
    for state, questions in client.requests:
        body = serialized_chars({"state": state, "questions": questions})
        assert body <= JEV_INPUT_LIMITS.request_chars
        assert len(questions) == len(state["parts"]), "one atomic question per item and slot"
        judged_files.extend(item["file"] for item in state["parts"])
    assert sorted(judged_files) == [f"part{index}.py" for index in range(4)]


def test_non_ascii_items_are_packed_by_their_escaped_size_so_every_request_fits_the_box() -> None:
    # Arrange: each item is about 4,000 characters as text but about 24,000 escaped, as Jev counts it
    client = _boxed_client()
    items = [
        {
            "file": f"part{index}.py",
            "lines": [1, 2],
            "code": f"def part{index}():\n    return '{'名' * 4_000}'",
        }
        for index in range(6)
    ]

    # Act
    results = Judge(client).check_every([DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts")

    # Assert: measured here with json.dumps itself, so a library measure that stopped escaping fails it
    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 6
    assert client.refusals == 0
    assert len(client.requests) > 1
    for state, questions in client.requests:
        longest = max(len(json.dumps(question)) for question in questions.values())
        assert len(json.dumps(state)) + longest <= JEV_INPUT_LIMITS.box_chars


def _boxed_client() -> BudgetedClient:
    return BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=JEV_INPUT_LIMITS.box_chars)


def _judge_padded_parts(client: BudgetedClient, count: int, chars: int):
    items = [_padding_item(f"part{index}", chars) for index in range(count)]
    return Judge(client).check_every([DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts")


def test_a_state_just_under_the_character_box_is_sent_unsplit() -> None:
    client = _boxed_client()

    results = _judge_padded_parts(client, count=4, chars=18_800)

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 4
    assert len(client.requests) == 1
    assert client.refusals == 0


def test_a_state_just_over_the_character_box_is_split_before_sending_and_every_item_is_judged() -> None:
    client = _boxed_client()

    results = _judge_padded_parts(client, count=4, chars=19_300)

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 4
    assert client.refusals == 0, "a request the measurement already rejects must not be paid for"
    assert len(client.requests) == 2
    judged_files = [item["file"] for state, _ in client.requests for item in state["parts"]]
    assert sorted(judged_files) == [f"part{index}.py" for index in range(4)]


def test_many_small_questions_over_a_moderate_state_fit_because_only_the_longest_question_counts() -> None:
    state = {"parts": [{"code": "y" * 60_000}]}
    questions = {f"q{index}": {"ask": "x" * 400} for index in range(200)}

    assert not JEV_INPUT_LIMITS.exceeded_by(state, questions)


def test_the_longest_question_counts_towards_the_character_box() -> None:
    state = {"parts": [{"code": "y" * 60_000}]}
    questions = {"short": {"ask": "x"}, "long": {"ask": "x" * 17_000}}

    assert JEV_INPUT_LIMITS.exceeded_by(state, questions)
    assert not JEV_INPUT_LIMITS.exceeded_by(state, {"short": questions["short"]})


def test_a_body_over_the_request_box_is_over_budget_although_state_and_question_fit() -> None:
    state = {"doc": {"sentence": "s"}}
    questions = {f"q{index}": {"ask": "x" * 40} for index in range(5_000)}

    assert JEV_INPUT_LIMITS.exceeded_by(state, questions)
    assert not JEV_INPUT_LIMITS.exceeded_by(state, dict(list(questions.items())[:100]))


def test_non_ascii_state_is_measured_as_the_escaped_body_the_engine_measures() -> None:
    chinese_comments = "\u4e2d" * 20_000
    state = {"parts": [{"code": chinese_comments}]}

    assert len(json.dumps(state, ensure_ascii=False)) < JEV_INPUT_LIMITS.box_chars
    assert JEV_INPUT_LIMITS.exceeded_by(state, {"q": {"ask": "x"}})


def test_every_box_derives_from_the_one_characters_per_token_constant() -> None:
    assert chars_for_tokens(JEV_STATE_TOKEN_LIMIT) == JEV_INPUT_LIMITS.box_chars == 76_800
    assert chars_for_tokens(JEV_REQUEST_TOKEN_LIMIT) == JEV_INPUT_LIMITS.request_chars == 153_600
    assert chars_for_tokens(8_192) == 19_660


def test_serialized_chars_counts_every_escaped_character() -> None:
    assert serialized_chars({"a": "\u4e2d" * 10}) == len('{"a": "' + "\\u4e2d" * 10 + '"}')


def test_provider_max_tokens_error_splits_the_batch_and_keeps_every_question_identity() -> None:
    client = BudgetedClient(34_000)  # stricter than the measured packing budget
    judge = Judge(client)
    items = [_padding_item(f"part{index}", 12_000) for index in range(4)]

    results = judge.check_every([DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts")

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 4
    assert client.refusals == 1, "the first over-budget request is the provider's own evidence"
    assert len(client.requests) == 2
    for state, questions in client.requests:
        body = serialized_chars({"state": state, "questions": questions})
        assert body <= 34_000
    # Each item keeps its own store identity across the split: every item is judged exactly once.
    keys = [result.item["file"] for result in results["describes"]]
    assert keys == ["part0.py", "part1.py", "part2.py", "part3.py"]


def test_unsplittable_single_question_is_reported_honestly_after_a_real_attempt(tmp_path: Path) -> None:
    client = BudgetedClient(34_000)
    journal = JsonlJournal(tmp_path / "journal.jsonl")
    judge = Judge(client, journal=journal)
    item = _hub_item(16_000, links=40)

    with pytest.raises(InputBudgetExceededError):
        judge.check_every([DESCRIBES], [item], {"doc": {"sentence": "s"}}, list_name="parts")

    assert client.refusals == 1, "the provider, not the estimate, reports the unsplittable request"
    failures = [
        json.loads(line)
        for line in journal.path.read_text().splitlines()
        if json.loads(line)["kind"] == "failure"
    ]
    assert len(failures) == 1
    assert failures[0]["error_type"] == "InputBudgetExceededError"


def test_split_answers_replay_from_the_store_without_new_calls(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    first = BudgetedClient(34_000)
    judge = Judge(first, store=JsonlAnswerStore(path), served_model="jev-scripted")
    items = [_padding_item(f"part{index}", 12_000) for index in range(4)]
    judge.check_every([DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts")

    replay_client = BudgetedClient(34_000)
    replay = Judge(replay_client, store=JsonlAnswerStore(path), served_model="jev-scripted")
    results = replay.check_every([DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts")

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 4
    assert replay_client.requests == [] and replay_client.refusals == 0


def test_a_normal_small_batch_is_unchanged_by_the_input_budget_boundary() -> None:
    client = BudgetedClient(JEV_INPUT_LIMITS.request_chars)
    judge = Judge(client)
    items = [{"code": "def a(): ..."}, {"code": "def b(): ..."}, {"code": "def c(): ..."}]

    results = judge.check_each(DESCRIBES, items, {"doc": {"sentence": "s"}})

    assert [result.verdict for result in results] == [NoulVerdict.YES] * 3
    assert len(client.requests) == 1
    assert client.refusals == 0
    state, questions = client.requests[0]
    assert set(questions) == {f"{DESCRIBES.question_id}#{slot}" for slot in range(3)}


def test_input_budget_error_is_typed_from_the_provider_report_without_the_sdk() -> None:
    from jev_navigator.judgments.client import input_budget_error

    class ProviderError(Exception):
        def __init__(self) -> None:
            super().__init__("POST https://gateway/v1/systemone: 400")
            self.status = 400
            self.body = {"detail": {"error_type": "max_tokens_exceeded"}}

    class OtherProviderError(Exception):
        status = 400
        body = {"detail": {"error_type": "question_malformed"}}

    assert isinstance(input_budget_error(ProviderError()), InputBudgetExceededError)
    assert input_budget_error(OtherProviderError()) is None
    assert input_budget_error(TypeError("no status at all")) is None


WORDING_ONLY_SECRET = "Zq9xW2pL7vB4mNc8"
"""A value the masker recognises only in a check's wording, where ``api_key = "..."`` marks it."""


def _judged_with_wording_secret(path: str, judge: Judge, check: Check, items: list[dict]) -> None:
    shared = {"doc": {"sentence": "s"}}
    if path == "check_each":
        judge.check_each(check, items, shared)
    elif path == "check_every":
        judge.check_every([check], items, shared)
    else:
        asyncio.run(judge.check_each_async(check, items, shared))


@pytest.mark.parametrize("path", ["check_each", "check_every", "check_each_async"])
def test_a_value_masked_in_check_wording_is_masked_in_the_batch_state_too(path: str) -> None:
    # Arrange
    check = Check(
        "uses_key",
        f'Does `{{item}}.code` call the service configured with api_key = "{WORDING_ONLY_SECRET}"?',
        Criterion("yes"),
        Criterion("no"),
    )
    items = [{"file": "a.py", "code": f'client.connect("{WORDING_ONLY_SECRET}")'}]
    client = AsyncScriptedJevClient() if path == "check_each_async" else ScriptedJevClient()

    # Act
    _judged_with_wording_secret(path, Judge(client), check, items)

    # Assert
    [(state, questions)] = client.requests
    assert WORDING_ONLY_SECRET not in json.dumps(questions)
    assert WORDING_ONLY_SECRET not in json.dumps(state)


def test_a_secret_in_check_wording_is_masked_once_per_plan_and_the_request_still_goes(tmp_path: Path) -> None:
    # Arrange
    token = f"ghp_{'d4' * 18}"
    wording = f"Does `{{item}}.code` use the token {token}?"
    leaky = Check("leaky", wording, Criterion("Yes."), Criterion("No."))
    masker = CountingMasker()
    client = ScriptedJevClient()
    items = [{"file": f"f{index}.py", "code": f"v = {index}"} for index in range(3)]

    # Act
    Judge(client, masker=masker, items_per_request=1).check_each(leaky, items, {"doc": {"sentence": "s"}})

    # Assert
    sent = json.dumps(client.requests)
    assert len(client.requests) == 3
    assert token not in sent and "[MASKED]" in sent
    wording_masks = sum(count for text, count in masker.masked.items() if token in text)
    assert wording_masks == 1, "every request here asks at slot 0, so its wording is masked once"


def _limited_judge(request_chars: int) -> Judge:
    client = BudgetedClient(request_chars, input_box=100_000)
    client.input_limits = InputLimits(100_000, request_chars)
    return Judge(client)


def test_fits_alone_measures_every_check_a_request_asks_of_the_item() -> None:
    # Arrange: the smallest request limit at which one question about the item fits
    first, second = match_check("first"), match_check("second")
    shared = {"targets": {"first": "the order limit", "second": "the audit call"}}
    item = {"file": "app/orders.py", "code": "def accept(order):\n    return len(order.items) <= 4"}
    request_chars = next(
        chars for chars in range(1, 20_000) if _limited_judge(chars).fits_alone([first], item, shared)
    )
    judge = _limited_judge(request_chars)

    # Act
    one_fits = judge.fits_alone([first], item, shared)
    both_fit = judge.fits_alone([first, second], item, shared)

    # Assert: the measure agrees with the requests check_every sends
    assert (one_fits, both_fit) == (True, False)
    assert judge.check_every([first], [item], shared)
    with pytest.raises(InputBudgetExceededError):
        judge.check_every([first, second], [item], shared)
