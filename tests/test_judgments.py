from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import BudgetedClient

from jev_navigator.judgments.answers import ChoiceAnswer, JevResponse, NoulAnswer
from jev_navigator.judgments.client import (
    JEV_INPUT_BOX_CHARS,
    MAX_REQUEST_CHARS,
    InputBudgetExceededError,
    MissingAnswerError,
    ReplayOnlyClient,
)
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import (
    CallCapReachedError,
    CallOffer,
    Judge,
    request_exceeds_input_budget,
)
from jev_navigator.judgments.questions import Check, Criterion, Pick
from jev_navigator.judgments.secrets import SecretInRequestError, SecretMasker
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.thresholds import NoulVerdict, Thresholds
from jev_navigator.testing import ScriptedJevClient

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
    client = ScriptedJevClient(nouls={"describes#0": 0.95, "describes#1": 0.5, "describes#2": 0.05})
    judge = Judge(client)
    items = [{"code": "def a(): ..."}, {"code": "def b(): ..."}, {"code": "def c(): ..."}]

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


def test_items_judged_before_are_answered_from_the_store(tmp_path: Path) -> None:
    # Arrange
    store = JsonlAnswerStore(tmp_path / "answers.jsonl")
    first_client = ScriptedJevClient(nouls={"describes": 0.9})
    Judge(first_client, store=store).check_each(DESCRIBES, [{"code": "x = 1"}], {"doc": {"sentence": "s"}})
    second_client = ScriptedJevClient(nouls={"describes": 0.1})
    second = Judge(
        second_client, store=JsonlAnswerStore(tmp_path / "answers.jsonl"), served_model="jev-scripted"
    )

    # Act
    results = second.check_each(DESCRIBES, [{"code": "x = 1"}, {"code": "y = 2"}], {"doc": {"sentence": "s"}})

    # Assert
    assert [(result.probability, result.from_store) for result in results] == [(0.9, True), (0.1, False)]
    assert len(second_client.requests) == 1


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
        DESCRIBES, items[:1], shared
    )
    client = ScriptedJevClient(nouls={"describes": 0.1, "changes#0": 0.5, "changes#1": 0.95})
    judge = Judge(client, store=JsonlAnswerStore(path), served_model="jev-scripted")

    results = judge.check_every([DESCRIBES, changes], items, shared)

    assert len(client.requests) == 1
    state, questions = client.requests[0]
    assert state["items"] == items
    assert set(questions) == {
        f"{DESCRIBES.question_id}#1",
        f"{changes.question_id}#0",
        f"{changes.question_id}#1",
    }
    assert [(r.item, r.probability, r.from_store) for r in results["describes"]] == [
        (items[0], 0.9, True),
        (items[1], 0.1, False),
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
    assert judge.input_total.complete_total() is None


def test_a_total_with_every_response_reported_is_complete() -> None:
    judge = Judge(_UsageClient(10, 0))

    for index in range(2):
        judge.ask({"s": index}, {"q": {"type": "noul"}}, thresholds=Thresholds())

    assert judge.input_total.complete_total() == 10


def test_a_scoped_judge_adds_unreported_responses_to_its_parent() -> None:
    judge = Judge(_UsageClient(None))
    scoped = judge.scope()

    scoped.ask({"s": 1}, {"q": {"type": "noul"}}, thresholds=Thresholds())

    assert (scoped.input_total.not_reported, judge.input_total.not_reported) == (1, 1)


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


def test_oversized_batch_is_split_before_sending_so_no_request_exceeds_the_input_budget() -> None:
    client = BudgetedClient(MAX_REQUEST_CHARS)
    judge = Judge(client)
    items = [_padding_item(f"part{index}", 28_000) for index in range(4)]

    results = judge.check_every(
        [DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts", batch_budget=200_000
    )

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 4
    assert client.refusals == 0, "a request the measurement already rejects must not be paid for"
    assert len(client.requests) == 2
    judged_files: list[str] = []
    for state, questions in client.requests:
        body = len(json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode())
        assert body <= MAX_REQUEST_CHARS
        assert len(questions) == len(state["parts"]), "one atomic question per item and slot"
        judged_files.extend(item["file"] for item in state["parts"])
    assert sorted(judged_files) == [f"part{index}.py" for index in range(4)]


def _boxed_client() -> BudgetedClient:
    return BudgetedClient(MAX_REQUEST_CHARS, input_box=JEV_INPUT_BOX_CHARS)


def _judge_padded_parts(client: BudgetedClient, count: int, chars: int):
    items = [_padding_item(f"part{index}", chars) for index in range(count)]
    return Judge(client).check_every(
        [DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts", batch_budget=500_000
    )


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

    assert not request_exceeds_input_budget(state, questions)


def test_the_longest_question_counts_towards_the_character_box() -> None:
    state = {"parts": [{"code": "y" * 60_000}]}
    questions = {"short": {"ask": "x"}, "long": {"ask": "x" * 17_000}}

    assert request_exceeds_input_budget(state, questions)
    assert not request_exceeds_input_budget(state, {"short": questions["short"]})


def test_a_body_over_the_request_box_is_over_budget_although_state_and_question_fit() -> None:
    state = {"doc": {"sentence": "s"}}
    questions = {f"q{index}": {"ask": "x" * 40} for index in range(5_000)}

    assert request_exceeds_input_budget(state, questions)
    assert not request_exceeds_input_budget(state, dict(list(questions.items())[:100]))


def test_provider_max_tokens_error_splits_the_batch_and_keeps_every_question_identity() -> None:
    client = BudgetedClient(34_000)  # stricter than the measured packing budget
    judge = Judge(client)
    items = [_padding_item(f"part{index}", 12_000) for index in range(4)]

    results = judge.check_every([DESCRIBES], items, {"doc": {"sentence": "s"}}, list_name="parts")

    assert [result.verdict for result in results["describes"]] == [NoulVerdict.YES] * 4
    assert client.refusals == 1, "the first over-budget request is the provider's own evidence"
    assert len(client.requests) == 2
    for state, questions in client.requests:
        body = len(json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode())
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
    assert "max_tokens_exceeded" in failures[0]["error"]


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
    client = BudgetedClient(MAX_REQUEST_CHARS)
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
