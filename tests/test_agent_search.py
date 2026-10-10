"""agent_search over a small real repository with a scripted judge: the requests it sends, how points
stop, and what it returns."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from git_repos import commit_all, write_files

from jev_navigator.directives.agent_search import Bands, agent_search, agent_search_async
from jev_navigator.directives.agent_search_result import AgentSearchResult
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.sources import ScentSource
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

ADMITTED_J1 = json.loads((Path(__file__).parent / "fixtures/j1_admitted_contract.json").read_text())
EXISTENCE_WORDING = json.loads(
    (Path(__file__).parents[1] / "src/jev_navigator/judgments/existence_question.json").read_text()
)["exists"]
MECHANISM = "MECHANISM-MARKER: one helper owns the item limit"
SHOP = {
    "shop/orders.py": (
        "from shop.limits import check_limit\n\n\n"
        "def place_order(order):\n    check_limit(order)\n    return save(order)\n\n\n"
        "def save(order):\n    return order\n"
    ),
    "shop/limits.py": (
        "MAX_ITEMS = 4\n\n\n"
        "def check_limit(order):\n    if count_items(order) > MAX_ITEMS:\n"
        '        raise ValueError("too many items")\n\n\n'
        "def count_items(order):\n    return len(order.items)\n"
    ),
    "shop/admin.py": (
        "from shop.limits import check_limit\n\n\n"
        "def bulk_import(orders):\n    for order in orders:\n        check_limit(order)\n"
    ),
    "tests/test_limits.py": (
        "from shop.limits import MAX_ITEMS, check_limit\n\n\n"
        "def test_check_limit():\n    check_limit(MAX_ITEMS)\n"
    ),
}
LIMIT = "code that refuses an order over the item limit"
REFUND = "code that refunds an order"


def request(**changes) -> dict:
    base = {
        "hypotheses": [
            {
                "id": "h1",
                "mechanism": MECHANISM,
                "evidence": [{"id": "e1", "point": LIMIT}],
                "refuted_by": [{"id": "r1", "point": REFUND}],
            }
        ],
        "terms": [],
        "anchors": [{"file": "shop/orders.py", "line": 5}],
        "files": ["shop/orders.py"],
        "scope": {"include": [], "exclude": [], "with_tests": False},
        "follow": ["callers", "callees"],
        "budget_requests": 8,
    }
    return {**base, **changes}


@pytest.fixture
def shop(tmp_path: Path) -> CodeIndex:
    write_files(tmp_path, SHOP)
    commit_all(tmp_path)
    return CodeIndex.from_git(tmp_path)


def scripted(
    match: Mapping[str, Callable[[str], float]], exists: Mapping[str, Callable[[str], float]]
) -> ScriptedJevClient:
    """J1-3 answers by point key and the unit's code; existence answers by point key and the joined
    code of the shown entries; every role label 0.3."""

    def answer(question_id: str, question: Mapping, state: Mapping) -> float:
        name = question_id.split("@")[0]
        if name.startswith("match_"):
            slot = int(question_id.split("#")[1])
            return match[name.removeprefix("match_")](state["items"][slot]["code"])
        if name.startswith("exists_"):
            return exists[name.removeprefix("exists_")](" ".join(entry["code"] for entry in state["fetched"]))
        return 0.3

    return ScriptedJevClient(nouls=answer)


def limit_match(code: str) -> float:
    if "raise ValueError" in code:
        return 0.92
    return 0.55 if "check_limit(order)" in code else 0.1


def constant(probability: float) -> Callable[[str], float]:
    return lambda code: probability


def limit_exists(code: str) -> float:
    return 0.9 if "raise ValueError" in code else 0.5


def kinds(client: ScriptedJevClient) -> list[str]:
    """Each sent request's kind, by the state it carries."""
    return [
        "ranking"
        if "items" in state and "match_" in next(iter(questions))
        else "existence"
        if "fetched" in state
        else "labels"
        for state, questions in client.requests
    ]


def sent_code(client: ScriptedJevClient) -> str:
    return json.dumps([state for state, _ in client.requests])


def searched(
    client: ScriptedJevClient, asked: dict, index: CodeIndex, *, asynchronous: bool, **options
) -> AgentSearchResult:
    """``agent_search``, or ``agent_search_async`` run to the end behind the async form of ``client``,
    which the Judge's sync methods refuse."""
    if not asynchronous:
        return agent_search(asked, index, Judge(client, masker=None, scanner=None), **options)
    judge = Judge(AsyncScriptedJevClient(client), masker=None, scanner=None)
    return asyncio.run(agent_search_async(asked, index, judge, **options))


def test_every_ranking_request_asks_every_point_with_the_admitted_j1_wording(shop: CodeIndex) -> None:
    # Arrange
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
    )

    # Act
    agent_search(request(), shop, Judge(client, masker=None, scanner=None))

    # Assert: each item slot asks each point the admitted question, only its point path bound
    ranking = [sent for sent, kind in zip(client.requests, kinds(client), strict=True) if kind == "ranking"]
    assert len(ranking) == 2
    for state, questions in ranking:
        assert list(state) == ["targets", "items"]
        assert state["targets"] == {"h1_e1": LIMIT, "h1_r1": REFUND}
        assert all(list(item) == ["file", "code"] for item in state["items"])
        assert 1 <= len(state["items"]) <= 16
        for slot in range(len(state["items"])):
            admitted = ADMITTED_J1["questions"][
                f"{next(iter(ADMITTED_J1['questions'])).split('#')[0]}#{slot}"
            ]
            for key in state["targets"]:
                [sent] = [
                    question
                    for qid, question in questions.items()
                    if qid.startswith(f"match_{key}@") and qid.endswith(f"#{slot}")
                ]
                assert sent == json.loads(json.dumps(admitted).replace("`targets.p0`", f"`targets.{key}`"))


def test_the_mechanism_never_reaches_any_request(shop: CodeIndex) -> None:
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
    )

    result = agent_search(request(), shop, Judge(client, masker=None, scanner=None))

    assert set(kinds(client)) == {"ranking", "existence", "labels"}
    assert all(MECHANISM not in json.dumps([state, questions]) for state, questions in client.requests)
    assert result.hypotheses[0].mechanism == MECHANISM


def test_existence_asks_every_open_point_once_over_the_union_of_their_shortlists(shop: CodeIndex) -> None:
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
    )

    agent_search(request(), shop, Judge(client, masker=None, scanner=None))

    first, second = [
        sent for sent, kind in zip(client.requests, kinds(client), strict=True) if kind == "existence"
    ]
    state, questions = first
    assert list(state) == ["targets", "fetched"]
    assert state["targets"] == {"h1_e1": LIMIT, "h1_r1": REFUND}
    assert [list(entry) for entry in state["fetched"]] == [["file", "code"]] * 2
    assert [question for question in questions.values()] == [
        json.loads(json.dumps(EXISTENCE_WORDING).replace("`point`", f"`targets.{key}`"))
        for key in ("h1_e1", "h1_r1")
    ]
    # the refuting point was low and had nothing left, so only the still open point is asked again
    assert second[0]["targets"] == {"h1_e1": LIMIT}
    assert len(second[0]["fetched"]) == 3


def test_only_open_points_whose_shortlist_changed_are_asked_again(shop: CodeIndex) -> None:
    # Arrange: one unit per request; e2's best unit is judged first and stays its whole shortlist
    [base] = request()["hypotheses"]
    two = {**base, "evidence": [{"id": "e1", "point": LIMIT}, {"id": "e2", "point": "code that places"}]}
    client = scripted(
        {
            "h1_e1": limit_match,
            "h1_e2": lambda code: 0.9 if "def place_order" in code else 0.1,
            "h1_r1": constant(0.05),
        },
        {"h1_e1": limit_exists, "h1_e2": constant(0.2), "h1_r1": constant(0.1)},
    )
    judge = Judge(client, masker=None, scanner=None, items_per_request=1)

    # Act
    agent_search(request(hypotheses=[{**two, "refuted_by": []}]), shop, judge, beam_width=1)

    # Assert
    asked = [sorted(state["targets"]) for state, _ in client.requests if "fetched" in state]
    assert asked[0] == ["h1_e1", "h1_e2"]
    assert asked[1:] and all(targets == ["h1_e1"] for targets in asked[1:])


def test_a_middle_point_expands_its_shortlist_and_is_established_once_the_shortlist_holds_it(
    shop: CodeIndex,
) -> None:
    # Arrange: the anchored place_order only calls the check (middle); its callee check_limit holds it
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
    )

    # Act
    result = agent_search(request(), shop, Judge(client, masker=None, scanner=None))

    # Assert
    limit, refund = result.hypotheses[0].points
    assert (limit.outcome, limit.stopped_by, limit.band) == ("established", "established", "high")
    assert limit.existence.probability == 0.9
    assert [place.symbol for place in limit.shortlist] == ["check_limit", "place_order", "save"]
    assert (limit.definite_files, limit.possible_files) == (("shop/limits.py",), ("shop/orders.py",))
    assert "def check_limit" not in json.dumps(client.requests[0][0])
    assert "def check_limit" in json.dumps(client.requests[2][0])
    assert "bulk_import" not in sent_code(client), "an established point's shortlist is never expanded"
    assert (refund.outcome, refund.stopped_by, refund.band) == (
        "not_found_in_scope",
        "frontier_exhausted",
        "low",
    )
    assert {place.symbol for place in refund.shortlist} == {"place_order", "save", "check_limit"}, (
        "every judged unit is asked every point, and the shortlist stays as where to look"
    )
    assert result.stopped_by == "points_settled"
    assert kinds(client) == ["ranking", "existence", "ranking", "existence", "labels"]
    assert result.requests.used == len(client.requests) == 5


def test_a_high_point_in_the_first_round_stops_without_expanding(shop: CodeIndex) -> None:
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": constant(0.9), "h1_r1": constant(0.1)}
    )

    result = agent_search(request(), shop, Judge(client, masker=None, scanner=None))

    assert result.point("h1.e1").outcome == "established"
    assert "def check_limit" not in sent_code(client)
    assert kinds(client)[:2] == ["ranking", "existence"] and kinds(client).count("ranking") == 1


def test_a_shortlist_unchanged_by_its_expansion_stops_its_point(shop: CodeIndex) -> None:
    # Arrange: check_limit, the only new hop, ranks below both shortlisted units
    scores = {"place_order": 0.55, "save": 0.3}

    def match(code: str) -> float:
        return next((p for name, p in scores.items() if f"def {name}" in code), 0.2)

    client = scripted(
        {"h1_e1": match, "h1_r1": constant(0.05)}, {"h1_e1": constant(0.5), "h1_r1": constant(0.1)}
    )

    # Act
    result = agent_search(request(), shop, Judge(client, masker=None, scanner=None), beam_width=2)

    # Assert
    limit = result.point("h1.e1")
    assert (limit.outcome, limit.stopped_by, limit.band) == ("undecided", "stable_beam", "middle")
    assert [place.symbol for place in limit.shortlist] == ["place_order", "save"]
    assert [place.symbol for place in limit.next] == ["check_limit"]
    assert kinds(client).count("existence") == 1, "an unchanged shortlist is not asked again"


def test_a_refuting_match_is_a_conflict_listed_before_the_hypotheses(shop: CodeIndex) -> None:
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": lambda code: 0.85 if "def save" in code else 0.05},
        {"h1_e1": limit_exists, "h1_r1": constant(0.8)},
    )

    result = agent_search(request(), shop, Judge(client, masker=None, scanner=None))

    [conflict] = result.conflicts
    assert (conflict.point, conflict.place.symbol, conflict.place.probability) == ("h1.r1", "save", 0.85)
    answer = result.to_json()
    assert list(answer).index("conflicts") < list(answer).index("hypotheses")
    assert answer["conflicts"][0]["place"] == "shop/orders.py:9-10"
    assert result.point("h1.r1").outcome == "established"


def test_the_budget_counts_role_labels_and_names_what_it_left_unlabelled(shop: CodeIndex) -> None:
    # Arrange: two matched points, three requests, one kept for labels
    [base] = request()["hypotheses"]
    two_points = {
        **base,
        "evidence": [{"id": "e1", "point": LIMIT}, {"id": "e2", "point": "code that places an order"}],
    }
    client = scripted(
        {"h1_e1": limit_match, "h1_e2": limit_match, "h1_r1": constant(0.05)},
        {"h1_e1": limit_exists, "h1_e2": limit_exists, "h1_r1": constant(0.1)},
    )

    # Act
    result = agent_search(
        request(hypotheses=[two_points], budget_requests=3),
        shop,
        Judge(client, masker=None, scanner=None),
        label_requests=1,
    )

    # Assert
    assert result.stopped_by == "budget"
    assert kinds(client) == ["ranking", "existence", "labels"]
    assert (
        result.requests.used,
        result.requests.ranking,
        result.requests.existence,
        result.requests.labels,
    ) == (3, 1, 1, 1)
    assert result.point("h1.e1").labels == "labelled"
    assert result.point("h1.e1").shortlist[0].roles == dict.fromkeys(
        ("decide", "guard", "value", "effect", "delegates", "satisfied"), 0.3
    )
    assert result.point("h1.e2").labels == "not labelled: budget"
    assert result.point("h1.e2").outcome == "undecided" and result.point("h1.e2").stopped_by == "budget"


def test_code_travels_for_the_best_places_first_within_the_allowance(shop: CodeIndex) -> None:
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
    )
    check_limit = SHOP["shop/limits.py"].split("\n\n\n")[1]

    result = agent_search(
        request(), shop, Judge(client, masker=None, scanner=None), max_code_chars=len(check_limit)
    )

    code = result.to_json()["code"]
    assert code["shop/limits.py:4-6"] == {"code": check_limit}
    omitted = [entry for place, entry in code.items() if place != "shop/limits.py:4-6"]
    assert omitted and all(entry["code"] is None and entry["omitted"] for entry in omitted)
    assert all("code" not in place for place in result.to_json()["hypotheses"][0]["points"][0]["shortlist"])


def test_scope_leaves_tests_out_and_names_files_it_left_out(shop: CodeIndex) -> None:
    client = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": constant(0.5), "h1_r1": constant(0.5)}
    )
    with_tests = request(terms=["MAX_ITEMS"], files=["shop/orders.py", "tests/test_limits.py"], follow=[])

    left_out = agent_search(with_tests, shop, Judge(client, masker=None, scanner=None))
    kept = agent_search(
        {**with_tests, "scope": {"include": [], "exclude": [], "with_tests": True}},
        shop,
        Judge(
            scripted(
                {"h1_e1": limit_match, "h1_r1": constant(0.05)},
                {"h1_e1": constant(0.5), "h1_r1": constant(0.5)},
            ),
            masker=None,
            scanner=None,
        ),
    )

    assert "def test_check_limit" not in sent_code(client)
    assert left_out.coverage.outside_scope == ("tests/test_limits.py",)
    assert "test_check_limit" in {
        place.symbol for place in kept.point("h1.e1").shortlist + kept.point("h1.e1").next
    }


def test_likely_units_pull_their_graph_neighbours_ahead_in_the_queue(tmp_path: Path) -> None:
    # Arrange: two queued units tie by code value; only count_items is linked to check_limit
    write_files(
        tmp_path,
        {
            "shop/zzz.py": (
                "from shop.aaa import helper\n\n\n"
                "def check_limit(order):\n    if count_items(order) > 4:\n        raise ValueError()\n\n\n"
                "def count_items(order):\n    return len(order.items)\n"
            ),
            "shop/aaa.py": "def helper(order):\n    return order\n",
        },
    )
    commit_all(tmp_path)
    index = CodeIndex.from_git(tmp_path)
    one_by_one = request(
        anchors=[{"file": "shop/zzz.py", "line": 4}], files=["shop/aaa.py", "shop/zzz.py"], follow=[]
    )

    def second_judged(check_limit: float) -> str:
        client = scripted(
            {"h1_e1": lambda code: check_limit if "raise" in code else 0.1, "h1_r1": constant(0.05)},
            {"h1_e1": constant(0.5), "h1_r1": constant(0.5)},
        )
        agent_search(
            one_by_one, index, Judge(client, masker=None, scanner=None, items_per_request=1), beam_width=1
        )
        ranking = [
            state
            for (state, _), kind in zip(client.requests, kinds(client), strict=True)
            if kind == "ranking"
        ]
        return ranking[1]["items"][0]["code"].splitlines()[0]

    # Act and Assert
    assert second_judged(check_limit=0.1) == "def helper(order):"
    assert second_judged(check_limit=0.9) == "def count_items(order):"


def test_bands_refuse_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="low < high"):
        Bands(high=0.3, low=0.6)


@pytest.mark.parametrize("budget", [8, 3])
def test_the_async_form_sends_the_same_requests_and_returns_the_same_result(
    shop: CodeIndex, budget: int
) -> None:
    # Arrange: the same answers behind a sync client and an async one
    def client() -> ScriptedJevClient:
        return scripted(
            {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
        )

    sync_client, async_client = client(), client()

    # Act
    expected = searched(sync_client, request(budget_requests=budget), shop, asynchronous=False)
    actual = searched(async_client, request(budget_requests=budget), shop, asynchronous=True)

    # Assert
    assert len(sync_client.requests) == {8: 5, 3: 2}[budget], "budget 3 keeps 2 for labels and uses 1"
    assert async_client.requests == sync_client.requests
    assert actual.to_json() == expected.to_json()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_failed_request_ends_the_search_failed_and_keeps_the_answers_before_it(
    shop: CodeIndex, asynchronous: bool
) -> None:
    def down(code: str) -> float:
        raise ConnectionError("Jev is down")

    client = scripted({"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": down, "h1_r1": down})

    result = searched(client, request(), shop, asynchronous=asynchronous)

    assert (result.stopped_by, repr(result.failure)) == ("failed", "ConnectionError('Jev is down')")
    assert result.point("h1.e1").shortlist[0].symbol == "place_order"
    assert result.point("h1.e1").labels == "labelled"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_failed_labelling_request_is_named_and_the_result_still_returns(
    shop: CodeIndex, asynchronous: bool
) -> None:
    # Arrange: every role-label question fails; ranking and existence answer
    labelling = scripted(
        {"h1_e1": limit_match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
    )

    def answer(question_id: str, question: Mapping, state: Mapping) -> float:
        if question_id.startswith(("match_", "exists_")):
            return labelling.nouls(question_id, question, state)
        raise ConnectionError("Jev is down")

    # Act
    result = searched(ScriptedJevClient(nouls=answer), request(), shop, asynchronous=asynchronous)

    # Assert
    assert result.stopped_by == "points_settled" and result.failure is None
    assert result.point("h1.e1").outcome == "established"
    assert result.point("h1.e1").labels == "not labelled: the request failed: ConnectionError: Jev is down"
    assert result.requests.labels == 1


def test_cancelling_the_async_search_raises_instead_of_returning(shop: CodeIndex) -> None:
    class Hanging(AsyncScriptedJevClient):
        async def send(self, state: Mapping, questions: Mapping):
            sending.set()
            await asyncio.Event().wait()

    async def cancelled_while_judging() -> None:
        task = asyncio.create_task(agent_search_async(request(), shop, Judge(Hanging())))
        await sending.wait()
        task.cancel()
        await task

    sending = asyncio.Event()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled_while_judging())


QUOTA = {
    "shop/quota.py": (
        "def refuse_oversized_order(order, item_limit):\n"
        "    if order.item_total > item_limit:\n        raise OrderRefused(order)\n"
    )
}


@pytest.fixture
def shop_with_quota(tmp_path: Path) -> CodeIndex:
    write_files(tmp_path, {**SHOP, **QUOTA})
    commit_all(tmp_path)
    return CodeIndex.from_git(tmp_path)


def quota_client() -> ScriptedJevClient:
    def match(code: str) -> float:
        if "raise" in code:
            return 0.92
        return 0.55 if "check_limit(order)" in code else 0.1

    return scripted(
        {"h1_e1": match, "h1_r1": constant(0.05)}, {"h1_e1": limit_exists, "h1_r1": constant(0.1)}
    )


def test_the_scent_source_is_off_by_default_and_reaches_what_no_other_source_does_when_passed(
    shop_with_quota: CodeIndex,
) -> None:
    # Arrange: nothing names, imports or calls the quota check
    default, scented = quota_client(), quota_client()

    # Act
    without = agent_search(request(), shop_with_quota, Judge(default, masker=None, scanner=None))
    with_scent = agent_search(
        request(), shop_with_quota, Judge(scented, masker=None, scanner=None), extra_sources=[ScentSource()]
    )

    # Assert
    assert "refuse_oversized_order" not in sent_code(default)
    assert "shop/quota.py" not in without.point("h1.e1").definite_files
    assert "refuse_oversized_order" in sent_code(scented)
    assert with_scent.point("h1.e1").definite_files == ("shop/limits.py", "shop/quota.py")
    assert with_scent.requests.used <= 8 and all(
        len(state["items"]) <= 16 for state, _ in scented.requests if "items" in state
    )


def test_scent_units_left_unjudged_are_counted_by_their_source_within_the_bounds(
    shop_with_quota: CodeIndex,
) -> None:
    # Arrange: two units per request and one ranking request
    client = quota_client()
    judge = Judge(client, masker=None, scanner=None, items_per_request=2)

    # Act
    result = agent_search(
        request(budget_requests=3), shop_with_quota, judge, extra_sources=[ScentSource()], max_code_chars=200
    )

    # Assert
    assert result.coverage.not_judged.get("not reached: from scent", 0) >= 1
    assert result.requests.used <= 3
    assert all(len(state["items"]) <= 2 for state, _ in client.requests if "items" in state)
    assert sum(len(code) for code in result.code.values() if code is not None) <= 200
