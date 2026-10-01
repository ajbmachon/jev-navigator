"""The route table: named decision-model routes with automatic fallback, each served by its
registered adapter or by the generic System-One client."""

from __future__ import annotations

import json
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from jev_navigator.adapters.local import LocalModelClient
from jev_navigator.adapters.routes import Route, RoutedJevClient, client_from_env, routes_from_env
from jev_navigator.adapters.system_one import AdapterError, HttpRequest, HttpResponse, SystemOneClient
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.thresholds import NoulVerdict, Thresholds


def test_no_routes_variable_resolves_to_no_routes():
    assert routes_from_env({}) == ()


def test_unknown_route_needs_endpoint_and_model():
    with pytest.raises(ValueError, match="route 'decider' is incomplete"):
        routes_from_env({"SYSTEM_ONE_ROUTES": "decider"})


def test_a_route_whose_endpoint_no_request_can_reach_fails_at_startup_naming_the_route():
    # Not on the first question, where routing would take the failure for an outage and fail over.
    with pytest.raises(ValueError, match=r"route 'mine'.*needs http:// or https://"):
        routes_from_env(
            {
                "SYSTEM_ONE_ROUTES": "mine",
                "SYSTEM_ONE_MINE_ENDPOINT": "gpu-box:8000",
                "SYSTEM_ONE_MINE_MODEL": "decider-1",
            }
        )


def test_per_route_settings_beat_the_adapter_defaults():
    routes = routes_from_env(
        {"SYSTEM_ONE_ROUTES": "drex", "SYSTEM_ONE_DREX_MODEL": "drex-v1.1", "DREX_API_KEY": "nace_sk_route"}
    )

    assert routes[0].client.model == "drex-v1.1"


def test_a_route_never_takes_another_services_key():
    with pytest.raises(
        ValueError, match="route 'drex' has no API key: set SYSTEM_ONE_DREX_API_KEY or DREX_API_KEY"
    ):
        routes_from_env({"SYSTEM_ONE_ROUTES": "drex", "TYPESAFE_API_KEY": "tsk-issued-for-jev"})


def test_a_self_hosted_route_runs_without_a_key_and_sends_none():
    sent: list[HttpRequest] = []

    def server(request):
        sent.append(request)
        return HttpResponse(200, {}, b'{"model": "mine-2026-09-30", "answers": {}}')

    routes = routes_from_env(
        {
            "SYSTEM_ONE_ROUTES": "mine",
            "SYSTEM_ONE_MINE_ENDPOINT": "http://gpu-box:8000",
            "SYSTEM_ONE_MINE_MODEL": "mine-2026-09-30",
        },
        transport=server,
    )

    routes[0].client.ask({}, {})

    assert sent[0].url == "http://gpu-box:8000/v1/systemone"
    assert "Authorization" not in sent[0].headers


def test_a_route_runs_the_adapter_class_its_adapter_setting_names():
    routes = routes_from_env(
        {
            "SYSTEM_ONE_ROUTES": "mine",
            "SYSTEM_ONE_MINE_ADAPTER": "jev_navigator.adapters.drex:DrexClient",
            "DREX_API_KEY": "nace_sk_route",
        }
    )

    assert (routes[0].name, type(routes[0].client).__name__) == ("mine", "DrexClient")


@pytest.mark.parametrize(
    ("path", "refusal"),
    [
        ("jev_navigator.adapters.drex", "must name the adapter as module:Class"),
        ("jev_navigator.adapters.nope:Model", "cannot be loaded"),
        ("json:loads", "is not an adapter class"),
        ("collections:OrderedDict", "is not an adapter class: it lacks name, endpoint"),
    ],
)
def test_an_adapter_setting_that_names_no_adapter_class_is_refused(path, refusal):
    with pytest.raises(ValueError, match=f"SYSTEM_ONE_MINE_ADAPTER.*{refusal}"):
        routes_from_env({"SYSTEM_ONE_ROUTES": "mine", "SYSTEM_ONE_MINE_ADAPTER": path})


def test_a_route_asks_with_its_own_timeout_and_retries():
    sent: list[HttpRequest] = []

    def busy(request):
        sent.append(request)
        return HttpResponse(429, {"retry-after-ms": "1"}, b"{}")

    routes = routes_from_env(
        {
            "SYSTEM_ONE_ROUTES": "decider",
            "SYSTEM_ONE_DECIDER_ENDPOINT": "http://gpu-box:8000",
            "SYSTEM_ONE_DECIDER_MODEL": "decider-4b",
            "SYSTEM_ONE_DECIDER_TIMEOUT": "2.5",
            "SYSTEM_ONE_DECIDER_RETRIES": "0",
        },
        transport=busy,
    )

    with pytest.raises(AdapterError, match="HTTP 429"):
        routes[0].client.ask({}, {})
    assert [request.timeout for request in sent] == [2.5]


@pytest.mark.parametrize(
    ("setting", "value"),
    [("TIMEOUT", "0"), ("TIMEOUT", "inf"), ("TIMEOUT", "soon"), ("RETRIES", "-1")],
)
def test_a_route_refuses_a_timeout_or_retry_count_it_cannot_use(setting, value):
    with pytest.raises(ValueError, match=f"SYSTEM_ONE_DREX_{setting} must be"):
        routes_from_env(
            {
                "SYSTEM_ONE_ROUTES": "drex",
                "DREX_API_KEY": "nace_sk_route",
                f"SYSTEM_ONE_DREX_{setting}": value,
            }
        )


def test_the_drex_route_runs_the_drex_adapter_with_the_route_key():
    sent: list[HttpRequest] = []

    def drex(request):
        sent.append(request)
        answers = {id_: {"type": "noul", "noul": 0.9} for id_ in json.loads(request.body)["questions"]}
        return HttpResponse(200, {}, json.dumps({"model": "drex-v1.5", "answers": answers}).encode())

    routes = routes_from_env(
        {
            "SYSTEM_ONE_ROUTES": "drex",
            "SYSTEM_ONE_DREX_API_KEY": "nace_sk_route",
            "DREX_API_KEY": "nace_sk_general",
        },
        transport=drex,
    )
    question = {
        "type": "noul",
        "instructions": "y?",
        "criteria": {"true": {"what": "yes"}, "false": {"what": "no"}},
    }

    routes[0].client.ask({"case": "x"}, {"adds_one": question})

    assert sent[0].url == "https://drex.nace.ai/v1/systemone"
    assert sent[0].headers["Authorization"] == "Bearer nace_sk_route"
    assert json.loads(sent[0].body)["questions"]["adds_one"]["criteria"] == {
        "true": '{"what": "yes"}',
        "false": '{"what": "no"}',
    }


def test_the_first_route_answers_and_the_second_never_runs():
    exchanges: list[tuple[bytes, bytes]] = []
    primary = _server(exchanges)
    backup = _server([])
    routed = RoutedJevClient((Route("primary", primary), Route("backup", backup)))

    answer = routed.ask({"case": "x"}, {"adds_one": {"type": "noul", "instructions": "y?"}})

    assert answer.answers["adds_one"].probability == 0.9
    assert len(exchanges) == 1


def test_failover_asks_the_next_route_after_a_failure():
    dead = _dead_server()
    exchanges: list[tuple[bytes, bytes]] = []
    backup = _server(exchanges)
    routed = RoutedJevClient((Route("dead", dead), Route("backup", backup)))

    answer = routed.ask({"case": "x"}, {"adds_one": {"type": "noul", "instructions": "y?"}})

    assert answer.answers["adds_one"].probability == 0.9
    assert len(exchanges) == 1


def test_all_routes_failing_names_every_route():
    routed = RoutedJevClient((Route("a", _dead_server()), Route("b", _dead_server())))

    with pytest.raises(ConnectionError, match="a.*b"):
        routed.ask({}, {})


def test_routed_client_without_routes_is_refused():
    with pytest.raises(ValueError, match="at least one route"):
        RoutedJevClient(())


def test_a_route_client_hits_its_own_endpoint_not_the_default():
    exchanges: list[tuple[bytes, bytes]] = []
    client = _server(exchanges, model="finetuned-4b")

    answer = client.ask({"case": "x"}, {"adds_one": {"type": "noul", "instructions": "y?"}})

    assert answer.model == "finetuned-4b"
    assert len(exchanges) == 1


CHECK = {"type": "noul", "instructions": "Does `code` add one?"}
PICK = {"type": "choice", "instructions": "What does `code` do?", "criteria": {"add": None, "sub": None}}
RATE = {"type": "score", "instructions": "How well?", "criteria": ["not at all", "exactly"]}
SPLIT_ROUTES = {
    "SYSTEM_ONE_ROUTES": "decider",
    "SYSTEM_ONE_DECIDER_ENDPOINT": "http://decider:8000",
    "SYSTEM_ONE_DECIDER_MODEL": "decider-1",
    "SYSTEM_ONE_ROUTES_PICK": "picker",
    "SYSTEM_ONE_PICKER_ENDPOINT": "http://picker:8000",
    "SYSTEM_ONE_PICKER_MODEL": "picker-2",
}


def test_a_mixed_request_is_split_by_question_type_and_merged_into_one_response():
    received: dict[str, list[list[str]]] = {}
    routed = client_from_env(SPLIT_ROUTES, transport=_services(received))

    raw = routed.send({"code": "x + 1"}, {"adds_one": CHECK, "kind": PICK, "fit": RATE})
    answer = routed.parse(raw)

    assert received == {"decider": [["adds_one", "fit"]], "picker": [["kind"]]}
    assert list(answer.answers) == ["adds_one", "kind", "fit"]
    assert answer.choice("kind").choice == "add"
    assert (answer.model, answer.input_tokens) == ("check=decider-1,pick=picker-2,rate=decider-1", 14)
    assert routed.model == "check=decider-1,pick=picker-2,rate=decider-1"
    assert raw.exact is False


def test_a_request_one_chain_answers_is_sent_whole_with_its_exact_bytes():
    received: dict[str, list[list[str]]] = {}
    routed = client_from_env(SPLIT_ROUTES, transport=_services(received))

    raw = routed.send({"code": "x + 1"}, {"adds_one": CHECK, "fit": RATE})

    assert received == {"decider": [["adds_one", "fit"]]}
    assert raw.exact is True
    assert json.loads(raw.body)["model"] == "decider-1"
    assert routed.parse(raw).model == "decider-1"


def test_a_split_request_fails_when_one_types_routes_all_fail():
    routes = {
        **SPLIT_ROUTES,
        "SYSTEM_ONE_PICKER_ENDPOINT": "http://127.0.0.1:1",
        "SYSTEM_ONE_PICKER_RETRIES": "0",
    }
    routed = client_from_env(routes, transport=_services({}, dead={"picker"}))

    with pytest.raises(ConnectionError, match="every route for pick questions failed: picker"):
        routed.ask({"code": "x + 1"}, {"adds_one": CHECK, "kind": PICK})


def test_a_route_named_in_two_tables_is_one_adapter():
    _CountingModel.loads.clear()
    routed = client_from_env(
        {
            "SYSTEM_ONE_ROUTES_CHECK": "toy",
            "SYSTEM_ONE_ROUTES_PICK": "toy,backup",
            "SYSTEM_ONE_ROUTES_RATE": "toy",
            "SYSTEM_ONE_TOY_ADAPTER": f"{__name__}:_CountingModel",
            "SYSTEM_ONE_BACKUP_ENDPOINT": "http://backup:8000",
            "SYSTEM_ONE_BACKUP_MODEL": "backup-1",
        }
    )

    try:
        answer = routed.ask({"code": "x + 1"}, {"adds_one": CHECK, "kind": PICK})
    finally:
        routed.close()

    assert answer.model == "toy@1"
    assert _CountingModel.loads == ["load"]


def test_cancel_stops_a_split_request_waiting_on_any_route():
    # Arrange: the pick part answers at once; the check part waits on an in-process model.
    _HeldModel.release.clear()
    routed = client_from_env(
        {
            **SPLIT_ROUTES,
            "SYSTEM_ONE_ROUTES_CHECK": "held",
            "SYSTEM_ONE_HELD_ADAPTER": f"{__name__}:_HeldModel",
        },
        transport=_services({}),
    )
    release = threading.Timer(3, _HeldModel.release.set)
    release.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            asked = pool.submit(routed.ask, {"code": "x + 1"}, {"kind": PICK, "adds_one": CHECK})
            assert _HeldModel.answering.wait(2)

            # Act
            routed.cancel()

            # Assert
            with pytest.raises(CancelledError):
                asked.result(timeout=1)
    finally:
        release.cancel()
        _HeldModel.release.set()
        routed.close()


def test_a_routes_own_bars_calibrate_its_answers_while_the_journal_keeps_them_as_sent():
    routed = client_from_env(
        {
            "SYSTEM_ONE_ROUTES": "decider",
            **_decider_settings(),
            "SYSTEM_ONE_DECIDER_NOUL_YES_AT": "0.9",
            "SYSTEM_ONE_DECIDER_CHOICE_MIN_CONFIDENCE": "0.8",
        },
        transport=_services({}, noul=0.88, confidence=0.75),
    )

    raw = routed.send({"code": "x + 1"}, {"adds_one": CHECK, "kind": PICK})
    answer = routed.parse(raw)

    sent = json.loads(raw.body)["answers"]
    assert (raw.exact, sent["adds_one"]["noul"], sent["kind"]["confidence"]) == (True, 0.88, 0.75)
    assert Thresholds().noul_verdict(answer.noul("adds_one").probability) == NoulVerdict.UNSURE
    assert not Thresholds().choice_is_confident(answer.choice("kind").confidence)
    assert routed.route_thresholds == {
        "decider": {"choice_min_confidence": 0.8, "noul_yes_at": 0.9, "noul_no_at": 0.2}
    }


def test_a_split_request_maps_only_the_barred_routes_answers_and_journals_them_as_sent():
    routed = client_from_env(
        {**SPLIT_ROUTES, "SYSTEM_ONE_PICKER_CHOICE_MIN_CONFIDENCE": "0.9"},
        transport=_services({}, confidence=0.85),
    )

    raw = routed.send({"code": "x + 1"}, {"adds_one": CHECK, "kind": PICK})

    assert json.loads(raw.body)["answers"]["kind"]["confidence"] == 0.85
    assert routed.parse(raw).choice("kind").confidence == pytest.approx(0.85 * 0.7 / 0.9)
    assert routed.parse(raw).noul("adds_one").probability == 0.9


@pytest.mark.parametrize(
    ("setting", "value", "refusal"),
    [
        ("NOUL_YES_AT", "1.5", "SYSTEM_ONE_DECIDER_NOUL_YES_AT must be a number from 0 to 1"),
        ("CHOICE_MIN_CONFIDENCE", "high", "SYSTEM_ONE_DECIDER_CHOICE_MIN_CONFIDENCE must be a number"),
        ("NOUL_NO_AT", "0.85", "route 'decider' thresholds: noul_no_at .* must be below noul_yes_at"),
        ("NOUL_YES_AT", "1", "route 'decider' thresholds: a bar at 0 or 1 cannot move"),
    ],
)
def test_a_route_refuses_bars_it_cannot_use(setting, value, refusal):
    with pytest.raises(ValueError, match=refusal):
        client_from_env(
            {"SYSTEM_ONE_ROUTES": "decider", **_decider_settings(), f"SYSTEM_ONE_DECIDER_{setting}": value}
        )


def test_question_types_without_a_table_go_to_jev_when_no_default_is_named():
    with pytest.raises(ValueError, match="route 'jev' has no API key"):
        client_from_env({"SYSTEM_ONE_ROUTES_CHECK": "decider", **_decider_settings()})


def test_a_misspelt_question_table_is_refused():
    with pytest.raises(ValueError, match="SYSTEM_ONE_ROUTES_CHEK names no question type"):
        client_from_env({"SYSTEM_ONE_ROUTES_CHEK": "decider", **_decider_settings()})


class _CountingModel(LocalModelClient):
    name = "toy"
    default_model = "toy@1"
    loads: list[str] = []

    def load(self) -> None:
        self.loads.append("load")

    def answer(self, state, questions):
        return {"answers": _answers(questions)}


class _HeldModel(LocalModelClient):
    name = "held"
    default_model = "held@1"
    answering = threading.Event()
    release = threading.Event()

    def answer(self, state, questions):
        self.answering.set()
        self.release.wait()
        return {"answers": _answers(questions)}


def _decider_settings() -> dict[str, str]:
    return {"SYSTEM_ONE_DECIDER_ENDPOINT": "http://decider:8000", "SYSTEM_ONE_DECIDER_MODEL": "decider-1"}


def _services(
    received: dict[str, list[list[str]]],
    dead: set[str] = frozenset(),
    noul: float = 0.9,
    confidence: float = 1.0,
):
    """A transport for several System-One hosts: each answers with the model it was asked for,
    ``noul`` for every check and ``confidence`` for every pick, and records the question ids of
    every request it receives."""

    def transport(request: HttpRequest) -> HttpResponse:
        host = urlsplit(request.url).hostname
        if host in dead or urlsplit(request.url).port == 1:
            raise ConnectionRefusedError(f"{host} is down")
        questions = json.loads(request.body)["questions"]
        received.setdefault(host, []).append(list(questions))
        model = json.loads(request.body)["model"]
        body = {
            "model": model,
            "usage": {"input_tokens": 7},
            "answers": _answers(questions, noul, confidence),
        }
        return HttpResponse(200, {"content-type": "application/json"}, json.dumps(body).encode())

    return transport


def _answers(questions, noul: float = 0.9, confidence: float = 1.0) -> dict:
    answers = {}
    for question_id, question in questions.items():
        if question["type"] == "noul":
            answers[question_id] = {"type": "noul", "noul": noul}
        elif question["type"] == "choice":
            labels = list(question["criteria"])
            answers[question_id] = {
                "type": "choice",
                "choice": labels[0],
                "probabilities": {label: float(label == labels[0]) for label in labels},
                "confidence": confidence,
            }
        else:
            answers[question_id] = {
                "type": "score",
                "score": 0.5,
                "probabilities": {"0": 0.5, "1": 0.5},
                "confidence": 0.5,
            }
    return answers


# --- local System-One endpoints -------------------------------------------------------------


def _server(exchanges: list[tuple[bytes, bytes]], model: str = "jev-1.13.0") -> SystemOneClient:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            sent = self.rfile.read(int(self.headers["content-length"]))
            answers = {id_: {"type": "noul", "noul": 0.9} for id_ in json.loads(sent)["questions"]}
            served = json.dumps(
                {
                    "model": model,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                    "answers": answers,
                }
            ).encode()
            exchanges.append((sent, served))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(served)))
            self.end_headers()
            self.wfile.write(served)

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading_daemon = __import__("threading").Thread(target=httpd.serve_forever, daemon=True)
    threading_daemon.start()
    return SystemOneClient(model="test", api_key="test-key", endpoint=f"http://127.0.0.1:{port}")


def _dead_server() -> SystemOneClient:
    return SystemOneClient(model="test", api_key="test-key", endpoint="http://127.0.0.1:1")


def test_a_routed_budget_refusal_reaches_the_judge_and_splits_without_failover():
    """The real route transport preserves a size refusal for the batching owner to split."""
    refused: list[bytes] = []
    accepted: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            sent = self.rfile.read(int(self.headers["content-length"]))
            if len(sent) > 40_000:
                refused.append(sent)
                body = b'{"detail":{"error_type":"max_tokens_exceeded"}}'
                self.send_response(400)
            else:
                accepted.append(sent)
                body = json.dumps(
                    {
                        "model": "drex-latest",
                        "usage": {"input_tokens": 12, "output_tokens": 1},
                        "answers": {
                            key: {"type": "noul", "noul": 0.9} for key in json.loads(sent)["questions"]
                        },
                    }
                ).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    client = SystemOneClient(model="drex-latest", api_key="test-key", endpoint=f"http://127.0.0.1:{port}")
    backup_exchanges: list[tuple[bytes, bytes]] = []
    routed = RoutedJevClient((Route("drex", client), Route("backup", _server(backup_exchanges))))
    check = Check(
        "has_code",
        "Does `{item}.code` contain code?",
        yes=Criterion("Code is present."),
        no=Criterion("Code is absent."),
    )
    results = Judge(routed).check_every(
        [check], [{"code": "x" * 24_000}, {"code": "y" * 24_000}], list_name="items"
    )

    assert len(results["has_code"]) == 2
    assert len(refused) == 1
    assert len(accepted) == 2
    assert backup_exchanges == []
