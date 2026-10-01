"""Drex by Nace.AI: the library's questions reach Drex in its text-criteria dialect, at its pinned
endpoint, and its answers parse. The contract shared with every adapter is in test_adapters.py."""

from __future__ import annotations

import json
import re

from jev_navigator.adapters.drex import DrexClient
from jev_navigator.adapters.system_one import HttpRequest, HttpResponse
from jev_navigator.judgments.questions import Check, Criterion, Rate

ADDS_ONE = Check(
    name="adds_one",
    instructions="Does `code` add one to its argument?",
    yes=Criterion("The code adds one.", not_for="Code that adds two.", examples=("`x + 1`.",)),
    no=Criterion("The code does something else."),
)
MATCH = Rate(
    name="match", instructions="How well does `code` match `sentence`?", levels=("no", "partly", "yes")
)
STATE = {"code": "def add_one(x):\n    return x + 1\n", "sentence": "increments a number"}


def test_a_check_and_a_score_reach_drex_as_text_and_its_answers_parse():
    # Arrange: Drex's live answer shape, including the score `legend` list it echoes.
    sent: list[HttpRequest] = []
    served = {
        "model": "drex-v1.5",
        "answers": {
            ADDS_ONE.question_id: {"type": "noul", "noul": 0.9938},
            MATCH.question_id: {
                "type": "score",
                "score": 1.9703,
                "legend": ["no", "partly", "yes"],
                "probabilities": {"0": 0.001, "1": 0.0277, "2": 0.9713},
                "confidence": 0.9555,
            },
        },
        "usage": {"input_tokens": 49, "output_tokens": 92},
        "evaluation_time_ms": 5.8,
        "request_id": "req_00000000000000000000000000000001",
    }

    def drex(request):
        sent.append(request)
        return HttpResponse(200, {"content-type": "application/json"}, json.dumps(served).encode())

    client = DrexClient(api_key="nace_sk_test", transport=drex)

    # Act
    answer = client.ask(
        STATE, {ADDS_ONE.question_id: ADDS_ONE.to_question(), MATCH.question_id: MATCH.to_question()}
    )

    # Assert
    request = sent[0]
    body = json.loads(request.body)
    assert request.url == "https://drex.nace.ai/v1/systemone"
    assert request.headers["Authorization"] == "Bearer nace_sk_test"
    assert body["model"] == "drex-latest"
    assert body["questions"][ADDS_ONE.question_id]["criteria"] == {
        "true": '{"what": "The code adds one.", "not_for": "Code that adds two.", "examples": ["`x + 1`."]}',
        "false": '{"what": "The code does something else."}',
    }
    assert body["questions"][MATCH.question_id]["criteria"] == ["no", "partly", "yes"]
    assert answer.model == "drex-v1.5"
    assert answer.noul(ADDS_ONE.question_id).probability == 0.9938
    assert answer.score(MATCH.question_id).score == 1.9703


def test_base64_data_urls_reach_drex_broken_and_other_data_text_unchanged():
    # Arrange: source code that builds data URLs, which Drex refuses as media (HTTP 422).
    sent: list[dict] = []

    def drex(request):
        sent.append(json.loads(request.body))
        answer = {"builds_url": {"type": "noul", "noul": 0.9}}
        return HttpResponse(200, {}, json.dumps({"model": "drex-v1.5", "answers": answer}).encode())

    state = {
        "candidates": [
            {"signature": "const toUri = (bytes) => `data:image/gif;base64,${encode(bytes)}`;"},
            {"signature": 'm = "metadata:IMAGE/PNG;BASE64,iVBOR"'},
        ],
        "code": 'rows = { data: rows }; raw = "data:image/png,bytes"',
    }
    question = {"type": "noul", "instructions": "Is it like data:application/pdf;base64,JV?"}
    client = DrexClient(api_key="nace_sk_test", transport=drex)

    # Act
    answer = client.ask(state, {"builds_url": question})

    # Assert
    assert re.search(r"(?i)data:[^\s,]*;base64,", json.dumps(sent[0], ensure_ascii=False)) is None
    unbroken = json.loads(json.dumps(sent[0]).replace("\\u200b", ""))
    assert unbroken["state"] == state
    assert unbroken["questions"] == {"builds_url": question}
    assert sent[0]["state"]["code"] == state["code"]
    assert answer.noul("builds_url").probability == 0.9
