"""Protect the real async dispatch boundary and typed selection admission, without model calls."""

import asyncio
import json

import pytest

from jev_navigator.judgments.answers import ChoiceAnswer, JevResponse, NoulAnswer
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.secrets import SecretInRequestError

from .consume import consume_jev, consume_llm, judge_group
from .designs import DESIGNS, identity, jev_request, llm_request, scan_llm_request
from .frozen import freeze_case


def test_frozen_groups_use_existing_async_judge_without_changing_companions():
    baseline = {
        "state": {
            "targets": {"p0": "The implementation that rejects negative quantities."},
            "items": [{"file": f"item{i}.py", "code": f"def item{i}(q): return q"} for i in range(16)],
        }
    }
    for design in (DESIGNS[0], DESIGNS[3], DESIGNS[12]):
        request = jev_request(design, baseline)
        observed = []

        class Provider:
            model = "fixture"

            async def ask(self, state, questions, observed=observed):
                observed.append({"state": state, "questions": questions})
                answers = {}
                for qid, question in questions.items():
                    answers[qid] = (
                        ChoiceAnswer.from_probabilities(
                            {key: float(key == "3") for key in question["criteria"]}
                        )
                        if question["type"] == "choice"
                        else NoulAnswer(0.9 if qid.endswith("#3") else 0.1)
                    )
                return JevResponse(answers, self.model)

        judge = Judge(Provider(), masker=None, max_calls=1)
        result = asyncio.run(judge_group(judge, design, request))
        assert observed == [{"state": request["state"], "questions": request["questions"]}]
        assert observed[0]["state"]["items"] == baseline["state"]["items"]
        assert len(result) == 16 and judge.calls == 1
        assert max(result, key=lambda row: row["score"])["slot"] == 3


def test_choice_keeps_probability_confidence_and_none_separate():
    design = DESIGNS[12]
    request = jev_request(
        design,
        {"state": {"targets": {"p0": "Reject negative quantity"}, "items": [{"code": "return other(q)"}]}},
    )
    qid = next(iter(request["questions"]))
    response = JevResponse({qid: ChoiceAnswer("none", {"0": 0.1, "none": 0.9}, 0.8)}, "fixture")
    rows = consume_jev(design, request, response)
    assert rows[0]["score"] == 0.1
    assert rows[0]["choice_confidence"] == 0.8
    assert not rows[0]["confirmed_pick"]


def test_llm_ids_are_closed_and_invalid_output_never_becomes_a_negative_answer():
    rows = consume_llm('{"p0":["1"],"p1":[]}', {"p0": "first", "p1": "second"}, 3)
    assert [(row["point"], row["slot"]) for row in rows if row["score"]] == [("p0", 1)]
    for reply in (
        '{"p0":["unlisted"]}',
        '{"p0":[1]}',
        '{"p0":["1","1"]}',
        '{"p0":null}',
        "{}",
        '{"p0":[],"extra":[]}',
    ):
        with pytest.raises(ValueError):
            consume_llm(reply, {"p0": "point"}, 3)


def test_llm_scan_preserves_code_references_and_rejects_literal_secrets():
    baseline = {
        "state": {
            "targets": {"p0": "Read a token from config"},
            "items": [
                {"file": "owner.py", "code": "token = config.token"},
                # Reduced from P6: a JavaScript object field is code, not a config secret.
                {"file": "src/Options/Definitions.js", "code": "  allowExpiredAuthDataToken: {"},
            ],
        }
    }
    request = llm_request(DESIGNS[-1], baseline)
    scan_llm_request(request)
    assert json.loads(request["messages"][0]["content"].split("STATE_JSON:\n", 1)[1]) == baseline["state"]
    baseline["state"]["items"][0]["code"] = "api_key = 'fixture-secret-value'"
    with pytest.raises(SecretInRequestError):
        scan_llm_request(llm_request(DESIGNS[-1], baseline))


def test_baseline_join_checks_raw_answers_not_just_source_names(tmp_path):
    folder = tmp_path
    wire = folder / "code/http/jev_one"
    wire.mkdir(parents=True)
    request = {
        "model": "fixture",
        "state": {"targets": {"p0": "reject"}, "items": [{"file": "a.py", "code": "raise ValueError()"}]},
        "questions": {
            f"{role}_p0@fixed#0": {"type": "noul", "instructions": "Reject?"}
            for role in ("decide", "guard", "value", "effect", "delegates", "satisfied")
        },
    }
    probabilities = dict.fromkeys(("decide", "guard", "value", "effect", "delegates", "satisfied"), 0.9)
    files = {
        "context.json": {"case": "fixture"},
        "labels.json": [],
        "code-shape.json": {"requests": [request]},
        "code/result.json": {},
        "code/observations.json": [
            {
                "point": "p0",
                "request_sha256": identity(request),
                "probabilities": probabilities,
                "place": {"id": "source", "file": "a.py", "ranges": [[1, 1]]},
            }
        ],
        "code/http/jev_one/request.json": request,
        "code/http/jev_one/response.json": {"answers": {qid: {"noul": 0.9} for qid in request["questions"]}},
        "code/http/jev_one/receipt.json": {},
    }
    for name, value in files.items():
        (folder / name).write_text(json.dumps(value))
    assert freeze_case(folder)["groups"][0]["baseline_rows"][0]["score"] == 0.9
    # Model presentation may hide a path component; original source identity remains exact.
    request["state"]["items"][0]["file"] = "[MASKED].py"
    files["code-shape.json"]["requests"] = [request]
    files["code/observations.json"][0]["request_sha256"] = identity(request)
    for name in ("code-shape.json", "code/http/jev_one/request.json", "code/observations.json"):
        (folder / name).write_text(json.dumps(files[name]))
    assert freeze_case(folder)["groups"][0]["baseline_rows"][0]["place"]["file"] == "a.py"
    response = json.loads((wire / "response.json").read_text())
    response["answers"][next(iter(request["questions"]))]["noul"] = 0.2
    (wire / "response.json").write_text(json.dumps(response))
    with pytest.raises(ValueError, match="slot join"):
        freeze_case(folder)
