"""Candidate contracts. JVN owns questions, profiles, masking and dispatch."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from jev_navigator.judgments.profiles import QuestionProfile

from jev_navigator.judgments.questions import Pick, request_sha256
from jev_navigator.judgments.secrets import SecretScanner, refuse_if_secret

YES = "Locally implements a condition, value or action described by the point."
NO = "Only calls, forwards, mentions or sits near that code, or implements something else."
EXAMPLE = "Negative q: 'if q < 0: raise ValueError()' decides; 'return validate(q)' forwards."
DECIDING_EXAMPLE = "if q < 0: raise ValueError()"
FORWARDING_EXAMPLE = "return validate(q)"
MATCH_WORDINGS = (
    "Does `unit.code` itself hold the condition, value or action described in `point`?",
    "Is `unit.code` the implementation of a behavior described in `point`?",
    "Does `unit.code` contain the code that settles a part of `point`?",
)
COUNTERFACTUAL_WORDINGS = (
    "Would changing the local logic in `unit.code` change whether the behavior in `point` happens?",
    "With callees fixed, could changing `unit.code` change whether the behavior described in `point` "
    "happens?",
    "Does whether the behavior described in `point` happens depend on a condition, value or action "
    "implemented in `unit.code`?",
)
CHOICE_WORDINGS = (
    "Which item in `items` itself holds the code that decides `targets.{point}`?",
    "Which item in `items` implements the condition, value or action described in `targets.{point}`?",
    "Which item's own implementation in `items` settles `targets.{point}`?",
)
LLM_WORDINGS = (
    "Return the ids of every supplied item whose own code holds a condition, value or action "
    "described by each point. Exclude code that merely calls, forwards, mentions or sits near it.",
    "For each point, select all supplied items implementing a part that decides the described "
    "behavior. A caller of that implementation is insufficient on its own.",
    "For each point, return all supplied items where changing the local condition, value or "
    "action could change the answer. Keep called implementations fixed; a forwarding call alone "
    "does not qualify.",
)


@dataclass(frozen=True)
class Design:
    name: str
    family: str
    wording: int
    richer: bool = False


DESIGNS = (
    *(Design(f"J1-{i + 1}", "J1", i) for i in range(3)),
    *(Design(f"J2-{i + 1}", "J2", i) for i in range(3)),
    *(Design(f"J3-{family}-{i + 1}", family, i, True) for family in ("J1", "J2") for i in range(3)),
    *(Design(f"J4-{i + 1}", "J4", i) for i in range(3)),
    *(Design(f"L1-{i + 1}", "L1", i) for i in range(3)),
)


def profile(design: Design) -> QuestionProfile:
    counterfactual = design.family == "J2"
    wording = (COUNTERFACTUAL_WORDINGS if counterfactual else MATCH_WORDINGS)[design.wording]
    if design.richer:
        wording += " Use `claim` and `cited_code` to identify the behavior the point concerns."
    yes = YES
    no = NO
    if counterfactual:
        yes = "A local decision change could change the behavior, with inputs, wiring and callees fixed."
        no = "Decision lives elsewhere or is unrelated. Redirecting a forwarding call does not count."
        if design.wording == 2:
            no = "Decision elsewhere or unrelated. Redirecting a forwarding call does not count."
    return QuestionProfile(
        "match",
        {
            "match": {
                "type": "noul",
                "instructions": wording,
                "criteria": {
                    "true": {"what": yes, "examples": [DECIDING_EXAMPLE]},
                    "false": {"what": no, "examples": [FORWARDING_EXAMPLE]},
                },
            }
        },
    )


def jev_request(design: Design, baseline: dict, rich_state: dict | None = None) -> dict:
    state = deepcopy(baseline["state"])
    if design.richer:
        if rich_state is None:
            raise ValueError("Richer state requires the original claim and cited source")
        state.update(deepcopy(rich_state))
    if design.family == "J4":
        questions = {}
        for point in state["targets"]:
            pick = Pick(
                f"deciding_{point}",
                CHOICE_WORDINGS[design.wording].replace("{point}", point)
                + " Options are item indices. Pick the strongest if several qualify. "
                + EXAMPLE,
                (("none", "No item holds that logic."),),
            )
            options = {str(i): "" for i in range(len(state["items"]))}
            questions[pick.question_id] = pick.to_question(options)
    else:
        asked = profile(design).asked(state["targets"])
        questions = {
            f"{check.question_id}#{slot}": check.to_question(f"items[{slot}]")
            for slot in range(len(state["items"]))
            for check, _, _ in asked.values()
        }
    return {"model": "jev-latest", "state": state, "questions": questions}


def llm_request(design: Design, baseline: dict) -> dict:
    state = baseline["state"]
    ids = [str(i) for i in range(len(state["items"]))]
    properties = {
        point: {
            "type": "array",
            "maxItems": len(ids),
            "uniqueItems": True,
            "items": {"type": "string", "enum": ids},
        }
        for point in state["targets"]
    }
    prompt = (
        LLM_WORDINGS[design.wording]
        + "\n"
        + EXAMPLE
        + "\nEach id is the decimal index of its item in items. Return an empty list when none "
        "qualifies. This selects evidence; it does not classify the finding as true or false. "
        "Return only the JSON object with one id list per supplied target.\n"
        + "STATE_JSON:\n"
        + json.dumps(state, ensure_ascii=False)
    )
    return {
        "model": "sference/deepseek-v4-flash-0731",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "deciding_items",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        },
    }


def scan_llm_request(request: dict) -> None:
    """Scan the complete semantic input with source paths, before JSON escaping loses that context."""
    prefix, separator, encoded = request["messages"][0]["content"].partition("STATE_JSON:\n")
    if not separator:
        raise ValueError("Missing structured LLM state")
    state = json.loads(encoded)
    refuse_if_secret({"instructions": prefix, "state": state}, {}, SecretScanner())


def identity(request: dict) -> str:
    if "questions" in request:
        return request_sha256(request["state"], request["questions"])
    import hashlib

    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
