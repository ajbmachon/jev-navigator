"""Typed answers become ranks in code; absence of an answer is never a negative."""

from __future__ import annotations

import json

from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2

from jev_navigator.judgments.answers import JevResponse
from jev_navigator.judgments.judge import Judge

from .designs import Design, profile


def baseline_scores(probabilities: dict) -> float:
    return ROLES_V2.rank(probabilities)


def consume_jev(design: Design, request: dict, response: JevResponse) -> list[dict]:
    """Consume only answers to this exact group, retaining all primitive-specific fields."""
    rows = []
    state = request["state"]
    if design.family == "J4":
        for point, question_id in zip(state["targets"], request["questions"], strict=True):
            answer = response.choice(question_id)
            options = request["questions"][question_id]["criteria"]
            if set(answer.probabilities) != set(options) or answer.choice not in options:
                raise ValueError("Choice options do not match the submitted request")
            for slot in range(len(state["items"])):
                rows.append(
                    {
                        "point": point,
                        "slot": slot,
                        "score": answer.probabilities[str(slot)],
                        "score_kind": "choice_option_probability",
                        "raw_question_id": question_id,
                        "winner": answer.choice,
                        "choice_confidence": answer.confidence,
                        "confirmed_pick": answer.choice == str(slot) and answer.confidence >= 0.70,
                    }
                )
        return rows
    asked = profile(design).asked(state["targets"])
    for slot in range(len(state["items"])):
        for check, _, point in asked.values():
            question_id = f"{check.question_id}#{slot}"
            probability = response.noul(question_id).probability
            rows.append(
                {
                    "point": point,
                    "slot": slot,
                    "score": probability,
                    "score_kind": "noul_yes_probability",
                    "raw_question_id": question_id,
                }
            )
    return rows


def consume_llm(text: str, targets: dict, item_count: int) -> list[dict]:
    """Reject unknown ids, points and duplicates, rather than partly accepting a malformed answer."""
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != set(targets):
        raise ValueError("LLM answer must contain exactly the submitted targets")
    offered = {str(i) for i in range(item_count)}
    rows = []
    for point in targets:
        ids = result[point]
        if (
            not isinstance(ids, list)
            or any(not isinstance(i, str) or i not in offered for i in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("LLM returned an invalid or repeated item id")
        rows.extend(
            {
                "point": point,
                "slot": slot,
                "score": float(str(slot) in ids),
                "score_kind": "llm_binary_selection",
            }
            for slot in range(item_count)
        )
    return rows


async def judge_group(judge: Judge, design: Design, request: dict) -> list[dict]:
    """JVN dispatches the frozen group, with no regrouping or independent judge implementation."""
    if judge.input_limits.exceeded_by(request["state"], request["questions"]):
        raise ValueError("Candidate does not fit the existing JVN input box; no regrouping permitted")
    response = await judge.ask_async(
        request["state"], request["questions"], thresholds=judge.thresholds, masked=frozenset()
    )
    return consume_jev(design, request, response)


def packing_policy(family: str) -> dict:
    """The existing allocator consumes ranks; new one-signal arms do not invent six role answers."""
    return {
        "required_roles": LOCAL_ROLES if family == "R0" else (),
        "score_at": 0.80,
        "choice_confidence_at": 0.70,
        "per_point": 5,
        "rooms": (7200, 20000, 36000),
    }
