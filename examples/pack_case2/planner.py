"""The development planner's prompt contract. Rendering and parsing are free."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from jev_navigator.llm_step import ReplyParseError
from jev_navigator.search_plan import SearchPlan, decode_plan, plan_schema

MODEL = "sference/deepseek-v4-flash-0731"
INSTRUCTIONS = """Plan where to read the code that decides the supplied finding.
Return at most ten independent, ranked calls conforming to the schema. Use fewer
when approaches duplicate one another. Give each argument its copied or
transformed source provenance, a short reason and the gap it addresses.
Paths and owners must be exact repository-relative files visible in the outline
or cited code. Names must come from the supplied context. Use matching_files to
test a convention hypothesis, and mark that provenance as a hypothesis.
A caller or callee follows code already known in this round. If an argument
requires a new result, defer it to another round. Empty results and invalid
targets will be reported, never repaired by inventing a replacement.
Attempted searches and coverage distinguish observed code, pending candidates,
exclusions, exhausted scopes and unknown roles. Do not require every role to be
high. A claim that a guard is missing needs a search of the concrete places where
it would apply. Propose those places; never infer absence from a missing score.
JVN executes the calls together. Jev judges the resulting shortlist with the
existing six-role profile; the plan is not a verdict. If a scope is broad, name
the smaller visible scope that would answer the same question. Return only JSON.
"""


@dataclass(frozen=True)
class PlannerInput:
    finding: str
    cited_code: tuple[dict, ...]
    outline: str
    attempted_searches: tuple[dict, ...]
    coverage: dict


@dataclass(frozen=True)
class PlannerContract:
    def render(self, context: PlannerInput, parse_error: str = "") -> str:
        retry = f"\nPrevious parse error: {parse_error}\n" if parse_error else ""
        return (
            INSTRUCTIONS
            + "\nContext:\n"
            + json.dumps(asdict(context), ensure_ascii=False)
            + "\nOutput schema:\n"
            + json.dumps(plan_schema(), ensure_ascii=False)
            + retry
        )

    def parse(self, reply: str, context: PlannerInput) -> SearchPlan:
        try:
            return decode_plan(reply)
        except (ValueError, TypeError) as error:
            raise ReplyParseError(str(error)) from error
