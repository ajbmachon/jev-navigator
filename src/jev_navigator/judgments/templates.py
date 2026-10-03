"""Find v2 question templates: fixed wording over variable state.

Every unit in a round gets the match question, jgrep's one question per unit without criteria, with
the request's behaviour moved out of the wording into state. Like jgrep's, it points at the whole
unit, so Jev reads its file and lines with its code. The best few units also get the four behaviour
role questions, one Noul per role because a unit can play several. All of them read one shared state
field, ``target``, built by ``target_state``; each per-unit question names its own unit and nothing
else.

The four role names are the behaviour role list (``behavior_role``) that a request's ``want`` and a
result's labels use.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..index.code_index import CodeIndex
from ..index.spans import Span
from .judge import CODE_FIELD
from .questions import Check, Criterion

TARGET = "target"
EXAMPLE_BEHAVIOUR = "For the check that limits how many items an order may hold"

MATCH = Check(
    name="match",
    instructions="Look only at `{item}`. Does that code match the description in `target`?",
)

PERFORMS = Check(
    name="performs",
    instructions="Do lines in `{item}.code` themselves carry out the behaviour described in `target`?",
    yes=Criterion(
        "Lines in `{item}.code` do the described work, all of it or a part of it.",
        examples=(f"{EXAMPLE_BEHAVIOUR}: the code compares the number of items with the limit.",),
    ),
    no=Criterion(
        "The described work happens only in code that `{item}.code` calls, registers or receives,"
        " or `{item}.code` does something else.",
        not_for="A unit that does part of the work itself and passes the rest on.",
    ),
)

HANDS_OFF = Check(
    name="hands_off",
    instructions=(
        "Does `{item}.code` pass the behaviour described in `target` on to other code that carries it out?"
    ),
    yes=Criterion(
        "`{item}.code` calls, registers, schedules or dispatches to the code that does the described work.",
        examples=(f"{EXAMPLE_BEHAVIOUR}: an order handler that calls the limit check before saving.",),
    ),
    no=Criterion(
        "`{item}.code` does not call, register or dispatch to the code that does the described work.",
        not_for="Code that only receives the outcome after the work is done.",
    ),
)

SELECTS_OR_CONFIGURES = Check(
    name="selects_or_configures",
    instructions="Does `{item}.code` choose or set what the behaviour described in `target` uses?",
    yes=Criterion(
        "`{item}.code` decides which implementation, handler or setting value the described behaviour"
        " uses, or writes a setting that behaviour reads.",
        examples=(
            f"{EXAMPLE_BEHAVIOUR}: code that reads the item limit from configuration, or registers which"
            " check runs.",
        ),
    ),
    no=Criterion("`{item}.code` neither chooses nor sets anything the described behaviour uses."),
)

CONSUMES = Check(
    name="consumes",
    instructions="Does `{item}.code` use what the behaviour described in `target` produces?",
    yes=Criterion(
        "`{item}.code` receives, reads or acts on the outcome of the described behaviour.",
        examples=(
            f"{EXAMPLE_BEHAVIOUR}: code that returns an error response when the check rejects the order.",
        ),
    ),
    no=Criterion("`{item}.code` does not use the outcome of the described behaviour."),
)

BEHAVIOR_ROLE_QUESTIONS: Mapping[str, Check] = {
    role.name: role for role in (PERFORMS, HANDS_OFF, SELECTS_OR_CONFIGURES, CONSUMES)
}
TEMPLATES: tuple[Check, ...] = (MATCH, *BEHAVIOR_ROLE_QUESTIONS.values())


def target_state(request: Mapping) -> dict:
    """The shared state every template reads: the request's behaviour and, only when it names any,
    its conditions as data. Nothing else from the request reaches Jev; ``want`` only orders the
    returned group, and scope, anchors and budget are code's."""
    behavior = request.get("behavior")
    if not behavior:
        raise ValueError(
            "the templates judge units against the request's behavior, and this request has none"
        )
    target: dict = {"behavior": behavior}
    if request.get("conditions"):
        target["conditions"] = dict(request["conditions"])
    return {TARGET: target}


def unit_entry(index: CodeIndex, span: Span) -> dict:
    """One unit as Jev sees it: its location and its code, nothing a search concluded about it. The
    location is what ``rebuild_request`` re-reads the code from."""
    return {"file": span.file, "lines": [span.start, span.end], CODE_FIELD: index.read_slice(span).text}
