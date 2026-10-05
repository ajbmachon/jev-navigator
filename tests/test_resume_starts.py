"""Starting evidence remains starting evidence when a search resumes."""

import asyncio

import pytest

from jev_navigator.directives.find_code import SearchBudget, find_code, find_code_async
from jev_navigator.directives.places import place_for_line
from jev_navigator.judgments.client import JEV_INPUT_LIMITS, InputLimits
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unopened_starts_keep_their_role_after_resume(sample_index, asynchronous):
    # Arrange: a budget interruption happens before the caller's evidence is opened.
    start = place_for_line(sample_index, "app/validation.py", 11, "start")
    client = ScriptedJevClient(default_noul=0.95)
    judge = Judge(AsyncScriptedJevClient(client) if asynchronous else client)

    def search(starts, **options):
        arguments = (sample_index, judge, "the item limit check", starts)
        return (
            asyncio.run(find_code_async(*arguments, moves={}, **options))
            if asynchronous
            else find_code(*arguments, moves={}, **options)
        )

    interrupted = search([start], budget=SearchBudget(max_steps=0))

    # Act: resume the same search with enough budget to inspect its waiting start.
    resumed = search([], resume=interrupted)

    # Assert: rediscovering the caller's own evidence is never a new finding.
    assert not interrupted.starts and len(interrupted.not_inspected) == 1
    assert not resumed.found
    assert [visit.place_key for visit in resumed.starts] == [start.key]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unshown_source_stays_uninspected_until_resume(sample_index, asynchronous):
    # Arrange: the first search asks a route whose input box cannot hold even the start's first line
    start = place_for_line(sample_index, "app/validation.py", 11, "start")

    def search(input_limits: InputLimits, starts, **options):
        client = ScriptedJevClient(default_noul=0.05)
        routed = AsyncScriptedJevClient(client) if asynchronous else client
        routed.input_limits = input_limits
        arguments = (sample_index, Judge(routed), "the item limit check", starts)
        return (
            asyncio.run(find_code_async(*arguments, moves={}, **options))
            if asynchronous
            else find_code(*arguments, moves={}, **options)
        )

    interrupted = search(InputLimits(box_chars=1), [start])

    assert interrupted.outcome == "budget"
    assert interrupted.calls == interrupted.steps == 0
    assert not interrupted.starts and not interrupted.visited and not interrupted.judged_code
    assert [(item.place_key, item.reason) for item in interrupted.not_inspected] == [(start.key, "budget")]

    resumed = search(JEV_INPUT_LIMITS, [], resume=interrupted)

    assert resumed.calls == resumed.steps == 1
    assert [visit.place_key for visit in resumed.starts] == [start.key]
    assert resumed.starts[0].code.text == start.open().text
    assert resumed.starts[0].code.span.start <= resumed.starts[0].code.span.end
    assert not resumed.not_inspected
