"""One search of many places, run in its own process by ``test_send_bound``: a judge that
deadlocks leaves worker threads Python waits for at exit, so only a process the test can kill
fails cleanly. Prints what the provider saw as one JSON object."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from git_repos import commit_files, git

from jev_navigator.directives.find_code import FindResult, SearchBudget, find_code, find_code_async
from jev_navigator.directives.places import Place, function_place
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import InputBudgetExceededError
from jev_navigator.judgments.journal import RawResponse
from jev_navigator.judgments.judge import DEFAULT_MAX_CONCURRENCY, Judge
from jev_navigator.testing import ScriptedJevClient

TARGET = "the check that limits how many items an order may have"
HELPERS = "\n\n".join(f"def h{index}(order):\n    return {index}" for index in range(4)) + "\n"
SEND_HOLD_SECONDS = 0.1


@dataclass
class HeldClient:
    """Stands in for the remote provider and counts the requests in flight. Each send stays in flight
    for ``SEND_HOLD_SECONDS``, or until more than ``bound`` are in flight together, so a judge without
    one shared bound shows a peak above it at once instead of by timing. With ``refuse_lists`` it
    refuses, for its input size, any request whose state carries more than one candidate, which
    splits every opening into its found question and one batch per neighbour."""

    bound: int = DEFAULT_MAX_CONCURRENCY
    refuse_lists: bool = False
    script: ScriptedJevClient = field(default_factory=lambda: ScriptedJevClient(default_noul=0.05))
    in_flight: int = 0
    peak: int = 0
    _changed: threading.Condition = field(default_factory=threading.Condition)

    @property
    def model(self) -> str:
        return self.script.model

    @property
    def requests(self) -> list:
        return self.script.requests

    def ask(self, state: Mapping, questions: Mapping):
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        self._arrive()
        try:
            with self._changed:
                self._changed.wait_for(self._over_bound, timeout=SEND_HOLD_SECONDS)
            return self._answer(state, questions)
        finally:
            self._leave()

    def parse(self, raw: RawResponse):
        return self.script.parse(raw)

    def _arrive(self) -> None:
        with self._changed:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self._changed.notify_all()

    def _leave(self) -> None:
        with self._changed:
            self.in_flight -= 1

    def _over_bound(self) -> bool:
        return self.in_flight > self.bound

    def _answer(self, state: Mapping, questions: Mapping) -> RawResponse:
        if self.refuse_lists and len(state.get("candidates", ())) > 1:
            raise InputBudgetExceededError("TypeSafeBadRequestError: 400 max_tokens_exceeded")
        return self.script.send(state, questions)


@dataclass
class AsyncHeldClient:
    """``HeldClient`` with an awaitable send, holding each request on the event loop."""

    held: HeldClient

    @property
    def model(self) -> str:
        return self.held.model

    async def ask(self, state: Mapping, questions: Mapping):
        return self.parse(await self.send(state, questions))

    async def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        self.held._arrive()
        try:
            deadline = time.monotonic() + SEND_HOLD_SECONDS
            while not self.held._over_bound() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            return self.held._answer(state, questions)
        finally:
            self.held._leave()

    def parse(self, raw: RawResponse):
        return self.held.parse(raw)


def _many_functions_index(tmp_path: Path, count: int) -> CodeIndex:
    """A repository of ``count`` distinct functions, one per file, each calling two shared helpers."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    files = {"app/__init__.py": "", "app/helpers.py": HELPERS}
    for index in range(count):
        files[f"app/f{index}.py"] = (
            f"from app.helpers import h{index % 4}, h{(index + 1) % 4}\n\n\n"
            f"def f{index}(order):\n    return h{index % 4}(order) + h{(index + 1) % 4}(order) + {index}\n"
        )
    commit_files(root, files)
    return CodeIndex.from_git(root, fact_cache_dir=tmp_path / "fact-cache")


def _starts(index: CodeIndex, count: int) -> list[Place]:
    return [function_place(index, index.enclosing_symbol(f"app/f{number}.py", 5)) for number in range(count)]


def _search(mode: str, index: CodeIndex, held: HeldClient, width: int) -> FindResult:
    """One round of a search opening ``width`` places at once, through the sync or the async path."""
    budget = SearchBudget(beam_width=width, max_steps=width, neighbours_per_kind=2)
    starts = _starts(index, width)
    if mode == "sync":
        return find_code(index, Judge(held, items_per_request=1), TARGET, starts, budget=budget)
    judge = Judge(AsyncHeldClient(held), items_per_request=1)
    return asyncio.run(find_code_async(index, judge, TARGET, starts, budget=budget))


def main(mode: str, width: int, refuse_lists: bool, workspace: Path) -> dict:
    index = _many_functions_index(workspace, width)
    held = HeldClient(refuse_lists=refuse_lists)
    result = _search(mode, index, held, width)
    candidates = [len(state.get("candidates", ())) for state, _ in held.requests]
    return {
        "steps": result.steps,
        "requests": len(held.requests),
        "neighbour_batches": candidates.count(1),
        "peak": held.peak,
    }


if __name__ == "__main__":
    mode, width, refuse_lists, workspace = sys.argv[1:]
    print(json.dumps(main(mode, int(width), refuse_lists == "refuse-lists", Path(workspace))))
