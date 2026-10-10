"""A value the masker hides anywhere in a search's repository is hidden in every request the search
sends, whichever request carries the code a rule finds it in; code that holds no copy is sent as before."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from git_repos import write_files

from jev_navigator.directives.agent_search import agent_search, agent_search_async
from jev_navigator.directives.find_all import find_all, find_all_async
from jev_navigator.directives.find_code import find_code
from jev_navigator.directives.places import function_place
from jev_navigator.directives.trace import trace_workflow
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.known_values import KNOWN_VALUE_MAX_FILES, KnownValuesMasker
from jev_navigator.judgments.secrets import SecretMasker
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

VALUE = "Qm4vT8xLr2Zp9wKc"
PLACEHOLDER = "changeme-local-only"
# The bare copy's file sorts first, so a search judging one unit per request and wave asks about it
# before the code a rule finds the value in, and trace and find_code, which start at it, may never reach
# that code at all.
USE = {"shop/a_session.py": f'def connect(order):\n    return dial(order, "{VALUE}")\n'}
CONFIG = {"shop/z_config.py": f'def configure(order):\n    password = "{VALUE}"\n    return order\n'}
TARGET = "code that opens a session for an order"
REQUEST = {
    "hypotheses": [
        {"id": "h1", "mechanism": "sessions", "evidence": [{"id": "e1", "point": TARGET}], "refuted_by": []}
    ],
    "terms": [],
    "anchors": [],
    "files": [*USE, *CONFIG],
    "scope": {"include": [], "exclude": [], "with_tests": False},
    "follow": [],
    "budget_requests": 8,
}
Search = Callable[[CodeIndex, ScriptedJevClient], object]


def repository(root: Path, files: dict[str, str]) -> CodeIndex:
    write_files(root, files)
    return CodeIndex(root, files)


def judge(client: ScriptedJevClient, *, asynchronous: bool = False) -> Judge:
    return Judge(AsyncScriptedJevClient(client) if asynchronous else client, items_per_request=1)


def connect(index: CodeIndex):
    return index.find_definition("connect")[0]


SEARCHES: dict[str, Search] = {
    "find_all": lambda index, client: find_all(
        index, judge(client), {"session": TARGET}, files=index.files, batches_per_wave=1
    ),
    "find_all_async": lambda index, client: asyncio.run(
        find_all_async(
            index,
            judge(client, asynchronous=True),
            {"session": TARGET},
            files=index.files,
            batches_per_wave=1,
        )
    ),
    "agent_search": lambda index, client: agent_search(REQUEST, index, judge(client)),
    "agent_search_async": lambda index, client: asyncio.run(
        agent_search_async(REQUEST, index, judge(client, asynchronous=True))
    ),
    "find_code": lambda index, client: find_code(
        index, judge(client), TARGET, [function_place(index, connect(index))]
    ),
    "trace_workflow": lambda index, client: trace_workflow(
        index, judge(client), "How is a session opened?", [connect(index)]
    ),
}


def sent(client: ScriptedJevClient) -> str:
    return json.dumps(client.requests)


@pytest.mark.parametrize("search", SEARCHES.values(), ids=SEARCHES)
def test_a_bare_copy_is_hidden_although_no_request_before_it_carried_the_code_a_rule_finds_it_in(
    tmp_path: Path, search: Search
) -> None:
    # Arrange
    index = repository(tmp_path, {**USE, **CONFIG})
    client = ScriptedJevClient(default_noul=0.1)

    # Act
    search(index, client)

    # Assert
    assert "dial(order" in sent(client)
    assert VALUE not in sent(client)


def test_code_holding_no_copy_is_sent_byte_for_byte_whatever_secrets_the_repository_holds(
    tmp_path: Path,
) -> None:
    # Arrange
    plain = {"shop/orders.py": "def place(order):\n    return save(order)\n"}
    clients = {"without": ScriptedJevClient(default_noul=0.1), "with": ScriptedJevClient(default_noul=0.1)}

    # Act
    for name, files in (("without", plain), ("with", {**plain, **CONFIG})):
        index = repository(tmp_path / name, files)
        find_all(index, judge(clients[name]), {"session": TARGET}, files=list(plain))

    # Assert
    assert clients["with"].requests == clients["without"].requests


@pytest.mark.parametrize(
    ("copies", "hidden"), [(KNOWN_VALUE_MAX_FILES - 1, True), (KNOWN_VALUE_MAX_FILES, False)]
)
def test_a_value_standing_in_more_files_than_a_secret_would_is_hidden_only_in_its_own_request(
    tmp_path: Path, copies: int, hidden: bool
) -> None:
    # Arrange: one file a rule finds the placeholder in, and ``copies`` files holding it bare
    config = {"shop/z_config.py": CONFIG["shop/z_config.py"].replace(VALUE, PLACEHOLDER)}
    uses = {
        f"shop/use_{n}.py": f'def use_{n}(order):\n    return dial(order, "{PLACEHOLDER}")\n'
        for n in range(copies)
    }
    index = repository(tmp_path, {**uses, **config})
    client = ScriptedJevClient(default_noul=0.1)

    # Act
    find_all(index, judge(client), {"session": TARGET}, files=["shop/use_0.py"])

    # Assert
    assert (PLACEHOLDER not in sent(client)) is hidden


def test_the_known_values_masker_keeps_the_token_pattern_of_the_masker_it_wraps() -> None:
    # Arrange
    class OwnTokens(SecretMasker):
        token_pattern = re.compile(r"<hidden:\d+>")

    # Act
    wrapped = KnownValuesMasker(OwnTokens(), frozenset({VALUE}))

    # Assert
    assert wrapped.token_pattern is OwnTokens.token_pattern
