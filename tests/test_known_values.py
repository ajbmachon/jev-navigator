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
from jev_navigator.directives.entry import choose_initial_candidates
from jev_navigator.directives.find_all import find_all, find_all_async
from jev_navigator.directives.find_code import find_code
from jev_navigator.directives.places import function_place
from jev_navigator.directives.shown import LINES_CUT_MARK
from jev_navigator.directives.trace import trace_workflow
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.known_values import KNOWN_VALUE_MAX_FILES, KnownValuesMasker, repository_values
from jev_navigator.judgments.secrets import DEFAULT_MASKER, SecretMasker, mask_request
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
    "files": ["shop/a_session.py", "shop/z_config.py"],
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


POINT = f"code that dials an order session with {VALUE}"
ENV_ONLY = {
    "deploy/app.env": f"DB_PASSWORD={VALUE}\n",
    "shop/a_session.py": "def connect(order):\n    return dial(order)\n",
    "shop/b_orders.py": "def place(order):\n    return order\n",
}
POINT_SEARCHES: dict[str, Search] = {
    "entry": lambda index, client: choose_initial_candidates(index, Judge(client), POINT),
    "agent_search": lambda index, client: agent_search(
        {**REQUEST, "hypotheses": [{**REQUEST["hypotheses"][0], "evidence": [{"id": "e1", "point": POINT}]}]},
        index,
        Judge(client),
    ),
}


@pytest.mark.parametrize("search", POINT_SEARCHES.values(), ids=POINT_SEARCHES)
def test_a_point_never_sends_a_value_its_writer_copied_from_a_file_no_request_shows(
    tmp_path: Path, search: Search
) -> None:
    # Arrange: the value stands only in an env file, which no request ever shows
    index = repository(tmp_path, ENV_ONLY)
    client = ScriptedJevClient(default_noul=0.6)

    # Act
    search(index, client)

    # Assert
    assert client.requests
    assert VALUE not in sent(client)


# "described" is a word of JVN's own question wording, which request masking leaves as it is.
@pytest.mark.parametrize("secret", [VALUE, "described"])
def test_code_holding_no_copy_and_jvns_wording_are_sent_byte_for_byte_whatever_secrets_the_repository_holds(
    tmp_path: Path, secret: str
) -> None:
    # Arrange
    plain = {"shop/orders.py": "def place(order):\n    return save(order)\n"}
    config = {"shop/z_config.py": CONFIG["shop/z_config.py"].replace(VALUE, secret)}
    clients = {"without": ScriptedJevClient(default_noul=0.1), "with": ScriptedJevClient(default_noul=0.1)}

    # Act
    for name, files in (("without", plain), ("with", {**plain, **config})):
        index = repository(tmp_path / name, files)
        find_all(index, judge(clients[name]), {"session": TARGET}, files=list(plain))

    # Assert
    assert secret in sent(clients["with"]) or secret == VALUE
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


def test_copies_in_binary_files_never_count_toward_the_limit(tmp_path: Path) -> None:
    # Arrange: the value stands in as many text files as a known value may, and in binary files too
    uses = {
        f"shop/use_{n}.py": f'def use_{n}(order):\n    return dial(order, "{VALUE}")\n'
        for n in range(KNOWN_VALUE_MAX_FILES - 1)
    }
    blobs = {f"assets/blob_{n}.bin": f"\0{VALUE}\0" for n in range(2)}
    index = repository(tmp_path, {**uses, **blobs, **CONFIG})

    # Act
    known = repository_values(index)

    # Assert
    assert VALUE in known


KEY_LINES = "\n".join(
    "".join(chr(ord("A") + (row * 7 + column) % 26) for column in range(64)) for row in range(8)
)
KEY = "-----BEGIN RSA " + "PRIVATE KEY-----\n" + KEY_LINES + "\n-----END RSA " + "PRIVATE KEY-----"


@pytest.mark.parametrize(
    ("files", "known"), [(KNOWN_VALUE_MAX_FILES, True), (KNOWN_VALUE_MAX_FILES + 1, False)]
)
def test_a_value_spanning_lines_is_counted_in_every_file_holding_it(
    tmp_path: Path, files: int, known: bool
) -> None:
    # Arrange
    keys = {f"keys/key_{n}.py": f'KEY = """\n{KEY}\n"""\n' for n in range(files)}
    index = repository(tmp_path, keys)

    # Act
    values = repository_values(index)

    # Assert
    spanning = [value for value in values if "\n" in value]
    assert bool(spanning) is known


def test_a_file_holding_only_one_line_of_a_value_spanning_lines_does_not_hold_it(tmp_path: Path) -> None:
    # Arrange: the key stands whole in as many files as a known value may, and its lines stand alone in more
    keys = {f"keys/key_{n}.py": f'KEY = """\n{KEY}\n"""\n' for n in range(KNOWN_VALUE_MAX_FILES)}
    parts = {f"docs/part_{n}.txt": f"{line}\n" for n, line in enumerate(KEY_LINES.split("\n"))}
    index = repository(tmp_path, {**keys, **parts})

    # Act
    values = repository_values(index)

    # Assert
    assert [value for value in values if "\n" in value]


def pieces_of(value: str, text: str, at_least: int = 4) -> list[str]:
    """The parts of ``value``, ``at_least`` characters or longer, that ``text`` holds."""
    return [
        value[start:end]
        for start in range(len(value))
        for end in range(start + at_least, len(value) + 1)
        if value[start:end] in text
    ]


def dial_line(cut_at: int, value_before_cut: int) -> str:
    """A ``dial`` line whose quoted VALUE starts ``value_before_cut`` characters before column ``cut_at``."""
    head, glue = '    return dial(order, "', '", "'
    line = f'{head}{"x" * (cut_at - value_before_cut - len(head) - len(glue))}{glue}{VALUE}")'
    assert line.index(VALUE) == cut_at - value_before_cut
    return line


def entry_file(cut_at: int, value_before_cut: int) -> str:
    """Two functions; the first one's three-line preview, its stripped lines joined by spaces, holds
    VALUE starting ``value_before_cut`` characters before preview character ``cut_at``."""
    head, glue = 'return dial(order, "', '", "'
    joined = len("def connect(order): x = 1 ")
    third = f'{head}{"y" * (cut_at - value_before_cut - joined - len(head) - len(glue))}{glue}{VALUE}")'
    return f"def connect(order):\n    x = 1\n    {third}\n\n\ndef other(order):\n    return order\n"


CUT_SEARCHES: dict[str, Search] = {
    "find_code_line_cut": lambda tmp_path: _cut_find_code(tmp_path),
    "entry_preview_cut": lambda tmp_path: _cut_entry(tmp_path),
}


def _cut_find_code(tmp_path: Path) -> ScriptedJevClient:
    index = repository(
        tmp_path, {"shop/a_session.py": f"def connect(order):\n{dial_line(240, 10)}\n", **CONFIG}
    )
    client = ScriptedJevClient(default_noul=0.1)
    find_code(index, judge(client), TARGET, [function_place(index, connect(index))])
    return client


def _cut_entry(tmp_path: Path) -> ScriptedJevClient:
    files = {"shop/a_session.py": entry_file(360, 10), "deploy/app.env": f"DB_PASSWORD={VALUE}\n"}
    index = repository(tmp_path, files)
    client = ScriptedJevClient(default_noul=0.6)
    choose_initial_candidates(index, Judge(client), TARGET)
    return client


@pytest.mark.parametrize("search", CUT_SEARCHES.values(), ids=CUT_SEARCHES)
def test_a_cut_never_keeps_the_start_of_a_known_value(tmp_path: Path, search) -> None:
    # Arrange and act: a long line or preview is cut ten characters into VALUE
    client = search(tmp_path)

    # Assert
    assert client.requests
    assert pieces_of(VALUE, sent(client)) == []


LONG_VALUE = "".join(f"{n:03x}" for n in range(200))
TWO_LINES = f"{VALUE}\n{VALUE[::-1]}"


@pytest.mark.parametrize(
    ("value", "kept", "mark"),
    [
        (VALUE, 10, " [line cut]"),
        (VALUE, 10, "[... 99 characters cut]"),
        (LONG_VALUE, 214, " [line cut]"),
        (TWO_LINES, len(VALUE) + 1, LINES_CUT_MARK.format(kept=1, total=2)),
    ],
    ids=["line", "history", "long value", "slice"],
)
def test_a_known_value_split_before_a_cut_mark_is_hidden_in_the_request(
    value: str, kept: int, mark: str
) -> None:
    # Arrange: a cut that keeps the value's start, as a long line, a history section or a slice's first
    # lines are cut
    masker = KnownValuesMasker(DEFAULT_MASKER, frozenset({value}))
    state = {"items": [{"file": "shop/a.py", "code": f'return dial(order, """{value[:kept]}{mark}'}]}

    # Act
    masked, _, _ = mask_request(state, {}, masker)

    # Assert
    assert pieces_of(value, json.dumps(masked)) == []
    assert mark in masked["items"][0]["code"]


@pytest.mark.parametrize("placeholder", ["instructions", "examples", "described"])
def test_a_fixture_word_under_a_secret_key_never_changes_or_refuses_another_request(
    tmp_path: Path, placeholder: str
) -> None:
    # Arrange: a word JVN's own requests use as a key or a point uses, as a fixture's password
    plain = {f"shop/m{n}.py": f"def connect{n}(order):\n    return dial(order, {n})\n" for n in range(3)}
    fixture = {"tests/fixture_users.py": f'USER = {{"password": "{placeholder}"}}\n'}
    point = "the session code described in the README"
    clients = {"without": ScriptedJevClient(default_noul=0.1), "with": ScriptedJevClient(default_noul=0.1)}

    # Act
    for name, files in (("without", plain), ("with", {**plain, **fixture})):
        index = repository(tmp_path / name, files)
        find_all(index, judge(clients[name]), {"session": point}, files=list(plain), batches_per_wave=1)

    # Assert
    assert clients["with"].requests
    assert clients["with"].requests == clients["without"].requests


def test_two_known_values_sharing_characters_are_both_hidden_whole() -> None:
    # Arrange: equal lengths, so no order between them; they overlap in one copy
    first, second = "Zx81Qw7Lm3Pk", "Lm3Pk5Tr9Ny2"
    masker = KnownValuesMasker(DEFAULT_MASKER, frozenset({first, second}))
    state = {"items": [{"file": "shop/a.py", "code": "dial(order, 'Zx81Qw7Lm3Pk5Tr9Ny2')"}]}

    # Act
    masked, _, _ = mask_request(state, {}, masker)

    # Assert
    assert masked["items"][0]["code"] == "dial(order, '[MASKED]')"


@pytest.mark.parametrize("letters", ["QmvTxLrZpwKcHdNy", "qmvtxlrzpwkchdny"], ids=["mixed case", "lowercase"])
def test_a_letters_only_secret_is_hidden_across_requests(tmp_path: Path, letters: str) -> None:
    # Arrange: no digit, as a generated token or an app password may have
    files = {name: text.replace(VALUE, letters) for name, text in {**USE, **CONFIG}.items()}
    index = repository(tmp_path, files)
    client = ScriptedJevClient(default_noul=0.1)

    # Act
    find_all(index, judge(client), {"session": TARGET}, files=index.files, batches_per_wave=1)

    # Assert
    assert "dial(order" in sent(client)
    assert letters not in sent(client)
