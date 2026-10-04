"""A small real repository (Python and TypeScript, two commits) that the index tests run against,
and every test's isolation from the developer's own decision-model settings."""

from __future__ import annotations

import os
import signal
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest
from git_repos import git, write_files
from isolated_jvn import NO_SETTINGS

from jev_navigator.cache_root import cache_root
from jev_navigator.data_root import data_root
from jev_navigator.environment import SETTING_PREFIXES
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.answers import JevResponse, NoulAnswer
from jev_navigator.judgments.client import InputBudgetExceededError
from jev_navigator.judgments.questions import serialized_chars


@pytest.fixture(autouse=True)
def no_developer_settings(monkeypatch):
    """`jvn` reads the checkout `.env`, `~/.config/jvn/env` and the exported environment on
    purpose, and any of them can name a live route with a real key, so a test would send its code
    to a paid service. Every test starts without them, and what a test loads into the environment
    is dropped when it ends. A test that runs `jvn` in a subprocess uses `isolated_jvn`."""
    monkeypatch.setattr("jev_navigator.environment.checkout_root", lambda: NO_SETTINGS)
    monkeypatch.setattr("jev_navigator.environment.LEGACY_CONFIG", NO_SETTINGS / "env")
    for name in list(os.environ):
        if name.startswith(SETTING_PREFIXES):
            monkeypatch.delenv(name)
    kept = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(kept)


ORDER_SERVICE = '''\
from app.validation import validate_order


class OrderService:
    def place(self, order):
        validate_order(order)
        return self.store.save(order)


def cancel(order_id):
    """Cancels an order that has not shipped."""
    order = load(order_id)
    return archive(order)
'''

VALIDATION = """\
LIMITS_KEY = "orders.max_items"


def validate_order(order):
    if not order.items:
        raise ValueError("empty order")
    return check_limits(order)


def check_limits(order):
    limit = read_setting("orders.max_items")
    return len(order.items) <= limit
"""

ROUTES = """\
import { handleOrder } from "./handlers";

export function registerRoutes(app) {
  app.post("/orders", (req, res) => handleOrder(req, res));
}
"""

HANDLERS = """\
export function handleOrder(req, res) {
  const order = parseOrder(req.body);
  return res.json(order);
}

export const parseOrder = (body) => JSON.parse(body);
"""

COMMENTS = """\
import functools

# Retries the payment twice before giving up.
@functools.cache
def charge(amount):
    return amount


# Normalises a currency code.

def normalise(code):
    return code.upper()


def total(items):
    subtotal = sum(items)  # adds every item price
    if subtotal > 100:  # large orders get a discount
        subtotal = subtotal * 0.9
        subtotal = round(subtotal, 2)

    return subtotal


# A basket of items.
class Basket:
    items = []
"""

SECRET_CONFIG = """\
API_TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
"""


@pytest.fixture
def python_sigint_handler():
    """Python's own Ctrl-C handler for a test that sends SIGINT. A suite started as a background job
    (``cmd &``) inherits SIGINT as ignored, so without this the signal never arrives."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write_files(
        root, {"app/__init__.py": "", "app/orders.py": ORDER_SERVICE, "app/validation.py": VALIDATION}
    )
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "orders and validation")
    write_files(
        root,
        {
            "web/routes.ts": ROUTES,
            "web/handlers.ts": HANDLERS,
            "app/settings.py": SECRET_CONFIG,
            "app/comments.py": COMMENTS,
            "app/validation.py": VALIDATION + "\n\ndef noop():\n    return None\n",
        },
    )
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "routes, and a validation change")
    write_files(
        root,
        {
            "app/orders.py": ORDER_SERVICE + "\n",
            "app/validation.py": VALIDATION + "\n\ndef noop():\n    return 1\n",
        },
    )
    git(root, "commit", "-qam", "orders and validation change together")
    return root


OUTER_CACHE_ROOT = cache_root()


@pytest.fixture
def outer_cache_root() -> Path:
    """The cache folder the suite's own environment names, before any test's private one replaces it."""
    return OUTER_CACHE_ROOT


@pytest.fixture(autouse=True)
def private_cache_root(
    no_developer_settings, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Each test starts with an empty cache folder of its own, holding its fact cache and its shared
    answer store, so no test reads what another run wrote, and no test, or jvn process a test
    starts, writes the user's caches. It runs after ``no_developer_settings`` has dropped every
    ``JEV_NAVIGATOR_`` variable, so the answer store variable is unset and the store lives here."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("cache")))
    return cache_root()


@pytest.fixture(autouse=True)
def private_data_root(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test starts with an empty data folder of its own, holding the run folders the CLI writes
    without ``--out``, so no test writes the user's run folders."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("data")))
    return data_root()


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """How many processes each tool started (``git`` counted by subcommand); every process still runs."""
    spawns: Counter[str] = Counter()

    class CountedPopen(subprocess.Popen):
        def __init__(self, arguments, *args, **kwargs) -> None:
            spawns[_tool_name(arguments)] += 1
            super().__init__(arguments, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", CountedPopen)
    return spawns


def _tool_name(arguments) -> str:
    if arguments[0] != "git":
        return arguments[0]
    return "git " + next(part for part in arguments[1:] if not part.startswith("-") and "=" not in part)


@pytest.fixture
def sample_index(sample_repo: Path) -> CodeIndex:
    return CodeIndex.from_git(sample_repo, fact_cache_dir=sample_repo.parent / "fact-cache")


class BudgetedClient:
    """A Jev client that refuses any request over a measured input budget, the way the real
    endpoint answered request 5 of the saved trace run: HTTP 400 ``max_tokens_exceeded``.

    ``budget`` bounds the whole body and ``input_box`` the state plus the longest single question,
    the way the provider measures its documented input limit. Both count characters of the
    ASCII-escaped serialization (``serialized_chars``), the one measure of the library and the Engine.

    It records the requests it accepted, so a test can prove no request over the budget was ever
    sent, and how many times the provider had to refuse one.
    """

    model = "jev-scripted"

    def __init__(self, budget: int, default_noul: float = 0.9, input_box: int | None = None) -> None:
        self.budget = budget
        self.input_box = input_box
        self.default_noul = default_noul
        self.requests: list[tuple[Mapping, Mapping]] = []
        self.refusals = 0

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        body = serialized_chars({"state": state, "questions": questions})
        box = self._state_and_longest_question(state, questions)
        if body > self.budget or (self.input_box is not None and box > self.input_box):
            self.refusals += 1
            raise InputBudgetExceededError(
                "TypeSafeBadRequestError: 400 "
                '{"detail":{"error_type":"max_tokens_exceeded"}} '
                f"(input of {body} bytes, {box} characters of state and question)"
            )
        self.requests.append((state, questions))
        answers = {question_id: NoulAnswer(self.default_noul) for question_id in questions}
        return JevResponse(answers, self.model, 100)

    @staticmethod
    def _state_and_longest_question(state: Mapping, questions: Mapping) -> int:
        longest = max((serialized_chars(question) for question in questions.values()), default=0)
        return serialized_chars(state) + longest
