"""A small real repository (Python and TypeScript, two commits) that the index tests run against,
and every test's isolation from the developer's own decision-model settings."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from git_repos import git, write_files
from isolated_jvn import NO_SETTINGS

from jev_navigator.adapters.registry import ADAPTERS
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.answers import JevResponse, NoulAnswer
from jev_navigator.judgments.client import InputBudgetExceededError

# Variables that can point `jvn` at a live service with a real key.
SERVICE_SETTINGS = ("SYSTEM_ONE_", "TYPESAFE_", *(adapter.api_key_env for adapter in ADAPTERS.values()))


@pytest.fixture(autouse=True)
def no_developer_settings(monkeypatch):
    """`jvn` reads the checkout `.env`, `~/.config/jvn/env` and the exported environment on
    purpose, and any of them can name a live route with a real key, so a test would send its code
    to a paid service. Every test starts without them, and what a test loads into the environment
    is dropped when it ends. A test that runs `jvn` in a subprocess uses `isolated_jvn`."""
    monkeypatch.setattr("jev_navigator.environment.checkout_root", lambda: NO_SETTINGS)
    monkeypatch.setattr("jev_navigator.environment.LEGACY_CONFIG", NO_SETTINGS / "env")
    for name in list(os.environ):
        if name.startswith(SERVICE_SETTINGS):
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


@pytest.fixture
def sample_index(sample_repo: Path) -> CodeIndex:
    return CodeIndex.from_git(sample_repo, fact_cache_dir=sample_repo.parent / "fact-cache")


class BudgetedClient:
    """A Jev client that refuses any request over a measured input budget, the way the real
    endpoint answered request 5 of the saved trace run: HTTP 400 ``max_tokens_exceeded``.

    It records the requests it accepted, so a test can prove no request over the budget was ever
    sent, and how many times the provider had to refuse one.
    """

    model = "jev-scripted"

    def __init__(self, budget: int, default_noul: float = 0.9) -> None:
        self.budget = budget
        self.default_noul = default_noul
        self.requests: list[tuple[Mapping, Mapping]] = []
        self.refusals = 0

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        body = len(json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode())
        if body > self.budget:
            self.refusals += 1
            raise InputBudgetExceededError(
                "TypeSafeBadRequestError: 400 "
                '{"detail":{"error_type":"max_tokens_exceeded"}} '
                f"(input of {body} bytes over the {self.budget}-byte budget)"
            )
        self.requests.append((state, questions))
        answers = {question_id: NoulAnswer(self.default_noul) for question_id in questions}
        return JevResponse(answers, self.model, 100)
