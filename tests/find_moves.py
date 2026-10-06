"""A small repository in which every one of find's neighbour moves lists a place, and one find search
over it that opens everything it can reach. ``fixtures/find_move_requests.json`` holds the requests that
search sent at 5600567e, before the moves became sources, written by ``frozen_requests``."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

from git_repos import git, write_files

from jev_navigator.directives.find_code import SearchBudget, find_code
from jev_navigator.directives.places import place_for_line
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient

ORDERS = """\
import os

from app import audit
from app.limits import MAX_ITEMS, check_limit

ORDER_LIMIT_KEY = "orders.max_items"


class OrderService:
    def place(self, order):
        check_limit(order)
        self.store.save(order)
        audit.record(order)
        return order


def cancel(order_id):
    return order_id


def register(handlers):
    handlers.append(cancel)
    return handlers


def submit(order):
    return OrderService().place(order)


TIMEOUT = os.environ.get("ORDER_TIMEOUT_SECONDS")
DEFAULT = submit({"items": [MAX_ITEMS]})
"""
LIMITS = """\
MAX_ITEMS = 4
SETTINGS_KEY = "orders.max_items"


def check_limit(order):
    if len(order.items) > MAX_ITEMS:
        raise ValueError("too many items")
    return order
"""
FILES = {
    "app/__init__.py": "",
    "app/orders.py": ORDERS,
    "app/limits.py": LIMITS,
    "app/audit.py": "def record(order):\n    return {'order': order}\n",
    "config/settings.yaml": "orders.max_items: 4\nORDER_TIMEOUT_SECONDS: 30\n",
    "web/website.ts": (
        "import prisma from '@/lib/prisma';\nimport * as format from './format';\n"
        "import * as labels from './labels';\n\n"
        "export async function renameWebsite(websiteId: string, name: string) {\n"
        "  const data = { name: format.title(name) };\n"
        "  return prisma.client.website.update({ where: { id: websiteId }, data });\n}\n"
    ),
    "web/format.ts": "export function title(name: string) {\n  return name.trim();\n}\n",
    "web/labels.ts": "export const RENAMED = 'renamed';\n",
    "web/stats.ts": (
        "import prisma from '@/lib/prisma';\n\n"
        "export async function countWebsites() {\n  return prisma.client.website.count();\n}\n"
    ),
    "prisma/schema.prisma": "model Website {\n  id   String @id\n  name String\n}\n",
    "tests/test_orders.py": (
        "from app.orders import cancel\n\n\ndef test_cancel_returns_the_id():\n    assert cancel(1) == 1\n"
    ),
}
FROZEN = Path(__file__).parent / "fixtures" / "find_move_requests.json"
STARTS = (("app/orders.py", 11), ("web/website.ts", 6))
TARGET = "the code that refunds an order"


def moves_index(root: Path) -> CodeIndex:
    """Each file in a commit of its own, then one commit changing orders.py and limits.py together, so
    those two are the only files committed together."""
    git(root, "init", "-q", "-b", "main")
    for file, text in FILES.items():
        write_files(root, {file: text})
        git(root, "add", file)
        git(root, "commit", "-qm", file)
    write_files(root, {"app/orders.py": ORDERS + "\n", "app/limits.py": LIMITS + "\n"})
    git(root, "commit", "-qam", "orders and limits change together")
    return CodeIndex.from_git(root)


def sent_requests(index: CodeIndex) -> list[Mapping]:
    """Each request a find search sends from ``OrderService.place`` and ``renameWebsite``, one place
    and one request at a time, as the opened lines, each neighbour's signature and the question ids.
    Nothing is ever found and every neighbour could hold the target, so the search opens every place
    it can reach."""
    client = ScriptedJevClient(default_noul=0.5, nouls={"contains_target": 0.05})
    judge = Judge(client, max_concurrency=1)
    starts = [place_for_line(index, file, line, "start") for file, line in STARTS]
    find_code(index, judge, TARGET, starts, budget=SearchBudget(beam_width=1))
    return [_request(state, questions) for state, questions in client.requests]


def _request(state: Mapping, questions: Mapping) -> dict:
    opened = state["slice"]
    return {
        "opened": f"{opened['file']}:{opened['lines']}",
        "candidates": [candidate["signature"] for candidate in state.get("candidates", ())],
        "questions": sorted(questions),
    }


def frozen_requests(root: Path) -> list[Mapping]:
    return sent_requests(moves_index(root))


if __name__ == "__main__":
    print(json.dumps(frozen_requests(Path(sys.argv[1])), indent=2))
