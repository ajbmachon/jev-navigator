"""One `jvn find` with ``--neighbours-per-kind 1`` under Drex's box, run in its own process by
``test_opened_code_size``, so a cut that never settles fails at the deadline instead of hanging
pytest. Prints what the run kept as one JSON object.

The opened function calls its own nested tail before ``audit``, whose preview is large, and is
committed on its own, so no file committed with it is a neighbour. Shown whole,
the tail is on screen and ``audit`` is the one callee; ``audit`` beside the whole function is too
large, so the function is cut. The cut leaves the tail's lines out, so the tail becomes a callee and
ranks first, and a cap of one keeps it in place of ``audit``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from conftest import BudgetedClient
from git_repos import commit_files
from isolated_jvn import NO_SETTINGS

import jev_navigator.environment as settings
from jev_navigator import cli
from jev_navigator.adapters.routes import DREX_INPUT_LIMITS
from jev_navigator.judgments.client import JEV_INPUT_LIMITS

LINES = 17_600
AUDIT = "def audit(order):\n" + "".join(f"    log('{letter * 220}')\n" for letter in "abcdefgh")
TOTALS = [f"    total_{line:05d} = order.amount * {line:05d}\n" for line in range(LINES)]
PLACE = (
    "def place(order):\n    place_tail(order)\n"
    + "".join(TOTALS[: LINES // len(TOTALS[0]) + 1])
    + "    audit(order)\n    def place_tail(order):\n        return order\n\n    return order\n"
)
OTHERS = "from app.audit import audit\n\n\ndef others(order):\n" + "".join(
    "    audit(order)\n" for _ in range(5)
)


class _DrexStandIn(BudgetedClient):
    """The provider behind the CLI: Drex's box, and nothing to release when the run ends."""

    def close(self) -> None:
        pass


def main(folder: Path) -> None:
    settings.checkout_root = lambda: NO_SETTINGS
    settings.LEGACY_CONFIG = NO_SETTINGS / "env"
    os.environ["TYPESAFE_API_KEY"] = "local-test-key"
    repository, output = folder / "repository", folder / "pack"
    commit_files(repository, {"app/audit.py": AUDIT, "app/other.py": OTHERS})
    commit_files(repository, {"app/orders.py": PLACE})
    client = _DrexStandIn(JEV_INPUT_LIMITS.request_chars, input_box=DREX_INPUT_LIMITS.box_chars)
    client.input_limits = DREX_INPUT_LIMITS
    cli.system_one_client = lambda environment: client
    arguments = ["find", "the order total", "--repo", str(repository), "--start", "app/orders.py:1"]
    limits = ["--out", str(output), "--max-steps", "1", "--beam-width", "1", "--neighbours-per-kind", "1"]
    with contextlib.redirect_stdout(sys.stderr):
        status = cli.main([*arguments, *limits])
    if status != 0:
        raise SystemExit(f"jvn find exited {status}")
    manifest = json.loads((output / "manifest.json").read_text())
    asked = [_asked(state) for state, _questions in client.requests]
    tail_line = PLACE.split("\n").index("    def place_tail(order):") + 1
    kept = {
        "refusals": client.refusals,
        "asked": asked,
        "function_lines": PLACE.count("\n"),
        "tail_line": tail_line,
    }
    print(json.dumps({**kept, "search": manifest["search"]}))


def _asked(state: dict) -> dict:
    """What one request showed: the shown lines, the code's last line and its neighbours' signatures."""
    shown = state["slice"]
    neighbours = [candidate["signature"] for candidate in state.get("candidates", [])]
    return {"lines": shown["lines"], "last_line": shown["code"].split("\n")[-1], "neighbours": neighbours}


if __name__ == "__main__":
    main(Path(sys.argv[1]))
