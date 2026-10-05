"""The one deadline for a search run in its own process: a search that never ends fails at it, and
its process is killed instead of hanging pytest."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence

import pytest

SEARCH_DEADLINE_SECONDS = 60


def searched_in_child(command: Sequence[str]) -> dict:
    """The JSON object ``command`` prints, or a test failure when it runs past the deadline or fails."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=SEARCH_DEADLINE_SECONDS, check=False
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the search did not finish within {SEARCH_DEADLINE_SECONDS} seconds")
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)
