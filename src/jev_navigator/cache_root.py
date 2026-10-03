"""The one folder every JVN cache lives in."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """``$XDG_CACHE_HOME/jev-navigator``, or ``~/.cache/jev-navigator`` when the variable is unset."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "jev-navigator"
