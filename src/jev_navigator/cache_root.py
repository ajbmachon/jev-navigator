"""The one folder every JVN cache lives in."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """``$XDG_CACHE_HOME/jev-navigator``, or ``~/.cache/jev-navigator`` when the variable is unset or
    relative."""
    return xdg_base("XDG_CACHE_HOME", Path.home() / ".cache") / "jev-navigator"


def xdg_base(variable: str, default: Path) -> Path:
    """The folder an XDG base directory variable names, or ``default`` when it is unset or relative.
    The XDG specification ignores a relative path, which would otherwise put JVN's files inside
    whatever repository JVN runs in."""
    configured = Path(os.environ.get(variable, ""))
    return configured if configured.is_absolute() else default
