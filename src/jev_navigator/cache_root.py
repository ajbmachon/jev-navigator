"""The one folder every JVN cache lives in."""

from __future__ import annotations

import os
from pathlib import Path

CACHE_HOME_VARIABLE = "JEV_NAVIGATOR_CACHE_HOME"


def cache_root() -> Path:
    """``$JEV_NAVIGATOR_CACHE_HOME`` itself, for a host that keeps each tenant's caches in their own
    folder; else ``$XDG_CACHE_HOME/jev-navigator``, or ``~/.cache/jev-navigator`` when that is unset.
    A relative path in either variable is ignored."""
    own = Path(os.environ.get(CACHE_HOME_VARIABLE, ""))
    if own.is_absolute():
        return own
    return xdg_base("XDG_CACHE_HOME", Path.home() / ".cache") / "jev-navigator"


def xdg_base(variable: str, default: Path) -> Path:
    """The folder an XDG base directory variable names, or ``default`` when it is unset or relative.
    The XDG specification ignores a relative path, which would otherwise put JVN's files inside
    whatever repository JVN runs in."""
    configured = Path(os.environ.get(variable, ""))
    return configured if configured.is_absolute() else default
