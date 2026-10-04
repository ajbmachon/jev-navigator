"""The one folder JVN keeps its run folders in, and how a run folder JVN names itself is named."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from .cache_root import xdg_base

_STAMP = "%Y%m%dT%H%M%S%fZ"
_DEFAULT_NAME = re.compile(r".+-(\d{8}T\d{12}Z)")


def data_root() -> Path:
    """``$XDG_DATA_HOME/jev-navigator``, or ``~/.local/share/jev-navigator`` when the variable is unset
    or relative."""
    return xdg_base("XDG_DATA_HOME", Path.home() / ".local" / "share") / "jev-navigator"


def runs_root() -> Path:
    """Where a run writes its folder when the user names none with ``--out``."""
    return data_root() / "runs"


def default_run_folder(repository: Path, started: datetime) -> Path:
    """``<runs_root>/<repository name>-<UTC start stamp>``."""
    return runs_root() / f"{repository.name}-{started.astimezone(UTC).strftime(_STAMP)}"


def run_folder_started(name: str) -> datetime | None:
    """When the run in the folder ``name`` started, or None for a folder JVN did not name."""
    match = _DEFAULT_NAME.fullmatch(name)
    return datetime.strptime(match[1], _STAMP).replace(tzinfo=UTC) if match else None
