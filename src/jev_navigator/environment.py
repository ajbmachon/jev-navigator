"""Environment resolution shared by every entry point: real environment, then `.env`, then the
legacy `~/.config/jvn/env`.

The lookup order is the family convention (`jvn`, `jvr`, comment-tool): a variable already set
in the process environment always wins; the tool's own checkout `.env` fills what is missing; the
legacy config file fills what is still missing. A file never overrides anything that is
already set, so shell exports and CI secrets keep precedence everywhere.

Two rules keep an untrusted repository from configuring the tool through a `.env` it ships:

- The `.env` is read only from this tool's own source checkout (`checkout_root`), never from the
  directory a search happens to run in. So a repository under analysis cannot point the API key at
  another host with its own `.env`, and an installed `jvn` that is not run from a checkout takes no
  `.env` at all — it uses the real environment and `~/.config/jvn/env`.
- A file may set only the tool's own recognised settings (`TYPESAFE_*`, `JEV_NAVIGATOR_*`,
  `SYSTEM_ONE_*`, `DREX_*`); every other name is ignored. So a config file cannot inject an unrelated variable
  such as `PATH`, `LD_PRELOAD` or `RIPGREP_CONFIG_PATH` into the tool or into the `rg`, `git` and
  `ast-grep` subprocesses it runs.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

from .adapters.routes import covers_every_question

TYPESAFE_SETTINGS = ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL")
# The tool's own settings namespace. Only names under these prefixes are honoured from a file, so a
# file can never inject an operating-system or subprocess variable. `JEV_NAVIGATOR_*` holds the
# search thresholds and budget; `SYSTEM_ONE_*` and `DREX_*` name decision-model routes and their
# keys where that feature is present.
SETTING_PREFIXES = ("TYPESAFE_", "JEV_NAVIGATOR_", "SYSTEM_ONE_", "DREX_")
LEGACY_CONFIG = Path.home() / ".config/jvn/env"


def checkout_root() -> Path | None:
    """This tool's own source checkout — the first directory above this file that holds the
    `jev-navigator` `pyproject.toml` — or None when `jvn` is installed outside such a checkout.

    The `.env` is trusted only from here. Returning None rather than the current directory is what
    stops an arbitrary working directory's `.env` from configuring the tool.
    """
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file() and _names_this_project(pyproject):
            return parent
    return None


def load_typesafe_environment(
    environment: MutableMapping[str, str] | None = None,
    root: Path | None = None,
    legacy: Path | None = None,
) -> dict[str, str]:
    """Fill `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL` from the tool's
    checkout `.env`, then the legacy `~/.config/jvn/env`; return what the files contributed.

    Real environment variables win over both files, and a file may set only the tool's own
    settings (`SETTING_PREFIXES`). Raises when no source provides an API key, unless the route
    tables leave no question type to Jev: each route resolves its own key. ``root`` overrides the
    checkout the `.env` is read from; passing it opts into reading that directory's `.env`.
    """
    environment = os.environ if environment is None else environment
    root = checkout_root() if root is None else root
    contributed: dict[str, str] = {}
    sources = ([root / ".env"] if root is not None else []) + [legacy or LEGACY_CONFIG]
    for source in sources:
        for name, value in _env_file(source).items():
            if not _is_setting(name):
                continue
            if not environment.get(name, "").strip() and value:
                environment[name] = value
                contributed[name] = value
    if not environment.get("TYPESAFE_API_KEY", "").strip() and not covers_every_question(environment):
        raise RuntimeError(
            "TYPESAFE_API_KEY is unset: export it, or set it in the checkout's .env "
            f"(see .env.example) or {LEGACY_CONFIG}"
        )
    return contributed


def _is_setting(name: str) -> bool:
    return name.startswith(SETTING_PREFIXES)


def _names_this_project(pyproject: Path) -> bool:
    """Whether ``pyproject`` is jev-navigator's own, so a parent project's `pyproject.toml` (and
    its `.env`) higher up the tree is never taken for the tool's checkout."""
    try:
        text = pyproject.read_text()
    except OSError:
        return False
    return 'name = "jev-navigator"' in text or "name = 'jev-navigator'" in text


def _env_file(path: Path) -> dict[str, str]:
    """`KEY=value` lines; `#` comments, `export ` prefixes and quotes handled; nothing overrides."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        values[name.strip()] = value.strip().strip("'\"")
    return values
