"""Environment resolution shared by every entry point: real environment, then `.env`, then the
legacy `~/.config/jvn/env`.

The lookup order is the family convention (`jvn`, `jvr`, comment-tool): a variable already set
in the process environment always wins; a checkout-root `.env` fills what is missing; the
legacy config file fills what is still missing. A file never overrides anything that is
already set, so shell exports and CI secrets keep precedence everywhere.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

TYPESAFE_SETTINGS = ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL")
LEGACY_CONFIG = Path.home() / ".config/jvn/env"


def checkout_root() -> Path:
    """The checkout root: the first directory above this file that holds a `pyproject.toml`."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def load_typesafe_environment(
    environment: MutableMapping[str, str] | None = None,
    root: Path | None = None,
    legacy: Path | None = None,
) -> dict[str, str]:
    """Fill `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL` from the
    checkout-root `.env`, then the legacy `~/.config/jvn/env`; return what the files contributed.

    Real environment variables win over both files. Raises when no source provides an API key.
    """
    environment = os.environ if environment is None else environment
    contributed: dict[str, str] = {}
    for source in ((root or checkout_root()) / ".env", legacy or LEGACY_CONFIG):
        for name, value in _env_file(source).items():
            if not environment.get(name, "").strip() and value:
                environment[name] = value
                contributed[name] = value
    if not environment.get("TYPESAFE_API_KEY", "").strip():
        raise RuntimeError(
            "TYPESAFE_API_KEY is unset: export it, or set it in the checkout's .env "
            f"(see .env.example) or {LEGACY_CONFIG}"
        )
    return contributed


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
