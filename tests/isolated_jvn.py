"""`jvn` in a subprocess without the developer's settings files, which the test process's own
isolation (`conftest.no_developer_settings`) cannot reach."""

from __future__ import annotations

import sys
from pathlib import Path

# A path that cannot exist: `/dev/null` is a file, so nothing lies below it.
NO_SETTINGS = Path("/dev/null/no-jvn-settings")

JVN = [
    sys.executable,
    "-c",
    "import pathlib\n"
    "import jev_navigator.environment as settings\n"
    f"settings.checkout_root = lambda: pathlib.Path({str(NO_SETTINGS)!r})\n"
    f"settings.LEGACY_CONFIG = pathlib.Path({str(NO_SETTINGS / 'env')!r})\n"
    "from jev_navigator.cli import main\n"
    "raise SystemExit(main())",
]
