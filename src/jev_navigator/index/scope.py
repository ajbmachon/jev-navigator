"""Which files a search covers, decided from paths, never by parsing."""

from __future__ import annotations

import re

TEST_FOLDERS = frozenset({"test", "tests", "__tests__", "spec"})
_TEST_FILE_NAME = re.compile(r"^test_|_test\.|\.test\.|\.spec\.|^conftest\.py$")


def is_test_file(path: str) -> bool:
    """A file in a test folder at any depth, or named like a test (``test_*``, ``*_test.*``,
    ``*.test.*``, ``*.spec.*``, ``conftest.py``)."""
    folders, _, name = path.rpartition("/")
    return not TEST_FOLDERS.isdisjoint(folders.split("/")) or bool(_TEST_FILE_NAME.search(name))
