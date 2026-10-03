"""Scope facts read from paths alone."""

from __future__ import annotations

import pytest

from jev_navigator.index.scope import is_test_file


@pytest.mark.parametrize(
    "path",
    [
        "tests/orders.py",
        "app/test/orders.py",
        "web/__tests__/routes.ts",
        "spec/routes.js",
        "app/test_orders.py",
        "app/orders_test.py",
        "web/routes.test.ts",
        "web/routes.spec.tsx",
        "conftest.py",
        "app/conftest.py",
    ],
)
def test_a_test_folder_or_test_file_name_marks_a_test_file(path: str) -> None:
    assert is_test_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "app/latest.py",
        "app/contest.py",
        "testing/helpers.py",
        "app/testdata.py",
        "app/attest_test_helpers/orders.py",
        "web/inspect.ts",
        "tests.py",
    ],
)
def test_a_name_that_only_contains_test_is_not_a_test_file(path: str) -> None:
    assert not is_test_file(path)
