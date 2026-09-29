"""The environment chain: real environment wins, then the checkout `.env`, then the legacy config."""

from __future__ import annotations

import os

import pytest

from jev_navigator.environment import (
    TYPESAFE_SETTINGS,
    _env_file,
    load_typesafe_environment,
)


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in TYPESAFE_SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_the_real_environment_wins_over_every_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from-env-file\n")
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-shell")

    load_typesafe_environment(root=tmp_path)

    assert os.environ["TYPESAFE_API_KEY"] == "from-shell"


def test_the_checkout_env_file_fills_what_the_environment_lacks(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# config\n"
        "TYPESAFE_BASE_URL=https://drex.nace.ai\n"
        'TYPESAFE_DEFAULT_MODEL="drex-latest"\n'
        "export TYPESAFE_API_KEY=nace-key\n"
    )

    contributed = load_typesafe_environment(root=tmp_path)

    assert os.environ["TYPESAFE_BASE_URL"] == "https://drex.nace.ai"
    assert os.environ["TYPESAFE_DEFAULT_MODEL"] == "drex-latest"
    assert os.environ["TYPESAFE_API_KEY"] == "nace-key"
    assert contributed == {
        "TYPESAFE_BASE_URL": "https://drex.nace.ai",
        "TYPESAFE_DEFAULT_MODEL": "drex-latest",
        "TYPESAFE_API_KEY": "nace-key",
    }


def test_the_legacy_config_fills_what_both_left_open(tmp_path, monkeypatch):
    from jev_navigator.environment import LEGACY_CONFIG

    (tmp_path / ".env").write_text("TYPESAFE_BASE_URL=https://drex.nace.ai\n")
    monkeypatch.setattr("jev_navigator.environment.LEGACY_CONFIG",
                        _written(tmp_path, {"TYPESAFE_API_KEY": "legacy-key"}))

    load_typesafe_environment(root=tmp_path)

    assert os.environ["TYPESAFE_API_KEY"] == "legacy-key"
    assert os.environ["TYPESAFE_BASE_URL"] == "https://drex.nace.ai"
    assert LEGACY_CONFIG  # the real default stays importable


def test_a_missing_key_everywhere_raises_with_every_source_named(tmp_path, monkeypatch):
    monkeypatch.setattr("jev_navigator.environment.LEGACY_CONFIG", tmp_path / "absent-legacy-env")
    with pytest.raises(RuntimeError, match=r"\.env|jvn/env"):
        load_typesafe_environment(root=tmp_path)


def test_env_file_parsing_is_tolerant(tmp_path):
    path = _written(tmp_path, {"A": "plain", "B": "quoted"},
                    extra=["", "# comment", "no equals sign", "=novalue"])
    assert _env_file(path) == {"A": "plain", "B": "quoted", "": "novalue"}


def _written(tmp_path, values: dict[str, str], extra: list[str] | None = None):
    path = tmp_path / "legacy-env"
    path.write_text("\n".join([*(f"{k}={v}" for k, v in values.items()), *(extra or [])]) + "\n")
    return path
