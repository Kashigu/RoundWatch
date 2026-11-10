import os
from pathlib import Path

import pytest

from scripts.runtime_paths import PROJECT_ROOT, configured_path, load_dotenv, sumo_binary


def test_load_dotenv_preserves_environment_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nROUNDWATCH_TEST_DOTENV=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ROUNDWATCH_TEST_DOTENV", "from-environment")

    load_dotenv(env_file)

    assert os.environ["ROUNDWATCH_TEST_DOTENV"] == "from-environment"


def test_load_dotenv_rejects_malformed_entries(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("not-a-key-value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        load_dotenv(env_file)


def test_configured_path_resolves_relative_to_repository(monkeypatch):
    monkeypatch.setenv("ROUNDWATCH_TEST_PATH", "external/tool")

    assert configured_path("ROUNDWATCH_TEST_PATH") == PROJECT_ROOT / "external" / "tool"


def test_sumo_binary_uses_optional_override(monkeypatch, tmp_path):
    executable = tmp_path / "custom-sumo"
    monkeypatch.setenv("SUMO_BINARY", str(executable))

    assert sumo_binary() == executable
