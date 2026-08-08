"""Shared fixtures.

The suite must pass with no credentials in the environment at all. That is not
just CI hygiene -- it is the property that proves the fixture backends work,
and therefore that the agent really does degrade instead of crashing when a
service is undeployed. A test that quietly passes because the developer
happened to have a key set is testing the wrong thing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from vichara.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_VARS = (
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "VIDYARAG_URL",
    "VICHARA_PROFILE",
    "SANDBOX_BACKEND",
    "WORKSPACE_ROOT",
    "CACHE_PATH",
    "CHECKPOINT_PATH",
    "LOG_LEVEL",
    "LOG_FORMAT",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real .env from leaking into tests.

    Clearing the environment variables is not sufficient on its own:
    pydantic-settings would still read the ``.env`` file next to the repo
    root. Disabling ``env_file`` for the duration of each test closes that
    second channel, so a bare ``Settings()`` in a test is genuinely
    unconfigured rather than configured by whatever the developer last used.
    """
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def config_dir() -> Path:
    """The repository's real config directory.

    Profiles are validated against the shipped configs on purpose: a broken
    profile should fail the test suite, not just runtime.
    """
    return REPO_ROOT / "config"


@pytest.fixture
def tmp_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point every writable path at a temporary directory.

    Depends on nothing from the developer's machine, and keeps `health` from
    creating session/cache directories in the working tree during a test run.
    """
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path / "cache" / "llm.sqlite"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "cache" / "checkpoints.sqlite"))
    yield tmp_path
