"""Command line surface.

`health` is a container healthcheck and a CI smoke test, so its exit code is
load-bearing: it must be 0 for every supported state, including states with no
credentials at all, and non-zero only when something is genuinely broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vichara import __version__
from vichara.cli import app

runner = CliRunner()


class TestVersion:
    def test_prints_version(self) -> None:
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestConfigCommand:
    def test_dumps_resolved_profile(self) -> None:
        result = runner.invoke(app, ["config", "--profile", "baseline"])

        assert result.exit_code == 0
        assert "baseline" in result.stdout

    def test_unknown_profile_fails(self) -> None:
        result = runner.invoke(app, ["config", "--profile", "nope"])

        assert result.exit_code != 0


class TestToolsCommand:
    def test_lists_tools_without_credentials(self) -> None:
        """Inspecting the capability set must not need a key or a network."""
        result = runner.invoke(app, ["tools"])

        assert result.exit_code == 0
        for name in ("textbook_search", "web_search", "run_python", "workspace_file"):
            assert name in result.stdout

    def test_shows_fixture_backends_when_undeployed(self) -> None:
        result = runner.invoke(app, ["tools"])

        assert "fixture" in result.stdout


class TestHealthCommand:
    def test_healthy_with_no_credentials(self, tmp_env: Path) -> None:
        """The headline Phase 0 property: a fresh clone is healthy.

        Missing credentials are a degraded state, not a failure. Reporting them
        as errors would make the command something you learn to ignore.
        """
        result = runner.invoke(app, ["health"])

        assert result.exit_code == 0, result.stdout
        assert "Healthy" in result.stdout
        assert "degraded" in result.stdout

    def test_creates_writable_paths(self, tmp_env: Path) -> None:
        result = runner.invoke(app, ["health"])

        assert result.exit_code == 0
        assert (tmp_env / "sessions").is_dir()
        assert (tmp_env / "cache").is_dir()

    def test_leaves_no_probe_files(self, tmp_env: Path) -> None:
        """Probing must not leave litter behind.

        Health builds the real registry, so the workspace tool's probe does
        create one session directory -- that is how it verifies the path is
        writable. It uses a fixed name so repeated invocations reuse it rather
        than accumulating one directory per run, and it must leave no files.
        """
        runner.invoke(app, ["health"])
        runner.invoke(app, ["health"])

        sessions = sorted(p.name for p in (tmp_env / "sessions").iterdir())
        assert sessions == ["cli-probe"], "probing must not accumulate directories"
        assert list((tmp_env / "sessions" / "cli-probe").iterdir()) == []
        assert not list(tmp_env.rglob(".vichara-write-probe"))

    def test_reports_provider_when_key_present(
        self, tmp_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 0
        assert "google" in result.stdout

    def test_does_not_print_the_key(self, tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "super-secret-value")

        result = runner.invoke(app, ["health"])

        assert "super-secret-value" not in result.stdout

    def test_unwritable_path_is_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A path the agent cannot write is genuinely fatal: no trajectory can
        be recorded and no run can be checkpointed.

        The failure modelled here is the realistic one -- a configured path
        that points *through* an existing file, which is what a typo in
        WORKSPACE_ROOT usually produces.
        """
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        monkeypatch.setenv("WORKSPACE_ROOT", str(blocker / "sessions"))

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
        assert "fail" in result.stdout

    def test_broken_config_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VICHARA_PROFILE", "does-not-exist")

        result = runner.invoke(app, ["health"])

        assert result.exit_code == 1
