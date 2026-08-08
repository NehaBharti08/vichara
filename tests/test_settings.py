"""Configuration layer."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vichara.settings import (
    REPO_ROOT,
    PipelineConfig,
    SandboxBackend,
    Settings,
    load_pipeline_config,
)


class TestSettings:
    def test_loads_with_no_credentials(self) -> None:
        """The agent must start with nothing configured.

        This is the property the whole degradation story rests on: if Settings
        required a key, `vichara health` on a fresh clone would fail and the
        fixture backends could never be exercised.
        """
        settings = Settings()

        assert settings.profile == "baseline"
        assert settings.has_google_key is False
        assert settings.has_tavily_key is False
        assert settings.vidyarag_url is None
        assert settings.sandbox_backend is None

    def test_env_vars_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
        monkeypatch.setenv("VICHARA_PROFILE", "hardened")
        monkeypatch.setenv("SANDBOX_BACKEND", "docker")

        settings = Settings()

        assert settings.has_google_key is True
        assert settings.profile == "hardened"
        assert settings.sandbox_backend is SandboxBackend.DOCKER

    def test_secrets_do_not_render(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A key must not appear in repr or str.

        Settings ends up in log lines and exception context more often than
        anyone intends; SecretStr is what keeps that from becoming a leak.
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "super-secret-value")

        settings = Settings()

        assert "super-secret-value" not in repr(settings)
        assert "super-secret-value" not in str(settings)
        assert settings.google_api_key.get_secret_value() == "super-secret-value"

    @pytest.mark.parametrize("url", ["localhost:8000", "vidyarag.example.com", "ftp://host"])
    def test_rejects_url_without_scheme(self, monkeypatch: pytest.MonkeyPatch, url: str) -> None:
        """Catch it at startup, not at the first retrieval call."""
        monkeypatch.setenv("VIDYARAG_URL", url)

        with pytest.raises(ValidationError, match="http"):
            Settings()

    def test_blank_url_is_treated_as_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`VIDYARAG_URL=` in a .env should degrade, not explode."""
        monkeypatch.setenv("VIDYARAG_URL", "   ")

        assert Settings().vidyarag_url is None

    def test_accepts_valid_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIDYARAG_URL", "https://example.hf.space")

        assert Settings().vidyarag_url == "https://example.hf.space"

    def test_resolved_path_handling(self, tmp_path: Path) -> None:
        settings = Settings()

        assert settings.resolved("sessions") == REPO_ROOT / "sessions"
        assert settings.resolved(str(tmp_path)) == tmp_path


class TestPipelineConfig:
    def test_baseline_profile_loads(self, config_dir: Path) -> None:
        cfg = load_pipeline_config("baseline", config_dir)

        assert cfg.name == "baseline"
        assert cfg.sandbox.backend is SandboxBackend.PYODIDE

    def test_default_yaml_matches_code_defaults(self, config_dir: Path) -> None:
        """default.yaml restates the pydantic defaults; keep them in step.

        The duplication is deliberate -- a profile is only a useful description
        of a variant if the whole configuration is readable in one file. This
        test is the cost of that decision: without it the two drift, and a
        reader trusts a YAML value that the code silently overrides.
        """
        from_yaml = load_pipeline_config("baseline", config_dir)
        from_code = PipelineConfig()

        ignored = {"name", "description"}
        assert from_yaml.model_dump(exclude=ignored) == from_code.model_dump(exclude=ignored)

    def test_unknown_profile_lists_available(self, config_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="baseline"):
            load_pipeline_config("does-not-exist", config_dir)

    def test_typo_in_profile_is_a_startup_error(self, tmp_path: Path) -> None:
        """`extra: forbid` turns a silent misconfiguration into a crash.

        A profile key that is ignored rather than rejected is the worst
        possible failure for a benchmark: the run completes, the number looks
        plausible, and it describes a configuration nobody chose.
        """
        (tmp_path / "profiles").mkdir()
        (tmp_path / "default.yaml").write_text("name: base\n", encoding="utf-8")
        (tmp_path / "profiles" / "typo.yaml").write_text(
            "budget:\n  max_stpes: 40\n", encoding="utf-8"
        )

        with pytest.raises(ValidationError, match="max_stpes"):
            load_pipeline_config("typo", tmp_path)

    def test_profile_overlay_is_a_deep_merge(self, tmp_path: Path) -> None:
        """Overriding one nested key must not wipe its siblings."""
        (tmp_path / "profiles").mkdir()
        (tmp_path / "default.yaml").write_text(
            "budget:\n  max_steps: 12\n  max_llm_requests: 25\n", encoding="utf-8"
        )
        (tmp_path / "profiles" / "tight.yaml").write_text(
            "budget:\n  max_steps: 6\n", encoding="utf-8"
        )

        cfg = load_pipeline_config("tight", tmp_path)

        assert cfg.budget.max_steps == 6
        assert cfg.budget.max_llm_requests == 25

    def test_tool_limits_fall_back_to_default(self) -> None:
        limits = PipelineConfig().tool_limits

        assert limits.limit_for("run_python") == 2
        assert limits.limit_for("some_future_tool") == limits.default


class TestGuardrailInvariants:
    """Thresholds whose value is an argument, not a preference.

    These are pinned because the reasoning behind them is written down in the
    plan and in docs/EVALUATION.md. Changing one should require editing a test
    and therefore noticing the claim it contradicts.
    """

    def test_identical_action_threshold_is_two(self) -> None:
        """One exact repeat is already a bug: nothing changed, so nothing will."""
        assert PipelineConfig().loops.identical_threshold == 2

    def test_provenance_survives_compression_by_default(self) -> None:
        """Otherwise summarisation launders untrusted content into narration."""
        assert PipelineConfig().memory.preserve_provenance_tags is True

    def test_baseline_ships_no_injection_defences(self, config_dir: Path) -> None:
        """The control must not defend, or the baseline attack rate is unmeasurable."""
        injection = load_pipeline_config("baseline", config_dir).injection

        assert injection.neutralise_delimiters is False
        assert injection.strip_imperatives is False
        assert injection.detector_enabled is False
        # Instrumentation, not a defence -- needed to attribute a compromise.
        assert injection.tag_provenance is True

    def test_sandbox_network_is_off(self) -> None:
        assert PipelineConfig().sandbox.network is False
