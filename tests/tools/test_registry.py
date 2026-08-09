"""Config-driven registration and graceful degradation.

This is the file that turns "a missing tool shrinks the agent's capability
rather than crashing it" from a design statement into something the build
enforces. Every test here removes something and asserts the rest survives.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from vichara.settings import PipelineConfig, Settings
from vichara.tools.base import HealthStatus
from vichara.tools.config import ToolRegistryConfig
from vichara.tools.registry import build_registry


@pytest.fixture
def settings(tmp_env: Path) -> Settings:
    return Settings()


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig()


def _registry(settings: Settings, config: PipelineConfig, **kwargs: object) -> object:
    return build_registry(settings, config, session_id="sess-test", **kwargs)  # type: ignore[arg-type]


class TestDefaultCapabilitySet:
    def test_builds_with_no_credentials_at_all(
        self, settings: Settings, config: PipelineConfig
    ) -> None:
        """The headline property. A fresh clone with an empty .env still works."""
        registry = build_registry(settings, config, session_id="sess-test")

        assert "textbook_search" in registry.capability_profile
        assert "web_search" in registry.capability_profile
        assert "workspace_file" in registry.capability_profile

    def test_run_python_tracks_the_sandbox_runtime(
        self, settings: Settings, config: PipelineConfig
    ) -> None:
        """Available exactly when a runtime is, and honest either way.

        Node is present on a developer machine and in CI, absent on a stripped
        image. Both are supported; what must not happen is the tool appearing
        in the capability set while being unable to run anything.
        """
        registry = build_registry(settings, config, session_id="sess-test")
        status = registry.status("run_python")

        assert status is not None
        assert status.available == status.health.healthy
        if not status.available:
            assert "Do computations by hand" in registry.capability_notice()

    def test_degraded_backends_are_reported_separately(
        self, settings: Settings, config: PipelineConfig
    ) -> None:
        """Working, but not on the backend that was asked for."""
        registry = build_registry(settings, config, session_id="sess-test")

        degraded = {s.spec.name for s in registry.degraded}
        assert "textbook_search" in degraded


class TestDegradation:
    def test_disabled_tool_is_dropped(self, settings: Settings, config: PipelineConfig) -> None:
        declared = ToolRegistryConfig.model_validate(
            {
                "tools": [
                    {"name": "textbook_search", "summary": "s", "enabled": False},
                    {"name": "web_search", "summary": "s"},
                ]
            }
        )

        registry = build_registry(
            settings, config, session_id="sess-test", registry_config=declared
        )

        assert registry.capability_profile == ["web_search"]
        assert registry.get("textbook_search") is None

    def test_an_unhealthy_tool_does_not_break_the_others(
        self, settings: Settings, config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vichara.tools.rag.fixture import FixtureRetrievalBackend

        monkeypatch.setattr(
            FixtureRetrievalBackend,
            "health",
            lambda self: HealthStatus(healthy=False, backend="fixture", detail="corpus gone"),
        )

        registry = build_registry(settings, config, session_id="sess-test")

        assert "textbook_search" not in registry.capability_profile
        assert "web_search" in registry.capability_profile

    def test_a_probe_that_raises_is_treated_as_dead_not_fatal(
        self, settings: Settings, config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A buggy health probe costs one capability, not the whole run."""
        from vichara.tools.rag.fixture import FixtureRetrievalBackend

        def explode(self: object) -> HealthStatus:
            raise RuntimeError("probe bug")

        monkeypatch.setattr(FixtureRetrievalBackend, "health", explode)

        registry = build_registry(settings, config, session_id="sess-test")

        assert "textbook_search" not in registry.capability_profile
        assert "web_search" in registry.capability_profile

    def test_a_constructor_that_raises_is_contained(
        self, settings: Settings, config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "vichara.tools.registry.build_web_search_tool",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no disk")),
        )

        registry = build_registry(settings, config, session_id="sess-test")

        assert "web_search" not in registry.capability_profile
        assert "textbook_search" in registry.capability_profile
        status = registry.status("web_search")
        assert status is not None
        assert status.reason == "construction failed"

    def test_everything_can_fail_without_raising(
        self, settings: Settings, config: PipelineConfig
    ) -> None:
        """The extreme case: an agent with no tools at all still starts."""
        declared = ToolRegistryConfig.model_validate(
            {"tools": [{"name": "unknown_tool", "summary": "s"}]}
        )

        registry = build_registry(
            settings, config, session_id="sess-test", registry_config=declared
        )

        assert registry.capability_profile == []
        assert registry.tools == []


class TestCapabilityNotice:
    def test_absent_tools_produce_instructions_not_status(
        self, settings: Settings, config: PipelineConfig
    ) -> None:
        """The notice tells the model how to behave, not what is broken.

        "Retrieval is down" invites it to work around the gap by inventing a
        citation; "say your answer is not textbook-grounded" describes what an
        honest answer looks like without one.
        """
        declared = ToolRegistryConfig.model_validate(
            {
                "tools": [
                    {"name": "textbook_search", "summary": "s", "enabled": False},
                    {"name": "web_search", "summary": "s"},
                ]
            }
        )

        registry = build_registry(
            settings, config, session_id="sess-test", registry_config=declared
        )
        notice = registry.capability_notice()

        assert "not textbook-grounded" in notice
        assert "Do not cite textbook pages" in notice

    def test_no_notice_when_nothing_is_missing(
        self, settings: Settings, config: PipelineConfig
    ) -> None:
        declared = ToolRegistryConfig.model_validate(
            {"tools": [{"name": "web_search", "summary": "s"}]}
        )

        registry = build_registry(
            settings, config, session_id="sess-test", registry_config=declared
        )

        assert registry.capability_notice() == ""


class TestCapabilityProfile:
    def test_profile_is_sorted_and_stable(self, settings: Settings, config: PipelineConfig) -> None:
        """Recorded on every trajectory and used as an eval grouping key, so
        the ordering must not depend on declaration order."""
        first = build_registry(settings, config, session_id="a").capability_profile
        second = build_registry(settings, config, session_id="b").capability_profile

        assert first == second == sorted(first)

    @respx.mock
    def test_profile_reflects_the_live_service_when_it_is_up(
        self, settings: Settings, config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VIDYARAG_URL", "https://vidyarag.test")
        respx.get("https://vidyarag.test/health").mock(return_value=httpx.Response(200))

        registry = build_registry(Settings(), config, session_id="sess-test")
        status = registry.status("textbook_search")

        assert status is not None
        assert status.health.backend == "http"
        assert status.health.degraded is False


class TestToolWiring:
    def test_tools_carry_config_limits(self, settings: Settings, config: PipelineConfig) -> None:
        """Timeouts and output ceilings come from the profile, so an
        experiment can change them without touching code."""
        config.tools.tool_timeout_s = 7.0
        config.tools.max_output_bytes = 1234

        registry = build_registry(settings, config, session_id="sess-test")
        tool = registry.get("web_search")

        assert tool is not None
        assert tool.timeout_s == 7.0
        assert tool.max_output_bytes == 1234

    def test_tools_are_bindable_to_a_model(
        self, settings: Settings, config: PipelineConfig
    ) -> None:
        """Phase 3 binds these; a schema that will not adapt fails here first."""
        registry = build_registry(settings, config, session_id="sess-test")

        adapted = [t.as_langchain_tool() for t in registry.tools]

        assert {a.name for a in adapted} == set(registry.capability_profile)
        assert all(a.description for a in adapted)
