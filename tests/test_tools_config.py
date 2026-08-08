"""Declarative tool registration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vichara.tools.config import (
    OutputTrust,
    RiskClass,
    ToolRegistryConfig,
    load_tool_registry,
)


class TestShippedRegistry:
    def test_loads(self, config_dir: Path) -> None:
        registry = load_tool_registry(config_dir)

        assert {t.name for t in registry.tools} == {
            "textbook_search",
            "web_search",
            "run_python",
            "workspace_file",
        }

    def test_no_tool_is_required(self, config_dir: Path) -> None:
        """Nothing may hard-depend on an external service being up.

        VidyaRAG in particular: this agent has to be buildable and evaluable
        before its sibling is deployed, so a required retrieval tool would
        block the whole project on someone else's uptime.
        """
        registry = load_tool_registry(config_dir)

        assert [t.name for t in registry.tools if t.required] == []

    def test_external_output_is_untrusted(self, config_dir: Path) -> None:
        """Anything originating outside the process is data, never instructions."""
        registry = load_tool_registry(config_dir)

        for name in ("textbook_search", "web_search"):
            spec = registry.by_name(name)
            assert spec is not None
            assert spec.output_trust is OutputTrust.UNTRUSTED

    def test_side_effecting_tools_need_approval(self, config_dir: Path) -> None:
        """Code execution and file writes route through the approval interrupt."""
        registry = load_tool_registry(config_dir)

        for name in ("run_python", "workspace_file"):
            spec = registry.by_name(name)
            assert spec is not None
            assert spec.risk is RiskClass.DESTRUCTIVE

    def test_read_only_tools_do_not(self, config_dir: Path) -> None:
        """Approving every retrieval would make the interrupt noise, not signal."""
        registry = load_tool_registry(config_dir)

        for name in ("textbook_search", "web_search"):
            spec = registry.by_name(name)
            assert spec is not None
            assert spec.risk is RiskClass.READ


class TestValidation:
    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate tool names: dup"):
            ToolRegistryConfig.model_validate(
                {
                    "tools": [
                        {"name": "dup", "summary": "one"},
                        {"name": "dup", "summary": "two"},
                    ]
                }
            )

    def test_enabled_tool_needs_a_summary(self) -> None:
        """The summary is the description the model selects on.

        An empty one does not degrade gracefully -- the tool becomes
        unselectable, which shows up in the eval as a reasoning failure rather
        than the configuration error it actually is.
        """
        with pytest.raises(ValidationError, match="no summary"):
            ToolRegistryConfig.model_validate({"tools": [{"name": "quiet", "summary": "  "}]})

    def test_disabled_tool_may_omit_summary(self) -> None:
        registry = ToolRegistryConfig.model_validate(
            {"tools": [{"name": "parked", "enabled": False}]}
        )

        assert registry.enabled == []
        assert registry.by_name("parked") is not None

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolRegistryConfig.model_validate(
                {"tools": [{"name": "x", "summary": "s", "tiemout": 5}]}
            )

    def test_missing_file_is_explicit(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="tool registry not found"):
            load_tool_registry(tmp_path)
