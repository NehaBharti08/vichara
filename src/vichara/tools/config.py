"""Declarative tool registration.

This module holds the *declaration* only -- what tools exist, whether they are
required, and what their output may be trusted for. Instantiation and health
probing live in ``tools/registry.py``.

The separation exists so that the capability set can be inspected without
importing an HTTP client, a Node subprocess, or anything else with a startup
cost: ``vichara tools`` answers "what could this agent do" from a YAML file
alone, which is also what makes the eval runner's capability-profile sweep
cheap to set up.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vichara.settings import CONFIG_DIR


class RiskClass(enum.StrEnum):
    """What approval a call to this tool needs."""

    READ = "read"
    """Observes without changing anything reachable outside the process."""

    DESTRUCTIVE = "destructive"
    """Writes to disk or executes agent-authored code. Routed through the
    approval interrupt in the graph before it is allowed to run."""


class OutputTrust(enum.StrEnum):
    """How the model is allowed to treat what this tool returns."""

    TRUSTED = "trusted"
    """Produced entirely within this process from values the agent supplied."""

    UNTRUSTED = "untrusted"
    """Originated outside the process. Data, never instructions. Wrapped in a
    provenance envelope, and that envelope survives summarisation -- a digest
    that drops the marker re-emits the payload as trusted narration."""


class ToolSpec(BaseModel):
    """One registered tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    required: bool = False
    """When false -- the default, and the only value used today -- an unhealthy
    tool shrinks the capability set instead of failing startup."""

    backend: str = "auto"
    risk: RiskClass = RiskClass.READ
    output_trust: OutputTrust = OutputTrust.UNTRUSTED
    summary: str = ""

    @model_validator(mode="after")
    def _check_summary(self) -> ToolSpec:
        """An enabled tool with no summary is invisible to the model.

        The summary becomes the tool description the model selects on, so an
        empty one does not degrade tool choice gracefully -- it makes the tool
        unusable in a way that looks like a reasoning failure in the eval.
        """
        if self.enabled and not self.summary.strip():
            raise ValueError(f"tool {self.name!r} is enabled but has no summary")
        return self


class ToolRegistryConfig(BaseModel):
    """The full declared tool set."""

    model_config = ConfigDict(extra="forbid")

    tools: list[ToolSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique(self) -> ToolRegistryConfig:
        names = [t.name for t in self.tools]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate tool names: {', '.join(duplicates)}")
        return self

    @property
    def enabled(self) -> list[ToolSpec]:
        return [t for t in self.tools if t.enabled]

    def by_name(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)


def load_tool_registry(config_dir: Path | None = None) -> ToolRegistryConfig:
    """Load ``config/tools.yaml``."""
    path = (config_dir or CONFIG_DIR) / "tools.yaml"
    if not path.exists():
        raise FileNotFoundError(f"tool registry not found: {path}")
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return ToolRegistryConfig.model_validate(loaded)
