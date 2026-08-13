"""Everything the nodes need, assembled once.

LangGraph nodes take only state, so the tools, provider, recorder and config
travel in a context object bound at graph-build time. Passing them through
state instead would put non-serialisable objects in the checkpoint and break
resumption -- which is exactly the property Phase 3 exists to deliver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vichara.llm.provider import Provider
from vichara.settings import PipelineConfig, Settings
from vichara.tools.base import BaseTool
from vichara.tools.registry import Registry
from vichara.trajectory.recorder import TrajectoryRecorder

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


@dataclass
class AgentContext:
    """Bound once per run."""

    settings: Settings
    config: PipelineConfig
    registry: Registry
    provider: Provider
    recorder: TrajectoryRecorder
    workspace: Path
    auto_approve: bool = False
    """Evaluation supplies a policy auto-approver instead of a human. The
    interrupt path is still taken, evaluated, and recorded -- untested
    human-in-the-loop is decoration, so it runs on every eval task."""

    prompts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompts:
            self.prompts = {
                name: load_prompt(name)
                for name in ("system", "plan", "act", "reflect", "synthesize")
            }

    @property
    def tools(self) -> list[BaseTool]:
        return self.registry.tools

    def tool(self, name: str) -> BaseTool | None:
        return self.registry.get(name)

    @property
    def tool_names(self) -> list[str]:
        return self.registry.capability_profile
