"""Config-driven tool registration and health probing.

This is where "a missing tool shrinks the agent's capability rather than
crashing it" stops being a design statement and becomes code.

At startup every declared tool is constructed and probed. A tool that is
disabled, unhealthy, or unbuildable is dropped from the capability set and a
notice is generated telling the model, in words, what it can no longer do and
what to say instead. The surviving set is recorded on the trajectory as a
``capability_profile``, which is what lets the evaluation runner sweep
configurations -- all tools, no retrieval, no search -- and report accuracy per
profile rather than treating a missing service as an outage.

Construction failures are caught for the same reason as probe failures. A tool
whose constructor raises because a directory is unwritable should cost the
agent that one capability, not the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vichara.logging import get_logger
from vichara.settings import PipelineConfig, Settings
from vichara.tools.base import BaseTool, HealthStatus
from vichara.tools.config import ToolRegistryConfig, ToolSpec, load_tool_registry
from vichara.tools.files.tool import build_workspace_file_tool
from vichara.tools.rag.tool import build_textbook_tool
from vichara.tools.websearch.tool import build_web_search_tool

log = get_logger(__name__)

# What the model is told when a capability is absent. Phrased as an instruction
# about behaviour, not a status report: "retrieval is down" invites the model
# to work around it by inventing citations, whereas "say your answer is not
# textbook-grounded" tells it what an honest answer looks like without one.
_NOTICES = {
    "textbook_search": (
        "Textbook retrieval is unavailable in this session. Do not cite textbook "
        "pages. If a question needs textbook grounding, answer from what you can "
        "verify and state plainly that the answer is not textbook-grounded."
    ),
    "web_search": (
        "Web search is unavailable in this session. You cannot check for recent "
        "developments. If a question depends on current information, say so "
        "rather than answering from memory."
    ),
    "run_python": (
        "Code execution is unavailable in this session. Do computations by hand, "
        "show your working, and flag any result you could not verify."
    ),
    "workspace_file": (
        "File access is unavailable in this session. Return any output in your "
        "answer rather than offering to save it."
    ),
}


@dataclass(frozen=True, slots=True)
class ToolStatus:
    """The outcome of trying to bring one declared tool up."""

    spec: ToolSpec
    tool: BaseTool | None
    health: HealthStatus
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.tool is not None and self.health.healthy


@dataclass
class Registry:
    """The capability set for one session."""

    statuses: list[ToolStatus] = field(default_factory=list)

    @property
    def tools(self) -> list[BaseTool]:
        return [s.tool for s in self.statuses if s.available and s.tool is not None]

    @property
    def capability_profile(self) -> list[str]:
        """Recorded on every trajectory. The key an eval report groups by."""
        return sorted(s.spec.name for s in self.statuses if s.available)

    @property
    def missing(self) -> list[ToolStatus]:
        return [s for s in self.statuses if not s.available]

    @property
    def degraded(self) -> list[ToolStatus]:
        """Available, but not on the backend that was asked for."""
        return [s for s in self.statuses if s.available and s.health.degraded]

    def get(self, name: str) -> BaseTool | None:
        return next((s.tool for s in self.statuses if s.spec.name == name and s.available), None)

    def status(self, name: str) -> ToolStatus | None:
        return next((s for s in self.statuses if s.spec.name == name), None)

    def capability_notice(self) -> str:
        """The paragraph appended to the system prompt. Empty when nothing is missing.

        Deliberately covers tools that are *disabled in config* as well as
        those that failed to come up. Turning a tool off is precisely how the
        evaluation runner builds a degraded capability profile, and a notice
        that skipped those would leave the agent unaware of the very
        constraint the experiment is measuring.
        """
        notices = [_NOTICES[s.spec.name] for s in self.missing if s.spec.name in _NOTICES]
        if not notices:
            return ""
        return "Capability limits for this session:\n" + "\n".join(f"- {n}" for n in notices)


def build_registry(
    settings: Settings,
    config: PipelineConfig,
    *,
    session_id: str,
    config_dir: Path | None = None,
    registry_config: ToolRegistryConfig | None = None,
    prefer_recorded_search: bool = False,
) -> Registry:
    """Construct and probe every declared tool.

    Never raises for a tool-level problem. The only failure that propagates is
    a malformed ``tools.yaml``, which is a configuration error the operator
    must fix rather than a capability the agent can do without.
    """
    declared = registry_config or load_tool_registry(config_dir)
    statuses: list[ToolStatus] = []
    for spec in declared.tools:
        if not spec.enabled:
            statuses.append(
                ToolStatus(
                    spec=spec,
                    tool=None,
                    health=HealthStatus(healthy=False, backend="-", detail="disabled in config"),
                    reason="disabled",
                )
            )
            continue

        try:
            tool = _construct(
                spec,
                settings,
                config,
                session_id=session_id,
                prefer_recorded_search=prefer_recorded_search,
            )
        except Exception as exc:  # noqa: BLE001 - one broken tool must not stop the rest
            log.warning(
                "tool construction failed", tool=spec.name, error=f"{type(exc).__name__}: {exc}"
            )
            statuses.append(
                ToolStatus(
                    spec=spec,
                    tool=None,
                    health=HealthStatus(
                        healthy=False, backend="-", detail=f"{type(exc).__name__}: {exc}"
                    ),
                    reason="construction failed",
                )
            )
            continue

        if tool is None:
            statuses.append(
                ToolStatus(
                    spec=spec,
                    tool=None,
                    health=HealthStatus(healthy=False, backend="-", detail="not implemented yet"),
                    reason="not implemented",
                )
            )
            continue

        health = _probe(tool)
        statuses.append(
            ToolStatus(
                spec=spec,
                tool=tool if health.healthy else None,
                health=health,
                reason="" if health.healthy else "unhealthy",
            )
        )

    registry = Registry(statuses=statuses)
    log.info(
        "tool registry built",
        available=registry.capability_profile,
        missing=[s.spec.name for s in registry.missing],
        degraded=[s.spec.name for s in registry.degraded],
    )
    return registry


def _construct(
    spec: ToolSpec,
    settings: Settings,
    config: PipelineConfig,
    *,
    session_id: str,
    prefer_recorded_search: bool,
) -> BaseTool | None:
    """Build one tool. ``None`` means declared but not yet implemented."""
    timeout_s = config.tools.tool_timeout_s
    max_retries = config.tools.tool_max_retries
    max_output_bytes = config.tools.max_output_bytes

    if spec.name == "textbook_search":
        return build_textbook_tool(
            settings,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_output_bytes=max_output_bytes,
        )
    if spec.name == "web_search":
        return build_web_search_tool(
            settings,
            prefer_recorded=prefer_recorded_search,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_output_bytes=max_output_bytes,
        )
    if spec.name == "workspace_file":
        return build_workspace_file_tool(
            settings, session_id, timeout_s=timeout_s, max_output_bytes=max_output_bytes
        )
    if spec.name == "run_python":
        # Arrives in Phase 2 behind the Sandbox protocol. Declared now so the
        # capability set is honest about what is missing rather than silently
        # omitting it.
        return None
    log.warning("unknown tool declared", tool=spec.name)
    return None


def _probe(tool: BaseTool) -> HealthStatus:
    """Health probes must not raise, but a buggy one might."""
    try:
        return tool.health()
    except Exception as exc:  # noqa: BLE001 - a broken probe is a dead tool, not a dead run
        log.warning("health probe raised", tool=tool.name, error=f"{type(exc).__name__}: {exc}")
        return HealthStatus(
            healthy=False, backend="-", detail=f"probe raised {type(exc).__name__}: {exc}"
        )
