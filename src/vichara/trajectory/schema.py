"""The trajectory record.

This is the artifact Phase 4 measures, Phase 5 audits, and Phase 6 renders. It
is written for every run, not only successful ones -- a run that hit the step
ceiling is exactly the run a step-efficiency metric needs to see.

Two things it must capture that are easy to omit and impossible to recover:

* **The prompt content hash.** A trajectory whose prompts have since been
  edited describes an agent that no longer exists. Comparing last week's
  numbers against this week's is only meaningful if you can tell whether the
  prompts changed, and the only reliable way is to hash them at the time.
* **Provenance on every observation.** Which tool result was untrusted, and
  whether an injection was detected in it. Phase 5 cannot attribute a
  compromised trajectory without it, and it cannot be reconstructed later.

Versioned, because the eval runner reads records written by older code and a
silently changed shape is a silently changed result.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class TerminalReason(enum.StrEnum):
    """How a run ended. Read directly by the Phase 4 refusal metric, so every
    exit path must set one -- an unset value is a measurement hole."""

    ANSWERED = "answered"
    REFUSED = "refused"
    """Correctly declined: impossible, out of scope, or needing an absent tool."""

    CLARIFY = "clarify"
    """Stopped to ask a question rather than guessing at an ambiguous task."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    STEP_CEILING = "step_ceiling"
    LOOP_DETECTED = "loop_detected"
    FATAL_ERROR = "fatal_error"


class StepKind(enum.StrEnum):
    PLAN = "plan"
    ACT = "act"
    GUARD = "guard"
    APPROVE = "approve"
    EXECUTE = "execute"
    OBSERVE = "observe"
    REFLECT = "reflect"
    COMPRESS = "compress"
    SYNTHESIZE = "synthesize"


class PlanStep(BaseModel):
    """One intended step. Typed rather than prose so Phase 4 can compare the
    annotated optimal path, the agent's plan, and what it actually did as
    three separate objects."""

    model_config = ConfigDict(extra="forbid")

    intent: str
    tool: str | None = None
    success_criterion: str = ""


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    answerable: bool = True
    needs_clarification: bool = False
    clarifying_question: str | None = None
    revision: int = 0


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ObservationRecord(BaseModel):
    """What a tool returned, with the provenance that must survive summarisation."""

    model_config = ConfigDict(extra="forbid")

    step: int
    tool: str
    backend: str = ""
    ok: bool = True
    content: str = ""
    trust: Literal["trusted", "untrusted"] = "untrusted"
    error_code: str | None = None
    retryable: bool = False
    latency_ms: float = 0.0
    truncated: bool = False
    raw_bytes: int = 0
    citations: list[dict[str, Any]] = Field(default_factory=list)
    injection_flagged: bool = False
    """Set by the Phase 5 detector. Present from Phase 3 so that baseline
    trajectories -- recorded before any defence exists -- have the field and
    can be compared against hardened ones without a schema migration."""

    externalised_ref: str | None = None
    """Set when the body was spilled to the workspace and replaced by a pointer."""


class GuardrailEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    rule: str
    action: Literal["allow", "block", "require_approval", "warn"]
    detail: str = ""


class StepRecord(BaseModel):
    """One node execution."""

    model_config = ConfigDict(extra="forbid")

    index: int
    kind: StepKind
    started_at: str
    duration_ms: float = 0.0
    thought: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    observations: list[ObservationRecord] = Field(default_factory=list)
    llm_requests: int = 0
    tokens: int = 0
    est_usd: float = 0.0
    note: str = ""


class TrajectoryRecord(BaseModel):
    """One complete run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    session_id: str
    task: str
    task_id: str | None = None
    seed: int | None = None

    profile: str = "baseline"
    capability_profile: list[str] = Field(default_factory=list)
    """Which tools were actually available. The key an eval report groups by,
    so that a degraded run is a comparable data point rather than an outage."""

    models: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    """Content hash per prompt file. Without it, a trajectory recorded before a
    prompt edit is indistinguishable from one recorded after."""

    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None

    plan: Plan | None = None
    plan_revisions: int = 0
    steps: list[StepRecord] = Field(default_factory=list)
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)

    final_answer: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    citations_fabricated: list[str] = Field(default_factory=list)
    """Citation-shaped spans in the answer that no tool produced. Recorded even
    when nothing is removed, so a baseline run shows how often the agent
    invents a source."""
    terminal_reason: TerminalReason | None = None

    llm_requests: int = 0
    cache_hits: int = 0
    total_tokens: int = 0
    est_usd: float = 0.0
    wall_clock_s: float = 0.0

    @property
    def tool_calls_made(self) -> list[str]:
        """Every tool invoked, in order. The input to tool-selection precision."""
        return [call.tool for step in self.steps for call in step.tool_calls]

    @property
    def distinct_tools(self) -> set[str]:
        return set(self.tool_calls_made)

    @property
    def agent_steps(self) -> int:
        """Tool calls made. The unit step efficiency is measured in.

        This counted graph nodes -- act, execute and reflect -- until the
        arithmetic was checked against a gold task. The optimal path is
        annotated as a sequence of *tool calls*, so dividing it by a node count
        compares two different units: a flawless single-tool run executes
        plan, act, execute, synthesize and scored 0.333 for doing exactly the
        right thing. The reported median of 0.333 was that artifact, not
        agent waste.

        Counting tool calls makes numerator and denominator the same unit, so
        1.0 means "took the annotated optimal path" and anything below it is
        real waste. Node count is still available as `graph_steps` for cost
        analysis, where it is the right measure.
        """
        return len(self.tool_calls_made)

    @property
    def graph_steps(self) -> int:
        """Nodes executed that consumed budget.

        The right measure for cost, and the wrong one for efficiency.
        Bookkeeping nodes -- compression, guard checks -- are excluded: charging
        the agent for the memory manager's work would measure the framework.
        """
        return sum(
            1
            for step in self.steps
            if step.kind in (StepKind.ACT, StepKind.EXECUTE, StepKind.REFLECT)
        )
