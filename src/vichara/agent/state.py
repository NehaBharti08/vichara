"""Graph state.

A ``TypedDict`` rather than a pydantic model because LangGraph merges partial
updates from each node, and pydantic validation on every partial merge costs
more than it catches. The typed records that *are* validated live in
:mod:`vichara.trajectory.schema`; this is the working state that produces them.

``terminal_reason`` is the load-bearing field. Phase 4's refusal metric reads
it directly, so every path out of the graph must set one -- an unset value is
not a neutral default, it is a measurement hole.
"""

from __future__ import annotations

import hashlib
import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from vichara.trajectory.schema import (
    GuardrailEvent,
    ObservationRecord,
    Plan,
    TerminalReason,
)


class ActionFingerprint(BaseModel):
    """A hash of one tool call, for repeat detection.

    Arguments are normalised before hashing -- lowercased, whitespace
    collapsed -- so that a query differing only in capitalisation counts as
    the same action. An agent that "varies" its query by changing the case is
    looping, not exploring.
    """

    model_config = ConfigDict(extra="forbid")

    step: int
    tool: str
    digest: str
    normalised: str

    @classmethod
    def of(cls, step: int, tool: str, args: dict[str, Any]) -> ActionFingerprint:
        normalised = json.dumps(
            {k: _normalise(v) for k, v in sorted(args.items())},
            sort_keys=True,
            ensure_ascii=False,
        )
        material = f"{tool}|{normalised}"
        return cls(
            step=step,
            tool=tool,
            digest=hashlib.sha256(material.encode("utf-8")).hexdigest()[:16],
            normalised=normalised,
        )


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in sorted(value.items())}
    return value


class PendingAction(BaseModel):
    """The tool call the guard is deciding on, and approval may block."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    risk: str = "read"


class BudgetState(BaseModel):
    """Consumption so far, checked against the profile's ceilings."""

    model_config = ConfigDict(extra="forbid")

    llm_requests: int = 0
    tokens: int = 0
    est_usd: float = 0.0
    wall_clock_s: float = 0.0


class AgentState(TypedDict, total=False):
    """What flows between nodes."""

    messages: Annotated[list[AnyMessage], add_messages]

    task: str
    session_id: str
    capability_profile: list[str]

    plan: Plan | None
    plan_revisions: int
    plan_cursor: int
    """Which plan step is being worked on. Advisory -- the agent may deviate,
    and the deviation is recorded rather than prevented."""

    step: int
    tool_calls: dict[str, int]
    action_history: Annotated[list[ActionFingerprint], operator.add]

    budget: BudgetState
    summary: str | None
    scratchpad: Annotated[list[ObservationRecord], operator.add]
    citations: Annotated[list[dict[str, Any]], operator.add]

    guardrail_events: Annotated[list[GuardrailEvent], operator.add]
    pending_action: PendingAction | None
    approval_denied: bool

    reflect_note: str
    """What the last reflection concluded. Carried into the next act prompt so
    reflection changes behaviour instead of being a log line."""

    force_synthesis: bool
    """Set when a *soft* ceiling fires -- a step or per-tool limit. The action
    is refused, but the run answers from the evidence it already has instead
    of discarding it. Hard stops (loop detected, wall clock, spend) still halt
    outright. The guardrail event records which ceiling fired either way, so
    the distinction survives into the eval."""

    final_answer: str | None
    terminal_reason: TerminalReason | None
    halt_detail: str


def initial_state(*, task: str, session_id: str, capability_profile: list[str]) -> AgentState:
    """A fresh run. Every field that a reducer appends to starts empty."""
    return AgentState(
        messages=[],
        task=task,
        session_id=session_id,
        capability_profile=capability_profile,
        plan=None,
        plan_revisions=0,
        plan_cursor=0,
        step=0,
        tool_calls={},
        action_history=[],
        budget=BudgetState(),
        summary=None,
        scratchpad=[],
        citations=[],
        guardrail_events=[],
        pending_action=None,
        approval_denied=False,
        reflect_note="",
        force_synthesis=False,
        final_answer=None,
        terminal_reason=None,
        halt_detail="",
    )
