"""Act, guard, approve, execute, observe.

The guard sits between choosing an action and performing it, which is the only
place both are visible. Putting the ceilings inside the tools would make them
invisible to the trajectory; putting them after execution would make them
pointless.

Approval interrupts live here too, and they are exercised on every evaluation
run through a policy auto-approver rather than being demo-only. Untested
human-in-the-loop is decoration.
"""

from __future__ import annotations

import difflib
import hashlib

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from vichara.agent.memory import render_observation
from vichara.agent.nodes.context import AgentContext
from vichara.agent.state import ActionFingerprint, AgentState, PendingAction
from vichara.guardrails.injection.defences import guard
from vichara.logging import bind_step, get_logger
from vichara.tools.config import RiskClass
from vichara.trajectory.schema import (
    GuardrailEvent,
    ObservationRecord,
    StepKind,
    TerminalReason,
    ToolCallRecord,
)

log = get_logger(__name__)


def act_node(state: AgentState, context: AgentContext) -> AgentState:
    """Choose one tool call, or decide the answer is ready."""
    step = state.get("step", 0) + 1
    bind_step(step, node="act")
    context.recorder.begin_step(StepKind.ACT)

    plan = state.get("plan")
    plan_text = (
        "\n".join(f"{i + 1}. [{s.tool or 'any'}] {s.intent}" for i, s in enumerate(plan.steps))
        if plan and plan.steps
        else "(no plan)"
    )

    evidence = state.get("summary") or ""
    recent = state.get("scratchpad", [])[-context.config.memory.verbatim_recent_observations :]
    if recent:
        evidence += "\n\n" + "\n\n".join(
            render_observation(o, tag_provenance=context.config.injection.tag_provenance)
            for o in recent
        )

    reflect_note = state.get("reflect_note", "")
    prompt = context.prompts["act"].format(
        task=state["task"],
        plan=plan_text,
        summary=evidence or "(nothing yet)",
        reflect_note=f"\n{reflect_note}\n" if reflect_note else "",
    )

    client = context.provider.get("agent")
    tools = [t.as_langchain_tool() for t in context.tools]
    try:
        response = client.invoke(
            [SystemMessage(content=context.prompts["system"]), HumanMessage(content=prompt)],
            tools=tools,
        )
    except Exception as exc:  # noqa: BLE001 - a dead provider ends the run legibly
        log.warning("act failed", error=str(exc)[:200])
        context.recorder.end_step(note=f"act failed: {exc}")
        return AgentState(
            step=step,
            terminal_reason=TerminalReason.FATAL_ERROR,
            halt_detail=f"The model could not be reached: {exc}",
        )

    calls = response.tool_calls or []
    if not calls:
        # No tool call means the model believes it can answer now.
        context.recorder.end_step(note="no tool call; ready to answer")
        return AgentState(step=step, pending_action=None)

    # One action at a time. The model may emit parallel calls, but the guard,
    # the approval interrupt and the per-tool budget all reason about a single
    # pending action -- and a trajectory of one-action steps is what makes step
    # efficiency comparable to an annotated optimal path.
    call = calls[0]
    pending = PendingAction(
        tool=call["name"],
        args=dict(call.get("args") or {}),
        call_id=call.get("id"),
        risk=_risk_of(context, call["name"]),
    )
    context.recorder.end_step(
        tool_calls=[ToolCallRecord(tool=pending.tool, args=pending.args, call_id=pending.call_id)],
        note=f"{len(calls)} call(s) proposed" if len(calls) > 1 else "",
    )
    return AgentState(step=step, pending_action=pending)


def _risk_of(context: AgentContext, tool_name: str) -> str:
    status = context.registry.status(tool_name)
    return status.spec.risk.value if status else RiskClass.READ.value


def route_after_act(state: AgentState) -> str:
    if state.get("terminal_reason") is not None:
        return "halt"
    return "guard" if state.get("pending_action") else "synthesize"


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


def guard_node(state: AgentState, context: AgentContext) -> AgentState:
    """Enforce ceilings before the action runs.

    Full enforcement lands in Phase 5; the ceilings that bound cost and
    non-termination are here from the start, because a graph that can run away
    is not safe to develop against.
    """
    pending = state.get("pending_action")
    if pending is None:
        return AgentState()

    step = state.get("step", 0)
    budget = context.config.budget
    events: list[GuardrailEvent] = []

    def block(rule: str, detail: str, reason: TerminalReason, *, soft: bool = False) -> AgentState:
        """Refuse the action.

        A *soft* ceiling -- steps used up, one tool called too often -- refuses
        the action but lets the run answer from what it already gathered. An
        agent holding thirteen citations that reports only "budget exhausted"
        has thrown away the work; the ceiling is there to stop it spending
        more, not to make it forget.

        Hard stops still end the run: a detected loop or an exhausted wall
        clock means the evidence is not going to improve and the run should
        not keep paying to discover that.
        """
        events.append(GuardrailEvent(step=step, rule=rule, action="block", detail=detail))
        for event in events:
            context.recorder.add_guardrail_event(event)
        log.info("guardrail blocked action", rule=rule, detail=detail, soft=soft)

        if soft and state.get("scratchpad"):
            return AgentState(guardrail_events=events, pending_action=None, force_synthesis=True)
        return AgentState(
            guardrail_events=events,
            pending_action=None,
            terminal_reason=reason,
            halt_detail=detail,
        )

    if step > budget.max_steps:
        return block(
            "max_steps",
            f"Reached the {budget.max_steps}-step ceiling.",
            TerminalReason.STEP_CEILING,
            soft=True,
        )

    ledger = context.provider.ledger
    if ledger.requests >= budget.max_llm_requests:
        return block(
            "max_llm_requests",
            f"Reached the {budget.max_llm_requests}-request ceiling.",
            TerminalReason.BUDGET_EXHAUSTED,
        )
    if ledger.total_tokens >= budget.max_tokens:
        return block("max_tokens", "Token budget exhausted.", TerminalReason.BUDGET_EXHAUSTED)
    if ledger.est_usd >= budget.max_est_usd:
        return block("max_est_usd", "Spend ceiling reached.", TerminalReason.BUDGET_EXHAUSTED)
    if ledger.wall_clock_s >= budget.max_wall_clock_s:
        return block("max_wall_clock", "Time limit reached.", TerminalReason.BUDGET_EXHAUSTED)

    used = state.get("tool_calls", {}).get(pending.tool, 0)
    limit = context.config.tool_limits.limit_for(pending.tool)
    if used >= limit:
        # Not merely a safety rail: the cap forces the reflect path instead of
        # letting the agent reformulate a failing query indefinitely.
        return block(
            "per_tool_limit",
            f"{pending.tool} has already been called {used} times (limit {limit}).",
            TerminalReason.BUDGET_EXHAUSTED,
            soft=True,
        )

    fingerprint = ActionFingerprint.of(step, pending.tool, pending.args)
    history = state.get("action_history", [])
    identical = sum(1 for f in history if f.digest == fingerprint.digest)
    if identical + 1 >= context.config.loops.identical_threshold:
        return block(
            "identical_action",
            f"{pending.tool} was already called with these exact arguments.",
            TerminalReason.LOOP_DETECTED,
        )

    if _near_repeat(fingerprint, history, context):
        return block(
            "near_repeat",
            f"{pending.tool} has been called with near-identical arguments repeatedly.",
            TerminalReason.LOOP_DETECTED,
        )

    if pending.risk == RiskClass.DESTRUCTIVE.value:
        events.append(
            GuardrailEvent(
                step=step,
                rule="approval_required",
                action="require_approval",
                detail=f"{pending.tool} is destructive",
            )
        )
    else:
        events.append(
            GuardrailEvent(step=step, rule="risk_class", action="allow", detail=pending.tool)
        )

    for event in events:
        context.recorder.add_guardrail_event(event)
    return AgentState(guardrail_events=events, action_history=[fingerprint])


def _near_repeat(
    fingerprint: ActionFingerprint, history: list[ActionFingerprint], context: AgentContext
) -> bool:
    """Catch paraphrase loops -- the same query with the wording shuffled."""
    window = context.config.loops.near_repeat_window
    same_tool = [f for f in history if f.tool == fingerprint.tool][-window:]
    if len(same_tool) < window:
        return False
    return all(
        difflib.SequenceMatcher(None, f.normalised, fingerprint.normalised).ratio()
        >= context.config.loops.near_repeat_similarity
        for f in same_tool
    )


def route_after_guard(state: AgentState) -> str:
    if state.get("terminal_reason") is not None:
        return "halt"
    if state.get("force_synthesis"):
        return "synthesize"
    pending = state.get("pending_action")
    if pending is None:
        return "halt"
    if pending.risk == RiskClass.DESTRUCTIVE.value:
        return "approve"
    return "execute"


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def approve_node(state: AgentState, context: AgentContext) -> AgentState:
    """Pause for a human decision on a destructive action.

    ``interrupt()`` suspends the graph; the checkpointer holds the state until
    a ``Command(resume=...)`` arrives, which is what makes a run resumable
    across process restarts rather than merely pausable.
    """
    pending = state.get("pending_action")
    if pending is None:
        return AgentState()

    if context.auto_approve:
        # Evaluation still takes this path and still records the decision, so
        # the interrupt is exercised on every task rather than only in a demo.
        context.recorder.add_guardrail_event(
            GuardrailEvent(
                step=state.get("step", 0),
                rule="approval",
                action="allow",
                detail="auto-approved (evaluation policy)",
            )
        )
        return AgentState(approval_denied=False)

    decision = interrupt(
        {
            "kind": "approval_request",
            "tool": pending.tool,
            "args": pending.args,
            "risk": pending.risk,
            "prompt": f"Allow {pending.tool} with these arguments?",
        }
    )

    approved = bool(decision) if not isinstance(decision, dict) else bool(decision.get("approved"))
    context.recorder.add_guardrail_event(
        GuardrailEvent(
            step=state.get("step", 0),
            rule="approval",
            action="allow" if approved else "block",
            detail=f"human {'approved' if approved else 'denied'} {pending.tool}",
        )
    )
    return AgentState(approval_denied=not approved)


def route_after_approve(state: AgentState) -> str:
    return "observe" if state.get("approval_denied") else "execute"


# ---------------------------------------------------------------------------
# Execute and observe
# ---------------------------------------------------------------------------


def execute_node(state: AgentState, context: AgentContext) -> AgentState:
    """Run the approved tool call."""
    pending = state.get("pending_action")
    if pending is None:
        return AgentState()

    bind_step(state.get("step", 0), node="execute", tool=pending.tool)
    context.recorder.begin_step(StepKind.EXECUTE)

    tool = context.tool(pending.tool)
    if tool is None:
        observation = ObservationRecord(
            step=state.get("step", 0),
            tool=pending.tool,
            ok=False,
            content=(
                f"{pending.tool} is not available in this session. "
                "Use a different tool, or say what you cannot establish without it."
            ),
            error_code="backend_unavailable",
        )
    else:
        result = tool.run(**pending.args)
        # Untrusted output is scanned before it ever reaches the model, and the
        # result is recorded on the observation. That is what lets a
        # compromised trajectory be attributed to a specific tool result
        # afterwards rather than inferred from the answer.
        content, scanned = guard(result.content, context.config.injection)
        observation = ObservationRecord(
            step=state.get("step", 0),
            tool=result.tool,
            backend=result.backend,
            ok=result.ok,
            content=content,
            injection_flagged=scanned.flagged,
            trust=result.trust.value,
            error_code=result.error_code.value if result.error_code else None,
            retryable=result.retryable,
            latency_ms=result.latency_ms,
            truncated=result.truncated,
            raw_bytes=result.raw_bytes,
            citations=[c.model_dump() for c in result.citations],
        )

    events: list[GuardrailEvent] = []
    if context.config.loops.detect_redundant_results and observation.ok:
        earlier = _first_matching_step(observation.content, state.get("scratchpad", []))
        if earlier is not None:
            observation = _as_redundant(observation, earlier)
            event = GuardrailEvent(
                step=state.get("step", 0),
                rule="redundant_result",
                action="warn",
                detail=f"{observation.tool} returned the result of step {earlier}",
            )
            events.append(event)
            context.recorder.add_guardrail_event(event)

    context.recorder.add_observation(observation)
    context.recorder.end_step()

    counts = dict(state.get("tool_calls", {}))
    counts[pending.tool] = counts.get(pending.tool, 0) + 1

    return AgentState(
        scratchpad=[observation],
        citations=list(observation.citations),
        tool_calls=counts,
        guardrail_events=events,
        pending_action=None,
    )


def _first_matching_step(content: str, scratchpad: list[ObservationRecord]) -> int | None:
    """The step of the earliest observation holding exactly these bytes."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    for prior in scratchpad:
        if (
            not prior.redundant
            and prior.ok
            and hashlib.sha256(prior.content.encode("utf-8")).hexdigest() == digest
        ):
            return prior.step
    return None


def _as_redundant(observation: ObservationRecord, earlier: int) -> ObservationRecord:
    """Replace a duplicate body with a pointer to the step that already has it.

    This warns rather than blocks, and the distinction is deliberate: the agent
    is not missing evidence, it is holding the evidence twice. Halting here
    would repeat the mistake the soft ceiling made when it ended a run holding
    thirteen unused citations -- the ceiling exists to stop the agent spending
    more, not to make it forget.

    Dropping the body is safe because ``citations`` was extracted before this
    ran and the identical earlier observation already contributed the same
    entries. It also removes the reason the loop was expensive: resending ~11KB
    the model has already read, on every subsequent step, at quadratic cost.
    """
    return observation.model_copy(
        update={
            "redundant": True,
            "content": (
                f"This returned exactly what {observation.tool} already returned at "
                f"step {earlier}; the text is there and is not repeated here. "
                "Rewording the query is not finding new evidence. Answer from what "
                "you have, or use a different tool to establish something new."
            ),
            "citations": [],
        }
    )


def observe_node(state: AgentState, context: AgentContext) -> AgentState:
    """Record a denied action as an observation the agent can react to.

    A refusal the agent never sees is indistinguishable from a tool that did
    nothing, and it would go on trying.
    """
    if not state.get("approval_denied"):
        return AgentState(approval_denied=False)

    pending = state.get("pending_action")
    observation = ObservationRecord(
        step=state.get("step", 0),
        tool=pending.tool if pending else "unknown",
        ok=False,
        content=(
            "The user declined this action. Do not attempt it again. "
            "Continue without it and say what you could not do."
        ),
        error_code="policy_violation",
    )
    context.recorder.begin_step(StepKind.OBSERVE)
    context.recorder.add_observation(observation)
    context.recorder.end_step(note="approval denied")
    return AgentState(scratchpad=[observation], pending_action=None, approval_denied=False)
