"""Planning and reflection.

The plan is **advisory**, and that is a measurement decision as much as a
design one. Pure ReAct has no notion of an optimal path, which makes step
efficiency unmeasurable; plan-first-then-execute is brittle when a tool fails.
An advisory plan gives Phase 4 three separate objects to compare -- the
annotated optimal path, the agent's plan, and its actual trajectory -- and
lets the agent abandon the plan when reality disagrees.

Refusal happens **here**, at step one, not after twelve steps of grinding.
"The agent eventually said 'I don't know'" is a failure if it took fifteen
steps to get there, and the refusal metric is gated on step count precisely to
catch that.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from vichara.agent.nodes.context import AgentContext
from vichara.agent.state import AgentState
from vichara.logging import bind_step, get_logger
from vichara.trajectory.schema import Plan, PlanStep, StepKind, TerminalReason

log = get_logger(__name__)


class PlanOutput(BaseModel):
    """Structured planner response."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="One or two sentences on the approach.")
    answerable: bool = Field(description="False if no available tool could answer this.")
    needs_clarification: bool = Field(default=False)
    clarifying_question: str | None = Field(default=None)
    steps: list[PlanStep] = Field(default_factory=list)


class ReflectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: str = Field(description="What has actually been established, concretely.")
    decision: Literal["continue", "revise", "give_up"]
    reason: str = Field(description="Why, specifically.")


def plan_node(state: AgentState, context: AgentContext) -> AgentState:
    """Produce or revise the plan."""
    bind_step(state.get("step", 0), node="plan")
    context.recorder.begin_step(StepKind.PLAN)

    revision = state.get("plan_revisions", 0)
    notice = context.registry.capability_notice()
    prompt = context.prompts["plan"].format(
        min_steps=context.config.planning.min_plan_steps,
        max_steps=context.config.planning.max_plan_steps,
        tools=", ".join(context.tool_names) or "(none)",
        capability_notice=f"\n{notice}\n" if notice else "",
        task=state["task"],
    )
    if revision:
        prompt += (
            f"\n\nThis is revision {revision}. The previous plan did not work. "
            f"What happened: {state.get('reflect_note', '')}\n"
            "Do not repeat the approach that failed."
        )

    client = context.provider.get("planner")
    try:
        output = client.structured(
            [SystemMessage(content=context.prompts["system"]), HumanMessage(content=prompt)],
            PlanOutput,
        )
    except Exception as exc:  # noqa: BLE001 - a dead planner ends the run legibly
        log.warning("planner failed", error=str(exc)[:200])
        context.recorder.end_step(note=f"planner failed: {exc}")
        return AgentState(
            terminal_reason=TerminalReason.FATAL_ERROR,
            halt_detail=f"The planner could not be reached: {exc}",
        )

    plan = Plan(
        reasoning=output.reasoning,
        steps=output.steps,
        answerable=output.answerable,
        needs_clarification=output.needs_clarification,
        clarifying_question=output.clarifying_question,
        revision=revision,
    )
    context.recorder.record.plan = plan
    context.recorder.record.plan_revisions = revision
    context.recorder.end_step(thought=output.reasoning)

    log.info(
        "planned",
        steps=len(plan.steps),
        answerable=plan.answerable,
        clarify=plan.needs_clarification,
        revision=revision,
    )
    return AgentState(plan=plan, plan_cursor=0, reflect_note="")


def route_after_plan(state: AgentState) -> str:
    """Impossible and ambiguous tasks leave immediately."""
    if state.get("terminal_reason") is not None:
        return "halt"
    plan = state.get("plan")
    if plan is None:
        return "halt"

    # Clarification is checked first, and the ordering is load-bearing. The
    # planner routinely marks an ambiguous task as `answerable: false` *and*
    # `needs_clarification: true`, which is not incoherent -- it cannot be
    # answered as written. Checking answerability first turned every ambiguous
    # task into a refusal, which the evaluation caught on its first run:
    # `ambiguous-the-cycle` expected `clarify` and got `refused`.
    #
    # The distinction matters beyond bookkeeping. Refusing tells the user
    # nothing can be done; asking which cycle they meant costs one sentence and
    # gets them an answer.
    if plan.needs_clarification and plan.clarifying_question:
        return "clarify"
    if not plan.answerable:
        return "refuse"
    return "act"


def refuse_node(state: AgentState, context: AgentContext) -> AgentState:
    """Decline on step one, which is what correct refusal looks like."""
    plan = state.get("plan")
    reason = plan.reasoning if plan else "No available tool can answer this."
    context.recorder.record.final_answer = reason
    log.info("refused early", step=state.get("step", 0))
    return AgentState(
        final_answer=reason,
        terminal_reason=TerminalReason.REFUSED,
        halt_detail="refused during planning",
    )


def clarify_node(state: AgentState, context: AgentContext) -> AgentState:
    """Ask rather than silently guessing at an ambiguous task."""
    plan = state.get("plan")
    question = (
        plan.clarifying_question
        if plan and plan.clarifying_question
        else "Could you clarify what you are asking for?"
    )
    context.recorder.record.final_answer = question
    return AgentState(
        final_answer=question,
        terminal_reason=TerminalReason.CLARIFY,
        halt_detail="asked a clarifying question",
    )


def should_reflect(state: AgentState, context: AgentContext) -> tuple[bool, str]:
    """Whether to reflect, and what triggered it.

    Reflecting every step roughly doubles request count for marginal gain, and
    requests are the binding budget. These triggers concentrate it where it
    changes behaviour.
    """
    config = context.config.reflection
    step = state.get("step", 0)
    if config.skip_first_step and step <= 1:
        return False, ""

    scratchpad = state.get("scratchpad", [])
    if config.on_tool_error and scratchpad and not scratchpad[-1].ok:
        return True, "The last tool call failed."
    if config.on_low_information and scratchpad and _low_information(scratchpad[-1]):
        return True, "The last observation returned nothing useful."
    if config.every_n_steps and step % config.every_n_steps == 0:
        return True, f"Routine check at step {step}."
    return False, ""


def _low_information(observation: object) -> bool:
    content = getattr(observation, "content", "") or ""
    markers = ("No textbook passages matched", "No web results for", "produced no output")
    return any(marker in content for marker in markers)


def reflect_node(state: AgentState, context: AgentContext) -> AgentState:
    """Decide whether to continue, replan, or stop."""
    bind_step(state.get("step", 0), node="reflect")
    context.recorder.begin_step(StepKind.REFLECT)

    _, trigger = should_reflect(state, context)
    recent = (
        "\n\n".join(
            f"[{obs.tool}] ok={obs.ok} {obs.content[:400]}"
            for obs in state.get("scratchpad", [])[-4:]
        )
        or "(nothing yet)"
    )

    prompt = context.prompts["reflect"].format(task=state["task"], recent=recent, trigger=trigger)
    client = context.provider.get("agent")
    try:
        output = client.structured(
            [SystemMessage(content=context.prompts["system"]), HumanMessage(content=prompt)],
            ReflectOutput,
        )
    except Exception as exc:  # noqa: BLE001
        # Reflection is advisory. Failing it costs a thought, not the run.
        log.warning("reflection failed, continuing", error=str(exc)[:160])
        context.recorder.end_step(note="reflection failed")
        return AgentState(reflect_note="")

    context.recorder.end_step(thought=f"{output.decision}: {output.reason}")
    log.info("reflected", decision=output.decision)

    note = f"Earlier assessment: {output.assessment} ({output.reason})"

    if output.decision == "give_up":
        return AgentState(
            reflect_note=note,
            final_answer=(f"I could not answer this with the tools available. {output.reason}"),
            terminal_reason=TerminalReason.REFUSED,
            halt_detail="gave up after reflection",
        )

    if output.decision == "revise":
        if state.get("plan_revisions", 0) >= context.config.budget.max_plan_revisions:
            # Plan-thrash is its own failure mode: an agent that keeps
            # rewriting the plan makes no progress while looking busy.
            return AgentState(
                reflect_note=note + " (revision limit reached; proceeding)",
            )
        return AgentState(
            reflect_note=note,
            plan_revisions=state.get("plan_revisions", 0) + 1,
        )

    return AgentState(reflect_note=note)


def route_after_reflect(state: AgentState) -> str:
    if state.get("terminal_reason") is not None:
        return "halt"
    plan = state.get("plan")
    if plan is not None and state.get("plan_revisions", 0) > plan.revision:
        return "plan"
    return "act"
