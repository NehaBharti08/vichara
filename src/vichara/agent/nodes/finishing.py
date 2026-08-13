"""Compress, synthesize, halt."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from vichara.agent.memory import compress, partition, render_observation
from vichara.agent.nodes.context import AgentContext
from vichara.agent.state import AgentState
from vichara.llm.provider import text_of
from vichara.logging import bind_step, get_logger
from vichara.trajectory.schema import StepKind, TerminalReason

log = get_logger(__name__)


def compress_node(state: AgentState, context: AgentContext) -> AgentState:
    """Digest older observations, keeping their provenance."""
    bind_step(state.get("step", 0), node="compress")
    scratchpad = state.get("scratchpad", [])
    to_digest, keep = partition(scratchpad, context.config.memory)
    if not to_digest:
        return AgentState()

    context.recorder.begin_step(StepKind.COMPRESS)
    summary = compress(
        to_digest,
        client=context.provider.get("compress"),
        config=context.config.memory,
        previous_summary=state.get("summary"),
    )
    context.recorder.end_step(note=f"digested {len(to_digest)} observations")
    log.info("compressed", digested=len(to_digest), kept=len(keep))

    # `scratchpad` uses an append reducer, so it cannot be shrunk by returning
    # a shorter list. The summary carries the digested content forward and the
    # act node reads only the most recent few, which is what actually bounds
    # the prompt.
    return AgentState(summary=summary)


def synthesize_node(state: AgentState, context: AgentContext) -> AgentState:
    """Write the final cited answer."""
    bind_step(state.get("step", 0), node="synthesize")
    context.recorder.begin_step(StepKind.SYNTHESIZE)

    scratchpad = state.get("scratchpad", [])
    parts = []
    if state.get("summary"):
        parts.append(str(state["summary"]))
    parts.extend(
        render_observation(o, tag_provenance=context.config.injection.tag_provenance)
        for o in scratchpad
    )
    evidence = "\n\n".join(parts) or "(no evidence was gathered)"

    prompt = context.prompts["synthesize"].format(task=state["task"], evidence=evidence)
    client = context.provider.get("agent")
    try:
        answer = text_of(
            client.invoke(
                [
                    SystemMessage(content=context.prompts["system"]),
                    HumanMessage(content=prompt),
                ]
            )
        ).strip()
    except Exception as exc:  # noqa: BLE001 - report the failure, do not lose the trajectory
        log.warning("synthesis failed", error=str(exc)[:200])
        context.recorder.end_step(note=f"synthesis failed: {exc}")
        return AgentState(
            terminal_reason=TerminalReason.FATAL_ERROR,
            halt_detail=f"Could not write the answer: {exc}",
        )

    context.recorder.end_step()
    context.recorder.record.final_answer = answer
    return AgentState(final_answer=answer, terminal_reason=TerminalReason.ANSWERED)


def halt_node(state: AgentState, context: AgentContext) -> AgentState:
    """Terminal node. Stamps the reason and closes the trajectory.

    Every exit reaches here, because `terminal_reason` unset is a measurement
    hole rather than a neutral default -- Phase 4's refusal metric reads it
    directly.
    """
    reason = state.get("terminal_reason") or TerminalReason.FATAL_ERROR
    detail = state.get("halt_detail", "")

    record = context.recorder.record
    record.terminal_reason = reason
    record.citations = state.get("citations", [])
    if record.final_answer is None:
        record.final_answer = state.get("final_answer") or _explain(reason, detail)

    ledger = context.provider.ledger
    record.llm_requests = ledger.requests
    record.cache_hits = ledger.cache_hits
    record.total_tokens = ledger.total_tokens
    record.est_usd = round(ledger.est_usd, 6)
    record.wall_clock_s = round(ledger.wall_clock_s, 2)

    log.info(
        "run finished",
        terminal=reason.value,
        steps=state.get("step", 0),
        requests=ledger.requests,
    )
    return AgentState(
        terminal_reason=reason,
        final_answer=record.final_answer,
    )


def _explain(reason: TerminalReason, detail: str) -> str:
    """What the user sees when the agent stopped without answering.

    Phrased as a statement about the run rather than an apology: a ceiling
    that fired is the system working, and the honest thing is to say what was
    established and what was not.
    """
    base = {
        TerminalReason.STEP_CEILING: (
            "I stopped after reaching the step limit without reaching a grounded answer."
        ),
        TerminalReason.BUDGET_EXHAUSTED: (
            "I stopped after reaching this run's budget without reaching a grounded answer."
        ),
        TerminalReason.LOOP_DETECTED: (
            "I stopped because I was repeating the same action without making progress."
        ),
        TerminalReason.FATAL_ERROR: "The run could not be completed.",
        TerminalReason.REFUSED: "I could not answer this with the tools available.",
        TerminalReason.CLARIFY: "I need a clarification before I can answer.",
        TerminalReason.ANSWERED: "",
    }.get(reason, "The run ended.")
    return f"{base} {detail}".strip()
