"""The trajectory viewer.

The demo, and the thing that makes this repository legible in thirty seconds.
An answer alone shows nothing that a chat window would not: the point is the
*trajectory* -- what it planned, which tool it reached for, what came back,
what each step cost, where a guardrail fired, and which citations trace back to
something a tool actually returned.

Three things here are deliberate and worth defending.

**Untrusted content is shown as untrusted.** Every tool result is rendered
inside its provenance fence, and anything the injection scanner flagged is
marked. A viewer that renders retrieved text identically to the agent's own
reasoning teaches the reader the wrong mental model of where the risk is.

**Citations are shown verified or not.** After the Phase 5 measurement put
false citation at the top of the risk list, "this citation came from a tool"
is the single most valuable fact the interface can display.

**Replay mode exists so the Space is never dead.** The free tier allows
roughly 150 demo queries a day. A recruiter who clicks a broken demo forms an
impression that cannot be retaken, so when quota is gone the viewer serves a
recorded trajectory and says plainly that it is doing so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from vichara.agent.runner import DEFAULT_TRAJECTORY_STORE, AgentSession
from vichara.logging import configure_logging, get_logger
from vichara.settings import PipelineConfig, Settings, load_pipeline_config
from vichara.trajectory.recorder import read_trajectories
from vichara.trajectory.schema import StepKind, TrajectoryRecord

log = get_logger(__name__)

EXAMPLES = [
    "What does the textbook say about how the sodium-potassium pump maintains the resting membrane potential?",
    "Find the resting membrane potential in the textbook, then compute the change to a +30 mV peak.",
    "Summarise the chapter on quantum biology in the OpenStax Biology textbook.",
    "Explain the cycle and how it is regulated.",
]

_STEP_ICON = {
    StepKind.PLAN: "plan",
    StepKind.ACT: "act",
    StepKind.EXECUTE: "tool",
    StepKind.OBSERVE: "observe",
    StepKind.REFLECT: "reflect",
    StepKind.COMPRESS: "compress",
    StepKind.SYNTHESIZE: "answer",
}


def render_trajectory(record: TrajectoryRecord) -> str:
    """The reasoning tree as markdown."""
    lines = [f"### Trajectory `{record.session_id}`", ""]

    blocked = {e.step for e in record.guardrail_events if e.action == "block"}

    for step in record.steps:
        label = _STEP_ICON.get(step.kind, step.kind.value)
        header = f"**{step.index}. {label}**  ·  {step.duration_ms:.0f} ms"
        if step.index in blocked:
            header += "  ·  **guardrail blocked**"
        lines.append(header)

        if step.thought:
            lines.append(f"> {step.thought}")

        for call in step.tool_calls:
            args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in call.args.items())
            lines.append(f"- calls `{call.tool}({args})`")

        for obs in step.observations:
            status = "ok" if obs.ok else f"FAILED ({obs.error_code})"
            flag = "  ·  **injection flagged**" if obs.injection_flagged else ""
            lines.append(
                f"- `{obs.tool}` -> {status}, {obs.raw_bytes} bytes, "
                f"{obs.latency_ms:.0f} ms{flag}"
            )
            # Untrusted output is shown inside its fence rather than as plain
            # prose. Rendering it like the agent's own reasoning would teach
            # the reader the wrong model of where the risk lives.
            body = obs.content[:600] + ("..." if len(obs.content) > 600 else "")
            fence = "untrusted tool output" if obs.trust == "untrusted" else "tool output"
            lines.append(f"  <details><summary>{fence}</summary>\n\n```\n{body}\n```\n</details>")

        lines.append("")

    return "\n".join(lines)


def render_guardrails(record: TrajectoryRecord) -> str:
    if not record.guardrail_events:
        return "_No guardrail fired._"
    rows = ["| step | rule | action | detail |", "| --- | --- | --- | --- |"]
    for event in record.guardrail_events:
        mark = f"**{event.action}**" if event.action == "block" else event.action
        rows.append(f"| {event.step} | `{event.rule}` | {mark} | {event.detail[:70]} |")
    return "\n".join(rows)


def render_citations(record: TrajectoryRecord) -> str:
    """Sources, and whether each traces back to a tool.

    The most valuable panel in the interface. Phase 5 measured false citation
    as the top risk, so 'a tool returned this' is the fact worth surfacing.
    """
    parts = []
    if record.citations:
        seen: set[str] = set()
        parts.append("**Verified — returned by a tool**\n")
        for citation in record.citations:
            source = str(citation.get("source", ""))
            if source and source not in seen:
                seen.add(source)
                locator = citation.get("locator")
                link = f" ({locator})" if locator and str(locator).startswith("http") else ""
                parts.append(f"- {source}{link}")
    else:
        parts.append("_No citations. Expected for a refusal or a clarification._")

    if record.citations_fabricated:
        parts.append("\n**Removed — no tool returned these**\n")
        parts.extend(f"- ~~{span[:120]}~~" for span in record.citations_fabricated)
    return "\n".join(parts)


def render_cost(record: TrajectoryRecord) -> str:
    return "\n".join(
        [
            "| metric | value |",
            "| --- | --- |",
            f"| terminal | `{record.terminal_reason}` |",
            f"| agent steps | {record.agent_steps} |",
            f"| model requests | {record.llm_requests} |",
            f"| cache hits | {record.cache_hits} |",
            f"| tokens | {record.total_tokens} |",
            f"| wall clock | {record.wall_clock_s} s |",
            f"| capability profile | {', '.join(record.capability_profile)} |",
            f"| profile | `{record.profile}` |",
        ]
    )


def _replays(path: Path) -> list[TrajectoryRecord]:
    """Recorded trajectories, newest first."""
    try:
        return list(read_trajectories(path))[::-1]
    except OSError:
        return []


def build_app(settings: Settings, config: PipelineConfig) -> Any:
    # Returns Any rather than gr.Blocks: gradio ships no usable stubs, so the
    # class resolves to Any under mypy --strict and annotating the real type
    # only produces a no-any-return error at the call site.
    store = settings.resolved(str(DEFAULT_TRAJECTORY_STORE))
    provider_ready = settings.has_google_key

    def answer(question: str, live: bool) -> tuple[str, str, str, str, str]:
        question = (question or "").strip()
        if not question:
            return "Ask something.", "", "", "", ""

        if live and provider_ready:
            try:
                with AgentSession(
                    settings, config, auto_approve=True, prefer_recorded_search=True
                ) as session:
                    record = session.run(question).record
            except Exception as exc:  # noqa: BLE001 - a dead demo is the worst outcome
                log.warning("live run failed, falling back to replay", error=str(exc)[:200])
                return _fallback(
                    store, f"Live run failed ({type(exc).__name__}). Showing a recorded run."
                )
        else:
            reason = "No API key configured." if not provider_ready else "Replay mode selected."
            return _fallback(store, f"{reason} Showing a recorded run.")

        return (
            record.final_answer or "_No answer._",
            render_citations(record),
            render_cost(record),
            render_guardrails(record),
            render_trajectory(record),
        )

    # analytics_enabled=False is not cosmetic. Gradio otherwise posts usage
    # telemetry to an external endpoint on launch, and a project that ships a
    # threat model should not make an undeclared outbound request on startup.
    with gr.Blocks(title="Vichara - trajectory viewer", analytics_enabled=False) as app:
        gr.Markdown(
            "# Vichara\n"
            "A study agent that plans, calls tools, and cites its sources. "
            "**The trajectory below is the point** — the answer is the least "
            "interesting part.\n\n"
            "Every tool result is shown inside its provenance fence, every "
            "citation is marked verified or removed, and every guardrail firing "
            "is listed. See "
            "[EVALUATION](https://github.com/NehaBharti08/vichara/blob/main/docs/EVALUATION.md) "
            "and [PROMPT_INJECTION](https://github.com/NehaBharti08/vichara/blob/main/docs/PROMPT_INJECTION.md)."
        )

        with gr.Row():
            question = gr.Textbox(label="Question", scale=4, placeholder=EXAMPLES[0])
            live = gr.Checkbox(
                label="Run live",
                value=provider_ready,
                interactive=provider_ready,
                scale=1,
                info="Off = replay a recorded trajectory (no quota used)",
            )
        run = gr.Button("Ask", variant="primary")
        gr.Examples(examples=EXAMPLES, inputs=question)

        answer_box = gr.Markdown(label="Answer")
        with gr.Row():
            citations_box = gr.Markdown(label="Sources")
            cost_box = gr.Markdown(label="Cost")
        guardrails_box = gr.Markdown(label="Guardrails")
        trajectory_box = gr.Markdown(label="Trajectory")

        outputs = [answer_box, citations_box, cost_box, guardrails_box, trajectory_box]
        run.click(answer, inputs=[question, live], outputs=outputs)
        question.submit(answer, inputs=[question, live], outputs=outputs)

    return app


def _fallback(store: Path, notice: str) -> tuple[str, str, str, str, str]:
    """Serve a recorded trajectory. The Space must never look broken."""
    records = _replays(store)
    if not records:
        return (f"_{notice} No recordings are available either._", "", "", "", "")
    record = records[0]
    return (
        f"> _{notice}_\n\n{record.final_answer or ''}",
        render_citations(record),
        render_cost(record),
        render_guardrails(record),
        render_trajectory(record),
    )


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    config = load_pipeline_config(settings.profile)
    build_app(settings, config).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
