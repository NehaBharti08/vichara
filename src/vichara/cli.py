"""Command line entry point.

``health`` is the important one. It is the container healthcheck, the CI smoke
test, and the first thing to run when something is wrong, so it reports the
*resolved* state of the system rather than the configured one -- which backend
each tool would actually use, not which one the YAML asks for.

It distinguishes two outcomes on purpose. A broken configuration is a failure
and exits non-zero. A missing credential is not: it degrades a tool to its
fixture backend, which is a supported way to run this agent, and reporting it
as an error would train you to ignore the command.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from vichara import __version__
from vichara.logging import configure_logging
from vichara.settings import (
    CONFIG_DIR,
    PipelineConfig,
    SandboxBackend,
    Settings,
    load_pipeline_config,
)
from vichara.tools.config import ToolRegistryConfig, load_tool_registry
from vichara.tools.registry import Registry, build_registry
from vichara.trajectory.schema import TerminalReason, TrajectoryRecord

_PROBE_SESSION = "cli-probe"
"""Session id used for health probing. The workspace tool needs one, and
using a fixed name keeps `vichara health` from littering a new session
directory on every invocation."""

app = typer.Typer(
    name="vichara",
    help="A study agent that plans, calls tools, and cites its sources.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ProfileOption = Annotated[
    str | None,
    typer.Option("--profile", "-p", help="Config profile. Defaults to VICHARA_PROFILE."),
]

OK = "[green]ok[/green]"
DEGRADED = "[yellow]degraded[/yellow]"
FAIL = "[red]fail[/red]"


@app.callback()
def _main() -> None:
    """Configure logging once, for every command.

    Without this only `health` configured it, so other commands fell back to
    the JSON default and printed machine-readable log lines at a human. The
    format is the operator's choice via LOG_FORMAT; the CLI just honours it.
    """
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)


def _load(profile: str | None) -> tuple[Settings, PipelineConfig, ToolRegistryConfig]:
    settings = Settings()
    config = load_pipeline_config(profile or settings.profile)
    registry = load_tool_registry()
    return settings, config, registry


def _probe_registry(settings: Settings, config: PipelineConfig) -> Registry:
    """Actually construct and probe the tools.

    Earlier this guessed the backend from which credentials were present. That
    was wrong in the case that matters most: a configured-but-down retrieval
    service reported ``http`` while the agent would really have used the
    fixture corpus. A health command that reports intent rather than outcome
    is worse than none, so it now builds the same registry the agent gets.
    """
    return build_registry(settings, config, session_id=_PROBE_SESSION)


def _node_version() -> str | None:
    """Node hosts the Pyodide sandbox. Absent is survivable until Phase 2."""
    if shutil.which("node") is None:
        return None
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"vichara {__version__}")


@app.command()
def config(profile: ProfileOption = None) -> None:
    """Show the resolved behavioural configuration for a profile."""
    settings = Settings()
    cfg = load_pipeline_config(profile or settings.profile)
    console.print_json(cfg.model_dump_json(indent=2))


@app.command()
def tools(profile: ProfileOption = None) -> None:
    """Show the capability set this environment actually has."""
    settings, config, declared = _load(profile)
    registry = _probe_registry(settings, config)

    table = Table(title="Tools", header_style="bold")
    table.add_column("tool")
    table.add_column("status")
    table.add_column("backend")
    table.add_column("risk")
    table.add_column("output")

    for status in registry.statuses:
        if status.available:
            state = DEGRADED if status.health.degraded else OK
        else:
            state = f"[dim]{status.reason or 'unavailable'}[/dim]"
        table.add_row(
            status.spec.name,
            state,
            status.health.backend,
            status.spec.risk.value,
            status.spec.output_trust.value,
        )

    console.print(table)
    console.print(
        f"\nCapability set: [bold]{len(registry.capability_profile)}[/bold] of "
        f"{len(declared.tools)} declared tools available."
    )

    notice = registry.capability_notice()
    if notice:
        console.print("\n[dim]The agent will be told:[/dim]")
        console.print(f"[dim]{notice}[/dim]")


@app.command()
def health(profile: ProfileOption = None) -> None:
    """Verify configuration, paths and runtimes. Exits non-zero on failure."""
    failures: list[str] = []

    try:
        settings, cfg, registry = _load(profile)
    except Exception as exc:
        console.print(f"{FAIL}  configuration did not load: {exc}")
        raise typer.Exit(code=1) from exc

    configure_logging(settings.log_level, settings.log_format)

    table = Table(title=f"vichara {__version__} health", header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")

    table.add_row("config", OK, f"profile={cfg.name} ({CONFIG_DIR})")
    table.add_row("tools", OK, f"{len(registry.enabled)}/{len(registry.tools)} enabled")

    # Writable paths. Failing here is a real failure: the agent cannot record a
    # trajectory or checkpoint a run, which makes every later phase unusable.
    for label, target in (
        ("workspace", settings.resolved(settings.workspace_root)),
        ("cache dir", settings.resolved(settings.cache_path).parent),
        ("checkpoints", settings.resolved(settings.checkpoint_path).parent),
    ):
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".vichara-write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            table.add_row(label, OK, str(_relative(target)))
        except OSError as exc:
            failures.append(f"{label}: {exc}")
            table.add_row(label, FAIL, str(exc))

    # Model provider. Degraded, not failed: the tool layer and its whole test
    # suite run without a key, and Phase 0 has nothing that calls a model.
    if settings.has_google_key:
        # ASCII only in CLI output: the Windows console defaults to cp1252 and
        # mangles anything outside it, which makes `health` look broken on the
        # exact platform where someone is most likely to be debugging.
        table.add_row("model provider", OK, f"{cfg.models.provider}, agent={cfg.models.agent}")
    else:
        table.add_row("model provider", DEGRADED, "GOOGLE_API_KEY not set")

    # Real construction and probing, not an inference from which keys are set.
    probed = _probe_registry(settings, cfg)
    for status in probed.statuses:
        if status.available:
            table.add_row(
                f"tool:{status.spec.name}",
                DEGRADED if status.health.degraded else OK,
                f"backend={status.health.backend}, {status.health.detail}",
            )
        else:
            table.add_row(
                f"tool:{status.spec.name}",
                DEGRADED,
                f"unavailable: {status.reason or status.health.detail}",
            )

    # Node hosts the Pyodide sandbox. Not a Phase 0 failure, but the sandbox is
    # the default code-execution backend, so silence here would be misleading.
    if cfg.sandbox.backend is SandboxBackend.PYODIDE:
        node = _node_version()
        if node:
            table.add_row("node runtime", OK, node)
        else:
            table.add_row("node runtime", DEGRADED, "node not found; sandbox unavailable")

    console.print(table)

    if failures:
        console.print(f"\n[red]{len(failures)} check(s) failed.[/red]")
        raise typer.Exit(code=1)
    console.print("\n[green]Healthy.[/green] Degraded entries are supported states.")


def _relative(path: Path) -> Path:
    """Shorten a path for display; absolute is fine if it is outside the repo."""
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="The question to answer.")],
    profile: ProfileOption = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Auto-approve destructive actions.")
    ] = False,
    recorded_search: Annotated[
        bool, typer.Option("--recorded-search", help="Replay recorded search, not live.")
    ] = False,
    show_trajectory: Annotated[
        bool, typer.Option("--trajectory", help="Print the step-by-step trajectory.")
    ] = False,
) -> None:
    """Answer a question, printing the answer and what it cost."""
    from vichara.agent.runner import AgentSession

    settings = Settings()
    cfg = load_pipeline_config(profile or settings.profile)
    configure_logging(settings.log_level, settings.log_format)

    with AgentSession(
        settings, cfg, auto_approve=yes, prefer_recorded_search=recorded_search
    ) as session:
        outcome = session.run(task)

        # Approvals are answered here rather than auto-allowed, so the
        # human-in-the-loop path is exercised by the ordinary CLI and not only
        # by the UI. Resumption goes through the checkpoint, which is what
        # makes it survive a restart rather than merely a pause.
        while outcome.interrupted and outcome.interrupt_payload:
            payload = outcome.interrupt_payload
            console.print(
                f"\n[yellow]Approval needed[/yellow]: {payload.get('tool')} "
                f"({payload.get('risk')})"
            )
            console.print_json(data=payload.get("args", {}))
            outcome = session.resume(
                {"approved": typer.confirm("Allow this action?", default=False)}
            )

        record = outcome.record
        console.print()
        if show_trajectory:
            _print_trajectory(record)

        console.print("[bold]Answer[/bold]")
        console.print(record.final_answer or "(none)")

        if record.citations:
            console.print("\n[bold]Sources[/bold]")
            seen: set[str] = set()
            for citation in record.citations:
                source = str(citation.get("source", ""))
                if source and source not in seen:
                    seen.add(source)
                    console.print(f"  - {source}")

        console.print(
            f"\n[dim]{record.terminal_reason} | {record.agent_steps} steps | "
            f"{record.llm_requests} requests ({record.cache_hits} cached) | "
            f"{record.total_tokens} tokens | {record.wall_clock_s}s | "
            f"session {record.session_id}[/dim]"
        )
        # Non-zero for anything but a real answer, so a shell script or CI job
        # can tell "refused" from "answered" without parsing the output.
        if record.terminal_reason is not TerminalReason.ANSWERED:
            raise typer.Exit(code=2)


def _print_trajectory(record: TrajectoryRecord) -> None:
    """The reasoning tree, in text. The Phase 6 viewer renders the same data."""
    table = Table(title="Trajectory", header_style="bold")
    table.add_column("#")
    table.add_column("node")
    table.add_column("detail")
    table.add_column("ms", justify="right")

    for step in record.steps:
        detail = step.thought or step.note
        for call in step.tool_calls:
            detail = f"{call.tool}({_brief(call.args)})"
        for obs in step.observations:
            detail = f"{obs.tool} -> {'ok' if obs.ok else 'FAILED'}, {obs.raw_bytes}B"
        table.add_row(str(step.index), step.kind.value, detail[:70], f"{step.duration_ms:.0f}")

    console.print(table)
    if record.guardrail_events:
        console.print("[bold]Guardrails[/bold]")
        for event in record.guardrail_events:
            colour = "red" if event.action == "block" else "dim"
            console.print(
                f"  [{colour}]step {event.step}: {event.rule} -> {event.action}[/{colour}]"
            )
    console.print()


def _brief(args: dict[str, object]) -> str:
    return ", ".join(f"{key}={str(value)[:40]}" for key, value in args.items())


@app.command()
def evaluate(
    profile: ProfileOption = None,
    repeats: Annotated[
        int, typer.Option("--repeats", "-n", help="Runs per task. 3 routine, 5 headline.")
    ] = 3,
    only: Annotated[str | None, typer.Option("--only", help="Comma-separated task ids.")] = None,
    disable: Annotated[
        str | None,
        typer.Option("--disable", help="Comma-separated tools to remove (degraded profile)."),
    ] = None,
    fault: Annotated[
        str | None,
        typer.Option("--fault", help="tool:kind, e.g. textbook_search:plausible_but_wrong"),
    ] = None,
    max_requests: Annotated[
        int | None, typer.Option("--max-requests", help="Stop before exhausting the quota.")
    ] = None,
    no_resume: Annotated[bool, typer.Option("--no-resume", help="Re-run completed pairs.")] = False,
    report_to: Annotated[
        Path | None, typer.Option("--report", help="Write a markdown report here.")
    ] = None,
) -> None:
    """Run an evaluation sweep. Resumable -- safe to interrupt and restart."""
    from vichara.agent.nodes.context import PROMPT_DIR
    from vichara.eval.faults import FaultKind, FaultSpec
    from vichara.eval.metrics import agent_version_of
    from vichara.eval.report import for_agent, summarise, to_markdown
    from vichara.eval.runner import DEFAULT_RESULTS, ResultStore, SweepConfig, run_sweep
    from vichara.eval.tasks.loader import load_tasks
    from vichara.trajectory.recorder import hash_prompts

    settings = Settings()
    cfg = load_pipeline_config(profile or settings.profile)
    tasks = load_tasks()

    spec = None
    if fault:
        tool_name, _, kind = fault.partition(":")
        spec = FaultSpec(tool=tool_name, kind=FaultKind(kind))

    sweep = SweepConfig(
        profile=cfg.name,
        repeats=repeats,
        only=[t.strip() for t in only.split(",")] if only else [],
        disable_tools=[t.strip() for t in disable.split(",")] if disable else [],
        fault=spec,
        max_requests=max_requests,
    )

    console.print(
        f"[bold]{cfg.name}[/bold]: {len(sweep.only) or len(tasks.tasks)} tasks "
        f"x {repeats} seeds"
        + (f", tools disabled: {sweep.disable_tools}" if sweep.disable_tools else "")
        + (f", fault: {fault}" if fault else "")
    )

    run_sweep(settings, cfg, tasks, sweep, resume=not no_resume)

    # Report over everything recorded for this profile *by this agent*, not
    # only what this invocation produced -- otherwise a resumed sweep reports
    # on its last fragment and the numbers look worse than the run was. The
    # agent filter is what keeps that from also pooling runs made before a
    # prompt edit into the same average.
    every = ResultStore(DEFAULT_RESULTS / f"{cfg.name}.jsonl").read()
    version = agent_version_of(hash_prompts(PROMPT_DIR))
    results = for_agent(every, version)
    if len(results) != len(every):
        console.print(
            f"[yellow]{len(every) - len(results)} run(s) from a different prompt version "
            f"excluded; reporting {len(results)} for {version}.[/yellow]"
        )
    summary = summarise(results)
    _print_summary(summary)

    if report_to:
        report_to.parent.mkdir(parents=True, exist_ok=True)
        report_to.write_text(
            to_markdown(summary, title=f"Evaluation: {cfg.name}"), encoding="utf-8"
        )
        console.print(f"\n[dim]report written to {report_to}[/dim]")


def _print_summary(summary: dict[str, object]) -> None:
    if not summary.get("n_runs"):
        console.print("[yellow]No results.[/yellow]")
        return

    overall = summary["overall"]
    assert isinstance(overall, dict)

    table = Table(
        title=f"{summary['n_runs']} runs / {summary['n_tasks']} tasks", header_style="bold"
    )
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in (
        "terminal_correct",
        "answer_correct",
        "forbidden_tool_rate",
        "cited_rate",
        "refusal_correct",
        "mean_steps_to_refusal",
        "false_refusal_rate",
    ):
        value = overall.get(key)
        table.add_row(key, "-" if value is None else str(value))
    for key in ("tool_precision", "step_efficiency", "steps", "llm_requests"):
        dist = overall.get(key)
        if isinstance(dist, dict):
            table.add_row(f"{key} (median, IQR)", f"{dist['median']} +/- {dist['iqr']}")
    console.print(table)

    by_task = summary["by_task"]
    assert isinstance(by_task, dict)
    unstable = [t for t, row in by_task.items() if isinstance(row, dict) and not row["consistent"]]
    if unstable:
        # Instability and a capability gap need different fixes, so they are
        # never reported as one number.
        console.print(f"\n[yellow]Inconsistent across seeds:[/yellow] {', '.join(unstable)}")


@app.command()
def attack(
    profile: ProfileOption = None,
    only: Annotated[str | None, typer.Option("--only", help="Comma-separated attack ids.")] = None,
    seeds: Annotated[int, typer.Option("--seeds", help="Runs per attack.")] = 1,
    no_resume: Annotated[
        bool, typer.Option("--no-resume", help="Re-run completed attacks.")
    ] = False,
) -> None:
    """Run the prompt-injection suite and report attack success rate."""
    from vichara.agent.nodes.context import PROMPT_DIR
    from vichara.eval.injection_suite import (
        DEFAULT_RESULTS,
        InjectionSweep,
        read_attack_results,
        run_injection_sweep,
        summarise_attacks,
    )
    from vichara.eval.metrics import agent_version_of
    from vichara.trajectory.recorder import hash_prompts

    settings = Settings()
    cfg = load_pipeline_config(profile or settings.profile)
    configure_logging(settings.log_level, settings.log_format)

    sweep = InjectionSweep(
        profile=cfg.name,
        seeds=tuple(range(seeds)),
        only=tuple(t.strip() for t in only.split(",")) if only else (),
    )
    console.print(f"[bold]{cfg.name}[/bold]: running injection suite")
    run_injection_sweep(settings, cfg, sweep, resume=not no_resume)

    # Only this agent's attacks, for the reason the sweep report is scoped:
    # the headline here is a comparison, and one profile measured on one agent
    # against another profile measured on a different one is not a defence
    # measurement.
    every = read_attack_results(DEFAULT_RESULTS / f"injection-{cfg.name}.jsonl")
    version = agent_version_of(hash_prompts(PROMPT_DIR))
    results = [r for r in every if r.agent_version == version]
    if len(results) != len(every):
        console.print(
            f"[yellow]{len(every) - len(results)} attack(s) from a different agent "
            f"excluded; reporting {len(results)} for {version}.[/yellow]"
        )
    summary = summarise_attacks(results)
    if not summary.get("n"):
        console.print("[yellow]No results.[/yellow]")
        return

    table = Table(title=f"Injection suite: {cfg.name}", header_style="bold")
    table.add_column("technique")
    table.add_column("n", justify="right")
    table.add_column("ASR", justify="right")
    by_technique = summary["by_technique"]
    assert isinstance(by_technique, dict)
    for name, row in by_technique.items():
        table.add_row(name, str(row["n"]), f"{row['asr']:.2f}")
    table.add_row("[bold]overall[/bold]", str(summary["n"]), f"[bold]{summary['asr']:.2f}[/bold]")
    console.print(table)
    console.print(
        f"detection rate {summary['detection_rate']:.2f} | "
        f"detected but still succeeded: {summary['detected_but_succeeded']}"
    )
    successful = summary["successful"]
    assert isinstance(successful, list)
    if successful:
        console.print(f"\n[red]Attacks that worked:[/red] {', '.join(successful)}")


@app.command()
def rescore(
    profile: ProfileOption = None,
    trajectories: Annotated[
        Path | None, typer.Option("--trajectories", help="Trajectory store to read.")
    ] = None,
) -> None:
    """Recompute every metric from stored trajectories. Spends no quota.

    Metrics are pure functions of (trajectory, gold task), so a metric fix does
    not require re-running the agent -- which is the whole reason they were
    written that way. Use this after changing a metric, so results from before
    and after the change are not silently incomparable.
    """
    from vichara.agent.runner import DEFAULT_TRAJECTORY_STORE
    from vichara.eval.rescore import rescore as rebuild
    from vichara.eval.runner import DEFAULT_RESULTS

    settings = Settings()
    source = trajectories or settings.resolved(str(DEFAULT_TRAJECTORY_STORE))
    if not source.exists():
        console.print(f"[red]No trajectory store at {source}[/red]")
        raise typer.Exit(code=1)

    counts = rebuild(source, DEFAULT_RESULTS, profile=profile)
    if not counts:
        console.print("[yellow]No trajectories matched a gold task.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Rescored from stored trajectories", header_style="bold")
    table.add_column("profile")
    table.add_column("runs", justify="right")
    for name, count in sorted(counts.items()):
        table.add_row(name, str(count))
    console.print(table)


if __name__ == "__main__":
    app()
