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


if __name__ == "__main__":
    app()
