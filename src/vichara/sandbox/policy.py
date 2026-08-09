"""Backend selection and limit construction."""

from __future__ import annotations

from vichara.logging import get_logger
from vichara.sandbox.base import Limits, Sandbox
from vichara.settings import PipelineConfig, SandboxBackend, Settings

log = get_logger(__name__)


def limits_from_config(config: PipelineConfig) -> Limits:
    """Translate the profile's sandbox block into enforced ceilings."""
    return Limits(
        wall_clock_s=config.sandbox.wall_clock_s,
        cpu_seconds=config.sandbox.cpu_seconds,
        memory_mb=config.sandbox.memory_mb,
        max_output_bytes=config.sandbox.max_output_bytes,
        network=config.sandbox.network,
    )


def build_sandbox(settings: Settings, config: PipelineConfig) -> Sandbox:
    """Construct the configured backend.

    ``SANDBOX_BACKEND`` in the environment overrides the profile, which is how
    CI runs the same protocol tests against Docker without a separate profile.
    Selection happens once; nothing downstream branches on which backend it got.
    """
    backend = settings.sandbox_backend or config.sandbox.backend

    if backend is SandboxBackend.DOCKER:
        from vichara.sandbox.docker.backend import DockerSandbox

        return DockerSandbox()

    from vichara.sandbox.pyodide.backend import PyodideSandbox

    return PyodideSandbox()
