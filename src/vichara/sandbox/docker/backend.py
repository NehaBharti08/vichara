"""Docker sandbox backend.

Not the default, and stronger than the default. The trade is deliberate:
Pyodide runs anywhere including a Hugging Face Space and starts in
milliseconds once warm, while Docker needs a daemon that the Space does not
have -- but Docker gives **a fresh process per execution**, which is the one
thing the warm worker cannot.

That single difference matters more than it sounds. In the Pyodide backend one
submission shares an interpreter with the next, so it can leave the sandbox in
a worse state than it found it (see the re-hardening in ``runner.mjs``). Here,
each execution gets a container that is created, used, and destroyed, so there
is no state to poison.

Isolation is by explicit flags, every one of which is doing something:

    --network=none              no interface at all, not a firewall rule
    --read-only                 root filesystem is immutable
    --tmpfs /tmp:size=16m       the only writable path, capped, wiped on exit
    --cap-drop=ALL              no capabilities, including the default set
    --security-opt no-new-privileges   setuid cannot regain them
    --pids-limit                a fork bomb hits a wall instead of the host
    --memory / --memory-swap    equal values, so swap cannot be used to evade
    --cpus                      bounded share, so a busy loop cannot starve
    --user 65534:65534          nobody, so nothing runs as root

Unlike Pyodide, "no network" here is a *configuration*: it is one missing flag
away from being false. That difference is stated in docs/THREAT_MODEL.md
rather than glossed over, and it is why the Pyodide claim is the stronger one
even though this backend is the more capable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

from vichara.logging import get_logger
from vichara.sandbox.base import Limits, Outcome, SandboxResult

log = get_logger(__name__)

DEFAULT_IMAGE = "python:3.11-slim"

# Runs inside the container. Mirrors the Pyodide preamble so both backends
# return the same shape for the same code -- the protocol tests depend on it.
_HARNESS = r"""
import ast, io, json, sys, traceback

src = sys.stdin.read()
out, err = io.StringIO(), io.StringIO()
payload = {"stdout": "", "stderr": "", "exception": None, "result_repr": None}
scope = {"__name__": "__main__"}
old_out, old_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = out, err
try:
    try:
        tree = ast.parse(src, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<agent>", "exec"), scope)
            value = eval(compile(last, "<agent>", "eval"), scope)
            if value is not None:
                payload["result_repr"] = repr(value)
        else:
            exec(compile(tree, "<agent>", "exec"), scope)
    except BaseException:
        payload["exception"] = traceback.format_exc()
finally:
    sys.stdout, sys.stderr = old_out, old_err
payload["stdout"] = out.getvalue()
payload["stderr"] = err.getvalue()
sys.__stdout__.write(json.dumps(payload))
"""


class DockerSandbox:
    """One throwaway container per execution."""

    name = "docker"

    def __init__(self, image: str = DEFAULT_IMAGE, *, docker_path: str | None = None) -> None:
        self.image = image
        self.docker_path = docker_path or shutil.which("docker") or "docker"

    def health(self) -> tuple[bool, str]:
        if shutil.which(self.docker_path) is None:
            return False, "docker not found"
        try:
            probe = subprocess.run(
                [self.docker_path, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"docker probe failed: {exc}"
        if probe.returncode != 0:
            return False, "docker daemon not reachable"
        return True, f"docker {probe.stdout.strip()}"

    def close(self) -> None:
        """Nothing to release: containers do not outlive their execution."""

    def _argv(self, limits: Limits) -> list[str]:
        memory = f"{limits.memory_mb}m"
        return [
            self.docker_path,
            "run",
            "--rm",
            "--interactive",
            "--network=none" if not limits.network else "--network=bridge",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            f"--memory={memory}",
            # Equal to --memory on purpose: with a larger swap allowance a
            # process can exceed the memory ceiling by swapping instead of
            # being killed, which makes the limit advisory.
            f"--memory-swap={memory}",
            f"--cpus={max(limits.cpu_seconds / limits.wall_clock_s, 0.1):.2f}",
            "--user=65534:65534",
            "--workdir=/tmp",
            self.image,
            "python",
            "-c",
            _HARNESS,
        ]

    def execute(self, code: str, limits: Limits) -> SandboxResult:
        """Run ``code`` in a fresh container. Never raises."""
        started = time.perf_counter()

        try:
            completed = subprocess.run(
                self._argv(limits),
                input=code,
                capture_output=True,
                text=True,
                timeout=limits.wall_clock_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # --rm reaps the container when the daemon notices the client is
            # gone, so there is nothing to clean up here.
            return SandboxResult(
                outcome=Outcome.TIMEOUT,
                backend=self.name,
                detail=f"killed after {limits.wall_clock_s:.0f}s",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return SandboxResult(
                outcome=Outcome.BACKEND_ERROR,
                backend=self.name,
                detail=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        duration_ms = (time.perf_counter() - started) * 1000

        if completed.returncode == 137:
            # 128 + SIGKILL. Under a memory cgroup this is the OOM killer.
            return SandboxResult(
                outcome=Outcome.MEMORY,
                backend=self.name,
                detail="container killed (memory limit)",
                duration_ms=duration_ms,
            )

        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            return SandboxResult(
                outcome=Outcome.BACKEND_ERROR,
                backend=self.name,
                detail=(completed.stderr or completed.stdout or "no output")[:500],
                duration_ms=duration_ms,
            )

        stdout = payload.get("stdout", "") or ""
        stderr = payload.get("stderr", "") or ""
        truncated = False
        cap = limits.max_output_bytes
        if len(stdout.encode()) > cap:
            stdout = stdout.encode()[:cap].decode("utf-8", errors="ignore")
            truncated = True
        if len(stderr.encode()) > cap:
            stderr = stderr.encode()[:cap].decode("utf-8", errors="ignore")
            truncated = True

        return SandboxResult(
            outcome=Outcome.OUTPUT_LIMIT if truncated else Outcome.OK,
            stdout=stdout,
            stderr=stderr,
            exception=payload.get("exception"),
            result_repr=payload.get("result_repr"),
            truncated=truncated,
            backend=self.name,
            duration_ms=duration_ms,
        )
