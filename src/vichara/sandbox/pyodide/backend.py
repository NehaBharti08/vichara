"""Pyodide-in-Node sandbox backend.

The default everywhere: local development, CI, and the deployed Space. It
needs no daemon, which is the only reason a Hugging Face Space can run it at
all, and its isolation is a property of WebAssembly rather than of
configuration.

**The warm worker is disposable, and that is load-bearing.** One Node process
is kept per sandbox instance so that execution costs ~50ms instead of the 1-2s
Pyodide takes to initialise. But the wall clock cannot be enforced inside the
runtime -- a synchronous Python loop in WebAssembly never yields to the
JavaScript event loop, so no timer there can interrupt it. The only mechanism
that stops ``while True: pass`` is this process killing that one. So a timeout
destroys the worker, and the next execution pays the cold start again. Slow on
the failure path, correct on it, which is the right way round.

The subprocess environment is scrubbed. Layer 1 in ``runner.mjs`` makes the
host environment unreachable through the JS bridge; scrubbing makes it absent
in the first place. Either alone would probably hold. Together they mean the
claim survives one of them being wrong.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from vichara.logging import get_logger
from vichara.sandbox.base import Limits, Outcome, SandboxResult

log = get_logger(__name__)

RUNNER = Path(__file__).parent / "runner.mjs"
REPO_ROOT = Path(__file__).resolve().parents[4]

_STARTUP_TIMEOUT_S = 60.0
"""Pyodide unpacks a large WebAssembly bundle on first start. Generous,
because exceeding it produces a confusing 'sandbox unavailable' rather than
an obviously slow one."""

# The worker's environment is built from nothing and filled only from this
# list. An allowlist rather than a denylist, because a denylist has to predict
# every future secret and this has to predict only what Node needs to boot.
#
# It is an allowlist rather than a genuinely empty environment for a reason
# found the hard way: on Windows, `env={}` makes Node abort during startup
# with `Assertion failed: ncrypto::CSPRNG(nullptr, 0)` -- it cannot seed its
# random number generator without SystemRoot to locate the platform crypto
# provider. Security work meeting platform reality; the compromise is named
# here rather than quietly widened.
_ENV_ALLOWLIST_WINDOWS = (
    "SystemRoot",
    "SystemDrive",
    "WINDIR",
    "TEMP",
    "TMP",
    "PATH",
    "PATHEXT",
    "COMSPEC",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)
_ENV_ALLOWLIST_POSIX = ("PATH", "LANG", "LC_ALL", "TMPDIR")


def worker_env() -> dict[str, str]:
    """The environment the sandbox worker is given.

    Exposed rather than inlined so the adversarial suite can assert directly
    that no credential survives into it -- a claim about what an attacker can
    read should be tested, not asserted in a comment.
    """
    names = _ENV_ALLOWLIST_WINDOWS if os.name == "nt" else _ENV_ALLOWLIST_POSIX
    return {name: os.environ[name] for name in names if name in os.environ}


class PyodideSandbox:
    """Runs Python inside WebAssembly, hosted by Node."""

    name = "pyodide"

    def __init__(
        self,
        *,
        node_path: str | None = None,
        runner: Path | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.node_path = node_path or shutil.which("node") or "node"
        self.runner = runner or RUNNER
        # Node resolves `import "pyodide"` from the *script's* directory
        # upwards, so the worker must run somewhere under the repo root where
        # node_modules lives -- not from wherever the agent was invoked.
        self.cwd = cwd or REPO_ROOT
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    # -- Lifecycle ----------------------------------------------------------

    def health(self) -> tuple[bool, str]:
        if shutil.which(self.node_path) is None and not Path(self.node_path).exists():
            return False, "node runtime not found"
        if not self.runner.exists():
            return False, f"runner missing: {self.runner}"
        if not (self.cwd / "node_modules" / "pyodide").exists():
            return False, "pyodide not installed (run `npm install`)"
        return True, f"node at {self.node_path}"

    def close(self) -> None:
        with self._lock:
            self._kill()

    def _kill(self) -> None:
        """Caller holds the lock."""
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            log.warning("sandbox worker did not exit cleanly")

    def _ensure_worker(self) -> subprocess.Popen[str]:
        """Caller holds the lock."""
        if self._process is not None and self._process.poll() is None:
            return self._process

        log.info("starting sandbox worker", backend=self.name)
        self._process = subprocess.Popen(
            [self.node_path, str(self.runner)],
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            # The scrub. The parent holds GOOGLE_API_KEY and TAVILY_API_KEY;
            # the worker is given an allowlist that contains neither, so there
            # is nothing to read even if the JS bridge were somehow restored.
            env=worker_env(),
        )

        handshake = self._read_line(self._process, _STARTUP_TIMEOUT_S)
        if handshake is None or not handshake.get("ready"):
            detail = (handshake or {}).get("error", "worker did not become ready")
            self._kill()
            raise RuntimeError(f"sandbox worker failed to start: {detail}")

        log.info("sandbox worker ready", pyodide=handshake.get("version"))
        return self._process

    # -- Execution ----------------------------------------------------------

    def execute(self, code: str, limits: Limits) -> SandboxResult:
        """Run ``code``. Never raises."""
        started = time.perf_counter()
        with self._lock:
            try:
                worker = self._ensure_worker()
            except (OSError, RuntimeError) as exc:
                return SandboxResult(
                    outcome=Outcome.BACKEND_ERROR,
                    backend=self.name,
                    detail=str(exc),
                    duration_ms=(time.perf_counter() - started) * 1000,
                )

            request = {
                "id": uuid.uuid4().hex,
                "code": code,
                "max_output_bytes": limits.max_output_bytes,
            }

            try:
                assert worker.stdin is not None
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
            except (OSError, ValueError) as exc:
                self._kill()
                return SandboxResult(
                    outcome=Outcome.BACKEND_ERROR,
                    backend=self.name,
                    detail=f"could not reach worker: {exc}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )

            response = self._read_line(worker, limits.wall_clock_s)

            if response is None:
                # The only mechanism that stops a synchronous WASM loop.
                self._kill()
                return SandboxResult(
                    outcome=Outcome.TIMEOUT,
                    backend=self.name,
                    detail=f"killed after {limits.wall_clock_s:.0f}s",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )

        return SandboxResult(
            outcome=Outcome(str(response.get("outcome", "backend_error"))),
            stdout=response.get("stdout") or "",
            stderr=response.get("stderr") or "",
            exception=response.get("exception"),
            result_repr=response.get("result_repr"),
            truncated=bool(response.get("truncated")),
            backend=self.name,
            detail=response.get("detail", ""),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def _read_line(self, process: subprocess.Popen[str], timeout_s: float) -> dict[str, Any] | None:
        """Read one JSON line, or return None if the deadline passes.

        A reader thread rather than a blocking read, because the whole point
        is to stop waiting on a worker that will never answer -- and on
        Windows there is no select() for pipes to do it any other way.
        """
        assert process.stdout is not None
        box: list[str] = []

        def read() -> None:
            try:
                line = process.stdout.readline()  # type: ignore[union-attr]
            except (OSError, ValueError):
                return
            if line:
                box.append(line)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(timeout_s)

        if not box:
            return None
        try:
            parsed: dict[str, Any] = json.loads(box[0])
        except json.JSONDecodeError:
            return None
        return parsed
