"""The Sandbox contract.

Two backends satisfy it -- Pyodide-in-Node and Docker -- and the code tool
never learns which one it got. What each actually stops, and more importantly
what neither stops, is documented in docs/THREAT_MODEL.md.

One design decision worth stating here because it shapes everything else:
**a sandbox failure is a result, not an exception.** Code that raises, loops
forever, or gets killed for exceeding a limit all produce a
:class:`SandboxResult` with the traceback or the reason in it. The agent is
supposed to read a traceback and fix its code -- that is the single most
common legitimate use of this tool -- so turning an error into a control-flow
event would destroy the loop's most valuable feedback signal.
"""

from __future__ import annotations

import enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Outcome(enum.StrEnum):
    """How an execution ended. Counted directly by the Phase 4 metrics."""

    OK = "ok"
    """Ran to completion. The code may still have raised -- see ``exception``."""

    TIMEOUT = "timeout"
    """Exceeded the wall clock and was killed at the process boundary."""

    MEMORY = "memory"
    """Exhausted the memory ceiling."""

    OUTPUT_LIMIT = "output_limit"
    """Produced more output than the cap allowed; what remains is truncated."""

    BLOCKED = "blocked"
    """Refused before running -- a forbidden import, a policy violation."""

    BACKEND_ERROR = "backend_error"
    """The sandbox itself failed: no runtime, a crashed worker, a bad install.
    Distinct from every other outcome because it is *our* fault, not the
    agent's, and the agent should stop using the tool rather than retry."""


class Limits(BaseModel):
    """Resource ceilings for one execution."""

    model_config = ConfigDict(extra="forbid")

    wall_clock_s: float = Field(default=10.0, gt=0)
    """Enforced by the parent killing the child.

    Deliberately not enforced inside the runtime: a synchronous Python loop
    inside WebAssembly never yields to the JavaScript event loop, so a
    JS-side timer can never fire to interrupt it. Only the process boundary
    can stop `while True: pass`. This is the reason the warm worker is
    disposable -- see the Pyodide backend."""

    cpu_seconds: float = Field(default=5.0, gt=0)
    memory_mb: int = Field(default=256, gt=0)
    max_output_bytes: int = Field(default=65_536, gt=0)
    network: bool = False
    """False everywhere, always. Present as a field so the threat model can
    point at a value rather than an absence, and so a future backend that
    genuinely needs egress has to opt in explicitly and visibly."""


class SandboxResult(BaseModel):
    """What one execution produced."""

    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    stdout: str = ""
    stderr: str = ""
    exception: str | None = None
    """The formatted traceback, when the code raised. Handed to the agent
    verbatim: reading a traceback and fixing the code is the point."""

    result_repr: str | None = None
    """``repr`` of the final expression, when there is one."""

    duration_ms: float = 0.0
    truncated: bool = False
    backend: str = ""
    detail: str = ""
    """Backend-side explanation for a non-OK outcome. Logged, and included in
    what the agent sees only when it is actionable."""

    @property
    def ok(self) -> bool:
        """Ran to completion without raising."""
        return self.outcome is Outcome.OK and self.exception is None


@runtime_checkable
class Sandbox(Protocol):
    """Executes untrusted Python under resource limits."""

    name: str

    def health(self) -> tuple[bool, str]:
        """``(healthy, detail)``. Must not raise, must not execute user code."""
        ...

    def execute(self, code: str, limits: Limits) -> SandboxResult:
        """Run ``code`` and return what happened. Must not raise."""
        ...

    def close(self) -> None:
        """Release any worker process. Idempotent."""
        ...
