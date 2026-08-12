"""Tool failure taxonomy.

The design constraint, stated plainly: **an agent can recover from "rate
limited, retry after 30s" and cannot recover from a bare stack trace.**

So every failure a tool can produce is one of a small number of named kinds,
and each kind carries three things the agent actually needs:

* ``retryable`` -- whether trying the same call again could plausibly work,
* ``retry_after_s`` -- when, if the backend told us,
* ``remediation`` -- one sentence of what to do instead, written *to the
  model*, not to a developer.

The remediation text is the part that is easy to skip and expensive to omit.
"Connection refused" tells the agent nothing; "textbook retrieval is
unavailable; answer from web search and say the answer is not textbook-grounded"
tells it exactly what its next action should be. Phase 4 measures recovery
rate, and recovery rate is largely a function of how good these sentences are.

Exceptions here are internal. They never reach the model directly -- the base
tool converts them into a :class:`~vichara.tools.base.ToolResult` so that a
failure is a value the graph can route on, not a control-flow event that
unwinds the loop.
"""

from __future__ import annotations

import enum


class ErrorCode(enum.StrEnum):
    """Stable identifiers. Logged, counted, and asserted on in tests.

    Stability matters more than expressiveness: these strings end up in
    trajectory records, and an eval that groups failures by code cannot
    tolerate the vocabulary drifting between runs.
    """

    INVALID_ARGUMENTS = "invalid_arguments"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    EXECUTION_FAILED = "execution_failed"
    POLICY_VIOLATION = "policy_violation"
    OUTPUT_TOO_LARGE = "output_too_large"
    UNEXPECTED = "unexpected"


class ToolError(Exception):
    """Base class. Never raised directly -- use a subclass."""

    code: ErrorCode = ErrorCode.UNEXPECTED
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        remediation: str,
        retry_after_s: float | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.remediation = remediation
        self.retry_after_s = retry_after_s
        self.detail = detail
        """Developer-facing context. Logged, never shown to the model -- it is
        the most likely place for a credential or an internal path to leak."""

    def as_model_text(self) -> str:
        """The sentence the model sees. Actionable, and free of internals."""
        parts = [f"{self.code.value}: {self.message}"]
        if self.retry_after_s is not None:
            parts.append(f"Retry after {self.retry_after_s:.0f}s.")
        parts.append(self.remediation)
        return " ".join(parts)


class InvalidArguments(ToolError):
    """The call was malformed. Retrying it unchanged cannot help.

    Distinct from every other error because the fix is the agent's: it must
    change the arguments, not wait or switch tools. Tool-selection precision
    in Phase 4 counts these separately from backend failures, since they mean
    very different things about the agent.
    """

    code = ErrorCode.INVALID_ARGUMENTS
    retryable = False


class RateLimited(ToolError):
    """Quota exhausted or throttled. The one error with a useful deadline."""

    code = ErrorCode.RATE_LIMITED
    retryable = True


class Timeout(ToolError):
    """Exceeded the per-tool wall clock."""

    code = ErrorCode.TIMEOUT
    retryable = True


class BackendUnavailable(ToolError):
    """The service behind the tool could not be reached.

    Retryable because a transient network fault looks identical to a dead
    service from here. The registry's health probe, not this error, is what
    decides a tool is gone for the whole session.
    """

    code = ErrorCode.BACKEND_UNAVAILABLE
    retryable = True


class ExecutionFailed(ToolError):
    """The tool ran and the work itself failed -- a traceback in the sandbox,
    a malformed response from a service.

    Not retryable by default: the same input will fail the same way. The agent
    should change what it asked for.
    """

    code = ErrorCode.EXECUTION_FAILED
    retryable = False


class PolicyViolation(ToolError):
    """A guardrail refused the call: path traversal, disabled network, a
    forbidden operation.

    Never retryable, and deliberately blunt in its remediation. An agent that
    is allowed to believe a refusal was transient will spend its whole step
    budget rephrasing the same forbidden request.
    """

    code = ErrorCode.POLICY_VIOLATION
    retryable = False


class OutputTooLarge(ToolError):
    """The result exceeded the size ceiling and was truncated or dropped."""

    code = ErrorCode.OUTPUT_TOO_LARGE
    retryable = False
