"""The Tool contract.

Every tool is built and tested here before the agent exists. That ordering is
deliberate: when a trajectory goes wrong later, the question should be "why did
it choose that" and never "did the tool even work".

Three things the base class does so that no individual tool can forget them:

* **Argument validation.** Arguments arrive from a language model, so they are
  untrusted input in the ordinary sense -- wrong types, missing fields,
  hallucinated parameters. Validation happens once, here, and produces an
  :class:`~vichara.tools.errors.InvalidArguments` the agent can act on.
* **Timeouts and retries.** Applied uniformly from config. A tool author who
  forgets a timeout does not get to hang the agent.
* **Failure conversion.** Every exception becomes a :class:`ToolResult` with
  ``ok=False``. A failure must be a *value the graph can route on*, not a
  control-flow event that unwinds the loop -- otherwise "recovery rate after a
  tool failure" is unmeasurable because there is no trajectory left to measure.

A note on the timeout, because the limitation is real and belongs in the open:
Python cannot kill a running thread. The timeout here bounds *how long the
agent waits*, not how long the work runs. An abandoned call keeps executing in
its worker thread until it finishes on its own. That is sufficient for
network-bound tools, whose sockets carry their own deadlines, and it is not
sufficient for arbitrary code -- which is exactly why the sandbox enforces its
limits at the process boundary instead. See docs/THREAT_MODEL.md.
"""

from __future__ import annotations

import abc
import threading
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vichara.logging import get_logger
from vichara.tools.config import OutputTrust, RiskClass
from vichara.tools.errors import (
    ErrorCode,
    ExecutionFailed,
    InvalidArguments,
    Timeout,
    ToolError,
)

log = get_logger(__name__)

_TRUNCATION_NOTICE = "\n\n[... truncated: {dropped} bytes omitted of {total} total ...]"


class Citation(BaseModel):
    """Provenance for one claim.

    Carried separately from the text rather than parsed out of it later. A
    citation recovered by regex from prose is a citation the agent can fake;
    one attached by the tool that fetched the source cannot be.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["textbook", "web", "computation", "file"]
    source: str
    """Human-readable, and the string the answer cites -- e.g.
    "Biology, 4.2 Prokaryotic Cells, p.188" or a page title."""

    locator: str | None = None
    """Machine-checkable pointer: a URL, a chunk id. Used by the grounding
    metric to verify the cited source actually supports the claim."""

    snippet: str | None = None


class HealthStatus(BaseModel):
    """The answer to "would this tool work right now"."""

    model_config = ConfigDict(extra="forbid")

    healthy: bool
    backend: str
    detail: str = ""
    degraded: bool = False
    """Healthy, but not on the backend that was asked for -- a fixture corpus
    standing in for an undeployed service. Reported distinctly because it is a
    supported state, and flattening it into `unhealthy` would make the health
    output something you learn to ignore."""


class ToolResult(BaseModel):
    """What a tool returns. Success and failure use the same shape.

    Deliberately not an exception on the failure path: the graph routes on
    this value, the trajectory records it, and Phase 4 counts it.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    ok: bool
    content: str
    """The text the model sees. On failure this is the actionable remediation
    sentence, never a traceback."""

    trust: OutputTrust
    backend: str
    citations: list[Citation] = Field(default_factory=list)

    error_code: ErrorCode | None = None
    retryable: bool = False
    attempts: int = 1
    latency_ms: float = 0.0
    truncated: bool = False
    raw_bytes: int = 0

    @property
    def is_untrusted(self) -> bool:
        return self.trust is OutputTrust.UNTRUSTED


class BaseTool(abc.ABC):
    """Base for every tool.

    Subclasses implement :meth:`_execute` and :meth:`health` and inherit
    validation, timeout, retry, truncation and failure conversion.
    """

    name: str
    summary: str
    args_schema: type[BaseModel]
    risk: RiskClass = RiskClass.READ
    output_trust: OutputTrust = OutputTrust.UNTRUSTED

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse to define a tool whose schema tolerates unknown arguments.

        Pydantic's default is to ignore an unexpected field. For a tool that
        means a hallucinated parameter is silently dropped: the model believes
        `recency_days=7` took effect, the call runs without it, and the agent
        gets a plausible wrong answer with no signal that anything went wrong.
        Rejecting the call instead turns an invisible failure into one the
        agent can correct, so this is enforced at class-definition time rather
        than left to each tool author to remember.
        """
        super().__init_subclass__(**kwargs)
        schema: type[BaseModel] | None = getattr(cls, "args_schema", None)
        if schema is not None and schema.model_config.get("extra") != "forbid":
            raise TypeError(
                f"{cls.__name__}.args_schema ({schema.__name__}) must set "
                'model_config = ConfigDict(extra="forbid")'
            )

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        max_output_bytes: int = 16_384,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_output_bytes = max_output_bytes

    # -- Subclass contract --------------------------------------------------

    @abc.abstractmethod
    def health(self) -> HealthStatus:
        """Cheap liveness probe. Must not raise; must not cost quota."""

    @abc.abstractmethod
    def _execute(self, **kwargs: Any) -> ToolResult:
        """Do the work. Raise a :class:`ToolError` subclass on failure."""

    @property
    def backend_name(self) -> str:
        return "default"

    # -- Public entry point -------------------------------------------------

    def run(self, **kwargs: Any) -> ToolResult:
        """Validate, execute with timeout and retry, and never raise."""
        started = time.perf_counter()

        try:
            validated = self._validate(kwargs)
        except InvalidArguments as exc:
            return self._failure(exc, attempts=0, started=started)

        attempt = 0
        last: ToolError | None = None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                result = self._run_once(validated)
            except ToolError as exc:
                last = exc
                if not exc.retryable or attempt > self.max_retries:
                    break
                self._backoff(exc, attempt)
                continue
            except Exception as exc:
                log.exception("tool raised an unhandled exception", tool=self.name)
                return self._failure(_unexpected(self.name, exc), attempts=attempt, started=started)
            else:
                return self._finalise(result, attempt=attempt, started=started)

        if last is None:  # pragma: no cover - the loop cannot exit without setting it
            last = _unexpected(self.name, RuntimeError("retry loop exited with no result"))
        return self._failure(last, attempts=attempt, started=started)

    # -- Interop ------------------------------------------------------------

    def as_langchain_tool(self) -> Any:
        """Adapt for ``llm.bind_tools``.

        Binding only advertises the schema; the graph's execute node calls
        :meth:`run` directly so that citations, trust level and error code
        survive into state rather than being flattened into a string.
        """
        from langchain_core.tools import StructuredTool

        def _call(**kwargs: Any) -> str:
            return self.run(**kwargs).content

        return StructuredTool.from_function(
            func=_call,
            name=self.name,
            description=self.summary,
            args_schema=self.args_schema,
        )

    # -- Internals ----------------------------------------------------------

    def _validate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            model = self.args_schema.model_validate(kwargs)
        except ValidationError as exc:
            raise InvalidArguments(
                f"{self.name} received arguments that do not match its schema.",
                remediation=(
                    f"Call {self.name} again with corrected arguments. "
                    f"Problems: {_summarise_validation(exc)}."
                ),
                detail=str(exc),
            ) from exc
        return model.model_dump()

    def _run_once(self, validated: dict[str, Any]) -> ToolResult:
        """Execute with a wall-clock bound on how long the agent waits.

        A bare daemon thread rather than a ThreadPoolExecutor, for two reasons
        that only show up under timeout. An executor used as a context manager
        calls ``shutdown(wait=True)`` on exit, which blocks until the abandoned
        work finishes -- so the timeout would bound nothing at all. And its
        worker threads are non-daemon, so a genuinely hung call would keep the
        interpreter from exiting. ``daemon=True`` means an abandoned call can
        never outlive the process.

        The work itself still runs to completion in the background; only the
        waiting is bounded. See the module docstring.
        """
        outcome: dict[str, Any] = {}

        def target() -> None:
            try:
                outcome["result"] = self._execute(**validated)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
                outcome["error"] = exc

        worker = threading.Thread(target=target, daemon=True, name=f"tool-{self.name}")
        worker.start()
        worker.join(self.timeout_s)

        if worker.is_alive():
            raise Timeout(
                f"{self.name} did not respond within {self.timeout_s:.0f}s.",
                remediation=(
                    "The call may simply be slow. Try once more with a narrower "
                    "request, or use a different tool."
                ),
            )

        if "error" in outcome:
            raise outcome["error"]
        result: ToolResult = outcome["result"]
        return result

    def _backoff(self, exc: ToolError, attempt: int) -> None:
        """Honour the backend's deadline when it gave one, else back off."""
        delay = exc.retry_after_s if exc.retry_after_s is not None else min(2.0**attempt, 8.0)
        log.info(
            "retrying tool", tool=self.name, attempt=attempt, delay_s=delay, code=exc.code.value
        )
        time.sleep(delay)

    def _finalise(self, result: ToolResult, *, attempt: int, started: float) -> ToolResult:
        result.attempts = attempt
        result.latency_ms = (time.perf_counter() - started) * 1000
        return self._truncate(result)

    def _truncate(self, result: ToolResult) -> ToolResult:
        encoded = result.content.encode("utf-8")
        result.raw_bytes = len(encoded)
        if len(encoded) <= self.max_output_bytes:
            return result

        # Cut on a character boundary, not a byte one.
        kept = encoded[: self.max_output_bytes].decode("utf-8", errors="ignore")
        result.content = kept + _TRUNCATION_NOTICE.format(
            dropped=len(encoded) - len(kept.encode("utf-8")), total=len(encoded)
        )
        result.truncated = True
        log.info("tool output truncated", tool=self.name, raw_bytes=result.raw_bytes)
        return result

    def _failure(self, exc: ToolError, *, attempts: int, started: float) -> ToolResult:
        log.warning(
            "tool failed",
            tool=self.name,
            code=exc.code.value,
            attempts=attempts,
            detail=exc.detail,
        )
        return ToolResult(
            tool=self.name,
            ok=False,
            content=exc.as_model_text(),
            trust=self.output_trust,
            backend=self.backend_name,
            error_code=exc.code,
            retryable=exc.retryable,
            attempts=attempts,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def _summarise_validation(exc: ValidationError, limit: int = 3) -> str:
    """Turn a pydantic error into something a model can act on.

    The raw ValidationError is long, nested, and mentions pydantic internals;
    an agent handed that spends a step parsing it instead of fixing the call.
    """
    parts = []
    for err in exc.errors()[:limit]:
        location = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{location}: {err['msg']}")
    remaining = len(exc.errors()) - limit
    if remaining > 0:
        parts.append(f"and {remaining} more")
    return "; ".join(parts)


def _unexpected(tool: str, exc: Exception) -> ToolError:
    """Wrap a tool bug so it reaches the agent as a value, not a crash."""
    return ExecutionFailed(
        f"{tool} failed unexpectedly.",
        remediation="Try a different approach or another tool; this one is not working.",
        detail=f"{type(exc).__name__}: {exc}",
    )
