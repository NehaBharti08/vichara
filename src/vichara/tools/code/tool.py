"""The code execution tool.

Thin over the sandbox on purpose: everything security-relevant belongs in
:mod:`vichara.sandbox`, where it is tested adversarially, not here.

What this layer does own is *how failure is presented*. A traceback is handed
to the agent verbatim, because reading one and fixing the code is the single
most valuable feedback loop this tool provides. A timeout is presented as a
statement about the code rather than about the sandbox, because the useful
next action is "write something that terminates", not "retry and hope".
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vichara.sandbox.base import Limits, Outcome, Sandbox, SandboxResult
from vichara.settings import PipelineConfig, Settings
from vichara.tools.base import BaseTool, Citation, HealthStatus, ToolResult
from vichara.tools.config import OutputTrust, RiskClass
from vichara.tools.errors import ExecutionFailed

SUMMARY = (
    "Run Python in an isolated sandbox and get back stdout plus the value of "
    "the final expression. numpy, pandas, scipy and sympy are available. There "
    "is no network and no filesystem, so the code must be self-contained: put "
    "any data it needs into the code itself. Use this for every calculation "
    "rather than doing arithmetic in your head."
)


class RunPythonArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        max_length=8_000,
        description=(
            "Python source. print() what you want to see, or end with a bare "
            "expression to have its value returned. Must terminate quickly."
        ),
    )


class RunPythonTool(BaseTool):
    """Executes agent-authored code under resource limits."""

    name = "run_python"
    summary = SUMMARY
    args_schema = RunPythonArgs
    risk = RiskClass.DESTRUCTIVE
    """Routed through the approval interrupt. Not because the sandbox is
    expected to fail, but because 'the agent executed code' is a decision a
    user should be able to see and refuse."""

    output_trust = OutputTrust.UNTRUSTED
    """The output is derived from code the model wrote, and the model may have
    written it after reading a poisoned document. Computation does not launder
    provenance."""

    def __init__(
        self,
        sandbox: Sandbox,
        limits: Limits,
        *,
        timeout_s: float = 30.0,
        max_output_bytes: int = 16_384,
    ) -> None:
        # No retries. Deterministic code fails identically the second time, and
        # a timeout has already cost the full wall clock once.
        super().__init__(timeout_s=timeout_s, max_retries=0, max_output_bytes=max_output_bytes)
        self.sandbox = sandbox
        self.limits = limits

    @property
    def backend_name(self) -> str:
        return self.sandbox.name

    def health(self) -> HealthStatus:
        healthy, detail = self.sandbox.health()
        return HealthStatus(healthy=healthy, backend=self.sandbox.name, detail=detail)

    def _execute(self, **kwargs: Any) -> ToolResult:
        args = RunPythonArgs.model_validate(kwargs)
        result = self.sandbox.execute(args.code, self.limits)

        if result.outcome is Outcome.BACKEND_ERROR:
            # Our fault, not the agent's. Raised rather than returned so it is
            # recorded as a tool failure the recovery metric can see.
            raise ExecutionFailed(
                "The code sandbox is not working.",
                remediation=(
                    "Do not try running code again. Do the calculation by hand, "
                    "show your working, and say it could not be verified."
                ),
                detail=result.detail,
            )

        return ToolResult(
            tool=self.name,
            ok=result.ok,
            content=_render(result),
            trust=self.output_trust,
            backend=self.backend_name,
            citations=(
                [Citation(kind="computation", source=f"sandbox:{self.backend_name}")]
                if result.ok
                else []
            ),
            truncated=result.truncated,
        )


def _render(result: SandboxResult) -> str:
    """Format an execution for the model.

    A timeout and a traceback read very differently on purpose: one says the
    code is wrong, the other says the approach is wrong.
    """
    if result.outcome is Outcome.TIMEOUT:
        return (
            "The code did not finish within the time limit and was stopped. "
            "It probably loops forever or is far too slow. Rewrite it to "
            "terminate quickly -- do not simply run it again."
        )
    if result.outcome is Outcome.MEMORY:
        return (
            "The code ran out of memory and was stopped. Work on a smaller "
            "amount of data, or compute the result without materialising it."
        )

    payload: dict[str, Any] = {}
    if result.stdout:
        payload["stdout"] = result.stdout
    if result.stderr:
        payload["stderr"] = result.stderr
    if result.result_repr is not None:
        payload["value"] = result.result_repr
    if result.exception:
        # Verbatim. The agent is expected to read this and fix its code.
        payload["traceback"] = result.exception
    if result.truncated:
        payload["note"] = "Output was truncated at the size limit."

    if not payload:
        return (
            "The code ran and produced no output. print() the values you want "
            "to see, or end with a bare expression."
        )
    return json.dumps(payload, ensure_ascii=False)


def build_run_python_tool(
    settings: Settings,
    config: PipelineConfig,
    *,
    timeout_s: float = 30.0,
    max_output_bytes: int = 16_384,
) -> RunPythonTool:
    from vichara.sandbox.policy import build_sandbox, limits_from_config

    sandbox = build_sandbox(settings, config)
    limits = limits_from_config(config)
    # The tool's own wait must outlast the sandbox's kill deadline, or the
    # thread-level timeout fires first and the sandbox never gets to report
    # a clean TIMEOUT the agent can act on.
    return RunPythonTool(
        sandbox,
        limits,
        timeout_s=max(timeout_s, limits.wall_clock_s + 10.0),
        max_output_bytes=max_output_bytes,
    )
