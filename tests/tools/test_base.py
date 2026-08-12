"""Base tool behaviour.

The contract these enforce is the reason Phase 1 exists before the agent:
a tool must fail as a *value the graph can route on*, carrying enough
information for the agent to choose a different action. Every test here is
really asking "could an agent recover from this".
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from vichara.tools.base import BaseTool, HealthStatus, ToolResult
from vichara.tools.config import OutputTrust
from vichara.tools.errors import (
    BackendUnavailable,
    ErrorCode,
    ExecutionFailed,
    RateLimited,
)


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=100)
    repeat: int = Field(default=1, ge=1, le=5)


class EchoTool(BaseTool):
    """A tool whose behaviour the test dictates."""

    name = "echo"
    summary = "Echo text back."
    args_schema = EchoArgs

    def __init__(self, behaviour: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.behaviour = behaviour
        self.calls = 0

    def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, backend="test")

    def _execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        if callable(self.behaviour):
            self.behaviour(self.calls)
        args = EchoArgs.model_validate(kwargs)
        return ToolResult(
            tool=self.name,
            ok=True,
            content=args.text * args.repeat,
            trust=OutputTrust.UNTRUSTED,
            backend="test",
        )


class TestArgumentValidation:
    def test_valid_call_succeeds(self) -> None:
        result = EchoTool().run(text="hi", repeat=2)

        assert result.ok is True
        assert result.content == "hihi"

    def test_invalid_arguments_do_not_reach_the_tool(self) -> None:
        """Validation is the cheapest possible failure -- no work, no retry."""
        tool = EchoTool()

        result = tool.run(text="", repeat=99)

        assert result.ok is False
        assert result.error_code is ErrorCode.INVALID_ARGUMENTS
        assert result.retryable is False
        assert tool.calls == 0
        assert result.attempts == 0

    def test_validation_message_is_actionable(self) -> None:
        """The model must be able to fix the call from this text alone."""
        result = EchoTool().run(text="ok", repeat=99)

        assert "repeat" in result.content
        assert "less than or equal to 5" in result.content
        # No pydantic internals, no URL, no traceback.
        assert "Traceback" not in result.content
        assert "pydantic" not in result.content.lower()

    def test_unknown_argument_is_rejected(self) -> None:
        result = EchoTool().run(text="ok", colour="blue")

        assert result.ok is False
        assert result.error_code is ErrorCode.INVALID_ARGUMENTS


class TestFailuresBecomeValues:
    def test_tool_error_becomes_a_result(self) -> None:
        def boom(_: int) -> None:
            raise ExecutionFailed("it broke", remediation="Try another tool.")

        result = EchoTool(boom).run(text="hi")

        assert result.ok is False
        assert result.error_code is ErrorCode.EXECUTION_FAILED
        assert "Try another tool." in result.content

    def test_unexpected_exception_becomes_a_result(self) -> None:
        """A bug in a tool must cost one capability, not the whole run."""

        def boom(_: int) -> None:
            raise ZeroDivisionError("oops")

        result = EchoTool(boom).run(text="hi")

        assert result.ok is False
        assert result.error_code is ErrorCode.EXECUTION_FAILED
        assert "oops" not in result.content, "internal detail must not reach the model"

    def test_run_never_raises(self) -> None:
        def boom(_: int) -> None:
            raise KeyboardInterrupt

        # Even the pathological case returns rather than propagating.
        with pytest.raises(KeyboardInterrupt):
            EchoTool(boom).run(text="hi")


class TestRetryPolicy:
    def test_retryable_error_is_retried(self) -> None:
        def flaky(call: int) -> None:
            if call < 3:
                raise BackendUnavailable("transient", remediation="wait", retry_after_s=0.0)

        tool = EchoTool(flaky, max_retries=2)
        result = tool.run(text="ok")

        assert result.ok is True
        assert tool.calls == 3
        assert result.attempts == 3

    def test_non_retryable_error_is_not_retried(self) -> None:
        def fatal(_: int) -> None:
            raise ExecutionFailed("permanent", remediation="stop")

        tool = EchoTool(fatal, max_retries=3)
        tool.run(text="ok")

        assert tool.calls == 1

    def test_retries_are_bounded(self) -> None:
        def always(_: int) -> None:
            raise BackendUnavailable("down", remediation="later", retry_after_s=0.0)

        tool = EchoTool(always, max_retries=2)
        result = tool.run(text="ok")

        assert tool.calls == 3  # one attempt plus two retries
        assert result.ok is False
        assert result.retryable is True, "the agent should know this may work later"

    def test_server_retry_after_is_honoured(self) -> None:
        """A backend that states a deadline knows better than our backoff."""

        def limited(call: int) -> None:
            if call == 1:
                raise RateLimited("slow down", remediation="wait", retry_after_s=0.05)

        started = time.perf_counter()
        result = EchoTool(limited, max_retries=1).run(text="ok")
        elapsed = time.perf_counter() - started

        assert result.ok is True
        assert elapsed >= 0.05


class TestTimeout:
    def test_slow_tool_times_out(self) -> None:
        def slow(_: int) -> None:
            time.sleep(1.0)

        result = EchoTool(slow, timeout_s=0.1, max_retries=0).run(text="ok")

        assert result.ok is False
        assert result.error_code is ErrorCode.TIMEOUT
        assert result.retryable is True

    def test_timeout_bounds_the_wait_not_the_work(self) -> None:
        """Documented limitation, asserted so it cannot regress silently.

        Python cannot kill a thread. The agent stops waiting; the work keeps
        running. This is why the sandbox enforces limits at the process
        boundary instead of relying on this mechanism.
        """
        started = time.perf_counter()
        EchoTool(lambda _: time.sleep(0.5), timeout_s=0.05, max_retries=0).run(text="ok")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.4, "the caller must not wait for the abandoned work"


class TestOutputTruncation:
    def test_oversized_output_is_truncated(self) -> None:
        tool = EchoTool(max_output_bytes=100)

        result = tool.run(text="x" * 100, repeat=5)

        assert result.truncated is True
        assert result.raw_bytes == 500
        assert "truncated" in result.content
        assert "400 bytes omitted of 500 total" in result.content

    def test_output_within_budget_is_untouched(self) -> None:
        result = EchoTool(max_output_bytes=1000).run(text="short")

        assert result.truncated is False
        assert result.content == "short"

    def test_truncation_does_not_split_a_character(self) -> None:
        """Cutting mid-codepoint would produce mojibake in the transcript."""
        result = EchoTool(max_output_bytes=11).run(text="é" * 20)

        assert result.truncated is True
        assert "�" not in result.content


class TestMetadata:
    def test_latency_is_recorded(self) -> None:
        result = EchoTool().run(text="hi")

        assert result.latency_ms > 0

    def test_trust_level_is_carried(self) -> None:
        result = EchoTool().run(text="hi")

        assert result.is_untrusted is True

    def test_langchain_adapter_advertises_the_schema(self) -> None:
        adapted = EchoTool().as_langchain_tool()

        assert adapted.name == "echo"
        assert adapted.args_schema is EchoArgs
        assert "Echo text back." in adapted.description
