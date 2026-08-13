"""Deliberate tool failures.

Recovery rate is only measurable if failures can be caused on demand. Waiting
for a real service to break gives you no control over which kind of breakage
you observed, and the kinds differ in what they test.

``plausible_but_wrong`` is the interesting one. A timeout is obvious and every
agent notices it; a tool that returns confident, well-formed, incorrect data
tests whether the agent checks anything at all -- and most do not. It is the
fault most likely to produce a wrong answer the agent is sure about.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass
from typing import Any

from vichara.logging import get_logger
from vichara.tools.base import BaseTool, ToolResult
from vichara.tools.config import OutputTrust
from vichara.tools.errors import RateLimited, Timeout

log = get_logger(__name__)


class FaultKind(enum.StrEnum):
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PLAUSIBLE_BUT_WRONG = "plausible_but_wrong"
    RATE_LIMITED = "rate_limited"
    EMPTY = "empty"


@dataclass(frozen=True)
class FaultSpec:
    """What to break, how, and how often."""

    tool: str
    kind: FaultKind
    probability: float = 1.0
    first_n_calls: int | None = 1
    """Fail only the first N calls. The default of 1 is what makes recovery
    *possible* -- a tool that always fails measures persistence, not recovery,
    because there is nothing to recover to."""

    seed: int = 0


def wrap_with_fault(tool: BaseTool, spec: FaultSpec) -> BaseTool:
    """Monkey-patch a tool instance to misbehave. Mutates in place.

    Wraps ``run`` rather than ``_execute`` so the fault is applied *after* the
    base class's retry policy, not before it. Injecting beneath the retries
    would let the tool quietly recover on its own and the agent would never
    see a failure to recover from.
    """
    original = tool.run
    rng = random.Random(spec.seed)
    calls = {"n": 0}

    def faulty(**kwargs: Any) -> ToolResult:
        calls["n"] += 1
        within_window = spec.first_n_calls is None or calls["n"] <= spec.first_n_calls
        if within_window and rng.random() < spec.probability:
            log.info("injecting fault", tool=tool.name, kind=spec.kind.value, call=calls["n"])
            return _fault_result(tool, spec)
        return original(**kwargs)

    tool.run = faulty  # type: ignore[method-assign]
    return tool


def _fault_result(tool: BaseTool, spec: FaultSpec) -> ToolResult:
    if spec.kind is FaultKind.TIMEOUT:
        error = Timeout(
            f"{tool.name} did not respond in time.",
            remediation="Try a narrower request, or use a different tool.",
        )
        return _from_error(tool, error, retryable=True)

    if spec.kind is FaultKind.RATE_LIMITED:
        error = RateLimited(  # type: ignore[assignment]
            f"{tool.name} is rate limited.",
            remediation="Wait, then retry the same query.",
            retry_after_s=30.0,
        )
        return _from_error(tool, error, retryable=True)

    if spec.kind is FaultKind.MALFORMED:
        return ToolResult(
            tool=tool.name,
            ok=True,
            content='{"results": [{"citation": ',
            trust=OutputTrust.UNTRUSTED,
            backend="fault",
        )

    if spec.kind is FaultKind.EMPTY:
        return ToolResult(
            tool=tool.name,
            ok=True,
            content="No results.",
            trust=OutputTrust.UNTRUSTED,
            backend="fault",
        )

    # plausible_but_wrong: well-formed, confident, and false. No error flag,
    # no malformed JSON, nothing structural to notice -- the only signal is
    # that the content contradicts what the agent should already know.
    return ToolResult(
        tool=tool.name,
        ok=True,
        content=(
            '[{"citation": "Anatomy and Physiology, 12.4. The Action Potential, p.524", '
            '"text": "The resting membrane potential of a typical neuron is +450 mV, '
            "maintained by the calcium-magnesium exchanger, which moves 7 calcium ions "
            'inward for every 2 magnesium ions outward."}]'
        ),
        trust=OutputTrust.UNTRUSTED,
        backend="fault",
    )


def _from_error(tool: BaseTool, error: Any, *, retryable: bool) -> ToolResult:
    return ToolResult(
        tool=tool.name,
        ok=False,
        content=error.as_model_text(),
        trust=tool.output_trust,
        backend="fault",
        error_code=error.code,
        retryable=retryable,
    )
