"""Per-call cost and latency accounting.

Every model call is measured, because Phase 4 reports cost and latency
distributions and you cannot report what you did not record at the time.

The unit of account is **requests, not dollars**. On a free tier the scarce
resource is requests per day, and a budget denominated in money would show
$0.00 while the agent silently exhausted its quota at two in the morning.
Dollars are still estimated so the guardrail is already real on the day the
provider changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

# USD per million tokens. Free-tier models are priced at zero on purpose: the
# estimate should read 0.00 there rather than pretending otherwise, and the
# request counter is what actually binds.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.0, 0.0),
    "gemini-3.5-flash": (0.0, 0.0),
    "gemini-2.5-flash-lite": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class CallRecord(BaseModel):
    """One model call, as it happened."""

    model_config = ConfigDict(extra="forbid")

    model: str
    role: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    """Prompt tokens the provider served from its own cache. Reported because
    it is the difference between a cheap step and an expensive one, and
    because it is free information the provider already sends."""

    latency_ms: float = 0.0
    cache_hit: bool = False
    """Served from *our* SQLite cache -- no request was made at all."""

    est_usd: float = 0.0
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Price a call. Unknown models cost nothing rather than guessing."""
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


def usage_from_response(response: Any) -> tuple[int, int, int]:
    """Pull ``(input, output, cached)`` out of a LangChain response.

    Defensive because ``usage_metadata`` is optional in the interface and
    absent on some providers -- and a missing token count must degrade the
    accounting, never break the run.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0, 0, 0
    details = usage.get("input_token_details") or {}
    cached = details.get("cache_read", 0) if isinstance(details, dict) else 0
    return (
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
        int(cached or 0),
    )


@dataclass
class Ledger:
    """Running totals for one task.

    Held by the graph and read by the budget guardrail, so the ceiling is
    checked against measured usage rather than an estimate of it.
    """

    calls: list[CallRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def record(self, call: CallRecord) -> CallRecord:
        self.calls.append(call)
        return call

    @property
    def requests(self) -> int:
        """Calls that actually reached the provider. The binding budget."""
        return sum(1 for c in self.calls if not c.cache_hit)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def est_usd(self) -> float:
        return sum(c.est_usd for c in self.calls)

    @property
    def wall_clock_s(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def cache_hits(self) -> int:
        return sum(1 for c in self.calls if c.cache_hit)

    def summary(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "total_tokens": self.total_tokens,
            "est_usd": round(self.est_usd, 6),
            "wall_clock_s": round(self.wall_clock_s, 2),
        }
