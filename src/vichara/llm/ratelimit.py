"""Client-side rate limiting.

A free tier throttles rather than refuses, so an evaluation sweep that fires
as fast as it can spends most of its time collecting 429s and backing off.
Self-pacing to just under the published limit finishes sooner than not pacing
at all -- the counter-intuitive result that makes this worth having.

Two mechanisms, doing different jobs:

* a **token bucket** paces requests per minute, so a sweep runs at a steady
  rate rather than in bursts;
* **exponential backoff with jitter** handles the 429s that arrive anyway,
  because the published limit is not the only limit.

Jitter matters more than it looks. Without it, several workers throttled at
the same moment retry at the same moment, and the burst that caused the limit
reproduces itself exactly.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from vichara.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Conservative defaults for a Gemini free tier. Deliberately below the
# published ceiling: the published number is where requests start failing,
# not where they start being a good idea.
DEFAULT_RPM = 12
MAX_BACKOFF_S = 64.0


class TokenBucket:
    """Paces calls to at most ``rate_per_minute``, blocking when empty."""

    def __init__(self, rate_per_minute: int = DEFAULT_RPM, burst: int | None = None) -> None:
        self.rate_per_second = max(rate_per_minute, 1) / 60.0
        self.capacity = float(burst if burst is not None else max(rate_per_minute // 2, 1))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout_s: float = 300.0) -> bool:
        """Block until a token is free. ``False`` if the deadline passes."""
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate_per_second
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                shortfall = (1.0 - self._tokens) / self.rate_per_second

            if time.monotonic() + shortfall > deadline:
                return False
            time.sleep(min(shortfall, 5.0))


def is_rate_limit(exc: BaseException) -> bool:
    """Recognise a throttle across providers.

    String matching because each SDK raises its own exception type and this
    module deliberately imports none of them -- the provider abstraction is
    only worth having if the plumbing around it stays vendor-neutral.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "resource_exhausted", "quota", "too many requests")
    )


def retry_after(exc: BaseException) -> float | None:
    """Read a server-supplied delay when the exception carries one."""
    for attribute in ("retry_after", "retry_delay"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int | float) and value > 0:
            return float(value)
    return None


def call_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    bucket: TokenBucket | None = None,
) -> T:
    """Run ``fn``, pacing and retrying on throttles only.

    Non-rate-limit errors propagate immediately. Retrying a malformed request
    or a bad key burns quota to reproduce the same failure.
    """
    attempt = 0
    while True:
        attempt += 1
        if bucket is not None:
            bucket.acquire()
        try:
            return fn()
        except Exception as exc:
            if not is_rate_limit(exc) or attempt >= max_attempts:
                raise
            delay = retry_after(exc) or min(2.0**attempt, MAX_BACKOFF_S)
            # Jitter breaks the synchronised-retry stampede that recreates the
            # burst which caused the throttle in the first place.
            delay *= 0.5 + random.random()
            log.info(
                "rate limited, backing off",
                attempt=attempt,
                delay_s=round(delay, 1),
                error=str(exc)[:120],
            )
            time.sleep(delay)
