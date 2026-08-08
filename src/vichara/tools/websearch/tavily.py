"""Tavily web search client.

Free tier, no card. The interesting design question here is not the HTTP call
but what happens when the monthly credit allowance runs out mid-evaluation:
Tavily answers 432 for an exhausted plan, which is neither a rate limit that
will clear in thirty seconds nor a transient fault worth retrying. It is
mapped to a non-retryable error with a remediation that tells the agent to
stop trying and work from the textbook instead.
"""

from __future__ import annotations

import httpx

from vichara.logging import get_logger
from vichara.tools.base import HealthStatus
from vichara.tools.errors import (
    BackendUnavailable,
    ExecutionFailed,
    PolicyViolation,
    RateLimited,
    Timeout,
)
from vichara.tools.websearch.backend import WebResult

log = get_logger(__name__)

ENDPOINT = "https://api.tavily.com/search"

_FALLBACK = (
    "Web search is unavailable. Use the textbook instead, and say in your answer "
    "that you could not check for more recent sources."
)

# Tavily returns 432 when the plan's credits are exhausted. Not in httpx.codes.
_PLAN_EXHAUSTED = 432


class TavilySearchBackend:
    """Calls the Tavily API."""

    name = "tavily"

    def __init__(self, api_key: str, *, timeout_s: float = 20.0) -> None:
        self._api_key = api_key
        self.timeout_s = timeout_s

    def health(self) -> HealthStatus:
        """Key presence only.

        Deliberately does not call the API: a health probe that spends a
        credit on every startup would consume a meaningful share of a 1000-call
        monthly allowance across a development session.
        """
        if not self._api_key:
            return HealthStatus(healthy=False, backend=self.name, detail="TAVILY_API_KEY not set")
        return HealthStatus(healthy=True, backend=self.name, detail="key present (not probed)")

    def search(
        self, query: str, *, max_results: int, recency_days: int | None = None
    ) -> list[WebResult]:
        payload: dict[str, object] = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        if recency_days is not None:
            # `days` only applies to the news topic in Tavily's API; setting one
            # without the other silently returns undated general results, which
            # would make a recency-sensitive eval task quietly meaningless.
            payload["topic"] = "news"
            payload["days"] = recency_days

        try:
            response = httpx.post(
                ENDPOINT,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise Timeout(
                f"Web search did not respond within {self.timeout_s:.0f}s.",
                remediation="Try a shorter query, or use the textbook instead.",
                detail=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailable(
                "Could not reach the web search service.",
                remediation=_FALLBACK,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        self._raise_for_status(response)

        try:
            body = response.json()
            raw = body.get("results", []) if isinstance(body, dict) else []
            return [WebResult.model_validate(item) for item in raw]
        except (ValueError, TypeError) as exc:
            raise ExecutionFailed(
                "Web search returned a malformed response.",
                remediation=_FALLBACK,
                detail=f"{exc}; body={response.text[:300]!r}",
            ) from exc

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status == httpx.codes.OK:
            return

        if status == httpx.codes.TOO_MANY_REQUESTS:
            raise RateLimited(
                "Web search is rate limited.",
                remediation="Wait, then retry the same query.",
                retry_after_s=_retry_after(response) or 30.0,
                detail=response.text[:200],
            )
        if status == _PLAN_EXHAUSTED:
            # Not retryable, and saying so matters: an agent that believes this
            # is transient will spend its entire remaining budget waiting for a
            # quota that resets next month.
            raise PolicyViolation(
                "The web search quota for this account is exhausted.",
                remediation=_FALLBACK,
                detail=response.text[:200],
            )
        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise PolicyViolation(
                "Web search rejected the API key.",
                remediation=_FALLBACK,
                detail=f"HTTP {status}",
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise BackendUnavailable(
                f"Web search returned HTTP {status}.",
                remediation=_FALLBACK,
                detail=response.text[:200],
            )
        raise ExecutionFailed(
            f"Web search rejected the request (HTTP {status}).",
            remediation=_FALLBACK,
            detail=response.text[:200],
        )


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
