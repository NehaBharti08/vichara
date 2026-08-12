"""HTTP client for the VidyaRAG retrieval service.

Every failure mode is mapped onto the error taxonomy rather than surfacing as
an httpx exception, because the agent's next action depends entirely on which
kind of failure this was: a 429 means wait, a 503 means try again, a 404 means
the endpoint contract is wrong and no amount of retrying will fix it.

The contract this speaks is documented in :mod:`vichara.tools.rag.backend`.
"""

from __future__ import annotations

import httpx

from vichara.logging import get_logger
from vichara.tools.base import HealthStatus
from vichara.tools.errors import (
    BackendUnavailable,
    ExecutionFailed,
    RateLimited,
    Timeout,
)
from vichara.tools.rag.backend import RetrievedPassage, SearchResponse

log = get_logger(__name__)

_UNAVAILABLE_REMEDIATION = (
    "Textbook retrieval is unavailable. Use web search instead, and say in your "
    "answer that it is not grounded in the textbook."
)


class HttpRetrievalBackend:
    """Calls a deployed VidyaRAG instance."""

    name = "http"

    def __init__(self, base_url: str, *, timeout_s: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> HealthStatus:
        """Probe ``GET /health``. Never raises -- the registry calls this at
        startup and a probe that throws would defeat graceful degradation."""
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
        except httpx.HTTPError as exc:
            return HealthStatus(
                healthy=False, backend=self.name, detail=f"{type(exc).__name__}: {exc}"
            )
        if response.status_code != httpx.codes.OK:
            return HealthStatus(
                healthy=False, backend=self.name, detail=f"HTTP {response.status_code}"
            )
        return HealthStatus(healthy=True, backend=self.name, detail=self.base_url)

    def search(
        self, query: str, *, top_k: int, book_slug: str | None = None
    ) -> list[RetrievedPassage]:
        payload: dict[str, object] = {"query": query, "top_k": top_k}
        if book_slug:
            payload["book_slug"] = book_slug

        try:
            response = httpx.post(f"{self.base_url}/search", json=payload, timeout=self.timeout_s)
        except httpx.TimeoutException as exc:
            raise Timeout(
                f"Textbook retrieval did not respond within {self.timeout_s:.0f}s.",
                remediation="Try a shorter query, or use web search instead.",
                detail=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailable(
                "Could not reach the textbook retrieval service.",
                remediation=_UNAVAILABLE_REMEDIATION,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        self._raise_for_status(response)

        try:
            return SearchResponse.model_validate(response.json()).results
        except ValueError as exc:
            # A malformed body is not retryable: the same request will produce
            # the same broken response. This is a contract violation, and the
            # agent should stop asking rather than burn its budget.
            raise ExecutionFailed(
                "The textbook retrieval service returned a malformed response.",
                remediation=_UNAVAILABLE_REMEDIATION,
                detail=f"{exc}; body={response.text[:300]!r}",
            ) from exc

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status == httpx.codes.OK:
            return

        if status == httpx.codes.TOO_MANY_REQUESTS:
            raise RateLimited(
                "Textbook retrieval is rate limited.",
                remediation="Wait, then retry the same query.",
                retry_after_s=_retry_after(response),
                detail=response.text[:200],
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise BackendUnavailable(
                f"The textbook retrieval service returned HTTP {status}.",
                remediation=_UNAVAILABLE_REMEDIATION,
                detail=response.text[:200],
            )
        # 4xx other than 429 is our bug, not a transient fault. Retrying a
        # wrong URL or a rejected schema just wastes the step budget.
        raise ExecutionFailed(
            f"The textbook retrieval service rejected the request (HTTP {status}).",
            remediation=_UNAVAILABLE_REMEDIATION,
            detail=response.text[:200],
        )


def _retry_after(response: httpx.Response) -> float | None:
    """Honour the server's own deadline when it gives one."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
