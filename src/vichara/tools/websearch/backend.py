"""Web search backends.

Same two-implementation pattern as retrieval, for a different reason. Retrieval
has a fixture backend because the service might not be deployed; search has one
because **the live web is not reproducible**. An eval task whose answer depends
on what Google indexed this morning cannot be re-run in three months and
compared, and a metric you cannot re-derive is not a metric.

So the recorded backend is not a fallback for a missing key. It is the backend
the evaluation of record should usually run against, with live search reserved
for demonstrating that the tool works end to end.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from vichara.tools.base import Citation, HealthStatus


class WebResult(BaseModel):
    """One search hit."""

    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    content: str
    """The extracted snippet. Untrusted text from a page nobody vetted -- this
    is the single most likely carrier of an injection payload in the system."""

    score: float = 0.0
    published_date: str | None = None

    def to_citation(self) -> Citation:
        return Citation(
            kind="web",
            source=self.title or self.url,
            locator=self.url,
            snippet=self.content[:200],
        )


@runtime_checkable
class SearchBackend(Protocol):
    """What the web search tool needs."""

    name: str

    def health(self) -> HealthStatus:
        """Cheap liveness probe. Must not raise, must not spend quota."""
        ...

    def search(
        self, query: str, *, max_results: int, recency_days: int | None = None
    ) -> list[WebResult]:
        """Return hits ranked most relevant first.

        An empty list is a result, not an error.
        """
        ...
