"""Retrieval backends.

The agent's primary tool is a textbook retrieval service that, at the time this
was written, did not exist yet. So the dependency is inverted: this module
defines the contract, and VidyaRAG implements it. Two backends satisfy it --
an HTTP client for the live service, and a committed corpus for everything
else -- and the tool cannot tell them apart.

That inversion is the whole reason the project is not blocked on someone
else's deployment, and it is why "the RAG service is down" is a capability
change rather than an outage.

## The HTTP contract

The service must expose::

    POST {base_url}/search
      request:  {"query": str, "top_k": int, "book_slug": str | null}
      response: {"results": [
                   {"chunk_id": str, "text": str, "citation": str,
                    "book_title": str, "chapter": str | null,
                    "section": str | null, "printed_page": str | null,
                    "source_url": str, "score": float}
                 ]}

    GET {base_url}/health  ->  200 when able to serve a query

Field names match VidyaRAG's existing Qdrant payload
(``vidyarag.store.collection.build_payload``) so the endpoint is a projection
of what it already stores, not a translation layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from vichara.tools.base import Citation, HealthStatus


class RetrievedPassage(BaseModel):
    """One retrieved unit, carrying everything a citation needs."""

    model_config = ConfigDict(extra="ignore")
    """``ignore`` rather than ``forbid``: the live service may add payload
    fields, and a retrieval tool that starts rejecting responses because the
    upstream schema grew is a worse failure than one that ignores a field."""

    chunk_id: str
    text: str
    citation: str
    """Pre-rendered by the source, e.g. "Biology, 4.2 Prokaryotic Cells, p.188".
    Not assembled here -- the service that knows the printed page number is the
    one that should format it."""

    book_slug: str = ""
    """Stable book identifier, used to scope a search to one title."""

    book_title: str = ""
    chapter: str | None = None
    section: str | None = None
    printed_page: str | None = None
    source_url: str = ""
    score: float = 0.0

    def to_citation(self) -> Citation:
        return Citation(
            kind="textbook",
            source=self.citation,
            locator=self.chunk_id,
            snippet=self.text[:200],
        )


class SearchResponse(BaseModel):
    """Wire format for ``POST /search``."""

    model_config = ConfigDict(extra="ignore")

    results: list[RetrievedPassage] = Field(default_factory=list)


@runtime_checkable
class RetrievalBackend(Protocol):
    """What the retrieval tool needs. Implemented by fixture and HTTP."""

    name: str

    def health(self) -> HealthStatus:
        """Cheap liveness probe. Must not raise."""
        ...

    def search(
        self, query: str, *, top_k: int, book_slug: str | None = None
    ) -> list[RetrievedPassage]:
        """Return passages ranked most relevant first.

        Raises a :class:`~vichara.tools.errors.ToolError` subclass on failure.
        An empty list is a *result*, not an error -- "the textbook does not
        cover this" is exactly the finding an impossible-task probe needs, and
        turning it into an exception would hide it.
        """
        ...
