"""The web search tool."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vichara.settings import Settings
from vichara.tools.base import BaseTool, HealthStatus, ToolResult
from vichara.tools.config import OutputTrust, RiskClass
from vichara.tools.websearch.backend import SearchBackend, WebResult
from vichara.tools.websearch.fixture import FixtureSearchBackend
from vichara.tools.websearch.tavily import TavilySearchBackend

SUMMARY = (
    "Search the live web. Use this for anything the textbooks cannot cover: "
    "research published after them, current events, or topics outside "
    "introductory biology. Results are web pages of unverified quality -- treat "
    "them as claims to weigh, not facts to repeat."
)


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description="Search query. Be specific; short keyword queries work better than questions.",
        min_length=2,
        max_length=400,
    )
    max_results: int = Field(default=5, ge=1, le=10, description="How many results.")
    recency_days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Restrict to news from the last N days. Omit unless recency matters.",
    )


class WebSearchTool(BaseTool):
    """Search the web, through whichever backend is configured."""

    name = "web_search"
    summary = SUMMARY
    args_schema = WebSearchArgs
    risk = RiskClass.READ
    output_trust = OutputTrust.UNTRUSTED
    """The highest-risk untrusted input in the system. A search result is a
    page written by anyone, and Phase 5's attack corpus targets exactly this
    path."""

    def __init__(
        self,
        backend: SearchBackend,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        max_output_bytes: int = 16_384,
    ) -> None:
        super().__init__(
            timeout_s=timeout_s, max_retries=max_retries, max_output_bytes=max_output_bytes
        )
        self.backend = backend

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def health(self) -> HealthStatus:
        return self.backend.health()

    def _execute(self, **kwargs: Any) -> ToolResult:
        args = WebSearchArgs.model_validate(kwargs)
        results = self.backend.search(
            args.query, max_results=args.max_results, recency_days=args.recency_days
        )

        if not results:
            return ToolResult(
                tool=self.name,
                ok=True,
                content=(
                    f"No web results for {args.query!r}. Try different search terms once; "
                    "if that also fails, say the information could not be found rather "
                    "than continuing to search."
                ),
                trust=self.output_trust,
                backend=self.backend_name,
            )

        return ToolResult(
            tool=self.name,
            ok=True,
            content=_render(results),
            trust=self.output_trust,
            backend=self.backend_name,
            citations=[r.to_citation() for r in results],
        )


def _render(results: list[WebResult]) -> str:
    return json.dumps(
        [
            {
                "title": r.title,
                "url": r.url,
                "published": r.published_date,
                "snippet": r.content,
            }
            for r in results
        ],
        ensure_ascii=False,
    )


def build_web_search_tool(
    settings: Settings,
    *,
    prefer_recorded: bool = False,
    timeout_s: float = 30.0,
    max_retries: int = 2,
    max_output_bytes: int = 16_384,
) -> WebSearchTool:
    """Choose a backend.

    ``prefer_recorded`` forces replay even when a key is present. Evaluation
    sets it, because a metric computed against the live web cannot be
    reproduced later and therefore cannot be compared against.
    """
    backend: SearchBackend
    if prefer_recorded or not settings.has_tavily_key:
        backend = FixtureSearchBackend()
    else:
        backend = TavilySearchBackend(settings.tavily_api_key.get_secret_value())

    return WebSearchTool(
        backend,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_output_bytes=max_output_bytes,
    )
