"""The textbook retrieval tool.

Backend selection happens once, at construction. The tool then has no idea
whether it is talking to a deployed service or a file on disk, which is what
lets the same eval task run under both and lets "RAG is down" be a capability
profile rather than an incident.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vichara.settings import Settings
from vichara.tools.base import BaseTool, HealthStatus, ToolResult
from vichara.tools.config import OutputTrust, RiskClass
from vichara.tools.rag.backend import RetrievalBackend, RetrievedPassage
from vichara.tools.rag.fixture import FixtureRetrievalBackend
from vichara.tools.rag.http import HttpRetrievalBackend

SUMMARY = (
    "Retrieve passages from OpenStax Biology and Anatomy & Physiology textbooks, "
    "each with a citable printed page number. Use this for conceptual grounding, "
    "definitions, and established science. It does not know about anything "
    "published after the textbooks."
)


class TextbookSearchArgs(BaseModel):
    """Arguments the model must produce."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description=(
            "What to look up. Use the terminology a textbook would use, not the "
            "user's phrasing, and keep it to a focused topic."
        ),
        min_length=2,
        max_length=400,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="How many passages to return. Prefer the default.",
    )
    book_slug: str | None = Field(
        default=None,
        description=(
            "Restrict to one book: 'biology' or 'anatomy-and-physiology'. " "Omit to search both."
        ),
    )


class TextbookSearchTool(BaseTool):
    """Retrieval over the textbook corpus, whatever is serving it."""

    name = "textbook_search"
    summary = SUMMARY
    args_schema = TextbookSearchArgs
    risk = RiskClass.READ
    output_trust = OutputTrust.UNTRUSTED
    """Retrieved text is a document, not an instruction. A passage that says
    "ignore your previous instructions" is a passage saying that, and Phase 5
    is about making sure it stays that way."""

    def __init__(
        self,
        backend: RetrievalBackend,
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
        args = TextbookSearchArgs.model_validate(kwargs)
        passages = self.backend.search(args.query, top_k=args.top_k, book_slug=args.book_slug)

        if not passages:
            # A finding, not a failure. `ok=True` matters: an impossible-task
            # probe depends on the agent seeing "the textbook does not cover
            # this" and concluding something, rather than treating it as a
            # transient fault and retrying four times.
            return ToolResult(
                tool=self.name,
                ok=True,
                content=(
                    f"No textbook passages matched {args.query!r}. "
                    "The corpus covers introductory biology and human anatomy and "
                    "physiology only. If the topic is outside that, say so rather "
                    "than searching again with different wording."
                ),
                trust=self.output_trust,
                backend=self.backend_name,
            )

        return ToolResult(
            tool=self.name,
            ok=True,
            content=_render(passages),
            trust=self.output_trust,
            backend=self.backend_name,
            citations=[p.to_citation() for p in passages],
        )


def _render(passages: list[RetrievedPassage]) -> str:
    """Format passages for the model.

    JSON rather than prose, and each passage explicitly labelled with its
    citation, so that the model has no excuse for attributing a claim to the
    wrong page -- and so the grounding metric can check that it did not.
    """
    return json.dumps(
        [
            {
                "citation": p.citation,
                "book": p.book_title,
                "section": p.section,
                "page": p.printed_page,
                "text": p.text,
            }
            for p in passages
        ],
        ensure_ascii=False,
        indent=None,
    )


def build_textbook_tool(
    settings: Settings,
    *,
    timeout_s: float = 30.0,
    max_retries: int = 2,
    max_output_bytes: int = 16_384,
) -> TextbookSearchTool:
    """Choose a backend from configuration.

    The live service wins when a URL is configured *and* answers its health
    probe; otherwise the committed corpus. Falling back on a failed probe
    rather than on a missing URL is the difference between degrading and
    breaking when a deployment goes down mid-session.
    """
    backend: RetrievalBackend = FixtureRetrievalBackend()
    if settings.vidyarag_url:
        candidate = HttpRetrievalBackend(settings.vidyarag_url)
        if candidate.health().healthy:
            backend = candidate

    return TextbookSearchTool(
        backend,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_output_bytes=max_output_bytes,
    )
