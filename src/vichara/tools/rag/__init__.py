"""Textbook retrieval.

The agent's primary tool, and the one most likely to be unavailable -- so the
backend is chosen at construction from a protocol with two implementations,
and the tool never learns which one it got.
"""

from vichara.tools.rag.backend import RetrievalBackend, RetrievedPassage, SearchResponse
from vichara.tools.rag.fixture import FixtureRetrievalBackend
from vichara.tools.rag.http import HttpRetrievalBackend
from vichara.tools.rag.tool import (
    TextbookSearchArgs,
    TextbookSearchTool,
    build_textbook_tool,
)

__all__ = [
    "FixtureRetrievalBackend",
    "HttpRetrievalBackend",
    "RetrievalBackend",
    "RetrievedPassage",
    "SearchResponse",
    "TextbookSearchArgs",
    "TextbookSearchTool",
    "build_textbook_tool",
]
