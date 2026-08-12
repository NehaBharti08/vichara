"""Web search.

Two backends, and unusually the *recorded* one is the default for evaluation:
the live web is not reproducible, so a number measured against it cannot be
re-derived three months later.
"""

from vichara.tools.websearch.backend import SearchBackend, WebResult
from vichara.tools.websearch.fixture import FixtureSearchBackend
from vichara.tools.websearch.tavily import TavilySearchBackend
from vichara.tools.websearch.tool import (
    WebSearchArgs,
    WebSearchTool,
    build_web_search_tool,
)

__all__ = [
    "FixtureSearchBackend",
    "SearchBackend",
    "TavilySearchBackend",
    "WebResult",
    "WebSearchArgs",
    "WebSearchTool",
    "build_web_search_tool",
]
