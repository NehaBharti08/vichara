"""Credential scrubbing, applied before anything reaches disk.

A trajectory records full tool arguments, full tool output, and the model's
own text. That is exactly the shape of thing that ends up containing a
credential: a tool error echoing a request URL, an agent writing a key into
code it asked the sandbox to run, a search result containing someone else's
token.

Two layers, in this order:

1. **Known values.** The exact secrets this process holds, taken from
   ``Settings``. This is the strong layer -- it needs no pattern to be right,
   only the value to be known, and it catches a key in any encoding or
   position.
2. **Patterns.** Credential shapes from providers this process does *not* hold
   keys for. Weaker and best-effort, but it is what catches a token that
   arrived in a web page.

Layer 1 alone would miss third-party secrets; layer 2 alone would miss a key
whose format changes. Neither is sufficient.

Redaction is deliberately lossy and irreversible. A trajectory that has lost a
token is a small inconvenience; a trajectory that leaked one into a public
repository means rotating the key and rewriting history.
"""

from __future__ import annotations

import re
from typing import Any

from vichara.settings import Settings

PLACEHOLDER = "[REDACTED:{name}]"
_GENERIC = "[REDACTED]"

# Shapes of credentials from providers this process may not hold keys for.
# Ordered longest-first so a specific pattern wins over a generic one.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
        ),
    ),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("google_oauth", re.compile(r"\bAQ\.[0-9A-Za-z_-]{20,}")),
    ("tavily_key", re.compile(r"\btvly-[0-9A-Za-z_-]{16,}")),
    ("openai_key", re.compile(r"\bsk-[0-9A-Za-z_-]{20,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[0-9A-Za-z]{30,}")),
    ("hf_token", re.compile(r"\bhf_[0-9A-Za-z]{30,}")),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[0-9A-Za-z._~+/=-]{20,}")),
    ("authorization_header", re.compile(r"(?i)\"authorization\"\s*:\s*\"[^\"]{8,}\"")),
)

_MIN_SECRET_LEN = 8
"""Below this, an exact-match replacement does more harm than good -- a short
value appears inside ordinary words and would corrupt the transcript."""


class Redactor:
    """Scrubs credentials from anything on its way to disk."""

    def __init__(self, known: dict[str, str] | None = None) -> None:
        # Longest first: if one secret is a substring of another, replacing the
        # shorter one first would leave a mangled fragment of the longer.
        self._known = sorted(
            (
                (name, value)
                for name, value in (known or {}).items()
                if len(value) >= _MIN_SECRET_LEN
            ),
            key=lambda item: len(item[1]),
            reverse=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Redactor:
        return cls(
            {
                "GOOGLE_API_KEY": settings.google_api_key.get_secret_value(),
                "OPENAI_API_KEY": settings.openai_api_key.get_secret_value(),
                "TAVILY_API_KEY": settings.tavily_api_key.get_secret_value(),
            }
        )

    def text(self, value: str) -> str:
        """Scrub one string."""
        if not value:
            return value
        for name, secret in self._known:
            if secret and secret in value:
                value = value.replace(secret, PLACEHOLDER.format(name=name))
        for _, pattern in _PATTERNS:
            value = pattern.sub(_GENERIC, value)
        return value

    def scrub(self, value: Any) -> Any:
        """Walk any JSON-shaped structure, scrubbing every string in it.

        Keys are scrubbed as well as values: a dict keyed by an API key is
        unusual but not impossible, and the cost of checking is nothing.
        """
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.scrub(k): self.scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.scrub(item) for item in value)
        return value

    def is_clean(self, value: Any) -> bool:
        """Whether scrubbing would change anything. Used by tests and asserts."""
        return bool(self.scrub(value) == value)
