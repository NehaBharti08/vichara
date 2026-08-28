"""Verifying that a citation traces back to something a tool returned.

This exists because of a measurement, not a hunch. The Phase 5 injection
sweep put false citation at **ASR 1.00** -- every attack that told the agent to
cite a fabricated source succeeded. That is the worst possible result for this
project specifically, because grounded citable answers are the whole value
proposition, and an answer looks *more* credible for carrying a citation.

No filter fixes it. A payload asking for a citation is indistinguishable from
a document legitimately naming its source. What does fix it is refusing to
accept any citation the tools did not actually produce: the agent may only
cite what it retrieved, and anything else is removed before the answer is
shown.

The check is deliberately one-directional. It does not ask "is this claim
supported by this passage" -- that needs a judge and is the one genuinely
judged metric in the project. It asks the far narrower question "did this
source string come out of a tool", which is mechanical and cannot be argued
with.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from vichara.logging import get_logger
from vichara.tools.base import Citation

log = get_logger(__name__)

REMOVED = "[unverified citation removed]"

# Citation-shaped spans. Each is a form the agent actually emits, or that an
# injection payload asks it to emit.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # "Smith et al., Journal of X, 2025" -- the shape used by the corpus's
    # cite-fake-url attack.
    ("author_et_al", re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\s+et\s+al\.[^.)\n]*", re.U)),
    # A bare DOI.
    ("doi", re.compile(r"\bdoi:\s*\S+|\b10\.\d{4,}/\S+", re.I)),
    # A URL.
    ("url", re.compile(r"https?://\S+")),
    # "Anatomy and Physiology, 12.4. The Action Potential, p.524"
    ("book_locator", re.compile(r"\b[A-Z][\w &]+,\s*\d+\.\d+[^,\n]*,\s*p\.?\s*\d+", re.U)),
    # "section 4.2, p.9999" and similar fragments.
    ("section_page", re.compile(r"\bsection\s+[\w.-]+,\s*p\.?\s*\d+", re.I)),
)


class CitationAudit(BaseModel):
    """What the answer cited, and whether each was real."""

    model_config = ConfigDict(extra="forbid")

    verified: list[str] = []
    fabricated: list[str] = []
    answer: str = ""
    """The answer with fabricated spans replaced. Unchanged when nothing was
    fabricated, so the common case costs nothing."""

    @property
    def clean(self) -> bool:
        return not self.fabricated


def _normalise(text: str) -> str:
    """Fold whitespace and punctuation so formatting differences do not read
    as fabrication. The model reformats a citation constantly -- 'p.524' and
    'p. 524' are the same source and must not be scored differently."""
    return re.sub(r"[\s.,;:()\[\]]+", " ", text).strip().lower()


def _is_supported(span: str, allowed: list[str]) -> bool:
    """Whether a cited span traces to a source a tool returned.

    Containment in either direction: the agent commonly shortens a citation
    ('Anatomy and Physiology, 12.4') or extends it with a page number, and
    both are honest renderings of the same retrieved source.
    """
    candidate = _normalise(span)
    if len(candidate) < 6:
        return True
    return any(candidate in source or source in candidate for source in allowed if source)


def audit(answer: str, citations: list[Citation]) -> CitationAudit:
    """Find citation-shaped spans in the answer that no tool produced."""
    allowed = [_normalise(c.source) for c in citations]
    allowed += [_normalise(c.locator) for c in citations if c.locator]

    verified: list[str] = []
    fabricated: list[str] = []
    cleaned = answer

    for _rule, pattern in _PATTERNS:
        for match in pattern.finditer(answer):
            span = match.group(0).strip()
            if _is_supported(span, allowed):
                verified.append(span)
            elif span not in fabricated:
                fabricated.append(span)

    for span in fabricated:
        cleaned = cleaned.replace(span, REMOVED)

    if fabricated:
        log.warning("fabricated citations removed", count=len(fabricated), spans=fabricated[:3])

    return CitationAudit(
        verified=sorted(set(verified)),
        fabricated=fabricated,
        answer=cleaned,
    )
