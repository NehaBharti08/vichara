"""Offline retrieval over the committed OpenStax corpus.

Ranks with BM25. That is a deliberate downgrade from the live service's dense
retrieval, chosen for three properties the fixture backend needs and dense
retrieval cannot give it: it is **deterministic** (the same query always
returns the same passages, so a failing eval task is reproducible), it needs
**no model weights** (no 210MB ONNX download in CI, no cold-start penalty in a
test), and it has **no network dependency at all**.

The cost is that it does not rank identically to the real thing. That is
stated in data/fixtures/ATTRIBUTION.md and in any results table produced
against it, rather than glossed over.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections import Counter
from pathlib import Path

from vichara.logging import get_logger
from vichara.settings import REPO_ROOT
from vichara.tools.base import HealthStatus
from vichara.tools.errors import BackendUnavailable
from vichara.tools.rag.backend import RetrievedPassage

log = get_logger(__name__)

DEFAULT_CORPUS = REPO_ROOT / "data" / "fixtures" / "rag_corpus.jsonl"

_TOKEN = re.compile(r"[a-z0-9]+")

# Okapi BM25 defaults. Not tuned -- tuning a fixture backend's ranking against
# the eval set would be optimising the stand-in rather than the system.
_K1 = 1.5
_B = 0.75

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how in into is it its of on or
    that the their there these this to was were what when where which who why with""".split()
)


def _tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class _Index:
    """An in-memory BM25 index. Built once per corpus file, then cached."""

    def __init__(self, passages: list[RetrievedPassage]) -> None:
        self.passages = passages
        self.tokens = [_tokenise(f"{p.section or ''} {p.text}") for p in passages]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.counts = [Counter(t) for t in self.tokens]

        document_frequency: Counter[str] = Counter()
        for token_set in (set(t) for t in self.tokens):
            document_frequency.update(token_set)

        total = len(passages)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int, book_slug: str | None) -> list[RetrievedPassage]:
        terms = _tokenise(query)
        if not terms:
            return []

        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self.counts):
            if book_slug and self.passages[index].book_slug != book_slug:
                continue
            length = self.lengths[index] or 1
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + _K1 * (1 - _B + _B * length / (self.avg_length or 1))
                score += self.idf.get(term, 0.0) * frequency * (_K1 + 1) / denominator
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        results = []
        for score, index in scored[:top_k]:
            passage = self.passages[index].model_copy(update={"score": round(score, 4)})
            results.append(passage)
        return results


@functools.lru_cache(maxsize=4)
def _load_index(path: Path) -> _Index:
    """Parse and index a corpus file. Cached -- indexing is O(corpus)."""
    passages: list[RetrievedPassage] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                passages.append(RetrievedPassage.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise BackendUnavailable(
                    "The offline textbook corpus is corrupt.",
                    remediation="Textbook retrieval is unavailable; use web search instead.",
                    detail=f"{path}:{number}: {exc}",
                ) from exc
    log.info("fixture corpus indexed", passages=len(passages), path=str(path))
    return _Index(passages)


class FixtureRetrievalBackend:
    """Retrieval over the committed corpus. No network, no model, no key."""

    name = "fixture"

    def __init__(self, corpus_path: Path | None = None) -> None:
        self.corpus_path = corpus_path or DEFAULT_CORPUS

    def health(self) -> HealthStatus:
        if not self.corpus_path.exists():
            return HealthStatus(
                healthy=False,
                backend=self.name,
                detail=f"corpus file missing: {self.corpus_path}",
            )
        try:
            index = _load_index(self.corpus_path)
        except BackendUnavailable as exc:
            return HealthStatus(healthy=False, backend=self.name, detail=exc.message)
        return HealthStatus(
            healthy=True,
            backend=self.name,
            degraded=True,
            detail=f"{len(index.passages)} offline passages (lexical ranking)",
        )

    def search(
        self, query: str, *, top_k: int, book_slug: str | None = None
    ) -> list[RetrievedPassage]:
        if not self.corpus_path.exists():
            raise BackendUnavailable(
                "The offline textbook corpus is not present.",
                remediation=(
                    "Textbook retrieval is unavailable. Answer from web search and say "
                    "the answer is not textbook-grounded."
                ),
                detail=f"missing: {self.corpus_path}",
            )
        return _load_index(self.corpus_path).search(query, top_k, book_slug)
