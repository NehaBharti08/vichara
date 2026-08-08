"""Textbook retrieval, both backends."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from vichara.settings import Settings
from vichara.tools.errors import ErrorCode
from vichara.tools.rag import (
    FixtureRetrievalBackend,
    HttpRetrievalBackend,
    TextbookSearchTool,
    build_textbook_tool,
)

CORPUS = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "rag_corpus.jsonl"

PASSAGE = {
    "chunk_id": "biology:0001:0",
    "book_slug": "biology",
    "book_title": "Biology",
    "chapter": "Chapter 4. Cell Structure",
    "section": "4.2. Prokaryotic Cells",
    "page_start": 200,
    "printed_page": "188",
    "citation": "Biology, 4.2. Prokaryotic Cells, p.188",
    "text": "Prokaryotic cells lack a membrane-bound nucleus.",
    "license": "CC BY 4.0",
    "source_url": "https://openstax.org/details/books/biology",
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    rows = [PASSAGE, {**PASSAGE, "chunk_id": "bio:2", "text": "Mitochondria produce ATP."}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


class TestFixtureBackend:
    def test_finds_relevant_passages(self, corpus: Path) -> None:
        results = FixtureRetrievalBackend(corpus).search("prokaryotic nucleus", top_k=5)

        assert results
        assert "Prokaryotic" in results[0].text

    def test_ranking_is_deterministic(self, corpus: Path) -> None:
        """A failing eval task must be reproducible.

        Dense retrieval is not deterministic across model versions; BM25 is.
        That is the whole reason this backend ranks lexically.
        """
        backend = FixtureRetrievalBackend(corpus)

        first = backend.search("mitochondria ATP", top_k=5)
        second = backend.search("mitochondria ATP", top_k=5)

        assert [p.chunk_id for p in first] == [p.chunk_id for p in second]
        assert [p.score for p in first] == [p.score for p in second]

    def test_no_match_returns_empty_not_an_error(self, corpus: Path) -> None:
        assert FixtureRetrievalBackend(corpus).search("quantum chromodynamics", top_k=5) == []

    def test_book_filter(self, corpus: Path) -> None:
        backend = FixtureRetrievalBackend(corpus)

        assert backend.search("prokaryotic", top_k=5, book_slug="biology")
        assert backend.search("prokaryotic", top_k=5, book_slug="anatomy-and-physiology") == []

    def test_top_k_is_respected(self, corpus: Path) -> None:
        assert len(FixtureRetrievalBackend(corpus).search("cells ATP", top_k=1)) == 1

    def test_health_reports_degraded(self, corpus: Path) -> None:
        """Healthy but not the real thing -- and the distinction is reported."""
        health = FixtureRetrievalBackend(corpus).health()

        assert health.healthy is True
        assert health.degraded is True

    def test_missing_corpus_is_unhealthy(self, tmp_path: Path) -> None:
        health = FixtureRetrievalBackend(tmp_path / "absent.jsonl").health()

        assert health.healthy is False

    def test_corrupt_corpus_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json", encoding="utf-8")

        assert FixtureRetrievalBackend(path).health().healthy is False


class TestShippedCorpus:
    """The committed corpus itself is an artifact worth testing."""

    def test_exists_and_is_indexable(self) -> None:
        health = FixtureRetrievalBackend().health()

        assert health.healthy is True
        assert "passages" in health.detail

    def test_every_passage_carries_a_citation(self) -> None:
        """A passage without a citable page is unusable for a grounded answer."""
        rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]

        assert rows
        assert all(r["citation"] for r in rows)
        assert all(r["printed_page"] for r in rows)
        assert all(r["license"] == "CC BY 4.0" for r in rows)

    @pytest.mark.parametrize(
        ("query", "expect"),
        [
            ("sodium potassium pump action potential", "Action Potential"),
            ("Mendel dihybrid cross", "12."),
        ],
    )
    def test_retrieval_is_topically_correct(self, query: str, expect: str) -> None:
        results = FixtureRetrievalBackend().search(query, top_k=3)

        assert results
        assert any(expect in (p.section or "") for p in results)


class TestTool:
    def test_returns_citations(self, corpus: Path) -> None:
        result = TextbookSearchTool(FixtureRetrievalBackend(corpus)).run(query="prokaryotic")

        assert result.ok is True
        assert result.citations
        assert result.citations[0].kind == "textbook"
        assert result.citations[0].source == PASSAGE["citation"]

    def test_output_is_untrusted(self, corpus: Path) -> None:
        result = TextbookSearchTool(FixtureRetrievalBackend(corpus)).run(query="prokaryotic")

        assert result.is_untrusted is True

    def test_empty_result_is_ok_not_a_failure(self, corpus: Path) -> None:
        """`ok=False` here would make the agent retry a query that cannot work.

        "The textbook does not cover this" is the finding an impossible-task
        probe depends on; dressing it up as an error hides it.
        """
        result = TextbookSearchTool(FixtureRetrievalBackend(corpus)).run(query="chromodynamics")

        assert result.ok is True
        assert result.citations == []
        assert "say so rather than searching again" in result.content


class TestHttpBackend:
    BASE = "https://vidyarag.test"

    @respx.mock
    def test_successful_search(self) -> None:
        respx.post(f"{self.BASE}/search").mock(
            return_value=httpx.Response(200, json={"results": [PASSAGE]})
        )

        results = HttpRetrievalBackend(self.BASE).search("prokaryotic", top_k=5)

        assert len(results) == 1
        assert results[0].citation == PASSAGE["citation"]

    @respx.mock
    def test_rate_limit_is_retryable_and_carries_a_deadline(self) -> None:
        respx.post(f"{self.BASE}/search").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "42"})
        )

        with pytest.raises(Exception) as caught:
            HttpRetrievalBackend(self.BASE).search("x", top_k=5)

        error = caught.value
        assert getattr(error, "code", None) is ErrorCode.RATE_LIMITED
        assert getattr(error, "retryable", None) is True
        assert getattr(error, "retry_after_s", None) == 42.0

    @respx.mock
    def test_server_error_is_retryable(self) -> None:
        respx.post(f"{self.BASE}/search").mock(return_value=httpx.Response(503))

        with pytest.raises(Exception) as caught:
            HttpRetrievalBackend(self.BASE).search("x", top_k=5)

        assert getattr(caught.value, "retryable", None) is True

    @respx.mock
    def test_client_error_is_not_retryable(self) -> None:
        """A 404 means the contract is wrong. Retrying burns the step budget."""
        respx.post(f"{self.BASE}/search").mock(return_value=httpx.Response(404))

        with pytest.raises(Exception) as caught:
            HttpRetrievalBackend(self.BASE).search("x", top_k=5)

        assert getattr(caught.value, "retryable", None) is False

    @respx.mock
    def test_malformed_body_is_not_retryable(self) -> None:
        respx.post(f"{self.BASE}/search").mock(
            return_value=httpx.Response(200, text="<html>not json</html>")
        )

        with pytest.raises(Exception) as caught:
            HttpRetrievalBackend(self.BASE).search("x", top_k=5)

        assert getattr(caught.value, "code", None) is ErrorCode.EXECUTION_FAILED

    @respx.mock
    def test_connection_failure_degrades_with_advice(self) -> None:
        respx.post(f"{self.BASE}/search").mock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(Exception) as caught:
            HttpRetrievalBackend(self.BASE).search("x", top_k=5)

        assert "web search" in getattr(caught.value, "remediation", "")

    @respx.mock
    def test_health_probe_never_raises(self) -> None:
        respx.get(f"{self.BASE}/health").mock(side_effect=httpx.ConnectError("refused"))

        health = HttpRetrievalBackend(self.BASE).health()

        assert health.healthy is False


class TestBackendSelection:
    def test_falls_back_to_fixture_without_a_url(self) -> None:
        tool = build_textbook_tool(Settings())

        assert tool.backend_name == "fixture"

    @respx.mock
    def test_uses_http_when_the_service_is_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIDYARAG_URL", "https://vidyarag.test")
        respx.get("https://vidyarag.test/health").mock(return_value=httpx.Response(200))

        assert build_textbook_tool(Settings()).backend_name == "http"

    @respx.mock
    def test_falls_back_when_the_service_is_configured_but_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Degrading on a failed probe, not on a missing URL, is what keeps a
        mid-session outage from becoming a broken run."""
        monkeypatch.setenv("VIDYARAG_URL", "https://vidyarag.test")
        respx.get("https://vidyarag.test/health").mock(return_value=httpx.Response(503))

        assert build_textbook_tool(Settings()).backend_name == "fixture"
