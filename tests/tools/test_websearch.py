"""Web search, both backends."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from vichara.settings import Settings
from vichara.tools.errors import ErrorCode
from vichara.tools.websearch import (
    FixtureSearchBackend,
    TavilySearchBackend,
    WebSearchTool,
    build_web_search_tool,
)

RECORDINGS = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "search_responses.jsonl"


@pytest.fixture
def recordings(tmp_path: Path) -> Path:
    path = tmp_path / "search.jsonl"
    rows = [
        {
            "query": "CRISPR base editing clinical trial results",
            "results": [
                {
                    "title": "Base editing trial",
                    "url": "https://example.org/base-editing",
                    "content": "A trial of base editing reported results.",
                    "score": 0.9,
                }
            ],
        },
        {
            "query": "photosynthesis light reactions",
            "results": [
                {
                    "title": "Light reactions",
                    "url": "https://example.org/light",
                    "content": "Light reactions occur in the thylakoid membrane.",
                    "score": 0.8,
                }
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


class TestFixtureBackend:
    def test_replays_a_recorded_response(self, recordings: Path) -> None:
        results = FixtureSearchBackend(recordings).search(
            "CRISPR base editing clinical trial results", max_results=5
        )

        assert len(results) == 1
        assert results[0].url == "https://example.org/base-editing"

    def test_matches_a_reworded_query(self, recordings: Path) -> None:
        """An agent phrases the same intent differently each run.

        Exact-match replay would turn a working recording into a miss and make
        two runs of one task incomparable for a reason unrelated to the agent.
        """
        results = FixtureSearchBackend(recordings).search(
            "clinical trial results for CRISPR base editing", max_results=5
        )

        assert results

    def test_unrelated_query_returns_nothing(self, recordings: Path) -> None:
        """A confidently wrong recording silently changes what the task was."""
        assert (
            FixtureSearchBackend(recordings).search("medieval french poetry", max_results=5) == []
        )

    def test_respects_max_results(self, recordings: Path) -> None:
        results = FixtureSearchBackend(recordings).search(
            "photosynthesis light reactions", max_results=1
        )

        assert len(results) <= 1

    def test_missing_file_is_healthy_but_empty(self, tmp_path: Path) -> None:
        backend = FixtureSearchBackend(tmp_path / "absent.jsonl")

        assert backend.health().healthy is True
        assert backend.search("anything", max_results=5) == []

    def test_health_reports_degraded(self, recordings: Path) -> None:
        health = FixtureSearchBackend(recordings).health()

        assert health.healthy is True
        assert health.degraded is True


class TestShippedRecordings:
    def test_recordings_exist_and_load(self) -> None:
        assert RECORDINGS.exists()
        assert FixtureSearchBackend().health().healthy is True

    def test_every_result_has_a_url(self) -> None:
        """A search citation without a URL cannot be checked by anyone."""
        rows = [json.loads(line) for line in RECORDINGS.read_text(encoding="utf-8").splitlines()]

        assert rows
        for row in rows:
            for result in row["results"]:
                assert result["url"].startswith("http")


class TestTool:
    def test_returns_citations_with_urls(self, recordings: Path) -> None:
        result = WebSearchTool(FixtureSearchBackend(recordings)).run(
            query="photosynthesis light reactions"
        )

        assert result.ok is True
        assert result.citations[0].kind == "web"
        assert result.citations[0].locator == "https://example.org/light"

    def test_output_is_untrusted(self, recordings: Path) -> None:
        result = WebSearchTool(FixtureSearchBackend(recordings)).run(query="photosynthesis")

        assert result.is_untrusted is True

    def test_no_results_is_ok_with_advice_to_stop(self, recordings: Path) -> None:
        result = WebSearchTool(FixtureSearchBackend(recordings)).run(query="medieval poetry")

        assert result.ok is True
        assert "rather than continuing to search" in result.content


class TestTavilyBackend:
    @respx.mock
    def test_successful_search(self) -> None:
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "T", "url": "https://e.org", "content": "c", "score": 0.5}
                    ]
                },
            )
        )

        results = TavilySearchBackend("key").search("q", max_results=5)

        assert results[0].url == "https://e.org"

    @respx.mock
    def test_recency_switches_to_the_news_topic(self) -> None:
        """`days` is ignored unless topic=news.

        Sending one without the other returns undated general results, which
        would make a recency-sensitive eval task quietly meaningless.
        """
        route = respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        TavilySearchBackend("key").search("q", max_results=5, recency_days=7)

        body = json.loads(route.calls[0].request.content)
        assert body["topic"] == "news"
        assert body["days"] == 7

    @respx.mock
    def test_exhausted_plan_is_not_retryable(self) -> None:
        """432 means the monthly credits are gone.

        An agent that treats this as transient waits for a quota that resets
        next month, burning its entire remaining budget.
        """
        respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(432))

        with pytest.raises(Exception) as caught:
            TavilySearchBackend("key").search("q", max_results=5)

        assert getattr(caught.value, "code", None) is ErrorCode.POLICY_VIOLATION
        assert getattr(caught.value, "retryable", None) is False

    @respx.mock
    def test_rate_limit_is_retryable(self) -> None:
        respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(429))

        with pytest.raises(Exception) as caught:
            TavilySearchBackend("key").search("q", max_results=5)

        assert getattr(caught.value, "retryable", None) is True
        assert getattr(caught.value, "retry_after_s", None) == 30.0

    @respx.mock
    def test_bad_key_is_not_retryable(self) -> None:
        respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(401))

        with pytest.raises(Exception) as caught:
            TavilySearchBackend("key").search("q", max_results=5)

        assert getattr(caught.value, "retryable", None) is False

    def test_health_does_not_spend_a_credit(self) -> None:
        """A probe on every startup would consume a meaningful share of a
        1000-call monthly allowance across a development session."""
        with respx.mock:
            route = respx.post("https://api.tavily.com/search")

            assert TavilySearchBackend("key").health().healthy is True
            assert route.call_count == 0

    def test_missing_key_is_unhealthy(self) -> None:
        assert TavilySearchBackend("").health().healthy is False


class TestBackendSelection:
    def test_no_key_selects_recorded(self) -> None:
        assert build_web_search_tool(Settings()).backend_name == "fixture"

    def test_key_selects_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "test-key-not-real")

        assert build_web_search_tool(Settings()).backend_name == "tavily"

    def test_evaluation_can_force_recorded_despite_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A number measured against the live web cannot be re-derived later,
        so evaluation pins the search path even when live search is available."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key-not-real")

        tool = build_web_search_tool(Settings(), prefer_recorded=True)

        assert tool.backend_name == "fixture"
