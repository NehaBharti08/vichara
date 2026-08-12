"""Replay of recorded web search responses.

Recordings are real Tavily responses captured by ``scripts/record_search.py``,
not invented results. A fabricated search result would be a citation pointing
at a URL that never said what it is quoted as saying, which is precisely the
failure this project exists to detect in other systems.

Matching is lexical and fuzzy on purpose. An agent asked the same question
twice will phrase its query differently each time, so exact-match replay would
turn a working recording into a miss and make trajectories non-comparable for
a reason that has nothing to do with the agent.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

from vichara.logging import get_logger
from vichara.settings import REPO_ROOT
from vichara.tools.base import HealthStatus
from vichara.tools.errors import BackendUnavailable
from vichara.tools.websearch.backend import WebResult

log = get_logger(__name__)

DEFAULT_RECORDINGS = REPO_ROOT / "data" / "fixtures" / "search_responses.jsonl"

_TOKEN = re.compile(r"[a-z0-9]+")

# Below this overlap with a recorded query, replay reports no results rather
# than returning something loosely related. A confidently wrong recording is
# worse than an honest miss: it silently changes what the task was.
_MATCH_THRESHOLD = 0.34


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class _Recording:
    __slots__ = ("query", "results", "tokens")

    def __init__(self, query: str, results: list[WebResult]) -> None:
        self.query = query
        self.results = results
        self.tokens = _tokens(query)


@functools.lru_cache(maxsize=4)
def _load(path: Path) -> list[_Recording]:
    recordings: list[_Recording] = []
    if not path.exists():
        return recordings
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                recordings.append(
                    _Recording(
                        row["query"],
                        [WebResult.model_validate(r) for r in row.get("results", [])],
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise BackendUnavailable(
                    "The recorded web search corpus is corrupt.",
                    remediation="Web search is unavailable; use the textbook instead.",
                    detail=f"{path}:{number}: {exc}",
                ) from exc
    log.info("search recordings loaded", count=len(recordings), path=str(path))
    return recordings


class FixtureSearchBackend:
    """Replays recorded responses. No network, no key, no quota."""

    name = "fixture"

    def __init__(self, recordings_path: Path | None = None) -> None:
        self.recordings_path = recordings_path or DEFAULT_RECORDINGS

    def health(self) -> HealthStatus:
        try:
            recordings = _load(self.recordings_path)
        except BackendUnavailable as exc:
            return HealthStatus(healthy=False, backend=self.name, detail=exc.message)
        return HealthStatus(
            healthy=True,
            backend=self.name,
            degraded=True,
            detail=f"{len(recordings)} recorded responses (replay only)",
        )

    def search(
        self, query: str, *, max_results: int, recency_days: int | None = None
    ) -> list[WebResult]:
        del recency_days  # recordings are fixed in time; filtering them would lie
        recordings = _load(self.recordings_path)
        wanted = _tokens(query)
        if not wanted or not recordings:
            return []

        best: _Recording | None = None
        best_score = 0.0
        for recording in recordings:
            overlap = len(wanted & recording.tokens)
            if not overlap:
                continue
            score = overlap / len(wanted | recording.tokens)
            if score > best_score:
                best, best_score = recording, score

        if best is None or best_score < _MATCH_THRESHOLD:
            return []
        log.info("replayed search", query=query, matched=best.query, score=round(best_score, 3))
        return best.results[:max_results]
