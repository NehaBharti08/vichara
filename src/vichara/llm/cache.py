"""Deterministic response cache.

The single biggest lever in this project's economics. A full evaluation sweep
is ~2,500 requests against a daily quota; without a cache, re-running it after
changing one prompt costs another 2,500. With one, it costs only the calls
that genuinely changed.

It is also what makes the development loop bearable: iterating on graph
plumbing replays instantly and spends nothing, which is strictly better than
the weak local model it replaces.

**The key must cover everything that can change an answer.** Model, messages,
bound tool schemas, temperature and seed all go in. A cache keyed on the
prompt alone would serve a flash-lite answer for a pro request and quietly
invalidate an ablation -- the exact failure that makes a benchmark worthless
while looking fine.

Only deterministic calls are cached. At ``temperature > 0`` the whole point is
that repeated calls differ, and caching would collapse the cross-seed variance
that Phase 4 reports as a result.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from vichara.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    role        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    day         TEXT PRIMARY KEY,
    requests    INTEGER NOT NULL DEFAULT 0
);
"""


def cache_key(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[str],
    temperature: float,
    seed: int | None,
    structured: str | None = None,
) -> str:
    """Hash everything that could change the response.

    ``tools`` is the sorted list of bound tool names: binding a different tool
    set changes what the model may answer with, so a cache that ignored it
    would serve an answer the current agent could not have produced.
    """
    material = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": sorted(tools),
            "temperature": temperature,
            "seed": seed,
            "structured": structured,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ResponseCache:
    """SQLite-backed, process-safe, and safe to delete at any time."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connect()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
            # WAL so a long eval sweep and an interactive session can share the
            # cache without the reader blocking the writer.
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        return self._connection

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            row = (
                self._connect()
                .execute("SELECT payload FROM responses WHERE key = ?", (key,))
                .fetchone()
            )
        if row is None:
            return None
        try:
            payload: dict[str, Any] = json.loads(row[0])
        except json.JSONDecodeError:
            # A corrupt row is a cache miss, never an error. The cache is an
            # optimisation and must never be able to fail a run.
            log.warning("corrupt cache row ignored", key=key[:12])
            return None
        return payload

    def put(self, key: str, *, model: str, role: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            connection = self._connect()
            connection.execute(
                "INSERT OR REPLACE INTO responses (key, model, role, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, model, role, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            connection.commit()

    # -- Daily request counter ---------------------------------------------

    def record_request(self, day: str) -> int:
        """Increment and return today's request count.

        Persisted rather than held in memory so an accidental loop in a
        notebook cannot burn the day's free-tier allowance without the number
        surviving the process that spent it.
        """
        if not self.enabled:
            return 0
        with self._lock:
            connection = self._connect()
            connection.execute(
                "INSERT INTO usage (day, requests) VALUES (?, 1) "
                "ON CONFLICT(day) DO UPDATE SET requests = requests + 1",
                (day,),
            )
            connection.commit()
            row = connection.execute("SELECT requests FROM usage WHERE day = ?", (day,)).fetchone()
        return int(row[0]) if row else 0

    def requests_today(self, day: str) -> int:
        if not self.enabled:
            return 0
        with self._lock:
            row = (
                self._connect()
                .execute("SELECT requests FROM usage WHERE day = ?", (day,))
                .fetchone()
            )
        return int(row[0]) if row else 0

    def stats(self) -> dict[str, int]:
        if not self.enabled:
            return {"entries": 0}
        with self._lock:
            row = self._connect().execute("SELECT COUNT(*) FROM responses").fetchone()
        return {"entries": int(row[0]) if row else 0}

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
