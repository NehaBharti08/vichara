"""Structured logging."""

from __future__ import annotations

import json

import pytest
import structlog

from vichara import logging as vlogging


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    """Logging config is process-global; keep tests from leaking into each other."""
    vlogging._configured = False
    structlog.contextvars.clear_contextvars()


class TestConfiguration:
    def test_is_idempotent(self) -> None:
        """The CLI and the API both configure on startup, and uvicorn --reload
        does it more than once per process."""
        vlogging.configure_logging("INFO", "json")
        first = structlog.get_config()["processors"]

        vlogging.configure_logging("DEBUG", "console")

        assert structlog.get_config()["processors"] is first

    def test_force_reconfigures(self) -> None:
        vlogging.configure_logging("INFO", "json")
        first = structlog.get_config()["processors"]

        vlogging.configure_logging("INFO", "console", force=True)

        assert structlog.get_config()["processors"] is not first

    def test_get_logger_configures_implicitly(self) -> None:
        logger = vlogging.get_logger("test")

        assert vlogging._configured is True
        assert logger is not None


class TestRunContext:
    def test_bound_context_appears_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Every line belongs to a trajectory.

        Under an eval sweep several tasks log concurrently; without the bound
        session id their lines cannot be separated, let alone joined against
        the trajectory records they describe.
        """
        vlogging.configure_logging("INFO", "json", force=True)
        vlogging.bind_run("sess-abc", task_id="t-01")
        vlogging.bind_step(3, node="act")

        vlogging.get_logger("test").info("calling tool", tool="web_search")

        record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert record["session_id"] == "sess-abc"
        assert record["task_id"] == "t-01"
        assert record["step"] == 3
        assert record["node"] == "act"
        assert record["tool"] == "web_search"
        assert record["event"] == "calling tool"

    def test_clear_run_drops_context(self, capsys: pytest.CaptureFixture[str]) -> None:
        vlogging.configure_logging("INFO", "json", force=True)
        vlogging.bind_run("sess-abc")
        vlogging.clear_run()

        vlogging.get_logger("test").info("after")

        record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert "session_id" not in record

    def test_level_filters(self, capsys: pytest.CaptureFixture[str]) -> None:
        vlogging.configure_logging("WARNING", "json", force=True)

        vlogging.get_logger("test").debug("noise")

        assert capsys.readouterr().out.strip() == ""
