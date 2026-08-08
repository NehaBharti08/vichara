"""Structured logging.

Every log line an agent emits belongs to a trajectory. A flat message like
"calling tool" is close to useless when five tasks run concurrently under an
evaluation sweep; the same line carrying ``session_id``, ``step`` and
``tool`` can be grouped, counted, and joined against the trajectory record.

So the contract here is: bind context once at the start of a run, and every
subsequent line inherits it automatically via structlog's contextvars. Nodes
never thread a logger through their arguments.

JSON is the default format because the deployed Space ships logs to a
collector that parses them; ``console`` exists for a human at a terminal.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog

_configured = False


def configure_logging(
    level: str = "INFO",
    fmt: Literal["json", "console"] = "json",
    *,
    force: bool = False,
) -> None:
    """Install the structlog pipeline. Idempotent unless ``force`` is set.

    Idempotence matters because both the CLI and the FastAPI app configure
    logging on startup, and under ``uvicorn --reload`` that happens more than
    once per process.
    """
    global _configured
    if _configured and not force:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any
    if fmt == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    else:
        # format_exc_info is applied only in JSON mode: ConsoleRenderer draws
        # its own tracebacks and would render a pre-formatted string instead.
        shared.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger. Call :func:`configure_logging` first, or get defaults."""
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_run(session_id: str, **extra: Any) -> None:
    """Attach run-scoped context to every subsequent log line in this task.

    Uses contextvars, so it is safe under concurrency: an evaluation sweep
    running five tasks at once keeps their log lines attributable.
    """
    structlog.contextvars.bind_contextvars(session_id=session_id, **extra)


def bind_step(step: int, **extra: Any) -> None:
    """Update the current step number. Call once per graph node entry."""
    structlog.contextvars.bind_contextvars(step=step, **extra)


def clear_run() -> None:
    """Drop all bound context. Call when a run terminates."""
    structlog.contextvars.clear_contextvars()
