"""Loading the annotated task set."""

from __future__ import annotations

import json
from pathlib import Path

from vichara.eval.tasks.schema import GoldTask, TaskSet
from vichara.settings import REPO_ROOT

DEFAULT_TASKS = REPO_ROOT / "data" / "eval" / "tasks.v1.jsonl"


def load_tasks(path: Path | None = None) -> TaskSet:
    """Read and validate the task set.

    A malformed record raises rather than being skipped. A silently dropped
    task changes the denominator of every metric, and nothing downstream would
    reveal that the number moved because the set shrank.
    """
    source = path or DEFAULT_TASKS
    if not source.exists():
        raise FileNotFoundError(f"task set not found: {source}")

    tasks: list[GoldTask] = []
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(GoldTask.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{source}:{number}: {exc}") from exc

    return TaskSet(version=source.stem.split(".")[-1], tasks=tasks)
