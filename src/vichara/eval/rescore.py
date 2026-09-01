"""Recompute metrics from stored trajectories.

The point of making every metric a pure function of ``(TrajectoryRecord,
GoldTask)`` was that a change to a metric should not require re-running the
agent. This is that promise being collected: when the step-efficiency unit
mismatch was found, correcting every published number cost no model requests
and no waiting -- the trajectories were already on disk.

It also means a sweep interrupted by a metric change is not wasted. Results
written before the change and after it would otherwise be silently
incomparable, which is the sort of thing that quietly invalidates a table.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from vichara.eval.metrics import TaskResult, score
from vichara.eval.tasks.loader import load_tasks
from vichara.logging import get_logger
from vichara.trajectory.recorder import read_trajectories
from vichara.trajectory.schema import TerminalReason, TrajectoryRecord

log = get_logger(__name__)


def rescore(
    trajectories: Path,
    results_dir: Path,
    *,
    profile: str | None = None,
) -> dict[str, int]:
    """Rebuild ``eval_results/<profile>.jsonl`` from recorded trajectories.

    Only trajectories carrying a ``task_id`` from the gold set are used, which
    excludes ad-hoc CLI runs and the injection suite -- those have no annotated
    optimal path, so scoring them would be meaningless rather than merely
    wrong.

    The most recent trajectory wins for a given (task, seed, profile). A sweep
    that was resumed can contain an earlier failed attempt at the same pair,
    and the retry is the one that describes the agent.
    """
    tasks = {t.id: t for t in load_tasks().tasks}
    latest: dict[tuple[str, str, int | None], TrajectoryRecord] = {}
    skipped = 0
    fatal = 0

    for record in read_trajectories(trajectories):
        if record.task_id not in tasks:
            skipped += 1
            continue
        if profile and record.profile != profile:
            continue
        if record.terminal_reason is TerminalReason.FATAL_ERROR:
            # A provider outage is not agent behaviour. On a free tier the
            # quota runs out mid-sweep, every remaining task returns
            # fatal_error, and scoring those as failures would report the
            # agent losing capability it never lost.
            #
            # Dropping the pair entirely -- rather than scoring it -- is what
            # makes the gap visible: the run count falls short of tasks x
            # seeds, which is a question a reader asks, where a quietly
            # depressed accuracy number is one nobody thinks to.
            fatal += 1
            continue
        # Later lines overwrite earlier ones: the file is append-only and in
        # chronological order.
        latest[(record.task_id, record.profile, record.seed)] = record

    by_profile: dict[str, list[TaskResult]] = defaultdict(list)
    for (task_id, prof, _seed), record in latest.items():
        by_profile[prof].append(score(record, tasks[task_id]))

    # prompt_hashes was recorded on every trajectory from Phase 3 precisely so
    # a run made before a prompt edit could be told apart from one made after,
    # and then nothing read it. Pooling two prompt versions into one mean and
    # calling it "the agent" is the exact failure that field exists to prevent,
    # and it would happen silently: the row count still looks right.
    #
    # This warns rather than refuses. A mixed set is a real state to be in
    # halfway through a re-run, and the caller needs the numbers to decide
    # which pairs still need redoing -- but it must not be able to miss it.
    for prof, rows in by_profile.items():
        versions = Counter(row.agent_version for row in rows)
        if len(versions) > 1:
            log.warning(
                "results pool more than one prompt version; these are not one agent",
                profile=prof,
                versions=dict(versions),
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for prof, rows in by_profile.items():
        rows.sort(key=lambda r: (r.task_id, r.seed if r.seed is not None else -1))
        target = results_dir / f"{prof}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(row.model_dump_json() + "\n")
        counts[prof] = len(rows)
        log.info("rescored", profile=prof, runs=len(rows), path=str(target))

    if skipped:
        log.info("ignored trajectories with no gold task", count=skipped)
    if fatal:
        log.warning(
            "dropped runs the provider failed; re-run those pairs to close the gap",
            count=fatal,
        )
    return counts


def load_results(path: Path) -> list[TaskResult]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(TaskResult.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
    return out
