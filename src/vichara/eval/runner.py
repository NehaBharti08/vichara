"""The evaluation runner.

Resumable, seeded, and quota-aware, because on a free tier a full sweep is
~2,500 requests and takes a day or two of wall clock. A runner that loses its
work when the laptop sleeps is not usable at that scale, so completed
``(task, seed)`` pairs are skipped on restart and every result is flushed as
it is produced rather than at the end.

Every task runs multiple times with different seeds. A single trajectory is an
anecdote: agents are stochastic, and the spread across seeds is a result in
its own right rather than noise to average away.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vichara.agent.nodes.context import PROMPT_DIR
from vichara.agent.runner import AgentSession
from vichara.eval.faults import FaultSpec, wrap_with_fault
from vichara.eval.metrics import TaskResult, agent_version_of, score, score_recovery
from vichara.eval.tasks.schema import GoldTask, TaskSet
from vichara.logging import get_logger
from vichara.settings import PipelineConfig, Settings
from vichara.trajectory.recorder import hash_prompts

log = get_logger(__name__)

DEFAULT_RESULTS = Path("eval_results")

MAX_CONSECUTIVE_FAILURES = 5
"""Stop the sweep after this many failures in a row.

One failure is a blip. Five in a row is the provider being down or the day's
quota being gone, and churning through the remaining tasks to confirm it wastes
an hour and produces nothing. Failed pairs are never recorded, so resuming
later picks them up unchanged."""


@dataclass
class SweepConfig:
    """One evaluation sweep."""

    profile: str = "baseline"
    repeats: int = 3
    """Three for routine sweeps, five for the headline table. Each repeat is a
    full trajectory, so this multiplies the quota cost directly."""

    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    only: list[str] = field(default_factory=list)
    disable_tools: list[str] = field(default_factory=list)
    """Builds a degraded capability profile. Reported as its own column rather
    than mixed in, so "accuracy without retrieval" is a finding instead of an
    outage."""

    fault: FaultSpec | None = None
    max_requests: int | None = None
    """Halts the sweep before it exhausts the day's allowance. Without it an
    overnight run can consume the quota and leave nothing for the morning."""


def _sweep_order(tasks: Sequence[GoldTask], seeds: Sequence[int]) -> list[tuple[int, GoldTask]]:
    """Seed-major: every task once at seed 0, then everything again at seed 1.

    Task-major ordering ran all five seeds of one task before starting the
    next, which is fine for a sweep that finishes and quietly corrupting for
    one that does not. On a free tier interruption is the normal case, and the
    quota died at the same place every time -- so the partial sweep this
    project reported for weeks covered 97% of single-tool tasks and about 20%
    of everything else:

        single_tool  92/95   ambiguous    5/25
        impossible    8/30   multi_tool  11/55

    Single-tool is the easiest category (0.989 complete) and ambiguous the
    hardest (0.840), so the interruption did not merely shrink the sample, it
    selected the easy end of it. The reported 0.966 was never a whole-set
    number.

    Seed-major makes an interrupted sweep an honest one. Stop it anywhere and
    every task has been attempted a near-equal number of times: breadth first,
    depth second. The n is smaller and the sample is not skewed, which is the
    trade worth making when the run may be cut off at any point.

    Order within a seed is left alone. It is the gold-set order, it is stable,
    and shuffling it would buy nothing once the category skew is gone.
    """
    return [(seed, task) for seed in seeds for task in tasks]


class ResultStore:
    """Append-only JSONL of scored results, with resume support."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def completed(self, agent_version: str) -> set[tuple[str, int | None]]:
        """``(task_id, seed)`` pairs already scored **for this agent**.

        Scoping this to the prompt set is the difference between resuming a
        sweep and silently finishing someone else's. Editing a prompt makes
        every pair recorded before the edit describe a different agent, and a
        resume keyed on ``(task, seed)`` alone would skip all of them and write
        the remainder into the same file -- producing one results table
        spanning two agents, with a row count that looks entirely correct.

        Rows carrying no version are treated as not done. They were produced by
        an unknown prompt set, and re-running a pair is cheap next to reporting
        a number nobody can attribute.
        """
        done: set[tuple[str, int | None]] = set()
        if not self.path.exists():
            return done
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("agent_version", "") != agent_version:
                        continue
                    done.add((row["task_id"], row.get("seed")))
                except (json.JSONDecodeError, KeyError):
                    # A truncated final line from an interrupted sweep. Skipping
                    # it means that pair simply runs again.
                    continue
        return done

    def append(self, result: TaskResult) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(result.model_dump_json() + "\n")

    def read(self) -> list[TaskResult]:
        if not self.path.exists():
            return []
        results = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        results.append(TaskResult.model_validate(json.loads(line)))
                    except (json.JSONDecodeError, ValueError):
                        continue
        return results


def run_sweep(
    settings: Settings,
    config: PipelineConfig,
    tasks: TaskSet,
    sweep: SweepConfig,
    *,
    results_dir: Path | None = None,
    resume: bool = True,
) -> list[TaskResult]:
    """Run every (task, seed) pair not already recorded."""
    store = ResultStore((results_dir or DEFAULT_RESULTS) / f"{sweep.profile}.jsonl")
    version = agent_version_of(hash_prompts(PROMPT_DIR))
    done = store.completed(version) if resume else set()
    if done:
        log.info("resuming sweep", already_done=len(done), agent_version=version)
    else:
        log.info("no prior results for this agent; running the full sweep", agent_version=version)

    selected = [t for t in tasks.tasks if not sweep.only or t.id in sweep.only]
    seeds = sweep.seeds[: sweep.repeats]
    requests_used = 0
    consecutive_failures = 0
    produced: list[TaskResult] = []

    for seed, task in _sweep_order(selected, seeds):
        if (task.id, seed) in done:
            continue
        if sweep.max_requests is not None and requests_used >= sweep.max_requests:
            log.warning(
                "sweep halted at its request ceiling",
                used=requests_used,
                ceiling=sweep.max_requests,
            )
            return produced

        result = _run_one(settings, config, task, seed, sweep)
        requests_used += result.llm_requests

        # A fatal error is almost always the provider being unreachable --
        # on a free tier, the day's request quota running out mid-sweep.
        # Persisting it would be the worst possible outcome: resume skips
        # completed (task, seed) pairs, so a quota outage would be baked
        # into the results permanently and reported as the agent failing.
        # It is not recorded, so the pair simply runs again later.
        if result.terminal_reason == "fatal_error":
            consecutive_failures += 1
            log.warning(
                "run failed, not recorded",
                task=task.id,
                seed=seed,
                consecutive=consecutive_failures,
            )
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                # One failure is a blip; several in a row is the provider
                # being down or the quota being gone. Churning through the
                # remaining tasks would spend an hour proving it.
                log.warning(
                    "halting: the provider looks unavailable",
                    consecutive=consecutive_failures,
                    hint="resume later; nothing was recorded for the failed pairs",
                )
                return produced
            continue

        consecutive_failures = 0
        store.append(result)
        produced.append(result)
        log.info(
            "task scored",
            task=task.id,
            seed=seed,
            terminal=result.terminal_reason,
            correct=result.terminal_correct,
            requests=result.llm_requests,
        )

    return produced


def _run_one(
    settings: Settings,
    config: PipelineConfig,
    task: GoldTask,
    seed: int,
    sweep: SweepConfig,
) -> TaskResult:
    """One trajectory, scored. Never raises.

    A crash on task 31 of 50 must not lose the previous 30, so an exception
    becomes a recorded failure rather than an abort.
    """
    started = time.perf_counter()
    session = AgentSession(
        settings,
        config,
        # Evaluation always auto-approves and always replays recorded search.
        # The first keeps the interrupt path exercised without a human; the
        # second keeps the run reproducible, since a number measured against
        # the live web cannot be re-derived later.
        auto_approve=True,
        prefer_recorded_search=True,
        seed=seed,
        task_id=task.id,
    )
    try:
        for name in sweep.disable_tools:
            _disable(session, name)
        if sweep.fault is not None:
            _inject(session, sweep.fault)

        outcome = session.run(task.task)
        result = score(outcome.record, task)
        if sweep.fault is not None:
            result = score_recovery(result, outcome.record)
        return result
    except Exception as exc:  # noqa: BLE001 - one bad task must not end the sweep
        log.warning("task raised", task=task.id, seed=seed, error=f"{type(exc).__name__}: {exc}")
        return TaskResult(
            task_id=task.id,
            category=task.category,
            split=task.split.value,
            seed=seed,
            terminal_reason="fatal_error",
            terminal_correct=False,
            wall_clock_s=round(time.perf_counter() - started, 2),
        )
    finally:
        session.close()


def _disable(session: AgentSession, tool_name: str) -> None:
    """Remove a tool from the capability set for this run."""
    session.registry.statuses = [
        (
            s
            if s.spec.name != tool_name
            else type(s)(spec=s.spec, tool=None, health=s.health, reason="disabled for sweep")
        )
        for s in session.registry.statuses
    ]


def _inject(session: AgentSession, fault: FaultSpec) -> None:
    """Wrap the targeted tool so it misbehaves in a defined way."""
    for status in session.registry.statuses:
        if status.tool is not None and status.spec.name == fault.tool:
            wrap_with_fault(status.tool, fault)


def iter_results(path: Path) -> Iterator[TaskResult]:
    yield from ResultStore(path).read()
