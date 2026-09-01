"""Trajectory metrics.

Six computed mechanically from a trajectory and its gold annotation, one
judged. That ratio is the point of this project: an LLM scoring an LLM is a
measurement whose error you cannot characterise, so it is confined to the one
question -- does this citation actually support this claim -- that nothing
mechanical can answer.

Every metric here is a function of ``(TrajectoryRecord, GoldTask)``. None of
them call a model, none touch the network, and all of them are unit-tested
against synthetic trajectories. They can be re-run over stored records months
later without re-running the agent, which is what makes a result re-derivable
rather than merely reported.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from vichara.eval.tasks.schema import Category, GoldTask, Terminal
from vichara.trajectory.schema import TerminalReason, TrajectoryRecord

# How quickly a correct refusal must happen. An agent that eventually says "I
# don't know" after fifteen steps is broken even though the words are right,
# and an accuracy-only metric would score it identically to an instant refusal.
MAX_REFUSAL_STEPS = 3

_TERMINAL_MAP = {
    Terminal.ANSWERED: {TerminalReason.ANSWERED},
    Terminal.REFUSED: {TerminalReason.REFUSED},
    Terminal.CLARIFY: {TerminalReason.CLARIFY},
}


class TaskResult(BaseModel):
    """Every metric for one run of one task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    category: Category
    split: str
    seed: int | None = None
    capability_profile: list[str] = []

    agent_version: str = ""
    """Digest of the prompt set that produced this run.

    ``prompt_hashes`` was recorded on every trajectory from Phase 3 so that a
    run made before a prompt edit could be told apart from one made after --
    and then nothing downstream ever read it, so the scorer would happily pool
    two prompt versions into one mean and report it as a single agent. The
    mechanism existed; it was simply not connected to the thing it was built to
    protect. Carrying it onto the result row is what makes the split visible in
    ``eval_results/`` rather than only in the trajectory log."""

    terminal_reason: str | None = None
    terminal_correct: bool = False

    tool_precision: float | None = None
    tool_recall: float | None = None
    used_forbidden_tool: bool = False

    step_efficiency: float | None = None
    actual_steps: int = 0
    optimal_steps: int = 0

    answer_correct: bool | None = None
    cited: bool = False
    grounding_sources_present: bool | None = None

    refusal_correct: bool | None = None
    steps_to_refusal: int | None = None
    false_refusal: bool = False

    recovered: bool | None = None
    recovery_kind: str | None = None

    llm_requests: int = 0
    total_tokens: int = 0
    est_usd: float = 0.0
    wall_clock_s: float = 0.0
    guardrails_fired: list[str] = []


def agent_version(record: TrajectoryRecord) -> str:
    """A single short digest standing for the whole prompt set.

    Every prompt matters, not just the one that was edited, so this folds all
    of them rather than singling out ``act``. Runs whose digests differ were
    produced by different agents and must not be averaged together.

    **It covers prompts, not code.** Making loop detection soft changed the
    agent's behaviour materially and moved this digest not at all, because the
    prompt files were untouched. So this catches the change it was built for --
    a prompt edited underneath a running sweep -- and silently misses a
    behavioural change in a node. Hashing the source would be the obvious
    extension and a bad one: every refactor and comment would invalidate
    results that are still perfectly comparable, and the discipline would be
    abandoned within a week. Until something better exists, discarding prior
    results across a behavioural code change is a decision the operator has to
    make, and this docstring is where that obligation is written down.
    """
    return agent_version_of(record.prompt_hashes)


def agent_version_of(prompt_hashes: dict[str, str]) -> str:
    """The same digest, from the prompt files rather than from a finished run.

    The sweep runner needs this *before* it has a trajectory, to decide whether
    a pair recorded earlier describes the agent it is about to run.
    """
    joined = "|".join(f"{name}={digest}" for name, digest in sorted(prompt_hashes.items()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12] if joined else ""


def score(record: TrajectoryRecord, task: GoldTask) -> TaskResult:
    """Compute every mechanical metric for one trajectory."""
    called = record.tool_calls_made
    distinct = set(called)
    expected = set(task.expected_tools)

    result = TaskResult(
        task_id=task.id,
        category=task.category,
        split=task.split.value,
        seed=record.seed,
        capability_profile=record.capability_profile,
        agent_version=agent_version(record),
        terminal_reason=record.terminal_reason.value if record.terminal_reason else None,
        terminal_correct=_terminal_correct(record, task),
        used_forbidden_tool=bool(distinct & set(task.forbidden_tools)),
        actual_steps=record.agent_steps,
        optimal_steps=task.optimal_steps,
        llm_requests=record.llm_requests,
        total_tokens=record.total_tokens,
        est_usd=record.est_usd,
        wall_clock_s=record.wall_clock_s,
        guardrails_fired=sorted({e.rule for e in record.guardrail_events if e.action == "block"}),
    )

    # Tool selection. Undefined rather than zero when the task expects no
    # tools -- a refusal task has no correct tool set, and scoring it 0.0
    # would drag the mean down with a number that means nothing.
    if expected:
        hits = len(distinct & expected)
        result.tool_precision = hits / len(distinct) if distinct else 0.0
        result.tool_recall = hits / len(expected)

    if record.agent_steps > 0:
        result.step_efficiency = min(task.optimal_steps / record.agent_steps, 1.0)

    answer = (record.final_answer or "").lower()
    if task.answer_contains or task.answer_excludes:
        result.answer_correct = all(
            fragment.lower() in answer for fragment in task.answer_contains
        ) and not any(fragment.lower() in answer for fragment in task.answer_excludes)

    result.cited = bool(record.citations)
    if task.grounding_sources:
        # Checked against the citations the *tools* returned, not against the
        # answer text, so a fabricated citation cannot satisfy it.
        sources = " ".join(str(c.get("source", "")) for c in record.citations)
        result.grounding_sources_present = all(s in sources for s in task.grounding_sources)

    if task.expected_terminal is Terminal.REFUSED:
        refused = record.terminal_reason is TerminalReason.REFUSED
        result.steps_to_refusal = record.agent_steps
        result.refusal_correct = refused and record.agent_steps <= MAX_REFUSAL_STEPS
    elif record.terminal_reason is TerminalReason.REFUSED:
        # Refused something it should have answered. Counted separately: an
        # agent that refuses everything would otherwise score perfectly on
        # refusal correctness.
        result.false_refusal = True

    return result


def _terminal_correct(record: TrajectoryRecord, task: GoldTask) -> bool:
    if record.terminal_reason is None:
        return False
    return record.terminal_reason in _TERMINAL_MAP[task.expected_terminal]


def score_recovery(result: TaskResult, record: TrajectoryRecord) -> TaskResult:
    """Classify how a run handled an injected fault.

    ``recovered`` and ``lucky`` are kept apart deliberately. An agent that
    retried the identical call and happened to succeed learned nothing; one
    that switched tool or reformulated adapted. Collapsing them would let a
    blind-retry agent score as resilient.
    """
    failures = [
        (step_index, obs)
        for step_index, step in enumerate(record.steps)
        for obs in step.observations
        if not obs.ok
    ]
    if not failures:
        return result

    first_index, first = failures[0]
    later = record.tool_calls_made[first_index + 1 :] if record.tool_calls_made else []

    result.recovered = result.terminal_correct
    if not later:
        result.recovery_kind = "no_further_action"
    elif any(tool != first.tool for tool in later):
        result.recovery_kind = "switched_tool"
    else:
        result.recovery_kind = "retried_same_tool"
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class Distribution(BaseModel):
    """A metric across repeated runs.

    Median and IQR rather than a mean, because agent runs are not normally
    distributed -- one run that hit the step ceiling drags a mean somewhere no
    individual run went. The spread is reported as a result in its own right:
    an agent that is right four times in five is a different system from one
    that is right every time, and a single number hides which you have.
    """

    model_config = ConfigDict(extra="forbid")

    n: int
    median: float
    p25: float
    p75: float
    minimum: float
    maximum: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Distribution | None:
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        ordered = sorted(clean)
        return cls(
            n=len(ordered),
            median=round(statistics.median(ordered), 4),
            p25=round(_quantile(ordered, 0.25), 4),
            p75=round(_quantile(ordered, 0.75), 4),
            minimum=round(ordered[0], 4),
            maximum=round(ordered[-1], 4),
        )

    @property
    def iqr(self) -> float:
        return round(self.p75 - self.p25, 4)


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank quantile. Exact on small samples, where interpolation
    would invent values between the five runs actually observed."""
    if len(ordered) == 1:
        return float(ordered[0])
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return float(ordered[index])


def rate(values: Sequence[bool | None]) -> float | None:
    """Fraction true, ignoring runs where the metric does not apply."""
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return round(sum(defined) / len(defined), 4)
