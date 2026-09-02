"""Turning scored results into a report.

Distributions, never single numbers. An agent that answers correctly four
times in five is a different system from one that answers correctly every
time, and a mean hides which you have -- so every metric is reported with its
spread, and the spread is treated as a finding rather than as noise.

Categories are never pooled. Refusal correctness and answer correctness have
different success criteria, and an average over them describes nothing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from vichara.eval.metrics import Distribution, TaskResult, rate
from vichara.eval.tasks.schema import Category


def for_agent(results: Sequence[TaskResult], version: str) -> list[TaskResult]:
    """Only the runs produced by the prompt set now on disk.

    A sweep reports over everything recorded for its profile rather than the
    fragment this invocation produced, so that resuming does not make the
    numbers look worse than the run was. That is right across a resume and
    wrong across a prompt edit: the first sweep after one printed
    "230 runs / 41 tasks", pooling 116 runs from the previous agent with 114
    from the current one into a single table -- the exact averaging that
    ``agent_version`` exists to prevent, in the one place a person actually
    reads the number.
    """
    return [r for r in results if r.agent_version == version]


def summarise(results: Sequence[TaskResult]) -> dict[str, object]:
    """Aggregate one sweep."""
    if not results:
        return {"n": 0}

    by_category: dict[str, list[TaskResult]] = defaultdict(list)
    for result in results:
        by_category[result.category.value].append(result)

    return {
        "n_runs": len(results),
        "n_tasks": len({r.task_id for r in results}),
        "seeds": sorted({r.seed for r in results if r.seed is not None}),
        "overall": _block(results),
        "by_category": {name: _block(rows) for name, rows in sorted(by_category.items())},
        "by_task": _per_task_variance(results),
    }


def _block(results: Sequence[TaskResult]) -> dict[str, object]:
    refusals = [r for r in results if r.refusal_correct is not None]
    return {
        "terminal_correct": rate([r.terminal_correct for r in results]),
        "answer_correct": rate([r.answer_correct for r in results]),
        "tool_precision": _dist([r.tool_precision for r in results]),
        "tool_recall": _dist([r.tool_recall for r in results]),
        "forbidden_tool_rate": rate([r.used_forbidden_tool for r in results]),
        "step_efficiency": _dist([r.step_efficiency for r in results]),
        "cited_rate": rate([r.cited for r in results]),
        "grounding_sources_present": rate([r.grounding_sources_present for r in results]),
        "refusal_correct": rate([r.refusal_correct for r in results]),
        "mean_steps_to_refusal": (
            round(sum(r.steps_to_refusal or 0 for r in refusals) / len(refusals), 2)
            if refusals
            else None
        ),
        "false_refusal_rate": rate([r.false_refusal for r in results]),
        "recovery_rate": rate([r.recovered for r in results]),
        "recovery_kinds": _counts([r.recovery_kind for r in results]),
        "llm_requests": _dist([float(r.llm_requests) for r in results]),
        "steps": _dist([float(r.actual_steps) for r in results]),
        "wall_clock_s": _dist([r.wall_clock_s for r in results]),
        "guardrails_fired": _counts([rule for r in results for rule in r.guardrails_fired]),
    }


def _per_task_variance(results: Sequence[TaskResult]) -> dict[str, object]:
    """Per-task agreement across seeds.

    The most useful column in the whole report: a task correct on 2 of 3 seeds
    is a different problem from one that fails consistently. The first is
    instability, the second is a capability gap, and they need different fixes.
    """
    grouped: dict[str, list[TaskResult]] = defaultdict(list)
    for result in results:
        grouped[result.task_id].append(result)

    out: dict[str, object] = {}
    for task_id, rows in sorted(grouped.items()):
        correct = [r.terminal_correct for r in rows]
        out[task_id] = {
            "runs": len(rows),
            "correct": sum(correct),
            "consistent": len(set(correct)) == 1,
            "requests_median": _median([float(r.llm_requests) for r in rows]),
        }
    return out


def _dist(values: Sequence[float | None]) -> dict[str, float] | None:
    distribution = Distribution.of([v for v in values if v is not None])
    if distribution is None:
        return None
    return {
        "median": distribution.median,
        "p25": distribution.p25,
        "p75": distribution.p75,
        "iqr": distribution.iqr,
        "min": distribution.minimum,
        "max": distribution.maximum,
        "n": float(distribution.n),
    }


def _median(values: Sequence[float]) -> float | None:
    distribution = Distribution.of(values)
    return distribution.median if distribution else None


def _counts(values: Sequence[str | None]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def to_markdown(summary: dict[str, object], *, title: str = "Evaluation") -> str:
    """Render for docs/EVALUATION.md."""
    if not summary.get("n_runs"):
        return f"# {title}\n\nNo results.\n"

    overall = summary["overall"]
    assert isinstance(overall, dict)
    lines = [
        f"# {title}",
        "",
        f"{summary['n_runs']} runs over {summary['n_tasks']} tasks, " f"seeds {summary['seeds']}.",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "terminal_correct",
        "answer_correct",
        "forbidden_tool_rate",
        "cited_rate",
        "grounding_sources_present",
        "refusal_correct",
        "mean_steps_to_refusal",
        "false_refusal_rate",
        "recovery_rate",
    ):
        value = overall.get(key)
        lines.append(f"| {key} | {'-' if value is None else value} |")

    lines += [
        "",
        "## Distributions (median, IQR)",
        "",
        "| metric | median | IQR | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in (
        "tool_precision",
        "tool_recall",
        "step_efficiency",
        "steps",
        "llm_requests",
        "wall_clock_s",
    ):
        dist = overall.get(key)
        if isinstance(dist, dict):
            lines.append(
                f"| {key} | {dist['median']} | {dist['iqr']} | {dist['min']} | {dist['max']} |"
            )

    lines += [
        "",
        "## By category",
        "",
        "| category | n | terminal correct | answer correct | step efficiency |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_category = summary["by_category"]
    assert isinstance(by_category, dict)
    for name in [c.value for c in Category]:
        block = by_category.get(name)
        if not isinstance(block, dict):
            continue
        steps = block.get("step_efficiency")
        efficiency = steps["median"] if isinstance(steps, dict) else "-"
        runs = block.get("llm_requests")
        n = int(runs["n"]) if isinstance(runs, dict) else 0
        lines.append(
            f"| {name} | {n} | {block.get('terminal_correct')} | "
            f"{block.get('answer_correct', '-')} | {efficiency} |"
        )

    lines += [
        "",
        "## Per-task consistency across seeds",
        "",
        "| task | correct / runs | consistent |",
        "| --- | --- | --- |",
    ]
    by_task = summary["by_task"]
    assert isinstance(by_task, dict)
    for task_id, row in by_task.items():
        assert isinstance(row, dict)
        mark = "yes" if row["consistent"] else "**no**"
        lines.append(f"| {task_id} | {row['correct']} / {row['runs']} | {mark} |")

    return "\n".join(lines) + "\n"
