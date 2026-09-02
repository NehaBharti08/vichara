"""Build the static demo payload.

Hugging Face removed free Docker Spaces, so the hosted demo is a static page
rather than a live agent. That is a smaller demo and a more reliable one: it
loads instantly, never sleeps, and cannot show a cold-start error or an
exhausted quota, which are the three ways a hosted agent demo usually
embarrasses its author.

What it shows is unchanged, because the viewer was always about *displaying* a
trajectory rather than producing one. The curation below is the whole design
decision: a reviewer with thirty seconds should land on runs that demonstrate
the interesting behaviour, not on whichever trajectory happened to be last.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = ROOT / "trajectories" / "runs.jsonl"
OUT = ROOT / "site" / "data.json"

MAX_OBS_CHARS = 900
"""Observation bodies are truncated. A single retrieval result runs to
kilobytes and the page has to stay small enough to load instantly on a phone."""


def load() -> list[dict[str, Any]]:
    rows = []
    for line in TRAJECTORIES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def tools_used(record: dict[str, Any]) -> set[str]:
    return {c["tool"] for s in record.get("steps", []) for c in s.get("tool_calls", [])}


def flagged(record: dict[str, Any]) -> bool:
    return any(
        o.get("injection_flagged")
        for s in record.get("steps", [])
        for o in s.get("observations", [])
    )


def blocked(record: dict[str, Any]) -> bool:
    return any(e.get("action") == "block" for e in record.get("guardrail_events", []))


def curate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick one clear example of each behaviour worth demonstrating.

    Ordered so the page tells a story: it works, it works on harder things, it
    declines when it should, it asks when the question is ambiguous, it stops
    itself, and it notices when a document tries to give it orders.
    """
    wanted: list[tuple[str, str, Any]] = [
        (
            "Grounded answer",
            "A textbook question answered with citations carrying real printed page numbers.",
            lambda r: r.get("terminal_reason") == "answered"
            and tools_used(r) == {"textbook_search"}
            and len(r.get("citations", [])) >= 2
            and not flagged(r),
        ),
        (
            "Multi-tool orchestration",
            "Retrieves a value from the textbook, then computes with it in the sandbox.",
            lambda r: r.get("terminal_reason") == "answered"
            and {"textbook_search", "run_python"} <= tools_used(r),
        ),
        (
            "Correct refusal",
            "An impossible question declined in the planner, before any tool is called. "
            "Refusing at step fifteen would be a failure even with the same words.",
            lambda r: r.get("terminal_reason") == "refused" and not tools_used(r),
        ),
        (
            "Asks instead of guessing",
            "An ambiguous question. Silently picking one reading and answering "
            "confidently is the failure an accuracy-only metric scores as success.",
            lambda r: r.get("terminal_reason") == "clarify",
        ),
        (
            "Guardrail stops a runaway",
            "A ceiling fires. The run answers from the evidence it already has "
            "rather than discarding it.",
            lambda r: blocked(r) and r.get("citations"),
        ),
        (
            "Prompt injection, detected",
            "A retrieved document contains instructions addressed to the agent. "
            "The scanner flags it and the agent reports it instead of obeying.",
            lambda r: flagged(r) and r.get("terminal_reason") == "answered",
        ),
        (
            "Fabricated citation removed",
            "The document told the agent to cite a source no tool returned. "
            "Citation verification stripped it before the answer was shown.",
            lambda r: bool(r.get("citations_fabricated")),
        ),
    ]

    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for title, why, predicate in wanted:
        for record in rows:
            if record["session_id"] in used:
                continue
            try:
                if predicate(record):
                    chosen.append({"title": title, "why": why, "record": trim(record)})
                    used.add(record["session_id"])
                    break
            except (KeyError, TypeError):
                continue
    return chosen


def trim(record: dict[str, Any]) -> dict[str, Any]:
    """Shrink a record to what the page renders."""
    out = dict(record)
    steps = []
    for step in record.get("steps", []):
        copy = dict(step)
        copy["observations"] = [
            {
                **obs,
                "content": (obs.get("content") or "")[:MAX_OBS_CHARS],
                "truncated_for_display": len(obs.get("content") or "") > MAX_OBS_CHARS,
            }
            for obs in step.get("observations", [])
        ]
        steps.append(copy)
    out["steps"] = steps
    return out


def metrics() -> dict[str, float]:
    """Compute what the page shows, rather than restating it.

    These were hardcoded, and the page drifted exactly as far as you would
    expect: it went on serving ``step_efficiency: 0.333`` for weeks after the
    README had retracted that number as a unit-mismatch bug in the metric, and
    ``terminal_correct: 0.976`` from a sweep two agent versions old. A literal
    in a build script has no way to go stale loudly.

    Scoped to the agent whose prompts are on disk, for the same reason the
    sweep report is: averaging two agent versions into one figure describes
    neither. A missing or empty results file raises rather than falling back to
    a default -- publishing a plausible wrong number is worse than not
    publishing.
    """
    from vichara.agent.nodes.context import PROMPT_DIR
    from vichara.eval.metrics import agent_version_of
    from vichara.eval.report import for_agent, summarise
    from vichara.eval.runner import ResultStore
    from vichara.trajectory.recorder import hash_prompts

    version = agent_version_of(hash_prompts(PROMPT_DIR))
    rows = for_agent(ResultStore(ROOT / "eval_results" / "baseline.jsonl").read(), version)
    if not rows:
        raise SystemExit(f"no baseline results for agent {version}; run the sweep first")

    overall = summarise(rows)["overall"]
    assert isinstance(overall, dict)
    out = {
        "tasks": float(len({r.task_id for r in rows})),
        "runs": float(len(rows)),
        "terminal_correct": overall["terminal_correct"],
        "tool_precision": overall["tool_precision"]["median"],
        "forbidden_tool_rate": overall["forbidden_tool_rate"],
        "refusal_correct": overall["refusal_correct"],
        "step_efficiency": overall["step_efficiency"]["median"],
    }
    for profile in ("baseline", "hardened"):
        out[f"asr_{profile}"] = _asr(profile, version)
    return {k: round(float(v), 4) for k, v in out.items()}


def _asr(profile: str, version: str) -> float:
    """Attack success rate, over attacks this agent actually faced.

    Scoped for the same reason the sweep metrics are, and it matters more here:
    the published claim is a *comparison*. A baseline rate measured on one agent
    set against a hardened rate measured on another does not describe a defence,
    and the page would present it as though it did.
    """
    path = ROOT / "eval_results" / f"injection-{profile}.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    matching = [r for r in rows if r.get("agent_version", "") == version]
    if not matching:
        raise SystemExit(
            f"no injection results for {profile} at agent {version} "
            f"({len(rows)} row(s) from other versions); re-run with: "
            f"uv run vichara attack --profile {profile}"
        )
    return sum(1 for r in matching if r["succeeded"]) / len(matching)


def main() -> int:
    rows = load()
    examples = curate(rows)

    payload = {
        "generated_from": len(rows),
        "examples": examples,
        "metrics": metrics(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"{len(examples)} examples from {len(rows)} trajectories -> {OUT}")
    for example in examples:
        print(f"  - {example['title']}: {example['record']['session_id']}")
    print(f"  {OUT.stat().st_size / 1024:.0f} KB")
    return 0 if examples else 1


if __name__ == "__main__":
    raise SystemExit(main())
