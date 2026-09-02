"""Resume support for the sweep runner.

A full n=5 sweep is ~2,000 requests against a free tier and spans days, so
resumability is a requirement rather than a nicety -- and until now it had no
test. It was verified by hand, once, which is exactly the kind of coverage that
survives right up until the moment the thing it guards changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from vichara.eval.metrics import TaskResult, agent_version_of
from vichara.eval.report import for_agent
from vichara.eval.runner import ResultStore
from vichara.eval.tasks.schema import Category

VERSION = agent_version_of({"act": "aaa", "plan": "bbb"})
OTHER = agent_version_of({"act": "CHANGED", "plan": "bbb"})


def write(path: Path, *rows: dict[str, object]) -> ResultStore:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return ResultStore(path)


def row(task_id: str, seed: int, version: str = VERSION) -> dict[str, object]:
    return {"task_id": task_id, "seed": seed, "agent_version": version}


class TestCompleted:
    def test_a_missing_file_means_nothing_is_done(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "absent.jsonl")

        assert store.completed(VERSION) == set()

    def test_pairs_from_this_agent_are_done(self, tmp_path: Path) -> None:
        store = write(tmp_path / "r.jsonl", row("t-1", 0), row("t-1", 1))

        assert store.completed(VERSION) == {("t-1", 0), ("t-1", 1)}

    def test_pairs_from_a_different_prompt_set_are_not_done(self, tmp_path: Path) -> None:
        """The bug this exists to prevent.

        Editing a prompt makes every earlier pair describe a different agent.
        Skipping them and writing only the remainder produces one results table
        spanning two agents, with a row count that looks entirely correct.
        """
        store = write(tmp_path / "r.jsonl", row("t-1", 0, OTHER), row("t-2", 0, OTHER))

        assert store.completed(VERSION) == set()

    def test_a_mixed_file_resumes_only_the_matching_half(self, tmp_path: Path) -> None:
        store = write(tmp_path / "r.jsonl", row("t-1", 0), row("t-2", 0, OTHER))

        assert store.completed(VERSION) == {("t-1", 0)}

    def test_rows_with_no_version_are_rerun(self, tmp_path: Path) -> None:
        """They came from an unknown prompt set.

        Re-running a pair is cheap next to reporting a number nobody can
        attribute to a specific agent.
        """
        store = write(tmp_path / "r.jsonl", {"task_id": "t-1", "seed": 0})

        assert store.completed(VERSION) == set()

    def test_a_truncated_final_line_just_reruns_that_pair(self, tmp_path: Path) -> None:
        """A sweep killed mid-write must not poison every later resume."""
        path = tmp_path / "r.jsonl"
        path.write_text(
            json.dumps(row("t-1", 0)) + "\n" + '{"task_id": "t-2", "se', encoding="utf-8"
        )

        assert ResultStore(path).completed(VERSION) == {("t-1", 0)}

    def test_blank_lines_are_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        path.write_text("\n" + json.dumps(row("t-1", 0)) + "\n\n", encoding="utf-8")

        assert ResultStore(path).completed(VERSION) == {("t-1", 0)}

    def test_a_seedless_run_is_tracked_distinctly(self, tmp_path: Path) -> None:
        """`--repeats 1` records seed None; it must not collide with seed 0."""
        store = write(tmp_path / "r.jsonl", {"task_id": "t-1", "agent_version": VERSION})

        assert store.completed(VERSION) == {("t-1", None)}


class TestReportScoping:
    """The first sweep after a prompt edit printed "230 runs / 41 tasks".

    116 of those came from the previous agent and 114 from the current one,
    averaged into a single table — the exact pooling `agent_version` exists to
    prevent, in the one place a person actually reads the number. `rescore`
    warned about it; the sweep's own end-of-run summary did not.
    """

    def result(self, task_id: str, version: str) -> TaskResult:
        return TaskResult(
            task_id=task_id, category=Category.SINGLE_TOOL, split="dev", agent_version=version
        )

    def test_only_this_agents_runs_are_reported(self) -> None:
        rows = [self.result("t-1", VERSION), self.result("t-2", OTHER)]

        assert [r.task_id for r in for_agent(rows, VERSION)] == ["t-1"]

    def test_a_resume_still_reports_the_whole_sweep(self) -> None:
        """The filter must not reintroduce the bug it replaced.

        Reporting only what one invocation produced makes a resumed sweep look
        worse than it was; every run by this agent still counts.
        """
        rows = [self.result(f"t-{i}", VERSION) for i in range(5)]

        assert len(for_agent(rows, VERSION)) == 5

    def test_unversioned_rows_are_excluded(self) -> None:
        """They came from an unknown prompt set and cannot be attributed."""
        assert for_agent([self.result("t-1", "")], VERSION) == []
