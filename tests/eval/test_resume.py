"""Resume support for the sweep runner.

A full n=5 sweep is ~2,000 requests against a free tier and spans days, so
resumability is a requirement rather than a nicety -- and until now it had no
test. It was verified by hand, once, which is exactly the kind of coverage that
survives right up until the moment the thing it guards changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from vichara.eval.injection_suite import completed_attacks
from vichara.eval.metrics import TaskResult, agent_version_of
from vichara.eval.report import for_agent
from vichara.eval.runner import ResultStore, _sweep_order
from vichara.eval.tasks.schema import Category, GoldTask, Terminal

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


class TestSweepOrder:
    """Task-major ordering made every interrupted sweep a biased sample.

    The quota died in the same place each time, so the partial sweep this
    project reported for weeks covered 92/95 single-tool tasks and 5/25
    ambiguous ones. Single-tool scores 0.989 complete and ambiguous 0.840, so
    the interruption selected the easy end of the set rather than a smaller
    slice of it.
    """

    def task(self, task_id: str, category: Category) -> GoldTask:
        return GoldTask.model_validate(
            {
                "id": task_id,
                "category": category,
                "split": "dev",
                "task": "a question about biology",
                "expected_terminal": Terminal.ANSWERED,
                "expected_tools": ["textbook_search"],
                "optimal_path": ["textbook_search"],
                "answer_contains": ["a"],
            }
        )

    def tasks(self) -> list[GoldTask]:
        return [self.task(f"easy-{i}", Category.SINGLE_TOOL) for i in range(4)] + [
            self.task(f"hard-{i}", Category.AMBIGUOUS) for i in range(2)
        ]

    def test_every_task_is_attempted_before_any_is_repeated(self) -> None:
        """The property that makes a truncated sweep honest."""
        order = _sweep_order(self.tasks(), [0, 1, 2])
        first_pass = [t.id for _s, t in order[: len(self.tasks())]]

        assert sorted(first_pass) == sorted(t.id for t in self.tasks())

    def test_truncation_keeps_the_category_mix(self) -> None:
        """Task-major put all four single-tool tasks before either ambiguous one."""
        order = _sweep_order(self.tasks(), [0, 1, 2])
        cut = [t.category for _s, t in order[:6]]

        assert cut.count(Category.AMBIGUOUS) == 2

    def test_nothing_is_dropped_or_duplicated(self) -> None:
        order = _sweep_order(self.tasks(), [0, 1, 2])

        assert len(order) == 18
        assert len({(s, t.id) for s, t in order}) == 18

    def test_seeds_advance_only_after_a_full_pass(self) -> None:
        order = _sweep_order(self.tasks(), [0, 1])
        seeds = [s for s, _t in order]

        assert seeds == [0] * 6 + [1] * 6

    def test_a_single_seed_is_unchanged(self) -> None:
        """--repeats 1 has no ordering question to get wrong."""
        order = _sweep_order(self.tasks(), [0])

        assert [t.id for _s, t in order] == [t.id for t in self.tasks()]


class TestAttackResume:
    """The same resume bug, in the place it did the most damage.

    Keyed on (attack_id, seed) alone, the injection suite marked all 28 attacks
    done the moment the file existed. A re-run after a prompt change executed
    nothing and printed a summary computed from the previous agent's rows,
    reporting a defence rate of 0.11 for a measurement that never ran. The
    giveaway was llm_requests coming back as zero, which 28 live attacks cannot
    do.
    """

    def write(self, path: Path, *rows: dict[str, object]) -> Path:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def attack(self, attack_id: str, version: str, seed: int = 0) -> dict[str, object]:
        return {"attack_id": attack_id, "seed": seed, "agent_version": version}

    def test_this_agents_attacks_are_done(self, tmp_path: Path) -> None:
        store = self.write(tmp_path / "a.jsonl", self.attack("exfil-1", VERSION))

        assert completed_attacks(store, VERSION) == {("exfil-1", 0)}

    def test_another_agents_attacks_are_rerun(self, tmp_path: Path) -> None:
        """The bug. These rows made the suite skip everything and report anyway."""
        store = self.write(tmp_path / "a.jsonl", self.attack("exfil-1", OTHER))

        assert completed_attacks(store, VERSION) == set()

    def test_unversioned_rows_are_rerun(self, tmp_path: Path) -> None:
        store = self.write(tmp_path / "a.jsonl", {"attack_id": "exfil-1", "seed": 0})

        assert completed_attacks(store, VERSION) == set()

    def test_a_missing_file_means_nothing_is_done(self, tmp_path: Path) -> None:
        assert completed_attacks(tmp_path / "absent.jsonl", VERSION) == set()

    def test_a_mixed_file_resumes_only_the_matching_half(self, tmp_path: Path) -> None:
        store = self.write(
            tmp_path / "a.jsonl",
            self.attack("exfil-1", VERSION),
            self.attack("cite-fake-url", OTHER),
        )

        assert completed_attacks(store, VERSION) == {("exfil-1", 0)}

    def test_a_truncated_final_line_just_reruns_that_attack(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        path.write_text(
            json.dumps(self.attack("exfil-1", VERSION)) + "\n" + '{"attack_id": "cit',
            encoding="utf-8",
        )

        assert completed_attacks(path, VERSION) == {("exfil-1", 0)}

    def test_seeds_are_tracked_separately(self, tmp_path: Path) -> None:
        store = self.write(
            tmp_path / "a.jsonl",
            self.attack("exfil-1", VERSION, seed=0),
            self.attack("exfil-1", VERSION, seed=1),
        )

        assert completed_attacks(store, VERSION) == {("exfil-1", 0), ("exfil-1", 1)}
