"""The metrics, and the annotations they read.

Scored against synthetic trajectories rather than real runs, so these cost no
quota and can gate every commit. A metric that only works on trajectories the
agent happens to produce today is not a measurement instrument.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vichara.eval.metrics import (
    MAX_REFUSAL_STEPS,
    Distribution,
    agent_version,
    rate,
    score,
)
from vichara.eval.tasks.loader import load_tasks
from vichara.eval.tasks.schema import Category, GoldTask, Split, Terminal
from vichara.trajectory.schema import (
    ObservationRecord,
    StepKind,
    StepRecord,
    TerminalReason,
    ToolCallRecord,
    TrajectoryRecord,
)


def trajectory(
    *,
    tools: list[str] | None = None,
    terminal: TerminalReason = TerminalReason.ANSWERED,
    answer: str = "an answer",
    citations: list[str] | None = None,
    failed_tool: str | None = None,
) -> TrajectoryRecord:
    tools = tools or []
    citations = citations or []
    steps = [
        StepRecord(
            index=i,
            kind=StepKind.ACT,
            started_at="2026-01-01T00:00:00Z",
            tool_calls=[ToolCallRecord(tool=tool)],
        )
        for i, tool in enumerate(tools)
    ]
    if failed_tool:
        steps.append(
            StepRecord(
                index=len(steps),
                kind=StepKind.EXECUTE,
                started_at="2026-01-01T00:00:00Z",
                observations=[
                    ObservationRecord(step=0, tool=failed_tool, ok=False, content="failed")
                ],
            )
        )
    return TrajectoryRecord(
        session_id="s1",
        task="t",
        steps=steps,
        terminal_reason=terminal,
        final_answer=answer,
        citations=[{"source": c} for c in citations],
    )


def gold(**overrides: object) -> GoldTask:
    base: dict[str, object] = {
        "id": "t-1",
        "category": Category.SINGLE_TOOL,
        "split": "dev",
        "task": "a question about biology",
        "expected_terminal": Terminal.ANSWERED,
        "expected_tools": ["textbook_search"],
        "optimal_path": ["textbook_search"],
        "answer_contains": ["answer"],
    }
    base.update(overrides)
    return GoldTask.model_validate(base)


class TestShippedTaskSet:
    def test_loads_and_validates(self) -> None:
        """A malformed gold record is a build failure, not a skipped task --
        a silently dropped task changes every denominator."""
        tasks = load_tasks()

        assert len(tasks.tasks) >= 10

    def test_every_category_is_represented(self) -> None:
        tasks = load_tasks()

        present = {t.category for t in tasks.tasks}
        assert Category.SINGLE_TOOL in present
        assert Category.MULTI_TOOL in present
        assert Category.IMPOSSIBLE in present
        assert Category.AMBIGUOUS in present

    def test_a_held_out_split_exists(self) -> None:
        """Prompts are tuned on dev only; overfitting to the eval set is the
        commonest silent failure in agent evaluation."""
        tasks = load_tasks()

        assert tasks.by_split(Split.TEST)
        assert tasks.by_split(Split.DEV)


class TestAnnotationValidation:
    def test_a_refusal_task_cannot_require_tools(self) -> None:
        with pytest.raises(ValidationError, match="cannot require tools"):
            gold(expected_terminal=Terminal.REFUSED, must_cite=False, answer_contains=[])

    def test_a_refusal_task_has_nothing_to_cite(self) -> None:
        with pytest.raises(ValidationError, match="nothing to cite"):
            gold(
                expected_terminal=Terminal.REFUSED,
                expected_tools=[],
                optimal_path=[],
                must_cite=True,
                answer_contains=[],
            )

    def test_a_tool_cannot_be_expected_and_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="both expected and forbidden"):
            gold(forbidden_tools=["textbook_search"])

    def test_an_answerable_task_needs_a_success_criterion(self) -> None:
        with pytest.raises(ValidationError, match="needs answer_contains"):
            gold(answer_contains=[])

    def test_an_impossible_task_cannot_expect_an_answer(self) -> None:
        with pytest.raises(ValidationError, match="cannot expect an answer"):
            gold(category=Category.IMPOSSIBLE)

    def test_optimal_path_must_use_expected_tools_when_answerable(self) -> None:
        with pytest.raises(ValidationError, match="optimal_path uses"):
            gold(optimal_path=["web_search"])

    def test_a_refusal_task_may_include_a_confirming_check(self) -> None:
        """The distinction the pilot surfaced: `expected_tools` is what a run
        must use, `optimal_path` is what a competent human would do. For a
        refusal they legitimately differ -- confirming a chapter does not
        exist is reasonable, but requiring it would penalise an agent that
        recognised the false premise outright."""
        task = gold(
            category=Category.IMPOSSIBLE,
            expected_terminal=Terminal.REFUSED,
            expected_tools=[],
            optimal_path=["textbook_search"],
            must_cite=False,
            answer_contains=[],
        )

        assert task.optimal_steps == 1


class TestScoring:
    def test_correct_terminal_state(self) -> None:
        result = score(trajectory(tools=["textbook_search"]), gold())

        assert result.terminal_correct is True

    def test_tool_precision_and_recall(self) -> None:
        record = trajectory(tools=["textbook_search", "web_search"])

        result = score(record, gold())

        assert result.tool_recall == 1.0
        assert result.tool_precision == 0.5

    def test_forbidden_tool_is_flagged(self) -> None:
        record = trajectory(tools=["textbook_search", "web_search"])

        result = score(record, gold(forbidden_tools=["web_search"]))

        assert result.used_forbidden_tool is True

    def test_tool_metrics_are_undefined_when_no_tool_is_expected(self) -> None:
        """Scoring a refusal task 0.0 would drag the mean down with a number
        that means nothing."""
        task = gold(
            expected_terminal=Terminal.REFUSED,
            expected_tools=[],
            optimal_path=[],
            must_cite=False,
            answer_contains=[],
        )

        result = score(trajectory(terminal=TerminalReason.REFUSED), task)

        assert result.tool_precision is None
        assert result.tool_recall is None

    def test_step_efficiency_is_capped_at_one(self) -> None:
        """Beating the annotated optimum means the annotation was wrong, not
        that the agent earned a score above 1."""
        result = score(
            trajectory(tools=["textbook_search"]), gold(optimal_path=["textbook_search"])
        )

        assert result.step_efficiency == 1.0

    def test_step_efficiency_falls_with_wasted_steps(self) -> None:
        record = trajectory(tools=["textbook_search"] * 4)

        result = score(record, gold())

        assert result.step_efficiency == pytest.approx(0.25)

    def test_answer_excludes_catches_a_hallucination(self) -> None:
        record = trajectory(answer="The chapter covers quantum coherence in birds.")

        result = score(
            record,
            gold(
                answer_contains=[],
                answer_excludes=["quantum coherence"],
                category=Category.IMPOSSIBLE,
                expected_terminal=Terminal.REFUSED,
                expected_tools=[],
                optimal_path=[],
                must_cite=False,
            ),
        )

        assert result.answer_correct is False

    def test_grounding_is_checked_against_tool_citations(self) -> None:
        """Checked against what the tools returned, so a citation the model
        invented in its prose cannot satisfy it."""
        record = trajectory(tools=["textbook_search"], citations=["Biology, 12.4 X, p.5"])

        result = score(record, gold(grounding_sources=["12.4"]))

        assert result.grounding_sources_present is True

    def test_missing_grounding_is_detected(self) -> None:
        record = trajectory(tools=["textbook_search"], citations=["Biology, 3.1 Y, p.9"])

        result = score(record, gold(grounding_sources=["12.4"]))

        assert result.grounding_sources_present is False


class TestRefusal:
    def _task(self) -> GoldTask:
        return gold(
            category=Category.IMPOSSIBLE,
            expected_terminal=Terminal.REFUSED,
            expected_tools=[],
            optimal_path=[],
            must_cite=False,
            answer_contains=[],
        )

    def test_a_fast_refusal_is_correct(self) -> None:
        result = score(trajectory(terminal=TerminalReason.REFUSED), self._task())

        assert result.refusal_correct is True

    def test_a_slow_refusal_is_not(self) -> None:
        """The whole point of gating on steps: an agent that grinds through
        fifteen steps before saying 'I don't know' is broken even though the
        words are right."""
        record = trajectory(
            tools=["textbook_search"] * (MAX_REFUSAL_STEPS + 2),
            terminal=TerminalReason.REFUSED,
        )

        result = score(record, self._task())

        assert result.refusal_correct is False

    def test_refusing_an_answerable_task_is_counted_separately(self) -> None:
        """Otherwise an agent that refuses everything scores perfectly on
        refusal correctness."""
        result = score(trajectory(terminal=TerminalReason.REFUSED), gold())

        assert result.false_refusal is True
        assert result.refusal_correct is None


class TestDistribution:
    def test_reports_spread_not_just_a_midpoint(self) -> None:
        dist = Distribution.of([0.0, 0.5, 0.5, 1.0])

        assert dist is not None
        assert dist.median == 0.5
        assert dist.minimum == 0.0
        assert dist.maximum == 1.0
        assert dist.iqr > 0

    def test_a_single_run_has_no_spread(self) -> None:
        dist = Distribution.of([0.7])

        assert dist is not None
        assert dist.median == dist.minimum == dist.maximum == 0.7

    def test_empty_is_none_not_zero(self) -> None:
        assert Distribution.of([]) is None

    def test_rate_ignores_inapplicable_runs(self) -> None:
        assert rate([True, False, None, True]) == pytest.approx(0.6667, abs=1e-3)

    def test_rate_of_nothing_is_none(self) -> None:
        assert rate([None, None]) is None


class TestAgentVersion:
    """`prompt_hashes` was recorded from Phase 3 and then never read.

    A live sweep was found still running against exhausted quota while the act
    prompt was edited underneath it. `AgentSession` reloads prompts from disk
    per task, so it would have written one results file describing two
    different agents, with a row count that still looked right.
    """

    def rec(self, **hashes: str) -> TrajectoryRecord:
        return TrajectoryRecord(session_id="s1", task="t", steps=[], prompt_hashes=dict(hashes))

    def test_the_same_prompt_set_is_the_same_version(self) -> None:
        a = agent_version(self.rec(act="aaa", plan="bbb"))

        assert a == agent_version(self.rec(act="aaa", plan="bbb"))

    def test_editing_any_prompt_changes_the_version(self) -> None:
        """Not just `act`. Every prompt shapes the run being scored."""
        before = agent_version(self.rec(act="aaa", plan="bbb"))

        assert agent_version(self.rec(act="aaa", plan="CHANGED")) != before

    def test_key_order_is_not_a_version_change(self) -> None:
        """Otherwise dict ordering alone would split one sweep into two."""
        assert agent_version(self.rec(act="a", plan="b")) == agent_version(
            self.rec(plan="b", act="a")
        )

    def test_a_swap_between_prompts_is_still_a_change(self) -> None:
        """Folding the digests must not lose which prompt held which."""
        assert agent_version(self.rec(act="a", plan="b")) != agent_version(
            self.rec(act="b", plan="a")
        )

    def test_a_trajectory_with_no_hashes_claims_no_version(self) -> None:
        """Empty is honest; a digest of nothing would look like a real version."""
        assert agent_version(self.rec()) == ""

    def test_the_version_reaches_the_result_row(self) -> None:
        """The whole point: the split has to be visible in eval_results/."""
        record = trajectory(tools=["textbook_search"], answer="answer")
        record.prompt_hashes = {"act": "aaa"}

        assert score(record, gold()).agent_version == agent_version(record)
