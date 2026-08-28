"""The trajectory viewer.

Renderers are tested as pure functions over a TrajectoryRecord, so these cost
no quota and need no browser. The viewer is the demo, and a demo that breaks
silently is worse than no demo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vichara.settings import Settings, load_pipeline_config
from vichara.trajectory.schema import (
    GuardrailEvent,
    ObservationRecord,
    StepKind,
    StepRecord,
    TerminalReason,
    ToolCallRecord,
    TrajectoryRecord,
)
from vichara.ui.app import (
    _fallback,
    build_app,
    render_citations,
    render_cost,
    render_guardrails,
    render_trajectory,
)


def record(**kwargs: object) -> TrajectoryRecord:
    base: dict[str, object] = {
        "session_id": "s1",
        "task": "a question",
        "terminal_reason": TerminalReason.ANSWERED,
        "final_answer": "an answer",
    }
    base.update(kwargs)
    return TrajectoryRecord.model_validate(base)


def step_with_observation(*, flagged: bool = False, untrusted: bool = True) -> StepRecord:
    return StepRecord(
        index=1,
        kind=StepKind.EXECUTE,
        started_at="2026-01-01T00:00:00Z",
        tool_calls=[ToolCallRecord(tool="textbook_search", args={"query": "atp"})],
        observations=[
            ObservationRecord(
                step=1,
                tool="textbook_search",
                content="retrieved passage",
                trust="untrusted" if untrusted else "trusted",
                injection_flagged=flagged,
            )
        ],
    )


class TestTrajectoryRendering:
    def test_untrusted_output_is_labelled_as_untrusted(self) -> None:
        """A viewer that renders retrieved text like the agent's own reasoning
        teaches the reader the wrong model of where the risk is."""
        rendered = render_trajectory(record(steps=[step_with_observation()]))

        assert "untrusted tool output" in rendered

    def test_a_flagged_observation_is_marked(self) -> None:
        rendered = render_trajectory(record(steps=[step_with_observation(flagged=True)]))

        assert "injection flagged" in rendered

    def test_tool_calls_show_their_arguments(self) -> None:
        rendered = render_trajectory(record(steps=[step_with_observation()]))

        assert "textbook_search" in rendered
        assert "atp" in rendered

    def test_a_blocked_step_is_highlighted(self) -> None:
        rendered = render_trajectory(
            record(
                steps=[step_with_observation()],
                guardrail_events=[
                    GuardrailEvent(step=1, rule="per_tool_limit", action="block", detail="cap")
                ],
            )
        )

        assert "guardrail blocked" in rendered

    def test_an_empty_trajectory_does_not_crash(self) -> None:
        assert render_trajectory(record()).strip()


class TestCitationPanel:
    def test_verified_sources_are_listed(self) -> None:
        rendered = render_citations(record(citations=[{"source": "Biology, 4.2, p.188"}]))

        assert "Biology, 4.2, p.188" in rendered
        assert "Verified" in rendered

    def test_removed_citations_are_shown_as_removed(self) -> None:
        """The most valuable fact in the interface after Phase 5: a citation
        the agent invented, and the fact that it was caught."""
        rendered = render_citations(record(citations_fabricated=["Fake et al., 2025"]))

        assert "Fake et al., 2025" in rendered
        assert "no tool returned" in rendered

    def test_no_citations_is_explained_not_blank(self) -> None:
        rendered = render_citations(record(terminal_reason=TerminalReason.REFUSED))

        assert "refusal" in rendered.lower()


class TestCostPanel:
    def test_reports_requests_and_cache_hits_separately(self) -> None:
        """Requests are the binding budget on a free tier; a cache hit spends
        none, so collapsing them would misreport the cost."""
        rendered = render_cost(record(llm_requests=4, cache_hits=2))

        assert "| model requests | 4 |" in rendered
        assert "| cache hits | 2 |" in rendered

    def test_reports_the_capability_profile(self) -> None:
        rendered = render_cost(record(capability_profile=["textbook_search"]))

        assert "textbook_search" in rendered


class TestGuardrailPanel:
    def test_no_events_says_so(self) -> None:
        assert "No guardrail" in render_guardrails(record())

    def test_a_block_is_emphasised_over_an_allow(self) -> None:
        rendered = render_guardrails(
            record(
                guardrail_events=[
                    GuardrailEvent(step=1, rule="risk_class", action="allow", detail="x"),
                    GuardrailEvent(step=2, rule="loop", action="block", detail="y"),
                ]
            )
        )

        assert "**block**" in rendered
        assert "| allow |" in rendered


class TestReplayFallback:
    def test_missing_recordings_still_return_five_panels(self, tmp_path: Path) -> None:
        """The Space must never render a broken page. A recruiter who clicks a
        dead demo forms an impression that cannot be retaken."""
        panels = _fallback(tmp_path / "absent.jsonl", "no key")

        assert len(panels) == 5
        assert all(isinstance(p, str) for p in panels)

    def test_a_recording_is_served_with_a_visible_notice(self, tmp_path: Path) -> None:
        store = tmp_path / "runs.jsonl"
        store.write_text(record(final_answer="recorded").model_dump_json() + "\n", encoding="utf-8")

        answer, *_ = _fallback(store, "Replay mode selected.")

        assert "recorded" in answer
        assert "Replay mode selected." in answer, "the reader must know it is not live"


class TestAppConstruction:
    @pytest.mark.parametrize("profile", ["baseline", "hardened"])
    def test_the_app_builds_without_credentials(self, profile: str) -> None:
        """Same property the rest of the project holds: no key required."""
        assert build_app(Settings(), load_pipeline_config(profile)) is not None
