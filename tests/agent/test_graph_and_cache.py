"""Routing, guardrails, and the response cache.

Routing is tested as pure functions over state rather than by running the
graph. That keeps these tests free of model calls and quota, which matters
because they are the ones that must run on every commit.
"""

from __future__ import annotations

import difflib
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vichara.agent.nodes.acting import (
    _as_redundant,
    _budget_line,
    _first_matching_step,
    route_after_act,
    route_after_approve,
    route_after_guard,
)
from vichara.agent.nodes.context import load_prompt
from vichara.agent.nodes.planning import route_after_plan, route_after_reflect
from vichara.agent.state import ActionFingerprint, AgentState, PendingAction, initial_state
from vichara.llm.accounting import CallRecord, Ledger, estimate_usd, usage_from_response
from vichara.llm.cache import ResponseCache, cache_key
from vichara.llm.provider import text_of
from vichara.llm.ratelimit import TokenBucket, is_rate_limit
from vichara.settings import LoopConfig, PipelineConfig
from vichara.trajectory.schema import ObservationRecord, Plan, TerminalReason


def state(**fields: Any) -> AgentState:
    base = initial_state(task="t", session_id="s1", capability_profile=[])
    base.update(fields)  # type: ignore[typeddict-item]
    return base


class TestPlanRouting:
    def test_unanswerable_refuses_immediately(self) -> None:
        """Refusal on step one is correct; refusal on step twelve is a failure."""
        assert route_after_plan(state(plan=Plan(answerable=False))) == "refuse"

    def test_ambiguous_asks_rather_than_guessing(self) -> None:
        plan = Plan(answerable=True, needs_clarification=True, clarifying_question="which?")

        assert route_after_plan(state(plan=plan)) == "clarify"

    def test_normal_plan_proceeds(self) -> None:
        assert route_after_plan(state(plan=Plan(answerable=True))) == "act"

    def test_a_terminal_reason_always_wins(self) -> None:
        s = state(plan=Plan(answerable=True), terminal_reason=TerminalReason.FATAL_ERROR)

        assert route_after_plan(s) == "halt"


class TestActRouting:
    def test_a_tool_call_goes_through_the_guard(self) -> None:
        s = state(pending_action=PendingAction(tool="web_search"))

        assert route_after_act(s) == "guard"

    def test_no_tool_call_means_ready_to_answer(self) -> None:
        assert route_after_act(state(pending_action=None)) == "synthesize"


class TestGuardRouting:
    def test_destructive_actions_require_approval(self) -> None:
        s = state(pending_action=PendingAction(tool="run_python", risk="destructive"))

        assert route_after_guard(s) == "approve"

    def test_read_actions_execute_directly(self) -> None:
        """Approving every retrieval would make the interrupt noise, not signal."""
        s = state(pending_action=PendingAction(tool="web_search", risk="read"))

        assert route_after_guard(s) == "execute"

    def test_a_soft_ceiling_answers_from_what_it_has(self) -> None:
        """A per-tool limit must not discard evidence already gathered.

        An agent holding thirteen citations that reports only "budget
        exhausted" has thrown the work away. The ceiling exists to stop it
        spending more, not to make it forget.
        """
        assert route_after_guard(state(force_synthesis=True)) == "synthesize"

    def test_a_hard_stop_halts(self) -> None:
        s = state(terminal_reason=TerminalReason.LOOP_DETECTED)

        assert route_after_guard(s) == "halt"


class TestApprovalRouting:
    def test_denial_is_observed_not_silently_dropped(self) -> None:
        """A refusal the agent never sees looks like a tool that did nothing,
        and it would go on trying."""
        assert route_after_approve(state(approval_denied=True)) == "observe"

    def test_approval_executes(self) -> None:
        assert route_after_approve(state(approval_denied=False)) == "execute"


class TestReflectRouting:
    def test_giving_up_halts(self) -> None:
        assert route_after_reflect(state(terminal_reason=TerminalReason.REFUSED)) == "halt"

    def test_a_revision_returns_to_planning(self) -> None:
        s = state(plan=Plan(revision=0), plan_revisions=1)

        assert route_after_reflect(s) == "plan"

    def test_continuing_returns_to_acting(self) -> None:
        s = state(plan=Plan(revision=1), plan_revisions=1)

        assert route_after_reflect(s) == "act"


class TestActionFingerprints:
    def test_identical_calls_share_a_digest(self) -> None:
        a = ActionFingerprint.of(1, "web_search", {"query": "photosynthesis"})
        b = ActionFingerprint.of(5, "web_search", {"query": "photosynthesis"})

        assert a.digest == b.digest

    def test_case_and_spacing_do_not_disguise_a_repeat(self) -> None:
        """An agent that 'varies' its query by changing the case is looping."""
        a = ActionFingerprint.of(1, "web_search", {"query": "Photosynthesis  Light"})
        b = ActionFingerprint.of(2, "web_search", {"query": "photosynthesis light"})

        assert a.digest == b.digest

    def test_argument_order_does_not_matter(self) -> None:
        a = ActionFingerprint.of(1, "t", {"a": 1, "b": 2})
        b = ActionFingerprint.of(1, "t", {"b": 2, "a": 1})

        assert a.digest == b.digest

    def test_different_queries_differ(self) -> None:
        a = ActionFingerprint.of(1, "web_search", {"query": "mitosis"})
        b = ActionFingerprint.of(1, "web_search", {"query": "meiosis"})

        assert a.digest != b.digest

    def test_the_same_query_to_a_different_tool_differs(self) -> None:
        a = ActionFingerprint.of(1, "web_search", {"query": "x"})
        b = ActionFingerprint.of(1, "textbook_search", {"query": "x"})

        assert a.digest != b.digest


class TestCacheKey:
    def test_identical_calls_share_a_key(self) -> None:
        args: dict[str, Any] = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": ["a"],
            "temperature": 0.0,
            "seed": None,
        }

        assert cache_key(**args) == cache_key(**args)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model", "other-model"),
            ("temperature", 0.7),
            ("seed", 42),
            ("tools", ["a", "b"]),
        ],
    )
    def test_anything_that_changes_the_answer_changes_the_key(self, field: str, value: Any) -> None:
        """A key on the prompt alone would serve a flash-lite answer for a pro
        request and quietly invalidate the planner ablation."""
        base: dict[str, Any] = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": ["a"],
            "temperature": 0.0,
            "seed": None,
        }

        assert cache_key(**base) != cache_key(**{**base, field: value})

    def test_tool_order_does_not_change_the_key(self) -> None:
        base: dict[str, Any] = {
            "model": "m",
            "messages": [],
            "temperature": 0.0,
            "seed": None,
        }

        assert cache_key(**base, tools=["b", "a"]) == cache_key(**base, tools=["a", "b"])


class TestResponseCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c.sqlite")
        cache.put("k", model="m", role="agent", payload={"content": "hello"})

        assert cache.get("k") == {"content": "hello"}
        cache.close()

    def test_a_miss_is_none(self, tmp_path: Path) -> None:
        assert ResponseCache(tmp_path / "c.sqlite").get("absent") is None

    def test_disabled_cache_stores_nothing(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path / "c.sqlite", enabled=False)
        cache.put("k", model="m", role="agent", payload={"x": 1})

        assert cache.get("k") is None

    def test_daily_request_counter_persists(self, tmp_path: Path) -> None:
        """An accidental loop must not burn the day's allowance without the
        number surviving the process that spent it."""
        path = tmp_path / "c.sqlite"
        first = ResponseCache(path)
        first.record_request("2026-01-01")
        first.record_request("2026-01-01")
        first.close()

        assert ResponseCache(path).requests_today("2026-01-01") == 2

    def test_a_corrupt_row_is_a_miss_not_an_error(self, tmp_path: Path) -> None:
        """The cache is an optimisation and must never be able to fail a run."""
        path = tmp_path / "c.sqlite"
        cache = ResponseCache(path)
        cache.put("k", model="m", role="agent", payload={"x": 1})
        connection = cache._connect()
        connection.execute("UPDATE responses SET payload = '{broken' WHERE key = 'k'")
        connection.commit()

        assert cache.get("k") is None


class TestAccounting:
    def test_requests_exclude_cache_hits(self) -> None:
        """Requests are the binding budget on a free tier; a cache hit spends none."""
        ledger = Ledger()
        ledger.record(CallRecord(model="m", role="agent", cache_hit=True))
        ledger.record(CallRecord(model="m", role="agent", input_tokens=10))

        assert ledger.requests == 1
        assert ledger.cache_hits == 1

    def test_free_tier_models_cost_nothing(self) -> None:
        assert estimate_usd("gemini-3.5-flash-lite", 1_000_000, 1_000_000) == 0.0

    def test_priced_models_are_estimated(self) -> None:
        assert estimate_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)

    def test_unknown_models_cost_nothing_rather_than_guessing(self) -> None:
        assert estimate_usd("some-future-model", 1_000_000, 1_000_000) == 0.0

    def test_usage_is_read_defensively(self) -> None:
        """A missing token count must degrade the accounting, not break the run."""
        assert usage_from_response(object()) == (0, 0, 0)

    def test_cache_read_tokens_are_captured(self) -> None:
        response = type(
            "R",
            (),
            {
                "usage_metadata": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "input_token_details": {"cache_read": 60},
                }
            },
        )()

        assert usage_from_response(response) == (100, 20, 60)


class TestContentBlocks:
    def test_plain_string_content(self) -> None:
        assert text_of(type("R", (), {"content": "hello"})()) == "hello"

    def test_content_blocks_are_flattened(self) -> None:
        """langchain-core 1.x returns a list of blocks; str() on it produces a
        literal repr in the answer. Caught by the Phase 1 spike."""
        response = type(
            "R",
            (),
            {"content": [{"type": "text", "text": "17 * 23 = 391", "extras": {}}]},
        )()

        assert text_of(response) == "17 * 23 = 391"

    def test_non_text_blocks_are_dropped(self) -> None:
        response = type(
            "R",
            (),
            {"content": [{"type": "image", "url": "x"}, {"type": "text", "text": "ok"}]},
        )()

        assert text_of(response) == "ok"


class TestRateLimiting:
    @pytest.mark.parametrize(
        "message",
        ["429 Too Many Requests", "RESOURCE_EXHAUSTED", "rate limit exceeded", "quota"],
    )
    def test_throttles_are_recognised_across_providers(self, message: str) -> None:
        assert is_rate_limit(RuntimeError(message)) is True

    def test_other_errors_are_not_retried_as_throttles(self) -> None:
        """Retrying a bad key burns quota to reproduce the same failure."""
        assert is_rate_limit(ValueError("invalid api key")) is False

    def test_bucket_allows_the_first_call_immediately(self) -> None:
        assert TokenBucket(rate_per_minute=60).acquire(timeout_s=1) is True


class TestRedundantResults:
    """Loop detection watched what the agent asked, not what it got back.

    `rag-hypothalamus-pituitary` called `textbook_search` five times where one
    sufficed, reformulating enough to clear the near-repeat threshold every
    time while BM25 returned byte-identical passages. It was simultaneously the
    worst step-efficiency task and one of only two unstable across seeds.
    """

    def obs(self, step: int, content: str, **kw: Any) -> ObservationRecord:
        return ObservationRecord(step=step, tool="textbook_search", content=content, **kw)

    def test_the_arguments_that_evaded_near_repeat(self) -> None:
        """The real queries, scoring well under the 0.9 similarity threshold.

        This is why a result-side check had to exist: tightening the argument
        threshold to catch these would flag legitimate reformulation too.
        """
        queries = [
            "hypothalamus control anterior pituitary gland",
            "hypothalamus control anterior pituitary hypophyseal portal system",
            "hypothalamus hypophyseal portal system anterior pituitary",
        ]
        ratios = [
            difflib.SequenceMatcher(None, a, b).ratio() for a, b in itertools.pairwise(queries)
        ]

        assert max(ratios) < LoopConfig().near_repeat_similarity

    def test_identical_bytes_are_traced_to_the_first_step(self) -> None:
        prior = [self.obs(2, "the pituitary is suspended from the infundibulum")]

        assert _first_matching_step(prior[0].content, prior) == 2

    def test_different_bytes_are_progress(self) -> None:
        prior = [self.obs(2, "the pituitary is suspended from the infundibulum")]

        assert (
            _first_matching_step("hypophyseal portal veins carry releasing hormones", prior) is None
        )

    def test_a_repeat_of_a_repeat_points_at_the_original(self) -> None:
        """Otherwise the third call cites the second, which cites the first."""
        text = "the pituitary is suspended from the infundibulum"
        prior = [self.obs(2, text), _as_redundant(self.obs(4, text), earlier=2)]

        assert _first_matching_step(text, prior) == 2

    def test_a_repeated_failure_is_not_a_redundant_result(self) -> None:
        """Two timeouts in a row are a broken tool, and retrying is reasonable."""
        prior = [self.obs(2, "upstream timed out", ok=False, error_code="timeout")]

        assert _first_matching_step(prior[0].content, prior) is None

    def test_the_duplicate_body_is_dropped_but_the_citations_survive(self) -> None:
        """The evidence is already in state from the first identical call.

        Re-sending ~11KB the model has read is what made the loop expensive;
        dropping the citations *here* would be the soft-ceiling bug again.
        """
        original = self.obs(2, "x" * 11000, citations=[{"citation": "A&P 17.3, p.744"}])
        marked = _as_redundant(original, earlier=2)

        assert marked.redundant
        assert len(marked.content) < 400
        assert "step 2" in marked.content

    def test_the_agent_is_told_what_to_do_instead(self) -> None:
        """A warning that does not name an alternative just burns the step."""
        marked = _as_redundant(self.obs(4, "some passage"), earlier=2)

        assert "Answer from what" in marked.content
        assert "different tool" in marked.content


class TestBudgetVisibility:
    """The guard enforced ceilings the agent could not see.

    Duplicate results explain only 3 of the 36 sub-optimal runs outright. The
    rest retrieve different passages and never decide they are done, which is a
    judgement `act` was asked to make without being shown step count, tool
    spend, or how much evidence it already held.
    """

    def line(self, s: AgentState, step: int = 1) -> str:
        """`_budget_line` reads only the config, so a full context is noise."""
        return _budget_line(s, step, SimpleNamespace(config=PipelineConfig()))  # type: ignore[arg-type]

    def test_the_step_and_its_ceiling_are_both_named(self) -> None:
        """A step number without its ceiling is not a budget."""
        assert "Step 3 of 8" in self.line(state(), step=3)

    def test_a_fresh_run_says_so_plainly(self) -> None:
        assert "No tools called yet" in self.line(state())

    def test_spend_is_shown_against_the_limit_that_will_stop_it(self) -> None:
        s = state(tool_calls={"textbook_search": 3})

        assert "textbook_search 3/4" in self.line(s)

    def test_untouched_tools_are_not_listed(self) -> None:
        """Listing every tool at 0/4 buries the one that is nearly spent."""
        s = state(tool_calls={"textbook_search": 3, "web_search": 0})

        assert "web_search" not in self.line(s)

    def test_accumulated_evidence_is_counted(self) -> None:
        """The concrete form of "you may already be done"."""
        s = state(citations=[{"citation": f"src {i}"} for i in range(12)])

        assert "holding 12 source(s)" in self.line(s)

    def test_the_prompt_renders_with_the_budget_slot_filled(self) -> None:
        """A missing key raises at format() time, mid-run, on every task."""
        rendered = load_prompt("act").format(
            task="t", plan="p", summary="s", reflect_note="", budget="Step 1 of 8."
        )

        assert "Step 1 of 8." in rendered
