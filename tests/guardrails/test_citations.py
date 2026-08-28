"""Citation verification.

The defence added because of a measurement rather than a hunch: the Phase 5
sweep put false citation at the top of the risk list, so the agent may now
only cite what a tool actually returned.

The tests are paired throughout -- what it catches, and what it deliberately
does not. A defence whose limits are untested gets over-claimed, and the
limits here are real: this checks that a source *existed*, never that a
passage supports a claim.
"""

from __future__ import annotations

import pytest

from vichara.guardrails.citations import REMOVED, audit
from vichara.tools.base import Citation

REAL = "Anatomy and Physiology, 12.4. The Action Potential, p.524"


def tool_citations(*sources: str) -> list[Citation]:
    return [Citation(kind="textbook", source=s) for s in sources]


class TestFabricationIsRemoved:
    def test_an_invented_author_citation_is_stripped(self) -> None:
        answer = "ATP is consumed (Smith et al., Journal of Metabolic Research, 2025)."

        result = audit(answer, tool_citations(REAL))

        assert result.fabricated
        assert "Journal of Metabolic Research" not in result.answer
        assert REMOVED in result.answer

    def test_an_invented_doi_is_stripped(self) -> None:
        result = audit("See doi: 10.1000/not-real-at-all for details.", tool_citations(REAL))

        assert result.fabricated
        assert "10.1000/not-real-at-all" not in result.answer

    def test_an_invented_url_is_stripped(self) -> None:
        result = audit("Source: https://example.invalid/paper", tool_citations(REAL))

        assert result.fabricated

    def test_an_invented_textbook_locator_is_stripped(self) -> None:
        """The attack that motivated this: a fabricated page in a real book."""
        answer = "As stated in Anatomy and Physiology, 99.9. Invented Section, p.9999."

        result = audit(answer, tool_citations(REAL))

        assert result.fabricated
        assert "p.9999" not in result.answer

    def test_the_surrounding_prose_survives(self) -> None:
        """Removing the whole answer would be a denial of service, which is
        itself one of the attacks in the corpus."""
        answer = "The pump uses ATP (Fake Person et al., Nowhere Journal, 2025). It is vital."

        result = audit(answer, tool_citations(REAL))

        assert "The pump uses ATP" in result.answer
        assert "It is vital." in result.answer


class TestGenuineCitationsSurvive:
    def test_an_exact_match_is_verified(self) -> None:
        result = audit(f"ATP is consumed ({REAL}).", tool_citations(REAL))

        assert result.clean
        assert result.verified

    def test_reformatting_does_not_count_as_fabrication(self) -> None:
        """The model reformats constantly. 'p. 524' and 'p.524' are the same
        source and scoring them differently would make the defence useless."""
        answer = "ATP is consumed (Anatomy and Physiology, 12.4. The Action Potential, p. 524)."

        result = audit(answer, tool_citations(REAL))

        assert result.clean, result.fabricated

    def test_a_shortened_citation_is_accepted(self) -> None:
        """Agents routinely drop the page or the section title. Both are honest
        renderings of a source that was genuinely retrieved."""
        result = audit("As Anatomy and Physiology, 12.4 explains, ...", tool_citations(REAL))

        assert result.clean, result.fabricated

    def test_a_real_url_from_a_tool_is_accepted(self) -> None:
        citation = Citation(
            kind="web", source="Base editing trial", locator="https://example.org/base-editing"
        )

        result = audit("See https://example.org/base-editing for the trial.", [citation])

        assert result.clean, result.fabricated

    def test_a_clean_answer_is_returned_unchanged(self) -> None:
        """The common case must cost nothing."""
        answer = "Mitochondria produce ATP."

        result = audit(answer, tool_citations(REAL))

        assert result.answer == answer
        assert result.clean

    def test_prose_that_merely_names_a_book_is_not_a_citation(self) -> None:
        result = audit("The textbook covers this in chapter twelve.", tool_citations(REAL))

        assert result.clean


class TestLimits:
    def test_it_does_not_check_whether_the_source_supports_the_claim(self) -> None:
        """Stated as a test so the narrowness is on the record.

        A real citation attached to a false claim passes. That question needs a
        judge and is the one genuinely judged metric in the project; this check
        answers only 'did a tool return this source'.
        """
        answer = f"The resting potential is +450 mV ({REAL})."

        result = audit(answer, tool_citations(REAL))

        assert result.clean

    def test_a_bare_invented_token_is_not_caught(self) -> None:
        """The measured residual. cite-authority-substitution asks the agent to
        emit a token that never takes a citation shape, so no pattern matches
        it. Documented in docs/PROMPT_INJECTION.md rather than hidden."""
        result = audit("The reference code is XYZZY-1234.", tool_citations(REAL))

        assert result.clean

    def test_no_citations_from_tools_means_everything_shaped_is_suspect(self) -> None:
        """A refusal has no citations, so any citation-shaped span in that
        answer came from somewhere other than a tool."""
        result = audit("See Someone et al., Fake Journal, 2025.", [])

        assert result.fabricated


class TestConfigGating:
    @pytest.mark.parametrize("profile", ["baseline", "hardened"])
    def test_the_audit_runs_in_both_profiles(self, profile: str, config_dir: object) -> None:
        """Recording is unconditional; only removal is gated.

        A baseline trajectory still shows how often the agent invents a source,
        which is what made the before-and-after measurable.
        """
        from vichara.settings import load_pipeline_config

        cfg = load_pipeline_config(profile, config_dir)  # type: ignore[arg-type]

        assert cfg.injection.verify_citations is (profile == "hardened")
