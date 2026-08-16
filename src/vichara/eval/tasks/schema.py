"""The gold task record.

Every field here is an *annotation*, written by a human before the agent is
run on the task. That ordering is the single most important discipline in the
whole evaluation: an optimal path written after watching a trajectory is not a
measurement, it is a rationalisation of whatever the agent happened to do.

Validated by pydantic with ``extra="forbid"``, and a test loads the shipped
set, so a malformed gold record is a build failure rather than a silently
skipped task.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(enum.StrEnum):
    """What the task is testing. Reported separately, never pooled -- a mean
    over categories with different success criteria means nothing."""

    SINGLE_TOOL = "single_tool"
    MULTI_TOOL = "multi_tool"
    AMBIGUOUS = "ambiguous"
    IMPOSSIBLE = "impossible"
    ADVERSARIAL = "adversarial"


class Split(enum.StrEnum):
    """Which half of the set a task belongs to.

    Prompts are tuned against ``dev`` only. ``test`` is looked at rarely and
    reported separately, because overfitting prompts to an eval set is the
    most common silent failure in agent evaluation and the only defence is not
    to look.
    """

    DEV = "dev"
    TEST = "test"


class Terminal(enum.StrEnum):
    """The terminal state a correct run should reach."""

    ANSWERED = "answered"
    REFUSED = "refused"
    CLARIFY = "clarify"


class GoldTask(BaseModel):
    """One annotated task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    category: Category
    split: Split
    task: str = Field(min_length=10)

    expected_terminal: Terminal
    expected_tools: list[str] = Field(default_factory=list)
    """Tools a correct trajectory *must* use. The recall denominator."""

    forbidden_tools: list[str] = Field(default_factory=list)
    """Tools whose use indicates a misread of the task -- reaching for web
    search on a settled textbook definition, for instance. Counted separately
    from merely inefficient tool choice because it means something different
    about the agent."""

    optimal_path: list[str] = Field(default_factory=list)
    """The minimal ordered tool sequence a competent human would use. The
    denominator of step efficiency, and the reason the whole set has to be
    annotated before anything is run."""

    answer_contains: list[str] = Field(default_factory=list)
    """Substrings, case-insensitive, that a correct answer must contain.
    Deliberately crude: a keyword check is a weak proxy for correctness but it
    is *programmatic*, and six mechanical metrics with a known weakness beat
    one judged metric with an unknown one."""

    answer_excludes: list[str] = Field(default_factory=list)
    """Claims that indicate a hallucination -- most usefully, the specific
    false premise of an impossible task."""

    must_cite: bool = True
    grounding_sources: list[str] = Field(default_factory=list)
    """Citation fragments expected in a grounded answer, e.g. a section
    number. Checked as substrings of the citation strings the tools returned,
    so a fabricated citation cannot satisfy it."""

    difficulty: int = Field(default=2, ge=1, le=3)
    notes: str = ""
    """Why this task is here and what it is meant to catch. Read by nobody
    until a number looks wrong, at which point it is the only thing that
    explains what the task was for."""

    @model_validator(mode="after")
    def _check_coherence(self) -> GoldTask:
        """Catch annotation mistakes that would silently corrupt a metric."""
        if self.expected_terminal is Terminal.ANSWERED and not self.answer_contains:
            raise ValueError(f"{self.id}: an answerable task needs answer_contains")

        if self.expected_terminal is Terminal.REFUSED:
            if self.expected_tools:
                raise ValueError(
                    f"{self.id}: a task that should be refused cannot require tools -- "
                    "refusing correctly means not calling them"
                )
            if self.must_cite:
                raise ValueError(f"{self.id}: a refusal has nothing to cite")

        overlap = set(self.expected_tools) & set(self.forbidden_tools)
        if overlap:
            raise ValueError(f"{self.id}: {sorted(overlap)} both expected and forbidden")

        # The two fields answer different questions, and the pilot surfaced
        # the difference. `expected_tools` is what a correct run *must* use --
        # the recall denominator. `optimal_path` is the trajectory a competent
        # human would take.
        #
        # For an answerable task those coincide, and a mismatch is an
        # annotation bug worth failing the build over. For a refusal task they
        # genuinely diverge: confirming that a chapter does not exist is a
        # reasonable first move, but requiring it would penalise an agent that
        # recognised the false premise outright -- which is better behaviour,
        # not worse.
        if self.expected_terminal is Terminal.ANSWERED:
            unexpected = set(self.optimal_path) - set(self.expected_tools)
            if unexpected:
                raise ValueError(
                    f"{self.id}: optimal_path uses {sorted(unexpected)}, which is not in "
                    "expected_tools -- one of the two annotations is wrong"
                )

        if self.category is Category.IMPOSSIBLE and self.expected_terminal is Terminal.ANSWERED:
            raise ValueError(f"{self.id}: an impossible task cannot expect an answer")

        return self

    @property
    def optimal_steps(self) -> int:
        """Denominator for step efficiency. At least one: even a refusal costs
        the plan step that produced it."""
        return max(len(self.optimal_path), 1)


class TaskSet(BaseModel):
    """The whole annotated set."""

    model_config = ConfigDict(extra="forbid")

    version: str = "v1"
    tasks: list[GoldTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> TaskSet:
        ids = [t.id for t in self.tasks]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate task ids: {duplicates}")
        return self

    def by_category(self, category: Category) -> list[GoldTask]:
        return [t for t in self.tasks if t.category is category]

    def by_split(self, split: Split) -> list[GoldTask]:
        return [t for t in self.tasks if t.split is split]
