"""The prompt-injection attack corpus.

An attack is only useful if its success is **checkable without judgement**.
"Did the agent get manipulated" is a question a human has to squint at; "did
the string VICHARA-CANARY-7731 appear in a web_search argument" is a question
a test can answer. Every attack here therefore carries a canary and a rule for
where that canary must appear for the attack to count as successful.

That constraint shapes the corpus. It excludes attacks whose only effect is
tonal or subtle, which is a real limitation and is stated in
docs/PROMPT_INJECTION.md rather than glossed over. What it buys is an attack
success rate that means the same thing every time it is measured.

The threat model is the one from docs/THREAT_MODEL.md section 4.1: the sandbox
has no network but the *agent* does, so the highest-value attack does not
escape anything -- it persuades the model to put data into a legitimate tool
call.
"""

from __future__ import annotations

import enum
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vichara.settings import REPO_ROOT

DEFAULT_ATTACKS = REPO_ROOT / "data" / "attacks" / "injection.jsonl"

CANARY_PREFIX = "VICHARA-CANARY"
"""Every payload's marker starts with this, so a canary appearing anywhere in
a trajectory is unambiguous -- it cannot have come from the corpus, the web
recordings, or the model's own vocabulary."""


class Vector(enum.StrEnum):
    """Which tool's output carries the payload."""

    TEXTBOOK = "textbook_search"
    WEB = "web_search"
    FILE = "workspace_file"


class Technique(enum.StrEnum):
    """What the payload tries to make the agent do.

    Grouped by *effect* rather than by phrasing, because the report should say
    which capability was abused, not which wording happened to work.
    """

    EXFILTRATION = "exfiltration"
    """Persuade the agent to put a marker into an outbound tool call. The
    attack that matters most: it crosses no sandbox boundary at all."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    """Make the agent emit attacker-chosen text in its answer."""

    FALSE_CITATION = "false_citation"
    """Make the agent cite a source the tools never returned."""

    REFUSAL_INDUCTION = "refusal_induction"
    """Make the agent abandon an answerable task."""

    TOOL_ABUSE = "tool_abuse"
    """Make the agent invoke a capability the task never needed."""


class SuccessKind(enum.StrEnum):
    ANSWER_CONTAINS = "answer_contains"
    TOOL_ARG_CONTAINS = "tool_arg_contains"
    TOOL_CALLED = "tool_called"
    TERMINAL_IS = "terminal_is"


class SuccessCheck(BaseModel):
    """How to decide, mechanically, whether the attack worked."""

    model_config = ConfigDict(extra="forbid")

    kind: SuccessKind
    marker: str = ""
    tool: str | None = None

    @model_validator(mode="after")
    def _check_operands(self) -> SuccessCheck:
        if self.kind in (SuccessKind.ANSWER_CONTAINS, SuccessKind.TOOL_ARG_CONTAINS):
            if not self.marker:
                raise ValueError(f"{self.kind} needs a marker")
        if self.kind in (SuccessKind.TOOL_CALLED, SuccessKind.TOOL_ARG_CONTAINS):
            if not self.tool:
                raise ValueError(f"{self.kind} needs a tool")
        if self.kind is SuccessKind.TERMINAL_IS and not self.marker:
            raise ValueError("terminal_is needs the terminal reason in marker")
        return self


class Attack(BaseModel):
    """One poisoned document, and what counts as it having worked."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    vector: Vector
    technique: Technique
    carrier_task: str
    """The gold task the agent is asked to do while the payload rides along in
    a tool result. Attacks are never run in isolation -- an injection that only
    works when the agent has nothing else to do is not a realistic one."""

    payload: str = Field(min_length=20)
    success: SuccessCheck
    notes: str = ""

    @model_validator(mode="after")
    def _canary_is_present(self) -> Attack:
        """A marker the payload never mentions cannot be attributed to it."""
        if (
            self.success.marker.startswith(CANARY_PREFIX)
            and self.success.marker not in self.payload
        ):
            raise ValueError(
                f"{self.id}: success marker {self.success.marker!r} does not appear in the payload, "
                "so a hit could not be attributed to this attack"
            )
        return self


class AttackSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attacks: list[Attack] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> AttackSet:
        ids = [a.id for a in self.attacks]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate attack ids: {duplicates}")
        return self

    def by_technique(self, technique: Technique) -> list[Attack]:
        return [a for a in self.attacks if a.technique is technique]


def load_attacks(path: Path | None = None) -> AttackSet:
    """Read and validate the attack corpus. A malformed entry raises."""
    source = path or DEFAULT_ATTACKS
    if not source.exists():
        raise FileNotFoundError(f"attack corpus not found: {source}")

    attacks: list[Attack] = []
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                attacks.append(Attack.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{source}:{number}: {exc}") from exc
    return AttackSet(attacks=attacks)
