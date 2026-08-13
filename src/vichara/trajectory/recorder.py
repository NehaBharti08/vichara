"""Building and persisting trajectory records.

The recorder is the only writer. Everything reaching disk passes through the
redactor first -- enforced here rather than left to each call site, because a
call site that forgets is a leaked credential, and there is no way to notice
after the fact.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from vichara.logging import get_logger
from vichara.trajectory.redact import Redactor
from vichara.trajectory.schema import (
    GuardrailEvent,
    ObservationRecord,
    StepKind,
    StepRecord,
    TrajectoryRecord,
)

log = get_logger(__name__)


def hash_prompts(directory: Path) -> dict[str, str]:
    """Content-hash every prompt file.

    Recorded on the trajectory so a run can be told apart from one made before
    a prompt was edited. Without it, comparing two weeks of numbers silently
    assumes the agent did not change.
    """
    if not directory.exists():
        return {}
    hashes: dict[str, str] = {}
    for path in sorted(directory.glob("*.md")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.stem] = digest[:16]
    return hashes


class TrajectoryRecorder:
    """Accumulates one run, then writes it once."""

    def __init__(
        self,
        record: TrajectoryRecord,
        *,
        redactor: Redactor,
        store_path: Path | None = None,
    ) -> None:
        self.record = record
        self.redactor = redactor
        self.store_path = store_path
        self._open_step: StepRecord | None = None
        self._step_started: float = 0.0

    # -- Steps --------------------------------------------------------------

    def begin_step(self, kind: StepKind) -> StepRecord:
        step = StepRecord(
            index=len(self.record.steps),
            kind=kind,
            started_at=datetime.now(UTC).isoformat(),
        )
        self.record.steps.append(step)
        self._open_step = step
        self._step_started = time.perf_counter()
        return step

    def end_step(self, **fields: object) -> None:
        if self._open_step is None:
            return
        self._open_step.duration_ms = (time.perf_counter() - self._step_started) * 1000
        for name, value in fields.items():
            if hasattr(self._open_step, name):
                setattr(self._open_step, name, value)
        self._open_step = None

    def add_observation(self, observation: ObservationRecord) -> None:
        if self._open_step is not None:
            self._open_step.observations.append(observation)

    def add_guardrail_event(self, event: GuardrailEvent) -> None:
        self.record.guardrail_events.append(event)

    # -- Persistence --------------------------------------------------------

    def finalise(self) -> TrajectoryRecord:
        """Scrub, stamp, and return the record.

        Redaction happens here, once, over the whole structure -- not per field
        as it is added. A single pass at the boundary is auditable; scattered
        calls are a checklist someone will eventually miss.
        """
        self.record.finished_at = datetime.now(UTC).isoformat()
        cleaned = TrajectoryRecord.model_validate(
            self.redactor.scrub(self.record.model_dump(mode="json"))
        )
        self.record = cleaned
        return cleaned

    def write(self) -> Path | None:
        """Append the finalised record to the JSONL store."""
        if self.store_path is None:
            return None
        record = self.finalise()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(record.model_dump_json() + "\n")
        log.info(
            "trajectory written",
            session_id=record.session_id,
            terminal=record.terminal_reason,
            steps=len(record.steps),
            path=str(self.store_path),
        )
        return self.store_path


def read_trajectories(path: Path) -> Iterator[TrajectoryRecord]:
    """Stream records back.

    A malformed line is skipped with a warning rather than raising: a sweep
    interrupted mid-write should not make every earlier result unreadable.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield TrajectoryRecord.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning(
                    "skipping malformed trajectory",
                    path=str(path),
                    line=number,
                    error=str(exc)[:120],
                )
