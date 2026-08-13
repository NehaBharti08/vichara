"""Assembling and running one agent session.

Everything a run needs is built here: the registry, the provider, the
recorder, the checkpointer and the graph. Kept out of the CLI so the API,
the UI and the evaluation runner all start a run the same way -- three
divergent setup paths would mean the eval measured something the demo does
not do.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from vichara.agent.graph import build_graph
from vichara.agent.nodes.context import PROMPT_DIR, AgentContext
from vichara.agent.state import AgentState, initial_state
from vichara.llm.provider import Provider
from vichara.logging import bind_run, clear_run, get_logger
from vichara.settings import PipelineConfig, Settings
from vichara.tools.registry import build_registry
from vichara.trajectory.recorder import TrajectoryRecorder, hash_prompts
from vichara.trajectory.redact import Redactor
from vichara.trajectory.schema import TrajectoryRecord

log = get_logger(__name__)

DEFAULT_TRAJECTORY_STORE = Path("trajectories") / "runs.jsonl"

# Types that travel in graph state and therefore through the checkpoint.
# LangGraph refuses to deserialise unregistered types (a warning today, an
# error in a later version) because a checkpoint is untrusted input: anything
# that can name an arbitrary class at load time is a deserialisation gadget.
# Naming them explicitly keeps resumption working and keeps the allowlist
# small enough to audit -- which is the point of the mechanism.
_CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = (
    ("vichara.agent.state", "ActionFingerprint"),
    ("vichara.agent.state", "BudgetState"),
    ("vichara.agent.state", "PendingAction"),
    ("vichara.trajectory.schema", "Plan"),
    ("vichara.trajectory.schema", "PlanStep"),
    ("vichara.trajectory.schema", "ObservationRecord"),
    ("vichara.trajectory.schema", "GuardrailEvent"),
    ("vichara.trajectory.schema", "TerminalReason"),
)


def build_serialiser() -> JsonPlusSerializer:
    """Checkpoint serialiser that knows this project's state types."""
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)


def new_session_id() -> str:
    """Short, filesystem-safe, and valid as a workspace directory name."""
    return f"s{uuid.uuid4().hex[:12]}"


@dataclass
class RunOutcome:
    """What a completed or suspended run produced."""

    session_id: str
    state: AgentState
    record: TrajectoryRecord
    interrupted: bool = False
    interrupt_payload: dict[str, Any] | None = None

    @property
    def answer(self) -> str:
        return self.state.get("final_answer") or ""


class AgentSession:
    """One agent, one session, one trajectory."""

    def __init__(
        self,
        settings: Settings,
        config: PipelineConfig,
        *,
        session_id: str | None = None,
        auto_approve: bool = False,
        prefer_recorded_search: bool = False,
        seed: int | None = None,
        task_id: str | None = None,
        store_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.session_id = session_id or new_session_id()

        self.registry = build_registry(
            settings,
            config,
            session_id=self.session_id,
            prefer_recorded_search=prefer_recorded_search,
        )
        self.provider = Provider(settings, config, seed=seed)

        record = TrajectoryRecord(
            session_id=self.session_id,
            task="",
            task_id=task_id,
            seed=seed,
            profile=config.name,
            capability_profile=self.registry.capability_profile,
            models={
                "agent": config.models.agent,
                "planner": config.models.planner,
                "compress": config.models.compress,
            },
            prompt_hashes=hash_prompts(PROMPT_DIR),
        )
        self.recorder = TrajectoryRecorder(
            record,
            redactor=Redactor.from_settings(settings),
            store_path=store_path or settings.resolved(str(DEFAULT_TRAJECTORY_STORE)),
        )

        workspace = settings.resolved(settings.workspace_root) / self.session_id
        self.context = AgentContext(
            settings=settings,
            config=config,
            registry=self.registry,
            provider=self.provider,
            recorder=self.recorder,
            workspace=workspace,
            auto_approve=auto_approve,
        )

        checkpoint_path = settings.resolved(settings.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        self.checkpointer = SqliteSaver(self._connection, serde=build_serialiser())
        self.graph = build_graph(self.context, checkpointer=self.checkpointer)

    @property
    def thread(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.session_id}}

    def run(self, task: str) -> RunOutcome:
        """Execute a task from the beginning."""
        self.recorder.record.task = task
        bind_run(self.session_id, task_id=self.recorder.record.task_id)
        log.info("run starting", capability_profile=self.registry.capability_profile)
        try:
            state = self.graph.invoke(
                initial_state(
                    task=task,
                    session_id=self.session_id,
                    capability_profile=self.registry.capability_profile,
                ),
                config=self.thread,
                # A hard stop below the graph's own ceilings. If routing ever
                # develops a cycle the guard cannot see, this ends the run
                # instead of letting it spend the day's quota discovering it.
                # Generous, because tripping it is a bug, not a budget.
                **{"recursion_limit": self.config.budget.max_steps * 8},
            )
            return self._finish(state)
        finally:
            clear_run()

    def resume(self, decision: Any) -> RunOutcome:
        """Continue a run suspended at an approval interrupt."""
        from langgraph.types import Command

        bind_run(self.session_id)
        try:
            state = self.graph.invoke(
                Command(resume=decision),
                config=self.thread,
                **{"recursion_limit": self.config.budget.max_steps * 8},
            )
            return self._finish(state)
        finally:
            clear_run()

    def _finish(self, state: AgentState) -> RunOutcome:
        pending = self._pending_interrupt()
        if pending is not None:
            # Suspended, not finished: the trajectory stays open so the
            # resumed half lands in the same record rather than a second one.
            return RunOutcome(
                session_id=self.session_id,
                state=state,
                record=self.recorder.record,
                interrupted=True,
                interrupt_payload=pending,
            )
        self.recorder.write()
        return RunOutcome(session_id=self.session_id, state=state, record=self.recorder.record)

    def _pending_interrupt(self) -> dict[str, Any] | None:
        snapshot = self.graph.get_state(self.thread)
        interrupts = getattr(snapshot, "interrupts", None) or ()
        for item in interrupts:
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
        return None

    def close(self) -> None:
        for tool in self.registry.tools:
            closer = getattr(getattr(tool, "sandbox", None), "close", None)
            if callable(closer):
                closer()
        self.provider.cache.close()
        self._connection.close()

    def __enter__(self) -> AgentSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
