"""The workspace file tool.

One tool with an ``operation`` argument rather than three separate tools. That
is a deliberate trade: three tools would give the model clearer affordances,
but they would also triple the surface the approval interrupt has to reason
about and split the per-tool call budget three ways. One tool with a typed
operation keeps the risk classification in one place.

``delete`` is not offered at all. A study agent has no need to destroy its own
notes, and the cheapest way to be sure a destructive operation is safe is not
to implement it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vichara.settings import Settings
from vichara.tools.base import BaseTool, Citation, HealthStatus, ToolResult
from vichara.tools.config import OutputTrust, RiskClass
from vichara.tools.files.workspace import SessionWorkspace

SUMMARY = (
    "Read, write and list files in this session's private workspace. Use it to "
    "save notes or a draft the user can keep. Paths are relative to the workspace "
    "and cannot reach outside it."
)


class WorkspaceFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["read", "write", "list"] = Field(description="What to do.")
    path: str | None = Field(
        default=None,
        max_length=200,
        description="Relative filename, e.g. 'notes.md'. Required for read and write.",
    )
    content: str | None = Field(
        default=None, description="Text to write. Required for write; ignored otherwise."
    )

    @model_validator(mode="after")
    def _check_operands(self) -> WorkspaceFileArgs:
        """Catch a malformed call here so the model gets one clear message.

        Without this the failure surfaces from inside the operation as
        something vaguer, and the agent spends a step guessing what was wrong.
        """
        if self.operation in ("read", "write") and not self.path:
            raise ValueError(f"'path' is required for operation={self.operation!r}")
        if self.operation == "write" and self.content is None:
            raise ValueError("'content' is required for operation='write'")
        return self


class WorkspaceFileTool(BaseTool):
    """Scoped file access for one session."""

    name = "workspace_file"
    summary = SUMMARY
    args_schema = WorkspaceFileArgs
    risk = RiskClass.DESTRUCTIVE
    """Writes are irreversible from the agent's side, so they route through the
    approval interrupt. Reads are not destructive, but the risk class is a
    property of the tool rather than the call -- splitting it per-operation
    would put the approval decision inside the tool, where the graph cannot
    see or record it."""

    output_trust = OutputTrust.UNTRUSTED
    """A file the agent wrote earlier may contain text it copied out of a
    poisoned search result. Reading it back does not launder that."""

    def __init__(
        self,
        workspace: SessionWorkspace,
        *,
        timeout_s: float = 10.0,
        max_retries: int = 0,
        max_output_bytes: int = 16_384,
    ) -> None:
        # No retries: every failure this tool produces is a permission refusal
        # or a missing file, and neither becomes true on a second attempt.
        super().__init__(
            timeout_s=timeout_s, max_retries=max_retries, max_output_bytes=max_output_bytes
        )
        self.workspace = workspace

    @property
    def backend_name(self) -> str:
        return "local"

    def health(self) -> HealthStatus:
        try:
            self.workspace.ensure()
        except OSError as exc:
            return HealthStatus(healthy=False, backend="local", detail=str(exc))
        return HealthStatus(healthy=True, backend="local", detail=str(self.workspace.path))

    def _execute(self, **kwargs: Any) -> ToolResult:
        args = WorkspaceFileArgs.model_validate(kwargs)

        if args.operation == "list":
            files = self.workspace.list()
            listing = json.dumps([{"path": f.name, "bytes": f.size_bytes} for f in files])
            return self._ok(
                listing if files else "The workspace is empty.",
            )

        if args.operation == "read":
            text = self.workspace.read(args.path or "")
            return self._ok(
                text,
                citations=[
                    Citation(kind="file", source=f"workspace:{args.path}", locator=args.path)
                ],
            )

        written = self.workspace.write(args.path or "", args.content or "")
        return self._ok(f"Wrote {written} bytes to {args.path!r} in the workspace.")

    def _ok(self, content: str, *, citations: list[Citation] | None = None) -> ToolResult:
        return ToolResult(
            tool=self.name,
            ok=True,
            content=content,
            trust=self.output_trust,
            backend=self.backend_name,
            citations=citations or [],
        )


def build_workspace_file_tool(
    settings: Settings,
    session_id: str,
    *,
    timeout_s: float = 10.0,
    max_output_bytes: int = 16_384,
) -> WorkspaceFileTool:
    workspace = SessionWorkspace(settings.resolved(settings.workspace_root), session_id)
    return WorkspaceFileTool(workspace, timeout_s=timeout_s, max_output_bytes=max_output_bytes)
