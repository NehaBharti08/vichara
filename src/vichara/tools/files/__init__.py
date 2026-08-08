"""Scoped file access.

The only tool in the set with a real filesystem boundary, so containment lives
in its own module with its own adversarial tests.
"""

from vichara.tools.files.tool import (
    WorkspaceFileArgs,
    WorkspaceFileTool,
    build_workspace_file_tool,
)
from vichara.tools.files.workspace import (
    MAX_FILE_BYTES,
    MAX_FILES,
    FileInfo,
    SessionWorkspace,
    WorkspaceViolation,
)

__all__ = [
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "FileInfo",
    "SessionWorkspace",
    "WorkspaceFileArgs",
    "WorkspaceFileTool",
    "WorkspaceViolation",
    "build_workspace_file_tool",
]
