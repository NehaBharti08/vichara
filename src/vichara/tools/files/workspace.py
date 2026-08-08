"""Per-session file workspace with path containment.

Every path the agent supplies is untrusted. It may be a hallucination, or it
may be an instruction that arrived inside a poisoned search result, and the
two are indistinguishable from here. So containment is enforced structurally
rather than by pattern-matching for ``..``.

**The actual boundary is one check**: resolve the candidate path fully --
following symlinks, normalising ``..``, expanding short names -- and verify the
result is inside the workspace. Everything else in this module is defence in
depth and better error messages. A blocklist of dangerous-looking strings is
not a security control; ``resolve()`` plus ``is_relative_to()`` is.

Resolution happens with ``strict=False`` so a file that does not exist yet
still gets normalised, which is what makes the check work for writes.

Known limitation, stated rather than hidden: on Windows this does not defend
against a race in which a directory component is replaced with a symlink
between the check and the open. Defeating that needs handle-based reopening
(``O_NOFOLLOW`` has no true Windows equivalent), which is out of proportion to
a single-user study agent. See docs/THREAT_MODEL.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from vichara.logging import get_logger
from vichara.tools.errors import InvalidArguments, PolicyViolation

log = get_logger(__name__)

MAX_FILENAME_LEN = 120
MAX_DEPTH = 2
"""Components allowed below the workspace root. One subdirectory, one file."""

MAX_FILE_BYTES = 256 * 1024
MAX_FILES = 50

# Windows resolves these to devices regardless of extension or directory, so
# `notes/CON.txt` is not a file. Rejected on every platform to keep behaviour
# identical between a developer's laptop and the Linux container.
_RESERVED = frozenset(
    """CON PRN AUX NUL
    COM1 COM2 COM3 COM4 COM5 COM6 COM7 COM8 COM9
    LPT1 LPT2 LPT3 LPT4 LPT5 LPT6 LPT7 LPT8 LPT9""".split()
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class FileInfo:
    """One file in the workspace."""

    name: str
    size_bytes: int


class WorkspaceViolation(PolicyViolation):
    """A path that would leave the workspace, or is otherwise not allowed."""


def _reject(message: str, *, detail: str) -> WorkspaceViolation:
    return WorkspaceViolation(
        message,
        remediation=(
            "Use a simple relative name inside the workspace, such as 'notes.md' or "
            "'draft/summary.md'. You cannot reach files outside it, and retrying with "
            "a different path spelling will not change that."
        ),
        detail=detail,
    )


class SessionWorkspace:
    """A directory the agent may read and write, and nothing else."""

    def __init__(self, root: Path, session_id: str) -> None:
        if not _SAFE_NAME.match(session_id):
            raise ValueError(f"unsafe session id: {session_id!r}")
        self.root = root.resolve()
        self.session_id = session_id
        self.path = (self.root / session_id).resolve()

    def ensure(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    # -- The security boundary ---------------------------------------------

    def resolve(self, relative: str) -> Path:
        """Map an agent-supplied name to a real path, or refuse.

        Raises:
            WorkspaceViolation: the path escapes the workspace or is disallowed.
            InvalidArguments: the name is empty or malformed.
        """
        if not relative or not relative.strip():
            raise InvalidArguments(
                "No filename was given.",
                remediation="Call the tool again with a 'path' such as 'notes.md'.",
            )

        # Normalise first: composed and decomposed Unicode forms of the same
        # name resolve to the same file on macOS, so comparing raw strings
        # would let two spellings of one name look like two different files.
        candidate = unicodedata.normalize("NFC", relative.strip()).replace("\\", "/")

        if "\x00" in candidate:
            raise _reject("The filename contains a null byte.", detail=repr(relative))

        # Reject anything absolute or drive/UNC qualified before parsing. Both
        # path flavours are consulted because "C:/x" is relative to PurePosixPath
        # and absolute to PureWindowsPath -- checking only the host's flavour
        # would let a Windows-style path through on Linux.
        if (
            PurePosixPath(candidate).is_absolute()
            or PureWindowsPath(candidate).is_absolute()
            or PureWindowsPath(candidate).drive
            or candidate.startswith("//")
        ):
            raise _reject("Absolute paths are not allowed.", detail=repr(relative))

        parts = [p for p in PurePosixPath(candidate).parts if p not in (".", "")]
        if not parts:
            raise _reject("The path does not name a file.", detail=repr(relative))
        if any(p == ".." for p in parts):
            raise _reject("Parent-directory references are not allowed.", detail=repr(relative))
        if len(parts) > MAX_DEPTH:
            raise _reject(f"Paths may be at most {MAX_DEPTH} levels deep.", detail=repr(relative))

        for part in parts:
            if len(part) > MAX_FILENAME_LEN:
                raise _reject("A path component is too long.", detail=repr(relative))
            if PurePosixPath(part).stem.upper() in _RESERVED:
                raise _reject(f"{part!r} is a reserved device name.", detail=repr(relative))
            if not _SAFE_NAME.match(part):
                raise _reject(
                    f"{part!r} contains characters that are not allowed.",
                    detail=repr(relative),
                )

        resolved = (self.path / PurePosixPath(*parts)).resolve()

        # The check that actually matters. Everything above is hygiene.
        if resolved != self.path and self.path not in resolved.parents:
            log.warning(
                "workspace escape blocked",
                session_id=self.session_id,
                requested=relative,
                resolved=str(resolved),
            )
            raise _reject("That path is outside the workspace.", detail=str(resolved))

        return resolved

    # -- Operations ---------------------------------------------------------

    def read(self, relative: str) -> str:
        target = self.resolve(relative)
        if not target.is_file():
            raise InvalidArguments(
                f"No file named {relative!r} exists in the workspace.",
                remediation="List the workspace to see what is there before reading.",
            )
        if target.stat().st_size > MAX_FILE_BYTES:
            raise _reject(
                f"{relative!r} is larger than the {MAX_FILE_BYTES // 1024}KB read limit.",
                detail=str(target),
            )
        return target.read_text(encoding="utf-8", errors="replace")

    def write(self, relative: str, content: str) -> int:
        target = self.resolve(relative)
        encoded = content.encode("utf-8")

        if len(encoded) > MAX_FILE_BYTES:
            raise _reject(
                f"Content is larger than the {MAX_FILE_BYTES // 1024}KB write limit.",
                detail=f"{len(encoded)} bytes",
            )

        self.ensure()
        if not target.exists() and len(self.list()) >= MAX_FILES:
            raise _reject(
                f"The workspace already holds {MAX_FILES} files.",
                detail=str(target),
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
        log.info("workspace write", session_id=self.session_id, path=relative, bytes=len(encoded))
        return len(encoded)

    def list(self) -> list[FileInfo]:
        if not self.path.exists():
            return []
        found = [
            FileInfo(
                name=str(item.relative_to(self.path)).replace("\\", "/"),
                size_bytes=item.stat().st_size,
            )
            for item in sorted(self.path.rglob("*"))
            if item.is_file()
        ]
        return found
