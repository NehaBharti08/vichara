"""Path containment.

These are the adversarial tests for the one tool with a real filesystem
boundary. Every case here is an escape attempt that must fail, and fail as a
:class:`WorkspaceViolation` rather than an unhandled OSError -- the agent needs
to be told it may not do that, not handed a traceback.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vichara.tools.errors import InvalidArguments
from vichara.tools.files.workspace import (
    MAX_FILE_BYTES,
    MAX_FILES,
    SessionWorkspace,
    WorkspaceViolation,
)


@pytest.fixture
def workspace(tmp_path: Path) -> SessionWorkspace:
    ws = SessionWorkspace(tmp_path / "sessions", "sess-test")
    ws.ensure()
    return ws


class TestTraversalIsBlocked:
    """The path cases an attacker or a confused model actually produces."""

    @pytest.mark.parametrize(
        "attack",
        [
            "../escape.txt",
            "../../etc/passwd",
            "notes/../../escape.txt",
            "./../../escape.txt",
            "..",
            "../",
            "a/../../b.txt",
        ],
    )
    def test_parent_references(self, workspace: SessionWorkspace, attack: str) -> None:
        with pytest.raises(WorkspaceViolation):
            workspace.resolve(attack)

    @pytest.mark.parametrize(
        "attack",
        [
            "/etc/passwd",
            "/tmp/evil.txt",
            "C:/Windows/System32/config/SAM",
            "C:\\Windows\\win.ini",
            "\\\\server\\share\\file.txt",
            "//server/share/file.txt",
            "D:relative.txt",
        ],
    )
    def test_absolute_and_unc_paths(self, workspace: SessionWorkspace, attack: str) -> None:
        """Both path flavours are rejected on every platform.

        'C:/x' is relative to PurePosixPath and absolute to PureWindowsPath.
        Checking only the host's flavour would let a Windows-style path through
        on the Linux container the Space actually runs.
        """
        with pytest.raises(WorkspaceViolation):
            workspace.resolve(attack)

    def test_backslash_separators_are_normalised_not_trusted(
        self, workspace: SessionWorkspace
    ) -> None:
        """`..\\..\\x` must not survive by being an unrecognised separator."""
        with pytest.raises(WorkspaceViolation):
            workspace.resolve("..\\..\\escape.txt")

    def test_null_byte(self, workspace: SessionWorkspace) -> None:
        with pytest.raises(WorkspaceViolation, match="null byte"):
            workspace.resolve("notes.md\x00.png")

    @pytest.mark.parametrize("name", ["CON", "con.txt", "NUL.md", "aux", "COM1.log", "LPT9"])
    def test_reserved_device_names(self, workspace: SessionWorkspace, name: str) -> None:
        """Windows resolves these to devices regardless of extension.

        Rejected on every platform so behaviour is identical between a
        developer's laptop and the container.
        """
        with pytest.raises(WorkspaceViolation, match="reserved"):
            workspace.resolve(name)

    def test_depth_limit(self, workspace: SessionWorkspace) -> None:
        with pytest.raises(WorkspaceViolation, match="levels deep"):
            workspace.resolve("a/b/c/d.txt")

    def test_overlong_component(self, workspace: SessionWorkspace) -> None:
        with pytest.raises(WorkspaceViolation, match="too long"):
            workspace.resolve("x" * 200 + ".md")

    @pytest.mark.parametrize("name", ["-rf.txt", ".hidden", "sp ace.txt", "semi;colon.txt"])
    def test_disallowed_characters(self, workspace: SessionWorkspace, name: str) -> None:
        with pytest.raises(WorkspaceViolation):
            workspace.resolve(name)

    def test_symlink_escape(self, workspace: SessionWorkspace, tmp_path: Path) -> None:
        """The check that actually matters: resolve() follows the link.

        A blocklist of '..' strings would pass this attack straight through --
        the path contains nothing suspicious at all.
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("stolen", encoding="utf-8")

        link = workspace.path / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this machine")

        with pytest.raises(WorkspaceViolation, match="outside the workspace"):
            workspace.resolve("link/secret.txt")

    @pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
    def test_junction_escape(self, workspace: SessionWorkspace, tmp_path: Path) -> None:
        """The Windows equivalent of the symlink attack.

        Worth covering separately because `mklink /J` needs no elevation, so
        this is the escape a non-administrator on Windows can actually create
        -- and the one the symlink test above skips on a default install.
        """
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("stolen", encoding="utf-8")

        junction = workspace.path / "jn"
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip("could not create a junction on this machine")

        with pytest.raises(WorkspaceViolation, match="outside the workspace"):
            workspace.resolve("jn/secret.txt")

    def test_empty_path_is_an_argument_error(self, workspace: SessionWorkspace) -> None:
        """Distinct from a violation: the model forgot an argument, it did not
        attempt anything forbidden, and the remediation differs."""
        with pytest.raises(InvalidArguments):
            workspace.resolve("   ")


class TestLegitimateUse:
    def test_simple_name(self, workspace: SessionWorkspace) -> None:
        resolved = workspace.resolve("notes.md")

        assert resolved.parent == workspace.path

    def test_one_subdirectory(self, workspace: SessionWorkspace) -> None:
        resolved = workspace.resolve("draft/summary.md")

        assert resolved.parent.name == "draft"
        assert workspace.path in resolved.parents

    def test_write_then_read(self, workspace: SessionWorkspace) -> None:
        written = workspace.write("notes.md", "photosynthesis notes")

        assert written == len(b"photosynthesis notes")
        assert workspace.read("notes.md") == "photosynthesis notes"

    def test_write_creates_subdirectory(self, workspace: SessionWorkspace) -> None:
        workspace.write("draft/summary.md", "text")

        assert (workspace.path / "draft" / "summary.md").is_file()

    def test_list_reports_relative_posix_paths(self, workspace: SessionWorkspace) -> None:
        workspace.write("a.md", "1")
        workspace.write("draft/b.md", "22")

        listing = {f.name: f.size_bytes for f in workspace.list()}

        assert listing == {"a.md": 1, "draft/b.md": 2}

    def test_list_is_empty_before_anything_is_written(self, tmp_path: Path) -> None:
        assert SessionWorkspace(tmp_path / "s", "sess-empty").list() == []

    def test_sessions_are_isolated(self, tmp_path: Path) -> None:
        """One session must not see another's files."""
        root = tmp_path / "sessions"
        first = SessionWorkspace(root, "sess-one")
        second = SessionWorkspace(root, "sess-two")
        first.write("private.md", "secret")

        assert second.list() == []
        with pytest.raises(InvalidArguments):
            second.read("private.md")


class TestResourceLimits:
    def test_oversized_write_refused(self, workspace: SessionWorkspace) -> None:
        with pytest.raises(WorkspaceViolation, match="write limit"):
            workspace.write("big.txt", "x" * (MAX_FILE_BYTES + 1))

    def test_file_count_capped(self, workspace: SessionWorkspace) -> None:
        for index in range(MAX_FILES):
            workspace.write(f"f{index}.txt", "x")

        with pytest.raises(WorkspaceViolation, match="already holds"):
            workspace.write("one-too-many.txt", "x")

    def test_overwriting_does_not_count_against_the_cap(self, workspace: SessionWorkspace) -> None:
        for index in range(MAX_FILES):
            workspace.write(f"f{index}.txt", "x")

        assert workspace.write("f0.txt", "replaced") == len(b"replaced")

    def test_reading_a_missing_file_is_an_argument_error(self, workspace: SessionWorkspace) -> None:
        with pytest.raises(InvalidArguments, match="No file named"):
            workspace.read("absent.md")


class TestSessionIdentifier:
    @pytest.mark.parametrize("bad", ["../other", "a/b", "", "a b", "..", "C:"])
    def test_unsafe_session_ids_rejected(self, tmp_path: Path, bad: str) -> None:
        """The session id is a path component too, and it is not always
        generated internally -- an API caller supplies it."""
        with pytest.raises(ValueError, match="unsafe session id"):
            SessionWorkspace(tmp_path, bad)
