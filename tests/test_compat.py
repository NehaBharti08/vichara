"""The 3.10 StrEnum shim must be indistinguishable from the real one.

Hugging Face hosts a free Gradio Space only on ZeroGPU, whose image is pinned
to python:3.10.13, where `enum.StrEnum` does not exist. The shim exists for
that one host -- but its members are interpolated into prompts, log lines and
trajectory records, so a rendering difference would change what the model reads
and what gets written to disk, on the deployed Space only, silently.

These tests run the fallback explicitly on every interpreter rather than only
on 3.10, because the environment that would catch a regression is the one no
developer runs.
"""

from __future__ import annotations

import enum

from vichara.compat import StrEnum
from vichara.trajectory.schema import StepKind, TerminalReason


class Fallback(str, enum.Enum):  # noqa: UP042 - the 3.10 shape, on purpose
    """The shim's definition, constructed directly so it is always exercised."""

    __str__ = str.__str__

    def __format__(self, format_spec: str) -> str:
        return str.__format__(self, format_spec)

    ANSWERED = "answered"
    LOOP = "loop_detected"


class TestShimMatchesStrEnum:
    def test_str_renders_the_value_not_the_member(self) -> None:
        """A plain (str, Enum) renders as 'Fallback.ANSWERED'. That is the bug."""
        assert str(Fallback.ANSWERED) == "answered"

    def test_fstring_renders_the_value(self) -> None:
        """Prompts and log lines interpolate these."""
        assert f"{Fallback.ANSWERED}" == "answered"

    def test_format_renders_the_value(self) -> None:
        assert format(Fallback.ANSWERED) == "answered"
        assert f"{Fallback.ANSWERED:>10}" == "  answered"

    def test_equality_with_a_plain_string(self) -> None:
        assert Fallback.ANSWERED == "answered"

    def test_usable_as_a_dict_key_interchangeably_with_its_value(self) -> None:
        """Config parsed from YAML arrives as plain strings."""
        assert {Fallback.LOOP: 1}["loop_detected"] == 1

    def test_value_is_unchanged(self) -> None:
        assert Fallback.LOOP.value == "loop_detected"

    def test_json_serialises_as_the_value(self) -> None:
        """Trajectory records are written as JSON."""
        import json

        assert json.dumps({"terminal": Fallback.LOOP}) == '{"terminal": "loop_detected"}'


class TestRealEnumsStillBehave:
    """The swap touched eight modules; these are the values that reach disk."""

    def test_terminal_reason_renders_as_its_value(self) -> None:
        assert f"{TerminalReason.LOOP_DETECTED}" == "loop_detected"

    def test_step_kind_renders_as_its_value(self) -> None:
        assert f"{StepKind.SYNTHESIZE}" == "synthesize"

    def test_terminal_reason_compares_to_a_plain_string(self) -> None:
        assert TerminalReason.ANSWERED == "answered"

    def test_the_export_is_a_str_subclass(self) -> None:
        """Whatever StrEnum resolves to, members must still be strings."""

        class Probe(StrEnum):
            VALUE = "v"

        assert isinstance(Probe.VALUE, str)


class TestShimDoesNotUseWhatItShims:
    """The shim was once autofixed into the bug it exists to prevent.

    `ruff --fix` rewrote `timezone.utc` in compat.py to `datetime.UTC`, because
    pyproject declares requires-python >=3.11 and the rewrite is correct there.
    Every test passed. The Space then failed to start with the exact ImportError
    the module was written to avoid.

    Nothing that runs on 3.11 can catch that by executing the code, so this
    reads the source instead.
    """

    def source(self) -> str:
        from pathlib import Path

        import vichara.compat

        return Path(vichara.compat.__file__).read_text(encoding="utf-8")

    def test_it_does_not_import_utc_from_datetime(self) -> None:
        assert "from datetime import UTC" not in self.source()

    def test_it_does_not_reference_datetime_utc(self) -> None:
        """`datetime.UTC` is the 3.11 name; the shim must reach for timezone.utc."""
        assert "_datetime.UTC" not in self.source()

    def test_it_still_defines_utc_from_timezone(self) -> None:
        assert "timezone.utc" in self.source()

    def test_the_autofix_exemption_is_still_in_place(self) -> None:
        """Removing the noqa re-arms the autofixer against this file."""
        assert "UP017" in self.source()
