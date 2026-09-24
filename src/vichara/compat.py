"""Standard-library gaps on older interpreters.

This project targets Python 3.11+, and everything here exists because one
deployment target does not offer it. Hugging Face will host a Gradio Space for
free only on ZeroGPU hardware -- creating one on ``cpu-basic`` returns 402
Payment Required -- and the ZeroGPU image is pinned to ``python:3.10.13``. The
``python_version`` field in a Space README does not move it.

The failures that led here are worth recording because they are invisible until
the app starts: the build succeeds, installs every dependency, and then every
launch dies on the first 3.11-only name it meets --

    AttributeError: module 'enum' has no attribute 'StrEnum'
    ImportError: cannot import name 'UTC' from 'datetime'

Keep this module empty of anything that is not a version shim.
"""

# ruff: noqa: UP017, UP036, UP042
#
# Every one of these rules is right about the declared target and wrong about
# this file, and UP017 is why the exemption is file-level rather than three
# inline ones: `ruff --fix` rewrote `timezone.utc` here into `datetime.UTC`,
# turning the shim into the exact ImportError it exists to prevent. The tests
# could not catch that, because they run on 3.11 where the rewrite is correct.
#
# pyproject still says requires-python >=3.11 and that stays true: 3.10 is not
# a supported development environment and CI does not test it. The claim here
# is narrower -- that importing this package under the one host offering free
# Gradio hosting does not raise before the app can start.

from __future__ import annotations

import datetime as _datetime
import enum
import sys

if sys.version_info >= (3, 11):  # pragma: no cover - depends on interpreter
    StrEnum = enum.StrEnum
else:  # pragma: no cover - depends on interpreter

    class StrEnum(str, enum.Enum):  # type: ignore[no-redef]
        """``enum.StrEnum`` for 3.10, matching the 3.11 semantics that matter.

        Mixing in ``str`` already gives equality and hashing against plain
        strings, which is what lets a member be used as a dict key or compared
        to a value parsed from YAML. The one thing it does not give is
        ``str(member)``: a plain ``(str, Enum)`` renders as
        ``ClassName.MEMBER``, where ``StrEnum`` renders as the value.

        That difference is not cosmetic here. Terminal reasons, tool names and
        error codes are interpolated into prompts, log lines and trajectory
        records, so the wrong one would change what the model reads and what
        gets written to disk -- quietly, and only on the deployed Space.
        """

        __str__ = str.__str__

        def __format__(self, format_spec: str) -> str:
            return str.__format__(self, format_spec)


UTC = _datetime.timezone.utc
"""``datetime.UTC`` under a name that exists on both interpreters.

3.11 added ``datetime.UTC`` as a plain alias -- ``datetime.UTC is
timezone.utc`` is true there -- so this is the same object rather than an
equivalent one. Timestamps written through it land in trajectory records, which
is why identity matters and not merely equality.

Reached through ``datetime as _datetime`` because importing ``timezone``
directly is what the autofixer latched onto the first time.
"""


__all__ = ["UTC", "StrEnum"]
