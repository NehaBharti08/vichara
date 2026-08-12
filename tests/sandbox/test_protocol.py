"""One conformance suite, run against every backend.

The Sandbox protocol only means something if the code tool genuinely cannot
tell which implementation it got. These cases are written once and
parametrised over the backends available on the machine: Pyodide everywhere,
Docker where a daemon exists (CI, not a dev laptop).

Anything backend-specific belongs in test_adversarial.py, not here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from vichara.sandbox.base import Limits, Outcome, Sandbox, SandboxResult
from vichara.sandbox.docker.backend import DockerSandbox
from vichara.sandbox.pyodide.backend import PyodideSandbox

LIMITS = Limits(wall_clock_s=30.0, max_output_bytes=4096)


@pytest.fixture(scope="module", params=["pyodide", "docker"])
def backend(request: pytest.FixtureRequest) -> Iterator[Sandbox]:
    box: Sandbox = PyodideSandbox() if request.param == "pyodide" else DockerSandbox()
    healthy, detail = box.health()
    if not healthy:
        pytest.skip(f"{request.param} unavailable: {detail}")
    yield box
    box.close()


class TestProtocolConformance:
    def test_satisfies_the_protocol(self, backend: Sandbox) -> None:
        assert isinstance(backend, Sandbox)

    def test_returns_the_final_expression(self, backend: Sandbox) -> None:
        result = backend.execute("17 * 23", LIMITS)

        assert result.result_repr == "391"
        assert result.outcome is Outcome.OK

    def test_captures_stdout(self, backend: Sandbox) -> None:
        result = backend.execute("print('hello')", LIMITS)

        assert result.stdout.strip() == "hello"

    def test_statements_without_a_trailing_expression(self, backend: Sandbox) -> None:
        result = backend.execute("x = 5\ny = x * 2\nprint(y)", LIMITS)

        assert result.stdout.strip() == "10"
        assert result.result_repr is None

    def test_raising_is_not_a_sandbox_failure(self, backend: Sandbox) -> None:
        """A traceback is the feedback the agent needs, not an error state."""
        result = backend.execute("1/0", LIMITS)

        assert result.outcome is Outcome.OK
        assert result.ok is False
        assert "ZeroDivisionError" in (result.exception or "")

    def test_infinite_loop_times_out(self, backend: Sandbox) -> None:
        result = backend.execute("while True: pass", Limits(wall_clock_s=4.0))

        assert result.outcome is Outcome.TIMEOUT

    def test_survives_a_timeout(self, backend: Sandbox) -> None:
        backend.execute("while True: pass", Limits(wall_clock_s=3.0))

        result = backend.execute("1 + 1", LIMITS)

        assert result.result_repr == "2"

    def test_output_is_capped(self, backend: Sandbox) -> None:
        result = backend.execute(
            "print('x' * 100000)", Limits(wall_clock_s=30.0, max_output_bytes=1024)
        )

        assert result.truncated is True
        assert len(result.stdout.encode()) <= 1024

    def test_no_network(self, backend: Sandbox) -> None:
        """The claim both backends must satisfy, by whatever mechanism."""
        result = backend.execute(
            "import socket\n"
            "s = socket.socket(); s.settimeout(3)\n"
            "s.connect(('1.1.1.1', 80))\n"
            "s.send(b'GET / HTTP/1.0\\r\\n\\r\\n')\n"
            "print('TRANSFERRED', s.recv(16))\n",
            Limits(wall_clock_s=20.0),
        )

        assert "TRANSFERRED" not in result.stdout

    def test_cannot_read_host_files(self, backend: Sandbox) -> None:
        result = backend.execute("print(open('/etc/shadow').read())", LIMITS)

        assert result.ok is False or "root:" not in result.stdout

    def test_no_agent_credentials_in_scope(self, backend: Sandbox) -> None:
        result = backend.execute(
            "import os; print(os.environ.get('GOOGLE_API_KEY', 'ABSENT'))", LIMITS
        )

        assert "ABSENT" in result.stdout

    def test_execute_never_raises(self, backend: Sandbox) -> None:
        for code in ("", "   ", "!!! not python", "\x00", "raise SystemExit(1)"):
            result = backend.execute(code, LIMITS)

            assert isinstance(result, SandboxResult)

    def test_health_never_raises(self, backend: Sandbox) -> None:
        healthy, detail = backend.health()

        assert isinstance(healthy, bool)
        assert isinstance(detail, str)

    def test_close_is_idempotent(self, backend: Sandbox) -> None:
        backend.close()
        backend.close()

        assert backend.execute("1", LIMITS).result_repr == "1"
