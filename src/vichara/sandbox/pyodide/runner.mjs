// Node host for the Pyodide sandbox.
//
// Speaks newline-delimited JSON on stdin/stdout so the Python side can keep
// one warm worker per session: Pyodide costs 1-2s to initialise and ~50ms per
// execution after that, which is the entire reason a separate calculator tool
// was never needed.
//
// Two independent hardening layers, both of which the probe work in Phase 2
// showed to be necessary:
//
//   1. `jsglobals` is an empty null-prototype object, so the `js` module that
//      Pyodide always provides proxies nothing. Without this, `js.fetch` is
//      live and `js.process.env` reads the *host* environment -- which is the
//      environment holding the agent's API keys.
//   2. A `sys.meta_path` finder refuses the dangerous modules by name. A guard
//      on `builtins.__import__` alone is not enough: `importlib.import_module`
//      bypasses it entirely. Clearing `sys.meta_path` to escape the guard does
//      not help either, because the modules are also gone from `sys.modules`,
//      so removing the finder leaves nothing able to import them at all.
//
// The Python side additionally spawns this process with a scrubbed environment.
// Layer 1 makes the host env unreachable; scrubbing makes it absent. Neither
// alone is enough to state the claim confidently.
//
// What this file cannot do is enforce the wall clock. A synchronous Python
// loop inside WebAssembly never yields to the JavaScript event loop, so no
// timer here can ever fire to interrupt it. Only the parent killing this
// process stops `while True: pass`. See docs/THREAT_MODEL.md.

import { loadPyodide } from "pyodide";
import { createInterface } from "node:readline";

const BLOCKED_MODULES = [
  "js",
  "pyodide_js",
  "socket",
  "_socket",
  "ssl",
  "subprocess",
  "ctypes",
  "_ctypes",
  "multiprocessing",
  "webbrowser",
  "http",
  "urllib",
  "ftplib",
  "telnetlib",
  "smtplib",
  "xmlrpc",
  "asyncio",
];

// `pyodide` itself is blocked as a root: pyodide.http exposes fetch, and
// pyodide.ffi hands back the JS bridge that layer 1 was built to neuter.
const BLOCKED_ROOTS = ["pyodide"];

// Hardening is a function, not a one-off, and it is re-applied before every
// execution. The warm worker shares one interpreter across calls, so anything
// a submission does to global state persists into the next one -- the
// adversarial suite found this by running `sys.meta_path.clear()` in one test
// and watching later tests get a different (still blocked, but weaker) error.
//
// Clearing the finder is not an escape, because the blocked modules are also
// gone from sys.modules and with no finder left nothing can import them at
// all. But it degrades the sandbox for every later call in the session, and a
// defence that one submission can switch off for the next is not a defence.
// Re-asserting costs microseconds. See docs/THREAT_MODEL.md for what this
// still does not fix.
const HARDEN_PY = `
import sys, os, importlib.abc

_BLOCKED = set(${JSON.stringify(BLOCKED_MODULES)})
_BLOCKED_ROOTS = set(${JSON.stringify(BLOCKED_ROOTS)})


class _SandboxImportBlocker(importlib.abc.MetaPathFinder):
    """Refuse dangerous imports however they are requested.

    Raising from find_spec rather than returning None makes the refusal
    explicit and identical for import, __import__ and importlib.import_module.
    """

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if fullname in _BLOCKED or root in _BLOCKED or root in _BLOCKED_ROOTS:
            raise ImportError(
                f"{fullname!r} is not available in this sandbox. "
                "There is no network and no host access here; use only "
                "computation over values you already have."
            )
        return None


def __vichara_harden():
    """Idempotent. Called before every execution, not once at startup."""
    if not any(isinstance(f, _SandboxImportBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _SandboxImportBlocker())

    for name in list(sys.modules):
        root = name.split(".")[0]
        if name in _BLOCKED or root in _BLOCKED or root in _BLOCKED_ROOTS:
            del sys.modules[name]

    # Pyodide synthesises os.environ rather than inheriting the host's, but it
    # leaks the host invocation path through '_'. Harmless on its own; it still
    # tells an attacker where the agent lives, so it goes.
    os.environ.pop("_", None)


__vichara_harden()
`;

const PREAMBLE_PY = `
import io, sys, traceback, json

def __vichara_run(src):
    """Execute one submission, capturing output and any traceback.

    The last statement is evaluated as an expression when possible, so a bare
    trailing value behaves the way it does in a notebook.
    """
    __vichara_harden()
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    payload = {"stdout": "", "stderr": "", "exception": None, "result_repr": None}
    scope = {"__name__": "__main__"}
    try:
        try:
            import ast
            tree = ast.parse(src, mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
                exec(compile(tree, "<agent>", "exec"), scope)
                value = eval(compile(last, "<agent>", "eval"), scope)
                if value is not None:
                    payload["result_repr"] = repr(value)
            else:
                exec(compile(tree, "<agent>", "exec"), scope)
        except BaseException:
            payload["exception"] = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    payload["stdout"] = out.getvalue()
    payload["stderr"] = err.getvalue()
    return json.dumps(payload)
`;

function send(message) {
  process.stdout.write(JSON.stringify(message) + "\n");
}

let pyodide;
try {
  pyodide = await loadPyodide({
    // Layer 1. An empty null-prototype object: `import js` still succeeds,
    // because Pyodide always registers the module, but it proxies nothing.
    jsglobals: Object.create(null),
    stdout: () => {},
    stderr: () => {},
  });
  pyodide.runPython(HARDEN_PY);
  pyodide.runPython(PREAMBLE_PY);
} catch (error) {
  send({ ready: false, error: String(error && error.message ? error.message : error) });
  process.exit(1);
}

send({ ready: true, version: pyodide.version });

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });

for await (const line of lines) {
  const trimmed = line.trim();
  if (!trimmed) continue;

  let request;
  try {
    request = JSON.parse(trimmed);
  } catch {
    send({ id: null, outcome: "backend_error", detail: "unparseable request" });
    continue;
  }

  const started = performance.now();
  try {
    // Load any bundled packages the code imports (numpy, pandas, sympy...).
    // These come from node_modules on disk, so this needs no network -- which
    // is what makes preloading unnecessary and keeps cold start low.
    try {
      await pyodide.loadPackagesFromImports(request.code);
    } catch {
      // An unavailable package should surface as the ImportError the agent
      // can read and work around, not as a backend failure.
    }

    pyodide.globals.set("__vichara_src", request.code);
    const raw = pyodide.runPython("__vichara_run(__vichara_src)");
    const payload = JSON.parse(raw);

    const cap = request.max_output_bytes ?? 65536;
    let truncated = false;
    for (const stream of ["stdout", "stderr"]) {
      const text = payload[stream] ?? "";
      if (Buffer.byteLength(text, "utf8") > cap) {
        payload[stream] = Buffer.from(text, "utf8").subarray(0, cap).toString("utf8");
        truncated = true;
      }
    }

    send({
      id: request.id,
      outcome: truncated ? "output_limit" : "ok",
      stdout: payload.stdout,
      stderr: payload.stderr,
      exception: payload.exception,
      result_repr: payload.result_repr,
      truncated,
      duration_ms: performance.now() - started,
    });
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    // Emscripten reports heap exhaustion this way; it is the agent's problem
    // (allocate less) rather than a broken sandbox, so it gets its own outcome.
    const isMemory = /memory|allocation failed|out of bounds/i.test(message);
    send({
      id: request.id,
      outcome: isMemory ? "memory" : "backend_error",
      stdout: "",
      stderr: "",
      exception: null,
      truncated: false,
      duration_ms: performance.now() - started,
      detail: message.slice(0, 500),
    });
  }
}
