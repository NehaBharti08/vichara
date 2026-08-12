# Threat model

This document exists because "it's sandboxed" is a claim, and an unqualified
claim is worth less than a qualified one. The useful half is
[§4, What this does not stop](#4-what-this-does-not-stop).

Every statement about a defence here has a test behind it in
[`tests/sandbox/`](../tests/sandbox/). Every statement about a *gap* is
something that was tried and worked.

---

## 1. What is being defended, and against whom

The agent executes Python that a language model wrote. That model reads web
search results and retrieved documents, which are written by anyone. So the
realistic attacker is not a human with a debugger on this machine — it is
**text in a document that the agent retrieves and then acts on.**

That shapes the priorities. An attacker who already has code execution on the
host has won regardless. An attacker who can only influence what the model
*decides to run* is the one this design is built against.

Three assets, in order of what losing them costs:

| Asset | Why it matters | Where it lives |
|---|---|---|
| **API credentials** | `GOOGLE_API_KEY`, `TAVILY_API_KEY`. Stolen keys mean someone else's bill and someone else's traffic attributed to you. | The agent process environment and `.env` |
| **The host filesystem** | The repo, the user's home directory, other projects, SSH keys. | Outside the sandbox |
| **Network access from this machine** | Egress means exfiltration; a sandbox that can make requests is a proxy. | The host's interface |

The trust boundaries, and the fact that there are **two of them**, matter more
than any single control:

```
 ┌─ trusted ─────────────────┐   ┌─ UNTRUSTED ───────────────┐
 │ agent process             │   │ sandbox (Pyodide/Docker)  │
 │  · holds API keys         │   │  · runs model-authored    │
 │  · HAS NETWORK ACCESS     │──▶│    code                   │
 │  · reads/writes workspace │   │  · NO network             │
 │                           │   │  · NO host filesystem     │
 └───────────────────────────┘   └───────────────────────────┘
        ▲                                  
        │ untrusted text (search results, retrieved documents)
        │ ── this is the attack surface that actually matters ──
```

**The sandbox boundary and the agent boundary are different boundaries, and
the interesting attacks cross between them.** See §4.1.

---

## 2. The Pyodide backend (default)

CPython compiled to WebAssembly, hosted in a Node subprocess. Default for
local development, CI, and the deployed Space — a Hugging Face Space offers no
Docker-in-Docker, so this is the only backend that can run there.

### 2.1 What it stops, and how

| Threat | Mechanism | Test |
|---|---|---|
| Reading host files | No host filesystem is mounted. The virtual root is MEMFS: `['dev','home','lib','proc','tmp']` | `test_cannot_read_host_files`, `test_filesystem_root_is_virtual` |
| Reading the agent's `.env` | Same — the repo is not reachable at any path | `test_cannot_reach_the_repository` |
| Network egress via `socket` | `socket` is unimportable; separately, Pyodide's socket emulation moves no data even unblocked | `test_socket_is_not_importable` |
| Network egress via the JS bridge | Two layers: `jsglobals` is an empty null-prototype object, so `js` proxies nothing; and `js` is unimportable | `test_js_bridge_is_not_importable` |
| Network egress via `pyodide.http` | The whole `pyodide` root is blocked — `pyodide.http` exposes `fetch` directly | `test_pyodide_http_is_not_importable` |
| Reading host env vars | The worker is spawned with an allowlist environment containing no credentials | `test_worker_environment_carries_no_credentials` |
| Fork bombs | `subprocess`, `multiprocessing` unimportable; no `os.fork` | `test_fork_bomb_cannot_be_constructed` |
| Infinite loops | The parent kills the worker process | `test_infinite_loop_is_killed` |
| Unbounded output | Capped in the runner, per stream | `test_oversized_output_is_truncated` |
| Native escape via `ctypes` | Unimportable | `test_ctypes_is_blocked` |

### 2.2 Three findings worth writing down

**`import js` is live by default, and it reaches the host environment.**
Out of the box, Pyodide exposes a `js` module proxying JavaScript's
`globalThis`. `js.fetch` is present. `js.process.env` reads the *Node process's*
environment — which, if the worker inherited the agent's environment, is where
`GOOGLE_API_KEY` lives. This is why there are two independent defences rather
than one: `jsglobals: Object.create(null)` makes the bridge proxy nothing, and
the scrubbed subprocess environment means there is nothing behind it to read.
Either would probably hold. Together, the claim survives one of them being
wrong.

**Guarding `builtins.__import__` does not work.** The first version of the
import block replaced `builtins.__import__`. `importlib.import_module("js")`
walks straight past it — it calls `importlib._bootstrap._find_and_load`
directly. Only a `sys.meta_path` finder catches every route: `import`,
`__import__`, and `importlib`. Tested from all three
(`test_importlib_does_not_bypass_the_block`).

**`socket.connect()` succeeds and means nothing.** Pyodide's emulated socket
accepts a `connect()` to an arbitrary address and returns success. It is a
lie: `send`/`recv` then time out, because no data path exists. Worth knowing,
because a naive test that asserts on `connect()` failing would report a
vulnerability that is not there — and a naive attacker would believe they had
egress when they do not.

### 2.3 The warm worker's cost

One Node process is reused across executions, because Pyodide takes ~2s to
start and ~1ms per execution afterwards. That 2000× difference is the entire
reason this project ships no separate calculator tool.

The cost is that **executions share one interpreter**. The adversarial suite
found this by accident: a test ran `sys.meta_path.clear()`, and every later
test in that worker got a different, weaker failure. No access was granted —
the blocked modules are also gone from `sys.modules`, so with no finder left
nothing can import them at all, and disarming the defence disarms importing
itself. But a defence one submission can switch off for the next is not a
defence, so hardening is now re-applied before **every** execution
(`test_one_submission_cannot_weaken_the_next`).

Re-hardening fixes the import path specifically. It does not make executions
fully independent: the in-memory filesystem persists for the worker's life,
and so does anything written to `builtins`. **If per-execution isolation
matters more than latency, use the Docker backend**, which creates and
destroys a container per call.

---

## 3. The Docker backend

Stronger isolation, not the default. It needs a daemon, which the deployed
Space does not have.

Every flag is doing something:

| Flag | Threat |
|---|---|
| `--network=none` | No interface exists, rather than being filtered |
| `--read-only` | Immutable root filesystem |
| `--tmpfs /tmp:size=16m,noexec,nosuid` | The one writable path; capped, non-executable, wiped on exit |
| `--cap-drop=ALL` | No capabilities, including Docker's default set |
| `--security-opt=no-new-privileges` | setuid binaries cannot regain privilege |
| `--pids-limit=64` | A fork bomb hits a wall instead of the host |
| `--memory` **and** `--memory-swap` equal | Without equal swap, a process evades the memory ceiling by swapping |
| `--cpus` | A busy loop cannot starve the host |
| `--user=65534:65534` | Runs as `nobody` |

**The important asymmetry**: here, "no network" is a *configuration* — one
missing flag from being false. In Pyodide it is closer to a property of the
runtime. That is why the Pyodide claim is stated more confidently even though
Docker is the stronger sandbox overall. A control you can forget to apply is a
weaker guarantee than one you would have to build to defeat.

Both backends satisfy one conformance suite (`tests/sandbox/test_protocol.py`),
so the code tool genuinely cannot tell them apart. Docker cases skip where no
daemon exists and run in CI.

---

## 4. What this does not stop

The section that matters.

### 4.1 The sandbox has no network. The agent does.

This is the most important limitation in the document, and it is structural
rather than a bug.

The sandbox cannot make a request. But the **agent** holds `web_search`, which
exists to make requests. So a prompt injection inside a retrieved document
does not need to escape the sandbox at all — it only needs to persuade the
model to put data into a search query:

```
retrieved document  ──▶  model reads it as instruction
                    ──▶  model calls web_search("... <data> ...")
                    ──▶  data leaves the machine through a legitimate tool
```

No sandbox boundary is crossed. Every control in §2 and §3 holds perfectly
while this succeeds. **Sandboxing code execution does nothing about it**,
because code execution was never involved.

This is Phase 5's problem, not Phase 2's, and it is why the prompt-injection
work is treated as the more serious security effort of the two. Sandboxing is
the easy half.

### 4.2 Prompt injection generally

Out of scope for this document. A poisoned document that persuades the agent
to write a wrong answer, cite a fabricated source, or call a legitimate tool
against the user's interest is unaffected by anything here. See
`docs/PROMPT_INJECTION.md` (Phase 5) for the measured attack success rate.

### 4.3 Denial of service within the limits

An execution may consume its full CPU and memory allowance every time. The
limits bound one call, not a sequence of them. The agent-level guardrails
(`max_steps`, per-tool call caps) bound the sequence — but those are budget
controls, not security controls, and an attacker who can drive the agent can
make it burn its quota.

### 4.4 Side channels

Timing, cache and memory-pressure side channels are not addressed by either
backend. WebAssembly does not defend against them and neither does a
container. Out of scope for a single-user study agent; it would not be for a
multi-tenant service.

### 4.5 Supply chain

Pyodide, its bundled wheels, the base image, and every Python dependency are
trusted. `npm install` and `uv sync` fetch code that then runs with the
agent's privileges. Lockfiles pin versions and CI runs `gitleaks`, but neither
verifies that a pinned artifact is benign. A compromised upstream defeats
everything in this document.

### 4.6 Persistent state within a warm worker

Covered in §2.3. Import hardening is re-applied per execution; the in-memory
filesystem and `builtins` are not reset. One task can leave data another task
can read, within a single session. Use the Docker backend where that matters.

### 4.7 A TOCTOU race in workspace path resolution

`SessionWorkspace.resolve()` resolves a path, checks containment, and later
opens it. On Windows there is no `O_NOFOLLOW` equivalent, so a directory
component replaced with a symlink between check and open would not be caught.
Defeating it needs handle-based reopening, which is out of proportion here —
the attacker would already need to be running code on the host.

### 4.8 Resource limits are approximate

`--cpus` is a share, not a hard CPU-seconds cap. Pyodide's memory ceiling is
whatever the WebAssembly heap permits, not `memory_mb` — that field is
enforced by Docker and is advisory under Pyodide. The wall clock is the limit
that actually holds in both.

### 4.9 The per-tool timeout bounds waiting, not work

`BaseTool` enforces its timeout with a daemon thread, and Python cannot kill a
thread. The agent stops waiting; the work keeps running until it finishes on
its own. Sufficient for network-bound tools, whose sockets carry their own
deadlines. Not sufficient for arbitrary code — which is exactly why the
sandbox enforces its wall clock by killing a process instead.

---

## 5. Residual risk

| Risk | Severity | Status |
|---|---|---|
| Exfiltration via the agent's own `web_search` | **High** | Open. Phase 5. |
| Prompt injection changing the answer | **High** | Open, unmeasured until Phase 5. |
| Supply chain compromise | Medium | Accepted. Pinned, not verified. |
| Cross-execution state in a warm worker | Low | Partially mitigated; Docker backend avoids it. |
| Side channels | Low | Accepted, out of scope. |
| Workspace TOCTOU on Windows | Low | Accepted. Requires host code execution already. |

**If you take one thing from this document**: the sandbox is the part of this
system that is easy to get right, and it is not where the risk is. The risk is
that a language model reads an untrusted document and treats it as an
instruction, and no amount of WebAssembly fixes that.

---

## Reproducing

```bash
npm install
uv run pytest tests/sandbox -q                    # Pyodide, ~35 cases
SANDBOX_BACKEND=docker uv run pytest tests/sandbox -q   # Docker, needs a daemon
```
