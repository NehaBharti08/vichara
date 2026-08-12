# Vichara

[![CI](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml/badge.svg)](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml)

A study agent that plans a multi-step approach to an academic question, calls tools, and synthesises a cited answer — with hard ceilings on steps and spend, and a UI that exposes its whole reasoning trajectory.

**The agent loop is the least interesting part of this repository.** A LangGraph ReAct loop is two days of work and thousands of identical ones exist. What this repo is actually for:

- **Trajectory evaluation that is programmatic and distributional.** Six of seven metrics computed mechanically against human-annotated gold trajectories written *before the agent ever ran* — tool-selection precision, step efficiency against an annotated optimal path, recovery rate under injected faults, refusal correctness gated on step count. Every task runs five times; the report shows distributions, not a trajectory that worked once.
- **A prompt-injection attack corpus targeting tool output**, with a published baseline attack success rate, a published post-defence rate, and a section listing the attacks that still work.
- **A threat model whose useful half is the limitations section** — [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

> **Status: Phase 2 of 7 complete.** Foundation, tool layer, and sandbox are built and tested. **There are no evaluation numbers yet** — the harness is Phase 4. This README will lead with the results table when there is one; until then it does not pretend otherwise.

| Phase | | |
|---|---|---|
| 0 — Foundation | ✅ | Config system, structured logging, health CLI |
| 1 — Tool layer | ✅ | 4 tools, 2 backends each, tested without an LLM |
| 2 — Sandbox | ✅ | Pyodide + Docker behind one protocol, [threat model](docs/THREAT_MODEL.md) |
| 3 — Agent | ⬜ | LangGraph loop, checkpointing, memory policy |
| 4 — Trajectory evaluation | ⬜ | The phase this repo exists for |
| 5 — Guardrails + injection | ⬜ | Measured attack/defence cycle |
| 6 — Trajectory viewer | ⬜ | The demo |
| 7 — Ship | ⬜ | Docker, HF Space, docs |

---

## Quick start

```bash
uv sync          # Python dependencies
npm install      # Pyodide, for the code sandbox

uv run vichara health
uv run vichara tools
```

No credentials are required. With an empty environment `health` exits 0 and reports a *degraded* capability set — tools fall back to fixture backends and the agent is told to say what it cannot do rather than guess. That is a supported way to run this project, not a broken one.

To add capability, copy `.env.example` to `.env` and fill in what you have. Every key is optional.

```bash
make check     # ruff · black · mypy --strict · pytest
make test
make format
```

## How it is put together

| Layer | What it does |
|---|---|
| [`settings.py`](src/vichara/settings.py) | Two config layers: `Settings` (environment, secrets) and `PipelineConfig` (behaviour, from YAML profiles). A profile is a committable description of one agent variant, so a results table can cite the exact configuration that produced it. |
| [`config/profiles/`](config/profiles/) | `baseline.yaml` is the frozen control. Once evaluation lands it does not change — a control that drifts makes every later comparison meaningless. |
| [`config/tools.yaml`](config/tools.yaml) | Declarative tool registration. Nothing is `required`, so an undeployed service shrinks the capability set instead of crashing the agent. |
| [`tools/`](src/vichara/tools/) | Built and unit-tested before the agent, with no LLM involved. When a trajectory goes wrong the question should be *why did it choose that*, never *did the tool even work*. |
| [`sandbox/`](src/vichara/sandbox/) | Two backends behind one protocol: Pyodide-in-Node by default (it runs on a Hugging Face Space), Docker for stronger isolation in CI. |
| `eval/` | Phase 4. The part that matters. |

## Design notes worth arguing about

**Summarisation is not about context overflow.** Gemini Flash holds a million tokens; a twelve-step trajectory never approaches it. Memory compression exists because cost is quadratic — every step resends the whole trajectory — and because a wall of raw tool output measurably degrades tool selection. Both are testable claims, and Phase 4 tests them.

**On a free tier the budget is requests per day, not dollars.** `max_llm_requests` is the ceiling that actually fires. `max_est_usd` is enforced and inert, so the guardrail is already real on the day the provider changes.

**The repeated-action threshold is two, not three.** One exact repeat of the same call with the same arguments is already a bug: nothing about the state changed, so nothing about the result will either.

**The baseline profile ships no injection defences.** A baseline attack success rate cannot be measured against a configuration that already defends. Provenance tagging stays on because it is instrumentation, not a defence.

**There is no calculator tool.** A Python sandbox subsumes it, and a warm Pyodide worker executes in ~1ms against a ~2s cold start — so the latency argument for shipping one does not survive contact with the measurement.

## The retrieval corpus

`textbook_search` calls [VidyaRAG](https://github.com/NehaBharti08/VidyaRAG) over HTTP when it is deployed, and otherwise serves 440 passages extracted from the real OpenStax PDFs — same citation format, same printed page numbers. A reviewer can open the printed book at a cited page and find the text.

The fixture backend ranks lexically (BM25) where the live service ranks densely. They do not rank identically, and any result produced against fixtures says so. See [`data/fixtures/ATTRIBUTION.md`](data/fixtures/ATTRIBUTION.md).

## Related

- [VidyaRAG](https://github.com/NehaBharti08/VidyaRAG) — the textbook retrieval service this agent calls as its primary tool.

## Licence

MIT. Textbook content is OpenStax, CC BY 4.0.
