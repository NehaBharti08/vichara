# Vichara

[![CI](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml/badge.svg)](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml)

A study agent that plans a multi-step approach to an academic question, calls tools, and synthesises a cited answer — evaluated on its **trajectory**, not just its answer.

The agent loop is the least interesting part of this repository. A LangGraph ReAct loop is two days of work and thousands of identical ones exist. What follows is what this repo is actually for.

---

## Results

**41 tasks, annotated by hand before the agent ever ran.** Six metrics computed mechanically from the trajectory; one judged.

| metric | value | what it means |
|---|---|---|
| terminal correctness | **0.976** | 40 of 41 reached the right terminal state |
| tool precision | **1.00** | every tool it called was one the task needed |
| forbidden-tool rate | **0.000** | never reached for a tool the task forbids |
| refusal correctness | **1.00** | every impossible task refused, all within the step gate |
| answer correctness | 0.903 | 3 substantive failures, all multi-tool |
| **step efficiency** | **0.333** | **takes ~3× the optimal path — the standing weakness** |

Step efficiency is the finding. The agent reaches the right answer by a wasteful route, and **every one of those runs scores *correct* on terminal state**. An accuracy-only evaluation would never surface it. That is the entire argument for annotating optimal paths by hand.

### Prompt injection

**28 attacks**, each with a canary and a mechanical success rule, riding inside real tool results while the agent works a real task.

| profile | attack success rate | |
|---|---|---|
| baseline | **0.11** | 3 of 28 |
| hardened | **0.04** | 1 of 28 |

Citation verification took false-citation attacks from 0.25 to **0.00**. The one attack that still works is stopped in practice by the human approval interrupt, not by a filter.

**The most useful thing in that document is a correction.** It first reported 0.43 — until I found my own scoring counted the agent *reporting* an attack as being compromised by it. The agent was quoting payloads as evidence, exactly as designed. Both sweeps were re-run. [The full write-up](docs/PROMPT_INJECTION.md) leads with that mistake.

> **Every number here is n=1.** A shape, not a rate. Agent runs are stochastic and one seed gives no spread. The n=5 sweep is the number of record and has not been run — stated here rather than left to inference.

**Evidence:** [`docs/EVALUATION.md`](docs/EVALUATION.md) · [`docs/PROMPT_INJECTION.md`](docs/PROMPT_INJECTION.md) · [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) · raw results in [`eval_results/`](eval_results/)

---

## Quick start

```bash
uv sync          # Python dependencies
npm install      # Pyodide, for the code sandbox

uv run vichara health                              # what this environment can actually do
uv run vichara run "..." --trajectory              # answer a question, show the reasoning
uv run vichara evaluate --repeats 3                # the evaluation sweep
uv run vichara attack --profile hardened           # the injection suite
```

No credentials are required. With an empty environment `health` exits 0 and reports a *degraded* capability set — tools fall back to fixture backends and the agent is told to say what it cannot do rather than guess. That is a supported way to run this project, not a broken one.

To add capability, copy `.env.example` to `.env`. Every key is optional.

## What makes this different from a tutorial agent

**Evaluation measures the path, not just the destination.** Tool-selection precision against a required-tool set, step efficiency against a hand-annotated optimal path, refusal correctness gated on *step count* — because an agent that says "I don't know" after fifteen steps is broken even though the words are right.

**The security work is measured, and the failures are published.** A 28-attack corpus, before-and-after rates, the attacks that still work, and a correction to a number this repo previously got wrong.

**The threat model's useful half is the limitations.** [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) §4.1 names the gap that matters: the sandbox has no network, but the *agent* does, so an injection can exfiltrate through a legitimate `web_search` without crossing any sandbox boundary. Measurement later showed that attack failing 7 times out of 7 — right about consequence, wrong about likelihood, and both halves are in the document.

**Degradation is a measured property, not an outage.** Nothing is `required`. An undeployed service shrinks the capability set, the agent is told what it can no longer do, and the eval reports accuracy per capability profile.

## How it is put together

| Layer | What it does |
|---|---|
| [`settings.py`](src/vichara/settings.py) | Two config layers: environment/secrets, and behaviour from YAML profiles. A profile is a committable description of one agent variant, so a results table can cite the exact configuration that produced it. |
| [`tools/`](src/vichara/tools/) | Built and tested before the agent, with no LLM involved. When a trajectory goes wrong the question should be *why did it choose that*, never *did the tool even work*. |
| [`sandbox/`](src/vichara/sandbox/) | Two backends behind one protocol — Pyodide-in-Node by default (it runs on a Space), Docker for stronger isolation in CI. |
| [`agent/`](src/vichara/agent/) | Twelve LangGraph nodes, checkpointed and resumable, with a memory policy whose provenance markers survive summarisation. |
| [`guardrails/`](src/vichara/guardrails/) | Budget and loop ceilings, approval interrupts, injection defences, citation verification. |
| [`eval/`](src/vichara/eval/) | The part that matters. Resumable, seeded, quota-aware. |

## Design notes worth arguing about

**Summarisation is not about context overflow.** Gemini Flash holds a million tokens; a twelve-step trajectory never approaches it. Compression exists because cost is quadratic — every step resends the whole trajectory — and because raw tool output measurably degrades tool selection.

**A summary that drops its provenance marker is an injection laundering channel.** Digesting a poisoned document strips the "this came from an untrusted source" framing and re-emits the payload as trusted narration. There is a dedicated test that this cannot happen.

**On a free tier the budget is requests per day, not dollars.** `max_llm_requests` is the ceiling that actually fires; `max_est_usd` is enforced and inert, so the guardrail is already real the day the provider changes.

**`max_steps` is derived, not guessed.** All 41 optimal paths were annotated first; the longest is 3 tool calls, so the ceiling is 8 — about 2.5× the worst case. It was 12 before the annotations existed.

**A soft ceiling answers from what it has.** A per-tool limit once halted a run holding thirteen unused citations. The ceiling exists to stop the agent spending more, not to make it forget.

**There is no calculator tool.** A warm Pyodide worker executes in ~1 ms against a ~2 s cold start, so the latency argument for shipping one does not survive measurement.

## The retrieval corpus

`textbook_search` calls [VidyaRAG](https://github.com/NehaBharti08/VidyaRAG) over HTTP when deployed, and otherwise serves **440 passages extracted from the real OpenStax PDFs** — same citation format, same printed page numbers. A reviewer can open the printed book at a cited page and find the text.

The fixture backend ranks lexically (BM25) where the live service ranks densely. They do not rank identically, and **every result in this repo was produced against fixtures**. See [`data/fixtures/ATTRIBUTION.md`](data/fixtures/ATTRIBUTION.md).

## Related

- [VidyaRAG](https://github.com/NehaBharti08/VidyaRAG) — the textbook retrieval service this agent calls as its primary tool.

## Licence

MIT. Textbook content is OpenStax, CC BY 4.0.
