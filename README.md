# Vichara

[![CI](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml/badge.svg)](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml)

**[Live demo — the trajectory viewer](https://huggingface.co/spaces/nehabharti0802/vichara)**

A study agent that plans a multi-step approach to an academic question, calls tools, and synthesises a cited answer — evaluated on its **trajectory**, not just its answer.

The agent loop is the least interesting part of this repository. A LangGraph ReAct loop is two days of work and thousands of identical ones exist. What follows is what this repo is actually for.

---

## Results

**41 tasks, annotated by hand before the agent ever ran. 116 runs across five seeds.** Six metrics computed mechanically from the trajectory; one judged.

| metric | value | what it means |
|---|---|---|
| terminal correctness | **0.966** | reached the right terminal state |
| tool precision | **1.00** (median, IQR 0) | every tool it called was one the task needed |
| forbidden-tool rate | **0.000** | never reached for a tool the task forbids |
| refusal correctness | **1.00** | every impossible task refused, all within the step gate |
| answer correctness | 0.952 | |
| **step efficiency** | **1.00** (median, IQR 0.50) | the median run is optimal; a third are not — see below |

**Step efficiency was reported as 0.333 and that was a bug in the metric, not the agent.** The numerator counted *tool calls a human would make*; the denominator counted *graph nodes executed*. A flawless single-tool run executes plan → act → execute → synthesize and scored 0.333 for doing exactly the right thing. Counting the same unit on both sides gives a median of **1.00**.

**Chasing the worst remaining run found a third bug, this time in a guardrail.** The agent called `textbook_search` five times where one sufficed — and BM25 returned *byte-identical* results every time. Neither loop rule fired, because both fingerprint **arguments**, and the reformulated queries sat at 0.62–0.76 similarity, under the 0.9 threshold. Across the sweep, **8.3% of all tool output was bytes the agent already held.** Loop detection now hashes what a tool returned, not just what was asked.

Being careful about what that fixes: duplicates fully explain **3** of the 36 sub-optimal runs, partially 10, and 23 have no duplicates at all. The majority retrieve genuinely different passages and never decide they are done — which `act` was asked to judge while being shown no step count, no tool spend, and no count of evidence held. It now sees all three.

**Measured, on the eight tasks the fixes targeted** — n=5 both sides, 40 runs each. Deliberately not a headline number: those tasks were picked because they were the worst, so the subset is biased by construction.

| | before | after |
|---|---|---|
| terminal correctness | 37/40 | **39/40** |
| step efficiency (median) | 0.333 | **0.500** |
| `loop_detected` | 3 | **0** |

Both previously-unstable tasks improved and none regressed. `rag-innate-immunity` sits in the **test** split, was never tuned against, and moved 0.20 → 0.50. **Step efficiency is improved, not solved** — a median of 0.50 is still two calls where one would do; the agent stops sooner, not yet at the right time.

Getting there cost two regressions, both caught before publication. The first budget line made the agent refuse answerable questions to save budget. Fixing that exposed a second underneath: three runs halted on `loop_detected` while holding 5, 5 and 10 citations — one of them the exact passage needed — reporting *"I stopped because I was repeating the same action without making progress"*. That is the thirteen-citations bug in the branch the soft ceiling never covered, and `block()`'s own docstring had the faulty reasoning written down. Loops are now soft.

That makes four defects found in this repo's own instruments — the injection scoring, the step-efficiency units, a guardrail that watched the wrong end of the tool call, and a scorer that never read the `prompt_hashes` recorded specifically so two agent versions could not be averaged together. Each was caught by checking one concrete case against the arithmetic, and the first two had been published as agent weaknesses.

### Prompt injection

**28 attacks**, each with a canary and a mechanical success rule, riding inside real tool results while the agent works a real task.

| profile | attack success rate | |
|---|---|---|
| baseline | **0.11** | 3 of 28 |
| hardened | **0.04** | 1 of 28 |

Citation verification took false-citation attacks from 0.25 to **0.00**. The one attack that still works is stopped in practice by the human approval interrupt, not by a filter.

**The most useful thing in that document is a correction.** It first reported 0.43 — until I found my own scoring counted the agent *reporting* an attack as being compromised by it. The agent was quoting payloads as evidence, exactly as designed. Both sweeps were re-run. [The full write-up](docs/PROMPT_INJECTION.md) leads with that mistake.

> **116 of a planned 205 runs.** The sweep halted when the free-tier daily quota ran out; the 57 runs the provider failed are dropped rather than scored, because a quota exhaustion is not agent behaviour. 25 of 41 tasks have all five seeds, the rest three or four, and three multi-tool tasks have one — per-task rates on those are not claimable and are labelled where they appear.

**Evidence:** [`docs/EVALUATION.md`](docs/EVALUATION.md) · [`docs/PROMPT_INJECTION.md`](docs/PROMPT_INJECTION.md) · [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) · raw results in [`eval_results/`](eval_results/)

---

## The demo

**[huggingface.co/spaces/nehabharti0802/vichara](https://huggingface.co/spaces/nehabharti0802/vichara)** — seven recorded runs, each showing one behaviour worth looking at: a grounded answer, multi-tool orchestration, a correct refusal, a clarifying question, a guardrail stopping a runaway, a detected prompt injection, and a fabricated citation being removed.

It is a **static** page. Hugging Face withdrew free Docker Spaces partway through this project, and rather than pay for a live agent the viewer now serves recorded trajectories. That turned out to suit it: the viewer was always about *displaying* a trajectory rather than producing one, and a static page loads instantly, never sleeps, and cannot show a cold start or an exhausted quota — the three ways a hosted agent demo usually embarrasses its author.

Regenerate and redeploy with:

```bash
uv run python scripts/export_static.py
uv run python scripts/deploy_space.py --repo-id <user>/vichara
```

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

**The live read path is verified working** — `POST /v1/search` answered against the real 768-dim index and returned the right sections (`17.3. The Pituitary Gland and Hypothalamus` for a hypothalamus question, `12.4. The Action Potential` for a resting-potential one). Testing it turned up the sharpest live-vs-fixture difference found so far, and it is not the ranking:

| query | BM25 (fixture) | dense (live) |
|---|---|---|
| in-corpus | 27–31 | 0.80–0.82 |
| **out-of-corpus** | **6.5** | **0.55** |

**Neither backend returns empty.** Ask either one about quantum chromodynamics and it hands back three genetics passages — BM25 because "chromodynamics" lexically matches "chromosomal", dense because a nearest neighbour always exists. BM25's 5× score gap makes the miss obvious; dense cosine's is far narrower.

And the agent sees neither, because **the tool computes the score and discards it** — it forwards only citation, book, section, page and text. So the agent cannot distinguish the best match in a corpus that covers a topic from the best match in one that does not, and must infer irrelevance by reading the passage. It does that well against fixtures (refusal correctness 1.00, false-refusal 0.026), but those numbers were earned where off-topic hits read obviously wrong. Whether they transfer to the live service is untested, and surfacing the score is the obvious next experiment rather than a change made on the way past.

## Related

- [VidyaRAG](https://github.com/NehaBharti08/VidyaRAG) — the textbook retrieval service this agent calls as its primary tool.

## Licence

MIT. Textbook content is OpenStax, CC BY 4.0.
