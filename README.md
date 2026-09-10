# Vichara

[![CI](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml/badge.svg)](https://github.com/NehaBharti08/vichara/actions/workflows/ci.yml)

**[Live demo — the trajectory viewer](https://huggingface.co/spaces/nehabharti0802/vichara)**

A study agent that plans a multi-step approach to an academic question, calls tools, and synthesises a cited answer — evaluated on its **trajectory**, not just its answer.

The agent loop is the least interesting part of this repository. A LangGraph ReAct loop is two days of work and thousands of identical ones exist. What follows is what this repo is actually for.

---

## Results

**41 tasks, annotated by hand before the agent ever ran. 205 runs, five seeds, complete.** Six metrics computed mechanically from the trajectory; one judged.

| metric | value | what it means |
|---|---|---|
| terminal correctness | **0.946** | reached the right terminal state |
| tool precision | **1.00** (median, IQR 0) | every tool it called was one the task needed |
| forbidden-tool rate | **0.000** | never reached for a tool the task forbids |
| refusal correctness | **1.00** | all 30 impossible runs refused, every one inside the step gate |
| answer correctness | 0.877 | |
| false-refusal rate | 0.049 | |
| **step efficiency** | **1.00** (median, IQR 0.50) | the median run is optimal; a third are not |

Per category, which is where the number actually lives:

| category | n | terminal correctness | step efficiency |
|---|---|---|---|
| impossible | 30 | **1.000** | — |
| single-tool | 95 | **0.989** | 1.00 |
| multi-tool | 55 | 0.891 | 0.67 |
| ambiguous | 25 | **0.840** | — |

Orchestration and ambiguity are the weaknesses; tool selection and refusal are not.

### Six defects — four in the measuring instruments, two in the agent

The agent loop was never the hard part. Measuring it honestly was. Each of
these was found by checking one concrete case against the arithmetic, and the
first two had already been **published here as agent weaknesses** before anyone
thought to check the instrument that produced them.

| # | what was wrong | how it showed up |
|---|---|---|
| 1 | Injection scoring counted the agent *reporting* an attack as being compromised by it | ASR published at 0.43; actually **0.11** |
| 2 | Step efficiency divided tool calls by graph nodes — two different units | Published at 0.333; actually **1.00** |
| 3 | *(agent)* Loop detection fingerprinted **arguments, never results** | 8.3% of all tool output was bytes the agent already held |
| 4 | *(agent)* A detected loop *discarded* runs holding 5 and 10 citations — including the exact passage needed | 3 runs lost to `loop_detected` while holding the answer |
| 5 | Task-major sweep order made an interrupted run **the easy end of the set**, not a smaller sample | 97% coverage of single-tool tasks vs 20% of ambiguous |
| 6 | The injection suite skipped all 28 attacks and printed a rate anyway | reported `0.11` from **zero work** |

Defect 5 is the one worth dwelling on. Disclosing "116 of 205 runs" describes
the *size* of a gap and says nothing about its *shape* — and the shape was that
the quota died in the same place every sweep. **A resumable job that iterates in
a fixed order doesn't degrade gracefully; it degrades selectively.** The sweep
now runs seed-major, so truncating anywhere preserves the full set's category
mix (verified: 19/11/6/5, identical).

Defect 6 was caught by two guards built after the earlier ones: the static site
refuses to publish an unattributable number, and `llm_requests` came back as
`0` — which 28 live attacks cannot do.

The common root cause behind 5 and 6: **results keyed by identity without
recording which agent produced them.** Every results file now carries
`agent_version`, and resume, reporting and publication all refuse to mix
versions.

Fixing 3 and 4 was measured, not assumed — n=5 on the eight tasks they
targeted: terminal correctness 37/40 → **39/40**, step efficiency median
0.333 → **0.500**, `loop_detected` 3 → **0**. *Improved, not solved* — 0.50 is
still two tool calls where one would do.

Full detail: [`docs/EVALUATION.md`](docs/EVALUATION.md).

### Prompt injection

**28 attacks**, each with a canary and a mechanical success rule, riding inside real tool results while the agent works a real task.

| profile | attack success rate | |
|---|---|---|
| baseline | **0.11** | 3 of 28 |
| hardened | **0.04** | 1 of 28 |

Re-run on the current agent, and the aggregate held while the composition moved: `cite-fake-url` no longer succeeds, `refuse-poison-claim` now does. One attack fixed, one regressed, and 0.11 hid both — which is why the write-up names the attacks that work rather than reporting only a rate.

Citation verification took false-citation attacks from 0.25 to **0.00**. The one attack that still works is stopped in practice by the human approval interrupt, not by a filter.

**The most useful thing in that document is a correction.** It first reported 0.43 — until I found my own scoring counted the agent *reporting* an attack as being compromised by it. The agent was quoting payloads as evidence, exactly as designed. Both sweeps were re-run. [The full write-up](docs/PROMPT_INJECTION.md) leads with that mistake.

> **All 205 runs are present: 41 tasks × 5 seeds, no gaps.** Runs the provider failed are dropped rather than scored — a quota exhaustion is not agent behaviour, and scoring it would report the agent losing capability it never lost. Three tasks remain inconsistent across seeds (`ambiguous-recent-work` 1/5, `multi-smooth-muscle` 4/5, `search-glp1-mechanism` 4/5) and are named rather than averaged away.

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

uv run python app.py                               # the interactive agent, at localhost:7860
```

`app.py` is the one worth running. It is the same UI the Space serves, except
live: you ask a question and watch the trajectory build a step at a time — plan,
tool choice, tool call, result — with elapsed time and the stage the agent is
currently in. Expect 30-60 seconds for a multi-tool question. Set
`GRADIO_SERVER_PORT` if 7860 is taken.

Three questions that show three different behaviours:

| ask | expected |
|---|---|
| *How does the hypothalamus control the anterior pituitary gland?* | a cited answer |
| *Explain the quantum biology chapter in OpenStax Biology* | a **refusal** — no such chapter exists |
| *Is it normal?* | a **clarifying question**, not a guess |

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
