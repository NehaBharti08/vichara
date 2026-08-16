# Evaluation

41 annotated tasks, six programmatic metrics, one judged. This document is the
argument that the agent works, and — more usefully — the record of where it
does not.

> **Status: n=1 sweep, dev and test pooled.** The headline table below is a
> single run per task. It is enough to establish the shape of the result and
> to find real defects, and it is **not** enough to claim a rate: with one
> seed there is no spread to report, and agent runs are stochastic. The n=5
> sweep is the number of record and is not yet run. Everything here is
> labelled accordingly rather than presented as final.

## Method

**Annotations are written before the agent runs, every time.** Each task
carries an expected terminal state, a required tool set, a forbidden tool set,
and an optimal path — the minimal ordered tool sequence a competent human
would use. An optimal path written after watching a trajectory is not a
measurement, it is a rationalisation of whatever the agent happened to do.

**Tasks are sourced from what the corpus actually contains.** Every retrieval
task was checked with [`scripts/check_coverage.py`](../scripts/check_coverage.py)
before being annotated. Of 28 candidate topics, 21 passed; the rest were
dropped because their best match is a glossary stub. This rule exists because
Phase 3 ended with the agent refusing a glycolysis question and being *right*
to — the corpus lacks the passage. Without the check, a retrieval miss and a
corpus gap are indistinguishable, and every number computed over such a task
is noise attributed to the agent.

**Six metrics are mechanical, one is judged.** Tool precision and recall, step
efficiency, refusal correctness, forbidden-tool rate, grounding, and
cost/latency are all computed from `(trajectory, gold task)` with no model
involved. They can be re-run over stored trajectories months later without
re-running the agent. Only "does this citation support this claim" is judged,
because nothing mechanical can answer it.

**dev / test split.** 28 dev, 13 test. Prompts are tuned against dev only.
Overfitting prompts to an eval set is the commonest silent failure in agent
evaluation, and the only defence is not to look.

## Results (n=1, all 41 tasks)

| metric | value | reading |
| --- | --- | --- |
| terminal_correct | **0.976** | 40 of 41 reached the right terminal state |
| answer_correct | 0.903 | 3 substantive failures |
| forbidden_tool_rate | **0.000** | never reached for a tool the task forbids |
| tool_precision (median) | **1.00** | every tool it called was one the task needed |
| refusal_correct | **1.00** | every impossible task refused within the step gate |
| mean_steps_to_refusal | 0.0 | refusals happen in the planner, before any tool call |
| false_refusal_rate | 0.024 | one answerable task refused |
| cited_rate | 0.732 | the remainder are refusals and clarifications, which have nothing to cite |
| **step_efficiency (median)** | **0.333** | **the agent uses roughly 3x the optimal number of steps** |
| llm_requests (median) | 4.0 | IQR 4.0 |

## What this says

**Tool selection is not the problem.** Precision 1.00 and a forbidden-tool
rate of exactly zero mean the agent reaches for the right tool and never for a
wrong one — including on the tasks designed to tempt it, like searching a
textbook for arithmetic.

**Refusal behaviour is the strongest result.** All six impossible tasks were
refused, all within the step gate, and `mean_steps_to_refusal` of 0 means the
refusal happens in the planner before any tool is called. That is the
behaviour the metric was gated on step count to demand: an agent that
eventually says "I don't know" after fifteen steps is broken even though the
words are right.

**Step efficiency is the real weakness, and it is the headline finding.** A
median of 0.333 means the agent takes about three times the annotated optimal
path. It reaches the right answer by a wasteful route. This is exactly the
defect an accuracy-only evaluation would never surface — every one of those
runs is scored *correct* on terminal state.

### The three failures are all multi-tool

| task | what happened |
| --- | --- |
| `multi-smooth-muscle` | answered, but the computed percentage was wrong |
| `multi-search-then-compute` | answered, but the unit conversion was wrong |
| `multi-three-tool` | **refused** a task it should have answered — the only false refusal in the set |

Single-tool tasks are essentially solved; every failure is in orchestration,
and the hardest orchestration task (the only three-tool one) is the single
false refusal. That is a coherent, actionable finding rather than a scattering
of unrelated errors.

## What was fixed because of this

**`max_steps` is now derived rather than guessed.** All 41 optimal paths were
annotated first; the longest is 3 tool calls. The ceiling is set at 8 —
roughly 2.5x the worst case, leaving room for two full failure→reflect→retry
cycles on the hardest task. It was 12 before the annotation existed, and 12
was a guess.

**A routing bug found on the first sweep.** `ambiguous-the-cycle` expected
`clarify` and got `refused`. The planner marks an ambiguous task as both
`answerable: false` and `needs_clarification: true` — not incoherent, since it
cannot be answered as written — and the routing checked answerability first,
silently turning *every* ambiguous task into a refusal. Clarify now wins.
Refusing tells the user nothing can be done; asking which cycle they meant
costs a sentence and gets them an answer.

**An annotation invariant that was wrong.** The schema initially required
`optimal_path ⊆ expected_tools`. For a refusal task those legitimately
diverge: confirming a chapter does not exist is a reasonable first move, but
*requiring* it would penalise an agent that recognised the false premise
outright — which is better behaviour, not worse. The check now applies only to
answerable tasks. Caught by the 10-task pilot, before 40 more were annotated
against a bad rule.

## Honest limitations

- **n=1.** No spread, so no rate can be claimed. Reported as a shape, not a
  result.
- **Retrieval is the fixture corpus, not the live service.** 440 real OpenStax
  passages ranked with BM25, where VidyaRAG ranks densely. Same citations,
  different ranking. See [ATTRIBUTION](../data/fixtures/ATTRIBUTION.md).
- **Web search is recorded, not live**, and deliberately so: a number measured
  against whatever was indexed this morning cannot be re-derived in three
  months.
- **`answer_contains` is a keyword check**, which is a weak proxy for
  correctness. It is used anyway because it is *mechanical*; six mechanical
  metrics with a known weakness beat one judged metric with an unknown one.
- **The grounding judge is not yet implemented**, so `grounding_sources_present`
  currently checks that the expected section appears among the citations the
  tools returned — not that the passage supports the claim.
- **Fault injection is implemented but not yet swept**, so no recovery rate is
  reported.
- **The corpus is sparse** at roughly two passages per section, which caps how
  specific a groundable question can be.

## Reproducing

```bash
uv run vichara evaluate --repeats 1 --report docs/EVALUATION.md
uv run vichara evaluate --repeats 5                     # the number of record
uv run vichara evaluate --disable textbook_search       # degraded capability profile
uv run vichara evaluate --fault textbook_search:plausible_but_wrong
```

The runner is resumable: completed `(task, seed)` pairs are skipped, and
`--max-requests` stops a sweep before it exhausts the day's free-tier
allowance. A full n=5 sweep is roughly 2,000 requests and takes a day or two
of unattended wall clock, which is why resumability is a requirement rather
than a nicety.
