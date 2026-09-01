# Evaluation

41 annotated tasks, six programmatic metrics, one judged. This document is the
argument that the agent works, and — more usefully — the record of where it
does not.

> **Status: n=5 across seeds 0-4, dev and test pooled, 116 scored runs.**
> The sweep halted at 182 of 205 pairs when the free-tier daily quota ran out.
> The 57 runs the provider failed are **dropped, not scored** — a quota
> exhaustion is not agent behaviour, and counting those as failures would
> report the agent losing capability it never lost. 25 of 41 tasks have all
> five seeds; the rest have three or four. Spread is reported as median and
> IQR throughout.

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

## Results (n=5, 41 tasks, 116 scored runs)

| metric | value | reading |
| --- | --- | --- |
| terminal_correct | **0.966** | mean over 116 runs |
| answer_correct | 0.952 | n=105; the remainder have nothing to answer |
| forbidden_tool_rate | **0.000** | never reached for a tool the task forbids |
| tool_precision (median) | **1.00** | IQR 0.00 — every tool it called was one the task needed |
| tool_recall (median) | **1.00** | IQR 0.00 |
| refusal_correct | **1.00** | every impossible task refused within the step gate |
| mean_steps_to_refusal | 0.0 | refusals happen in the planner, before any tool call |
| false_refusal_rate | 0.026 | 3 of 116 |
| cited_rate | 0.888 | the remainder are refusals and clarifications |
| step_efficiency (median) | **1.00** | IQR 0.50, min 0.20 — the median run is optimal, a third are not |
| llm_requests (median) | 4.0 | IQR 4.0 |

### Step efficiency was published as 0.333, and that was a bug in the metric

The numerator counted *tool calls a human would make*; the denominator counted
*graph nodes executed*. A flawless single-tool run executes plan → act →
execute → synthesize and scored 0.333 for doing exactly the right thing. The
median was an artifact of dividing two different units, and it had been
published here as the headline weakness of the agent.

Counting tool calls on both sides gives a median of **1.00**. This is the
second measurement bug found in this repo's own instruments, after the
injection scoring; both were caught by checking the arithmetic against a
single concrete case, and both had been reported as agent weaknesses.

**The real signal survives the correction.** 36 of 103 scored runs (35%) still
come in under the annotated optimal path, and the worst takes five tool calls
where one suffices. That is the thing an accuracy-only evaluation would never
surface — every one of those runs is scored *correct* on terminal state.

### Chasing the worst run found a third bug, this time in a guardrail

`rag-hypothalamus-pituitary` was both the worst step-efficiency task and one of
only two inconsistent across seeds (4/5). The trajectory is unambiguous: the
first `textbook_search` returns the passage that answers the question, and the
agent then reformulates four more times while BM25 hands back **byte-identical
results every time**. On seed 4 it finally repeats a query verbatim and trips
`loop_detected`.

Neither loop rule fired, because both fingerprint *arguments*. The successive
queries scored 0.62–0.76 pairwise similarity, comfortably under the 0.9
`near_repeat` threshold. Tightening that threshold is not the fix — it would
flag legitimate reformulation, which is the behaviour that recovers a failed
retrieval. `LoopConfig`'s own docstring already stated the right principle,
*"the observation did not change, so neither will the result"*, and then only
ever checked the arguments.

Replaying all 197 stored baseline trajectories through a result-side digest:

| | |
| --- | --- |
| redundant calls | **19**, across 7 tasks |
| share of all tool output | **8.3%** |
| duplicate content | 183 KB, each resent up to 3× under the verbatim window |
| citations lost by suppressing it | **0** — the identical earlier call supplied them |

It warns rather than blocks. The agent is not missing evidence, it is holding
the same evidence twice, so halting would repeat the mistake the soft ceiling
made when it ended a run holding thirteen unused citations.

### Two causes, and duplicates are the smaller one

Attributing each of the 36 sub-optimal runs to a cause is the part worth being
careful about, because it would be easy to claim the fix above solves step
efficiency. It does not:

| | runs |
| --- | --- |
| waste fully explained by byte-identical repeats | 3 |
| partially explained | 10 |
| **no duplicate results at all** | **23** |

The majority retrieve genuinely *different* passages and simply never decide
they are done. That is a sufficiency judgement, and `act` was asked to make it
blind: the prompt said to answer "if you already have enough" while showing the
model no step count, no tool spend, and no count of the evidence it already
held. The guard knew all three and enforced ceilings on them, so every limit
arrived as an unexplained stop rather than a pressure the agent could plan
against. `act` now renders that budget.

**Neither fix has a measured effect yet, and this document will not claim one
until it does.** Editing the act prompt moves `prompt_hashes`
(`17c132fdfa89` → `23e3495c581c`), so the 116 runs above are correctly no
longer comparable to anything recorded afterwards — that is the mechanism
working as designed. The re-run is blocked on the free-tier daily quota. What
is established is that the detector fires on the right 19 calls and loses no
citations; what is *not* established is that the agent stops when told.

### Consistency across seeds

39 of 41 tasks reach the same terminal state on every seed. The two that do not
are the two the analysis above is about:

| task | correct on |
| --- | --- |
| `rag-hypothalamus-pituitary` | 4 of 5 seeds |
| `search-glp1-mechanism` | 3 of 5 seeds |

Both are single-tool retrieval tasks that the agent over-searches. Instability
and inefficiency have the same root here, which is the most actionable shape
this result could have taken.

### The substantive failures

| task | seeds scored | what happened |
| --- | --- | --- |
| `multi-smooth-muscle` | 1 | answered, but the computed percentage was wrong |
| `multi-search-then-compute` | 1 | answered, but the unit conversion was wrong |
| `multi-three-tool` | 1 | **refused** a task it should have answered |
| `rag-hypothalamus-pituitary` | 5 | 1 seed hit `loop_detected` after five identical retrievals |
| `search-glp1-mechanism` | 5 | 2 seeds false-refused |

**The three multi-tool rows have one seed each, and that is a limit on what
can be said about them.** Those tasks are late in the sweep and the quota ran
out before their remaining seeds ran, so a single failure cannot be
distinguished from a task that fails every time. They are listed because the
failure is real, not because the rate is known.

The pattern that *does* have five seeds behind it: single-tool tasks are solved
except where the agent over-searches, and both unstable tasks are over-search
cases. Orchestration failures are arithmetic and unit handling inside an
otherwise correct trajectory — the tool selection was right in every one.

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

- **The sweep is incomplete.** 116 of a planned 205 runs. 25 of 41 tasks have
  all five seeds; the rest have three or four, and three multi-tool tasks have
  one. Per-task rates on those are not claimable, and are labelled where they
  appear.
- **`agent_version` covers prompts, not code.** It folds the prompt-file
  hashes, so a prompt edited underneath a running sweep is caught. A
  behavioural change in a node — making loop detection soft, for instance —
  moves it not at all. Hashing the source would invalidate results on every
  refactor, so discarding prior runs across a behavioural code change is a
  judgement the operator has to make. Superseded runs are kept in
  `eval_results/archive/` rather than deleted.
- **The two fixes above are unmeasured.** The result-side loop detector and the
  act budget line were both derived from the runs in this document and have not
  been evaluated against a fresh sweep. `prompt_hashes` changed, so no future
  run will be silently pooled with these.
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
