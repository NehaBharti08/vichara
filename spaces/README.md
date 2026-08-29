---
title: Vichara
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Evaluated on its trajectory, not just its answer
---

# Vichara

A study agent that plans, calls tools, and cites its sources — **evaluated on its trajectory, not just its answer.**

The panel below the answer is the point: what it planned, which tool it reached for, what came back, what each step cost, where a guardrail fired, and **which citations trace back to something a tool actually returned**.

## Results

41 hand-annotated tasks. Terminal correctness **0.976**, tool precision **1.00**, forbidden-tool rate **0.000**, refusal correctness **1.00** — and step efficiency **0.333**, meaning it takes about three times the optimal path. That last one is the standing weakness, and every one of those runs still scores *correct* on terminal state. An accuracy-only evaluation would never surface it.

28 prompt-injection attacks: success rate **0.11** undefended, **0.04** hardened.

All numbers are n=1 — a shape, not a rate.

## Notes on this demo

- Retrieval uses a committed corpus of **440 real OpenStax passages**, not a live service, so citations are checkable against the printed books.
- Web search **replays recorded responses**. The live web is not reproducible, and a number measured against this morning's index cannot be re-derived.
- This page is **static**: Hugging Face no longer offers free Docker Spaces, so it serves recorded trajectories rather than running the agent live. It loads instantly and cannot show a cold start or an exhausted quota.

Source, evaluation, threat model and injection study: **[github.com/NehaBharti08/vichara](https://github.com/NehaBharti08/vichara)**
