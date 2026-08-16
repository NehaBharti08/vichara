"""Check whether a candidate task is answerable from the corpus.

Run this *before* annotating a task, never after running the agent on it.

The reason is the finding that ended Phase 3: the agent refused a glycolysis
question and was right to -- the corpus genuinely lacks that passage. Without
this check, a retrieval miss and a corpus gap are indistinguishable, and every
number computed over such a task is noise attributed to the agent.

    uv run python scripts/check_coverage.py "how the hypothalamus controls the pituitary"
    uv run python scripts/check_coverage.py --file candidates.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vichara.tools.rag import FixtureRetrievalBackend

# Below this BM25 score the top hit is usually a glossary stub or an
# incidental mention -- present in the corpus, but not enough to ground an
# answer on. Calibrated by inspection, and deliberately conservative: a task
# wrongly marked answerable poisons a metric, one wrongly rejected costs
# nothing but a replacement.
USABLE_SCORE = 6.0


def check(backend: FixtureRetrievalBackend, query: str, top_k: int = 3) -> bool:
    results = backend.search(query, top_k=top_k)
    if not results:
        print(f"  NO HITS      {query}")
        return False

    best = results[0]
    usable = best.score >= USABLE_SCORE and "Glossary" not in (best.section or "")
    print(f"  {'USABLE  ' if usable else 'WEAK    '} {query}")
    for passage in results:
        marker = "*" if passage is best else " "
        print(f"     {marker} {passage.score:6.2f}  {passage.citation}")
    return usable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Candidate question or key phrase")
    parser.add_argument("--file", type=Path, help="File with one candidate per line")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    queries = list(args.query)
    if args.file:
        queries += [
            q.strip() for q in args.file.read_text(encoding="utf-8").splitlines() if q.strip()
        ]
    if not queries:
        parser.error("give a query or --file")

    backend = FixtureRetrievalBackend()
    usable = sum(check(backend, q, args.top_k) for q in queries)
    print(f"\n{usable}/{len(queries)} candidates are usable as grounded tasks")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
