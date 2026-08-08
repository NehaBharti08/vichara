"""Record real Tavily responses for offline replay.

Run when you have a key and want to extend the recorded corpus:

    uv run python scripts/record_search.py --queries scripts/seed_queries.txt

Each query costs one Tavily credit. Existing recordings are preserved unless
--overwrite is passed, so this is safe to re-run to add new queries.

Why record at all: the live web is not reproducible. An eval number measured
against whatever was indexed this morning cannot be re-derived in three months
and therefore cannot be compared against. Recordings make the search path
deterministic without making it fictional.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from vichara.settings import REPO_ROOT, Settings
from vichara.tools.websearch.tavily import TavilySearchBackend

DEFAULT_OUT = REPO_ROOT / "data" / "fixtures" / "search_responses.jsonl"

SEED_QUERIES = [
    "CRISPR base editing clinical trial results 2025",
    "mRNA cancer vaccine trial outcomes",
    "CAR-T cell therapy solid tumors progress",
    "gut microbiome and depression recent research",
    "telomere-to-telomere complete human genome assembly",
    "AlphaFold impact on structural biology",
    "new class of antibiotics discovered resistance",
    "lab-grown organ transplant milestone",
    "amyloid hypothesis Alzheimers current status",
    "obesity GLP-1 drugs mechanism of action research",
]


def load_existing(path: Path) -> dict[str, dict]:  # type: ignore[type-arg]
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["query"]] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--queries", type=Path, help="File with one query per line")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="Re-record existing queries")
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    settings = Settings()
    if not settings.has_tavily_key:
        print("TAVILY_API_KEY not set -- nothing to record.", file=sys.stderr)
        return 1

    queries = (
        [q.strip() for q in args.queries.read_text(encoding="utf-8").splitlines() if q.strip()]
        if args.queries
        else SEED_QUERIES
    )

    existing = load_existing(args.out)
    backend = TavilySearchBackend(settings.tavily_api_key.get_secret_value())

    recorded = 0
    for query in queries:
        if query in existing and not args.overwrite:
            print(f"  skip (already recorded): {query}", file=sys.stderr)
            continue
        try:
            results = backend.search(query, max_results=args.max_results)
        except Exception as exc:  # noqa: BLE001 - a recorder should report and continue
            print(f"  FAILED {query}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        existing[query] = {
            "query": query,
            "results": [r.model_dump() for r in results],
        }
        recorded += 1
        print(f"  recorded {len(results)} results: {query}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for query in sorted(existing):
            handle.write(json.dumps(existing[query], ensure_ascii=False) + "\n")

    print(
        f"\n{recorded} new, {len(existing)} total -> {args.out} "
        f"({args.out.stat().st_size / 1024:.0f} KB)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
