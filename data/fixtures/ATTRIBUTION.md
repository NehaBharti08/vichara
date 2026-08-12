# Fixture corpus attribution

`rag_corpus.jsonl` contains 440 passages extracted verbatim from two OpenStax
textbooks:

- **Biology** — OpenStax, Rice University. <https://openstax.org/details/books/biology>
- **Anatomy and Physiology** — OpenStax, Rice University. <https://openstax.org/details/books/anatomy-and-physiology>

Both are licensed **Creative Commons Attribution 4.0 International (CC BY 4.0)**
<https://creativecommons.org/licenses/by/4.0/>.

Text is unmodified. Passages were selected, packed to a character budget, and
trimmed at sentence boundaries; section titles had a trailing typesetting
asterisk removed. No wording was altered, paraphrased, or generated.

## Why this file exists

This corpus is a **sample of the real corpus**, not an imitation of one. It was
produced by [`scripts/build_fixtures.py`](../../scripts/build_fixtures.py) using
VidyaRAG's own PDF parser, so the section titles, printed page numbers, and
citation strings are the ones the live retrieval service emits.

That matters for a specific reason. Hand-written passages attributed to
"Biology, 4.2, p.188" would be citations nobody could check, and this project's
whole argument is that its numbers are checkable. A reviewer can open the
printed book at page 188 and find this text.

## What it is not

The fixture backend retrieves **lexically** (BM25). The live service retrieves
**densely** (`BAAI/bge-base-en-v1.5` over Qdrant). They return the same shape
and cite the same way, but they do not rank identically.

Any evaluation result produced against this backend says so in the table
caption. It is a supported configuration, not a stand-in pretending to be the
real thing.

## Regenerating

```bash
"…/VidyaRAG/.venv/Scripts/python.exe" scripts/build_fixtures.py \
    --vidyarag …/VidyaRAG
```

Requires VidyaRAG checked out with `data/raw/*.pdf` downloaded
(`uv run vidyarag download`). Nothing at runtime imports `vidyarag` or `pymupdf`.
