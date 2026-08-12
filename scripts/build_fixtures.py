"""Build the committed fixture retrieval corpus from the real OpenStax PDFs.

Run once, at build time, under **VidyaRAG's** interpreter -- it is the only
place the PDF parser lives:

    "d:/AI_Portfolio/Projects/VidyaRAG/.venv/Scripts/python.exe" \
        scripts/build_fixtures.py --vidyarag d:/AI_Portfolio/Projects/VidyaRAG

Why reuse VidyaRAG's parser rather than write passages by hand: the fixture
corpus has to be a genuine *sample of the real corpus*, not an imitation of
one. Hand-written text attributed to "Biology, 4.2, p.188" would be a citation
the agent could not actually be checked against, and this project's entire
argument is that its numbers are checkable. Real pages, real section titles,
real printed page numbers, and the same citation format the live service
emits, so swapping the fixture backend for the HTTP one changes the retrieval
mechanism and nothing else.

The output JSONL is committed. Nothing at runtime imports vidyarag or pymupdf.

Content is OpenStax, CC BY 4.0. See data/fixtures/ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Target size of one fixture passage, in characters. Roughly comparable to the
# live service's 512-token chunks without needing its tokeniser (and the 210MB
# ONNX download that comes with it).
PASSAGE_CHARS = 1400
MIN_PASSAGE_CHARS = 400
PASSAGES_PER_SECTION = 2
MAX_PASSAGES_PER_BOOK = 220

# Hard ceiling on one passage. A whole PDF page can exceed 6000 characters, and
# five of those retrieved together would blow the tool-output truncation limit
# before the model saw the last two citations.
MAX_PASSAGE_CHARS = 2200


def _clean(text: str) -> str:
    """Collapse the whitespace two-column extraction leaves behind."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title(raw: str | None) -> str | None:
    """Normalise an outline title.

    OpenStax marks many section headings with a trailing asterisk in the PDF
    outline. It is a typesetting artifact, and carrying it into a citation
    makes the citation not match the printed book -- which defeats the point
    of citing a printed page number at all.
    """
    if raw is None:
        return None
    cleaned = raw.strip().rstrip("*").strip()
    return cleaned or None


def _clip(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    """Trim to a paragraph or sentence boundary at or before ``limit``."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    for boundary in ("\n\n", ". ", "; "):
        cut = window.rfind(boundary)
        if cut > limit // 2:
            return window[: cut + len(boundary)].strip()
    return window.strip()


def _pack(pages: list[Any]) -> list[dict[str, Any]]:
    """Greedily pack consecutive pages of one section into passages."""
    passages: list[dict[str, Any]] = []
    buffer: list[str] = []
    start_page: int | None = None
    label: str | None = None

    def flush() -> None:
        nonlocal buffer, start_page, label
        if not buffer:
            return
        text = _clean("\n\n".join(buffer))
        if len(text) >= MIN_PASSAGE_CHARS and start_page is not None:
            passages.append({"text": text, "page_start": start_page, "printed_page": label})
        buffer = []
        start_page = None
        label = None

    for page in pages:
        text = _clean(page.text)
        if not text:
            continue
        if start_page is None:
            start_page, label = page.page, page.label
        buffer.append(text)
        if sum(len(b) for b in buffer) >= PASSAGE_CHARS:
            flush()
    flush()
    return passages


def build_book(spec: Any, pdf_path: Path, extract_pages: Any) -> list[dict[str, Any]]:
    """Extract a breadth-first sample of passages across a book's sections."""
    by_section: dict[tuple[str | None, str | None], list[Any]] = defaultdict(list)
    for page in extract_pages(spec, pdf_path):
        by_section[(_title(page.chapter), _title(page.section))].append(page)

    print(f"  {spec.slug}: {len(by_section)} sections parsed", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    # Breadth before depth: a couple of passages from many sections beats many
    # passages from a few, because eval questions are drawn from across the
    # syllabus and a corpus gap is indistinguishable from a retrieval miss.
    for (chapter, section), pages in sorted(by_section.items(), key=lambda kv: str(kv[0])):
        for index, passage in enumerate(_pack(pages)[:PASSAGES_PER_SECTION]):
            where = section or chapter
            printed = passage["printed_page"] or str(passage["page_start"])
            citation = (
                f"{spec.title}, {where}, p.{printed}" if where else f"{spec.title}, p.{printed}"
            )
            rows.append(
                {
                    "chunk_id": f"{spec.slug}:{len(rows):04d}:{index}",
                    "book_slug": spec.slug,
                    "book_title": spec.title,
                    "chapter": chapter,
                    "section": section,
                    "page_start": passage["page_start"],
                    "printed_page": passage["printed_page"],
                    "citation": citation,
                    "text": _clip(passage["text"]),
                    "license": spec.license_name,
                    "source_url": spec.source_url,
                }
            )
            if len(rows) >= MAX_PASSAGES_PER_BOOK:
                return rows
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vidyarag", type=Path, required=True, help="VidyaRAG repo root")
    parser.add_argument(
        "--out", type=Path, default=Path("data/fixtures/rag_corpus.jsonl"), help="Output JSONL"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.vidyarag / "src"))
    try:
        from vidyarag.ingest.corpus import CORPUS
        from vidyarag.ingest.parse import extract_pages
    except ImportError as exc:
        print(f"cannot import vidyarag: {exc}", file=sys.stderr)
        print("Run this with VidyaRAG's interpreter -- see the module docstring.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    for spec in CORPUS:
        pdf = args.vidyarag / "data" / "raw" / f"{spec.slug}.pdf"
        if not pdf.exists():
            print(f"  skipping {spec.slug}: {pdf} not found", file=sys.stderr)
            continue
        print(f"parsing {spec.slug} ...", file=sys.stderr)
        rows.extend(build_book(spec, pdf, extract_pages))

    if not rows:
        print("no passages extracted", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    size_kb = args.out.stat().st_size / 1024
    books = sorted({r["book_slug"] for r in rows})
    print(f"\nwrote {len(rows)} passages to {args.out} ({size_kb:.0f} KB)", file=sys.stderr)
    for book in books:
        print(f"  {book}: {sum(1 for r in rows if r['book_slug'] == book)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
