#!/usr/bin/env python3
from __future__ import annotations

"""
Build a JSON manifest of ArXiv GCS PDF URLs for selected CS papers.

This script:

- Streams the local Cornell metadata snapshot from `/data/arxiv`.
- Filters records by:
  - category prefix (e.g. `cs.`),
  - year range (e.g. 2017–present),
  - presence of any keyword(s) in the title or abstract
    (e.g. transformer/attention/rnn/agent/llm).
- Checks which corresponding PDFs are *not* already present locally
  under a year-organised directory such as `/arxiv/pdfs`.
- Writes a JSON file describing the missing PDFs (ID, GCS URL, local path),
  which you can then feed into `gsutil` or other tooling.

Typical usage to match your request (CS, 2017+, transformer/attention/rnn/agent/llm):

    ./scripts/build_arxiv_keyword_gcs_manifest.py \\
        --category-prefix cs. \\
        --min-year 2017 \\
        --keywords "transformer attention rnn agent llm" \\
        --pdf-dir /arxiv/pdfs \\
        --output-json /data/arxiv/cs_keywords_2017plus_missing_pdfs.json

Then, for example, to download all missing PDFs listed in the manifest:

    jq -r '.[].gcs_url' /data/arxiv/cs_keywords_2017plus_missing_pdfs.json \\
      | gsutil -m cp -I /arxiv/pdfs

Requirements:
- `modules.arxiv_library.iter_metadata` must be able to see
  `/data/arxiv/arxiv-metadata-oai-snapshot.json`.
- `jq` and `gsutil` must be installed for the example shell pipeline.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

# Ensure we can import the local `modules` and `scripts` packages when
# running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.arxiv_library import iter_metadata  # type: ignore

# We reuse the same filtering semantics as `download_arxiv_pdfs_from_gcs`
# for category/year/keyword handling.
try:
    from scripts.download_arxiv_pdfs_from_gcs import (  # type: ignore
        _iter_filtered_ids,
    )
except Exception:
    _iter_filtered_ids = None  # type: ignore


ARXIV_ROOT = Path("/data/arxiv")
DEFAULT_PDF_DIR = Path("/arxiv/pdfs")
DEFAULT_GCS_PREFIX = "gs://arxiv-dataset/arxiv/arxiv/pdf"
DEFAULT_KEYWORDS = ["transformer", "attention", "rnn", "agent", "llm"]


@dataclass
class ManifestEntry:
    """
    JSON-serializable description of a single missing ArXiv PDF.
    """

    id: str
    gcs_url: str
    local_path: str


def _iter_filtered_ids_builtin(
    *,
    category_prefix: str,
    min_year: int,
    max_year: int,
    keywords: Sequence[str],
) -> Iterable[str]:
    """
    Fallback implementation of ID filtering if we cannot import the
    helper from `download_arxiv_pdfs_from_gcs`.

    This version omits the year filter (to avoid reimplementing the
    same logic twice) and simply matches category + keywords.
    """
    cpref = category_prefix.strip()
    kw = [k.strip().lower() for k in keywords if k.strip()]

    for rec in iter_metadata():
        if cpref and not rec.categories.startswith(cpref):
            continue

        if kw:
            haystack_parts: List[str] = []
            if getattr(rec, "title", ""):
                haystack_parts.append(rec.title)
            if getattr(rec, "abstract", ""):
                haystack_parts.append(rec.abstract)
            if not haystack_parts:
                continue
            haystack = " ".join(haystack_parts).lower()
            if not any(term in haystack for term in kw):
                continue

        rid = str(rec.id or "").strip()
        if not rid:
            continue
        base_id = rid.split("v", 1)[0]
        if len(base_id) < 4 or not base_id[:4].isdigit():
            continue

        yield base_id


def _select_ids(
    *,
    category_prefix: str,
    min_year: int,
    max_year: int,
    keywords: Sequence[str],
) -> List[str]:
    """
    Collect filtered ArXiv IDs using the shared helper if available,
    otherwise fall back to a local implementation.
    """
    ids: List[str] = []

    if _iter_filtered_ids is not None:
        for base_id, _raw in _iter_filtered_ids(
            category_prefix=category_prefix,
            min_year=min_year,
            max_year=max_year,
            keywords=keywords,
        ):
            ids.append(base_id)
    else:
        for base_id in _iter_filtered_ids_builtin(
            category_prefix=category_prefix,
            min_year=min_year,
            max_year=max_year,
            keywords=keywords,
        ):
            ids.append(base_id)

    # Deduplicate while preserving order.
    seen = set()
    unique_ids: List[str] = []
    for bid in ids:
        if bid in seen:
            continue
        seen.add(bid)
        unique_ids.append(bid)
    return unique_ids


def _build_manifest_entries(
    *,
    ids: Iterable[str],
    gcs_prefix: str,
    pdf_dir: Path,
    existing_ids: Set[str],
) -> List[ManifestEntry]:
    """
    Given base ArXiv IDs, build manifest entries for PDFs that are not
    already present under `pdf_dir`.
    """
    entries: List[ManifestEntry] = []
    prefix = gcs_prefix.rstrip("/")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for base_id in ids:
        # Skip if we already have a PDF with this ID somewhere under pdf_dir.
        if base_id in existing_ids:
            continue

        yymm = base_id[:4]
        # Canonical local path mirrors the GCS layout: /arxiv/pdfs/YYMM/<id>.pdf
        local_path = pdf_dir / yymm / f"{base_id}.pdf"
        gcs_url = f"{prefix}/{yymm}/{base_id}.pdf"
        entries.append(
            ManifestEntry(
                id=base_id,
                gcs_url=gcs_url,
                local_path=str(local_path),
            )
        )

    return entries


def _collect_existing_ids(pdf_dir: Path) -> Set[str]:
    """
    Recursively scan `pdf_dir` for existing PDFs and return their ID stems.

    This supports year-organised layouts such as `/arxiv/pdfs/YYMM/<id>.pdf`.
    """
    existing: Set[str] = set()
    if not pdf_dir.exists():
        return existing

    for p in pdf_dir.rglob("*.pdf"):
        if not p.is_file():
            continue
        existing.add(p.stem)

    return existing


def main(
    *,
    category_prefix: str,
    min_year: int,
    max_year: int,
    keywords: Sequence[str],
    pdf_dir: Path,
    gcs_prefix: str,
    output_json: Path,
    max_papers: int,
) -> None:
    if not ARXIV_ROOT.exists():
        print(
            f"[ERROR] Expected {ARXIV_ROOT} to exist with metadata.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Building manifest with category_prefix='{category_prefix}', "
        f"min_year={min_year or 'none'}, max_year={max_year or 'none'}, "
        f"keywords={', '.join(keywords) if keywords else 'none'}",
        flush=True,
    )

    ids = _select_ids(
        category_prefix=category_prefix,
        min_year=min_year,
        max_year=max_year,
        keywords=keywords,
    )

    if max_papers:
        ids = ids[:max_papers]

    if not ids:
        print("No metadata records matched the given filters.")
        output_json.write_text("[]\n", encoding="utf-8")
        return

    print(f"Matched {len(ids)} metadata records before local-file check.", flush=True)

    existing_ids = _collect_existing_ids(pdf_dir)
    print(
        f"Found {len(existing_ids)} existing PDF files under {pdf_dir} (recursively).",
        flush=True,
    )

    entries = _build_manifest_entries(
        ids=ids,
        gcs_prefix=gcs_prefix,
        pdf_dir=pdf_dir,
        existing_ids=existing_ids,
    )

    print(
        f"{len(entries)} PDFs are not yet present locally and will be listed in the manifest.",
        flush=True,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fh:
        json.dump([asdict(e) for e in entries], fh, indent=2)
        fh.write("\n")

    print(f"Wrote JSON manifest with {len(entries)} entries to {output_json}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build a JSON manifest of GCS URLs for ArXiv CS PDFs matching "
            "keyword and year filters, skipping those already downloaded locally."
        ),
    )
    parser.add_argument(
        "--category-prefix",
        type=str,
        default="cs.",
        help="ArXiv category prefix to filter by (e.g. 'cs.', 'cs.LG'). Default: cs.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2017,
        help="Only include papers with year >= this value. Default: 2017.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=0,
        help="Only include papers with year <= this value (0 = no upper bound).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=str(DEFAULT_PDF_DIR),
        help=f"Local directory where PDFs are or will be stored. Default: {DEFAULT_PDF_DIR}",
    )
    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default=DEFAULT_GCS_PREFIX,
        help=f"GCS prefix where PDFs live. Default: {DEFAULT_GCS_PREFIX}",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=" ".join(DEFAULT_KEYWORDS),
        help=(
            "Comma- or space-separated keywords to match in title/abstract "
            "(case-insensitive, any-match). "
            "Default: 'transformer attention rnn agent llm'."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="arxiv_gcs_manifest.json",
        help=(
            "Path to write the JSON manifest of missing PDFs. "
            "Default: ./arxiv_gcs_manifest.json"
        ),
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Maximum number of matching PDFs to consider (0 = no limit).",
    )

    args = parser.parse_args()
    # Support both comma- and space-separated keywords.
    raw_kw = args.keywords.replace(",", " ").split()

    main(
        category_prefix=args.category_prefix,
        min_year=args.min_year,
        max_year=args.max_year,
        keywords=raw_kw,
        pdf_dir=Path(args.pdf_dir),
        gcs_prefix=args.gcs_prefix,
        output_json=Path(args.output_json),
        max_papers=max(0, int(args.max_papers or 0)),
    )


