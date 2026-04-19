#!/usr/bin/env python3
from __future__ import annotations

"""
Download ArXiv PDFs based on the local metadata snapshot.

This script is designed to:

- Read paper IDs from `/data/arxiv/arxiv-metadata-oai-snapshot.json`
  via `modules.arxiv_library.iter_metadata`.
- Download PDFs from `https://export.arxiv.org/pdf/{id}.pdf`.
- Respect a configurable delay between downloads to be polite to arXiv.

It does NOT use the arXiv API for search; it assumes you already have
the metadata locally.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from modules.arxiv_library import iter_metadata  # type: ignore


DEFAULT_PDF_DIR = Path("/arxiv/pdfs")
DEFAULT_PDF_CACHE_DIR = Path(__file__).resolve().parents[1] / "exports" / "arxiv_pdfs"
DEFAULT_DELAY = 3.0  # seconds between PDF downloads


def _pdf_roots(out_dir: Path) -> list[Path]:
    roots = [out_dir, DEFAULT_PDF_DIR, DEFAULT_PDF_CACHE_DIR]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _writable_pdf_root(out_dir: Path) -> Path:
    for root in _pdf_roots(out_dir):
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".codex_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return root
        except Exception:
            continue
    return out_dir


def _local_pdf_candidates(out_dir: Path, arxiv_id: str) -> list[Path]:
    norm_id = str(arxiv_id or "").strip().split("/")[-1]
    if not norm_id:
        return []
    candidates: list[Path] = []
    for root in _pdf_roots(out_dir):
        if len(norm_id) >= 4 and norm_id[:4].isdigit():
            candidates.append(root / norm_id[:4] / f"{norm_id}.pdf")
        candidates.append(root / f"{norm_id}.pdf")
    return candidates


def download_pdf(arxiv_id: str, out_dir: Path, *, timeout: int = 60) -> bool:
    """
    Download a single ArXiv PDF by id into `out_dir`.

    Returns True on success, False on failure.
    """
    # Some ids may contain URL prefixes; normalize to the trailing segment.
    arxiv_id = arxiv_id.split("/")[-1]
    pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
    write_root = _writable_pdf_root(out_dir)
    if len(arxiv_id) >= 4 and arxiv_id[:4].isdigit():
        out_path = write_root / arxiv_id[:4] / f"{arxiv_id}.pdf"
    else:
        out_path = write_root / f"{arxiv_id}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if any(path.exists() for path in _local_pdf_candidates(out_dir, arxiv_id)):
        return False

    resp = requests.get(pdf_url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with out_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            fh.write(chunk)
    return True


def main(
    *,
    max_papers: int,
    category_prefix: Optional[str],
    delay: float,
    out_dir: Path,
) -> None:
    count = 0
    skipped_existing = 0
    category_prefix = (category_prefix or "").strip()

    for rec in iter_metadata():
        if max_papers and count >= max_papers:
            break

        if category_prefix and not rec.categories.startswith(category_prefix):
            continue

        arxiv_id = rec.id
        try:
            changed = download_pdf(arxiv_id, out_dir=out_dir)
        except Exception as exc:
            print(f"[WARN] failed to download {arxiv_id}: {exc}", file=sys.stderr)
            time.sleep(delay)
            continue

        if changed:
            count += 1
            print(f"Downloaded {count}: {rec.title} ({arxiv_id})")
            time.sleep(delay)
        else:
            skipped_existing += 1

    print(
        f"Done. Downloaded {count} new PDFs"
        + (f", skipped {skipped_existing} existing files." if skipped_existing else ".")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ArXiv PDFs based on the local metadata snapshot."
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=100,
        help="Maximum number of new PDFs to download (default: 100).",
    )
    parser.add_argument(
        "--category-prefix",
        type=str,
        default="",
        help="Optional arXiv category prefix to filter by (e.g. 'cs.', 'cs.LG').",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay in seconds between downloads (default: 3.0).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_PDF_DIR),
        help=f"Output directory for PDFs (default: {DEFAULT_PDF_DIR}).",
    )

    args = parser.parse_args()
    main(
        max_papers=max(0, int(args.max_papers or 0)),
        category_prefix=args.category_prefix,
        delay=float(args.delay or DEFAULT_DELAY),
        out_dir=Path(args.out_dir),
    )
