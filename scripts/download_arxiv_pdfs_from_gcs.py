#!/usr/bin/env python3
from __future__ import annotations

"""
Bulk-download ArXiv PDFs from the public GCS bucket using local metadata.

This script:

- Streams the local Cornell metadata snapshot from `/data/arxiv`.
- Filters records by:
  - category prefix (e.g. `cs.`),
  - year range (e.g. 2017–present),
- Constructs the corresponding GCS paths under `gs://arxiv-dataset/arxiv/arxiv/pdf/`,
- Invokes `gsutil -m cp -I DEST` to download only the matching PDFs.

Requirements:
- `modules.arxiv_library.iter_metadata` must be able to see
  `/data/arxiv/arxiv-metadata-oai-snapshot.json`.
- `gsutil` must be installed and authenticated on this machine.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Ensure we can import the local `modules` package when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.arxiv_library import iter_metadata  # type: ignore


ARXIV_ROOT = Path("/data/arxiv")
DEFAULT_OUT_DIR = Path("/arxiv/pdfs")
DEFAULT_GCS_PREFIX = "gs://arxiv-dataset/arxiv/arxiv/pdf"


def _extract_year(raw: dict) -> Optional[int]:
    """
    Best-effort extraction of a year from a Cornell arXiv metadata record.

    Prefers `update_date` if present, otherwise falls back to the first
    version's `created` field (which is typically `YYYY-MM-DD ...`).
    """
    upd = str(raw.get("update_date", "")).strip()
    if len(upd) >= 4 and upd[:4].isdigit():
        return int(upd[:4])

    versions = raw.get("versions")
    if isinstance(versions, list) and versions:
        created = str(versions[0].get("created", "")).strip()
        if len(created) >= 4 and created[:4].isdigit():
            return int(created[:4])

    return None


def _iter_filtered_ids(
    *,
    category_prefix: str,
    min_year: int,
    max_year: int,
    keywords: Sequence[str],
) -> Iterable[Tuple[str, dict]]:
    """
    Yield (arxiv_id_without_version, raw_record) for records matching filters.

    Filters applied:
    - Category prefix (e.g. 'cs.').
    - Year range [min_year, max_year] (0 = no bound).
    - Any of the `keywords` appearing (case-insensitive) in title or abstract.
    """
    cpref = category_prefix.strip()
    min_year = int(min_year or 0)
    max_year = int(max_year or 0)
    # Normalise keywords once
    kw = [k.strip().lower() for k in keywords if k.strip()]

    for rec in iter_metadata():
        # Category filter
        if cpref and not rec.categories.startswith(cpref):
            continue

        raw = rec.raw if isinstance(rec.raw, dict) else {}

        # Year filter
        if min_year or max_year:
            year = _extract_year(raw) if raw else None
            if year is None:
                continue
            if min_year and year < min_year:
                continue
            if max_year and year > max_year:
                continue

        # Keyword filter over title + abstract
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

        # Normalise ID: strip version suffix (vN)
        rid = str(rec.id or "").strip()
        if not rid:
            continue
        base_id = rid.split("v", 1)[0]

        # We only expect YYMM.xxxx style IDs in 2017+; skip weird ones defensively.
        if len(base_id) < 4 or not base_id[:4].isdigit():
            continue

        yield base_id, raw


def _build_gcs_and_local_paths(
    *,
    ids: Iterable[str],
    gcs_prefix: str,
    out_dir: Path,
) -> List[str]:
    """
    Given base arxiv IDs, build the list of GCS URLs to download.

    Layout assumptions:
    - Remote: {gcs_prefix}/{yymm}/{base_id}.pdf
    - Local:  {out_dir}/{base_id}.pdf
    """
    urls: List[str] = []
    prefix = gcs_prefix.rstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)

    for base_id in ids:
        yymm = base_id[:4]
        local_path = out_dir / f"{base_id}.pdf"
        if local_path.exists():
            continue
        urls.append(f"{prefix}/{yymm}/{base_id}.pdf")

    return urls


def _run_gsutil_cp(urls: List[str], dest_dir: Path) -> int:
    """
    Run `gsutil -m cp -I DEST` with the given URLs.

    Returns the subprocess return code.
    """
    if not urls:
        print("No new PDFs to download (all already present locally).")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "-m", "cp", "-I", str(dest_dir)]
    print(f"Running: {' '.join(cmd)}")
    print(f"Total PDFs to download: {len(urls)}")

    proc = subprocess.run(
        cmd,
        input=("\n".join(urls) + "\n").encode("utf-8"),
    )
    return proc.returncode


def main(
    *,
    category_prefix: str,
    min_year: int,
    max_year: int,
    keywords: Sequence[str],
    out_dir: Path,
    gcs_prefix: str,
    max_papers: int,
    dry_run: bool,
    paths_file: Optional[Path],
) -> None:
    if not ARXIV_ROOT.exists():
        print(f"[ERROR] Expected {ARXIV_ROOT} to exist with metadata.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Filtering metadata with category_prefix='{category_prefix}', "
        f"min_year={min_year or 'none'}, max_year={max_year or 'none'}, "
        f"keywords={', '.join(keywords) if keywords else 'none'}"
    )

    # Collect matching IDs, optionally capped by max_papers.
    ids: List[str] = []
    for base_id, _raw in _iter_filtered_ids(
        category_prefix=category_prefix,
        min_year=min_year,
        max_year=max_year,
        keywords=keywords,
    ):
        ids.append(base_id)
        if max_papers and len(ids) >= max_papers:
            break

    if not ids:
        print("No metadata records matched the given filters.")
        return

    print(f"Matched {len(ids)} metadata records before local-file check.")

    urls = _build_gcs_and_local_paths(ids=ids, gcs_prefix=gcs_prefix, out_dir=out_dir)
    print(f"{len(urls)} PDFs are not yet present locally and will be downloaded.")

    # Optionally write paths to a file for inspection/reuse.
    if paths_file is not None:
        paths_file.parent.mkdir(parents=True, exist_ok=True)
        paths_file.write_text("\n".join(urls) + "\n", encoding="utf-8")
        print(f"Wrote GCS paths to {paths_file}")

    if dry_run:
        print("Dry run enabled; not invoking gsutil.")
        return

    rc = _run_gsutil_cp(urls, dest_dir=out_dir)
    if rc != 0:
        print(f"[ERROR] gsutil exited with code {rc}", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk-download filtered ArXiv PDFs from GCS using local metadata.",
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
        help="Only download papers with year >= this value. Default: 2017.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=0,
        help="Only download papers with year <= this value (0 = no upper bound).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_GCS_PREFIX),
        help=f"Local directory for PDFs. Default: {DEFAULT_GCS_PREFIX}",
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
        default="",
        help=(
            "Comma- or space-separated keywords to match in title/abstract "
            "(case-insensitive, any-match). Example: 'transformer,llm,agent'."
        ),
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Maximum number of matching PDFs to consider (0 = no limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call gsutil; just print counts and optionally write the paths file.",
    )
    parser.add_argument(
        "--paths-file",
        type=str,
        default="",
        help="Optional path to write the list of GCS URLs that would be downloaded.",
    )

    args = parser.parse_args()
    paths_file: Optional[Path] = Path(args.paths_file) if args.paths_file else None

    # Support both comma- and space-separated keywords.
    raw_kw = args.keywords.replace(",", " ").split()

    main(
        category_prefix=args.category_prefix,
        min_year=args.min_year,
        max_year=args.max_year,
        keywords=raw_kw,
        out_dir=Path(args.out_dir),
        gcs_prefix=args.gcs_prefix,
        max_papers=max(0, int(args.max_papers or 0)),
        dry_run=bool(args.dry_run),
        paths_file=paths_file,
    )


