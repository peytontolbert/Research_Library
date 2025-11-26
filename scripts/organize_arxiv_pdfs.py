#!/usr/bin/env python3
from __future__ import annotations

"""
Organize locally downloaded ArXiv PDFs under /arxiv/pdfs into YYMM subdirectories.

- Assumes files are named like "{arxiv_id}.pdf", where recent-style IDs look like:
  "2101.00001", "1704.12345v2", etc.
- For PDFs in the root of /arxiv/pdfs (not already in a subdirectory) whose
  filename starts with 4 digits, we:
    - Take the first 4 digits as YYMM (e.g. "2101" -> /arxiv/pdfs/2101/),
    - Move the PDF into that directory.
- If the destination file already exists, we treat the root-level file as a
  duplicate and delete it.

Safe to re-run: once all matching PDFs have been moved into subdirectories,
subsequent runs will be effectively no-ops.
"""

import os
from pathlib import Path


PDF_ROOT = Path("/arxiv/pdfs")


def is_yymm_style_pdf(path: Path) -> bool:
    """Return True if this looks like a YYMM-style arxiv id PDF."""
    if not path.is_file() or path.suffix.lower() != ".pdf":
        return False
    stem = path.stem  # e.g. "2101.12345v2"
    return len(stem) >= 4 and stem[:4].isdigit()


def organize_pdfs(root: Path) -> None:
    if not root.is_dir():
        print(f"[ERROR] PDF root {root} does not exist or is not a directory.")
        return

    moved = 0
    deleted = 0
    skipped = 0

    for entry in root.iterdir():
        # Only consider files directly in the root (not subdirectories).
        if entry.is_dir():
            continue
        if not is_yymm_style_pdf(entry):
            skipped += 1
            continue

        stem = entry.stem
        yymm = stem[:4]
        dest_dir = root / yymm
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / entry.name

        if dest_path.exists():
            # Destination already has this PDF; treat the root-level copy as duplicate.
            entry.unlink()
            deleted += 1
            continue

        entry.rename(dest_path)
        moved += 1

    print(
        f"Done organizing PDFs under {root}.\n"
        f"  Moved: {moved}\n"
        f"  Deleted (duplicates): {deleted}\n"
        f"  Skipped (non-YYMM-style or already in subdirs): {skipped}"
    )


if __name__ == "__main__":
    organize_pdfs(PDF_ROOT)


