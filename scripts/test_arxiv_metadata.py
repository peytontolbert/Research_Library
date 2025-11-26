from __future__ import annotations

"""
Quick sanity check for the local ArXiv metadata snapshot.

Usage (from project root):

    python -m scripts.test_arxiv_metadata

This will:
- Verify that `/data/arxiv/arxiv-metadata-oai-snapshot.json` exists.
- Stream a small number of records via `modules.arxiv_library`.
- Print a short summary and one example record.
"""

from pathlib import Path
from typing import Any, Dict

from modules.arxiv_library import ARXIV_METADATA_PATH, ArxivRecord, iter_metadata  # type: ignore


def _record_to_brief(rec: ArxivRecord) -> Dict[str, Any]:
    return {
        "id": rec.id,
        "title": rec.title[:160] + ("..." if len(rec.title) > 160 else ""),
        "authors": rec.authors[:160] + ("..." if len(rec.authors) > 160 else ""),
        "categories": rec.categories,
        "abstract": rec.abstract[:200] + ("..." if len(rec.abstract) > 200 else ""),
        "raw_keys": sorted(list(rec.raw.keys())),
    }


def main() -> None:
    path = ARXIV_METADATA_PATH
    print(f"ArXiv metadata path: {path}")
    if not Path(path).is_file():
        print("ERROR: metadata file does not exist. Did you download the Kaggle snapshot to /data/arxiv?")
        return

    total = 0
    example: ArxivRecord | None = None
    for rec in iter_metadata():
        total += 1
        if example is None:
            example = rec
        if total >= 1000:
            break

    print(f"Read {total} metadata records (up to 1000) without JSON errors.")
    if example is not None:
        print("Example record:")
        print(_record_to_brief(example))
    else:
        print("WARNING: no valid records were found; file may be empty or malformed.")


if __name__ == "__main__":
    main()


