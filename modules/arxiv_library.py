from __future__ import annotations

"""
Local ArXiv metadata utilities.

This module provides a small, streaming interface over the local ArXiv
metadata snapshot under `/data/arxiv`, so that the server can expose
simple search APIs without depending on external services.

See `docs/arxiv_metadata.md` for details about the expected layout.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


ARXIV_ROOT = Path("/data/arxiv")
ARXIV_METADATA_PATH = ARXIV_ROOT / "arxiv-metadata-oai-snapshot.json"


@dataclass
class ArxivRecord:
    """
    Lightweight view over a single ArXiv metadata record.
    """

    id: str
    title: str
    abstract: str
    authors: str
    categories: str
    raw: Dict[str, object]


def _to_record(obj: Dict[str, object]) -> Optional[ArxivRecord]:
    # Field names are based on the Cornell snapshot; we default missing
    # fields to empty strings so that search remains robust.
    pid = str(obj.get("id") or "").strip()
    if not pid:
        return None
    title = str(obj.get("title") or "").strip()
    abstract = str(obj.get("abstract") or "").strip()
    authors = str(obj.get("authors") or "").strip()
    categories = str(obj.get("categories") or "").strip()
    return ArxivRecord(
        id=pid,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        raw=obj,
    )


def iter_metadata() -> Iterator[ArxivRecord]:
    """
    Stream ArxivRecord objects from the local metadata snapshot.

    This reads `arxiv-metadata-oai-snapshot.json` line by line and
    yields well-formed records. It skips empty/invalid lines.
    """
    if not ARXIV_METADATA_PATH.is_file():
        return iter(())

    def _gen() -> Iterator[ArxivRecord]:
        with ARXIV_METADATA_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj_any = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj_any, dict):
                    continue
                rec = _to_record(obj_any)
                if rec is not None:
                    yield rec

    return _gen()


def search_keyword(
    query: str,
    *,
    max_results: int = 50,
    fields: Optional[Iterable[str]] = None,
    category_prefix: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    Simple keyword search over the local Arxiv metadata snapshot.

    Args:
        query: Case-insensitive keyword to search for.
        max_results: Maximum number of results to return.
        fields: Optional iterable of fields to search within
                (subset of: "title", "abstract", "authors").
                Defaults to title+abstract.
        category_prefix: Optional ArXiv category prefix (e.g. "cs.",
                "cs.CL") to filter by.

    Returns:
        A list of JSON-serializable dicts with the most relevant fields
        for downstream display or further processing.
    """
    q = str(query or "").strip().lower()
    if not q:
        return []

    fields_list = [f.lower() for f in (fields or ["title", "abstract"])]
    want_title = "title" in fields_list
    want_abstract = "abstract" in fields_list
    want_authors = "authors" in fields_list

    out: List[Dict[str, object]] = []
    cpref = str(category_prefix or "").strip()

    for rec in iter_metadata():
        if cpref and not rec.categories.startswith(cpref):
            continue

        haystack_parts: List[str] = []
        if want_title and rec.title:
            haystack_parts.append(rec.title)
        if want_abstract and rec.abstract:
            haystack_parts.append(rec.abstract)
        if want_authors and rec.authors:
            haystack_parts.append(rec.authors)

        if not haystack_parts:
            continue

        haystack = " ".join(haystack_parts).lower()
        if q not in haystack:
            continue

        out.append(
            {
                "id": rec.id,
                "title": rec.title,
                "abstract": rec.abstract,
                "authors": rec.authors,
                "categories": rec.categories,
            }
        )

        if len(out) >= max_results:
            break

    return out



