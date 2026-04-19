#!/usr/bin/env python3
from __future__ import annotations

"""
Count how many arXiv metadata records match domain groupings.

This is a quick planning/estimation tool before attempting multi-terabyte
downloads.

Example:
  ./scripts/arxiv_domain_counts.py --domains cs,math,physics,chem
"""

import argparse
from typing import Dict, Iterable, Optional, Tuple

from modules.arxiv_categories import category_matches_any_prefix, prefixes_for_domains
from modules.arxiv_library import iter_metadata


def _extract_year(raw: dict) -> Optional[int]:
    upd = str(raw.get("update_date", "")).strip()
    if len(upd) >= 4 and upd[:4].isdigit():
        return int(upd[:4])
    versions = raw.get("versions")
    if isinstance(versions, list) and versions:
        created = str(versions[0].get("created", "")).strip()
        if len(created) >= 4 and created[:4].isdigit():
            return int(created[:4])
    return None


def _id_style(arxiv_id: str) -> str:
    s = (arxiv_id or "").strip()
    if "/" in s:
        return "old"
    # "new" covers 0704.0001+ style; "weird" for anything else.
    if len(s) >= 4 and s[:4].isdigit() and "." in s:
        return "new"
    return "weird"


def _iter_domains(domains_csv: str) -> Iterable[str]:
    for part in (domains_csv or "").replace(",", " ").split():
        part = part.strip().lower()
        if part:
            yield part


def main(*, domains: str, min_year: int, max_year: int) -> None:
    domain_list = list(_iter_domains(domains))
    if not domain_list:
        domain_list = ["cs", "math", "physics", "chem"]
    prefixes = prefixes_for_domains(domain_list)

    totals: Dict[str, int] = {"all": 0, "matched_any": 0, "new": 0, "old": 0, "weird": 0}
    matched_styles: Dict[str, int] = {"new": 0, "old": 0, "weird": 0}

    for rec in iter_metadata():
        totals["all"] += 1
        raw = rec.raw if isinstance(rec.raw, dict) else {}
        if min_year or max_year:
            y = _extract_year(raw) if raw else None
            if y is None:
                continue
            if min_year and y < min_year:
                continue
            if max_year and y > max_year:
                continue

        style = _id_style(rec.id)
        totals[style] = totals.get(style, 0) + 1

        if category_matches_any_prefix(rec.categories, prefixes):
            totals["matched_any"] += 1
            matched_styles[style] = matched_styles.get(style, 0) + 1

    span = ""
    if min_year or max_year:
        span = f" (year filter: {min_year or '...'}–{max_year or '...'})"

    print(f"Domains: {', '.join(domain_list)}")
    print(f"Category prefixes: {', '.join(prefixes)}")
    print(f"Total records scanned{span}: {totals['all']}")
    print(f"Matched any domain: {totals['matched_any']}")
    print(
        "ID style among matches: "
        + ", ".join(f"{k}={matched_styles.get(k,0)}" for k in ("new", "old", "weird"))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count arXiv records by domain groupings.")
    parser.add_argument(
        "--domains",
        type=str,
        default="cs,math,physics,chem",
        help="Comma/space separated list: cs, math, physics, chem (default: all four).",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=0,
        help="Optional year lower bound (0 = none).",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=0,
        help="Optional year upper bound (0 = none).",
    )
    args = parser.parse_args()
    main(domains=args.domains, min_year=int(args.min_year or 0), max_year=int(args.max_year or 0))


