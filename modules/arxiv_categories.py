from __future__ import annotations

"""
ArXiv category helpers.

The Cornell/Kaggle metadata snapshot stores categories as a whitespace-separated
string, e.g.:

  "cs.CL cs.LG stat.ML"

Important: the "primary" category is not always first, so callers should not
use `startswith()` on the raw string.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence


def split_categories(categories: str) -> List[str]:
    """Split a metadata `categories` field into individual tags."""
    cats = (categories or "").strip()
    if not cats:
        return []
    return [c for c in cats.split() if c]


def category_matches_any_prefix(categories: str, prefixes: Sequence[str]) -> bool:
    """
    Return True if any category tag matches any prefix.

    Matching is token-based:
    - exact match OR
    - tag.startswith(prefix)
    """
    prefs = [p.strip() for p in prefixes if str(p).strip()]
    if not prefs:
        return True
    tags = split_categories(categories)
    if not tags:
        return False
    for tag in tags:
        for pref in prefs:
            if tag == pref or tag.startswith(pref):
                return True
    return False


@dataclass(frozen=True)
class DomainSpec:
    """
    A convenience mapping from a user-friendly domain name to category prefixes.
    """

    name: str
    prefixes: List[str]


# Practical domain groupings (covers both old-style and new-style category tags).
DOMAIN_SPECS: List[DomainSpec] = [
    DomainSpec(name="cs", prefixes=["cs."]),
    DomainSpec(name="math", prefixes=["math."]),
    # "Physics" spans many legacy primary categories on arXiv.
    DomainSpec(
        name="physics",
        prefixes=[
            "physics.",
            "astro-ph",
            "cond-mat",
            "gr-qc",
            "hep-",
            "math-ph",
            "nlin",
            "nucl-",
            "quant-ph",
            "plasm-ph",
            "acc-phys",
        ],
    ),
    # arXiv "chemistry" is mostly chem-ph (legacy) / physics.chem-ph (new).
    DomainSpec(name="chem", prefixes=["physics.chem-ph", "chem-ph"]),
]


def prefixes_for_domains(domains: Iterable[str]) -> List[str]:
    """
    Expand user-friendly domains into category prefixes.

    Supported domains: cs, math, physics, chem
    """
    wanted = {str(d).strip().lower() for d in domains if str(d).strip()}
    out: List[str] = []
    for spec in DOMAIN_SPECS:
        if spec.name in wanted:
            out.extend(spec.prefixes)
    # Deduplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


