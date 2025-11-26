#!/usr/bin/env python3
from __future__ import annotations

"""
Ingest concrete Implementations from the local `python_algorithms` repository
into the Algorithms Library (`/data/algorithms/implementations.jsonl`).

This complements `scripts/algorithms_seed_from_python_algorithms.py`, which
bootstraps `algorithms.jsonl` and `problems.jsonl` from `DIRECTORY.md`.

High-level behavior:

- Parse `DIRECTORY.md` under:

      /data/repository_library/python_algorithms

- For each entry that points to a Python file:
  - Map its human-readable name back to a canonical `algo_id` using the same
    normalization + overrides as the seeding script.
  - Create (or update) an `Implementation` row that points from that `algo_id`
    to the concrete file in `python_algorithms`.

- Rewrite `/data/algorithms/implementations.jsonl` with the merged results.

Usage (from project root):

    python scripts/ingest_algorithms_from_python_algorithms.py

Optional flags:

    --repo-root           Root of the python_algorithms repo
    --implementations-path  Path to implementations.jsonl
    --repo-id             Repo id to embed in Implementation rows
    --dry-run             Print summary only; do not write files
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure the project root (which contains the `modules` package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.algorithms_library import (  # type: ignore
    ALGORITHMS_PATH,
    IMPLEMENTATIONS_PATH,
)


DEFAULT_REPO_ROOT = Path("/data/repository_library/python_algorithms")
DEFAULT_REPO_ID = "python_algorithms"


def _load_jsonl_list(path: Path) -> List[Dict[str, object]]:
    """
    Load a JSONL file into a list of dicts.
    """
    rows: List[Dict[str, object]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj_any = json.loads(line)
            except Exception:
                continue
            if isinstance(obj_any, dict):
                rows.append(obj_any)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    """
    Rewrite a JSONL file with the given rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def _normalize_id(name: str) -> str:
    """
    Normalize a human-readable name into a lowercase slug suitable for
    use as an algo_id or problem_id.
    """
    s = name.strip().lower()
    # Drop common punctuation and apostrophes.
    s = re.sub(r"[\"'’]", "", s)
    # Replace non-alphanumeric sequences with underscores.
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # Collapse multiple underscores and trim.
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Same overrides as in `algorithms_seed_from_python_algorithms.py` so that we
# link files to the same canonical algo_ids used during seeding.
_NAME_TO_ALGO_OVERRIDES: Dict[str, str] = {
    # Graph traversal
    "breadth_first_search": "bfs",
    "depth_first_search": "dfs",
    # Dijkstra variants
    "dijkstra": "dijkstra",
    "dijkstras_algorithm": "dijkstra",
    "dijkstra_algorithm": "dijkstra",
    "dijkstra_2": "dijkstra",
    "dijkstra_alternate": "dijkstra",
    "dijkstra_binary_grid": "dijkstra",
    "bi_directional_dijkstra": "dijkstra",
    # Bellman-Ford
    "bellman_ford": "bellman_ford",
    # Sorting
    "quick_sort": "quicksort",
    "merge_sort": "mergesort",
    "heap_sort": "heapsort",
    # Binary search
    "binary_search": "binary_search",
}


def _parse_directory_md(path: Path) -> List[Tuple[str, str, str]]:
    """
    Parse DIRECTORY.md and return a list of (name, file_path, category_name).
    """
    entries: List[Tuple[str, str, str]] = []
    if not path.is_file():
        return entries

    current_category = ""
    pattern = re.compile(r"\*\s+\[(.+?)\]\((.+?)\)")

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            # Category headings: "## Category"
            if line.startswith("## "):
                current_category = line[3:].strip()
                continue
            m = pattern.search(line)
            if not m:
                continue
            name = m.group(1).strip()
            rel_path = m.group(2).strip()
            entries.append((name, rel_path, current_category))
    return entries


def _build_existing_name_index(
    algorithms: Dict[str, Dict[str, object]]
) -> Dict[str, str]:
    """
    Build a map from normalized name -> algo_id using existing Algorithm
    entries' `algo_id` and `names` fields.
    """
    name_to_algo: Dict[str, str] = {}
    for algo_id, obj in algorithms.items():
        name_to_algo[_normalize_id(algo_id)] = algo_id
        names_any = obj.get("names") or []
        if isinstance(names_any, (list, tuple)):
            for n in names_any:
                nn = _normalize_id(str(n))
                name_to_algo.setdefault(nn, algo_id)
    return name_to_algo


def _choose_algo_id(
    display_name: str,
    existing_name_index: Dict[str, str],
) -> Tuple[str, str]:
    """
    Given a human-readable display name, choose a canonical algo_id.

    Returns (algo_id, norm_name) where:
        - algo_id: chosen identifier (may be existing or new).
        - norm_name: normalized form of the display name.

    For ingestion we only ever *attach* to existing algorithms; if the
    normalized name does not map to a known algo_id, we still emit an
    Implementation but keep the normalized name for debugging.
    """
    norm = _normalize_id(display_name)
    # First, explicit overrides.
    if norm in _NAME_TO_ALGO_OVERRIDES:
        return _NAME_TO_ALGO_OVERRIDES[norm], norm
    # Then, any existing algorithm that already has this name.
    existing = existing_name_index.get(norm)
    if existing:
        return existing, norm
    # Otherwise, fall back to the normalized name; this may or may not
    # already exist in algorithms.jsonl.
    return norm, norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Implementation rows for the Algorithms Library from the "
            "local python_algorithms repository (DIRECTORY.md)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(DEFAULT_REPO_ROOT),
        help="Root of the python_algorithms repository "
        f"(default: {DEFAULT_REPO_ROOT}).",
    )
    parser.add_argument(
        "--implementations-path",
        type=str,
        default=str(IMPLEMENTATIONS_PATH),
        help=f"Path to implementations.jsonl (default: {IMPLEMENTATIONS_PATH}).",
    )
    parser.add_argument(
        "--algorithms-path",
        type=str,
        default=str(ALGORITHMS_PATH),
        help=f"Path to algorithms.jsonl (default: {ALGORITHMS_PATH}).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Repo id to embed in Implementation rows (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write any files; just print a summary.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    implementations_path = Path(args.implementations_path)
    algorithms_path = Path(args.algorithms_path)
    repo_id = args.repo_id

    directory_md = repo_root / "DIRECTORY.md"
    if not directory_md.is_file():
        raise SystemExit(f"DIRECTORY.md not found under {repo_root}")

    # Load existing ontology + implementations.
    algorithms_list = _load_jsonl_list(algorithms_path)
    algorithms: Dict[str, Dict[str, object]] = {}
    for obj in algorithms_list:
        algo_id = str(obj.get("algo_id") or "").strip()
        if not algo_id:
            continue
        algorithms[algo_id] = obj

    existing_name_index = _build_existing_name_index(algorithms)

    existing_impls = _load_jsonl_list(implementations_path)
    impls_by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for impl in existing_impls:
        aid = str(impl.get("algo_id") or "").strip()
        rid = str(impl.get("repo_id") or "").strip()
        fpath = str(impl.get("file_path") or "").strip()
        if not aid or not rid or not fpath:
            continue
        impls_by_key[(aid, rid, fpath)] = impl

    entries = _parse_directory_md(directory_md)
    if not entries:
        raise SystemExit(f"No entries parsed from {directory_md}")

    added_impls = 0
    updated_impls = 0

    for display_name, rel_path, _category in entries:
        # Only consider Python source files.
        if not rel_path.endswith(".py"):
            continue

        algo_id, norm_name = _choose_algo_id(display_name, existing_name_index)
        # Skip entries that do not map to any known Algorithm and also do not
        # look like a plausible algo_id; this keeps noise low.
        if algo_id not in algorithms:
            # Still allow cases where the normalized name exactly matches the
            # algo_id in DIRECTORY.md; this is useful for purely new entries.
            if norm_name not in algorithms:
                continue

        # Stable implementation id based on algo_id + repo_id + file_path.
        impl_id = f"{algo_id}_py_{repo_id}_{rel_path}"
        impl_id = _normalize_id(impl_id)

        key = (algo_id, repo_id, rel_path)
        existing = impls_by_key.get(key)
        if existing:
            # Update existing implementation with any missing fields.
            changed = False
            if not existing.get("impl_id"):
                existing["impl_id"] = impl_id
                changed = True
            if not existing.get("language"):
                existing["language"] = "python"
                changed = True
            if not existing.get("repo_root"):
                existing["repo_root"] = str(repo_root)
                changed = True
            notes_any = existing.get("notes") or ""
            if not notes_any:
                existing["notes"] = f"Imported from python_algorithms {rel_path}"
                changed = True
            if changed:
                updated_impls += 1
        else:
            impl = {
                "impl_id": impl_id,
                "algo_id": algo_id,
                "language": "python",
                "repo_id": repo_id,
                "repo_root": str(repo_root),
                "file_path": rel_path,
                "entry_symbol": None,
                "constraints": {},
                "environment": {
                    "python": "3.11",
                },
                "notes": f"Imported from python_algorithms {rel_path}",
            }
            impls_by_key[key] = impl
            added_impls += 1

    out_impls = list(impls_by_key.values())

    if args.dry_run:
        print(
            f"[DRY RUN] Parsed {len(entries)} DIRECTORY entries.\n"
            f"- Algorithms known: {len(algorithms)}\n"
            f"- Implementations total (existing+new): {len(out_impls)}\n"
            f"- Implementations added: {added_impls}, updated: {updated_impls}"
        )
        return

    _write_jsonl(implementations_path, out_impls)

    print(
        f"Done ingesting implementations from {directory_md}.\n"
        f"- Algorithms known: {len(algorithms)}\n"
        f"- Implementations written: {len(out_impls)}\n"
        f"- Implementations added: {added_impls}, updated: {updated_impls}"
    )


if __name__ == "__main__":
    main()


