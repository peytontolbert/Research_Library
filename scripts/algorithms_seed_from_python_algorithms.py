#!/usr/bin/env python3
from __future__ import annotations

"""
Seed and update `/data/algorithms/algorithms.jsonl` and `problems.jsonl`
from the local `python_algorithms` repository.

This script parses `DIRECTORY.md` under:

    /data/repository_library/python_algorithms

and:

- Ensures that each listed algorithm has a corresponding Algorithm entry
  in `algorithms.jsonl` (creating new ones when needed).
- Enriches existing Algorithm entries with additional `names` and `topics`.
- Optionally creates placeholder Problem entries for algorithms that do
  not yet reference any `problem_id`.

The intent is to *bootstrap* the ontology from a large algorithm zoo
repository (e.g. a clone of `TheAlgorithms/Python`), while keeping the
seeded `algo_id` and `problem_id` values canonical and stable.

Usage (from project root):

    python scripts/algorithms_seed_from_python_algorithms.py \
        --repo-root /data/repository_library/python_algorithms \
        --algorithms-path /data/algorithms/algorithms.jsonl \
        --problems-path /data/algorithms/problems.jsonl

By default, the paths above are used if the corresponding flags are
omitted.
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
    PROBLEMS_PATH,
)


DEFAULT_REPO_ROOT = Path("/data/repository_library/python_algorithms")


def _load_jsonl_by_key(path: Path, key_field: str) -> Dict[str, Dict[str, object]]:
    """
    Load a JSONL file into a dict keyed by `key_field`.
    """
    out: Dict[str, Dict[str, object]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
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
            key = str(obj_any.get(key_field) or "").strip()
            if not key:
                continue
            out[key] = obj_any
    return out


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


# Overrides to connect common names back to canonical algo_ids from seeds.
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
    """
    norm = _normalize_id(display_name)
    # First, explicit overrides.
    if norm in _NAME_TO_ALGO_OVERRIDES:
        return _NAME_TO_ALGO_OVERRIDES[norm], norm
    # Then, any existing algorithm that already has this name.
    existing = existing_name_index.get(norm)
    if existing:
        return existing, norm
    # Otherwise, use the normalized name as a new algo_id.
    return norm, norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Seed and update /data/algorithms/algorithms.jsonl and problems.jsonl "
            "from the local python_algorithms repository (DIRECTORY.md)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(DEFAULT_REPO_ROOT),
        help="Root of the python_algorithms repository "
        "(default: /data/repository_library/python_algorithms).",
    )
    parser.add_argument(
        "--algorithms-path",
        type=str,
        default=str(ALGORITHMS_PATH),
        help=f"Path to algorithms.jsonl (default: {ALGORITHMS_PATH}).",
    )
    parser.add_argument(
        "--problems-path",
        type=str,
        default=str(PROBLEMS_PATH),
        help=f"Path to problems.jsonl (default: {PROBLEMS_PATH}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write any files; just print a summary.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    algorithms_path = Path(args.algorithms_path)
    problems_path = Path(args.problems_path)

    directory_md = repo_root / "DIRECTORY.md"
    if not directory_md.is_file():
        raise SystemExit(f"DIRECTORY.md not found under {repo_root}")

    # Load existing ontology.
    algorithms = _load_jsonl_by_key(algorithms_path, "algo_id")
    problems = _load_jsonl_by_key(problems_path, "problem_id")

    existing_name_index = _build_existing_name_index(algorithms)

    entries = _parse_directory_md(directory_md)
    if not entries:
        raise SystemExit(f"No entries parsed from {directory_md}")

    # Track a primary display name per algo_id for problem descriptions.
    primary_display_name: Dict[str, str] = {}

    added_algorithms = 0
    updated_algorithms = 0
    added_problems = 0

    for display_name, rel_path, category in entries:
        algo_id, norm_name = _choose_algo_id(display_name, existing_name_index)
        cat_slug = _normalize_id(category) if category else ""

        algo = algorithms.get(algo_id)
        if algo is None:
            # Create a new Algorithm entry with minimal metadata.
            topics: List[str] = []
            if cat_slug:
                topics.append(cat_slug)
            algo = {
                "algo_id": algo_id,
                "names": [display_name],
                "category": cat_slug,
                "problems": [],
                "time_complexity": {},
                "space_complexity": {},
                "properties": {},
                "constraints": {},
                "notes": (
                    "Auto-imported from python_algorithms DIRECTORY.md; "
                    "fill in complexity, properties, and constraints manually."
                ),
                "topics": topics,
                "tags": ["auto_imported_from_python_algorithms"],
            }
            algorithms[algo_id] = algo
            added_algorithms += 1
            # Keep the primary display name for later problem seeding.
            primary_display_name.setdefault(algo_id, display_name)
        else:
            changed = False
            names_any = algo.get("names") or []
            if isinstance(names_any, list):
                if display_name not in names_any:
                    names_any.append(display_name)
                    algo["names"] = names_any
                    changed = True
            # Extend topics with the category slug.
            if cat_slug:
                topics_any = algo.get("topics") or []
                if isinstance(topics_any, list):
                    if cat_slug not in topics_any:
                        topics_any.append(cat_slug)
                        algo["topics"] = topics_any
                        changed = True
            if changed:
                updated_algorithms += 1
            # Do not overwrite an existing primary name.
            primary_display_name.setdefault(algo_id, display_name)

        # Refresh the name index so that subsequent entries can map to this algo_id.
        existing_name_index.setdefault(norm_name, algo_id)

    # Seed simple placeholder problems for algorithms that do not yet
    # reference any `problem_id`. This is intentionally conservative and
    # meant as a bootstrap; ontology curation can refine/replace these.
    for algo_id, algo in algorithms.items():
        problems_any = algo.get("problems")
        if isinstance(problems_any, list) and problems_any:
            continue

        problem_id = f"{algo_id}_problem"
        if problem_id in problems:
            # Link existing problem to this algorithm if not already present.
            if isinstance(problems_any, list):
                if problem_id not in problems_any:
                    problems_any.append(problem_id)
                    algo["problems"] = problems_any
            continue

        display = primary_display_name.get(algo_id) or algo_id
        topics_any = algo.get("topics") or []
        problem = {
            "problem_id": problem_id,
            "names": [f"{display} problem"],
            "description": (
                f"Auto-generated placeholder problem for algorithm '{display}'. "
                "Refine this description and constraints manually."
            ),
            "topics": topics_any if isinstance(topics_any, list) else [],
            "constraints": {},
            "notes": "Auto-imported from python_algorithms; this is a placeholder.",
        }
        problems[problem_id] = problem
        # Attach to the algorithm.
        if isinstance(problems_any, list):
            problems_any.append(problem_id)
            algo["problems"] = problems_any
        else:
            algo["problems"] = [problem_id]
        added_problems += 1

    if args.dry_run:
        print(
            f"[DRY RUN] Parsed {len(entries)} entries from DIRECTORY.md. "
            f"Algorithms: {len(algorithms)} total "
            f"({added_algorithms} added, {updated_algorithms} updated). "
            f"Problems: {len(problems)} total ({added_problems} added)."
        )
        return

    # Persist updates.
    _write_jsonl(algorithms_path, list(algorithms.values()))
    _write_jsonl(problems_path, list(problems.values()))

    print(
        f"Done. Parsed {len(entries)} entries from {directory_md}.\n"
        f"- Algorithms: {len(algorithms)} total "
        f"({added_algorithms} added, {updated_algorithms} updated).\n"
        f"- Problems: {len(problems)} total ({added_problems} added)."
    )


if __name__ == "__main__":
    main()


