#!/usr/bin/env python3
from __future__ import annotations

"""
Validate the DSA-style Algorithms Library JSONL files.

This script:

- Loads `algorithms.jsonl` and `problems.jsonl` using `modules.dsa_library`.
- Reports schema validation errors (via Pydantic).
- Checks for dangling references between algorithms and problems.

Usage (from project root):

    python scripts/validate_dsa_library.py \
        --algorithms /data/algorithms/algorithms.jsonl \
        --problems /data/algorithms/problems.jsonl
"""

import argparse
from pathlib import Path

from modules.dsa_library import (  # type: ignore
    load_algorithms,
    load_problems,
    validate_references,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate DSA-style algorithms.jsonl and problems.jsonl."
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="/data/algorithms/algorithms.jsonl",
        help="Path to algorithms.jsonl (default: /data/algorithms/algorithms.jsonl).",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default="/data/algorithms/problems.jsonl",
        help="Path to problems.jsonl (default: /data/algorithms/problems.jsonl).",
    )
    args = parser.parse_args()

    algo_path = Path(args.algorithms)
    prob_path = Path(args.problems)

    algos, algo_errors = load_algorithms(algo_path)
    probs, prob_errors = load_problems(prob_path)
    ref_errors = validate_references(algos, probs)

    total_errors = len(algo_errors) + len(prob_errors) + len(ref_errors)

    if algo_errors:
        print("Algorithm schema errors:")
        for msg in algo_errors:
            print("  -", msg)
    if prob_errors:
        print("Problem schema errors:")
        for msg in prob_errors:
            print("  -", msg)
    if ref_errors:
        print("Reference integrity errors:")
        for msg in ref_errors:
            print("  -", msg)

    print()
    print(f"Loaded {len(algos)} algorithms and {len(probs)} problems from:")
    print(f"  algorithms: {algo_path}")
    print(f"  problems:   {prob_path}")
    print(f"Total validation errors: {total_errors}")

    if total_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


