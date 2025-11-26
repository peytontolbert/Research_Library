from __future__ import annotations

"""
Structured models and loaders for the DSA-style Algorithms Library.

This module implements the richer schema described in the conceptual
model:

- Algorithm (logical concept: "merge sort")
- Implementation (language-specific realization)
- Problem (task that uses one or more algorithms)
- Example (input/output pair for a Problem)

Storage format:

- `algorithms.jsonl` — one Algorithm per line (with nested implementations)
- `problems.jsonl`   — one Problem per line (with nested examples)

The goal is to treat these files as a small knowledge graph serialized
into JSONL, with explicit validation and reference checking.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator


class Implementation(BaseModel):
    """
    Language-specific realization of an Algorithm.

    `source_type` controls where the implementation code lives:
    - "inline": `code` contains the full implementation.
    - "file_path": `file_path` points into a local repo checkout.
    - "url": `file_path` or `code` are omitted; `notes` should contain the URL.
    - "repo_ref": future extension for (repo_id, path, commit) style refs.
    """

    impl_id: str = Field(..., description="Stable identifier for this implementation")
    algorithm_id: Optional[str] = Field(
        default=None,
        description="Back-reference to Algorithm.id; may be omitted when nested.",
    )
    language: str = Field(..., description="Programming language, e.g. 'python'")
    paradigm: Optional[str] = Field(
        default=None,
        description="High-level paradigm, e.g. 'imperative', 'functional', 'oop'",
    )
    source_type: str = Field(
        ...,
        description='One of "inline" | "file_path" | "url" | "repo_ref"',
    )
    code: Optional[str] = Field(
        default=None, description="Inline source code when source_type == 'inline'"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Repo-relative path to the implementation when source_type == 'file_path'",
    )
    repo_id: Optional[str] = Field(
        default=None,
        description="Repository identifier when the implementation lives in a repo",
    )
    entry_point: Optional[str] = Field(
        default=None,
        description="Function or class name that is the recommended entry point",
    )
    is_reference: bool = Field(
        default=False,
        description="True if this is a known-good reference implementation",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Free-form notes, URLs, or caveats about this implementation",
    )

    @field_validator("source_type")
    @classmethod
    def _validate_source_type(cls, v: str) -> str:
        allowed = {"inline", "file_path", "url", "repo_ref"}
        if v not in allowed:
            raise ValueError(f"source_type must be one of {sorted(allowed)}, got {v!r}")
        return v


class TimeComplexity(BaseModel):
    """
    Big-O time complexity summary.

    All fields are free-form strings (e.g. 'O(n log n)', 'O(V + E)').
    """

    best: Optional[str] = None
    average: Optional[str] = None
    worst: Optional[str] = None


class Algorithm(BaseModel):
    """
    Logical algorithm concept (e.g. 'merge sort').
    """

    id: str = Field(..., description="Stable, machine-friendly identifier")
    name: str = Field(..., description="Human-readable primary name")
    category: Optional[str] = Field(
        default=None,
        description="Primary conceptual bucket, e.g. 'sorting', 'graph', 'dp'",
    )
    description: Optional[str] = Field(
        default=None, description="Short natural-language description"
    )
    canonical_pseudocode: Optional[str] = Field(
        default=None,
        description="Language-agnostic pseudocode capturing the essence of the algorithm",
    )
    data_structures: List[str] = Field(
        default_factory=list,
        description="Core data structures used by this algorithm (e.g. 'array', 'heap')",
    )
    time_complexity: Optional[TimeComplexity] = Field(
        default=None, description="Big-O time complexity summary"
    )
    space: Optional[str] = Field(
        default=None,
        description="Space complexity as a free-form string (e.g. 'O(n)', 'O(V)').",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Additional labels (e.g. 'stable', 'in_place', 'classic')",
    )
    related_algorithm_ids: List[str] = Field(
        default_factory=list,
        description="IDs of related algorithms (variants, alternatives, etc.)",
    )
    related_problem_ids: List[str] = Field(
        default_factory=list,
        description="IDs of problems that prominently use this algorithm",
    )
    implementations: List[Implementation] = Field(
        default_factory=list,
        description="Language-specific implementations attached to this algorithm",
    )


class Example(BaseModel):
    """
    Concrete example input/output pair for a problem.
    """

    input: str = Field(..., description="Example input (text or code-like)")
    output: str = Field(..., description="Expected output")
    explanation: Optional[str] = Field(
        default=None, description="Explanation of how input maps to output"
    )


class Problem(BaseModel):
    """
    Problem/task that is solved by one or more algorithms.
    """

    id: str = Field(..., description="Stable, machine-friendly identifier")
    title: str = Field(..., description="Human-readable title")
    difficulty: Optional[str] = Field(
        default=None,
        description='One of "easy" | "medium" | "hard" (convention, not enforced)',
    )
    categories: List[str] = Field(
        default_factory=list,
        description="High-level categories/tags (e.g. 'array', 'hash_map')",
    )
    description: Optional[str] = Field(
        default=None, description="Full problem statement"
    )
    input_format: Optional[str] = Field(
        default=None, description="Description of input types/format"
    )
    output_format: Optional[str] = Field(
        default=None, description="Description of expected output format"
    )
    constraints: Optional[str] = Field(
        default=None,
        description="Text block with constraints (sizes, time limits, etc.)",
    )
    examples: List[Example] = Field(
        default_factory=list,
        description="Example input/output pairs illustrating the problem",
    )
    canonical_algorithm_ids: List[str] = Field(
        default_factory=list,
        description="Recommended algorithms for this problem",
    )
    acceptable_algorithm_ids: List[str] = Field(
        default_factory=list,
        description="Other valid (possibly less optimal) algorithms",
    )
    required_data_structures: List[str] = Field(
        default_factory=list,
        description="Data structures that are expected/required to solve the problem",
    )
    source: Optional[str] = Field(
        default=None,
        description='Origin of the problem, e.g. "custom", "leetcode-1-two-sum"',
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Additional labels (e.g. 'leetcode-style', 'introductory')",
    )


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    """
    Yield raw JSON objects from a JSONL file, skipping invalid lines.
    """
    if not path.is_file():
        return []

    def _gen() -> Iterable[Dict[str, object]]:
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
                    yield obj_any

    return _gen()


def load_algorithms(path: Path) -> Tuple[Dict[str, Algorithm], List[str]]:
    """
    Load Algorithms from a JSONL file, returning:

    - dict[id] -> Algorithm
    - list of validation error messages (if any)
    """
    algos: Dict[str, Algorithm] = {}
    errors: List[str] = []
    for idx, obj in enumerate(_iter_jsonl(path), start=1):
        try:
            algo = Algorithm.model_validate(obj)
        except ValidationError as exc:
            errors.append(f"{path}:{idx}: {exc}")
            continue
        if algo.id in algos:
            errors.append(f"{path}:{idx}: duplicate algorithm id {algo.id!r}")
            continue
        # Ensure nested implementations carry algorithm_id for convenience.
        for impl in algo.implementations:
            if impl.algorithm_id is None:
                impl.algorithm_id = algo.id
        algos[algo.id] = algo
    return algos, errors


def load_problems(path: Path) -> Tuple[Dict[str, Problem], List[str]]:
    """
    Load Problems from a JSONL file, returning:

    - dict[id] -> Problem
    - list of validation error messages (if any)
    """
    problems: Dict[str, Problem] = {}
    errors: List[str] = []
    for idx, obj in enumerate(_iter_jsonl(path), start=1):
        try:
            prob = Problem.model_validate(obj)
        except ValidationError as exc:
            errors.append(f"{path}:{idx}: {exc}")
            continue
        if prob.id in problems:
            errors.append(f"{path}:{idx}: duplicate problem id {prob.id!r}")
            continue
        problems[prob.id] = prob
    return problems, errors


def validate_references(
    algorithms: Dict[str, Algorithm],
    problems: Dict[str, Problem],
) -> List[str]:
    """
    Perform cross-entity validation:

    - All `canonical_algorithm_ids` / `acceptable_algorithm_ids` on Problems
      must exist in `algorithms`.
    - All `related_algorithm_ids` on Algorithms must exist in `algorithms`.
    - All `related_problem_ids` on Algorithms must exist in `problems`.
    - All Implementation.algorithm_id values (if set) must exist in `algorithms`.
    """
    errors: List[str] = []

    algo_ids = set(algorithms.keys())
    problem_ids = set(problems.keys())

    # Problem -> Algorithm references.
    for prob in problems.values():
        for aid in prob.canonical_algorithm_ids:
            if aid not in algo_ids:
                errors.append(
                    f"Problem {prob.id!r} references missing canonical_algorithm_id {aid!r}"
                )
        for aid in prob.acceptable_algorithm_ids:
            if aid not in algo_ids:
                errors.append(
                    f"Problem {prob.id!r} references missing acceptable_algorithm_id {aid!r}"
                )

    # Algorithm -> related Algorithm / Problem references and implementation back-refs.
    for algo in algorithms.values():
        for aid in algo.related_algorithm_ids:
            if aid not in algo_ids:
                errors.append(
                    f"Algorithm {algo.id!r} references missing related_algorithm_id {aid!r}"
                )
        for pid in algo.related_problem_ids:
            if pid not in problem_ids:
                errors.append(
                    f"Algorithm {algo.id!r} references missing related_problem_id {pid!r}"
                )
        for impl in algo.implementations:
            if impl.algorithm_id and impl.algorithm_id not in algo_ids:
                errors.append(
                    f"Implementation {impl.impl_id!r} has unknown algorithm_id {impl.algorithm_id!r}"
                )

    return errors



