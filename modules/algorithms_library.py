from __future__ import annotations

"""
Algorithms Library utilities.

This module provides a small, streaming interface over the local
Algorithms Library snapshot under `/data/algorithms`, so that the
server can expose simple listing/search APIs without depending on
external services or a graph database.

See `docs/algorithms_library.md` for the on-disk layout and schemas.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


ALGORITHMS_ROOT = Path("/data/algorithms")
ALGORITHMS_PATH = ALGORITHMS_ROOT / "algorithms.jsonl"
PROBLEMS_PATH = ALGORITHMS_ROOT / "problems.jsonl"
IMPLEMENTATIONS_PATH = ALGORITHMS_ROOT / "implementations.jsonl"
BENCHMARKS_PATH = ALGORITHMS_ROOT / "benchmarks.jsonl"


@dataclass
class Algorithm:
    """
    Lightweight view over a single Algorithm node.

    Fields mirror the JSONL schema but are intentionally permissive:
    unknown fields are kept in `raw` for forward compatibility.
    """

    algo_id: str
    names: List[str]
    category: str
    problems: List[str]
    topics: List[str]
    time_complexity: Dict[str, object]
    space_complexity: Dict[str, object]
    properties: Dict[str, object]
    constraints: Dict[str, object]
    notes: str
    tags: List[str]
    raw: Dict[str, object]


@dataclass
class Problem:
    """
    Lightweight view over a single Problem node.
    """

    problem_id: str
    names: List[str]
    description: str
    topics: List[str]
    constraints: Dict[str, object]
    notes: str
    raw: Dict[str, object]


@dataclass
class Implementation:
    """
    Thin pointer from an Algorithm to concrete code in a repository.
    """

    impl_id: str
    algo_id: str
    language: str
    repo_id: Optional[str]
    repo_root: Optional[str]
    file_path: Optional[str]
    entry_symbol: Optional[str]
    constraints: Dict[str, object]
    environment: Dict[str, object]
    notes: str
    raw: Dict[str, object]


@dataclass
class Benchmark:
    """
    Benchmark measurements for a specific implementation.
    """

    benchmark_id: str
    impl_id: str
    dataset_id: Optional[str]
    dataset_description: str
    metrics: Dict[str, object]
    environment: Dict[str, object]
    notes: str
    raw: Dict[str, object]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    """
    Stream JSON objects from a JSONL file, skipping invalid lines.
    """
    if not path.is_file():
        return iter(())

    def _gen() -> Iterator[Dict[str, object]]:
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
                yield obj_any

    return _gen()


def _to_algorithm(obj: Dict[str, object]) -> Optional[Algorithm]:
    algo_id = str(obj.get("algo_id") or "").strip()
    if not algo_id:
        return None
    names_any = obj.get("names") or []
    problems_any = obj.get("problems") or []
    topics_any = obj.get("topics") or []
    tags_any = obj.get("tags") or []

    names = [str(x) for x in names_any] if isinstance(names_any, (list, tuple)) else []
    problems = (
        [str(x) for x in problems_any] if isinstance(problems_any, (list, tuple)) else []
    )
    topics = (
        [str(x) for x in topics_any] if isinstance(topics_any, (list, tuple)) else []
    )
    tags = [str(x) for x in tags_any] if isinstance(tags_any, (list, tuple)) else []

    return Algorithm(
        algo_id=algo_id,
        names=names,
        category=str(obj.get("category") or "").strip(),
        problems=problems,
        topics=topics,
        time_complexity=dict(obj.get("time_complexity") or {}),
        space_complexity=dict(obj.get("space_complexity") or {}),
        properties=dict(obj.get("properties") or {}),
        constraints=dict(obj.get("constraints") or {}),
        notes=str(obj.get("notes") or "").strip(),
        tags=tags,
        raw=obj,
    )


def _to_problem(obj: Dict[str, object]) -> Optional[Problem]:
    pid = str(obj.get("problem_id") or "").strip()
    if not pid:
        return None
    names_any = obj.get("names") or []
    topics_any = obj.get("topics") or []

    names = [str(x) for x in names_any] if isinstance(names_any, (list, tuple)) else []
    topics = (
        [str(x) for x in topics_any] if isinstance(topics_any, (list, tuple)) else []
    )

    return Problem(
        problem_id=pid,
        names=names,
        description=str(obj.get("description") or "").strip(),
        topics=topics,
        constraints=dict(obj.get("constraints") or {}),
        notes=str(obj.get("notes") or "").strip(),
        raw=obj,
    )


def _to_implementation(obj: Dict[str, object]) -> Optional[Implementation]:
    impl_id = str(obj.get("impl_id") or "").strip()
    algo_id = str(obj.get("algo_id") or "").strip()
    if not impl_id or not algo_id:
        return None

    return Implementation(
        impl_id=impl_id,
        algo_id=algo_id,
        language=str(obj.get("language") or "").strip(),
        repo_id=(str(obj["repo_id"]).strip() if "repo_id" in obj else None),
        repo_root=(str(obj["repo_root"]).strip() if "repo_root" in obj else None),
        file_path=(str(obj["file_path"]).strip() if "file_path" in obj else None),
        entry_symbol=(
            str(obj["entry_symbol"]).strip() if "entry_symbol" in obj else None
        ),
        constraints=dict(obj.get("constraints") or {}),
        environment=dict(obj.get("environment") or {}),
        notes=str(obj.get("notes") or "").strip(),
        raw=obj,
    )


def _to_benchmark(obj: Dict[str, object]) -> Optional[Benchmark]:
    bid = str(obj.get("benchmark_id") or "").strip()
    impl_id = str(obj.get("impl_id") or "").strip()
    if not bid or not impl_id:
        return None

    return Benchmark(
        benchmark_id=bid,
        impl_id=impl_id,
        dataset_id=(str(obj["dataset_id"]).strip() if "dataset_id" in obj else None),
        dataset_description=str(obj.get("dataset_description") or "").strip(),
        metrics=dict(obj.get("metrics") or {}),
        environment=dict(obj.get("environment") or {}),
        notes=str(obj.get("notes") or "").strip(),
        raw=obj,
    )


def iter_algorithms() -> Iterator[Algorithm]:
    """
    Stream Algorithm objects from the local algorithms snapshot.
    """
    for obj in _iter_jsonl(ALGORITHMS_PATH):
        algo = _to_algorithm(obj)
        if algo is not None:
            yield algo


def iter_problems() -> Iterator[Problem]:
    """
    Stream Problem objects from the local problems snapshot.
    """
    for obj in _iter_jsonl(PROBLEMS_PATH):
        prob = _to_problem(obj)
        if prob is not None:
            yield prob


def iter_implementations() -> Iterator[Implementation]:
    """
    Stream Implementation objects from the local implementations snapshot.
    """
    for obj in _iter_jsonl(IMPLEMENTATIONS_PATH):
        impl = _to_implementation(obj)
        if impl is not None:
            yield impl


def iter_benchmarks() -> Iterator[Benchmark]:
    """
    Stream Benchmark objects from the local benchmarks snapshot.
    """
    for obj in _iter_jsonl(BENCHMARKS_PATH):
        bench = _to_benchmark(obj)
        if bench is not None:
            yield bench


def search_algorithms(
    query: str,
    *,
    problem_id: Optional[str] = None,
    topic: Optional[str] = None,
    max_results: int = 50,
    fields: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    """
    Simple keyword-style search over the Algorithms Library.

    Args:
        query: Case-insensitive keyword to search for.
        problem_id: Optional problem_id to filter algorithms by.
        topic: Optional topic/tag to filter algorithms by (matches topics or tags).
        max_results: Maximum number of algorithms to return.
        fields: Optional iterable of algorithm fields to search within.
                Subset of: "algo_id", "names", "notes", "category".
                Defaults to algo_id + names + notes.

    Returns:
        A list of JSON-serializable dicts with summary information suitable
        for API responses and UI display.
    """
    q = str(query or "").strip().lower()
    if not q and not problem_id and not topic:
        return []

    fields_list = [f.lower() for f in (fields or ["algo_id", "names", "notes"])]
    want_algo_id = "algo_id" in fields_list
    want_names = "names" in fields_list
    want_notes = "notes" in fields_list
    want_category = "category" in fields_list

    pid = str(problem_id or "").strip()
    t = str(topic or "").strip().lower()

    out: List[Dict[str, object]] = []
    for algo in iter_algorithms():
        if pid and pid not in algo.problems:
            continue
        if t:
            topics_lower = [x.lower() for x in (algo.topics or [])]
            tags_lower = [x.lower() for x in (algo.tags or [])]
            if t not in topics_lower and t not in tags_lower:
                continue

        if q:
            haystack_parts: List[str] = []
            if want_algo_id and algo.algo_id:
                haystack_parts.append(algo.algo_id)
            if want_names and algo.names:
                haystack_parts.extend(algo.names)
            if want_notes and algo.notes:
                haystack_parts.append(algo.notes)
            if want_category and algo.category:
                haystack_parts.append(algo.category)

            if not haystack_parts:
                continue

            haystack = " ".join(haystack_parts).lower()
            if q not in haystack:
                continue

        out.append(
            {
                "algo_id": algo.algo_id,
                "names": algo.names,
                "category": algo.category,
                "problems": algo.problems,
                "topics": algo.topics,
                "time_complexity": algo.time_complexity,
                "space_complexity": algo.space_complexity,
                "properties": algo.properties,
                "constraints": algo.constraints,
                "notes": algo.notes,
                "tags": algo.tags,
            }
        )

        if len(out) >= max_results:
            break

    return out



