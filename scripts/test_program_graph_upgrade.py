from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set

from .library_repo_scanner import DEFAULT_LIBRARY_ROOT, RepoInfo, discover_repositories
from .python_repo_graph import PythonRepoGraph


def _summarize_repo(repo: RepoInfo) -> Dict[str, Any]:
    """
    Build an in-memory PythonRepoGraph for `repo` and return a summary.

    This is a **dry run**: it does not write any JSONL exports or update
    the export manifest. It is intended to validate the upgraded graph
    behavior (file entities, language tags, entity kinds) over a subset
    of repositories.
    """
    graph = PythonRepoGraph(repo.root)

    entities = list(graph.entities())
    edges = list(graph.edges())
    artifacts = list(graph.artifacts(kind="source"))

    kind_counts: Counter[str] = Counter()
    languages: Set[str] = set()

    for e in entities:
        kind = str(getattr(e, "kind", "") or "")
        kind_counts[kind] += 1
        if kind == "file":
            labels = getattr(e, "labels", None) or []
            if isinstance(labels, list):
                for lab in labels:
                    if isinstance(lab, str) and lab.startswith("lang:"):
                        languages.add(lab.split("lang:", 1)[-1])

    return {
        "repo_id": repo.repo_id,
        "root": repo.root,
        "entities": len(entities),
        "edges": len(edges),
        "artifacts": len(artifacts),
        "entity_kinds": dict(sorted(kind_counts.items())),
        "languages": sorted(languages),
    }


def _select_repos(
    *,
    library_root: str,
    only_repo_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[RepoInfo]:
    repos = discover_repositories(root=library_root)
    if only_repo_ids:
        wanted = {r for r in only_repo_ids}
        repos = [r for r in repos if r.repo_id in wanted]
    if limit is not None and limit >= 0:
        repos = repos[: int(limit)]
    return repos


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run tester for the upgraded program graph.\n\n"
            "Builds in-memory graphs for a subset of repositories and prints\n"
            "per-repo summaries (entity kind counts, inferred languages, etc.)\n"
            "without touching the export manifest or JSONL files."
        )
    )
    parser.add_argument(
        "--library-root",
        type=str,
        default=DEFAULT_LIBRARY_ROOT,
        help=(
            "Root directory that contains individual repositories "
            f"(default: {DEFAULT_LIBRARY_ROOT})."
        ),
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        help=(
            "Limit the dry run to one or more specific repo ids. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "Maximum number of repositories to include in the dry run "
            "(default: 5). Ignored if --repo-id is provided."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON summaries instead of compact JSON.",
    )
    args = parser.parse_args()

    library_root = os.path.abspath(args.library_root)
    repos = _select_repos(
        library_root=library_root,
        only_repo_ids=args.repo_ids,
        limit=None if args.repo_ids else args.limit,
    )

    summaries: List[Dict[str, Any]] = []
    for repo in repos:
        try:
            summaries.append(_summarize_repo(repo))
        except Exception as exc:  # pragma: no cover - diagnostic helper
            summaries.append(
                {
                    "repo_id": repo.repo_id,
                    "root": repo.root,
                    "error": str(exc),
                }
            )

    if args.pretty:
        print(json.dumps(summaries, indent=2))
    else:
        print(json.dumps(summaries))


if __name__ == "__main__":
    main()


