from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List


DEFAULT_LIBRARY_ROOT = "/data/repositories"


@dataclass(frozen=True)
class RepoInfo:
    """Lightweight description of a discovered repository root."""

    repo_id: str
    root: str


def _looks_like_repo(path: str) -> bool:
    """Heuristics to decide if a directory is a repository root."""
    if not os.path.isdir(path):
        return False
    # Common signals of a project root
    markers = [
        ".git",
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "environment.yml",
        "Pipfile",
        "src",
    ]
    for m in markers:
        if os.path.exists(os.path.join(path, m)):
            return True
    return False


def discover_repositories(root: str = DEFAULT_LIBRARY_ROOT) -> List[RepoInfo]:
    """
    Recursively scan `root` for plausible repository roots.

    This is intentionally simple and file-system based; the caller can
    decide how to persist or filter the results.
    """
    root = os.path.abspath(root)
    repos: List[RepoInfo] = []
    seen_roots: set[str] = set()

    for dirpath, dirnames, _filenames in os.walk(root):
        # If this directory looks like a repo, register it and do not
        # descend into its children (treat nested structure as one repo).
        if _looks_like_repo(dirpath):
            norm = os.path.abspath(dirpath)
            if norm not in seen_roots:
                seen_roots.add(norm)
                repo_id = os.path.basename(norm) or "repo"
                repos.append(RepoInfo(repo_id=repo_id, root=norm))
            # Prevent descending into subdirectories of this repo
            dirnames[:] = []
            continue

    return sorted(repos, key=lambda r: r.repo_id)


def iter_repositories(root: str = DEFAULT_LIBRARY_ROOT) -> Iterable[RepoInfo]:
    """Generator wrapper around `discover_repositories`."""
    for r in discover_repositories(root=root):
        yield r


def main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Discover repository roots under a library directory."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_LIBRARY_ROOT,
        help="Root directory that contains individual repositories (default: /data/repositories).",
    )
    args = parser.parse_args()

    repos = discover_repositories(root=args.root)
    out = [
        {
            "repo_id": r.repo_id,
            "root": r.root,
        }
        for r in repos
    ]
    json.dump(out, sys.stdout, indent=2)
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()


