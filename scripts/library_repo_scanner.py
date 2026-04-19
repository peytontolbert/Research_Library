from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


DEFAULT_LIBRARY_ROOT = "/data/repositories"


@dataclass(frozen=True)
class RepoInfo:
    """Lightweight description of a discovered repository root."""

    repo_id: str
    root: str
    library_root: str


def _normalize_roots(
    *,
    root: Optional[str] = None,
    roots: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return an ordered, de-duplicated list of repository library roots."""
    normalized: List[str] = []
    seen: set[str] = set()

    candidates: List[str] = []
    if root:
        candidates.append(root)
    if roots:
        candidates.extend(roots)

    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        abs_root = os.path.abspath(raw)
        if abs_root in seen:
            continue
        seen.add(abs_root)
        normalized.append(abs_root)

    return normalized


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


def discover_repositories(
    root: str = DEFAULT_LIBRARY_ROOT,
    *,
    roots: Optional[Sequence[str]] = None,
) -> List[RepoInfo]:
    """
    Recursively scan `root` for plausible repository roots.

    This is intentionally simple and file-system based; the caller can
    decide how to persist or filter the results.
    """
    repos: List[RepoInfo] = []
    seen_roots: set[str] = set()
    seen_repo_ids: set[str] = set()

    for scan_root in _normalize_roots(root=root, roots=roots):
        if not os.path.isdir(scan_root):
            continue
        for dirpath, dirnames, _filenames in os.walk(scan_root):
            # If this directory looks like a repo, register it and do not
            # descend into its children (treat nested structure as one repo).
            if _looks_like_repo(dirpath):
                repo_root = os.path.abspath(dirpath)
                canonical_root = os.path.realpath(repo_root)
                if canonical_root not in seen_roots:
                    repo_id = os.path.basename(repo_root) or "repo"
                    if repo_id not in seen_repo_ids:
                        seen_roots.add(canonical_root)
                        seen_repo_ids.add(repo_id)
                        repos.append(
                            RepoInfo(
                                repo_id=repo_id,
                                root=repo_root,
                                library_root=scan_root,
                            )
                        )
                # Prevent descending into subdirectories of this repo
                dirnames[:] = []
                continue

    return sorted(repos, key=lambda r: r.repo_id)


def iter_repositories(
    root: str = DEFAULT_LIBRARY_ROOT,
    *,
    roots: Optional[Sequence[str]] = None,
) -> Iterable[RepoInfo]:
    """Generator wrapper around `discover_repositories`."""
    for r in discover_repositories(root=root, roots=roots):
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
    parser.add_argument(
        "--extra-root",
        action="append",
        dest="extra_roots",
        help=(
            "Additional root directory to scan for repositories. "
            "May be passed multiple times."
        ),
    )
    args = parser.parse_args()

    repos = discover_repositories(root=args.root, roots=args.extra_roots)
    out = [
        {
            "repo_id": r.repo_id,
            "root": r.root,
            "library_root": r.library_root,
        }
        for r in repos
    ]
    json.dump(out, sys.stdout, indent=2)
    if out:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()

