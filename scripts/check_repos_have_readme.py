from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

from scripts.library_repo_scanner import DEFAULT_LIBRARY_ROOT, RepoInfo, iter_repositories


README_CANDIDATES: Tuple[str, ...] = (
    "README",
    "README.md",
    "README.rst",
    "README.txt",
)


def has_readme(repo: RepoInfo) -> bool:
    """Return True if the repo root contains a README (case-insensitive)."""
    try:
        entries = os.listdir(repo.root)
    except OSError:
        # If we can't list the directory, treat it as missing a README.
        return False

    lower_to_name = {name.lower(): name for name in entries}
    for candidate in README_CANDIDATES:
        if candidate.lower() in lower_to_name:
            return True
    return False


def find_repos_missing_readme(
    root: str,
    *,
    extra_roots: Optional[Sequence[str]] = None,
) -> List[RepoInfo]:
    """Scan one or more library roots for repositories that do not have a README."""
    missing: List[RepoInfo] = []
    for repo in iter_repositories(root=root, roots=extra_roots):
        if not has_readme(repo):
            missing.append(repo)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that repositories under a library root have a README."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_LIBRARY_ROOT,
        help="Root directory that contains individual repositories "
        f"(default: {DEFAULT_LIBRARY_ROOT}).",
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

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"ERROR: Root directory does not exist or is not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    missing = find_repos_missing_readme(root=root, extra_roots=args.extra_roots)

    if not missing:
        print(f"All repositories under {root} have a README.")
        sys.exit(0)

    print(f"Repositories under {root} missing a README ({len(missing)}):")
    for repo in missing:
        print(f"- {repo.repo_id} ({repo.root})")

    # Non-zero exit code so this can be used in CI.
    sys.exit(1)


if __name__ == "__main__":
    main()

