from __future__ import annotations

"""
Build script for the repository library.

This orchestrates a full library scan over `/data/repositories` and exports
per-repository Python code graphs as JSONL files under
`/data/repository_library/exports` by default.

It is intentionally lightweight and file-based so that downstream components
(graph database ingester, vector index builder, LLM retrieval service) can
consume the exported data without needing to know about the internal graph
implementation.
"""

import argparse
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

from scripts.library_repo_graph_export import (
    DEFAULT_EXPORT_ROOT,
    DEFAULT_LIBRARY_ROOT,
    EXPORT_SCHEMA_VERSION,
    export_library,
)


def build_library(
    library_root: str = DEFAULT_LIBRARY_ROOT,
    extra_library_roots: Optional[Sequence[str]] = None,
    export_root: str = DEFAULT_EXPORT_ROOT,
) -> List[Dict[str, Any]]:
    """
    Run the library build:

    - Discover repositories under `library_root`.
    - For each repo, construct a `PythonRepoGraph`.
    - Export entities, edges, and artifacts as JSONL files under `export_root`.

    Returns a list of summary dicts, one per repository.
    """
    library_root_abs = os.path.abspath(library_root)
    export_root_abs = os.path.abspath(export_root)
    os.makedirs(export_root_abs, exist_ok=True)

    summaries = export_library(
        library_root=library_root_abs,
        extra_library_roots=extra_library_roots,
        export_root=export_root_abs,
    )
    return summaries


def main() -> None:
    # Suppress noisy SyntaxWarning instances coming from third-party
    # libraries that use non-raw regex strings (e.g. "\d", "\s").
    # They are harmless for this build pipeline but clutter output.
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    parser = argparse.ArgumentParser(
        description="Build the repository library (scan repos and export graphs)."
    )
    parser.add_argument(
        "--library-root",
        type=str,
        default=DEFAULT_LIBRARY_ROOT,
        help="Root directory that contains individual repositories (default: /data/repositories).",
    )
    parser.add_argument(
        "--extra-library-root",
        action="append",
        dest="extra_library_roots",
        help=(
            "Additional root directory to scan and persist as part of the "
            "library. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--export-root",
        type=str,
        default=DEFAULT_EXPORT_ROOT,
        help="Directory where JSONL exports will be written (default: /data/repository_library/exports).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON summary to stdout (default: pretty text).",
    )
    args = parser.parse_args()

    summaries = build_library(
        library_root=args.library_root,
        extra_library_roots=args.extra_library_roots,
        export_root=args.export_root,
    )

    if args.json:
        json.dump(summaries, fp=os.fdopen(1, "w"), indent=2)  # type: ignore[arg-type]
        return

    # Human-readable summary
    if not summaries:
        print("No repositories discovered; nothing to build.")
        return

    print(f"Built library from root: {os.path.abspath(args.library_root)}")
    if args.extra_library_roots:
        print(
            "Extended with roots: "
            + ", ".join(os.path.abspath(root) for root in args.extra_library_roots)
        )
    print(f"Exports written under: {os.path.abspath(args.export_root)}")
    print(f"Export schema version: {EXPORT_SCHEMA_VERSION}")
    print()
    for s in summaries:
        rid = s.get("repo_id")
        root = s.get("root")
        ents = s.get("entities", 0)
        edges = s.get("edges", 0)
        arts = s.get("artifacts", 0)
        skipped = bool(s.get("skipped"))
        reason = s.get("reason") if skipped else "rebuilt"
        status = f"skipped ({reason})" if skipped else "rebuilt"
        print(f"- {rid} ({root}) [{status}]")
        if not skipped:
            print(f"  entities={ents}, edges={edges}, artifacts={arts}")


if __name__ == "__main__":
    main()

