from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from modules.program_graph import Artifact, Edge, Entity  # type: ignore

from .library_repo_scanner import DEFAULT_LIBRARY_ROOT, RepoInfo, discover_repositories
from .python_repo_graph import PythonRepoGraph


DEFAULT_EXPORT_ROOT = "/data/repository_library/exports"
EXPORT_SCHEMA_VERSION = 2
_MANIFEST_FILENAME = "_manifest.json"


@dataclass
class ExportEntity:
    """Serializable representation of a program entity for export."""

    repo_id: str
    id: str
    uri: str
    kind: str
    name: str
    owner: Optional[str]


@dataclass
class ExportEdge:
    """Serializable representation of a graph edge for export."""

    repo_id: str
    src: str
    dst: str
    type: str


@dataclass
class ExportArtifact:
    """Serializable representation of a file-level artifact for export."""

    repo_id: str
    uri: str
    type: str
    hash: str


def _load_manifest(export_root: str) -> Dict[str, Any]:
    """Load the export manifest if present; otherwise return an empty dict."""
    path = os.path.join(export_root, _MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.loads(fh.read())
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _save_manifest(export_root: str, manifest: Dict[str, Any]) -> str:
    """Persist the manifest alongside the exports root."""
    os.makedirs(export_root, exist_ok=True)
    path = os.path.join(export_root, _MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2))
    return path


def _compute_repo_state(repo: RepoInfo) -> Dict[str, Any]:
    """
    Compute a lightweight snapshot of repository state.

    Prefer VCS metadata (e.g., git HEAD) when available; otherwise fall back
    to a coarse file timestamp snapshot. This is used to decide whether an
    existing export is up-to-date.
    """
    git_dir = os.path.join(repo.root, ".git")
    if os.path.isdir(git_dir):
        state: Dict[str, Any] = {"vcs": "git"}
        try:
            head = subprocess.check_output(
                ["git", "-C", repo.root, "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            head = ""
        state["head"] = head
        try:
            branch = subprocess.check_output(
                ["git", "-C", repo.root, "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
            ).strip()
            state["branch"] = branch
        except Exception:
            # Branch is optional (detached HEAD or shallow clone, etc.)
            pass
        return state

    # Fallback: approximate repo state by latest mtime across files
    latest_mtime = 0.0
    for dirpath, _dirnames, filenames in os.walk(repo.root):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                m = os.path.getmtime(fp)
            except Exception:
                continue
            if m > latest_mtime:
                latest_mtime = m
    return {"vcs": "none", "snapshot_mtime": int(latest_mtime)}


def _export_repo_graph(repo: RepoInfo, out_dir: str) -> Dict[str, Any]:
    """
    Build a PythonRepoGraph for `repo` and export entities/edges/artifacts as JSON.

    Returns a small summary dictionary describing what was written.
    """
    graph = PythonRepoGraph(repo.root)

    entities: List[ExportEntity] = []
    languages: Set[str] = set()
    for e in graph.entities():
        assert isinstance(e, Entity)
        entities.append(
            ExportEntity(
                repo_id=repo.repo_id,
                id=e.id,
                uri=e.uri,
                kind=e.kind,
                name=e.name,
                owner=str(e.owner) if getattr(e, "owner", None) is not None else None,
            )
        )
        # Track coarse per-repo language set based on file entities' labels.
        if getattr(e, "kind", "") == "file":
            labels = getattr(e, "labels", None) or []
            if isinstance(labels, list):
                for lab in labels:
                    if isinstance(lab, str) and lab.startswith("lang:"):
                        languages.add(lab.split("lang:", 1)[-1])

    edges: List[ExportEdge] = []
    for ed in graph.edges():
        assert isinstance(ed, Edge)
        edges.append(
            ExportEdge(
                repo_id=repo.repo_id,
                src=ed.src,
                dst=ed.dst,
                type=ed.type,
            )
        )

    artifacts: List[ExportArtifact] = []
    for a in graph.artifacts(kind="source"):
        assert isinstance(a, Artifact)
        artifacts.append(
            ExportArtifact(
                repo_id=repo.repo_id,
                uri=a.uri,
                type=a.type,
                hash=a.hash or "",
            )
        )

    os.makedirs(out_dir, exist_ok=True)
    ent_path = os.path.join(out_dir, f"{repo.repo_id}.entities.jsonl")
    edge_path = os.path.join(out_dir, f"{repo.repo_id}.edges.jsonl")
    art_path = os.path.join(out_dir, f"{repo.repo_id}.artifacts.jsonl")

    def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_jsonl(ent_path, (asdict(e) for e in entities))
    _write_jsonl(edge_path, (asdict(e) for e in edges))
    _write_jsonl(art_path, (asdict(a) for a in artifacts))

    return {
        "repo_id": repo.repo_id,
        "root": repo.root,
        "entities": len(entities),
        "edges": len(edges),
        "artifacts": len(artifacts),
        "entities_path": ent_path,
        "edges_path": edge_path,
        "artifacts_path": art_path,
        "languages": sorted(languages),
    }


def export_library(
    library_root: str = DEFAULT_LIBRARY_ROOT,
    export_root: str = DEFAULT_EXPORT_ROOT,
) -> List[Dict[str, Any]]:
    """
    Run PythonRepoGraph over all repositories under `library_root`
    and export graph data as JSONL files under `export_root`.

    The exporter is **incremental**:
    - It maintains a manifest (`_manifest.json`) under `export_root`.
    - For each repo, it computes a lightweight `repo_state` (e.g., git HEAD).
    - If both the export schema version and `repo_state` match the manifest,
      the repo is considered up-to-date and its export is skipped.

    This remains a pure file-based export; another component can ingest
    the JSONL files into a graph DB or vector index.
    """
    os.makedirs(export_root, exist_ok=True)
    manifest = _load_manifest(export_root)
    # Initialize manifest structure
    if not isinstance(manifest.get("repos"), dict):
        manifest["repos"] = {}
    repos_meta: Dict[str, Any] = manifest["repos"]
    manifest["manifest_version"] = int(manifest.get("manifest_version") or 1)
    manifest["export_schema_version"] = EXPORT_SCHEMA_VERSION

    repos = discover_repositories(root=library_root)
    results: List[Dict[str, Any]] = []

    for repo in repos:
        repo_state = _compute_repo_state(repo)
        prev_entry: Dict[str, Any] = repos_meta.get(repo.repo_id) or {}
        prev_schema = prev_entry.get("export_schema_version")
        prev_state = prev_entry.get("repo_state")

        # Decide whether to skip or rebuild this repo's export
        if (prev_schema == EXPORT_SCHEMA_VERSION) and (prev_state == repo_state):
            summary: Dict[str, Any] = {
                "repo_id": repo.repo_id,
                "root": repo.root,
                "skipped": True,
                "reason": "up_to_date",
                "export_schema_version": EXPORT_SCHEMA_VERSION,
                "repo_state": repo_state,
            }
            results.append(summary)
            continue

        repo_out_dir = os.path.join(export_root, repo.repo_id)
        summary = _export_repo_graph(repo, repo_out_dir)
        summary["skipped"] = False
        summary["export_schema_version"] = EXPORT_SCHEMA_VERSION
        summary["repo_state"] = repo_state
        summary["last_indexed_at"] = int(time.time())
        results.append(summary)

        # Preserve any existing index/skill metadata for this repo
        prev_indices = prev_entry.get("indices") or {}
        prev_skills = prev_entry.get("skills") or {}

        # Update manifest entry for this repo
        repos_meta[repo.repo_id] = {
            "repo_root": repo.root,
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "repo_state": repo_state,
            "last_indexed_at": summary["last_indexed_at"],
            "languages": summary.get("languages") or [],
            # Optional, populated by downstream indexing / adapter tooling
            "indices": prev_indices,
            "skills": prev_skills,
        }

    _save_manifest(export_root, manifest)
    return results


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Export Python repository graphs under a library root as JSONL files."
    )
    parser.add_argument(
        "--library-root",
        type=str,
        default=DEFAULT_LIBRARY_ROOT,
        help="Root directory that contains individual repositories (default: /data/repositories).",
    )
    parser.add_argument(
        "--export-root",
        type=str,
        default=DEFAULT_EXPORT_ROOT,
        help="Directory where JSONL exports will be written (default: /data/repository_library/exports).",
    )
    args = parser.parse_args()

    summaries = export_library(
        library_root=os.path.abspath(args.library_root),
        export_root=os.path.abspath(args.export_root),
    )
    json.dump(summaries, sys.stdout, indent=2)
    if summaries:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()


