"""
Build commit-level episodic episodes from git history.

This script walks git repositories listed in exports/_manifest.json and
creates `Episode` objects for commit messages, including a coarse
approximation of `ΔV_r,t` by attaching changed file paths into the
episode's graph_context.

This moves the implementation closer to the temporal / evolution
variables described in models/paper.md (K_r, t(k_t), ΔV_r,t).

Usage (example):
    python -m models.mirrormind.scripts.build_commit_episodic_from_git \\
        --output models/exports/commit_episodes.jsonl \\
        --max-commits-per-repo 256
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from models.mirrormind.memory import Episode, EpisodicMemoryStore, EpisodeType


MANIFEST_PATH = Path("/data/repository_library/exports/_manifest.json")


def _load_manifest_repos() -> Dict[str, Dict]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    repos = obj.get("repos")
    return repos if isinstance(repos, dict) else {}


def _iter_commits(repo_root: Path, max_commits: int) -> Iterable[Dict[str, Optional[str]]]:
    """
    Yield commits as dicts {hash, timestamp, summary} using `git log`.
    Falls back gracefully if git is unavailable.
    """
    if not (repo_root / ".git").exists():
        return []
    cmd = ["git", "-C", str(repo_root), "log", f"-n{max_commits}", "--pretty=%H%x01%ct%x01%s", "--no-merges"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    for line in proc.stdout.splitlines():
        parts = line.split("\x01")
        if len(parts) != 3:
            continue
        h, ts, msg = parts
        yield {"hash": h.strip(), "timestamp": ts.strip(), "summary": msg.strip()}


def _changed_files_for_commit(repo_root: Path, commit_hash: str, max_files: int = 128) -> List[str]:
    """
    Return a list of changed file paths for a commit, relative to repo root.
    This is a coarse proxy for ΔV_r,t; callers that know the ProgramGraph
    can later refine it into actual node IDs.
    """
    cmd = ["git", "-C", str(repo_root), "show", "--pretty=format:", "--name-only", commit_hash]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    paths: List[str] = []
    for line in proc.stdout.splitlines():
        p = line.strip()
        if not p:
            continue
        paths.append(p)
        if len(paths) >= max_files:
            break
    return paths


def _load_repo_entities(repo_id: str) -> List[Dict[str, str]]:
    """
    Load ProgramGraph entities for a repo from the library exports.

    This lets us map changed file paths to concrete ProgramGraph node IDs,
    turning the coarse ΔV_r,t proxy into a real subset of V_r when possible.
    """
    export_root = Path("/data/repository_library/exports")
    ent_path = export_root / repo_id / f"{repo_id}.entities.jsonl"
    entities: List[Dict[str, str]] = []
    if not ent_path.exists():
        return entities
    try:
        with ent_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                eid = obj.get("id")
                uri = obj.get("uri")
                if eid and uri:
                    entities.append({"id": str(eid), "uri": str(uri)})
    except Exception:
        return []
    return entities


def _map_files_to_nodes(repo_id: str, changed_files: List[str]) -> List[str]:
    """
    Best-effort mapping from changed file paths to ProgramGraph node IDs.

    Strategy:
    - Load {repo}.entities.jsonl.
    - For each entity, look at the path-like portion of its URI and match
      on filename or suffix against changed file paths.
    """
    if not changed_files:
        return []
    entities = _load_repo_entities(repo_id)
    if not entities:
        return []

    filenames = {Path(p).name: p for p in changed_files}
    node_ids: List[str] = []
    for ent in entities:
        uri = ent["uri"]
        # Extract a rough path from the URI: "program://repo/REST#L..." -> "REST".
        try:
            tail = uri.split("://", 1)[-1].split("#", 1)[0]
        except Exception:
            tail = uri
        tail_name = Path(tail).name
        if tail_name in filenames:
            node_ids.append(ent["id"])
    # De-duplicate while preserving order.
    seen = set()
    uniq: List[str] = []
    for nid in node_ids:
        if nid in seen:
            continue
        seen.add(nid)
        uniq.append(nid)
    return uniq


def build_commit_episodes_from_git(max_commits_per_repo: int = 256) -> EpisodicMemoryStore:
    """
    Build commit_message episodes for each git repo in the manifest.

    Each episode:
        - id: "{repo_id}:commit:{hash}"
        - entity_id: repo_id
        - time: commit unix timestamp (seconds)
        - type: "commit_message"
        - text: commit summary line
        - graph_context: ProgramGraph node IDs for changed entities, when
          available, otherwise a fallback list of changed file paths.
    """
    store = EpisodicMemoryStore()
    repos = _load_manifest_repos()
    for repo_id, meta in repos.items():
        root = meta.get("repo_root")
        if not root:
            continue
        repo_root = Path(root)
        for c in _iter_commits(repo_root, max_commits=max_commits_per_repo):
            ts = c.get("timestamp")
            summary = c.get("summary") or ""
            if not ts or not summary:
                continue
            changed = _changed_files_for_commit(repo_root, c.get("hash") or "")
            node_ids = _map_files_to_nodes(repo_id, changed)
            if node_ids:
                graph_ctx = node_ids
            else:
                graph_ctx = [f"file://{p}" for p in changed]
            ep = Episode(
                id=f"{repo_id}:commit:{c.get('hash')}",
                entity_id=repo_id,
                time=str(ts),
                type=EpisodeType.COMMIT_MESSAGE,
                text=summary,
                graph_context=graph_ctx,
                dense=[float(len(summary))],
                sparse={},
            )
            store.add(ep)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build commit-level episodic episodes from git history.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for commit episodes.",
    )
    parser.add_argument(
        "--max-commits-per-repo",
        type=int,
        default=256,
        help="Maximum number of commits to sample per repository.",
    )
    args = parser.parse_args()

    store = build_commit_episodes_from_git(max_commits_per_repo=args.max_commits_per_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entity_id in store.entities():
            for ep in store.episodes_for(entity_id):
                f.write(json.dumps(ep.__dict__, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


