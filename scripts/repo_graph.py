from __future__ import annotations

import os
import re
import hashlib
from typing import Iterable, List, Tuple, Dict, Optional, Set

from modules.program_graph import (
    ProgramGraph,
    Entity,
    Edge,
    Artifact,
    Span,
    ResolvedAnchor,
    EntityId,
)


def program_id_for_repo(repo_root: str) -> str:
    base = os.path.basename(os.path.abspath(repo_root)) or "repo"
    return base


def artifact_uri(program_id: str, rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"program://{program_id}/artifact/{rel}"


def parse_program_uri(uri: str) -> Tuple[str, str, str, Optional[Tuple[int, int]]]:
    m = re.match(r"^program://([^/]+)/([^/]+)/(.+?)(?:#L(\d+)-L(\d+))?$", uri)
    if not m:
        raise ValueError(f"invalid program uri: {uri}")
    pid, kind, res, a, b = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    span = (int(a), int(b)) if (a and b) else None
    return pid, kind, res, span


class RepoGraph(ProgramGraph):
    def __init__(self, repo_root: str, ignore: Optional[List[str]] = None):
        self.repo_root = os.path.abspath(repo_root)
        self.program_id = program_id_for_repo(self.repo_root)
        self.ignore_rules = [s for s in (ignore or []) if s]
        self._file_hash: Dict[str, str] = {}

    # ProgramGraph defaults (repo-agnostic)
    def entities(self) -> Iterable[Entity]:
        return []

    def edges(self) -> Iterable[Edge]:
        return []

    def search_refs(self, token: str) -> Iterable[Tuple[EntityId, Span]]:
        return []

    def subgraph(self, seeds: List[EntityId], radius: int) -> "ProgramGraph":
        if not seeds or radius <= 0:
            return self
        # Generic BFS over current edges view
        adj: Dict[str, List[str]] = {}
        for e in self.edges():
            adj.setdefault(e.src, []).append(e.dst)
            adj.setdefault(e.dst, []).append(e.src)
        cur = set(seeds)
        seen = set(cur)
        for _ in range(max(1, radius)):
            nxt: Set[str] = set()
            for s in list(cur):
                for nb in adj.get(s, []):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            cur = nxt
        return self  # default view is whole repo; subclasses may return filtered views

    def artifacts(self, kind: str) -> Iterable[Artifact]:
        if kind not in ("artifact", "source"):
            return []
        out: List[Artifact] = []
        for fp in self._discover_files(self.repo_root, self.ignore_rules):
            rel = os.path.relpath(fp, self.repo_root).replace("\\", "/")
            out.append(Artifact(uri=artifact_uri(self.program_id, rel), type="source", hash=self._hash_for(fp), span=None))
        return out

    def resolve(self, uri: str) -> ResolvedAnchor:
        pid, kind, res, span = parse_program_uri(uri)
        if pid != self.program_id:
            raise ValueError(f"program id mismatch: {pid} != {self.program_id}")
        if kind == "artifact":
            abs_fp = os.path.abspath(os.path.join(self.repo_root, res))
            if not os.path.isfile(abs_fp):
                raise FileNotFoundError(f"artifact not found: {abs_fp}")
            a = int(span[0]) if span else 1
            b = int(span[1]) if span else self._safe_count_lines(abs_fp)
            rel = os.path.relpath(abs_fp, self.repo_root).replace("\\", "/")
            return ResolvedAnchor(artifact_uri=artifact_uri(self.program_id, rel), span=Span(start_line=a, end_line=b), hash=self._hash_for(abs_fp))
        # Let subclass handle entity URIs
        return self._resolve_entity_uri(kind, res, span)

    # Hooks for subclasses
    def _resolve_entity_uri(self, kind: str, resource: str, span: Optional[Tuple[int, int]]) -> ResolvedAnchor:
        raise KeyError(f"unrecognized entity uri for kind={kind}, resource={resource}")

    # Utilities
    def _discover_files(self, root: str, ignore: List[str]) -> List[str]:
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            if any(ig and ig in dirpath for ig in ignore):
                continue
            for fn in filenames:
                ap = os.path.abspath(os.path.join(dirpath, fn))
                out.append(ap)
        return out

    def _language_labels_for(self, rel_path: str) -> List[str]:
        """
        Best-effort language tags for a repo-relative path based on extension.

        These are intentionally coarse and are only used to annotate entities
        (e.g., file-level entities) so that downstream tools can filter or
        group by language family when needed.
        """
        _, ext = os.path.splitext(rel_path.lower())
        labels: List[str] = []
        if ext == ".py":
            labels.append("lang:python")
        elif ext in (".c", ".h"):
            labels.append("lang:c")
        elif ext in (".cc", ".cpp", ".cxx", ".hpp"):
            labels.append("lang:cpp")
        elif ext in (".js", ".jsx", ".mjs"):
            labels.append("lang:js")
        elif ext in (".ts", ".tsx"):
            labels.append("lang:ts")
        elif ext == ".md":
            labels.append("lang:markdown")
        return labels

    def _safe_count_lines(self, abs_file: str) -> int:
        try:
            with open(abs_file, "r", encoding="utf-8", errors="ignore") as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 1

    def _hash_for(self, abs_file: str) -> str:
        if abs_file in self._file_hash:
            return self._file_hash[abs_file]
        try:
            with open(abs_file, "rb") as fh:
                raw = fh.read()
            h = hashlib.sha256(raw).hexdigest()
        except Exception:
            h = ""
        self._file_hash[abs_file] = h
        return h


