"""
DomainGraph and DomainAgent scaffolding.
Provides concept search/expansion/pathing backed by exported repo concepts as a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable
import json
import math
import random

from models.mirrormind.embeddings import TextEmbedder


@dataclass
class ConceptNode:
    id: str
    name: str
    neighbors: List[str] = field(default_factory=list)
    repos: List[str] = field(default_factory=list)
    papers: List[str] = field(default_factory=list)
    embedding: Sequence[float] = field(default_factory=list)
    # Optional per-neighbor edge types; kept lightweight to avoid breaking callers.
    # Example: edge_types["neighbor_id"] = ["co_occurs_with", "appears_in_same_repo_as"]
    edge_types: Dict[str, List[str]] = field(default_factory=dict)


class DomainGraph:
    """Concept graph built from repo (and optional paper) concept exports with lightweight edges."""

    def __init__(
        self,
        repo_concepts_path: Path = Path("models/exports/repo_concepts.jsonl"),
        paper_concepts_path: Path = Path("models/exports/paper_concepts.jsonl"),
        paper_repo_align_path: Path = Path("models/exports/paper_repo_align.jsonl"),
    ) -> None:
        self.repo_concepts_path = repo_concepts_path
        self.paper_concepts_path = paper_concepts_path
        self.paper_repo_align_path = paper_repo_align_path
        self.nodes: Dict[str, ConceptNode] = {}
        self._embedder = TextEmbedder()
        self._load()

    def _load(self) -> None:
        """
        Populate concept nodes from repo/paper exports, build lightweight
        neighbor links, and attach simple text embeddings for use in search.

        This intentionally keeps the schema minimal and backwards-compatible:
        - Neighbor relations are treated as generic "co_occurs_with /
          appears_in_same_repo_as" edges.
        - Embeddings are derived from `name` using TextEmbedder and default to
          length-based scalars when no model is available.
        """
        # Repo concepts
        if self.repo_concepts_path.exists():
            with self.repo_concepts_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    cid = obj.get("id") or obj.get("name")
                    if not cid:
                        continue
                    node = self.nodes.setdefault(cid, ConceptNode(id=cid, name=obj.get("name") or cid))
                    repo_id = obj.get("repo_id")
                    if repo_id:
                        node.repos.append(repo_id)
        # Paper concepts (if available)
        if self.paper_concepts_path.exists():
            with self.paper_concepts_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    cid = obj.get("id") or obj.get("name")
                    if not cid:
                        continue
                    node = self.nodes.setdefault(cid, ConceptNode(id=cid, name=obj.get("name") or cid))
                    paper_id = obj.get("paper_id") or obj.get("id")
                    if paper_id:
                        node.papers.append(paper_id)
        # Paper↔repo alignment to tie repos/papers together
        if self.paper_repo_align_path.exists():
            try:
                with self.paper_repo_align_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        repo_id = obj.get("repo_id")
                        paper_id = obj.get("paper_id") or obj.get("paper")
                        concepts = obj.get("concepts") or []
                        for cid in concepts:
                            node = self.nodes.setdefault(cid, ConceptNode(id=cid, name=cid))
                            if repo_id:
                                node.repos.append(repo_id)
                            if paper_id:
                                node.papers.append(paper_id)
            except Exception:
                # Best-effort: alignment is optional.
                pass
        # Build lightweight neighbor links: connect nodes that share a repo_id.
        # We record neighbors symmetrically and tag them as "co_occurs_with"
        # / "appears_in_same_repo_as" in the edge_types map for future use.
        repos_to_nodes: Dict[str, List[str]] = {}
        for cid, node in self.nodes.items():
            for rid in node.repos:
                repos_to_nodes.setdefault(rid, []).append(cid)
        for _, cids in repos_to_nodes.items():
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    a, b = cids[i], cids[j]
                    if b not in self.nodes[a].neighbors:
                        self.nodes[a].neighbors.append(b)
                    if a not in self.nodes[b].neighbors:
                        self.nodes[b].neighbors.append(a)
                    # Tag edge types in a lightweight fashion.
                    for src, dst in ((a, b), (b, a)):
                        edge_map = self.nodes[src].edge_types.setdefault(dst, [])
                        if "appears_in_same_repo_as" not in edge_map:
                            edge_map.append("appears_in_same_repo_as")
                        if "co_occurs_with" not in edge_map:
                            edge_map.append("co_occurs_with")

        # Build paper-based co-occurrence edges to inch closer to the paper-level
        # graph described in the spec.
        papers_to_nodes: Dict[str, List[str]] = {}
        for cid, node in self.nodes.items():
            for pid in node.papers:
                papers_to_nodes.setdefault(pid, []).append(cid)
        for _, cids in papers_to_nodes.items():
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    a, b = cids[i], cids[j]
                    if b not in self.nodes[a].neighbors:
                        self.nodes[a].neighbors.append(b)
                    if a not in self.nodes[b].neighbors:
                        self.nodes[b].neighbors.append(a)
                    for src, dst in ((a, b), (b, a)):
                        edge_map = self.nodes[src].edge_types.setdefault(dst, [])
                        if "appears_in_same_paper_as" not in edge_map:
                            edge_map.append("appears_in_same_paper_as")
                        if "co_occurs_with" not in edge_map:
                            edge_map.append("co_occurs_with")

        # Attach simple text embeddings to each concept name for semantic search.
        if self.nodes:
            names = [node.name for node in self.nodes.values()]
            vecs = self._embedder.encode(names)
            for node, vec in zip(self.nodes.values(), vecs):
                node.embedding = vec

    def search(self, query: str, top_k: int = 8) -> List[Tuple[ConceptNode, float]]:
        q_tokens = set(query.lower().split())
        q_vec: Sequence[float] = []
        try:
            q_vec = self._embedder.encode([query])[0]
        except Exception:
            q_vec = []

        def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored: List[Tuple[float, ConceptNode]] = []
        for node in self.nodes.values():
            toks = set(node.name.lower().split())
            overlap = len(toks & q_tokens) / float(len(toks | q_tokens) or 1) if q_tokens else 0.0
            embed_score = _cosine(q_vec, node.embedding)
            score = overlap * 0.4 + embed_score * 0.6
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(n, s) for s, n in scored[:top_k]]

    def expand(self, concept_id: str) -> List[ConceptNode]:
        node = self.nodes.get(concept_id)
        if not node:
            return []
        return [self.nodes[nid] for nid in node.neighbors if nid in self.nodes]

    def path_between(self, a: str, b: str, k: int = 3) -> List[List[str]]:
        """Very small BFS up to k steps."""
        if a not in self.nodes or b not in self.nodes:
            return []
        paths: List[List[str]] = []
        frontier = [[a]]
        visited = {a}
        while frontier and len(paths) < k:
            path = frontier.pop(0)
            last = path[-1]
            if last == b:
                paths.append(path)
                continue
            for neigh in self.nodes[last].neighbors:
                if neigh in visited:
                    continue
                visited.add(neigh)
                frontier.append(path + [neigh])
        return paths

    def experts_for_concept(self, concept_id: str, k: int = 4) -> List[Tuple[str, str, float]]:
        node = self.nodes.get(concept_id)
        if not node:
            return []
        if not node.repos:
            return []

        # Score repos by how frequently they appear for this concept and by
        # the local connectivity (degree) of the concept node. This is a
        # lightweight proxy for "expertise" without requiring a full graph DB.
        freq: Dict[str, int] = {}
        for rid in node.repos:
            freq[rid] = freq.get(rid, 0) + 1
        max_freq = max(freq.values()) if freq else 1

        degree = len(node.neighbors)
        degree_bonus = min(1.0, degree / 10.0)

        scored: List[Tuple[float, str]] = []
        for rid, cnt in freq.items():
            freq_score = cnt / max_freq
            score = 0.7 * freq_score + 0.3 * degree_bonus
            scored.append((score, rid))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [(rid, "repo", float(score)) for score, rid in scored[:k]]


class DomainAgent:
    """LLM-friendly wrapper exposing DomainGraph tools."""

    def __init__(
        self,
        domain_graph: Optional[DomainGraph] = None,
        graph_client: Optional[Callable[[str], Dict[str, object]]] = None,
    ) -> None:
        self.domain_graph = domain_graph or DomainGraph()
        self.graph_client = graph_client

    def search_concepts(self, query: str, top_k: int = 8) -> List[Dict[str, object]]:
        if self.graph_client and hasattr(self.graph_client, "search"):
            try:
                return self.graph_client.search(query, top_k=top_k)  # type: ignore[attr-defined]
            except Exception:
                pass
        results = []
        for node, score in self.domain_graph.search(query, top_k=top_k):
            results.append(
                {
                    "concept_id": node.id,
                    "name": node.name,
                    "score": score,
                    "neighbors": node.neighbors,
                    "edge_types": node.edge_types,
                    "top_repos": node.repos[:4],
                    "top_papers": node.papers[:4],
                }
            )
        return results

    def expand_concepts(self, concept_id: str) -> List[Dict[str, str]]:
        if self.graph_client and hasattr(self.graph_client, "expand"):
            try:
                return self.graph_client.expand(concept_id)  # type: ignore[attr-defined]
            except Exception:
                pass
        return [{"concept_id": n.id, "name": n.name} for n in self.domain_graph.expand(concept_id)]

    def find_path_between(self, concept_id_a: str, concept_id_b: str, k: int = 3) -> List[List[str]]:
        if self.graph_client and hasattr(self.graph_client, "path"):
            try:
                return self.graph_client.path(concept_id_a, concept_id_b, k=k)  # type: ignore[attr-defined]
            except Exception:
                pass
        return self.domain_graph.path_between(concept_id_a, concept_id_b, k=k)

    def get_expert_entities_for_concept(self, concept_id: str, k: int = 4) -> List[Tuple[str, str, float]]:
        if self.graph_client and hasattr(self.graph_client, "experts"):
            try:
                res = self.graph_client.experts(concept_id, k=k)  # type: ignore[attr-defined]
                if isinstance(res, list):
                    # Normalize to tuple form if using FileGraphClient style dicts.
                    if res and isinstance(res[0], dict):
                        tuples = []
                        for r in res:
                            tuples.append((r.get("entity_id") or r.get("id") or "", r.get("type") or "repo", r.get("score") or 0.0))
                        return tuples  # type: ignore[return-value]
                    return res  # type: ignore[return-value]
            except Exception:
                pass
        return self.domain_graph.experts_for_concept(concept_id, k=k)
