"""
DomainGraph and DomainAgent scaffolding.
Provides concept search/expansion/pathing backed by exported repo concepts as a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable, Any
import json
import math
import random

from models.mirrormind.embeddings import TextEmbedder


_DOMAIN_KEYWORDS_CACHE: Optional[Dict[str, List[str]]] = None


def _load_domain_keywords(path: Path = Path("models/mirrormind/domain_keywords.json")) -> Dict[str, List[str]]:
    """
    Load domain→keyword mappings from a JSON file, with a safe fallback.

    The JSON is expected to look like:
      {
        "deep-learning": ["transformer", "attention", ...],
        "compilers": ["compiler", "llvm", ...],
        ...
      }
    """
    global _DOMAIN_KEYWORDS_CACHE
    if _DOMAIN_KEYWORDS_CACHE is not None:
        return _DOMAIN_KEYWORDS_CACHE

    default: Dict[str, List[str]] = {
        "deep-learning": ["transformer", "attention", "llm", "bert", "gpt", "diffusion", "neural", "cnn"],
        "compilers": ["compiler", "llvm", "bytecode", "parser", "interpreter"],
        "rl-control": ["reinforcement", "rl", "reward", "policy", "environment", "gym", "agent", "trajectory", "control"],
    }

    if not path.exists():
        _DOMAIN_KEYWORDS_CACHE = default
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            obj: Any = json.load(f)
    except Exception:
        _DOMAIN_KEYWORDS_CACHE = default
        return default

    if not isinstance(obj, dict):
        _DOMAIN_KEYWORDS_CACHE = default
        return default

    parsed: Dict[str, List[str]] = {}
    for dom, kws in obj.items():
        if not isinstance(dom, str):
            continue
        if isinstance(kws, list):
            parsed[dom] = [str(k).lower() for k in kws if isinstance(k, (str, bytes))]
    if not parsed:
        _DOMAIN_KEYWORDS_CACHE = default
        return default
    _DOMAIN_KEYWORDS_CACHE = parsed
    return parsed


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
        # Add lightweight taxonomic and domain-level structure on top of the
        # co-occurrence graph. This remains heuristic but helps close the
        # gap with the spec's is_subconcept_of and domain pseudo-nodes.
        self._attach_taxonomy_edges()
        self._attach_domain_pseudo_nodes()

        # Attach simple text embeddings to each concept name (including
        # domain pseudo-nodes) for semantic search.
        if self.nodes:
            names = [node.name for node in self.nodes.values()]
            vecs = self._embedder.encode(names)
            for node, vec in zip(self.nodes.values(), vecs):
                node.embedding = vec

    def _attach_taxonomy_edges(self) -> None:
        """
        Heuristically add `is_subconcept_of` / `has_subconcept` edges by
        interpreting dotted, slash-separated, or scoped identifiers as
        simple hierarchies, e.g.:

            "foo.bar" -> parent "foo"
            "pkg/module" -> parent "pkg"

        This runs in O(N) over nodes and avoids quadratic name comparisons.
        """
        for cid, node in list(self.nodes.items()):
            parents: List[str] = []
            for raw in (node.id, node.name):
                if not raw:
                    continue
                # Treat common separators as hierarchical boundaries.
                for sep in (".", "/", "::"):
                    if sep in raw:
                        parent = raw.rsplit(sep, 1)[0]
                        if parent and parent != raw:
                            parents.append(parent)
            for pid in parents:
                if pid not in self.nodes or pid == cid:
                    continue
                parent_node = self.nodes[pid]
                # Undirected neighbor links.
                if pid not in node.neighbors:
                    node.neighbors.append(pid)
                if cid not in parent_node.neighbors:
                    parent_node.neighbors.append(cid)
                # Directed edge types: child -> parent is_subconcept_of, parent -> child has_subconcept.
                child_edges = node.edge_types.setdefault(pid, [])
                if "is_subconcept_of" not in child_edges:
                    child_edges.append("is_subconcept_of")
                parent_edges = parent_node.edge_types.setdefault(cid, [])
                if "has_subconcept" not in parent_edges:
                    parent_edges.append("has_subconcept")

    def _attach_domain_pseudo_nodes(self) -> None:
        """
        Create a small set of domain-level pseudo-nodes and connect concepts
        to them via `belongs_to_domain` / `domain_has_concept` edges.

        Domains and keywords are heuristic but cheap:
        - deep-learning: transformer, attention, llm, bert, gpt, diffusion
        - compilers: compiler, llvm, bytecode, parser, interpreter
        - rl-control: reward, policy, environment, gym, agent, trajectory, control
        """
        if not self.nodes:
            return

        domain_keywords = _load_domain_keywords()

        # Ensure domain pseudo-nodes exist.
        domain_nodes: Dict[str, ConceptNode] = {}
        for dom, _ in domain_keywords.items():
            dom_id = f"__domain:{dom}__"
            node = self.nodes.get(dom_id)
            if node is None:
                node = ConceptNode(id=dom_id, name=dom.replace("-", " ").title())
                self.nodes[dom_id] = node
            domain_nodes[dom] = node

        # Map each concept to zero or more domains based on name tokens.
        concept_domains: Dict[str, List[str]] = {}
        for cid, node in self.nodes.items():
            name_l = node.name.lower()
            toks = set(name_l.split())
            assigned: List[str] = []
            for dom, kws in domain_keywords.items():
                if any((kw in toks) or (kw in name_l) for kw in kws):
                    assigned.append(dom)
            if assigned:
                concept_domains[cid] = assigned
                for dom in assigned:
                    dom_node = domain_nodes[dom]
                    # Undirected neighbor link domain <-> concept.
                    if cid not in dom_node.neighbors:
                        dom_node.neighbors.append(cid)
                    if dom_node.id not in node.neighbors:
                        node.neighbors.append(dom_node.id)
                    # Edge types.
                    d_edges = dom_node.edge_types.setdefault(cid, [])
                    if "domain_has_concept" not in d_edges:
                        d_edges.append("domain_has_concept")
                    c_edges = node.edge_types.setdefault(dom_node.id, [])
                    if "belongs_to_domain" not in c_edges:
                        c_edges.append("belongs_to_domain")

        # Cross-domain connectors: if a single concept maps to multiple
        # domains, link those domains together to reflect shared topics.
        for _, doms in concept_domains.items():
            if len(doms) < 2:
                continue
            for i in range(len(doms)):
                for j in range(i + 1, len(doms)):
                    da, db = doms[i], doms[j]
                    na = domain_nodes[da]
                    nb = domain_nodes[db]
                    if nb.id not in na.neighbors:
                        na.neighbors.append(nb.id)
                    if na.id not in nb.neighbors:
                        nb.neighbors.append(na.id)
                    eab = na.edge_types.setdefault(nb.id, [])
                    if "cross_domain_connector" not in eab:
                        eab.append("cross_domain_connector")
                    eba = nb.edge_types.setdefault(na.id, [])
                    if "cross_domain_connector" not in eba:
                        eba.append("cross_domain_connector")

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

    def repo_neighbors(self, repo_id: str, k: int = 8) -> List[Tuple[str, float]]:
        """
        Lightweight RepoGraph view induced from the ConceptGraph:
        two repos are neighbors if they share concepts, scored by
        shared-concept frequency.
        """
        if not repo_id:
            return []
        co_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            if repo_id not in node.repos:
                continue
            for rid in node.repos:
                if rid == repo_id:
                    continue
                co_counts[rid] = co_counts.get(rid, 0) + 1
        if not co_counts:
            return []
        max_c = max(co_counts.values()) or 1
        scored: List[Tuple[str, float]] = [(rid, count / max_c) for rid, count in co_counts.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def paper_neighbors(self, paper_id: str, k: int = 8) -> List[Tuple[str, float]]:
        """
        Lightweight PaperGraph view induced from the ConceptGraph:
        two papers are neighbors if they share concepts, scored by
        shared-concept frequency.
        """
        if not paper_id:
            return []
        co_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            if paper_id not in node.papers:
                continue
            for pid in node.papers:
                if pid == paper_id:
                    continue
                co_counts[pid] = co_counts.get(pid, 0) + 1
        if not co_counts:
            return []
        max_c = max(co_counts.values()) or 1
        scored: List[Tuple[str, float]] = [(pid, count / max_c) for pid, count in co_counts.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


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

    def get_repo_neighbors(self, repo_id: str, k: int = 8) -> List[Dict[str, object]]:
        """
        Expose a simple RepoGraph-style neighborhood using the underlying
        DomainGraph. Each neighbor is a repo_id with an associated score.
        """
        neighbors = self.domain_graph.repo_neighbors(repo_id, k=k)
        return [{"repo_id": rid, "score": float(score)} for rid, score in neighbors]

    def get_paper_neighbors(self, paper_id: str, k: int = 8) -> List[Dict[str, object]]:
        """
        Expose a simple PaperGraph-style neighborhood using the underlying
        DomainGraph. Each neighbor is a paper_id with an associated score.
        """
        neighbors = self.domain_graph.paper_neighbors(paper_id, k=k)
        return [{"paper_id": pid, "score": float(score)} for pid, score in neighbors]
