"""
Simple graph client interface and a file-backed implementation using DomainGraph.
This makes it easier to swap in a real graph DB client later.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from models.mirrormind.domain import DomainGraph


class GraphClient:
    """Interface for concept graph operations."""

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, object]]:
        raise NotImplementedError

    def expand(self, concept_id: str) -> List[Dict[str, object]]:
        raise NotImplementedError

    def path(self, a: str, b: str, k: int = 3) -> List[List[str]]:
        raise NotImplementedError

    def experts(self, concept_id: str, k: int = 4) -> List[Dict[str, object]]:
        raise NotImplementedError


class FileGraphClient(GraphClient):
    """Uses DomainGraph (repo/paper concept files) to serve graph queries."""

    def __init__(self, domain_graph: Optional[DomainGraph] = None) -> None:
        self.graph = domain_graph or DomainGraph()

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for node, score in self.graph.search(query, top_k=top_k):
            out.append(
                {
                    "concept_id": node.id,
                    "name": node.name,
                    "score": score,
                    "neighbors": node.neighbors,
                    "top_repos": node.repos[:4],
                    "top_papers": node.papers[:4],
                }
            )
        return out

    def expand(self, concept_id: str) -> List[Dict[str, object]]:
        return [{"concept_id": n.id, "name": n.name} for n in self.graph.expand(concept_id)]

    def path(self, a: str, b: str, k: int = 3) -> List[List[str]]:
        return self.graph.path_between(a, b, k=k)

    def experts(self, concept_id: str, k: int = 4) -> List[Dict[str, object]]:
        exps = self.graph.experts_for_concept(concept_id, k=k)
        return [{"entity_id": eid, "type": etype, "score": score} for eid, etype, score in exps]
