"""
Neo4j-backed GraphClient implementation.
Assumes a Neo4j instance with Concept nodes and relationships:
- Nodes: (:Concept {id, name})
- Relationships: :RELATED or typed edges between concepts
- Optional relationships to repos/papers if modeled as nodes.
Requires environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None  # type: ignore


class Neo4jGraphClient:
    """GraphClient implementation using Neo4j."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
        if GraphDatabase is None:
            raise ImportError("neo4j driver not installed")
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        logging.getLogger(__name__).info("Using Neo4j URI=%s user=%s (dotenv/env loaded)", self.uri, self.user)
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, object]]:
        cypher = """
        MATCH (c:Concept)
        WHERE toLower(c.name) CONTAINS toLower($q) OR c.id CONTAINS $q
        OPTIONAL MATCH (c)-[:RELATED]-(n:Concept)
        RETURN c.id AS concept_id, c.name AS name, collect(distinct n.id)[0..8] AS neighbors
        LIMIT $k
        """
        with self.driver.session() as session:
            res = session.run(cypher, q=query, k=top_k)
            rows = res.data()
        return [
            {
                "concept_id": r.get("concept_id"),
                "name": r.get("name"),
                "neighbors": r.get("neighbors") or [],
                "top_repos": [],
                "top_papers": [],
                "score": 1.0,
            }
            for r in rows
        ]

    def expand(self, concept_id: str) -> List[Dict[str, object]]:
        cypher = """
        MATCH (c:Concept {id: $cid})-[:RELATED]-(n:Concept)
        RETURN n.id AS concept_id, n.name AS name
        LIMIT 16
        """
        with self.driver.session() as session:
            res = session.run(cypher, cid=concept_id)
            return res.data()

    def path(self, a: str, b: str, k: int = 3) -> List[List[str]]:
        cypher = """
        MATCH p = shortestPath((a:Concept {id: $a})-[:RELATED*..5]-(b:Concept {id: $b}))
        RETURN [n IN nodes(p) | n.id] AS path
        LIMIT $k
        """
        with self.driver.session() as session:
            res = session.run(cypher, a=a, b=b, k=k)
            return [r["path"] for r in res if r.get("path")]

    def experts(self, concept_id: str, k: int = 4) -> List[Dict[str, object]]:
        cypher = """
        MATCH (c:Concept {id: $cid})<-[:USES]-(e)
        RETURN e.id AS entity_id, labels(e)[0] AS type
        LIMIT $k
        """
        with self.driver.session() as session:
            res = session.run(cypher, cid=concept_id, k=k)
            return [{"entity_id": r.get("entity_id"), "type": r.get("type"), "score": 1.0} for r in res]
