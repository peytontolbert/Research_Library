"""
Graph-aware data utilities for models that operate over repository graphs.

We consume the JSONL exports produced under /data/repository_library/exports:
- <repo>/<repo>.entities.jsonl
- <repo>/<repo>.edges.jsonl
- exports/_manifest.json (repo metadata)

Each entity contains an `id`, `name`, `kind`, and `uri`. Each edge contains
`src`, `dst`, and `type`. We build positive/negative samples for link
prediction-style tasks and render them into textual pairs that the HF
tokenizer can consume.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
from collections import defaultdict

EXPORT_ROOT = Path("/data/repository_library/exports")
MANIFEST_PATH = EXPORT_ROOT / "_manifest.json"
ARXIV_METADATA_PATH = Path("/data/arxiv/arxiv-metadata-oai-snapshot.json")


@dataclass
class GraphSample:
    """Simple container for a link prediction example."""

    repo_id: str
    src: Dict[str, str]
    dst: Dict[str, str]
    label: int
    edge_type: Optional[str] = None
    repo_meta: Optional[Dict[str, Any]] = None
    subgraph: Optional[List[Dict[str, str]]] = None
    domain: str = "repo"  # "repo" or "paper"


def _load_jsonl(path: Path) -> Iterable[Dict[str, any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_repo_entities(repo_id: str) -> Dict[str, Dict[str, str]]:
    ent_path = EXPORT_ROOT / repo_id / f"{repo_id}.entities.jsonl"
    entities: Dict[str, Dict[str, str]] = {}
    if not ent_path.exists():
        return entities
    for row in _load_jsonl(ent_path):
        entities[row["id"]] = {
            "id": row["id"],
            "name": row.get("name") or row["id"],
            "kind": row.get("kind") or "",
            "uri": row.get("uri", ""),
        }
    return entities


def _load_repo_edges(repo_id: str, max_edges: int) -> List[Dict[str, str]]:
    edge_path = EXPORT_ROOT / repo_id / f"{repo_id}.edges.jsonl"
    edges: List[Dict[str, str]] = []
    if not edge_path.exists():
        return edges
    for row in _load_jsonl(edge_path):
        edges.append(row)
        if len(edges) >= max_edges:
            break
    return edges


def _select_repos(limit: int) -> List[str]:
    if not MANIFEST_PATH.exists():
        return []
    manifest = json.load(MANIFEST_PATH.open())
    repos = list((manifest.get("repos") or manifest).keys())
    random.seed(42)
    random.shuffle(repos)
    return repos[:limit]


def _load_repo_meta(repo_id: str) -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    manifest = json.load(MANIFEST_PATH.open())
    repos = manifest.get("repos") or manifest
    meta = repos.get(repo_id, {})
    return meta if isinstance(meta, dict) else {}


def load_graph_samples(
    *, max_samples: int = 1024, repos_limit: int = 8, hops: int = 1, max_subgraph_nodes: int = 16
) -> List[GraphSample]:
    """
    Build positive/negative graph edge samples from exported repo graphs.

    - Positive edges come directly from edges.jsonl.
    - Structural negatives are random pairs of entities not observed as edges,
      typed with an edge type drawn from the repo’s vocabulary so models learn
      to distinguish missing links.
    """
    repo_ids = _select_repos(repos_limit)
    samples: List[GraphSample] = []
    random.seed(42)

    for repo_id in repo_ids:
        entities = _load_repo_entities(repo_id)
        if not entities:
            continue
        edges = _load_repo_edges(repo_id, max_edges=max_samples)
        ent_ids = list(entities.keys())
        repo_meta = _load_repo_meta(repo_id)

        # Build a set for quick negative sampling checks.
        edge_set = {(e["src"], e["dst"]) for e in edges}
        edge_types = [e.get("type") for e in edges if e.get("type")]
        # Adjacency for subgraph extraction.
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for e in edges:
            adj.setdefault(e["src"], []).append((e["dst"], e.get("type", "")))
            adj.setdefault(e["dst"], []).append((e["src"], e.get("type", "")))  # undirected neighborhood for context

        def _collect_subgraph(seed_a: str, seed_b: str) -> List[Dict[str, str]]:
            frontier = {seed_a, seed_b}
            visited = set(frontier)
            nodes: List[Dict[str, str]] = []
            for _ in range(max(0, hops)):
                next_frontier = set()
                for nid in list(frontier):
                    for neigh, etype in adj.get(nid, []):
                        if neigh in visited:
                            continue
                        visited.add(neigh)
                        next_frontier.add(neigh)
                        if len(nodes) < max_subgraph_nodes and neigh in entities:
                            nodes.append(
                                {
                                    "id": neigh,
                                    "kind": entities[neigh].get("kind", ""),
                                    "name": entities[neigh].get("name", ""),
                                    "edge_type": etype,
                                }
                            )
                frontier = next_frontier
                if len(nodes) >= max_subgraph_nodes:
                    break
            return nodes

        for e in edges:
            src = entities.get(e["src"])
            dst = entities.get(e["dst"])
            if not src or not dst:
                continue
            sub = _collect_subgraph(e["src"], e["dst"])
            samples.append(
                GraphSample(
                    repo_id=repo_id,
                    src=src,
                    dst=dst,
                    label=1,
                    edge_type=e.get("type"),
                    repo_meta=repo_meta,
                    subgraph=sub,
                )
            )
            if len(samples) >= max_samples:
                return samples

        # Structural negatives: pick random pairs not in edge_set, assign an existing edge type.
        for _ in range(len(edges) * 2):
            if len(samples) >= max_samples:
                break
            src = random.choice(ent_ids)
            dst = random.choice(ent_ids)
            if src == dst or (src, dst) in edge_set:
                continue
            neg_type = random.choice(edge_types) if edge_types else None
            sub = _collect_subgraph(src, dst)
            samples.append(
                GraphSample(
                    repo_id=repo_id,
                    src=entities[src],
                    dst=entities[dst],
                    label=0,
                    edge_type=neg_type,
                    repo_meta=repo_meta,
                    subgraph=sub,
                )
            )
            edge_set.add((src, dst))

    return samples[:max_samples]


def graph_sample_to_text(sample: GraphSample) -> Dict[str, str]:
    """Render a GraphSample into textual fields for the HF tokenizer."""
    def fmt(node: Dict[str, str]) -> str:
        return f"[{node.get('kind','')}] {node.get('name','')} ({node.get('id','')})"

    text_a = fmt(sample.src)
    text_b = fmt(sample.dst)
    edge_info = f"edge={sample.edge_type}" if sample.edge_type else "edge=unknown"
    repo_meta_text = ""
    if sample.repo_meta:
        head = sample.repo_meta.get("repo_state", {}).get("head")
        branch = sample.repo_meta.get("repo_state", {}).get("branch")
        repo_meta_text = f"\nMETA: head={head or 'n/a'} branch={branch or 'n/a'}"
    subgraph_txt = ""
    if sample.subgraph:
        rendered = [f"{n.get('edge_type','')} -> [{n.get('kind','')}] {n.get('name','')} ({n.get('id','')})" for n in sample.subgraph]
        subgraph_txt = "\nSUBGRAPH:\n" + "\n".join(rendered)

    text = (
        f"REPO: {sample.repo_id}{repo_meta_text}\n"
        f"SRC: {text_a}\n"
        f"DST: {text_b}\n"
        f"{edge_info}"
        f"{subgraph_txt}"
    )
    return {
        "text_a": text_a,
        "text_b": text_b,
        "label": sample.label,
        "text": text,
        "edge_type": sample.edge_type or "unknown",
        "repo_id": sample.repo_id,
        "subgraph": sample.subgraph or [],
        "domain": sample.domain,
    }


# --- Paper graph sampling (ArXiv metadata) --- #

@dataclass
class PaperRecord:
    id: str
    title: str
    abstract: str
    authors: str
    categories: str


def _iter_arxiv_metadata(limit: int = 0) -> Iterable[PaperRecord]:
    if not ARXIV_METADATA_PATH.exists():
        return []
    count = 0
    with ARXIV_METADATA_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if limit and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = str(obj.get("id") or "").strip()
            if not pid:
                continue
            rec = PaperRecord(
                id=pid,
                title=str(obj.get("title") or "").strip(),
                abstract=str(obj.get("abstract") or "").strip(),
                authors=str(obj.get("authors") or "").strip(),
                categories=str(obj.get("categories") or "").strip(),
            )
            yield rec
            count += 1


def load_paper_graph_samples(max_samples: int = 1024) -> List[GraphSample]:
    """
    Build paper-level graph-like samples using ArXiv metadata.

    - Positive edges: same primary category; or co-authorship overlap.
    - Structural negatives: random paper pairs with disjoint categories/authors.
    """
    # Simple buckets by primary category and by author token to create edges.
    by_cat: Dict[str, List[PaperRecord]] = defaultdict(list)
    by_author: Dict[str, List[PaperRecord]] = defaultdict(list)

    # Cap metadata scan for speed; adjust if needed.
    for rec in _iter_arxiv_metadata(limit=50000):
        if rec.categories:
            cat = rec.categories.split()[0]
            by_cat[cat].append(rec)
        if rec.authors:
            for a in rec.authors.split(","):
                at = a.strip()
                if at:
                    by_author[at].append(rec)

    samples: List[GraphSample] = []
    random.seed(42)

    def _mk_node(r: PaperRecord) -> Dict[str, str]:
        return {
            "id": r.id,
            "kind": r.categories.split()[0] if r.categories else "paper",
            "name": r.title[:128],
            "uri": f"arxiv://{r.id}",
        }

    # Positive edges from category buckets
    for cat, papers in by_cat.items():
        if len(samples) >= max_samples:
            break
        for i in range(len(papers) - 1):
            if len(samples) >= max_samples:
                break
            a = papers[i]
            b = papers[i + 1]
            samples.append(
                GraphSample(
                    repo_id="arxiv",
                    src=_mk_node(a),
                    dst=_mk_node(b),
                    label=1,
                    edge_type=f"category:{cat}",
                    repo_meta={"category": cat},
                    subgraph=None,
                    domain="paper",
                )
            )

    # Positive edges from co-authors
    for author, papers in by_author.items():
        if len(samples) >= max_samples:
            break
        for i in range(len(papers) - 1):
            if len(samples) >= max_samples:
                break
            a = papers[i]
            b = papers[i + 1]
            samples.append(
                GraphSample(
                    repo_id="arxiv",
                    src=_mk_node(a),
                    dst=_mk_node(b),
                    label=1,
                    edge_type=f"author:{author}",
                    repo_meta={"author": author},
                    subgraph=None,
                    domain="paper",
                )
            )

    # Structural negatives: random mismatched categories/authors
    all_papers = [p for plist in by_cat.values() for p in plist]
    for _ in range(len(samples) * 2):
        if len(samples) >= max_samples:
            break
        a, b = random.sample(all_papers, 2) if len(all_papers) >= 2 else (None, None)
        if not a or not b:
            break
        if a.id == b.id:
            continue
        cat_a = a.categories.split()[0] if a.categories else ""
        cat_b = b.categories.split()[0] if b.categories else ""
        if cat_a and cat_b and cat_a == cat_b:
            continue  # keep negatives dissimilar
        samples.append(
            GraphSample(
                repo_id="arxiv",
                src=_mk_node(a),
                dst=_mk_node(b),
                label=0,
                edge_type="category_mismatch",
                repo_meta={"category_a": cat_a, "category_b": cat_b},
                subgraph=None,
                domain="paper",
            )
        )

    random.shuffle(samples)
    return samples[:max_samples]


def paper_sample_to_text(sample: GraphSample) -> Dict[str, str]:
    """Render a paper GraphSample into textual fields for the HF tokenizer."""
    def fmt(node: Dict[str, str]) -> str:
        return f"[{node.get('kind','paper')}] {node.get('name','')[:200]} ({node.get('id','')})"

    text_a = fmt(sample.src)
    text_b = fmt(sample.dst)
    edge_info = f"edge={sample.edge_type}" if sample.edge_type else "edge=unknown"
    text = f"PAPER GRAPH\nSRC: {text_a}\nDST: {text_b}\n{edge_info}"
    return {
        "text_a": text_a,
        "text_b": text_b,
        "label": sample.label,
        "text": text,
        "edge_type": sample.edge_type or "unknown",
        "repo_id": sample.repo_id,
        "subgraph": sample.subgraph or [],
        "domain": sample.domain,
    }
