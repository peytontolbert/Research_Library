from __future__ import annotations

"""
Universe builder: construct a heterogeneous repository universe with embeddings,
k-NN similarity, and 3D coordinates.

Outputs under `{export_root}/_universe/`:
- nodes.jsonl: per-entity metadata (repo/file/function/class/module)
- edges.jsonl: intra-repo graph edges (typed)
- node_embeddings.npy: embeddings aligned with nodes.jsonl order
- repo_vectors.npy: pooled per-repo vectors
- repo_coords.npy: 3D coordinates (PCA) for repos
- repo_knn_edges.jsonl: similarity edges between repos
- manifest.json: summary + index mappings

The pipeline:
1) Load all repos from exports/_manifest.json.
2) For each repo, open its ProgramGraph, extract entities + edges.
3) Derive textual views for entities (code spans for functions/classes/modules,
   file contents for file entities, fall back to names).
4) Embed all entity texts with `modules.embeddings.embed_texts`.
5) Pool embeddings per repo to get repo vectors, build k-NN graph.
6) Project repo vectors to 3D via PCA for visualization.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from modules.embeddings import embed_texts
from scripts.repo_graph import parse_program_uri  # type: ignore
from scripts.repo_library import load_manifest, open_repository  # type: ignore


DEFAULT_EXPORT_ROOT = "/data/repository_library/exports"
UNIVERSE_DIRNAME = "_universe"


def _read_lines(abs_path: str, start: int, end: int, max_bytes: int) -> str:
    """
    Read a bounded slice of a file (1-based inclusive line numbers).
    """
    if not os.path.isfile(abs_path):
        return ""
    buf: List[str] = []
    byte_budget = int(max_bytes)
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            for idx, line in enumerate(fh, start=1):
                if idx < start:
                    continue
                if idx > end:
                    break
                if byte_budget <= 0:
                    break
                chunk = line.rstrip("\n")
                byte_budget -= len(chunk.encode("utf-8", errors="ignore"))
                buf.append(chunk)
    except Exception:
        return ""
    return "\n".join(buf)


def _entity_text(repo_root: str, entity: Any, max_lines: int, max_bytes: int, resolver) -> str:
    """
    Build a textual view for an entity using code spans when available.
    """
    kind = str(getattr(entity, "kind", "") or "")
    name = str(getattr(entity, "name", "") or "")
    uri = str(getattr(entity, "uri", "") or "")

    # File entities: read a prefix of the file.
    if kind == "file":
        rel = name
        abs_path = os.path.join(repo_root, rel)
        return _read_lines(abs_path, 1, max_lines, max_bytes) or f"{kind} {name}"

    # Structured code entities: resolve to a span and read it.
    try:
        anchor = resolver(uri)
        _, _, rel_path, span = parse_program_uri(anchor.artifact_uri)
        abs_path = os.path.join(repo_root, rel_path)
        a = int(getattr(anchor.span, "start_line", 1))
        b = int(getattr(anchor.span, "end_line", a))
        b = min(b, a + max_lines - 1)
        text = _read_lines(abs_path, a, b, max_bytes)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback: simple descriptor.
    return f"{kind} {name}".strip()


def build_universe(
    export_root: str = DEFAULT_EXPORT_ROOT,
    *,
    max_lines: int = 200,
    max_bytes: int = 20000,
    repo_knn: int = 10,
    embed_batch_size: int = 512,
) -> Dict[str, Any]:
    lib_manifest = load_manifest(export_root)
    repos_meta = lib_manifest.get("repos") or {}
    if not isinstance(repos_meta, dict):
        repos_meta = {}
    repo_ids = list(repos_meta.keys())
    if not repo_ids:
        raise RuntimeError("No repositories found in manifest; run export first.")

    nodes: List[Dict[str, Any]] = []
    node_texts: List[str] = []
    edges: List[Dict[str, Any]] = []
    repo_node_indices: Dict[str, List[int]] = defaultdict(list)

    for repo_id in repo_ids:
        repo = open_repository(repo_id, export_root=export_root)
        graph = repo.graph
        resolver = getattr(graph, "resolve", None)
        if resolver is None:
            continue
        # Repo-level node
        repo_node_idx = len(nodes)
        repo_node_id = f"{repo_id}:repo"
        entry = repos_meta.get(repo_id) or {}
        languages = entry.get("languages") if isinstance(entry, dict) else None
        repo_desc = f"repo {repo_id}"
        if languages:
            repo_desc = f"{repo_desc} languages: {', '.join(languages)}"
        nodes.append(
            {
                "node_id": repo_node_id,
                "repo_id": repo_id,
                "kind": "repo",
                "name": repo_id,
                "uri": "",
                "labels": languages,
            }
        )
        node_texts.append(repo_desc)
        repo_node_indices[repo_id].append(repo_node_idx)
        for ent in graph.entities():
            if getattr(ent, "kind", "") == "file":
                # Skip file-level nodes to avoid huge text slices; keep finer-grained nodes.
                continue
            node_idx = len(nodes)
            node_id = f"{repo_id}:{getattr(ent, 'id', '')}"
            text = _entity_text(str(repo.root_path), ent, max_lines, max_bytes, resolver)
            node = {
                "node_id": node_id,
                "repo_id": repo_id,
                "kind": getattr(ent, "kind", ""),
                "name": getattr(ent, "name", ""),
                "uri": getattr(ent, "uri", ""),
                "labels": getattr(ent, "labels", None),
            }
            nodes.append(node)
            node_texts.append(text or node["name"] or node["kind"])
            repo_node_indices[repo_id].append(node_idx)

        for e in graph.edges():
            edges.append(
                {
                    "src": f"{repo_id}:{getattr(e, 'src', '')}",
                    "dst": f"{repo_id}:{getattr(e, 'dst', '')}",
                    "type": getattr(e, "type", ""),
                    "repo_id": repo_id,
                }
            )
        # File nodes are skipped above, so no repo->file ownership edges needed.

    # Embed all node texts.
    embeddings = embed_texts(node_texts, batch_size=embed_batch_size)
    if embeddings.size == 0:
        raise RuntimeError("Embedding model returned empty embeddings.")

    # Pool per-repo vectors (mean of normalized embeddings).
    repo_vectors: Dict[str, np.ndarray] = {}
    for rid, idxs in repo_node_indices.items():
        if not idxs:
            continue
        vecs = embeddings[idxs]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normed = vecs / norms
        repo_vectors[rid] = normed.mean(axis=0)

    if not repo_vectors:
        raise RuntimeError("No repo vectors were computed.")

    repo_ids_ordered = list(repo_vectors.keys())
    repo_mat = np.stack([repo_vectors[rid] for rid in repo_ids_ordered], axis=0)

    # Compute k-NN similarity edges between repos.
    norms = np.linalg.norm(repo_mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    repo_normed = repo_mat / norms
    sims = repo_normed @ repo_normed.T
    knn_edges: List[Dict[str, Any]] = []
    k = max(1, min(int(repo_knn), len(repo_ids_ordered) - 1))
    for i, src_id in enumerate(repo_ids_ordered):
        sim_row = sims[i]
        order = np.argsort(-sim_row)
        for j in order[1 : k + 1]:  # skip self
            dst_id = repo_ids_ordered[int(j)]
            knn_edges.append(
                {
                    "src_repo": src_id,
                    "dst_repo": dst_id,
                    "weight": float(sim_row[int(j)]),
                }
            )

    # PCA to 3D.
    centered = repo_mat - repo_mat.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    coords = (u[:, :3] * s[:3]) if s.size else np.zeros((len(repo_ids_ordered), 3), dtype=np.float32)
    if coords.shape[1] < 3:
        # Pad to 3 dims if fewer components exist.
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)

    # Persist outputs.
    uni_root = os.path.join(export_root, UNIVERSE_DIRNAME)
    os.makedirs(uni_root, exist_ok=True)

    def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    nodes_path = os.path.join(uni_root, "nodes.jsonl")
    edges_path = os.path.join(uni_root, "edges.jsonl")
    node_emb_path = os.path.join(uni_root, "node_embeddings.npy")
    repo_vec_path = os.path.join(uni_root, "repo_vectors.npy")
    repo_coord_path = os.path.join(uni_root, "repo_coords.npy")
    repo_knn_path = os.path.join(uni_root, "repo_knn_edges.jsonl")
    manifest_path = os.path.join(uni_root, "manifest.json")

    _write_jsonl(nodes_path, nodes)
    _write_jsonl(edges_path, edges)
    _write_jsonl(repo_knn_path, knn_edges)
    np.save(node_emb_path, embeddings)
    np.save(repo_vec_path, repo_mat)
    np.save(repo_coord_path, coords)

    manifest = {
        "built_at": int(time.time()),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "repo_count": len(repo_ids_ordered),
        "embedding_dim": int(embeddings.shape[1]),
        "universe_root": uni_root,
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "node_embeddings_path": node_emb_path,
        "repo_vectors_path": repo_vec_path,
        "repo_coords_path": repo_coord_path,
        "repo_knn_path": repo_knn_path,
        "repo_ids": repo_ids_ordered,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2))

    # Update top-level library manifest with a pointer to the universe.
    lib_manifest = load_manifest(export_root)
    lib_manifest["universe"] = {
        "built_at": manifest["built_at"],
        "node_count": manifest["node_count"],
        "edge_count": manifest["edge_count"],
        "repo_count": manifest["repo_count"],
        "embedding_dim": manifest["embedding_dim"],
        "path": uni_root,
    }
    top_manifest_path = os.path.join(export_root, "_manifest.json")
    with open(top_manifest_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(lib_manifest, indent=2))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a heterogeneous repository universe with embeddings and 3D coords.")
    parser.add_argument("--export-root", type=str, default=DEFAULT_EXPORT_ROOT, help="Exports root containing _manifest.json (default: /data/repository_library/exports)")
    parser.add_argument("--max-lines", type=int, default=200, help="Max lines per entity snippet (default: 200)")
    parser.add_argument("--max-bytes", type=int, default=20000, help="Max bytes per entity snippet (default: 20000)")
    parser.add_argument("--repo-knn", type=int, default=10, help="Top-K nearest neighbors per repo (default: 10)")
    parser.add_argument("--embed-batch-size", type=int, default=512, help="Batch size for embedding nodes to avoid OOM (default: 512)")
    args = parser.parse_args()

    manifest = build_universe(
        export_root=os.path.abspath(args.export_root),
        max_lines=int(args.max_lines),
        max_bytes=int(args.max_bytes),
        repo_knn=int(args.repo_knn),
        embed_batch_size=int(args.embed_batch_size),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
