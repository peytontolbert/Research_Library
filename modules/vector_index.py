from __future__ import annotations

"""
Simple per-repository vector index implementation.

This module provides:

- A minimal `RepoIndex` implementation backed by NumPy arrays.
- Helper functions to build and load a QA-oriented index for a single
  repository, stored alongside the existing graph exports.

The intent is to:

- Use the program graph to enumerate entities for a repo.
- Derive a lightweight textual representation per-entity.
- Embed those texts using `modules.embeddings.embed_texts`.
- Persist embeddings + metadata under:
    /data/repository_library/exports/{repo_id}/indices/qa/

This keeps the "vector database" local and file-based while still
supporting retrieval-augmented QA flows.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .embeddings import embed_texts
from .repository import RepoIndex, Repository


class SimpleNumpyRepoIndex(RepoIndex):
    """
    In-memory RepoIndex backed by NumPy arrays.

    The index stores:
    - `embeddings`: (N, D) float32 array.
    - `items`: list of N metadata dictionaries (entity_id, name, kind, etc.).
    """

    def __init__(self, embeddings: np.ndarray, items: List[Dict[str, Any]]):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D; got shape={embeddings.shape!r}")
        if embeddings.shape[0] != len(items):
            raise ValueError(
                f"embeddings/items length mismatch: {embeddings.shape[0]} vs {len(items)}"
            )
        self._emb = embeddings.astype("float32", copy=False)
        self._items = list(items)
        norms = np.linalg.norm(self._emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._norms = norms

    def search(self, query: str, *, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Return top-k items most similar to `query` using cosine similarity.
        """
        q = str(query or "")
        if not q:
            return []
        q_vec = embed_texts([q])
        if q_vec.size == 0:
            return []
        qv = q_vec[0]
        q_norm = float(np.linalg.norm(qv))
        if q_norm == 0.0:
            q_norm = 1.0

        sims = (self._emb @ qv) / (self._norms[:, 0] * q_norm)
        k = int(max(1, min(top_k, sims.shape[0])))
        idxs = np.argsort(-sims)[:k]

        out: List[Dict[str, Any]] = []
        for i in idxs:
            meta = dict(self._items[int(i)])
            meta["score"] = float(sims[int(i)])
            out.append(meta)
        return out


def _qa_index_paths(base_dir: str) -> Dict[str, str]:
    emb_path = os.path.join(base_dir, "qa_index_embeddings.npy")
    items_path = os.path.join(base_dir, "qa_index_items.json")
    return {"embeddings_path": emb_path, "items_path": items_path}


def build_repo_qa_index(
    repo: Repository,
    *,
    out_dir: str,
    max_entities: int = 20000,
) -> Dict[str, Any]:
    """
    Build a QA-oriented vector index for a single Repository.

    Strategy:
    - Enumerate up to `max_entities` entities from the repo's ProgramGraph.
    - For each entity, derive a compact textual representation:
        "<kind> <name>"
      (this can be enriched later with docstrings/snippets).
    - Compute embeddings via `embed_texts`.
    - Persist embeddings + metadata under `out_dir`.

    Returns:
        A small metadata dictionary describing the index, suitable for
        recording under `manifest['repos'][repo_id]['indices']['qa']`
        and in adapter metadata.
    """
    os.makedirs(out_dir, exist_ok=True)

    entities = list(repo.graph.entities())
    if not entities:
        # Nothing to index; return a marker so callers can skip retrieval.
        return {
            "type": "simple_numpy",
            "size": 0,
            "embeddings_path": "",
            "items_path": "",
        }

    # Prefer structured code entities (modules/classes/functions) and then
    # fall back to file-level entities for known source/doc languages.
    def _is_indexable(e: Any) -> bool:
        kind = str(getattr(e, "kind", "") or "")
        name = str(getattr(e, "name", "") or "")
        labels = getattr(e, "labels", None) or []
        if not isinstance(labels, list):
            labels = []
        # Always include classic Python structure.
        if kind in ("module", "class", "function"):
            return True
        # Include file-level entities only for known text/source-ish languages.
        if kind == "file":
            label_set = {str(x) for x in labels}
            if any(
                tag in label_set
                for tag in (
                    "lang:python",
                    "lang:c",
                    "lang:cpp",
                    "lang:js",
                    "lang:ts",
                    "lang:markdown",
                )
            ):
                # Require a non-empty, somewhat meaningful name.
                return bool(name.strip())
        return False

    indexable = [e for e in entities if _is_indexable(e)]
    if not indexable:
        return {
            "type": "simple_numpy",
            "size": 0,
            "embeddings_path": "",
            "items_path": "",
        }

    # Cap entities by max_entities to keep build tractable on large repos.
    ent_slice = indexable[: int(max_entities)]
    texts: List[str] = []
    items: List[Dict[str, Any]] = []

    for e in ent_slice:
        kind = getattr(e, "kind", "") or ""
        name = getattr(e, "name", "") or ""
        text = f"{kind} {name}".strip() or name or kind
        texts.append(text)
        items.append(
            {
                "entity_id": getattr(e, "id", ""),
                "entity_name": name,
                "entity_kind": kind,
            }
        )

    if not texts:
        return {
            "type": "simple_numpy",
            "size": 0,
            "embeddings_path": "",
            "items_path": "",
        }

    embeddings = embed_texts(texts)
    paths = _qa_index_paths(out_dir)

    np.save(paths["embeddings_path"], embeddings)
    with open(paths["items_path"], "w", encoding="utf-8") as fh:
        fh.write(json.dumps(items, indent=2))

    return {
        "type": "simple_numpy",
        "size": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        **paths,
    }


def load_simple_repo_index(index_meta: Dict[str, Any]) -> Optional[SimpleNumpyRepoIndex]:
    """
    Load a SimpleNumpyRepoIndex from metadata produced by `build_repo_qa_index`.
    """
    if not isinstance(index_meta, dict):
        return None
    if index_meta.get("type") != "simple_numpy":
        return None

    emb_path = index_meta.get("embeddings_path") or ""
    items_path = index_meta.get("items_path") or ""
    if not emb_path or not items_path:
        return None
    if not (os.path.isfile(emb_path) and os.path.isfile(items_path)):
        return None

    try:
        embeddings = np.load(emb_path)
        with open(items_path, "r", encoding="utf-8") as fh:
            items_any = json.loads(fh.read())
        items: List[Dict[str, Any]] = (
            list(items_any) if isinstance(items_any, list) else []
        )
        if not items or embeddings.shape[0] != len(items):
            return None
        return SimpleNumpyRepoIndex(embeddings=embeddings, items=items)
    except Exception:
        # Corrupt or incompatible index; let callers fall back to graph-only QA.
        return None



