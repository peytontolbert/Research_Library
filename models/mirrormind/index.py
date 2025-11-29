"""
Lightweight dense/sparse indexing helpers for episodic memory.
Pure-Python (numpy optional) with optional persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Optional
import math
import json
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None


@dataclass
class IndexedItem:
    entity_id: str
    episode_id: str
    vec: Sequence[float]
    text: str
    type: str = ""


class DenseIndex:
    """Minimal dense cosine index."""

    def __init__(self) -> None:
        self.items: List[IndexedItem] = []

    def add(self, item: IndexedItem) -> None:
        self.items.append(item)

    def build(self, items: Sequence[IndexedItem]) -> None:
        self.items = list(items)

    def query(self, query_vec: Sequence[float], top_k: int = 5) -> List[IndexedItem]:
        if not self.items:
            return []
        if np is not None:
            q = np.array(query_vec, dtype=float)
            mat = np.array([it.vec for it in self.items], dtype=float)
            denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) + 1e-8)) + 1e-8
            sims = (mat @ q) / denom
            order = sims.argsort()[::-1][:top_k]
            return [self.items[i] for i in order]
        # Fallback pure-Python cosine
        def cos(a: Sequence[float], b: Sequence[float]) -> float:
            num = sum(x * y for x, y in zip(a, b))
            den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            return num / den if den else 0.0

        scored: List[Tuple[float, IndexedItem]] = []
        for it in self.items:
            scored.append((cos(query_vec, it.vec), it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        idx = cls()
        if not path.exists():
            return idx
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx.add(IndexedItem(**obj))
                except Exception:
                    continue
        return idx


class FaissIndex:
    """Optional FAISS-backed index (inner product on normalized vectors)."""

    def __init__(self) -> None:
        self.items: List[IndexedItem] = []
        self.index = None

    def build(self, items: Sequence[IndexedItem]) -> None:
        if faiss is None or np is None or not items:
            self.items = list(items)
            self.index = None
            return
        self.items = list(items)
        vecs = np.array([it.vec for it in self.items], dtype="float32")
        # normalize for cosine similarity
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

    def query(self, query_vec: Sequence[float], top_k: int = 5) -> List[IndexedItem]:
        if self.index is None or faiss is None or np is None or not self.items:
            return []
        q = np.array([query_vec], dtype="float32")
        faiss.normalize_L2(q)
        _, idxs = self.index.search(q, top_k)
        results: List[IndexedItem] = []
        for idx in idxs[0]:
            if 0 <= idx < len(self.items):
                results.append(self.items[int(idx)])
        return results

    def save(self, path: Path) -> None:
        """Persist metadata + vectors; requires faiss."""
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_suffix(".meta.jsonl")
        with meta_path.open("w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissIndex":
        fi = cls()
        meta_path = path.with_suffix(".meta.jsonl")
        if not meta_path.exists():
            return fi
        items: List[IndexedItem] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(IndexedItem(**obj))
                except Exception:
                    continue
        fi.items = items
        if faiss is not None and path.exists():
            try:
                fi.index = faiss.read_index(str(path))
            except Exception:
                fi.index = None
        return fi


class SparseIndex:
    """Very simple tf-idf-like index."""

    def __init__(self) -> None:
        self.items: List[IndexedItem] = []
        self.df: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def build(self, items: Sequence[IndexedItem]) -> None:
        self.items = list(items)
        self.df = {}
        for it in self.items:
            toks = set(self._tokenize(it.text))
            for t in toks:
                self.df[t] = self.df.get(t, 0) + 1

    def query(self, text: str, top_k: int = 5) -> List[IndexedItem]:
        if not self.items:
            return []
        q_toks = self._tokenize(text)
        if not q_toks:
            return []
        # Prefer BM25 if available.
        if BM25Okapi is not None:
            corpus = [self._tokenize(it.text) for it in self.items]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(q_toks)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [self.items[i] for i in order[:top_k]]

        scores: List[Tuple[float, IndexedItem]] = []
        N = len(self.items)
        for it in self.items:
            toks = self._tokenize(it.text)
            tf = {t: toks.count(t) for t in set(toks)}
            score = 0.0
            for t in q_toks:
                idf = math.log((N + 1) / (1 + self.df.get(t, 0))) + 1
                score += tf.get(t, 0) * idf
            scores.append((score, it))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scores[:top_k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it.__dict__, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "SparseIndex":
        idx = cls()
        if not path.exists():
            return idx
        with path.open("r", encoding="utf-8") as f:
            items: List[IndexedItem] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(IndexedItem(**obj))
                except Exception:
                    continue
        idx.build(items)
        return idx
