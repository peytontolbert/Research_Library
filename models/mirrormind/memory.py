"""
Memory primitives for MirrorMind RepoTwin/PaperTwin.
Provides in-memory episodic and semantic stores with simple retrieval heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Callable
import math
import random
import json
from pathlib import Path

from models.mirrormind.index import DenseIndex, SparseIndex, IndexedItem, FaissIndex
from models.mirrormind.embeddings import TextEmbedder


@dataclass
class Episode:
    """Fine-grained artifact slice for repos or papers."""

    id: str
    entity_id: str  # repo_id or paper_id
    time: Optional[str]
    type: str  # function_def, commit_message, abstract_chunk, etc.
    text: str
    graph_context: Sequence[str] = field(default_factory=list)
    dense: Sequence[float] = field(default_factory=list)
    sparse: Dict[str, int] = field(default_factory=dict)


@dataclass
class SemanticSummary:
    """Trajectory summary over a time window."""

    id: str
    entity_id: str
    time_window: str
    scope: str
    summary_text: str
    key_concepts: Sequence[str] = field(default_factory=list)
    dense: Sequence[float] = field(default_factory=list)


class EpisodicMemoryStore:
    """
    Hybrid dense/sparse episodic memory.
    The store is append-only and keeps a per-entity index for quick filtering.
    """

    def __init__(self) -> None:
        self._by_entity: Dict[str, List[Episode]] = {}
        self.dense_index: Optional[DenseIndex] = None
        self.sparse_index: Optional[SparseIndex] = None
        self.faiss_index: Optional[FaissIndex] = None
        self._text_embedder: Optional[TextEmbedder] = None

    def add(self, episode: Episode) -> None:
        bucket = self._by_entity.setdefault(episode.entity_id, [])
        bucket.append(episode)

    def entities(self) -> List[str]:
        """Return all entity IDs present in the store."""
        return list(self._by_entity.keys())

    def episodes_for(self, entity_id: str) -> List[Episode]:
        """Return a shallow copy of episodes for an entity."""
        return list(self._by_entity.get(entity_id, []))

    def bulk_add(self, episodes: Sequence[Episode]) -> None:
        for ep in episodes:
            self.add(ep)

    def query(
        self,
        entity_id: Optional[str] = None,
        text: str = "",
        types: Optional[Sequence[str]] = None,
        type_weights: Optional[Dict[str, float]] = None,
        time_range: Optional[Tuple[int, int]] = None,
        top_k: int = 5,
    ) -> List[Episode]:
        """
        Retrieve candidate episodes using optional dense/sparse indexes; falls back to overlap heuristic.
        """
        candidates: List[Episode] = []
        if entity_id:
            candidates.extend(self._by_entity.get(entity_id, []))
        else:
            for vals in self._by_entity.values():
                candidates.extend(vals)

        if types:
            type_set = set(types)
            candidates = [ep for ep in candidates if ep.type in type_set]
        if time_range:
            # Interpret `time` as an integer-like timestamp prefix; keep
            # episodes with missing or unparsable times to avoid silently
            # dropping data, but prefer ones in the requested window later
            # in scoring.
            t0, t1 = time_range
            filtered = []
            for ep in candidates:
                try:
                    t_val = int(str(ep.time)[:10]) if ep.time is not None else None
                except Exception:
                    t_val = None
                if t_val is None or (t0 <= t_val <= t1):
                    filtered.append(ep)
            candidates = filtered

        if not candidates:
            return []

        # If indexes exist, use them first.
        indexed_hits: List[Episode] = []
        qvec = [float(len(text))] if text else []
        if text and self._text_embedder is not None:
            try:
                qvec = self._text_embedder.encode([text])[0]
            except Exception:
                qvec = [float(len(text))]
        if self.faiss_index and text:
            for hit in self.faiss_index.query(qvec, top_k=top_k * 2):
                ep = self._resolve(hit)
                if ep:
                    indexed_hits.append(ep)
        if self.dense_index and text:
            for hit in self.dense_index.query(qvec, top_k=top_k * 2):
                ep = self._resolve(hit)
                if ep:
                    indexed_hits.append(ep)
        if self.sparse_index and text:
            for hit in self.sparse_index.query(text, top_k=top_k * 2):
                ep = self._resolve(hit)
                if ep:
                    indexed_hits.append(ep)
        if indexed_hits:
            # De-dup preserving order.
            seen = set()
            uniq: List[Episode] = []
            for ep in indexed_hits:
                if ep.id in seen:
                    continue
                seen.add(ep.id)
                uniq.append(ep)
            return uniq[:top_k]

        # Fallback heuristic with lightweight recency + type awareness.
        query_tokens = set(text.lower().split()) if text else set()

        # Pre-compute a simple recency window over all candidates.
        times: List[int] = []
        for ep in candidates:
            try:
                t_val = int(str(ep.time)[:10]) if ep.time is not None else None
            except Exception:
                t_val = None
            if t_val is not None:
                times.append(t_val)
        t_min = min(times) if times else None
        t_max = max(times) if times else None

        def _recency_boost(ep: Episode) -> float:
            if t_min is None or t_max is None:
                return 0.0
            try:
                t_val = int(str(ep.time)[:10]) if ep.time is not None else None
            except Exception:
                t_val = None
            if t_val is None or t_max == t_min:
                return 0.0
            # Normalized 0..1, where more recent episodes get higher scores.
            return (t_val - t_min) / float(t_max - t_min)

        def _builtin_type_weight(ep_type: str) -> float:
            et = ep_type.lower()
            if "test" in et:
                return 1.2
            if "doc" in et or "readme" in et or "comment" in et:
                return 1.1
            if "commit" in et or "issue" in et:
                return 1.05
            return 1.0

        scored: List[Tuple[float, Episode]] = []
        for ep in candidates:
            dense_score = math.sqrt(sum(x * x for x in ep.dense)) if ep.dense else 0.0
            dense_score = dense_score / (1 + dense_score)  # bound 0..1
            overlap = 0.0
            if query_tokens and ep.text:
                tokens = set(ep.text.lower().split())
                overlap = len(tokens & query_tokens) / float(len(tokens | query_tokens) or 1)

            recency = _recency_boost(ep)
            type_weight_base = _builtin_type_weight(ep.type)
            if type_weights and ep.type in type_weights:
                type_weight_base *= type_weights[ep.type]

            # Combined score:
            # - dense similarity proxy (0.6)
            # - token overlap (0.3)
            # - recency (0.1)
            base_score = dense_score * 0.6 + overlap * 0.3 + recency * 0.1
            score = base_score * type_weight_base
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def _resolve(self, item: IndexedItem) -> Optional[Episode]:
        for ep in self._by_entity.get(item.entity_id, []):
            if ep.id == item.episode_id:
                return ep
        return None

    def build_indexes(
        self,
        embedder: Optional[Callable[[Episode], Sequence[float]]] = None,
        use_faiss: bool = True,
        text_embedder: Optional[TextEmbedder] = None,
    ) -> None:
        """Build dense and sparse indexes from current episodes."""
        if text_embedder is None:
            text_embedder = TextEmbedder()
        # Cache for query-time embedding to ensure consistent dimensions.
        self._text_embedder = text_embedder
        dense_items: List[IndexedItem] = []
        sparse_items: List[IndexedItem] = []
        # Batch embed via text_embedder for dense vectors, with embedder override per-episode if provided.
        all_eps: List[Episode] = [ep for eps in self._by_entity.values() for ep in eps]
        if embedder is not None:
            dense_vecs = [list(embedder(ep)) for ep in all_eps]
        else:
            dense_vecs = text_embedder.encode([ep.text for ep in all_eps]) if all_eps else []

        for ep, vec in zip(all_eps, dense_vecs):
            dense_items.append(IndexedItem(entity_id=ep.entity_id, episode_id=ep.id, vec=vec, text=ep.text, type=ep.type))
            sparse_items.append(IndexedItem(entity_id=ep.entity_id, episode_id=ep.id, vec=vec, text=ep.text, type=ep.type))
        if use_faiss:
            self.faiss_index = FaissIndex()
            self.faiss_index.build(dense_items)
        self.dense_index = DenseIndex()
        self.dense_index.build(dense_items)
        self.sparse_index = SparseIndex()
        self.sparse_index.build(sparse_items)

    def save(self, path: Path) -> None:
        """Persist episodes to JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for bucket in self._by_entity.values():
                for ep in bucket:
                    f.write(json.dumps(ep.__dict__, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EpisodicMemoryStore":
        store = cls()
        if not path.exists():
            return store
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    store.add(Episode(**obj))
                except Exception:
                    continue
        return store


class SemanticMemoryStore:
    """Semantic summaries keyed per entity with simple similarity scoring."""

    def __init__(self) -> None:
        self._by_entity: Dict[str, List[SemanticSummary]] = {}
        self._embedder = TextEmbedder()
        self.last_query_metrics: Dict[str, object] = {}

    def add(self, summary: SemanticSummary) -> None:
        bucket = self._by_entity.setdefault(summary.entity_id, [])
        if not summary.dense and summary.summary_text:
            try:
                summary.dense = self._embedder.encode([summary.summary_text])[0]
            except Exception:
                summary.dense = [float(len(summary.summary_text))]
        bucket.append(summary)

    def bulk_add(self, summaries: Sequence[SemanticSummary]) -> None:
        for s in summaries:
            self.add(s)

    def query(
        self,
        entity_id: Optional[str],
        text: str = "",
        scope: Optional[str] = None,
        top_k: int = 3,
    ) -> List[SemanticSummary]:
        candidates: List[SemanticSummary] = []
        if entity_id:
            candidates.extend(self._by_entity.get(entity_id, []))
        else:
            for vals in self._by_entity.values():
                candidates.extend(vals)

        if scope:
            candidates = [s for s in candidates if scope.lower() in s.scope.lower()]

        if not candidates:
            self.last_query_metrics = {"candidates": 0, "returned": 0}
            return []

        query_tokens = set(text.lower().split()) if text else set()
        try:
            q_vec = self._embedder.encode([text])[0] if text else []
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

        scored: List[Tuple[float, SemanticSummary]] = []
        for summ in candidates:
            tokens = set(summ.summary_text.lower().split())
            overlap = len(tokens & query_tokens) / float(len(tokens | query_tokens) or 1) if query_tokens else 0.0
            dense_score = _cosine(q_vec, summ.dense)
            key_bonus = 0.1 if summ.key_concepts and query_tokens.intersection(set(summ.key_concepts)) else 0.0
            score = overlap * 0.5 + dense_score * 0.4 + key_bonus
            scored.append((score, summ))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [s for _, s in scored[:top_k]]
        self.last_query_metrics = {
            "candidates": len(candidates),
            "returned": len(results),
            "used_embedding": bool(q_vec),
            "scope_filtered": bool(scope),
            "top_k": top_k,
        }
        return results

    def random_span(self, entity_id: str, num: int = 1) -> List[SemanticSummary]:
        """Lightweight helper to pull a random semantic window for persona prompts."""
        bucket = self._by_entity.get(entity_id, [])
        if not bucket:
            return []
        random.seed(42)
        random.shuffle(bucket)
        return bucket[:num]

    def save(self, path: Path) -> None:
        """Persist summaries to JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for bucket in self._by_entity.values():
                for summ in bucket:
                    f.write(json.dumps(summ.__dict__, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "SemanticMemoryStore":
        store = cls()
        if not path.exists():
            return store
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    store.add(SemanticSummary(**obj))
                except Exception:
                    continue
        return store


def build_semantic_summaries(
    entity_id: str,
    episodes: Sequence[Episode],
    *,
    summarize_fn: Optional[Callable[[str], str]] = None,
    scope_label: str = "generic",
    include_raw: bool = False,
) -> List[SemanticSummary]:
    """
    Lightweight helper to aggregate episodes into SemanticSummary objects.

    This does NOT perform any IO; callers are expected to persist summaries via
    SemanticMemoryStore.save(). The default summarization strategy is a
    heuristic concatenation/truncation of episode texts; pass a custom
    `summarize_fn` to integrate a real LLM-based summarizer.
    """
    if not episodes:
        return []

    # Concatenate episode texts in order; treat the whole span as a single
    # coarse time window for now (callers can shard by commit/release if
    # desired).
    texts: List[str] = []
    times: List[int] = []
    for ep in episodes:
        if ep.text:
            texts.append(ep.text.strip())
        try:
            t_val = int(str(ep.time)[:10]) if ep.time is not None else None
        except Exception:
            t_val = None
        if t_val is not None:
            times.append(t_val)

    if not texts:
        return []

    raw = "\n".join(texts)
    if summarize_fn is not None:
        try:
            summary_text = summarize_fn(raw)
        except Exception:
            # Fall back to full concatenation if the custom summarizer fails.
            summary_text = raw
    else:
        # Default: keep the full concatenation to preserve complete scope.
        summary_text = raw

    if include_raw and summarize_fn is not None and summary_text != raw:
        summary_text = f"{summary_text}\n\nRAW_CONTEXT:\n{raw}"

    time_window = ""
    if times:
        t_min = min(times)
        t_max = max(times)
        if t_min == t_max:
            time_window = str(t_min)
        else:
            time_window = f"{t_min}-{t_max}"

    # Use a very small dense proxy: length of summary.
    dense_vec = [float(len(summary_text))]

    summary = SemanticSummary(
        id=str(len(texts)),  # caller can re-id if necessary
        entity_id=entity_id,
        time_window=time_window,
        scope=scope_label,
        summary_text=summary_text,
        key_concepts=[],
        dense=dense_vec,
    )
    return [summary]
