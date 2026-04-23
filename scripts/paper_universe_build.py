from __future__ import annotations

"""
Build a standalone paper universe from a paper-text parquet dataset.

The paper universe intentionally does not duplicate full paper text. It stores
lightweight paper metadata, graph structure, embeddings, and 3D coordinates in
parquet form.

Final outputs:

- `paper_nodes.parquet`: one lightweight node row per paper, including 3D coords
- `category_nodes.parquet`: one node row per arXiv category anchor
- `edges.parquet`: paper -> category membership edges
- `paper_embeddings.parquet`: one metadata embedding row per paper
- `paper_fulltext_embeddings.parquet`: optional full-text embedding row per paper
- optional `paper_knn_edges.parquet`: approximate similarity edges when faiss is
  available and `--paper-knn > 0`
- `topic_nodes.parquet`: lightweight keyword/topic anchors derived from paper text
- `paper_topic_edges.parquet`: paper -> topic membership edges
- `manifest.json`: summary of the completed build
- `progress.json`: live build progress while the build is running

Resume model:

- Expensive embedding work is staged into committed temp parquet shards under
  `_build_tmp/`.
- Each committed shard has a manifest marker written last.
- On restart, the builder resumes from the last committed shard instead of
  rebuilding from zero.
- Final outputs are rewritten from committed temp shards if the build was
  interrupted during finalization.
"""

import argparse
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from modules.embeddings import embed_texts


DEFAULT_DATASET_DIR = Path("/arxiv/huggingface/paper_text_1m_dedup_v1")
DEFAULT_OUTPUT_DIR = Path("/data/repository_library/exports/_paper_universe")
DEFAULT_BATCH_ROWS = 1024
DEFAULT_EMBED_BATCH_SIZE = 256
DEFAULT_TEMP_SHARD_ROWS = 16384
DEFAULT_TEXT_PREFIX_CHARS = 0
DEFAULT_FULLTEXT_CHUNK_CHARS = 2000
DEFAULT_FULLTEXT_MAX_CHUNKS = 0
DEFAULT_MAX_TOPICS_PER_PAPER = 3
DEFAULT_PAPER_KNN = 0
DEFAULT_CATEGORY_KNN = 8
DEFAULT_PCA_CHUNK_ROWS = 16384
DEFAULT_PARQUET_COMPRESSION = "zstd"
TEMP_DIRNAME = "_build_tmp"
PROGRESS_FILENAME = "progress.json"

_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "approach",
    "analysis",
    "as",
    "at",
    "based",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "method",
    "model",
    "models",
    "new",
    "of",
    "on",
    "paper",
    "study",
    "system",
    "systems",
    "the",
    "their",
    "this",
    "to",
    "toward",
    "using",
    "via",
    "with",
}


def _paper_parquet_paths(dataset_dir: Path) -> List[Path]:
    if dataset_dir.is_file():
        return [dataset_dir]
    paths = sorted(dataset_dir.glob("train_*.parquet"))
    if not paths and dataset_dir.is_dir():
        paths = sorted(dataset_dir.glob("*.parquet"))
    return [p for p in paths if p.is_file()]


def _row_count(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            continue
    return total


def _iter_rows(
    paths: Sequence[Path],
    *,
    columns: Sequence[str],
    batch_rows: int,
    max_rows: int = 0,
    skip_rows: int = 0,
) -> Iterator[Dict[str, Any]]:
    emitted = 0
    seen = 0
    limit = max(0, int(max_rows or 0))
    skip = max(0, int(skip_rows or 0))
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=list(columns), batch_size=max(1, int(batch_rows or 1))):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    continue
                if limit and seen >= limit:
                    return
                if seen < skip:
                    seen += 1
                    continue
                seen += 1
                emitted += 1
                yield row


def _iter_embedding_chunks(paths: Sequence[Path], *, batch_rows: int) -> Iterator[np.ndarray]:
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=["embedding"], batch_size=max(1, int(batch_rows or 1))):
            values = batch.column(0).to_pylist()
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.size == 0:
                continue
            yield arr


def _paper_year(canonical_paper_id: str) -> Optional[int]:
    paper_id = str(canonical_paper_id or "").strip()
    if not paper_id:
        return None
    if "/" in paper_id:
        _, local_id = paper_id.split("/", 1)
        if len(local_id) >= 2 and local_id[:2].isdigit():
            yy = int(local_id[:2])
            return 1900 + yy if yy >= 90 else 2000 + yy
        return None
    if len(paper_id) >= 4 and paper_id[:4].isdigit():
        return 2000 + int(paper_id[:2])
    return None


def _category_list(raw_categories: Any) -> List[str]:
    text = str(raw_categories or "").strip()
    if not text:
        return []
    return [tok for tok in text.split() if tok]


def _embedding_text(row: Dict[str, Any], *, text_prefix_chars: int) -> str:
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    parts = [part for part in [title, abstract] if part]
    if text_prefix_chars and int(text_prefix_chars) > 0:
        prefix = str(row.get("text") or "")[: int(text_prefix_chars)]
        if prefix.strip():
            parts.append(prefix.strip())
    return "\n\n".join(parts).strip() or str(row.get("canonical_paper_id") or "").strip()


def _fulltext_chunks(
    row: Dict[str, Any],
    *,
    chunk_chars: int,
    max_chunks: int,
) -> List[str]:
    title = str(row.get("title") or "").strip()
    raw_text = str(row.get("text") or "").strip()
    if not raw_text:
        fallback = _embedding_text(row, text_prefix_chars=0)
        return [fallback] if fallback else []
    size = max(256, int(chunk_chars or DEFAULT_FULLTEXT_CHUNK_CHARS))
    limit = max(1, int(max_chunks or 1))
    chunks: List[str] = []
    for idx, start in enumerate(range(0, len(raw_text), size)):
        if idx >= limit:
            break
        chunk = raw_text[start : start + size].strip()
        if not chunk:
            continue
        chunks.append(f"{title}\n\n{chunk}".strip() if title else chunk)
    return chunks or [(_embedding_text(row, text_prefix_chars=0) or raw_text[:size])]


def _embed_fulltext_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    chunk_chars: int,
    max_chunks: int,
    embed_batch_size: int,
    device: str,
) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    texts: List[str] = []
    owners: List[int] = []
    for idx, row in enumerate(rows):
        for chunk in _fulltext_chunks(row, chunk_chars=chunk_chars, max_chunks=max_chunks):
            texts.append(chunk)
            owners.append(idx)
    if not texts:
        return np.zeros((len(rows), 0), dtype=np.float32)
    chunk_embeddings = embed_texts(
        texts,
        batch_size=max(1, int(embed_batch_size or 1)),
        device=str(device or "").strip() or None,
    )
    if chunk_embeddings.size == 0:
        return np.zeros((len(rows), 0), dtype=np.float32)
    dim = int(chunk_embeddings.shape[1])
    sums = np.zeros((len(rows), dim), dtype=np.float32)
    counts = np.zeros((len(rows), 1), dtype=np.float32)
    for emb, owner in zip(chunk_embeddings, owners):
        sums[owner] += np.asarray(emb, dtype=np.float32)
        counts[owner, 0] += 1.0
    counts[counts == 0.0] = 1.0
    return sums / counts


def _extract_topic_terms(row: Dict[str, Any], *, max_topics: int) -> List[str]:
    cap = max(0, int(max_topics or 0))
    if cap <= 0:
        return []

    def _tokens(text: str) -> List[str]:
        return [tok.lower() for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")]

    title_tokens = [tok for tok in _tokens(str(row.get("title") or "")) if tok not in _TOPIC_STOPWORDS]
    abstract_tokens = [tok for tok in _tokens(str(row.get("abstract") or "")) if tok not in _TOPIC_STOPWORDS]
    counts: Counter[str] = Counter()
    counts.update({tok: 3 for tok in title_tokens})
    counts.update({tok: 1 for tok in abstract_tokens[:128]})

    def _weighted_bigrams(tokens: Sequence[str], weight: int) -> None:
        for a, b in zip(tokens, tokens[1:]):
            if a in _TOPIC_STOPWORDS or b in _TOPIC_STOPWORDS:
                continue
            phrase = f"{a} {b}"
            counts[phrase] += weight

    _weighted_bigrams(title_tokens, 4)
    _weighted_bigrams(abstract_tokens[:64], 1)

    chosen: List[str] = []
    seen_terms: set[str] = set()
    for term, _score in sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0])):
        if term in seen_terms:
            continue
        seen_terms.add(term)
        chosen.append(term)
        if len(chosen) >= cap:
            break
    return chosen


def _chunk_slices(total: int, chunk_rows: int) -> Iterator[Tuple[int, int]]:
    step = max(1, int(chunk_rows or 1))
    for start in range(0, max(0, int(total)), step):
        end = min(total, start + step)
        yield start, end


def _compute_pca_projection_from_embedding_shards(
    embedding_paths: Sequence[Path],
    *,
    chunk_rows: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    dim = 0
    total_rows = 0
    sum_vec: Optional[np.ndarray] = None
    sum_xx: Optional[np.ndarray] = None

    for chunk in _iter_embedding_chunks(embedding_paths, batch_rows=chunk_rows):
        if chunk.size == 0:
            continue
        if dim == 0:
            dim = int(chunk.shape[1])
            sum_vec = np.zeros((dim,), dtype=np.float64)
            sum_xx = np.zeros((dim, dim), dtype=np.float64)
        total_rows += int(chunk.shape[0])
        chunk64 = np.asarray(chunk, dtype=np.float64)
        sum_vec += chunk64.sum(axis=0)  # type: ignore[operator]
        sum_xx += chunk64.T @ chunk64  # type: ignore[operator]

    if dim <= 0 or total_rows <= 0 or sum_vec is None or sum_xx is None:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), 0

    mean = sum_vec / float(total_rows)
    centered_xx = sum_xx - float(total_rows) * np.outer(mean, mean)
    denom = max(1, total_rows - 1)
    cov = centered_xx / float(denom)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = np.zeros((dim, 3), dtype=np.float32)
    top = eigvecs[:, order[: min(3, eigvecs.shape[1])]].astype(np.float32)
    components[:, : top.shape[1]] = top
    return mean.astype(np.float32), components, dim


def _paper_row_schema() -> pa.Schema:
    return pa.schema(
        [
            ("paper_idx", pa.int64()),
            ("paper_id", pa.string()),
            ("canonical_paper_id", pa.string()),
            ("paper_version", pa.string()),
            ("title", pa.string()),
            ("authors", pa.string()),
            ("categories", pa.list_(pa.string())),
            ("primary_category", pa.string()),
            ("license", pa.string()),
            ("update_date", pa.string()),
            ("year", pa.int32()),
            ("text_char_count", pa.int64()),
            ("page_count", pa.int64()),
            ("token_count", pa.int64()),
            ("text_source", pa.string()),
            ("text_is_partial", pa.bool_()),
            ("metadata_found", pa.bool_()),
            ("pdf_path", pa.string()),
        ]
    )


def _paper_node_schema() -> pa.Schema:
    return pa.schema(list(_paper_row_schema()) + [("x", pa.float32()), ("y", pa.float32()), ("z", pa.float32())])


def _category_node_schema() -> pa.Schema:
    return pa.schema(
        [
            ("category_idx", pa.int64()),
            ("category_id", pa.string()),
            ("name", pa.string()),
            ("paper_count", pa.int64()),
            ("x", pa.float32()),
            ("y", pa.float32()),
            ("z", pa.float32()),
        ]
    )


def _edge_schema() -> pa.Schema:
    return pa.schema(
        [
            ("src_paper_idx", pa.int64()),
            ("dst_category_id", pa.string()),
            ("type", pa.string()),
        ]
    )


def _paper_embedding_schema() -> pa.Schema:
    return pa.schema(
        [
            ("paper_idx", pa.int64()),
            ("paper_id", pa.string()),
            ("canonical_paper_id", pa.string()),
            ("embedding", pa.list_(pa.float32())),
        ]
    )


def _paper_fulltext_embedding_schema() -> pa.Schema:
    return _paper_embedding_schema()


def _paper_knn_schema() -> pa.Schema:
    return pa.schema(
        [
            ("src_paper_idx", pa.int64()),
            ("dst_paper_idx", pa.int64()),
            ("type", pa.string()),
            ("weight", pa.float32()),
        ]
    )


def _category_knn_schema() -> pa.Schema:
    return pa.schema(
        [
            ("src_category_id", pa.string()),
            ("dst_category_id", pa.string()),
            ("type", pa.string()),
            ("weight", pa.float32()),
        ]
    )


def _year_node_schema() -> pa.Schema:
    return pa.schema(
        [
            ("year", pa.int32()),
            ("paper_count", pa.int64()),
            ("x", pa.float32()),
            ("y", pa.float32()),
            ("z", pa.float32()),
        ]
    )


def _paper_year_edge_schema() -> pa.Schema:
    return pa.schema(
        [
            ("src_paper_idx", pa.int64()),
            ("dst_year", pa.int32()),
            ("type", pa.string()),
        ]
    )


def _topic_node_schema() -> pa.Schema:
    return pa.schema(
        [
            ("topic_idx", pa.int64()),
            ("topic_id", pa.string()),
            ("name", pa.string()),
            ("paper_count", pa.int64()),
            ("x", pa.float32()),
            ("y", pa.float32()),
            ("z", pa.float32()),
        ]
    )


def _paper_topic_edge_schema() -> pa.Schema:
    return pa.schema(
        [
            ("src_paper_idx", pa.int64()),
            ("dst_topic_id", pa.string()),
            ("type", pa.string()),
        ]
    )


class _ParquetRowWriter:
    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        *,
        batch_rows: int,
        compression: str,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.writer = pq.ParquetWriter(str(path), self.schema, compression=str(compression or DEFAULT_PARQUET_COMPRESSION))
        self.batch_rows = max(1, int(batch_rows or 1))
        self.rows: List[Dict[str, Any]] = []
        self.count = 0

    def write(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _path_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


def _embedding_dim_for_path(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        table = pq.read_table(path, columns=["embedding"]).slice(0, 1)
        rows = table.to_pylist()
        if not rows:
            return 0
        values = rows[0].get("embedding") or []
        return int(len(values))
    except Exception:
        return 0


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except Exception:
        pass


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_table_write(path: Path, schema: pa.Schema, rows: List[Dict[str, Any]], *, compression: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    _safe_unlink(tmp)
    writer = pq.ParquetWriter(str(tmp), schema, compression=str(compression or DEFAULT_PARQUET_COMPRESSION))
    try:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    finally:
        writer.close()
    tmp.replace(path)


def _legacy_output_paths(output_dir: Path) -> List[Path]:
    return [
        output_dir / "paper_nodes.jsonl",
        output_dir / "category_nodes.jsonl",
        output_dir / "edges.jsonl",
        output_dir / "paper_embeddings.npy",
        output_dir / "paper_coords.npy",
        output_dir / "paper_knn_edges.jsonl",
    ]


def _progress_path(output_dir: Path) -> Path:
    return output_dir / PROGRESS_FILENAME


def _temp_root(output_dir: Path) -> Path:
    return output_dir / TEMP_DIRNAME


def _temp_paths(output_dir: Path, shard_index: int) -> Dict[str, Path]:
    root = _temp_root(output_dir)
    stem = f"{int(shard_index):06d}"
    return {
        "paper_rows": root / f"paper_rows_{stem}.parquet",
        "paper_embeddings": root / f"paper_embeddings_{stem}.parquet",
        "edges": root / f"edges_{stem}.parquet",
        "manifest": root / f"shard_{stem}.json",
    }


def _cleanup_incomplete_temp_files(temp_root: Path) -> None:
    if not temp_root.is_dir():
        return
    for path in temp_root.glob("*.tmp"):
        _safe_unlink(path)


def _committed_temp_manifests(output_dir: Path) -> List[Dict[str, Any]]:
    root = _temp_root(output_dir)
    if not root.is_dir():
        return []
    manifests: List[Dict[str, Any]] = []
    manifest_paths = sorted(root.glob("shard_*.json"))
    expected_index = 0
    for path in manifest_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        raw_shard_index = data.get("shard_index")
        shard_index = -1 if raw_shard_index is None else int(raw_shard_index)
        if shard_index != expected_index:
            break
        paths = _temp_paths(output_dir, shard_index)
        if not all(paths[key].is_file() for key in ["paper_rows", "paper_embeddings", "edges"]):
            break
        data["_paths"] = {key: str(value) for key, value in paths.items()}
        manifests.append(data)
        expected_index += 1
    return manifests


def _remove_final_outputs(output_dir: Path) -> None:
    for path in [
        output_dir / "paper_nodes.parquet",
        output_dir / "category_nodes.parquet",
        output_dir / "year_nodes.parquet",
        output_dir / "edges.parquet",
        output_dir / "paper_year_edges.parquet",
        output_dir / "paper_embeddings.parquet",
        output_dir / "paper_fulltext_embeddings.parquet",
        output_dir / "paper_knn_edges.parquet",
        output_dir / "category_knn_edges.parquet",
        output_dir / "topic_nodes.parquet",
        output_dir / "paper_topic_edges.parquet",
        output_dir / "manifest.json",
        output_dir / ".paper_nodes.parquet.tmp",
        output_dir / ".category_nodes.parquet.tmp",
        output_dir / ".year_nodes.parquet.tmp",
        output_dir / ".edges.parquet.tmp",
        output_dir / ".paper_year_edges.parquet.tmp",
        output_dir / ".paper_embeddings.parquet.tmp",
        output_dir / ".paper_fulltext_embeddings.parquet.tmp",
        output_dir / ".paper_knn_edges.parquet.tmp",
        output_dir / ".category_knn_edges.parquet.tmp",
        output_dir / ".topic_nodes.parquet.tmp",
        output_dir / ".paper_topic_edges.parquet.tmp",
        output_dir / ".paper_rows_tmp.parquet",
        output_dir / ".paper_embeddings_tmp.npy",
        output_dir / ".paper_fulltext_knn_embeddings.npy",
        output_dir / ".paper_coords_tmp.npy",
    ]:
        _safe_unlink(path)
    for legacy in _legacy_output_paths(output_dir):
        _safe_unlink(legacy)


def _write_progress(
    output_dir: Path,
    *,
    status: str,
    phase: str,
    started_at: float,
    total_rows: int,
    processed_papers: int,
    temp_shards_written: int,
    committed_edge_rows: int,
    resumed_from_existing_temp: bool,
    embed_device: str,
    parquet_compression: str,
    temp_shard_rows: int,
    batch_rows: int,
    embed_batch_size: int,
    embedding_dim: int = 0,
    final_paper_nodes_written: int = 0,
    final_category_nodes_written: int = 0,
    paper_knn: int = 0,
    knn_edge_count: int = 0,
) -> None:
    payload = {
        "status": str(status),
        "phase": str(phase),
        "built_at": int(started_at),
        "elapsed_seconds": float(max(0.0, time.time() - started_at)),
        "total_rows": int(total_rows),
        "processed_papers": int(processed_papers),
        "remaining_papers": int(max(0, total_rows - processed_papers)),
        "temp_shards_written": int(temp_shards_written),
        "committed_edge_rows": int(committed_edge_rows),
        "resumed_from_existing_temp": bool(resumed_from_existing_temp),
        "temp_root": str(_temp_root(output_dir)),
        "embed_device": str(embed_device),
        "parquet_compression": str(parquet_compression),
        "temp_shard_rows": int(temp_shard_rows),
        "batch_rows": int(batch_rows),
        "embed_batch_size": int(embed_batch_size),
        "embedding_dim": int(embedding_dim),
        "final_paper_nodes_written": int(final_paper_nodes_written),
        "final_category_nodes_written": int(final_category_nodes_written),
        "paper_knn": int(paper_knn),
        "knn_edge_count": int(knn_edge_count),
        "resume_supported": True,
    }
    _atomic_json_write(_progress_path(output_dir), payload)


def _commit_temp_shard(
    output_dir: Path,
    *,
    shard_index: int,
    paper_rows: List[Dict[str, Any]],
    embedding_rows: List[Dict[str, Any]],
    edge_rows: List[Dict[str, Any]],
    paper_idx_start: int,
    compression: str,
) -> Dict[str, Any]:
    if len(paper_rows) != len(embedding_rows):
        raise RuntimeError("paper row buffer and embedding row buffer length mismatch")

    paths = _temp_paths(output_dir, shard_index)
    paths["paper_rows"].parent.mkdir(parents=True, exist_ok=True)

    _atomic_table_write(paths["paper_rows"], _paper_row_schema(), paper_rows, compression=compression)
    _atomic_table_write(paths["paper_embeddings"], _paper_embedding_schema(), embedding_rows, compression=compression)
    _atomic_table_write(paths["edges"], _edge_schema(), edge_rows, compression=compression)

    manifest = {
        "shard_index": int(shard_index),
        "paper_rows": int(len(paper_rows)),
        "edge_rows": int(len(edge_rows)),
        "paper_idx_start": int(paper_idx_start),
        "paper_idx_end": int(paper_idx_start + len(paper_rows)),
    }
    _atomic_json_write(paths["manifest"], manifest)
    return manifest


def _write_paper_knn_edges(
    *,
    embedding_paths: Sequence[Path],
    node_count: int,
    embedding_dim: int,
    output_path: Path,
    paper_knn: int,
    batch_rows: int,
    compression: str,
) -> Optional[Dict[str, Any]]:
    if int(paper_knn or 0) <= 0 or node_count <= 1:
        return {"enabled": False, "reason": "paper_knn_disabled_or_too_small"}

    try:
        import faiss  # type: ignore
    except Exception:
        return {"enabled": False, "reason": "faiss_not_available"}

    tmp_embeddings = output_path.parent / ".paper_knn_embeddings.npy"
    _safe_unlink(tmp_embeddings)
    vectors = np.lib.format.open_memmap(tmp_embeddings, mode="w+", dtype=np.float32, shape=(node_count, embedding_dim))
    cursor = 0
    for chunk in _iter_embedding_chunks(embedding_paths, batch_rows=max(1, int(batch_rows or 1))):
        end = cursor + int(chunk.shape[0])
        chunk = np.asarray(chunk, dtype=np.float32)
        faiss.normalize_L2(chunk)
        vectors[cursor:end] = chunk
        cursor = end
    vectors.flush()

    index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = max(80, int(paper_knn or 0) * 8)
    index.hnsw.efSearch = max(64, int(paper_knn or 0) * 8)
    for start, end in _chunk_slices(node_count, batch_rows):
        chunk = np.asarray(vectors[start:end], dtype=np.float32)
        index.add(chunk)

    tmp_output = output_path.parent / f".{output_path.name}.tmp"
    _safe_unlink(tmp_output)
    writer = _ParquetRowWriter(
        tmp_output,
        _paper_knn_schema(),
        batch_rows=max(1, int(batch_rows)),
        compression=compression,
    )
    edge_count = 0
    try:
        for start, end in _chunk_slices(node_count, batch_rows):
            query = np.asarray(vectors[start:end], dtype=np.float32)
            scores, indices = index.search(query, int(paper_knn) + 1)
            for local_idx in range(indices.shape[0]):
                src_idx = start + local_idx
                for dst_idx, score in zip(indices[local_idx], scores[local_idx]):
                    dst_idx = int(dst_idx)
                    if dst_idx < 0 or dst_idx == src_idx:
                        continue
                    writer.write(
                        {
                            "src_paper_idx": int(src_idx),
                            "dst_paper_idx": dst_idx,
                            "type": "paper_knn",
                            "weight": float(score),
                        }
                    )
                    edge_count += 1
    finally:
        writer.close()
    tmp_output.replace(output_path)
    _safe_unlink(tmp_embeddings)
    return {"enabled": True, "edge_count": edge_count, "backend": "faiss_index_hnsw_ip"}


def _aggregate_from_base_outputs(
    *,
    paper_nodes_path: Path,
    paper_embeddings_path: Path,
    batch_rows: int,
) -> Dict[str, Any]:
    node_pf = pq.ParquetFile(paper_nodes_path)
    emb_pf = pq.ParquetFile(paper_embeddings_path)
    node_iter = node_pf.iter_batches(
        columns=["paper_idx", "categories", "year", "x", "y", "z"],
        batch_size=max(1, int(batch_rows or 1)),
    )
    emb_iter = emb_pf.iter_batches(columns=["embedding"], batch_size=max(1, int(batch_rows or 1)))

    category_coord_sums: Dict[str, np.ndarray] = {}
    category_embedding_sums: Dict[str, np.ndarray] = {}
    category_counts: Dict[str, int] = {}
    year_coord_sums: Dict[int, np.ndarray] = {}
    year_counts: Dict[int, int] = {}
    processed_papers = 0
    embedding_dim = 0

    while True:
        try:
            node_batch = next(node_iter)
        except StopIteration:
            node_batch = None
        try:
            emb_batch = next(emb_iter)
        except StopIteration:
            emb_batch = None
        if node_batch is None and emb_batch is None:
            break
        if node_batch is None or emb_batch is None:
            raise RuntimeError("paper node and embedding parquet row streams are misaligned")

        node_rows = node_batch.to_pylist()
        embeddings = np.asarray(emb_batch.column(0).to_pylist(), dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if len(node_rows) != int(embeddings.shape[0]):
            raise RuntimeError("paper node and embedding batch size mismatch")
        if embeddings.size and embedding_dim <= 0:
            embedding_dim = int(embeddings.shape[1])

        for row, emb in zip(node_rows, embeddings):
            processed_papers += 1
            coord = np.asarray(
                [
                    float(row.get("x") or 0.0),
                    float(row.get("y") or 0.0),
                    float(row.get("z") or 0.0),
                ],
                dtype=np.float64,
            )
            raw_year = row.get("year")
            year = None if raw_year is None else int(raw_year)
            if year is not None:
                if year not in year_coord_sums:
                    year_coord_sums[year] = np.zeros((3,), dtype=np.float64)
                    year_counts[year] = 0
                year_coord_sums[year] += coord
                year_counts[year] += 1

            for category in row.get("categories") or []:
                category_id = str(category or "").strip()
                if not category_id:
                    continue
                if category_id not in category_coord_sums:
                    category_coord_sums[category_id] = np.zeros((3,), dtype=np.float64)
                    category_embedding_sums[category_id] = np.zeros((emb.shape[0],), dtype=np.float64)
                    category_counts[category_id] = 0
                category_coord_sums[category_id] += coord
                category_embedding_sums[category_id] += np.asarray(emb, dtype=np.float64)
                category_counts[category_id] += 1

    return {
        "processed_papers": int(processed_papers),
        "embedding_dim": int(embedding_dim),
        "category_coord_sums": category_coord_sums,
        "category_embedding_sums": category_embedding_sums,
        "category_counts": category_counts,
        "year_coord_sums": year_coord_sums,
        "year_counts": year_counts,
    }


def _write_year_nodes_and_edges(
    *,
    paper_nodes_path: Path,
    year_nodes_path: Path,
    year_edges_path: Path,
    year_coord_sums: Dict[int, np.ndarray],
    year_counts: Dict[int, int],
    batch_rows: int,
    compression: str,
) -> Dict[str, int]:
    year_nodes_tmp = year_nodes_path.parent / f".{year_nodes_path.name}.tmp"
    year_edges_tmp = year_edges_path.parent / f".{year_edges_path.name}.tmp"
    _safe_unlink(year_nodes_tmp)
    _safe_unlink(year_edges_tmp)

    year_writer = _ParquetRowWriter(
        year_nodes_tmp,
        _year_node_schema(),
        batch_rows=max(1, min(256, int(batch_rows or 1))),
        compression=compression,
    )
    year_node_count = 0
    try:
        for year in sorted(year_coord_sums):
            count = int(year_counts.get(year) or 0)
            if count > 0:
                coords = (year_coord_sums[year] / float(count)).astype(np.float32)
                x, y, z = [float(v) for v in coords.tolist()]
            else:
                x = y = z = 0.0
            year_writer.write(
                {
                    "year": int(year),
                    "paper_count": count,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            year_node_count += 1
    finally:
        year_writer.close()
    year_nodes_tmp.replace(year_nodes_path)

    edge_writer = _ParquetRowWriter(
        year_edges_tmp,
        _paper_year_edge_schema(),
        batch_rows=max(1024, int(batch_rows or 1)),
        compression=compression,
    )
    year_edge_count = 0
    try:
        for row in _iter_rows([paper_nodes_path], columns=["paper_idx", "year"], batch_rows=max(1, int(batch_rows or 1))):
            raw_year = row.get("year")
            if raw_year is None:
                continue
            edge_writer.write(
                {
                    "src_paper_idx": int(row.get("paper_idx") or 0),
                    "dst_year": int(raw_year),
                    "type": "has_year",
                }
            )
            year_edge_count += 1
    finally:
        edge_writer.close()
    year_edges_tmp.replace(year_edges_path)
    return {"year_node_count": int(year_node_count), "year_edge_count": int(year_edge_count)}


def _write_topic_nodes_and_edges(
    *,
    dataset_paths: Sequence[Path],
    paper_nodes_path: Path,
    topic_nodes_path: Path,
    topic_edges_path: Path,
    batch_rows: int,
    compression: str,
    max_topics_per_paper: int,
    max_rows: int = 0,
) -> Dict[str, int]:
    topic_nodes_tmp = topic_nodes_path.parent / f".{topic_nodes_path.name}.tmp"
    topic_edges_tmp = topic_edges_path.parent / f".{topic_edges_path.name}.tmp"
    _safe_unlink(topic_nodes_tmp)
    _safe_unlink(topic_edges_tmp)

    topic_coord_sums: Dict[str, np.ndarray] = {}
    topic_counts: Dict[str, int] = {}
    edge_writer = _ParquetRowWriter(
        topic_edges_tmp,
        _paper_topic_edge_schema(),
        batch_rows=max(1024, int(batch_rows or 1)),
        compression=compression,
    )
    topic_edge_count = 0
    try:
        node_iter = _iter_rows(
            [paper_nodes_path],
            columns=["paper_idx", "x", "y", "z"],
            batch_rows=max(1, int(batch_rows or 1)),
            max_rows=max_rows,
        )
        dataset_iter = _iter_rows(
            dataset_paths,
            columns=["title", "abstract"],
            batch_rows=max(1, int(batch_rows or 1)),
            max_rows=max_rows,
        )
        for node_row, paper_row in zip(node_iter, dataset_iter):
            topics = _extract_topic_terms(paper_row, max_topics=max_topics_per_paper)
            if not topics:
                continue
            coord = np.asarray(
                [
                    float(node_row.get("x") or 0.0),
                    float(node_row.get("y") or 0.0),
                    float(node_row.get("z") or 0.0),
                ],
                dtype=np.float64,
            )
            src_idx = int(node_row.get("paper_idx") or 0)
            for topic_id in topics:
                if topic_id not in topic_coord_sums:
                    topic_coord_sums[topic_id] = np.zeros((3,), dtype=np.float64)
                    topic_counts[topic_id] = 0
                topic_coord_sums[topic_id] += coord
                topic_counts[topic_id] += 1
                edge_writer.write(
                    {
                        "src_paper_idx": src_idx,
                        "dst_topic_id": topic_id,
                        "type": "has_topic",
                    }
                )
                topic_edge_count += 1
    finally:
        edge_writer.close()
    topic_edges_tmp.replace(topic_edges_path)

    topic_writer = _ParquetRowWriter(
        topic_nodes_tmp,
        _topic_node_schema(),
        batch_rows=max(1, min(256, int(batch_rows or 1))),
        compression=compression,
    )
    topic_node_count = 0
    try:
        for topic_idx, topic_id in enumerate(sorted(topic_coord_sums)):
            count = int(topic_counts.get(topic_id) or 0)
            if count > 0:
                coords = (topic_coord_sums[topic_id] / float(count)).astype(np.float32)
                x, y, z = [float(v) for v in coords.tolist()]
            else:
                x = y = z = 0.0
            topic_writer.write(
                {
                    "topic_idx": int(topic_idx),
                    "topic_id": topic_id,
                    "name": topic_id,
                    "paper_count": count,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            topic_node_count += 1
    finally:
        topic_writer.close()
    topic_nodes_tmp.replace(topic_nodes_path)
    return {"topic_node_count": int(topic_node_count), "topic_edge_count": int(topic_edge_count)}


def _write_fulltext_embeddings(
    *,
    dataset_paths: Sequence[Path],
    output_path: Path,
    batch_rows: int,
    embed_batch_size: int,
    device: str,
    compression: str,
    chunk_chars: int,
    max_chunks: int,
    max_rows: int = 0,
) -> Dict[str, Any]:
    if int(max_chunks or 0) <= 0:
        return {"enabled": False, "reason": "fulltext_embeddings_disabled"}
    tmp_output = output_path.parent / f".{output_path.name}.tmp"
    _safe_unlink(tmp_output)
    writer = _ParquetRowWriter(
        tmp_output,
        _paper_fulltext_embedding_schema(),
        batch_rows=max(32, min(256, int(batch_rows or 1))),
        compression=compression,
    )
    row_count = 0
    embedding_dim = 0
    rows_batch: List[Dict[str, Any]] = []
    try:
        for row in _iter_rows(
            dataset_paths,
            columns=["paper_id", "canonical_paper_id", "title", "abstract", "text"],
            batch_rows=max(1, int(batch_rows or 1)),
            max_rows=max_rows,
        ):
            rows_batch.append(row)
            if len(rows_batch) < max(1, int(batch_rows or 1)):
                continue
            embeddings = _embed_fulltext_rows(
                rows_batch,
                chunk_chars=chunk_chars,
                max_chunks=max_chunks,
                embed_batch_size=embed_batch_size,
                device=device,
            )
            if embeddings.size:
                embedding_dim = max(embedding_dim, int(embeddings.shape[1]))
            for local_idx, row_obj in enumerate(rows_batch):
                writer.write(
                    {
                        "paper_idx": int(row_count + local_idx),
                        "paper_id": str(row_obj.get("paper_id") or "").strip(),
                        "canonical_paper_id": str(row_obj.get("canonical_paper_id") or "").strip(),
                        "embedding": np.asarray(embeddings[local_idx], dtype=np.float32).tolist(),
                    }
                )
            row_count += len(rows_batch)
            rows_batch = []
        if rows_batch:
            embeddings = _embed_fulltext_rows(
                rows_batch,
                chunk_chars=chunk_chars,
                max_chunks=max_chunks,
                embed_batch_size=embed_batch_size,
                device=device,
            )
            if embeddings.size:
                embedding_dim = max(embedding_dim, int(embeddings.shape[1]))
            for local_idx, row_obj in enumerate(rows_batch):
                writer.write(
                    {
                        "paper_idx": int(row_count + local_idx),
                        "paper_id": str(row_obj.get("paper_id") or "").strip(),
                        "canonical_paper_id": str(row_obj.get("canonical_paper_id") or "").strip(),
                        "embedding": np.asarray(embeddings[local_idx], dtype=np.float32).tolist(),
                    }
                )
            row_count += len(rows_batch)
    finally:
        writer.close()
    tmp_output.replace(output_path)
    return {"enabled": True, "row_count": int(row_count), "embedding_dim": int(embedding_dim)}


def _write_category_knn_edges(
    *,
    category_embedding_sums: Dict[str, np.ndarray],
    category_counts: Dict[str, int],
    output_path: Path,
    category_knn: int,
    compression: str,
) -> Dict[str, Any]:
    if int(category_knn or 0) <= 0 or len(category_embedding_sums) <= 1:
        return {"enabled": False, "reason": "category_knn_disabled_or_too_small"}

    category_ids = sorted(category_embedding_sums)
    vectors = []
    for category_id in category_ids:
        count = max(1, int(category_counts.get(category_id) or 0))
        vec = np.asarray(category_embedding_sums[category_id], dtype=np.float32) / float(count)
        vectors.append(vec)
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms
    sims = mat @ mat.T

    top_k = max(1, min(int(category_knn), len(category_ids) - 1))
    tmp_output = output_path.parent / f".{output_path.name}.tmp"
    _safe_unlink(tmp_output)
    writer = _ParquetRowWriter(
        tmp_output,
        _category_knn_schema(),
        batch_rows=max(1, min(256, len(category_ids))),
        compression=compression,
    )
    edge_count = 0
    try:
        for i, src_id in enumerate(category_ids):
            order = np.argsort(-sims[i])
            for j in order[1 : top_k + 1]:
                dst_id = category_ids[int(j)]
                writer.write(
                    {
                        "src_category_id": src_id,
                        "dst_category_id": dst_id,
                        "type": "category_knn",
                        "weight": float(sims[i, int(j)]),
                    }
                )
                edge_count += 1
    finally:
        writer.close()
    tmp_output.replace(output_path)
    return {"enabled": True, "edge_count": int(edge_count), "backend": "exact_cosine"}


def build_paper_universe(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_papers: int = 0,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    temp_shard_rows: int = DEFAULT_TEMP_SHARD_ROWS,
    text_prefix_chars: int = DEFAULT_TEXT_PREFIX_CHARS,
    fulltext_chunk_chars: int = DEFAULT_FULLTEXT_CHUNK_CHARS,
    fulltext_max_chunks: int = DEFAULT_FULLTEXT_MAX_CHUNKS,
    max_topics_per_paper: int = DEFAULT_MAX_TOPICS_PER_PAPER,
    paper_knn: int = DEFAULT_PAPER_KNN,
    category_knn: int = DEFAULT_CATEGORY_KNN,
    pca_chunk_rows: int = DEFAULT_PCA_CHUNK_ROWS,
    embed_device: str = "",
    parquet_compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> Dict[str, Any]:
    started_at = time.time()
    dataset_dir_path = Path(dataset_dir).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    _temp_root(output_dir_path).mkdir(parents=True, exist_ok=True)
    _cleanup_incomplete_temp_files(_temp_root(output_dir_path))

    parquet_paths = _paper_parquet_paths(dataset_dir_path)
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found under {dataset_dir_path}")

    total_rows = _row_count(parquet_paths)
    if int(max_papers or 0) > 0:
        total_rows = min(total_rows, int(max_papers))
    if total_rows <= 0:
        raise RuntimeError("No paper rows available to build a paper universe.")

    resolved_embed_device = str(embed_device or "").strip()
    if not resolved_embed_device:
        resolved_embed_device = str(os.environ.get("EMBED_DEVICE") or "").strip() or "default"

    paper_nodes_path = output_dir_path / "paper_nodes.parquet"
    category_nodes_path = output_dir_path / "category_nodes.parquet"
    year_nodes_path = output_dir_path / "year_nodes.parquet"
    edges_path = output_dir_path / "edges.parquet"
    paper_year_edges_path = output_dir_path / "paper_year_edges.parquet"
    paper_embeddings_path = output_dir_path / "paper_embeddings.parquet"
    paper_fulltext_embeddings_path = output_dir_path / "paper_fulltext_embeddings.parquet"
    paper_knn_path = output_dir_path / "paper_knn_edges.parquet"
    category_knn_path = output_dir_path / "category_knn_edges.parquet"
    topic_nodes_path = output_dir_path / "topic_nodes.parquet"
    paper_topic_edges_path = output_dir_path / "paper_topic_edges.parquet"

    committed_manifests = _committed_temp_manifests(output_dir_path)
    processed_papers = int(sum(int(item.get("paper_rows") or 0) for item in committed_manifests))
    committed_edge_rows = int(sum(int(item.get("edge_rows") or 0) for item in committed_manifests))
    resumed_from_existing_temp = bool(committed_manifests)
    next_shard_index = len(committed_manifests)

    base_outputs_ready = (
        _path_row_count(paper_nodes_path) == total_rows
        and _path_row_count(paper_embeddings_path) == total_rows
        and edges_path.is_file()
    )
    reused_existing_base = bool(base_outputs_ready and not resumed_from_existing_temp)
    if reused_existing_base:
        processed_papers = total_rows
        committed_edge_rows = _path_row_count(edges_path)

    _write_progress(
        output_dir_path,
        status="running",
        phase="reuse_existing_base" if reused_existing_base else "source_embeddings",
        started_at=started_at,
        total_rows=total_rows,
        processed_papers=processed_papers,
        temp_shards_written=next_shard_index,
        committed_edge_rows=committed_edge_rows,
        resumed_from_existing_temp=resumed_from_existing_temp,
        embed_device=resolved_embed_device,
        parquet_compression=parquet_compression,
        temp_shard_rows=temp_shard_rows,
        batch_rows=batch_rows,
        embed_batch_size=embed_batch_size,
        paper_knn=paper_knn,
    )

    columns = [
        "paper_id",
        "canonical_paper_id",
        "paper_version",
        "title",
        "abstract",
        "authors",
        "categories",
        "license",
        "update_date",
        "text_char_count",
        "page_count",
        "token_count",
        "text_source",
        "text_is_partial",
        "metadata_found",
        "pdf_path",
        "text",
    ]

    paper_rows_buffer: List[Dict[str, Any]] = []
    embedding_rows_buffer: List[Dict[str, Any]] = []
    edge_rows_buffer: List[Dict[str, Any]] = []
    embedding_dim = 0

    def _flush_temp_buffers() -> None:
        nonlocal next_shard_index, processed_papers, committed_edge_rows, embedding_dim
        if not paper_rows_buffer:
            return
        manifest = _commit_temp_shard(
            output_dir_path,
            shard_index=next_shard_index,
            paper_rows=list(paper_rows_buffer),
            embedding_rows=list(embedding_rows_buffer),
            edge_rows=list(edge_rows_buffer),
            paper_idx_start=int(paper_rows_buffer[0]["paper_idx"]),
            compression=parquet_compression,
        )
        processed_papers += int(manifest["paper_rows"])
        committed_edge_rows += int(manifest["edge_rows"])
        next_shard_index += 1
        paper_rows_buffer.clear()
        embedding_rows_buffer.clear()
        edge_rows_buffer.clear()
        _write_progress(
            output_dir_path,
            status="running",
            phase="source_embeddings",
            started_at=started_at,
            total_rows=total_rows,
            processed_papers=processed_papers,
            temp_shards_written=next_shard_index,
            committed_edge_rows=committed_edge_rows,
            resumed_from_existing_temp=resumed_from_existing_temp,
            embed_device=resolved_embed_device,
            parquet_compression=parquet_compression,
            temp_shard_rows=temp_shard_rows,
            batch_rows=batch_rows,
            embed_batch_size=embed_batch_size,
            embedding_dim=embedding_dim,
            paper_knn=paper_knn,
        )

    if processed_papers < total_rows:
        rows_batch: List[Dict[str, Any]] = []
        for row in _iter_rows(
            parquet_paths,
            columns=columns,
            batch_rows=batch_rows,
            max_rows=total_rows,
            skip_rows=processed_papers,
        ):
            rows_batch.append(row)
            if len(rows_batch) < max(1, int(batch_rows or 1)):
                continue

            texts = [_embedding_text(item, text_prefix_chars=text_prefix_chars) for item in rows_batch]
            embeddings = embed_texts(
                texts,
                batch_size=max(1, int(embed_batch_size or 1)),
                device=str(embed_device or "").strip() or None,
            )
            if embedding_dim <= 0:
                embedding_dim = int(embeddings.shape[1])
            batch_start = processed_papers + len(paper_rows_buffer)
            for local_idx, row_obj in enumerate(rows_batch):
                paper_idx = batch_start + local_idx
                categories = _category_list(row_obj.get("categories"))
                paper_rows_buffer.append(
                    {
                        "paper_idx": int(paper_idx),
                        "paper_id": str(row_obj.get("paper_id") or "").strip(),
                        "canonical_paper_id": str(row_obj.get("canonical_paper_id") or "").strip(),
                        "paper_version": str(row_obj.get("paper_version") or "").strip(),
                        "title": str(row_obj.get("title") or "").strip(),
                        "authors": str(row_obj.get("authors") or "").strip(),
                        "categories": categories,
                        "primary_category": categories[0] if categories else "",
                        "license": str(row_obj.get("license") or "").strip(),
                        "update_date": str(row_obj.get("update_date") or "").strip(),
                        "year": _paper_year(str(row_obj.get("canonical_paper_id") or "").strip()),
                        "text_char_count": int(row_obj.get("text_char_count") or 0),
                        "page_count": int(row_obj.get("page_count") or 0),
                        "token_count": int(row_obj.get("token_count") or 0),
                        "text_source": str(row_obj.get("text_source") or "").strip(),
                        "text_is_partial": bool(row_obj.get("text_is_partial")),
                        "metadata_found": bool(row_obj.get("metadata_found")),
                        "pdf_path": str(row_obj.get("pdf_path") or "").strip(),
                    }
                )
                embedding_rows_buffer.append(
                    {
                        "paper_idx": int(paper_idx),
                        "paper_id": str(row_obj.get("paper_id") or "").strip(),
                        "canonical_paper_id": str(row_obj.get("canonical_paper_id") or "").strip(),
                        "embedding": np.asarray(embeddings[local_idx], dtype=np.float32).tolist(),
                    }
                )
                for category in categories:
                    edge_rows_buffer.append(
                        {
                            "src_paper_idx": int(paper_idx),
                            "dst_category_id": category,
                            "type": "has_category",
                        }
                    )
            rows_batch = []
            if len(paper_rows_buffer) >= max(1, int(temp_shard_rows or 1)):
                _flush_temp_buffers()

        if rows_batch:
            texts = [_embedding_text(item, text_prefix_chars=text_prefix_chars) for item in rows_batch]
            embeddings = embed_texts(
                texts,
                batch_size=max(1, int(embed_batch_size or 1)),
                device=str(embed_device or "").strip() or None,
            )
            if embedding_dim <= 0:
                embedding_dim = int(embeddings.shape[1])
            batch_start = processed_papers + len(paper_rows_buffer)
            for local_idx, row_obj in enumerate(rows_batch):
                paper_idx = batch_start + local_idx
                categories = _category_list(row_obj.get("categories"))
                paper_rows_buffer.append(
                    {
                        "paper_idx": int(paper_idx),
                        "paper_id": str(row_obj.get("paper_id") or "").strip(),
                        "canonical_paper_id": str(row_obj.get("canonical_paper_id") or "").strip(),
                        "paper_version": str(row_obj.get("paper_version") or "").strip(),
                        "title": str(row_obj.get("title") or "").strip(),
                        "authors": str(row_obj.get("authors") or "").strip(),
                        "categories": categories,
                        "primary_category": categories[0] if categories else "",
                        "license": str(row_obj.get("license") or "").strip(),
                        "update_date": str(row_obj.get("update_date") or "").strip(),
                        "year": _paper_year(str(row_obj.get("canonical_paper_id") or "").strip()),
                        "text_char_count": int(row_obj.get("text_char_count") or 0),
                        "page_count": int(row_obj.get("page_count") or 0),
                        "token_count": int(row_obj.get("token_count") or 0),
                        "text_source": str(row_obj.get("text_source") or "").strip(),
                        "text_is_partial": bool(row_obj.get("text_is_partial")),
                        "metadata_found": bool(row_obj.get("metadata_found")),
                        "pdf_path": str(row_obj.get("pdf_path") or "").strip(),
                    }
                )
                embedding_rows_buffer.append(
                    {
                        "paper_idx": int(paper_idx),
                        "paper_id": str(row_obj.get("paper_id") or "").strip(),
                        "canonical_paper_id": str(row_obj.get("canonical_paper_id") or "").strip(),
                        "embedding": np.asarray(embeddings[local_idx], dtype=np.float32).tolist(),
                    }
                )
                for category in categories:
                    edge_rows_buffer.append(
                        {
                            "src_paper_idx": int(paper_idx),
                            "dst_category_id": category,
                            "type": "has_category",
                        }
                    )
            rows_batch = []
        _flush_temp_buffers()

    committed_manifests = _committed_temp_manifests(output_dir_path)
    processed_papers = int(sum(int(item.get("paper_rows") or 0) for item in committed_manifests))
    committed_edge_rows = int(sum(int(item.get("edge_rows") or 0) for item in committed_manifests))
    if not reused_existing_base and processed_papers != total_rows:
        raise RuntimeError(f"paper row mismatch after temp commit: expected {total_rows}, have {processed_papers}")

    paper_row_shard_paths = [Path(item["_paths"]["paper_rows"]) for item in committed_manifests]
    embedding_shard_paths = [Path(item["_paths"]["paper_embeddings"]) for item in committed_manifests]
    edge_shard_paths = [Path(item["_paths"]["edges"]) for item in committed_manifests]

    if not reused_existing_base:
        _write_progress(
            output_dir_path,
            status="running",
            phase="pca",
            started_at=started_at,
            total_rows=total_rows,
            processed_papers=processed_papers,
            temp_shards_written=len(committed_manifests),
            committed_edge_rows=committed_edge_rows,
            resumed_from_existing_temp=resumed_from_existing_temp,
            embed_device=resolved_embed_device,
            parquet_compression=parquet_compression,
            temp_shard_rows=temp_shard_rows,
            batch_rows=batch_rows,
            embed_batch_size=embed_batch_size,
            paper_knn=paper_knn,
        )

        mean_vec, components, inferred_embedding_dim = _compute_pca_projection_from_embedding_shards(
            embedding_shard_paths,
            chunk_rows=pca_chunk_rows,
        )
        if inferred_embedding_dim <= 0:
            raise RuntimeError("paper embedding shards were not created")
        embedding_dim = inferred_embedding_dim

        _remove_final_outputs(output_dir_path)
        paper_nodes_tmp = output_dir_path / ".paper_nodes.parquet.tmp"
        paper_embeddings_tmp = output_dir_path / ".paper_embeddings.parquet.tmp"
        edges_tmp = output_dir_path / ".edges.parquet.tmp"

        edge_writer = _ParquetRowWriter(
            edges_tmp,
            _edge_schema(),
            batch_rows=max(1024, int(batch_rows or 1)),
            compression=parquet_compression,
        )
        try:
            for edge_path in edge_shard_paths:
                for row in _iter_rows([edge_path], columns=["src_paper_idx", "dst_category_id", "type"], batch_rows=max(1024, int(batch_rows or 1))):
                    edge_writer.write(row)
        finally:
            edge_writer.close()
        edges_tmp.replace(edges_path)

        _write_progress(
            output_dir_path,
            status="running",
            phase="finalizing_nodes",
            started_at=started_at,
            total_rows=total_rows,
            processed_papers=processed_papers,
            temp_shards_written=len(committed_manifests),
            committed_edge_rows=committed_edge_rows,
            resumed_from_existing_temp=resumed_from_existing_temp,
            embed_device=resolved_embed_device,
            parquet_compression=parquet_compression,
            temp_shard_rows=temp_shard_rows,
            batch_rows=batch_rows,
            embed_batch_size=embed_batch_size,
            embedding_dim=embedding_dim,
            paper_knn=paper_knn,
        )

        paper_nodes_writer = _ParquetRowWriter(
            paper_nodes_tmp,
            _paper_node_schema(),
            batch_rows=max(1, int(batch_rows or 1)),
            compression=parquet_compression,
        )
        embeddings_writer = _ParquetRowWriter(
            paper_embeddings_tmp,
            _paper_embedding_schema(),
            batch_rows=max(32, min(256, int(batch_rows or 1))),
            compression=parquet_compression,
        )
        final_paper_nodes_written = 0
        try:
            paper_row_columns = [
                "paper_idx",
                "paper_id",
                "canonical_paper_id",
                "paper_version",
                "title",
                "authors",
                "categories",
                "primary_category",
                "license",
                "update_date",
                "year",
                "text_char_count",
                "page_count",
                "token_count",
                "text_source",
                "text_is_partial",
                "metadata_found",
                "pdf_path",
            ]
            embedding_columns = ["paper_idx", "paper_id", "canonical_paper_id", "embedding"]
            for shard_idx, (paper_row_path, embedding_path) in enumerate(zip(paper_row_shard_paths, embedding_shard_paths)):
                paper_rows = list(_iter_rows([paper_row_path], columns=paper_row_columns, batch_rows=max(1, int(batch_rows or 1))))
                embedding_rows = list(_iter_rows([embedding_path], columns=embedding_columns, batch_rows=max(1, int(batch_rows or 1))))
                if len(paper_rows) != len(embedding_rows):
                    raise RuntimeError(f"temp shard row mismatch at shard {shard_idx}")

                if embedding_rows:
                    matrix = np.asarray([row["embedding"] for row in embedding_rows], dtype=np.float32)
                    if matrix.ndim == 1:
                        matrix = matrix.reshape(1, -1)
                    coords = (matrix - mean_vec) @ components
                else:
                    coords = np.zeros((0, 3), dtype=np.float32)

                for local_idx, (paper_row, embedding_row) in enumerate(zip(paper_rows, embedding_rows)):
                    coord = np.asarray(coords[local_idx], dtype=np.float32)
                    paper_nodes_writer.write(
                        {
                            **paper_row,
                            "x": float(coord[0]) if coord.shape[0] > 0 else 0.0,
                            "y": float(coord[1]) if coord.shape[0] > 1 else 0.0,
                            "z": float(coord[2]) if coord.shape[0] > 2 else 0.0,
                        }
                    )
                    embeddings_writer.write(embedding_row)
                    final_paper_nodes_written += 1

                _write_progress(
                    output_dir_path,
                    status="running",
                    phase="finalizing_nodes",
                    started_at=started_at,
                    total_rows=total_rows,
                    processed_papers=processed_papers,
                    temp_shards_written=len(committed_manifests),
                    committed_edge_rows=committed_edge_rows,
                    resumed_from_existing_temp=resumed_from_existing_temp,
                    embed_device=resolved_embed_device,
                    parquet_compression=parquet_compression,
                    temp_shard_rows=temp_shard_rows,
                    batch_rows=batch_rows,
                    embed_batch_size=embed_batch_size,
                    embedding_dim=embedding_dim,
                    final_paper_nodes_written=final_paper_nodes_written,
                    paper_knn=paper_knn,
                )
        finally:
            paper_nodes_writer.close()
            embeddings_writer.close()
        paper_nodes_tmp.replace(paper_nodes_path)
        paper_embeddings_tmp.replace(paper_embeddings_path)

    _write_progress(
        output_dir_path,
        status="running",
        phase="graph_layers",
        started_at=started_at,
        total_rows=total_rows,
        processed_papers=total_rows,
        temp_shards_written=len(committed_manifests),
        committed_edge_rows=max(committed_edge_rows, _path_row_count(edges_path)),
        resumed_from_existing_temp=resumed_from_existing_temp,
        embed_device=resolved_embed_device,
        parquet_compression=parquet_compression,
        temp_shard_rows=temp_shard_rows,
        batch_rows=batch_rows,
        embed_batch_size=embed_batch_size,
        paper_knn=paper_knn,
    )

    aggregates = _aggregate_from_base_outputs(
        paper_nodes_path=paper_nodes_path,
        paper_embeddings_path=paper_embeddings_path,
        batch_rows=max(1, int(batch_rows or 1)),
    )
    processed_papers = int(aggregates["processed_papers"])
    embedding_dim = int(aggregates["embedding_dim"])
    if processed_papers != total_rows:
        raise RuntimeError(f"paper row mismatch in final base outputs: expected {total_rows}, have {processed_papers}")

    category_nodes_tmp = output_dir_path / ".category_nodes.parquet.tmp"
    category_writer = _ParquetRowWriter(
        category_nodes_tmp,
        _category_node_schema(),
        batch_rows=max(1, min(256, int(batch_rows or 1))),
        compression=parquet_compression,
    )
    final_category_nodes_written = 0
    try:
        for category_idx, category_id in enumerate(sorted(aggregates["category_coord_sums"])):
            count = int(aggregates["category_counts"].get(category_id) or 0)
            if count > 0:
                coords = (aggregates["category_coord_sums"][category_id] / float(count)).astype(np.float32)
                x, y, z = [float(v) for v in coords.tolist()]
            else:
                x = y = z = 0.0
            category_writer.write(
                {
                    "category_idx": int(category_idx),
                    "category_id": category_id,
                    "name": category_id,
                    "paper_count": count,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            final_category_nodes_written += 1
    finally:
        category_writer.close()
    category_nodes_tmp.replace(category_nodes_path)

    year_info = _write_year_nodes_and_edges(
        paper_nodes_path=paper_nodes_path,
        year_nodes_path=year_nodes_path,
        year_edges_path=paper_year_edges_path,
        year_coord_sums=aggregates["year_coord_sums"],
        year_counts=aggregates["year_counts"],
        batch_rows=max(1, int(batch_rows or 1)),
        compression=parquet_compression,
    )

    topic_info = _write_topic_nodes_and_edges(
        dataset_paths=parquet_paths,
        paper_nodes_path=paper_nodes_path,
        topic_nodes_path=topic_nodes_path,
        topic_edges_path=paper_topic_edges_path,
        batch_rows=max(1, int(batch_rows or 1)),
        compression=parquet_compression,
        max_topics_per_paper=max_topics_per_paper,
        max_rows=total_rows,
    )

    category_knn_info = _write_category_knn_edges(
        category_embedding_sums=aggregates["category_embedding_sums"],
        category_counts=aggregates["category_counts"],
        output_path=category_knn_path,
        category_knn=int(category_knn or 0),
        compression=parquet_compression,
    )
    if not category_knn_info.get("enabled"):
        _safe_unlink(category_knn_path)

    fulltext_info: Dict[str, Any] = {"enabled": False, "reason": "fulltext_embeddings_disabled"}
    if int(fulltext_max_chunks or 0) > 0:
        existing_fulltext_rows = _path_row_count(paper_fulltext_embeddings_path)
        if existing_fulltext_rows == total_rows:
            fulltext_info = {
                "enabled": True,
                "row_count": int(existing_fulltext_rows),
                "embedding_dim": int(_embedding_dim_for_path(paper_fulltext_embeddings_path)),
                "reused_existing": True,
            }
        else:
            _write_progress(
                output_dir_path,
                status="running",
                phase="fulltext_embeddings",
                started_at=started_at,
                total_rows=total_rows,
                processed_papers=processed_papers,
                temp_shards_written=len(committed_manifests),
                committed_edge_rows=max(committed_edge_rows, _path_row_count(edges_path)),
                resumed_from_existing_temp=resumed_from_existing_temp,
                embed_device=resolved_embed_device,
                parquet_compression=parquet_compression,
                temp_shard_rows=temp_shard_rows,
                batch_rows=batch_rows,
                embed_batch_size=embed_batch_size,
                embedding_dim=embedding_dim,
                final_paper_nodes_written=processed_papers,
                final_category_nodes_written=final_category_nodes_written,
                paper_knn=paper_knn,
            )
            fulltext_info = _write_fulltext_embeddings(
                dataset_paths=parquet_paths,
                output_path=paper_fulltext_embeddings_path,
                batch_rows=max(1, int(batch_rows or 1)),
                embed_batch_size=max(1, int(embed_batch_size or 1)),
                device=resolved_embed_device,
                compression=parquet_compression,
                chunk_chars=max(256, int(fulltext_chunk_chars or DEFAULT_FULLTEXT_CHUNK_CHARS)),
                max_chunks=max(1, int(fulltext_max_chunks or 1)),
                max_rows=total_rows,
            )
    else:
        _safe_unlink(paper_fulltext_embeddings_path)

    _write_progress(
        output_dir_path,
        status="running",
        phase="paper_knn",
        started_at=started_at,
        total_rows=total_rows,
        processed_papers=processed_papers,
        temp_shards_written=len(committed_manifests),
        committed_edge_rows=max(committed_edge_rows, _path_row_count(edges_path)),
        resumed_from_existing_temp=resumed_from_existing_temp,
        embed_device=resolved_embed_device,
        parquet_compression=parquet_compression,
        temp_shard_rows=temp_shard_rows,
        batch_rows=batch_rows,
        embed_batch_size=embed_batch_size,
        embedding_dim=embedding_dim,
        final_paper_nodes_written=processed_papers,
        final_category_nodes_written=final_category_nodes_written,
        paper_knn=paper_knn,
    )

    knn_info = _write_paper_knn_edges(
        embedding_paths=[paper_embeddings_path],
        node_count=processed_papers,
        embedding_dim=embedding_dim,
        output_path=paper_knn_path,
        paper_knn=int(paper_knn or 0),
        batch_rows=max(1024, int(batch_rows or 1)),
        compression=parquet_compression,
    )
    if not isinstance(knn_info, dict):
        knn_info = {"enabled": False, "reason": "unknown"}
    if not knn_info.get("enabled"):
        _safe_unlink(paper_knn_path)

    manifest = {
        "built_at": int(time.time()),
        "dataset_dir": str(dataset_dir_path),
        "output_dir": str(output_dir_path),
        "paper_count": int(processed_papers),
        "category_count": int(final_category_nodes_written),
        "year_count": int(year_info.get("year_node_count") or 0),
        "embedding_dim": int(embedding_dim),
        "paper_nodes_path": str(paper_nodes_path),
        "category_nodes_path": str(category_nodes_path),
        "year_nodes_path": str(year_nodes_path),
        "edges_path": str(edges_path),
        "paper_year_edges_path": str(paper_year_edges_path),
        "paper_embeddings_path": str(paper_embeddings_path),
        "paper_embeddings_kind": "metadata_title_abstract",
        "paper_fulltext_embeddings_path": str(paper_fulltext_embeddings_path) if fulltext_info.get("enabled") else "",
        "paper_fulltext_embeddings": fulltext_info,
        "topic_nodes_path": str(topic_nodes_path),
        "paper_topic_edges_path": str(paper_topic_edges_path),
        "topic_count": int(topic_info.get("topic_node_count") or 0),
        "paper_topic_edges": int(topic_info.get("topic_edge_count") or 0),
        "paper_knn_path": str(paper_knn_path) if knn_info.get("enabled") else "",
        "category_knn_path": str(category_knn_path) if category_knn_info.get("enabled") else "",
        "paper_knn": knn_info,
        "category_knn": category_knn_info,
        "paper_year_edges": int(year_info.get("year_edge_count") or 0),
        "text_prefix_chars": int(text_prefix_chars or 0),
        "fulltext_chunk_chars": int(fulltext_chunk_chars or 0),
        "fulltext_max_chunks": int(fulltext_max_chunks or 0),
        "max_topics_per_paper": int(max_topics_per_paper or 0),
        "embed_device": resolved_embed_device,
        "parquet_compression": str(parquet_compression or DEFAULT_PARQUET_COMPRESSION),
        "stores_full_text": False,
        "paper_reference_fields": ["paper_id", "canonical_paper_id", "paper_version", "pdf_path"],
        "temp_shard_rows": int(temp_shard_rows or 0),
        "temp_shards_written": int(len(committed_manifests)),
        "resumed_from_existing_temp": resumed_from_existing_temp,
        "reused_existing_base": reused_existing_base,
    }
    _atomic_json_write(output_dir_path / "manifest.json", manifest)

    _write_progress(
        output_dir_path,
        status="completed",
        phase="completed",
        started_at=started_at,
        total_rows=total_rows,
        processed_papers=processed_papers,
        temp_shards_written=len(committed_manifests),
        committed_edge_rows=committed_edge_rows,
        resumed_from_existing_temp=resumed_from_existing_temp,
        embed_device=resolved_embed_device,
        parquet_compression=parquet_compression,
        temp_shard_rows=temp_shard_rows,
        batch_rows=batch_rows,
        embed_batch_size=embed_batch_size,
        embedding_dim=embedding_dim,
        final_paper_nodes_written=processed_papers,
        final_category_nodes_written=final_category_nodes_written,
        paper_knn=paper_knn,
        knn_edge_count=int(knn_info.get("edge_count") or 0),
    )

    if not reused_existing_base:
        _safe_rmtree(_temp_root(output_dir_path))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone paper universe from a paper-text parquet dataset.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument(
        "--temp-shard-rows",
        type=int,
        default=DEFAULT_TEMP_SHARD_ROWS,
        help="Number of papers per committed temp shard. Default: 16384.",
    )
    parser.add_argument(
        "--text-prefix-chars",
        type=int,
        default=DEFAULT_TEXT_PREFIX_CHARS,
        help="Optional number of prefix chars from full paper text to append after title+abstract. Default: 0.",
    )
    parser.add_argument(
        "--fulltext-chunk-chars",
        type=int,
        default=DEFAULT_FULLTEXT_CHUNK_CHARS,
        help="Approximate characters per full-text chunk when building full-text paper embeddings. Default: 2000.",
    )
    parser.add_argument(
        "--fulltext-max-chunks",
        type=int,
        default=DEFAULT_FULLTEXT_MAX_CHUNKS,
        help="Maximum number of full-text chunks per paper to average into paper_fulltext_embeddings.parquet. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--max-topics-per-paper",
        type=int,
        default=DEFAULT_MAX_TOPICS_PER_PAPER,
        help="Maximum number of heuristic topic terms to assign to each paper. Default: 3.",
    )
    parser.add_argument(
        "--paper-knn",
        type=int,
        default=DEFAULT_PAPER_KNN,
        help="Optional number of paper->paper similarity edges to compute with faiss. Default: 0.",
    )
    parser.add_argument(
        "--category-knn",
        type=int,
        default=DEFAULT_CATEGORY_KNN,
        help="Optional number of category->category similarity edges to compute. Default: 8.",
    )
    parser.add_argument("--pca-chunk-rows", type=int, default=DEFAULT_PCA_CHUNK_ROWS)
    parser.add_argument(
        "--embed-device",
        default="",
        help="Optional embedding device override such as 'cuda:1'. Default: EMBED_DEVICE env or automatic device.",
    )
    parser.add_argument(
        "--parquet-compression",
        default=DEFAULT_PARQUET_COMPRESSION,
        help="Compression codec for final parquet outputs. Default: zstd.",
    )
    args = parser.parse_args()

    result = build_paper_universe(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_papers=int(args.max_papers or 0),
        batch_rows=int(args.batch_rows or DEFAULT_BATCH_ROWS),
        embed_batch_size=int(args.embed_batch_size or DEFAULT_EMBED_BATCH_SIZE),
        temp_shard_rows=int(args.temp_shard_rows or DEFAULT_TEMP_SHARD_ROWS),
        text_prefix_chars=int(args.text_prefix_chars or 0),
        fulltext_chunk_chars=int(args.fulltext_chunk_chars or DEFAULT_FULLTEXT_CHUNK_CHARS),
        fulltext_max_chunks=int(args.fulltext_max_chunks or 0),
        max_topics_per_paper=int(args.max_topics_per_paper or DEFAULT_MAX_TOPICS_PER_PAPER),
        paper_knn=int(args.paper_knn or 0),
        category_knn=int(args.category_knn or 0),
        pca_chunk_rows=int(args.pca_chunk_rows or DEFAULT_PCA_CHUNK_ROWS),
        embed_device=str(args.embed_device or "").strip(),
        parquet_compression=str(args.parquet_compression or DEFAULT_PARQUET_COMPRESSION),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
