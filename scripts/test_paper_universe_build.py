from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pytest

from scripts.paper_universe_build import (
    _commit_temp_shard,
    _progress_path,
    _temp_root,
    build_paper_universe,
)


def _rows(path: Path):
    table = pq.read_table(path)
    return table.to_pylist()


def _make_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "paper_id": ["2401.00001v1", "2401.00002v1", "2402.00003v1"],
            "canonical_paper_id": ["2401.00001", "2401.00002", "2402.00003"],
            "paper_version": ["v1", "v1", "v1"],
            "pdf_path": [
                "/arxiv/pdfs/2401/2401.00001v1.pdf",
                "/arxiv/pdfs/2401/2401.00002v1.pdf",
                "/arxiv/pdfs/2402/2402.00003v1.pdf",
            ],
            "title": ["paper one", "paper two", "paper three"],
            "abstract": ["abstract one", "abstract two", "abstract three"],
            "authors": ["a", "b", "c"],
            "categories": ["cs.LG cs.AI", "cs.AI", "math.OC"],
            "license": ["cc-by", "cc-by", "nonexclusive"],
            "update_date": ["2024-01-01", "2024-01-02", "2024-02-01"],
            "version_count": [1, 1, 1],
            "metadata_found": [True, True, True],
            "text": ["full text one", "full text two", "full text three"],
            "text_source": ["raw_pdf_preferred", "raw_pdf_preferred", "raw_pdf_preferred"],
            "text_is_partial": [False, False, False],
            "text_char_count": [100, 120, 140],
            "text_line_count": [10, 12, 14],
            "token_count": [20, 24, 28],
            "page_count": [5, 6, 7],
            "token_types": [["raw_text_pdf"], ["raw_text_pdf"], ["raw_text_pdf"]],
            "token_type_counts_json": ["{}", "{}", "{}"],
        }
    )
    pq.write_table(table, dataset_dir / "train_00000.parquet")


def _fake_embed_texts(texts, batch_size=None, device=None):
    rows = []
    for idx, text in enumerate(texts):
        base = float(len(str(text)))
        rows.append([base, base + 1.0, float(idx) + 1.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def test_build_paper_universe_writes_lightweight_parquet_outputs(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "paper_dataset"
    _make_dataset(dataset_dir)
    monkeypatch.setattr("scripts.paper_universe_build.embed_texts", _fake_embed_texts)

    output_dir = tmp_path / "paper_universe"
    pytest.importorskip("faiss")

    result = build_paper_universe(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        batch_rows=2,
        embed_batch_size=2,
        temp_shard_rows=2,
        fulltext_max_chunks=2,
        paper_knn=1,
        category_knn=1,
        embed_device="cuda:1",
    )

    assert result["paper_count"] == 3
    assert result["category_count"] == 3
    assert result["stores_full_text"] is False
    assert result["embed_device"] == "cuda:1"

    paper_nodes = _rows(output_dir / "paper_nodes.parquet")
    assert len(paper_nodes) == 3
    assert paper_nodes[0]["paper_idx"] == 0
    assert paper_nodes[0]["canonical_paper_id"] == "2401.00001"
    assert paper_nodes[0]["primary_category"] == "cs.LG"
    assert "abstract" not in paper_nodes[0]
    assert "text" not in paper_nodes[0]
    assert {"x", "y", "z"} <= set(paper_nodes[0].keys())

    category_nodes = _rows(output_dir / "category_nodes.parquet")
    category_ids = {row["category_id"] for row in category_nodes}
    assert category_ids == {"cs.AI", "cs.LG", "math.OC"}
    category_counts = {row["category_id"]: row["paper_count"] for row in category_nodes}
    assert category_counts["cs.LG"] == 1
    assert category_counts["cs.AI"] == 2
    assert category_counts["math.OC"] == 1

    edges = _rows(output_dir / "edges.parquet")
    assert len(edges) == 4
    assert edges[0]["type"] == "has_category"

    embedding_rows = _rows(output_dir / "paper_embeddings.parquet")
    assert len(embedding_rows) == 3
    assert len(embedding_rows[0]["embedding"]) == 4

    year_nodes = _rows(output_dir / "year_nodes.parquet")
    assert len(year_nodes) == 1
    assert year_nodes[0]["year"] == 2024
    paper_year_edges = _rows(output_dir / "paper_year_edges.parquet")
    assert len(paper_year_edges) == 3

    category_knn_edges = _rows(output_dir / "category_knn_edges.parquet")
    assert len(category_knn_edges) > 0
    paper_knn_edges = _rows(output_dir / "paper_knn_edges.parquet")
    assert len(paper_knn_edges) > 0
    topic_nodes = _rows(output_dir / "topic_nodes.parquet")
    assert len(topic_nodes) > 0
    paper_topic_edges = _rows(output_dir / "paper_topic_edges.parquet")
    assert len(paper_topic_edges) > 0
    fulltext_embedding_rows = _rows(output_dir / "paper_fulltext_embeddings.parquet")
    assert len(fulltext_embedding_rows) == 3
    assert len(fulltext_embedding_rows[0]["embedding"]) == 4

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_count"] == 3
    assert manifest["paper_knn"]["enabled"] is True
    assert manifest["category_knn"]["enabled"] is True
    assert manifest["paper_nodes_path"].endswith("paper_nodes.parquet")
    assert manifest["paper_embeddings_path"].endswith("paper_embeddings.parquet")
    assert manifest["paper_fulltext_embeddings_path"].endswith("paper_fulltext_embeddings.parquet")
    assert manifest["year_nodes_path"].endswith("year_nodes.parquet")
    assert manifest["topic_nodes_path"].endswith("topic_nodes.parquet")
    assert manifest["paper_topic_edges_path"].endswith("paper_topic_edges.parquet")

    progress = json.loads(_progress_path(output_dir).read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["resume_supported"] is True

    assert not _temp_root(output_dir).exists()


def test_build_paper_universe_resumes_from_committed_temp_shards(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "paper_dataset"
    _make_dataset(dataset_dir)
    monkeypatch.setattr("scripts.paper_universe_build.embed_texts", _fake_embed_texts)

    output_dir = tmp_path / "paper_universe_resume"
    temp_root = _temp_root(output_dir)
    temp_root.mkdir(parents=True, exist_ok=True)

    first_rows = [
        {
            "paper_idx": 0,
            "paper_id": "2401.00001v1",
            "canonical_paper_id": "2401.00001",
            "paper_version": "v1",
            "title": "paper one",
            "authors": "a",
            "categories": ["cs.LG", "cs.AI"],
            "primary_category": "cs.LG",
            "license": "cc-by",
            "update_date": "2024-01-01",
            "year": 2024,
            "text_char_count": 100,
            "page_count": 5,
            "token_count": 20,
            "text_source": "raw_pdf_preferred",
            "text_is_partial": False,
            "metadata_found": True,
            "pdf_path": "/arxiv/pdfs/2401/2401.00001v1.pdf",
        },
        {
            "paper_idx": 1,
            "paper_id": "2401.00002v1",
            "canonical_paper_id": "2401.00002",
            "paper_version": "v1",
            "title": "paper two",
            "authors": "b",
            "categories": ["cs.AI"],
            "primary_category": "cs.AI",
            "license": "cc-by",
            "update_date": "2024-01-02",
            "year": 2024,
            "text_char_count": 120,
            "page_count": 6,
            "token_count": 24,
            "text_source": "raw_pdf_preferred",
            "text_is_partial": False,
            "metadata_found": True,
            "pdf_path": "/arxiv/pdfs/2401/2401.00002v1.pdf",
        },
    ]
    first_embeddings = [
        {
            "paper_idx": 0,
            "paper_id": "2401.00001v1",
            "canonical_paper_id": "2401.00001",
            "embedding": [1.0, 2.0, 1.0, 1.0],
        },
        {
            "paper_idx": 1,
            "paper_id": "2401.00002v1",
            "canonical_paper_id": "2401.00002",
            "embedding": [2.0, 3.0, 2.0, 1.0],
        },
    ]
    first_edges = [
        {"src_paper_idx": 0, "dst_category_id": "cs.LG", "type": "has_category"},
        {"src_paper_idx": 0, "dst_category_id": "cs.AI", "type": "has_category"},
        {"src_paper_idx": 1, "dst_category_id": "cs.AI", "type": "has_category"},
    ]
    _commit_temp_shard(
        output_dir,
        shard_index=0,
        paper_rows=first_rows,
        embedding_rows=first_embeddings,
        edge_rows=first_edges,
        paper_idx_start=0,
        compression="zstd",
    )

    result = build_paper_universe(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        batch_rows=2,
        embed_batch_size=2,
        temp_shard_rows=2,
        fulltext_max_chunks=2,
        paper_knn=0,
        category_knn=1,
        embed_device="cuda:1",
    )

    assert result["paper_count"] == 3
    assert result["resumed_from_existing_temp"] is True

    paper_nodes = _rows(output_dir / "paper_nodes.parquet")
    assert len(paper_nodes) == 3

    progress = json.loads(_progress_path(output_dir).read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["resumed_from_existing_temp"] is True
    assert progress["processed_papers"] == 3
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["reused_existing_base"] is False

    assert not _temp_root(output_dir).exists()
