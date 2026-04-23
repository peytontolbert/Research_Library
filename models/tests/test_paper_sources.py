from __future__ import annotations

from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.shared.data import _build_metadata_embedding_samples, build_dataset
from models.shared.training import Trainer


def _write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_build_dataset_uses_paper_text_parquet_for_full_text_models(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "paper_text"
    dataset_dir.mkdir()
    _write_parquet(
        dataset_dir / "train_00000.parquet",
        [
            {
                "paper_id": "2401.00001",
                "canonical_paper_id": "2401.00001",
                "paper_version": "v1",
                "pdf_path": "/tmp/2401.00001.pdf",
                "title": "Chunked Paper Training",
                "abstract": "We propose a chunk-friendly language-model training path.",
                "authors": "Ada Lovelace, Alan Turing",
                "categories": "cs.LG cs.AI",
                "license": "cc-by-4.0",
                "update_date": "2024-01-15",
                "metadata_found": True,
                "text": " ".join(f"Section {idx}: finding {idx}." for idx in range(120)),
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 960,
                "token_count": 200,
                "page_count": 8,
            }
        ],
    )
    config = {
        "model_id": "P1",
        "dataset": {
            "sources": ["paper_text_parquet"],
            "filters": {"years": [2020, 2025]},
            "construction": {
                "paper_dataset_dir": str(dataset_dir),
                "max_samples": 3,
                "chunk_chars_text": 120,
                "chunk_overlap_text": 20,
            },
        },
        "training": {"model_type": "seq2seq"},
    }

    samples = build_dataset(config)

    assert samples
    assert len(samples) == 3
    assert all(sample.get("text") for sample in samples)
    assert all(sample.get("target") for sample in samples)
    assert all(sample.get("target") != sample.get("text") for sample in samples)
    assert any("Chunked Paper Training" in sample["text"] for sample in samples)
    assert all(sample.get("paper_id") == "2401.00001" for sample in samples)


def test_metadata_embedding_samples_use_title_query_and_metadata_card() -> None:
    samples = _build_metadata_embedding_samples(
        [
            {
                "id": "2401.00001",
                "title": "Metadata Paper One",
                "abstract": "This paper studies metadata embeddings.",
                "categories": "cs.LG",
                "authors": ["Ada Lovelace"],
            },
            {
                "id": "2401.00002",
                "title": "Metadata Paper Two",
                "abstract": "This paper studies retrieval over metadata.",
                "categories": "cs.IR",
                "authors": ["Alan Turing"],
            },
        ],
        max_samples=4,
    )

    assert len(samples) == 4
    assert {int(sample["label"]) for sample in samples} == {0, 1}
    assert samples[0]["text_a"].startswith("TITLE:")
    assert samples[0]["text_b"].startswith("METADATA_CARD:")


def test_build_dataset_uses_paper_universe_graph_when_requested(tmp_path: Path) -> None:
    universe_dir = tmp_path / "paper_universe"
    universe_dir.mkdir()
    _write_parquet(
        universe_dir / "paper_nodes.parquet",
        [
            {
                "paper_idx": 1,
                "paper_id": "2401.00001",
                "canonical_paper_id": "2401.00001",
                "title": "Paper One",
                "authors": "Ada",
                "primary_category": "cs.LG",
                "categories": ["cs.LG", "cs.AI"],
                "pdf_path": "/tmp/2401.00001.pdf",
            },
            {
                "paper_idx": 2,
                "paper_id": "2401.00002",
                "canonical_paper_id": "2401.00002",
                "title": "Paper Two",
                "authors": "Alan",
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "pdf_path": "/tmp/2401.00002.pdf",
            },
            {
                "paper_idx": 3,
                "paper_id": "2401.00003",
                "canonical_paper_id": "2401.00003",
                "title": "Paper Three",
                "authors": "Grace",
                "primary_category": "math.OC",
                "categories": ["math.OC"],
                "pdf_path": "/tmp/2401.00003.pdf",
            },
        ],
    )
    _write_parquet(
        universe_dir / "paper_knn_edges.parquet",
        [
            {"src_paper_idx": 1, "dst_paper_idx": 2, "type": "paper_knn", "weight": 0.92},
            {"src_paper_idx": 2, "dst_paper_idx": 3, "type": "paper_knn", "weight": 0.73},
        ],
    )
    config = {
        "model_id": "M4",
        "dataset": {
            "sources": ["paper_universe_graph"],
            "construction": {
                "paper_universe_dir": str(universe_dir),
                "max_samples": 4,
            },
        },
        "training": {"objective": "link_prediction"},
    }

    samples = build_dataset(config)

    assert samples
    assert all(sample.get("domain") == "paper" for sample in samples)
    assert all(sample.get("repo_id") == "paper_universe" for sample in samples)
    assert any(sample.get("label") == 1 for sample in samples)
    assert all(sample.get("text", "").startswith("PAPER GRAPH") for sample in samples)


def test_trainer_build_hf_dataset_uses_direct_paper_parquet_for_p1(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "paper_text"
    dataset_dir.mkdir()
    _write_parquet(
        dataset_dir / "train_00000.parquet",
        [
            {
                "paper_id": "2402.00001",
                "canonical_paper_id": "2402.00001",
                "paper_version": "v1",
                "pdf_path": "/tmp/2402.00001.pdf",
                "title": "Direct HF Dataset",
                "abstract": "This paper explains direct parquet-backed training.",
                "authors": "Ada Lovelace",
                "categories": "cs.LG",
                "license": "cc-by-4.0",
                "update_date": "2024-02-01",
                "metadata_found": True,
                "text": " ".join(f"Body section {idx}." for idx in range(120)),
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 900,
                "token_count": 180,
                "page_count": 6,
            }
        ],
    )
    config = {
        "model_id": "P1",
        "dataset": {
            "sources": ["paper_text_parquet"],
            "filters": {"years": [2020, 2025]},
            "construction": {
                "paper_dataset_dir": str(dataset_dir),
                "max_samples": 1,
                "chunk_chars_text": 150,
                "chunk_overlap_text": 25,
            },
        },
        "training": {"model_type": "seq2seq"},
    }

    ds = Trainer(config, model_stub=None)._build_hf_dataset()

    assert ds is not None
    assert len(ds) > 1
    first = ds[0]
    assert "text" in first
    assert first["target"] != first["text"]
    assert "Direct HF Dataset" in first["text"]
    assert first["paper_id"] == "2402.00001"


def test_trainer_build_hf_dataset_uses_direct_paper_parquet_for_m6_pairs_with_teacher_embeddings(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "paper_text"
    dataset_dir.mkdir()
    universe_dir = tmp_path / "paper_universe"
    universe_dir.mkdir()
    _write_parquet(
        dataset_dir / "train_00000.parquet",
        [
            {
                "paper_id": "2403.00001",
                "canonical_paper_id": "2403.00001",
                "paper_version": "v1",
                "pdf_path": "/tmp/2403.00001.pdf",
                "title": "Paper One",
                "abstract": "Abstract one.",
                "authors": "Ada",
                "categories": "cs.LG",
                "license": "cc-by-4.0",
                "update_date": "2024-03-01",
                "metadata_found": True,
                "text": "Text one.",
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 50,
                "token_count": 10,
                "page_count": 1,
            },
            {
                "paper_id": "2403.00002",
                "canonical_paper_id": "2403.00002",
                "paper_version": "v1",
                "pdf_path": "/tmp/2403.00002.pdf",
                "title": "Paper Two",
                "abstract": "Abstract two.",
                "authors": "Alan",
                "categories": "cs.AI",
                "license": "cc-by-4.0",
                "update_date": "2024-03-02",
                "metadata_found": True,
                "text": "Text two.",
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 50,
                "token_count": 10,
                "page_count": 1,
            },
            {
                "paper_id": "2403.00003",
                "canonical_paper_id": "2403.00003",
                "paper_version": "v1",
                "pdf_path": "/tmp/2403.00003.pdf",
                "title": "Paper Three",
                "abstract": "Abstract three.",
                "authors": "Grace",
                "categories": "math.OC",
                "license": "cc-by-4.0",
                "update_date": "2024-03-03",
                "metadata_found": True,
                "text": "Text three.",
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 50,
                "token_count": 10,
                "page_count": 1,
            },
        ],
    )
    _write_parquet(
        universe_dir / "paper_fulltext_embeddings.parquet",
        [
            {"paper_idx": 0, "paper_id": "2403.00001", "canonical_paper_id": "2403.00001", "embedding": [0.1, 0.2, 0.3, 0.4]},
            {"paper_idx": 1, "paper_id": "2403.00002", "canonical_paper_id": "2403.00002", "embedding": [0.5, 0.6, 0.7, 0.8]},
            {"paper_idx": 2, "paper_id": "2403.00003", "canonical_paper_id": "2403.00003", "embedding": [0.9, 1.0, 1.1, 1.2]},
        ],
    )
    config = {
        "model_id": "M6",
        "dataset": {
            "sources": ["paper_text_parquet"],
            "filters": {"years": [2020, 2025]},
            "quality_filters": {"papers_min_chars": 16, "require_full_text": False},
            "construction": {
                "paper_dataset_dir": str(dataset_dir),
                "paper_universe_dir": str(universe_dir),
                "teacher_embeddings": "fulltext",
                "document_chars": 256,
                "max_chunks_per_paper": 1,
                "max_samples": 3,
            },
        },
        "training": {"objective": "contrastive", "distillation_weight": 0.2},
    }

    ds = Trainer(config, model_stub=None)._build_hf_dataset()

    assert ds is not None
    assert len(ds) >= 4
    labels = {int(example["label"]) for example in ds}
    assert labels == {0, 1}
    first = ds[0]
    assert "text_a" in first and "text_b" in first
    assert "TITLE:" in first["text_a"]
    assert first["text_b"].startswith("PAPER_DOCUMENT:") or first["text_b"].startswith("PAPER_SPAN:")
    assert first["teacher_mask"] == 1
    assert len(first["teacher_embedding"]) == 4


def test_trainer_build_hf_dataset_uses_sentence_pairs_for_m7(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "paper_text"
    dataset_dir.mkdir()
    _write_parquet(
        dataset_dir / "train_00000.parquet",
        [
            {
                "paper_id": "2406.00001",
                "canonical_paper_id": "2406.00001",
                "paper_version": "v1",
                "pdf_path": "/tmp/2406.00001.pdf",
                "title": "Sentence Grounding Paper One",
                "abstract": "We study sentence retrieval. Our method links abstract claims to evidence.",
                "authors": "Ada",
                "categories": "cs.CL",
                "license": "cc-by-4.0",
                "update_date": "2024-06-01",
                "metadata_found": True,
                "text": "Our method links abstract claims to evidence sentences in the paper body. Sentence retrieval improves grounding and QA.",
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 180,
                "token_count": 36,
                "page_count": 1,
            },
            {
                "paper_id": "2406.00002",
                "canonical_paper_id": "2406.00002",
                "paper_version": "v1",
                "pdf_path": "/tmp/2406.00002.pdf",
                "title": "Sentence Grounding Paper Two",
                "abstract": "We analyze citation structure. Our study focuses on graph edges.",
                "authors": "Alan",
                "categories": "cs.IR",
                "license": "cc-by-4.0",
                "update_date": "2024-06-02",
                "metadata_found": True,
                "text": "This study focuses on graph edges and citation analysis. It does not discuss grounding evidence.",
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 150,
                "token_count": 30,
                "page_count": 1,
            },
        ],
    )
    config = {
        "model_id": "M7",
        "dataset": {
            "sources": ["paper_text_parquet"],
            "quality_filters": {"papers_min_chars": 64},
            "construction": {
                "paper_dataset_dir": str(dataset_dir),
                "max_query_sentences": 2,
                "max_body_sentences": 6,
                "max_samples": 4,
            },
        },
        "training": {"objective": "contrastive"},
    }

    ds = Trainer(config, model_stub=None)._build_hf_dataset()

    assert ds is not None
    assert len(ds) >= 4
    labels = {int(example["label"]) for example in ds}
    assert labels == {0, 1}
    first = ds[0]
    assert "ABSTRACT_SENTENCE:" in first["text_a"]
    assert first["text_b"].startswith("PAPER_SENTENCE:")


def test_trainer_build_hf_dataset_uses_keyword_targets_for_a3(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "paper_text"
    dataset_dir.mkdir()
    _write_parquet(
        dataset_dir / "train_00000.parquet",
        [
            {
                "paper_id": "2404.00001",
                "canonical_paper_id": "2404.00001",
                "paper_version": "v1",
                "pdf_path": "/tmp/2404.00001.pdf",
                "title": "Graph Neural Networks for Scientific Retrieval",
                "abstract": "We propose a graph neural network retrieval system for scientific papers and citation search.",
                "authors": "Ada",
                "categories": "cs.LG cs.IR",
                "license": "cc-by-4.0",
                "update_date": "2024-04-01",
                "metadata_found": True,
                "text": "Our graph neural retrieval model improves citation search and scientific document ranking." * 8,
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 900,
                "token_count": 180,
                "page_count": 6,
            }
        ],
    )
    config = {
        "model_id": "A3",
        "dataset": {
            "sources": ["paper_text_parquet"],
            "construction": {
                "paper_dataset_dir": str(dataset_dir),
                "max_samples": 1,
            },
        },
        "training": {"model_type": "seq2seq"},
    }

    ds = Trainer(config, model_stub=None)._build_hf_dataset()

    assert ds is not None
    first = ds[0]
    assert first["text"].startswith("We propose")
    assert "graph" in first["target"]
    assert "," in first["target"]


def test_trainer_build_hf_dataset_uses_paper_qa_targets_for_p5(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "paper_text"
    dataset_dir.mkdir()
    _write_parquet(
        dataset_dir / "train_00000.parquet",
        [
            {
                "paper_id": "2405.00001",
                "canonical_paper_id": "2405.00001",
                "paper_version": "v1",
                "pdf_path": "/tmp/2405.00001.pdf",
                "title": "Retriever-Augmented Paper QA",
                "abstract": "We study question answering over scientific papers and propose a retrieval-augmented method.",
                "authors": "Ada",
                "categories": "cs.CL cs.IR",
                "license": "cc-by-4.0",
                "update_date": "2024-05-01",
                "metadata_found": True,
                "text": (
                    "We propose a retrieval-augmented question answering method for papers. "
                    "Our experiments show improved answer accuracy over baseline systems. "
                    "The method uses abstract and paragraph retrieval before generation."
                ),
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "text_char_count": 420,
                "token_count": 90,
                "page_count": 3,
            }
        ],
    )
    config = {
        "model_id": "P5",
        "dataset": {
            "sources": ["paper_text_parquet"],
            "construction": {
                "paper_dataset_dir": str(dataset_dir),
                "max_samples": 3,
            },
        },
        "training": {"model_type": "seq2seq"},
    }

    ds = Trainer(config, model_stub=None)._build_hf_dataset()

    assert ds is not None
    assert len(ds) == 3
    first = ds[0]
    assert "question" in first and "context" in first and "target" in first
    assert "Retriever-Augmented Paper QA" in first["question"]
    assert "Title:" in first["context"]
    assert len(first["target"]) > 20
