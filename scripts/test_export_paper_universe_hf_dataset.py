from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.export_paper_universe_hf_dataset import _render_dataset_card, export_paper_universe_hf_dataset


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def test_export_paper_universe_hf_dataset_writes_splits_and_assets(tmp_path: Path) -> None:
    universe_dir = tmp_path / "paper_universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    (universe_dir / "manifest.json").write_text(
        json.dumps(
            {
                "paper_count": 2,
                "category_count": 1,
                "year_count": 1,
                "topic_count": 1,
                "embedding_dim": 4,
                "paper_fulltext_embeddings": {"enabled": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (universe_dir / "progress.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (universe_dir / "viewer_manifest.json").write_text(json.dumps({"html_path": "universe_3d_hover.html"}), encoding="utf-8")
    (universe_dir / "render_manifest.json").write_text(json.dumps({"overview": "universe_3d.png"}), encoding="utf-8")
    (universe_dir / "universe_3d.png").write_bytes(b"png")
    (universe_dir / "universe_3d_detailed.png").write_bytes(b"png2")
    (universe_dir / "nodes_3d_sample.html").write_text("<html></html>", encoding="utf-8")
    (universe_dir / "universe_3d_hover.html").write_text("<html></html>", encoding="utf-8")
    interactive_dir = universe_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    (interactive_dir / "manifest.json").write_text(json.dumps({"levels": [50000]}), encoding="utf-8")
    (interactive_dir / "papers_50000.json").write_text("[]", encoding="utf-8")

    _write_parquet(
        universe_dir / "paper_nodes.parquet",
        [
            {
                "paper_idx": 0,
                "paper_id": "2401.00001",
                "canonical_paper_id": "2401.00001",
                "paper_version": "v1",
                "title": "Paper One",
                "authors": "Ada",
                "categories": ["cs.LG"],
                "primary_category": "cs.LG",
                "license": "cc-by-4.0",
                "update_date": "2024-01-01",
                "year": 2024,
                "text_char_count": 1000,
                "page_count": 5,
                "token_count": 200,
                "text_source": "raw_pdf",
                "text_is_partial": False,
                "metadata_found": True,
                "pdf_path": "/tmp/paper1.pdf",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
            }
        ],
    )
    _write_parquet(
        universe_dir / "edges.parquet",
        [{"src_paper_idx": 0, "dst_category_id": "cs.LG", "type": "has_category"}],
    )
    _write_parquet(
        universe_dir / "paper_knn_edges.parquet",
        [{"src_paper_idx": 0, "dst_paper_idx": 1, "type": "paper_knn", "weight": 0.9}],
    )
    _write_parquet(
        universe_dir / "category_nodes.parquet",
        [{"category_idx": 0, "category_id": "cs.LG", "name": "cs.LG", "paper_count": 2, "x": 0.0, "y": 0.1, "z": 0.2}],
    )
    _write_parquet(
        universe_dir / "category_knn_edges.parquet",
        [{"src_category_id": "cs.LG", "dst_category_id": "cs.AI", "type": "category_knn", "weight": 0.8}],
    )
    _write_parquet(
        universe_dir / "topic_nodes.parquet",
        [{"topic_idx": 0, "topic_id": "retrieval", "name": "retrieval", "paper_count": 1, "x": 0.2, "y": 0.3, "z": 0.4}],
    )
    _write_parquet(
        universe_dir / "paper_topic_edges.parquet",
        [{"src_paper_idx": 0, "dst_topic_id": "retrieval", "type": "has_topic"}],
    )
    _write_parquet(
        universe_dir / "year_nodes.parquet",
        [{"year": 2024, "paper_count": 2, "x": 0.4, "y": 0.5, "z": 0.6}],
    )
    _write_parquet(
        universe_dir / "paper_year_edges.parquet",
        [{"src_paper_idx": 0, "dst_year": 2024, "type": "has_year"}],
    )
    _write_parquet(
        universe_dir / "paper_embeddings.parquet",
        [{"paper_idx": 0, "paper_id": "2401.00001", "canonical_paper_id": "2401.00001", "embedding": [0.1, 0.2, 0.3, 0.4]}],
    )
    _write_parquet(
        universe_dir / "paper_fulltext_embeddings.parquet",
        [{"paper_idx": 0, "paper_id": "2401.00001", "canonical_paper_id": "2401.00001", "embedding": [0.4, 0.3, 0.2, 0.1]}],
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_universe_hf_dataset(
        universe_dir=str(universe_dir),
        output_dir=str(output_dir),
        materialize_mode="copy",
    )

    stats = result["stats"]
    assert stats["paper_count"] == 2
    assert stats["category_count"] == 1
    assert stats["year_count"] == 1
    assert stats["topic_count"] == 1
    assert stats["splits"]["paper_nodes"] == 1
    assert stats["splits"]["paper_category_edges"] == 1
    assert stats["splits"]["paper_knn"] == 1
    assert stats["splits"]["paper_embeddings"] == 1
    assert stats["splits"]["paper_fulltext_embeddings"] == 1
    assert "universe_3d.png" in stats["assets"]
    assert "papers_50000.json" in stats["interactive_files"]

    assert (output_dir / "parquet" / "paper_nodes.parquet").is_file()
    assert (output_dir / "parquet" / "paper_category_edges.parquet").is_file()
    assert (output_dir / "parquet" / "paper_knn.parquet").is_file()
    assert (output_dir / "parquet" / "paper_embeddings.parquet").is_file()
    assert (output_dir / "parquet" / "paper_fulltext_embeddings.parquet").is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "README.remote.md").is_file()
    assert (output_dir / "stats.json").is_file()
    assert (output_dir / "universe_3d.png").is_file()
    assert (output_dir / "interactive" / "papers_50000.json").is_file()

    node_rows = _rows(output_dir / "parquet" / "paper_nodes.parquet")
    assert node_rows[0]["paper_id"] == "2401.00001"
    assert abs(node_rows[0]["x"] - 0.1) < 1e-6
    edge_rows = _rows(output_dir / "parquet" / "paper_category_edges.parquet")
    assert edge_rows[0]["dst_category_id"] == "cs.LG"
    fulltext_rows = _rows(output_dir / "parquet" / "paper_fulltext_embeddings.parquet")
    assert fulltext_rows[0]["embedding"] == [0.4, 0.3, 0.2, 0.1]

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    remote_readme = (output_dir / "README.remote.md").read_text(encoding="utf-8")
    assert "Paper Universe Graph Dataset" in readme
    assert "universe_3d.png" in readme
    assert "interactive/" in readme
    assert 'path: "paper_nodes/*.parquet"' in remote_readme


def test_render_dataset_card_can_target_hub_config_directories() -> None:
    readme = _render_dataset_card(
        split_counts={"paper_nodes": 1, "paper_embeddings": 1},
        manifest={"paper_count": 1, "category_count": 1, "year_count": 1, "topic_count": 1, "embedding_dim": 4, "paper_fulltext_embeddings": {"enabled": True}},
        asset_names=[],
        interactive_files=[],
        path_template="{name}/*.parquet",
    )

    assert 'path: "paper_nodes/*.parquet"' in readme
    assert 'path: "paper_embeddings/*.parquet"' in readme
