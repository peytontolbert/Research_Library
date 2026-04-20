from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.export_library_repo_graph_hf_dataset import export_library_repo_graph_hf_dataset


def _rows(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return table.to_pylist()


def test_export_library_repo_graph_hf_dataset_writes_repo_and_universe_parquet(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "repos": {
            "demo": {
                "export_schema_version": 2,
                "repo_state": {"vcs": "git", "head": "abc123", "branch": "main"},
                "last_indexed_at": 123,
                "languages": ["python", "markdown"],
                "indices": {
                    "qa": {
                        "type": "simple_numpy",
                        "embeddings_path": str(export_root / "demo" / "indices" / "qa.npy"),
                    }
                },
                "skills": {
                    "qa": {"status": "up_to_date"}
                },
                "extensions": {
                    "repo_skills_miner": {
                        "paths": {"summary": "demo/structured/summary.json"}
                    }
                },
            },
            "other": {
                "export_schema_version": 2,
                "repo_state": {"vcs": "none", "snapshot_mtime": 99},
                "last_indexed_at": 456,
                "languages": ["rust"],
                "indices": {},
                "skills": {},
                "extensions": {},
            },
        }
    }
    (export_root / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    demo_dir = export_root / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "demo.entities.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "repo_id": "demo",
                        "id": "py:demo",
                        "uri": "program://demo/module/demo#L1-L10",
                        "kind": "module",
                        "name": "demo",
                        "owner": None,
                    }
                ),
                json.dumps(
                    {
                        "repo_id": "demo",
                        "id": "py:demo.fn",
                        "uri": "program://demo/function/demo.fn#L2-L3",
                        "kind": "function",
                        "name": "fn",
                        "owner": "py:demo",
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (demo_dir / "demo.edges.jsonl").write_text(
        json.dumps({"repo_id": "demo", "src": "py:demo", "dst": "py:demo.fn", "type": "owns"}) + "\n",
        encoding="utf-8",
    )
    (demo_dir / "demo.artifacts.jsonl").write_text(
        json.dumps({"repo_id": "demo", "uri": "program://demo/artifact/demo.py", "type": "source", "hash": "abc"}) + "\n",
        encoding="utf-8",
    )

    other_dir = export_root / "other"
    other_dir.mkdir(parents=True, exist_ok=True)
    (other_dir / "other.entities.jsonl").write_text(
        json.dumps(
            {
                "repo_id": "other",
                "id": "rs:lib",
                "uri": "program://other/module/lib#L1-L5",
                "kind": "module",
                "name": "lib",
                "owner": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (other_dir / "other.edges.jsonl").write_text(
        json.dumps({"repo_id": "other", "src": "rs:lib", "dst": "rs:lib", "type": "self"}) + "\n",
        encoding="utf-8",
    )
    (other_dir / "other.artifacts.jsonl").write_text(
        json.dumps({"repo_id": "other", "uri": "program://other/artifact/lib.rs", "type": "source", "hash": "def"}) + "\n",
        encoding="utf-8",
    )

    universe_root = export_root / "_universe"
    universe_root.mkdir(parents=True, exist_ok=True)
    (universe_root / "manifest.json").write_text(
        json.dumps(
            {
                "node_count": 2,
                "repo_ids": ["demo"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (universe_root / "nodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "node_id": "demo:repo",
                        "repo_id": "demo",
                        "kind": "repo",
                        "name": "demo",
                        "uri": "",
                        "labels": ["python"],
                    }
                ),
                json.dumps(
                    {
                        "node_id": "demo:py:demo",
                        "repo_id": "demo",
                        "kind": "module",
                        "name": "demo",
                        "uri": "program://demo/module/demo#L1-L10",
                        "labels": None,
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (universe_root / "edges.jsonl").write_text(
        json.dumps({"repo_id": "demo", "src": "demo:repo", "dst": "demo:py:demo", "type": "owns"}) + "\n",
        encoding="utf-8",
    )
    (universe_root / "repo_knn_edges.jsonl").write_text(
        json.dumps({"src_repo": "demo", "dst_repo": "other", "weight": 0.75}) + "\n",
        encoding="utf-8",
    )
    np.save(universe_root / "repo_coords.npy", np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    np.save(
        universe_root / "node_coords.npy",
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    )
    (universe_root / "universe_3d.png").write_bytes(b"png")

    output_dir = tmp_path / "hf_out"
    result = export_library_repo_graph_hf_dataset(
        export_root=str(export_root),
        output_dir=str(output_dir),
    )

    stats = result["stats"]
    assert stats["repo_count_manifest"] == 2
    assert stats["universe_available"] is True
    assert stats["splits"]["repos"] == 2
    assert stats["splits"]["entities"] == 3
    assert stats["splits"]["edges"] == 2
    assert stats["splits"]["artifacts"] == 2
    assert stats["splits"]["universe_nodes"] == 2
    assert stats["splits"]["universe_edges"] == 1
    assert stats["splits"]["repo_knn"] == 1
    assert "universe_3d.png" in stats["universe_assets"]

    assert (output_dir / "parquet" / "repos.parquet").is_file()
    assert (output_dir / "parquet" / "entities.parquet").is_file()
    assert (output_dir / "parquet" / "edges.parquet").is_file()
    assert (output_dir / "parquet" / "artifacts.parquet").is_file()
    assert (output_dir / "parquet" / "universe_nodes.parquet").is_file()
    assert (output_dir / "parquet" / "universe_edges.parquet").is_file()
    assert (output_dir / "parquet" / "repo_knn.parquet").is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "stats.json").is_file()

    repo_rows = _rows(output_dir / "parquet" / "repos.parquet")
    demo_row = next(row for row in repo_rows if row["repo_id"] == "demo")
    other_row = next(row for row in repo_rows if row["repo_id"] == "other")
    assert demo_row["in_universe"] is True
    assert demo_row["universe_repo_x"] == 1.0
    assert demo_row["universe_repo_y"] == 2.0
    assert demo_row["universe_repo_z"] == 3.0
    assert demo_row["index_names"] == ["qa"]
    assert "qa.npy" in demo_row["indices_json"]
    assert "/tmp/" not in demo_row["indices_json"]
    assert other_row["in_universe"] is False
    assert other_row["universe_repo_x"] is None

    universe_node_rows = _rows(output_dir / "parquet" / "universe_nodes.parquet")
    assert universe_node_rows[0]["node_id"] == "demo:repo"
    assert abs(universe_node_rows[0]["x"] - 0.1) < 1e-6
    assert abs(universe_node_rows[0]["y"] - 0.2) < 1e-6
    assert abs(universe_node_rows[0]["z"] - 0.3) < 1e-6
    assert abs(universe_node_rows[1]["x"] - 0.4) < 1e-6

    repo_knn_rows = _rows(output_dir / "parquet" / "repo_knn.parquet")
    assert repo_knn_rows[0]["src_repo"] == "demo"
    assert repo_knn_rows[0]["dst_repo"] == "other"
    assert repo_knn_rows[0]["weight"] == 0.75
