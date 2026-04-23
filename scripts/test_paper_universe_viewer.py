from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.paper_universe_viewer import build_paper_universe_viewer


def _write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_build_paper_universe_viewer_writes_assets(tmp_path: Path) -> None:
    universe_dir = tmp_path / "paper_universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(
        universe_dir / "paper_nodes.parquet",
        [
            {
                "paper_id": "2401.00001v1",
                "canonical_paper_id": "2401.00001",
                "title": "Paper One",
                "primary_category": "cs.AI",
                "year": 2024,
                "x": 0.0,
                "y": 0.1,
                "z": 0.2,
            },
            {
                "paper_id": "2401.00002v1",
                "canonical_paper_id": "2401.00002",
                "title": "Paper Two",
                "primary_category": "cs.LG",
                "year": 2024,
                "x": 1.0,
                "y": 1.1,
                "z": 1.2,
            },
            {
                "paper_id": "2401.00003v1",
                "canonical_paper_id": "2401.00003",
                "title": "Paper Three",
                "primary_category": "math.OC",
                "year": 2025,
                "x": -1.0,
                "y": -1.1,
                "z": -1.2,
            },
        ],
    )
    _write_parquet(
        universe_dir / "category_nodes.parquet",
        [
            {"category_id": "cs.AI", "paper_count": 2, "x": 0.2, "y": 0.2, "z": 0.0},
            {"category_id": "cs.LG", "paper_count": 1, "x": 1.1, "y": 1.0, "z": 1.0},
        ],
    )
    _write_parquet(
        universe_dir / "year_nodes.parquet",
        [
            {"year": 2024, "paper_count": 2, "x": 0.4, "y": 0.5, "z": 0.1},
            {"year": 2025, "paper_count": 1, "x": -0.8, "y": -0.7, "z": -0.1},
        ],
    )

    result = build_paper_universe_viewer(
        universe_dir=str(universe_dir),
        levels=[2],
        batch_rows=2,
    )

    assert Path(result["html_path"]).is_file()
    assert (universe_dir / "nodes_3d_sample.html").is_file()
    assert (universe_dir / "interactive" / "manifest.json").is_file()
    assert (universe_dir / "interactive" / "papers_2.json").is_file()
    assert (universe_dir / "interactive" / "categories.json").is_file()
    assert (universe_dir / "interactive" / "years.json").is_file()
    manifest = json.loads((universe_dir / "interactive" / "manifest.json").read_text())
    assert manifest["categories"]["rows"] == 2
    assert manifest["years"]["rows"] == 2
    papers = json.loads((universe_dir / "interactive" / "papers_2.json").read_text())
    assert len(papers) == 2
