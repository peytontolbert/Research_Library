from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.paper_universe_render import render_paper_universe_assets


def _write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_render_paper_universe_assets_writes_pngs(tmp_path: Path) -> None:
    universe_dir = tmp_path / "paper_universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(
        universe_dir / "paper_nodes.parquet",
        [
            {"x": 0.0, "y": 0.0, "z": 0.0, "primary_category": "cs.AI", "year": 2024, "title": "Paper One"},
            {"x": 1.0, "y": 0.5, "z": -0.5, "primary_category": "cs.LG", "year": 2024, "title": "Paper Two"},
            {"x": -0.5, "y": 1.0, "z": 0.25, "primary_category": "math.OC", "year": 2025, "title": "Paper Three"},
        ],
    )
    _write_parquet(
        universe_dir / "category_nodes.parquet",
        [
            {"category_id": "cs.AI", "paper_count": 2, "x": 0.2, "y": 0.1, "z": 0.0},
            {"category_id": "cs.LG", "paper_count": 1, "x": 0.9, "y": 0.3, "z": -0.2},
        ],
    )
    _write_parquet(
        universe_dir / "year_nodes.parquet",
        [
            {"year": 2024, "paper_count": 2, "x": 0.1, "y": 0.2, "z": 0.05},
            {"year": 2025, "paper_count": 1, "x": 0.7, "y": 0.8, "z": -0.1},
        ],
    )

    result = render_paper_universe_assets(
        universe_dir=str(universe_dir),
        overview_sample=3,
        detailed_sample=3,
        seed=7,
    )

    assert Path(result["overview_image"]).is_file()
    assert Path(result["detailed_image"]).is_file()
    assert Path(result["overview_image"]).stat().st_size > 0
    assert Path(result["detailed_image"]).stat().st_size > 0
    assert (universe_dir / "render_manifest.json").is_file()
    assert result["overview"]["paper_points"] == 3
    assert result["overview"]["categories_represented"] == 3
    assert result["overview"]["years_represented"] == 2
    assert result["detailed"]["paper_points"] == 3
    assert result["detailed"]["labeled_papers"] >= 1
