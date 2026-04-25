from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.python_repo_graph import PythonRepoGraph


def _load_run_module(monkeypatch):
    monkeypatch.setenv("PRELOAD_LLM", "0")
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_api_source_resolves_file_uri(monkeypatch, tmp_path: Path) -> None:
    run = _load_run_module(monkeypatch)

    repo_root = tmp_path / "demo_repo"
    (repo_root / "pkg").mkdir(parents=True, exist_ok=True)
    (repo_root / "pkg" / "demo.py").write_text(
        "def greet(name: str) -> str:\n    return f'hello {name}'\n",
        encoding="utf-8",
    )

    graph = PythonRepoGraph(str(repo_root))
    repo = SimpleNamespace(root_path=repo_root, graph=graph)
    monkeypatch.setattr(run, "open_repository", lambda _repo_id: repo)

    uri = f"program://{graph.program_id}/file/pkg/demo.py"
    payload = asyncio.run(run.api_source("demo_repo", uri, context=0, max_lines=20))

    assert payload["artifact_uri"] == f"program://{graph.program_id}/artifact/pkg/demo.py"
    assert payload["path"] == "pkg/demo.py"
    assert payload["span"]["start_line"] == 1
    assert payload["span"]["end_line"] == 2
    assert any("def greet" in line["text"] for line in payload["snippet"]["lines"])


def test_paper_universe_neighborhood_payload(monkeypatch, tmp_path: Path) -> None:
    run = _load_run_module(monkeypatch)

    universe_dir = tmp_path / "paper_universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(
        universe_dir / "paper_nodes.parquet",
        [
            {
                "paper_idx": 1,
                "paper_id": "2401.00001v2",
                "canonical_paper_id": "2401.00001",
                "paper_version": "v2",
                "title": "Paper One",
                "authors": "Author One",
                "categories": ["cs.AI"],
                "primary_category": "cs.AI",
                "update_date": "2024-01-10",
                "year": 2024,
                "pdf_path": "",
                "x": 0.0,
                "y": 0.1,
                "z": 0.2,
            },
            {
                "paper_idx": 2,
                "paper_id": "2401.00002v1",
                "canonical_paper_id": "2401.00002",
                "paper_version": "v1",
                "title": "Paper Two",
                "authors": "Author Two",
                "categories": ["cs.LG"],
                "primary_category": "cs.LG",
                "update_date": "2024-01-11",
                "year": 2024,
                "pdf_path": "",
                "x": 1.0,
                "y": 1.1,
                "z": 1.2,
            },
            {
                "paper_idx": 3,
                "paper_id": "2401.00003v1",
                "canonical_paper_id": "2401.00003",
                "paper_version": "v1",
                "title": "Paper Three",
                "authors": "Author Three",
                "categories": ["math.OC"],
                "primary_category": "math.OC",
                "update_date": "2024-01-12",
                "year": 2024,
                "pdf_path": "",
                "x": -1.0,
                "y": -1.1,
                "z": -1.2,
            },
        ],
    )
    _write_parquet(
        universe_dir / "paper_knn_edges.parquet",
        [
            {"src_paper_idx": 1, "dst_paper_idx": 2, "type": "paper_knn", "weight": 0.91},
            {"src_paper_idx": 1, "dst_paper_idx": 3, "type": "paper_knn", "weight": 0.82},
        ],
    )
    _write_parquet(
        universe_dir / "edges.parquet",
        [
            {"src_paper_idx": 1, "dst_category_id": "cs.AI", "type": "has_category"},
            {"src_paper_idx": 1, "dst_category_id": "cs.LG", "type": "has_category"},
        ],
    )
    _write_parquet(
        universe_dir / "category_nodes.parquet",
        [
            {"category_id": "cs.AI", "name": "cs.AI", "paper_count": 10},
            {"category_id": "cs.LG", "name": "cs.LG", "paper_count": 20},
        ],
    )
    _write_parquet(
        universe_dir / "paper_topic_edges.parquet",
        [
            {"src_paper_idx": 1, "dst_topic_id": "paper graph", "type": "has_topic"},
            {"src_paper_idx": 1, "dst_topic_id": "library viewer", "type": "has_topic"},
        ],
    )
    _write_parquet(
        universe_dir / "topic_nodes.parquet",
        [
            {"topic_id": "paper graph", "name": "paper graph", "paper_count": 3},
            {"topic_id": "library viewer", "name": "library viewer", "paper_count": 2},
        ],
    )
    _write_parquet(
        universe_dir / "paper_year_edges.parquet",
        [
            {"src_paper_idx": 1, "dst_year": 2024, "type": "has_year"},
        ],
    )

    monkeypatch.setattr(
        run,
        "_find_local_arxiv_pdf",
        lambda paper_id: Path(f"/tmp/{paper_id}.pdf") if paper_id == "2401.00002" else None,
    )
    monkeypatch.setattr(
        run,
        "_selected_paper_section_nodes",
        lambda _paper_id, max_sections=10: [
            {"id": "section:1:0", "title": "Abstract", "page": 1},
            {"id": "section:2:1", "title": "1 Introduction", "page": 2},
        ],
    )

    payload = run._paper_universe_neighborhood_payload(
        "2401.00001v7",
        neighbor_limit=2,
        universe_root=universe_dir,
    )

    assert payload["paper"]["canonical_paper_id"] == "2401.00001"
    assert payload["type"] == "selected_paper_graph"
    assert payload["neighbor_count"] == 2
    paper_nodes = [node for node in payload["nodes"] if node["kind"] == "paper"]
    assert {node["canonical_paper_id"] for node in paper_nodes} == {
        "2401.00001",
        "2401.00002",
        "2401.00003",
    }
    selected = next(node for node in paper_nodes if node["role"] == "selected")
    assert selected["label"] == "Paper One"
    paper_two = next(node for node in paper_nodes if node["canonical_paper_id"] == "2401.00002")
    assert paper_two["has_pdf"] is True
    assert {node["kind"] for node in payload["nodes"]} >= {"paper", "category", "topic", "year", "section"}
    assert payload["category_count"] == 2
    assert payload["topic_count"] == 2
    assert payload["section_count"] == 2
    assert any(edge["type"] == "paper_knn" and edge["source"] == "paper:2401.00001" and edge["target"] == "paper:2401.00002" for edge in payload["edges"])
    assert any(edge["type"] == "has_category" and edge["target"] == "category:cs.AI" for edge in payload["edges"])
    assert any(edge["type"] == "has_topic" and edge["target"] == "topic:paper graph" for edge in payload["edges"])
    assert any(edge["type"] == "has_section" and edge["target"] == "section:1:0" for edge in payload["edges"])
