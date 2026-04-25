from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path


def _load_run_module(monkeypatch):
    monkeypatch.setenv("PRELOAD_LLM", "0")
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def test_repo_universe_payload_builds_library_graph(monkeypatch, tmp_path: Path) -> None:
    run = _load_run_module(monkeypatch)
    export_root = tmp_path / "exports"
    for repo_id, entity_count in {"alpha": 3, "beta": 2, "docs_only": 1}.items():
        repo_dir = export_root / repo_id
        repo_dir.mkdir(parents=True)
        (repo_dir / f"{repo_id}.entities.jsonl").write_text(
            "\n".join("{}" for _ in range(entity_count)) + "\n",
            encoding="utf-8",
        )
        (repo_dir / f"{repo_id}.edges.jsonl").write_text("{}\n", encoding="utf-8")
        (repo_dir / f"{repo_id}.artifacts.jsonl").write_text("{}\n", encoding="utf-8")

    manifest = {
        "repos": {
            "alpha": {
                "repo_root": "/repos/alpha",
                "library_root": "/repos",
                "repo_state": {"branch": "main", "head": "abc123"},
                "languages": ["python", "markdown"],
                "skills": {"qa": {"status": "up_to_date"}},
                "indices": {"qa": {}},
                "extensions": {"repo_skills_miner": {"counts": {"skills": 4}}},
            },
            "beta": {
                "repo_root": "/repos/beta",
                "library_root": "/repos",
                "repo_state": {"branch": "main", "head": "def456"},
                "languages": ["python"],
                "skills": {},
                "extensions": {},
            },
            "docs_only": {
                "repo_root": "/repos/docs_only",
                "library_root": "/repos",
                "repo_state": {"branch": "main", "head": "fedcba"},
                "languages": ["markdown"],
                "skills": {},
                "extensions": {},
            },
        }
    }

    payload = run._repo_universe_payload(
        manifest=manifest,
        export_root=export_root,
        max_similarity_edges=20,
    )

    assert payload["type"] == "repo_universe"
    assert payload["repo_count"] == 3
    assert payload["language_count"] == 2
    assert payload["qa_ready_count"] == 1
    node_ids = {node["id"] for node in payload["nodes"]}
    assert {"repo:alpha", "repo:beta", "language:python", "language:markdown", "skill:qa"} <= node_ids
    alpha = next(node for node in payload["nodes"] if node["id"] == "repo:alpha")
    assert alpha["entity_count"] == 3
    assert alpha["qa_ready"] is True
    assert all(isinstance(alpha[key], float) for key in ("x", "y", "z"))
    assert all("x" in node and "y" in node and "z" in node for node in payload["nodes"])
    edge_types = {edge["type"] for edge in payload["edges"]}
    assert {"uses_language", "in_library_root", "has_skill", "similar_repo"} <= edge_types


def test_index_html_exposes_repository_universe(monkeypatch) -> None:
    run = _load_run_module(monkeypatch)

    response = asyncio.run(run.index())
    html = response.body.decode("utf-8")

    assert "Repository Universe" in html
    assert 'id="repo-universe-container"' in html
    assert "#repo-universe-container { position: relative;" in html
    assert "#repo-universe-container canvas { position: absolute !important;" in html
    assert "https://unpkg.com/deck.gl@latest/dist.min.js" in html
    assert "/api/repo-universe" in html
    assert "renderRepoUniverse" in html
    assert "loadRepoUniverse" in html
    assert "new OrbitView" in html
    assert "new ScatterplotLayer" in html
    assert "new LineLayer" in html
    assert "3D repository universe" in html
    assert "repoUniverseHubFilter" in html
    assert "clearRepoUniverseHubFilter" in html
    assert "Clear hub filter" in html
    assert "Filtering by language" in html
    assert "Click a repository node to load it." in html
