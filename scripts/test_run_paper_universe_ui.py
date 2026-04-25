from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_run_module(monkeypatch):
    monkeypatch.setenv("PRELOAD_LLM", "0")
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def test_paper_universe_assets_payload_reports_available_assets(tmp_path: Path, monkeypatch) -> None:
    run = _load_run_module(monkeypatch)

    universe_dir = tmp_path / "paper_universe"
    interactive_dir = universe_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)

    (universe_dir / "universe_3d.png").write_bytes(b"png")
    (universe_dir / "universe_3d_detailed.png").write_bytes(b"png2")
    (universe_dir / "universe_3d_hover.html").write_text("<html></html>", encoding="utf-8")
    (interactive_dir / "manifest.json").write_text(json.dumps({"paper_levels": []}), encoding="utf-8")
    (universe_dir / "render_manifest.json").write_text(
        json.dumps({"overview": {"paper_points": 1200}, "detailed": {"paper_points": 400}}),
        encoding="utf-8",
    )
    (universe_dir / "viewer_manifest.json").write_text(
        json.dumps({"paper_levels": [{"label": "50,000 papers", "rows": 50000}]}),
        encoding="utf-8",
    )
    (universe_dir / "progress.json").write_text(
        json.dumps({"processed_papers": 123456}),
        encoding="utf-8",
    )

    payload = run._paper_universe_assets_payload(universe_dir)

    assert payload["available"] is True
    assert payload["overview_image_url"] == "/paper-universe/universe_3d.png"
    assert payload["detailed_image_url"] == "/paper-universe/universe_3d_detailed.png"
    assert payload["interactive_viewer_url"] == "/paper-universe/universe_3d_hover.html"
    assert payload["interactive_manifest_url"] == "/paper-universe/interactive/manifest.json"
    assert payload["render_manifest"]["overview"]["paper_points"] == 1200
    assert payload["viewer_manifest"]["paper_levels"][0]["rows"] == 50000
    assert payload["progress"]["processed_papers"] == 123456


def test_index_html_exposes_unified_papers_library(monkeypatch) -> None:
    run = _load_run_module(monkeypatch)

    response = asyncio.run(run.index())
    html = response.body.decode("utf-8")

    assert "<title>Research Library</title>" in html
    assert '<option value="papers">Papers</option>' in html
    assert '<option value="arxiv">Arxiv</option>' not in html
    assert '<option value="paper_universe">Paper Universe</option>' not in html
    assert 'id="library-list-label"' in html
    assert "refreshActiveLibrary" in html
    assert 'Search papers to populate the sidebar.' in html
    assert "data.answer + '\\n\\n---\\n\\n' + raw" in html
    assert "loadPaperUniverse" in html
    assert "paper_universe_select_paper" in html
    assert "selectPaperFromViewer" in html
    assert "Double-clicking a paper there should select it in the library" in html
    assert "Selected Paper Graph" in html
    assert "categories, topics, year, and text sections" in html
    assert "Selection loads details, full text, and the selected-paper graph together." in html
    assert 'id="arxiv-detail" class="paper-detail-card"' in html
    assert 'id="paper-universe-meta" class="paper-universe-grid"' in html
    assert "renderSelectedPaperDetail" in html
    assert "renderPaperSelectionPending" in html
    assert "Click received" in html
    assert "Paper click received" in html
    assert "renderPaperUniverseMeta" in html
    assert "Paper Text" in html
    assert "loadPaperText" in html
    assert "/api/paper-text/" in html
    assert "Open source PDF" in html
    assert "loadPaperNeighborhood" in html
    assert "/api/paper-universe/neighborhood/" in html
    assert "Universe 3D Overview" not in html
    assert "Universe 3D Detailed View" not in html
    assert "paper-universe-overview-img" not in html
    assert "paper-universe-detailed-img" not in html
    assert "paper-universe-open-overview" not in html
    assert "paper-universe-open-detailed" not in html
    assert html.index("Interactive 3D Viewer") < html.index("Paper details")
    assert "* { box-sizing: border-box; }" in html
    assert "html, body { max-width: 100%; overflow-x: hidden; }" in html
    assert "grid-template-columns: 320px minmax(0, 1fr)" in html
    assert "#repo-main-section, #arxiv-panel-root, #algorithms-panel-root { min-width: 0; max-width: 100%; overflow-x: hidden; }" in html
    assert "overflow-wrap:anywhere" in html
    assert "/api/paper-universe" in html


def test_find_local_arxiv_record_by_id_normalizes_versions(monkeypatch) -> None:
    run = _load_run_module(monkeypatch)
    monkeypatch.setattr(
        run,
        "arxiv_iter_metadata",
        lambda: iter(
            [
                SimpleNamespace(
                    id="2401.00001",
                    title="Paper One",
                    abstract="Abstract",
                    authors="Author",
                    categories="cs.AI",
                )
            ]
        ),
    )
    monkeypatch.setattr(run, "_find_local_arxiv_pdf", lambda _paper_id: Path("/tmp/2401.00001.pdf"))

    record = run._find_local_arxiv_record_by_id("2401.00001v3")

    assert record is not None
    assert record["id"] == "2401.00001"
    assert record["title"] == "Paper One"
    assert record["has_pdf"] is True
