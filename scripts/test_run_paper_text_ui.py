from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_run_module(monkeypatch):
    monkeypatch.setenv("PRELOAD_LLM", "0")
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def test_paper_text_payload_infers_pages_and_source_url(monkeypatch) -> None:
    run = _load_run_module(monkeypatch)
    monkeypatch.setattr(
        run,
        "_paper_text_row_by_id",
        lambda _paper_id: {
            "paper_id": "1701.04743v1",
            "canonical_paper_id": "1701.04743",
            "title": "Computing Egomotion with Local Loop Closures for Egocentric Videos",
            "authors": "Author One",
            "categories": "cs.CV",
            "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "update_date": "2017-01-17",
            "text": (
                "Abstract\n\n"
                "This is the abstract paragraph for the paper.\n\n"
                "1 Introduction\n\n"
                "The introduction runs for a while and explains the setup.\n\n"
                "2 Method\n\n"
                "The method paragraph is also fairly long and should land on a later inferred page.\n\n"
                "3 Results\n\n"
                "The results paragraph closes out the sample."
            ),
            "text_source": "raw_pdf_preextracted",
            "text_is_partial": False,
            "text_char_count": 0,
            "text_line_count": 0,
            "token_count": 0,
            "page_count": 3,
            "token_types": ["raw_text_preextracted"],
            "token_type_counts_json": "{\"raw_text_preextracted\":42}",
        },
    )
    monkeypatch.setattr(run, "_paper_text_existing_pdf_path", lambda _row: None)

    payload = run._paper_text_payload("1701.04743")

    assert payload is not None
    assert payload["paper_id"] == "1701.04743v1"
    assert payload["canonical_paper_id"] == "1701.04743"
    assert payload["page_mode"] == "inferred"
    assert payload["page_count"] == 3
    assert payload["source_pdf_url"] == "https://arxiv.org/pdf/1701.04743v1.pdf"
    assert payload["has_local_pdf"] is False
    assert any(section["title"] == "1 Introduction" for section in payload["sections"])
    assert payload["pages"][0]["blocks"][0]["kind"] == "heading"


def test_paper_text_payload_prefers_exact_pdf_pages(monkeypatch, tmp_path: Path) -> None:
    run = _load_run_module(monkeypatch)
    fake_pdf = tmp_path / "sample.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        run,
        "_paper_text_row_by_id",
        lambda _paper_id: {
            "paper_id": "2401.00001v2",
            "canonical_paper_id": "2401.00001",
            "title": "Paper One",
            "authors": "Author One",
            "categories": "cs.AI",
            "license": "",
            "update_date": "2024-01-01",
            "text": "Fallback text that should not be used for page splitting.",
            "text_source": "raw_pdf_preferred",
            "text_is_partial": False,
            "text_char_count": 0,
            "text_line_count": 0,
            "token_count": 0,
            "page_count": 9,
            "token_types": [],
            "token_type_counts_json": "",
        },
    )
    monkeypatch.setattr(run, "_paper_text_existing_pdf_path", lambda _row: fake_pdf)
    monkeypatch.setattr(
        run,
        "_extract_pdf_pages",
        lambda _path: ["Title Page\n\n1 Introduction", "2 Method\n\nMore text"],
    )

    payload = run._paper_text_payload("2401.00001")

    assert payload is not None
    assert payload["page_mode"] == "exact_pdf"
    assert payload["page_count"] == 2
    assert payload["reported_page_count"] == 9
    assert payload["has_local_pdf"] is True
    assert payload["pages"][1]["heading_titles"][0] == "2 Method"


def test_paper_text_heading_filter_rejects_equation_fragments(monkeypatch) -> None:
    run = _load_run_module(monkeypatch)

    assert run._looks_like_heading("9 Magnetically dominated jets are naturally expected")
    assert not run._looks_like_heading("-12 -12 -15 -15 AT")
    assert not run._looks_like_heading("x = -12 - 15")
    assert not run._looks_like_heading("A-B-C-D")
