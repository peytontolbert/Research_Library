from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore

from scripts.backfill_missing_paper_text_shards import backfill_missing_paper_text_shards


def test_backfill_missing_paper_text_shards_skips_existing_and_writes_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    existing_structured_dir = tmp_path / "existing_structured"
    existing_structured_dir.mkdir(parents=True, exist_ok=True)
    (existing_structured_dir / "pdf_structured_00000.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "2401.00001v1",
                "pdf_path": "/arxiv/pdfs/2401/2401.00001v1.pdf",
                "tokens": [{"type": "text", "text": "Already present.", "page": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pdf_root = tmp_path / "pdfs"
    (pdf_root / "2401").mkdir(parents=True, exist_ok=True)
    existing_pdf = pdf_root / "2401" / "2401.00001v1.pdf"
    missing_pdf = pdf_root / "2401" / "2401.00002v1.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4 existing")
    missing_pdf.write_bytes(b"%PDF-1.4 missing")

    monkeypatch.setattr(
        "scripts.backfill_missing_paper_text_shards._extract_pdf_text_fast",
        lambda path, max_chars, timeout_seconds: (
            "Ignored existing paper"
            if Path(path).name == "2401.00001v1.pdf"
            else "Page one.\n\x0cPage two.\n"
        ),
    )

    out_dir = tmp_path / "pdfs_structured_backfill"
    result = backfill_missing_paper_text_shards(
        existing_structured_dirs=[str(existing_structured_dir)],
        out_dir=str(out_dir),
        pdf_root=str(pdf_root),
        shard_size=10,
        row_group_rows=1,
        progress_every=0,
        raw_pdf_max_chars=0,
        parquet_compression="zstd",
        max_papers=0,
    )

    stats = result["stats"]
    assert stats["extracted_rows"] == 1
    assert stats["skipped_existing"] == 1
    assert stats["parquet_compression"] == "zstd"
    assert stats["rows_per_parquet_file"] == 10
    assert stats["row_group_rows"] == 1
    assert stats["parquet_shards_written"] == 1

    shard_path = out_dir / "paper_text_backfill_00000.parquet"
    assert shard_path.is_file()
    rows = pq.read_table(str(shard_path)).to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["paper_id"] == "2401.00002v1"
    assert row["canonical_paper_id"] == "2401.00002"
    assert row["text"] == "Page one.\n\nPage two."
    assert row["text_source"] == "raw_pdf_preextracted"
    assert row["text_is_partial"] is False
    assert row["page_count"] == 2
    assert row["text_line_count"] == 2
    assert row["token_types"] == ["raw_text_preextracted"]
