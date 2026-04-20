from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.merge_paper_text_parquets import merge_paper_text_parquets


def test_merge_paper_text_parquets_dedupes_and_fills_metadata(tmp_path: Path) -> None:
    base_path = tmp_path / "base.parquet"
    backfill_path = tmp_path / "backfill.parquet"
    output_dir = tmp_path / "merged"
    metadata_path = tmp_path / "metadata.jsonl"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "paper_id": "2401.00001v1",
                    "canonical_paper_id": "2401.00001",
                    "paper_version": "v1",
                    "pdf_path": "/arxiv/pdfs/2401/2401.00001v1.pdf",
                    "title": "Old Title",
                    "abstract": "Old abstract",
                    "authors": "Alice",
                    "categories": "cs.AI",
                    "license": "",
                    "update_date": "",
                    "version_count": 1,
                    "metadata_found": True,
                    "text": "short text",
                    "text_source": "combined_structured_tokens",
                    "text_is_partial": True,
                    "text_char_count": 10,
                    "text_line_count": 1,
                    "token_count": 1,
                    "page_count": 1,
                    "token_types": ["text"],
                    "token_type_counts_json": "{\"text\":1}",
                },
                {
                    "paper_id": "2401.00001v2",
                    "canonical_paper_id": "2401.00001",
                    "paper_version": "v2",
                    "pdf_path": "/arxiv/pdfs/2401/2401.00001v2.pdf",
                    "title": "Better Title",
                    "abstract": "Better abstract",
                    "authors": "Alice, Bob",
                    "categories": "cs.AI",
                    "license": "",
                    "update_date": "",
                    "version_count": 2,
                    "metadata_found": True,
                    "text": "much longer full text",
                    "text_source": "raw_pdf_preferred",
                    "text_is_partial": False,
                    "text_char_count": 21,
                    "text_line_count": 1,
                    "token_count": 1,
                    "page_count": 5,
                    "token_types": ["raw_text_pdf"],
                    "token_type_counts_json": "{\"raw_text_pdf\":1}",
                },
            ]
        ),
        str(base_path),
        compression="zstd",
    )

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "paper_id": "2401.00002v1",
                    "canonical_paper_id": "2401.00002",
                    "paper_version": "v1",
                    "pdf_path": "/arxiv/pdfs/2401/2401.00002v1.pdf",
                    "title": "",
                    "abstract": "",
                    "authors": "",
                    "categories": "",
                    "license": "",
                    "update_date": "",
                    "version_count": 0,
                    "metadata_found": False,
                    "text": "backfilled text",
                    "text_source": "raw_pdf_preextracted",
                    "text_is_partial": False,
                    "text_char_count": 15,
                    "text_line_count": 1,
                    "token_count": 1,
                    "page_count": 7,
                    "token_types": ["raw_text_preextracted"],
                    "token_type_counts_json": "{\"raw_text_preextracted\":1}",
                }
            ]
        ),
        str(backfill_path),
        compression="zstd",
    )

    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2401.00001",
                        "title": "Metadata Title 1",
                        "abstract": "Metadata Abstract 1",
                        "authors": "Alice, Bob",
                        "categories": "cs.AI",
                        "license": "http://creativecommons.org/licenses/by/4.0/",
                        "update_date": "2024-01-01",
                        "versions": [{"version": "v1"}, {"version": "v2"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.00002",
                        "title": "Metadata Title 2",
                        "abstract": "Metadata Abstract 2",
                        "authors": "Carol",
                        "categories": "cs.LG",
                        "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                        "update_date": "2024-01-02",
                        "versions": [{"version": "v1"}],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = merge_paper_text_parquets(
        base_parquets=[str(base_path)],
        backfill_parquets=[str(backfill_path)],
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        rows_per_output_file=1,
        compression="zstd",
    )

    stats = result["stats"]
    assert stats["base_rows"] == 2
    assert stats["backfill_rows"] == 1
    assert stats["merged_rows"] == 2
    assert stats["merged_unique_canonical_ids"] == 2
    assert len(stats["output_files"]) == 2

    rows = []
    for shard_path in sorted(output_dir.glob("train_*.parquet")):
        rows.extend(pq.read_table(str(shard_path)).to_pylist())
    rows_by_id = {row["canonical_paper_id"]: row for row in rows}

    row1 = rows_by_id["2401.00001"]
    assert row1["paper_id"] == "2401.00001v2"
    assert row1["text"] == "much longer full text"
    assert row1["title"] == "Metadata Title 1"
    assert row1["license"] == "http://creativecommons.org/licenses/by/4.0/"
    assert row1["metadata_found"] is True

    row2 = rows_by_id["2401.00002"]
    assert row2["paper_id"] == "2401.00002v1"
    assert row2["text"] == "backfilled text"
    assert row2["title"] == "Metadata Title 2"
    assert row2["authors"] == "Carol"
    assert row2["metadata_found"] is True
