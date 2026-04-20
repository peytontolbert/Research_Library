from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.export_paper_text_hf_dataset import export_paper_text_hf_dataset


def test_export_paper_text_hf_dataset_combines_tokens_and_attaches_metadata(tmp_path: Path) -> None:
    structured_dir = tmp_path / "pdfs_structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    shard = structured_dir / "pdf_structured_00000.jsonl"
    shard.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pdf_path": "/arxiv/pdfs/2401/2401.00001v1.pdf",
                        "tokens": [
                            {"type": "heading", "text": "Abstract", "page": 1, "line_no": 1},
                            {"type": "text", "text": "First line.", "page": 1, "line_no": 2},
                            {"type": "text", "text": "First line.", "page": 1, "line_no": 3},
                            {"type": "figure", "text": "[IMAGE]", "page": 1, "line_no": 4},
                            {"type": "text", "text": "Second line.", "page": 2, "line_no": 1},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "pdf_path": "/arxiv/pdfs/2401/2401.00002.pdf",
                        "tokens": [
                            {"type": "text", "text": "Other paper text.", "page": 1, "line_no": 1},
                        ],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    metadata_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2401.00001",
                        "title": "Paper One",
                        "abstract": "Abs one",
                        "authors": "Alice, Bob",
                        "categories": "cs.CL",
                        "license": "http://creativecommons.org/licenses/by/4.0/",
                        "update_date": "2024-01-10",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.00002",
                        "title": "Paper Two",
                        "abstract": "Abs two",
                        "authors": "Carol",
                        "categories": "cs.AI",
                        "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                        "update_date": "2024-01-11",
                        "versions": [{"version": "v1"}, {"version": "v2"}],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_text_hf_dataset(
        structured_dir=str(structured_dir),
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        write_jsonl=True,
        write_dataset_dict=True,
    )

    stats = result["stats"]
    assert stats["rows_written"] == 2
    assert stats["metadata_covered"] == 2
    assert (output_dir / "papers.jsonl").is_file()
    assert (output_dir / "dataset_dict").is_dir()
    assert (output_dir / "train.parquet").is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "stats.json").is_file()

    rows = [json.loads(line) for line in (output_dir / "papers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    row1 = rows[0]
    assert row1["paper_id"] == "2401.00001v1"
    assert row1["canonical_paper_id"] == "2401.00001"
    assert row1["paper_version"] == "v1"
    assert row1["title"] == "Paper One"
    assert row1["license"] == "http://creativecommons.org/licenses/by/4.0/"
    assert row1["token_count"] == 3
    assert row1["page_count"] == 2
    assert row1["text"].startswith("Abstract")
    assert "[IMAGE]" not in row1["text"]
    assert row1["text"].count("First line.") == 1
    assert row1["text_is_partial"] is True

    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk(str(output_dir / "dataset_dict"))
    assert ds["train"].num_rows == 2
    assert ds["train"][0]["paper_id"] == "2401.00001v1"
    assert ds["train"][1]["version_count"] == 2


def test_export_paper_text_hf_dataset_filters_by_license(tmp_path: Path) -> None:
    structured_dir = tmp_path / "pdfs_structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    (structured_dir / "pdf_structured_00000.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pdf_path": "/arxiv/pdfs/2401/2401.00001.pdf",
                        "tokens": [{"type": "text", "text": "Paper one.", "page": 1, "line_no": 1}],
                    }
                ),
                json.dumps(
                    {
                        "pdf_path": "/arxiv/pdfs/2401/2401.00002.pdf",
                        "tokens": [{"type": "text", "text": "Paper two.", "page": 1, "line_no": 1}],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    metadata_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2401.00001",
                        "title": "Paper One",
                        "license": "http://creativecommons.org/licenses/by/4.0/",
                        "versions": [],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.00002",
                        "title": "Paper Two",
                        "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                        "versions": [],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_text_hf_dataset(
        structured_dir=str(structured_dir),
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        license_allow=["http://creativecommons.org/licenses/by/4.0/"],
        require_metadata=True,
    )
    assert result["stats"]["rows_written"] == 1
    assert not (output_dir / "papers.jsonl").exists()


def test_export_paper_text_hf_dataset_uses_raw_pdf_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    structured_dir = tmp_path / "pdfs_structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    raw_text_path = tmp_path / "arxiv_cache" / "2401" / "raw_fallback.txt"
    raw_text_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text_path.write_text("placeholder", encoding="utf-8")

    (structured_dir / "pdf_structured_00000.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "2401.00003v2",
                "pdf_path": str(raw_text_path),
                "tokens": [{"type": "text", "text": "PDF_PATH::" + str(raw_text_path), "page": 1, "line_no": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "2401.00003",
                "title": "Paper Three",
                "versions": [{"version": "v1"}, {"version": "v2"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.export_paper_text_hf_dataset.extract_pdf_text",
        lambda path, max_chars=0: "Fallback page one.\n\x0cFallback page two.\n",
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_text_hf_dataset(
        structured_dir=str(structured_dir),
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        write_dataset_dict=True,
    )

    assert result["stats"]["rows_written"] == 1
    assert result["stats"]["text_source_counts"]["raw_pdf_fallback"] == 1
    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk(str(output_dir / "dataset_dict"))
    row = ds["train"][0]
    assert row["paper_id"] == "2401.00003v2"
    assert row["canonical_paper_id"] == "2401.00003"
    assert row["paper_version"] == "v2"
    assert row["text_source"] == "raw_pdf_fallback"
    assert row["page_count"] == 2
    assert row["token_types"] == ["raw_text_fallback"]
    assert "Fallback page one." in row["text"]


def test_export_paper_text_hf_dataset_skips_jsonl_by_default(tmp_path: Path) -> None:
    structured_dir = tmp_path / "pdfs_structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    (structured_dir / "pdf_structured_00000.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "2401.00004v1",
                "pdf_path": "/arxiv/pdfs/2401/2401.00004v1.pdf",
                "tokens": [{"type": "text", "text": "Paper four.", "page": 1, "line_no": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "2401.00004",
                "title": "Paper Four",
                "versions": [{"version": "v1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_text_hf_dataset(
        structured_dir=str(structured_dir),
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
    )

    assert result["stats"]["jsonl_written"] is False
    assert result["stats"]["dataset_dict_written"] is False
    assert result["stats"]["jsonl_path"] == ""
    assert result["stats"]["dataset_disk_dir"] == ""
    assert not (output_dir / "papers.jsonl").exists()
    assert not (output_dir / "dataset_dict").exists()
    assert (output_dir / "train.parquet").is_file()


def test_export_paper_text_hf_dataset_prefers_raw_pdf_text_for_full_export(tmp_path: Path) -> None:
    structured_dir = tmp_path / "pdfs_structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    raw_text_path = tmp_path / "arxiv_cache" / "2401" / "2401.00005v1.txt"
    raw_text_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text_path.write_text("Full page one.\n\x0cFull page two.\n", encoding="utf-8")

    (structured_dir / "pdf_structured_00000.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "2401.00005v1",
                "pdf_path": str(raw_text_path),
                "tokens": [{"type": "text", "text": "Only first page snippet.", "page": 1, "line_no": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "2401.00005",
                "title": "Paper Five",
                "versions": [{"version": "v1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_text_hf_dataset(
        structured_dir=str(structured_dir),
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        write_dataset_dict=True,
        prefer_raw_pdf_text=True,
        raw_pdf_max_chars=0,
    )

    assert result["stats"]["prefer_raw_pdf_text"] is True
    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk(str(output_dir / "dataset_dict"))
    row = ds["train"][0]
    assert row["text_source"] == "raw_pdf_preferred"
    assert row["text"] == "Full page one.\n\nFull page two."
    assert row["page_count"] == 2
    assert row["token_types"] == ["raw_text_pdf"]
    assert row["text_is_partial"] is False


def test_export_paper_text_hf_dataset_uses_preextracted_raw_text(tmp_path: Path) -> None:
    structured_dir = tmp_path / "pdfs_structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    raw_text_dir = structured_dir / "raw_text" / "2401"
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    raw_text_path = raw_text_dir / "2401.00006v1.txt"
    raw_text_path.write_text("Full cached paper text.\n\nSecond paragraph.", encoding="utf-8")

    (structured_dir / "pdf_structured_00000.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "2401.00006v1",
                "pdf_path": "/arxiv/pdfs/2401/2401.00006v1.pdf",
                "tokens": [],
                "raw_text_path": str(raw_text_path),
                "raw_text_line_count": 2,
                "raw_text_page_count": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "2401.00006",
                "title": "Paper Six",
                "versions": [{"version": "v1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_text_hf_dataset(
        structured_dir=str(structured_dir),
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        write_dataset_dict=True,
        prefer_raw_pdf_text=True,
        raw_pdf_max_chars=0,
    )

    assert result["stats"]["text_source_counts"]["raw_pdf_preextracted"] == 1
    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk(str(output_dir / "dataset_dict"))
    row = ds["train"][0]
    assert row["text_source"] == "raw_pdf_preextracted"
    assert row["text"] == "Full cached paper text.\n\nSecond paragraph."
    assert row["page_count"] == 7
    assert row["token_types"] == ["raw_text_preextracted"]
    assert row["text_is_partial"] is False


def test_export_paper_text_hf_dataset_reads_backfill_parquet_rows(tmp_path: Path) -> None:
    structured_dir = tmp_path / "pdfs_structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "paper_id": "2401.00007v1",
                    "canonical_paper_id": "2401.00007",
                    "paper_version": "v1",
                    "pdf_path": "/arxiv/pdfs/2401/2401.00007v1.pdf",
                    "title": "",
                    "abstract": "",
                    "authors": "",
                    "categories": "",
                    "license": "",
                    "update_date": "",
                    "version_count": 0,
                    "metadata_found": False,
                    "text": "Backfilled full text.",
                    "text_source": "raw_pdf_preextracted",
                    "text_is_partial": False,
                    "text_char_count": 21,
                    "text_line_count": 1,
                    "token_count": 1,
                    "page_count": 9,
                    "token_types": ["raw_text_preextracted"],
                    "token_type_counts_json": "{\"raw_text_preextracted\":1}",
                }
            ]
        ),
        str(structured_dir / "paper_text_backfill_00000.parquet"),
        compression="zstd",
    )

    metadata_path = tmp_path / "arxiv-metadata-oai-snapshot.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "2401.00007",
                "title": "Paper Seven",
                "versions": [{"version": "v1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "hf_out"
    result = export_paper_text_hf_dataset(
        structured_dir=str(structured_dir),
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        write_dataset_dict=True,
    )

    assert result["stats"]["rows_written"] == 1
    assert result["stats"]["text_source_counts"]["raw_pdf_preextracted"] == 1

    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk(str(output_dir / "dataset_dict"))
    row = ds["train"][0]
    assert row["paper_id"] == "2401.00007v1"
    assert row["title"] == "Paper Seven"
    assert row["text"] == "Backfilled full text."
    assert row["page_count"] == 9
    assert row["text_is_partial"] is False
