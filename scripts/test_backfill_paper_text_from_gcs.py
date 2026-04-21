from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.backfill_paper_text_from_gcs import (
    _download_candidates,
    _partition_matches,
    backfill_paper_text_from_gcs,
)


def test_backfill_paper_text_from_gcs_skips_existing_and_deletes_temp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    existing_dir = tmp_path / "existing_dataset"
    existing_dir.mkdir(parents=True, exist_ok=True)
    existing_parquet = existing_dir / "train_00000.parquet"
    metadata_path = tmp_path / "metadata.jsonl"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp_pdfs"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "paper_id": "2401.00001v1",
                    "canonical_paper_id": "2401.00001",
                }
            ]
        ),
        str(existing_parquet),
        compression="zstd",
    )

    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2401.00001",
                        "title": "Existing",
                        "abstract": "Already covered",
                        "authors": "Alice",
                        "categories": "cs.AI",
                        "license": "",
                        "update_date": "2024-01-01",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.00002",
                        "title": "New Two",
                        "abstract": "Abstract two",
                        "authors": "Bob",
                        "categories": "cs.LG",
                        "license": "cc-by",
                        "update_date": "2024-01-02",
                        "versions": [{"version": "v1"}, {"version": "v2"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.00003",
                        "title": "New Three",
                        "abstract": "Abstract three",
                        "authors": "Carol",
                        "categories": "cs.CL",
                        "license": "cc-by",
                        "update_date": "2024-01-03",
                        "versions": [{"version": "v1"}],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    def fake_gsutil(urls, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            base_id = Path(url).stem
            (dest_dir / f"{base_id}.pdf").write_bytes(b"%PDF-1.4 fake\n")
        return 0

    def fake_extract_pdf_text(path: Path, *, max_chars: int, timeout_seconds: int) -> str:
        return f"full text for {path.stem}"

    monkeypatch.setattr("scripts.backfill_paper_text_from_gcs._run_gsutil_cp", fake_gsutil)
    monkeypatch.setattr(
        "scripts.backfill_paper_text_from_gcs._extract_pdf_text_fast",
        fake_extract_pdf_text,
    )

    result = backfill_paper_text_from_gcs(
        existing_structured_dirs=[],
        existing_parquet_dirs=[str(existing_dir)],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        target_total_papers=3,
        download_batch_size=2,
        progress_every=0,
        delete_temp_pdfs=True,
    )

    stats = result["stats"]
    assert stats["covered_ids_before"] == 1
    assert stats["covered_ids_after"] == 3
    assert stats["extracted_rows"] == 2
    assert stats["download_batches"] == 1
    assert stats["downloaded_pdfs"] == 2
    assert stats["deleted_temp_pdfs"] == 2

    shard_paths = sorted(out_dir.glob("paper_text_backfill_*.parquet"))
    assert len(shard_paths) == 1
    rows = pq.read_table(str(shard_paths[0])).to_pylist()
    rows_by_id = {row["canonical_paper_id"]: row for row in rows}

    assert set(rows_by_id.keys()) == {"2401.00002", "2401.00003"}
    assert rows_by_id["2401.00002"]["paper_id"] == "2401.00002v2"
    assert rows_by_id["2401.00002"]["paper_version"] == "v2"
    assert rows_by_id["2401.00002"]["metadata_found"] is True
    assert rows_by_id["2401.00002"]["pdf_path"].endswith("/2401/2401.00002v2.pdf")
    assert rows_by_id["2401.00003"]["paper_id"] == "2401.00003v1"

    remaining_pdfs = list(temp_pdf_dir.rglob("*.pdf"))
    assert remaining_pdfs == []

    progress_path = out_dir / "gcs_backfill_progress.json"
    assert progress_path.is_file()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["extracted_rows"] == 2


def test_backfill_paper_text_from_gcs_partitions_work(tmp_path: Path, monkeypatch) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp_pdfs"

    records = []
    for idx in range(1, 6):
        paper_id = f"2401.{idx:05d}"
        records.append(
            json.dumps(
                {
                    "id": paper_id,
                    "title": f"Paper {idx}",
                    "abstract": f"Abstract {idx}",
                    "authors": "Author",
                    "categories": "cs.AI",
                    "license": "",
                    "update_date": "2024-01-01",
                    "versions": [{"version": "v1"}],
                }
            )
        )
    metadata_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    def fake_gsutil(urls, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            base_id = Path(url).stem
            (dest_dir / f"{base_id}.pdf").write_bytes(b"%PDF-1.4 fake\n")
        return 0

    def fake_extract_pdf_text(path: Path, *, max_chars: int, timeout_seconds: int) -> str:
        return f"full text for {path.stem}"

    monkeypatch.setattr("scripts.backfill_paper_text_from_gcs._run_gsutil_cp", fake_gsutil)
    monkeypatch.setattr(
        "scripts.backfill_paper_text_from_gcs._extract_pdf_text_fast",
        fake_extract_pdf_text,
    )

    result = backfill_paper_text_from_gcs(
        existing_structured_dirs=[],
        existing_parquet_dirs=[],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        progress_every=0,
        partition_count=2,
        partition_index=1,
    )

    stats = result["stats"]
    expected_ids = {
        f"2401.{idx:05d}"
        for idx in range(1, 6)
        if _partition_matches(f"2401.{idx:05d}", partition_count=2, partition_index=1)
    }
    assert stats["extracted_rows"] == len(expected_ids)
    assert stats["partition_count"] == 2
    assert stats["partition_index"] == 1

    shard_paths = sorted(out_dir.glob("paper_text_backfill_*.parquet"))
    rows = []
    for shard_path in shard_paths:
        rows.extend(pq.read_table(str(shard_path)).to_pylist())
    assert {row["canonical_paper_id"] for row in rows} == expected_ids


def test_backfill_paper_text_from_gcs_uses_older_version_fallback_and_cleans_temp_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp_pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_tmp = out_dir / ".paper_text_backfill_00000.parquet.tmp"
    stale_tmp.write_text("stale", encoding="utf-8")

    metadata_path.write_text(
        json.dumps(
            {
                "id": "2401.12345",
                "title": "Fallback Paper",
                "abstract": "Needs older version",
                "authors": "Alice",
                "categories": "cs.AI",
                "license": "cc-by",
                "update_date": "2024-01-10",
                "versions": [{"version": "v1"}, {"version": "v2"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_gsutil(urls, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        return 0

    def fake_gsutil_one(url, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(str(url)).stem
        if stem.endswith("v1"):
            (dest_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4 fallback\n")
            return True
        return False

    def fake_extract_pdf_text(path: Path, *, max_chars: int, timeout_seconds: int) -> str:
        return f"full text for {path.stem}"

    monkeypatch.setattr("scripts.backfill_paper_text_from_gcs._run_gsutil_cp", fake_gsutil)
    monkeypatch.setattr("scripts.backfill_paper_text_from_gcs._run_gsutil_cp_one", fake_gsutil_one)
    monkeypatch.setattr(
        "scripts.backfill_paper_text_from_gcs._extract_pdf_text_fast",
        fake_extract_pdf_text,
    )

    result = backfill_paper_text_from_gcs(
        existing_structured_dirs=[],
        existing_parquet_dirs=[],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        progress_every=0,
        download_batch_size=1,
        retry_missing_downloads=True,
    )

    stats = result["stats"]
    assert stats["extracted_rows"] == 1
    assert stats["version_fallback_uses"] == 1
    assert stats["stale_temp_shards_removed"] == 1

    shard_paths = sorted(out_dir.glob("paper_text_backfill_*.parquet"))
    assert len(shard_paths) == 1
    row = pq.read_table(str(shard_paths[0])).to_pylist()[0]
    assert row["paper_id"] == "2401.12345v1"
    assert row["paper_version"] == "v1"
    assert row["pdf_path"].endswith("/2401/2401.12345v1.pdf")
    assert not list(out_dir.glob("*.tmp"))


def test_download_candidates_support_legacy_slash_style_ids() -> None:
    candidates = _download_candidates(
        {
            "id": "acc-phys/9411001",
            "versions": [{"version": "v1"}],
        },
        gcs_prefix="gs://arxiv-dataset/arxiv/pdf",
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["paper_id"] == "acc-phys/9411001v1"
    assert candidate["paper_version"] == "v1"
    assert candidate["gcs_url"] == "gs://arxiv-dataset/arxiv/acc-phys/pdf/9411/9411001v1.pdf"


def test_backfill_paper_text_from_gcs_supports_legacy_slash_style_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp_pdfs"

    metadata_path.write_text(
        json.dumps(
            {
                "id": "acc-phys/9411001",
                "title": "Legacy Paper",
                "abstract": "Legacy abstract",
                "authors": "Alice",
                "categories": "physics.acc-ph",
                "license": "cc-by",
                "update_date": "1994-11-01",
                "versions": [{"version": "v1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_gsutil(urls, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            base_id = Path(str(url)).stem
            (dest_dir / f"{base_id}.pdf").write_bytes(b"%PDF-1.4 fake\n")
        return 0

    def fake_extract_pdf_text(path: Path, *, max_chars: int, timeout_seconds: int) -> str:
        return f"full text for {path.stem}"

    monkeypatch.setattr("scripts.backfill_paper_text_from_gcs._run_gsutil_cp", fake_gsutil)
    monkeypatch.setattr(
        "scripts.backfill_paper_text_from_gcs._extract_pdf_text_fast",
        fake_extract_pdf_text,
    )

    result = backfill_paper_text_from_gcs(
        existing_structured_dirs=[],
        existing_parquet_dirs=[],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        progress_every=0,
        delete_temp_pdfs=True,
    )

    stats = result["stats"]
    assert stats["covered_ids_before"] == 0
    assert stats["covered_ids_after"] == 1
    assert stats["extracted_rows"] == 1
    assert stats["downloaded_pdfs"] == 1

    shard_paths = sorted(out_dir.glob("paper_text_backfill_*.parquet"))
    assert len(shard_paths) == 1
    row = pq.read_table(str(shard_paths[0])).to_pylist()[0]
    assert row["canonical_paper_id"] == "acc-phys/9411001"
    assert row["paper_id"] == "acc-phys/9411001v1"
    assert row["paper_version"] == "v1"
    assert row["pdf_path"] == "gs://arxiv-dataset/arxiv/acc-phys/pdf/9411/9411001v1.pdf"
