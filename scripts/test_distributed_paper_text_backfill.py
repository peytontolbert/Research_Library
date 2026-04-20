from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore

from scripts.distributed_paper_text_backfill import (
    DistributedBackfillCoordinator,
    run_coordinator,
)


def test_distributed_coordinator_leases_submits_and_writes_progress(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp"

    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2401.00001",
                        "title": "Paper 1",
                        "abstract": "Abstract 1",
                        "authors": "Alice",
                        "categories": "cs.AI",
                        "license": "cc-by",
                        "update_date": "2024-01-01",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.00002",
                        "title": "Paper 2",
                        "abstract": "Abstract 2",
                        "authors": "Bob",
                        "categories": "cs.LG",
                        "license": "cc-by",
                        "update_date": "2024-01-02",
                        "versions": [{"version": "v1"}, {"version": "v2"}],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    coordinator = DistributedBackfillCoordinator(
        auth_token="secret",
        existing_structured_dirs=[],
        existing_parquet_dirs=[],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        target_total_papers=2,
        max_papers=0,
        download_batch_size=2,
        shard_size=100000,
        row_group_rows=1,
        raw_pdf_max_chars=0,
        raw_pdf_timeout_seconds=20,
        parquet_compression="zstd",
        category_prefix="",
        min_year=0,
        max_year=0,
        keywords=[],
        gcs_prefix="gs://arxiv-dataset/arxiv/pdf",
        lease_timeout_seconds=600,
        progress_every=0,
        progress_path="",
    )

    lease = coordinator.lease(worker_id="worker-1", max_records=2)
    assert lease["status"] == "running"
    assert len(lease["records"]) == 2
    assert lease["lease_id"]
    heartbeat = coordinator.heartbeat(worker_id="worker-1", lease_id=lease["lease_id"])
    assert heartbeat["lease_id"] == lease["lease_id"]

    records = lease["records"]
    rows = [
        {
            "paper_id": record["paper_id"],
            "canonical_paper_id": record["canonical_paper_id"],
            "paper_version": record["paper_version"],
            "pdf_path": record["pdf_path"],
            "title": record["title"],
            "abstract": record["abstract"],
            "authors": record["authors"],
            "categories": record["categories"],
            "license": record["license"],
            "update_date": record["update_date"],
            "version_count": record["version_count"],
            "metadata_found": True,
            "text": f"text for {record['paper_id']}",
            "text_source": "raw_pdf_preextracted",
            "text_is_partial": False,
            "text_char_count": 10,
            "text_line_count": 1,
            "token_count": 1,
            "page_count": 1,
            "token_types": ["raw_text_preextracted"],
            "token_type_counts_json": '{"raw_text_preextracted":1}',
        }
        for record in records
    ]

    progress = coordinator.submit(
        worker_id="worker-1",
        lease_id=lease["lease_id"],
        rows=rows,
        worker_stats={
            "download_batches": 1,
            "download_requested": 2,
            "downloaded_pdfs": 2,
            "deleted_temp_pdfs": 2,
            "skipped_empty": 0,
            "missing_downloads": 0,
        },
    )
    assert progress["extracted_rows"] == 2

    stats = coordinator.finalize()
    assert stats["status"] == "completed"
    assert stats["extracted_rows"] == 2
    assert stats["downloaded_pdfs"] == 2

    shard_paths = sorted(out_dir.glob("paper_text_backfill_*.parquet"))
    assert len(shard_paths) == 1
    rows_out = pq.read_table(str(shard_paths[0])).to_pylist()
    assert {row["canonical_paper_id"] for row in rows_out} == {"2401.00001", "2401.00002"}

    progress_path = out_dir / "distributed_backfill_progress.json"
    assert progress_path.is_file()
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress_payload["status"] == "completed"


def test_run_coordinator_processes_locally_without_remote_workers(tmp_path: Path, monkeypatch) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "2401.00010",
                "title": "Paper 10",
                "abstract": "Abstract 10",
                "authors": "Alice",
                "categories": "cs.AI",
                "license": "cc-by",
                "update_date": "2024-01-10",
                "versions": [{"version": "v1"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_gsutil(urls, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            stem = Path(str(url)).stem
            (dest_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4 fake\n")
        return 0

    def fake_extract_record_row(record, *, pdf_path, raw_pdf_max_chars, raw_pdf_timeout_seconds):
        return {
            "paper_id": str(record.get("paper_id") or ""),
            "canonical_paper_id": str(record.get("canonical_paper_id") or ""),
            "paper_version": str(record.get("paper_version") or ""),
            "pdf_path": str(record.get("pdf_path") or ""),
            "title": str(record.get("title") or ""),
            "abstract": str(record.get("abstract") or ""),
            "authors": str(record.get("authors") or ""),
            "categories": str(record.get("categories") or ""),
            "license": str(record.get("license") or ""),
            "update_date": str(record.get("update_date") or ""),
            "version_count": int(record.get("version_count") or 0),
            "metadata_found": bool(record.get("metadata_found")),
            "text": f"text for {pdf_path.stem}",
            "text_source": "raw_pdf_preextracted",
            "text_is_partial": False,
            "text_char_count": 12,
            "text_line_count": 1,
            "token_count": 1,
            "page_count": 1,
            "token_types": ["raw_text_preextracted"],
            "token_type_counts_json": '{"raw_text_preextracted":1}',
        }

    monkeypatch.setattr("scripts.distributed_paper_text_backfill._run_gsutil_cp", fake_gsutil)
    monkeypatch.setattr(
        "scripts.distributed_paper_text_backfill._extract_record_row",
        fake_extract_record_row,
    )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    thread = threading.Thread(
        target=run_coordinator,
        kwargs={
            "auth_token": "secret",
            "bind_host": "127.0.0.1",
            "port": port,
            "existing_structured_dirs": [],
            "existing_parquet_dirs": [],
            "existing_parquet_paths": [],
            "metadata_path": str(metadata_path),
            "out_dir": str(out_dir),
            "temp_pdf_dir": str(temp_pdf_dir),
            "target_total_papers": 1,
            "max_papers": 0,
            "download_batch_size": 1,
            "shard_size": 100000,
            "row_group_rows": 1,
            "raw_pdf_max_chars": 0,
            "raw_pdf_timeout_seconds": 20,
            "parquet_compression": "zstd",
            "category_prefix": "",
            "min_year": 0,
            "max_year": 0,
            "keywords": [],
            "gcs_prefix": "gs://arxiv-dataset/arxiv/pdf",
            "lease_timeout_seconds": 600,
            "progress_every": 0,
            "progress_path": "",
            "local_workers": 1,
            "local_max_records_per_lease": 1,
            "local_extract_workers": 1,
            "local_retry_missing_downloads": False,
            "local_heartbeat_interval_seconds": 5,
        },
        daemon=True,
    )
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()

    shard_paths = sorted(out_dir.glob("paper_text_backfill_*.parquet"))
    assert len(shard_paths) == 1
    rows_out = pq.read_table(str(shard_paths[0])).to_pylist()
    assert {row["canonical_paper_id"] for row in rows_out} == {"2401.00010"}
