from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.backfill_missing_paper_text_shards import _backfill_schema
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


def test_target_total_papers_counts_existing_output_rows_only_for_output_not_completion(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    base_dir = tmp_path / "base"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2401.01001",
                        "title": "Base paper",
                        "abstract": "Base",
                        "authors": "Alice",
                        "categories": "cs.AI",
                        "license": "cc-by",
                        "update_date": "2024-01-01",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.01002",
                        "title": "Existing out paper",
                        "abstract": "Out",
                        "authors": "Bob",
                        "categories": "cs.LG",
                        "license": "cc-by",
                        "update_date": "2024-01-02",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.01003",
                        "title": "New paper 1",
                        "abstract": "New 1",
                        "authors": "Carol",
                        "categories": "cs.CL",
                        "license": "cc-by",
                        "update_date": "2024-01-03",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.01004",
                        "title": "New paper 2",
                        "abstract": "New 2",
                        "authors": "Dave",
                        "categories": "cs.IR",
                        "license": "cc-by",
                        "update_date": "2024-01-04",
                        "versions": [{"version": "v1"}],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    def row_for(paper_id: str) -> dict:
        return {
            "paper_id": f"{paper_id}v1",
            "canonical_paper_id": paper_id,
            "paper_version": "v1",
            "pdf_path": f"gs://arxiv-dataset/arxiv/pdf/{paper_id}.pdf",
            "title": f"title {paper_id}",
            "abstract": f"abstract {paper_id}",
            "authors": "Author",
            "categories": "cs.AI",
            "license": "cc-by",
            "update_date": "2024-01-01",
            "version_count": 1,
            "metadata_found": True,
            "text": f"text for {paper_id}",
            "text_source": "raw_pdf_preextracted",
            "text_is_partial": False,
            "text_char_count": 16,
            "text_line_count": 1,
            "token_count": 1,
            "page_count": 1,
            "token_types": ["raw_text_preextracted"],
            "token_type_counts_json": '{"raw_text_preextracted":1}',
        }

    pq.write_table(
        pa.Table.from_pylist([row_for("2401.01001")], schema=_backfill_schema()),
        str(base_dir / "train_00000.parquet"),
    )
    pq.write_table(
        pa.Table.from_pylist([row_for("2401.01002")], schema=_backfill_schema()),
        str(out_dir / "paper_text_backfill_00000.parquet"),
    )

    coordinator = DistributedBackfillCoordinator(
        auth_token="secret",
        existing_structured_dirs=[],
        existing_parquet_dirs=[str(base_dir)],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        target_total_papers=4,
        max_papers=0,
        download_batch_size=1,
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

    first_lease = coordinator.lease(worker_id="worker-1", max_records=1)
    assert first_lease["status"] == "running"
    assert len(first_lease["records"]) == 1
    first_record = first_lease["records"][0]
    progress = coordinator.submit(
        worker_id="worker-1",
        lease_id=first_lease["lease_id"],
        rows=[row_for(str(first_record["canonical_paper_id"]))],
        worker_stats={"download_batches": 1, "download_requested": 1, "downloaded_pdfs": 1},
    )
    assert progress["covered_ids_after"] == 3
    assert progress["remaining_target_rows"] == 1
    assert progress["status"] == "running"

    second_lease = coordinator.lease(worker_id="worker-1", max_records=1)
    assert second_lease["status"] == "running"
    assert len(second_lease["records"]) == 1
    second_record = second_lease["records"][0]
    progress = coordinator.submit(
        worker_id="worker-1",
        lease_id=second_lease["lease_id"],
        rows=[row_for(str(second_record["canonical_paper_id"]))],
        worker_stats={"download_batches": 1, "download_requested": 1, "downloaded_pdfs": 1},
    )
    assert progress["covered_ids_after"] == 4
    assert progress["remaining_target_rows"] == 0
    assert progress["status"] == "completed"


def test_distributed_coordinator_resumes_from_saved_metadata_offset(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    out_dir = tmp_path / "out"
    temp_pdf_dir = tmp_path / "tmp"

    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2401.10001",
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
                        "id": "2401.10002",
                        "title": "Paper 2",
                        "abstract": "Abstract 2",
                        "authors": "Bob",
                        "categories": "cs.LG",
                        "license": "cc-by",
                        "update_date": "2024-01-02",
                        "versions": [{"version": "v1"}],
                    }
                ),
                json.dumps(
                    {
                        "id": "2401.10003",
                        "title": "Paper 3",
                        "abstract": "Abstract 3",
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

    coordinator = DistributedBackfillCoordinator(
        auth_token="secret",
        existing_structured_dirs=[],
        existing_parquet_dirs=[],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        target_total_papers=3,
        max_papers=0,
        download_batch_size=1,
        shard_size=1,
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

    lease = coordinator.lease(worker_id="worker-1", max_records=1)
    record = lease["records"][0]
    progress = coordinator.submit(
        worker_id="worker-1",
        lease_id=lease["lease_id"],
        rows=[
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
        ],
        worker_stats={},
    )
    assert progress["covered_ids_after"] == 1
    assert progress["resume_metadata_offset"] > 0

    resumed = DistributedBackfillCoordinator(
        auth_token="secret",
        existing_structured_dirs=[],
        existing_parquet_dirs=[],
        existing_parquet_paths=[],
        metadata_path=str(metadata_path),
        out_dir=str(out_dir),
        temp_pdf_dir=str(temp_pdf_dir),
        target_total_papers=3,
        max_papers=0,
        download_batch_size=1,
        shard_size=1,
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

    assert resumed.covered_ids_before == 1
    assert resumed.resume_metadata_offset > 0
    resumed_lease = resumed.lease(worker_id="worker-2", max_records=1)
    resumed_ids = {row["canonical_paper_id"] for row in resumed_lease["records"]}
    assert resumed_ids == {"2401.10002"}
