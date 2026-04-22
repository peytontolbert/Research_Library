from __future__ import annotations

import argparse
import gzip
import json
import secrets
import shutil
import threading
import time
import uuid
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.backfill_missing_paper_text_shards import (
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_RAW_PDF_MAX_CHARS,
    DEFAULT_RAW_PDF_TIMEOUT_SECONDS,
    DEFAULT_ROW_GROUP_ROWS,
    DEFAULT_SHARD_SIZE,
    _cleanup_temporary_shards,
    _backfill_schema,
    _finalize_writer,
    _next_shard_index,
    _open_writer,
)
from scripts.backfill_paper_text_from_gcs import (
    DEFAULT_CATEGORY_PREFIX,
    DEFAULT_DIRECT_ARXIV_PDF_FALLBACK,
    DEFAULT_DOWNLOAD_BATCH_SIZE,
    DEFAULT_EXTRACT_WORKERS,
    DEFAULT_GCS_PREFIX,
    DEFAULT_METADATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROGRESS_EVERY,
    DEFAULT_RETRY_MISSING_DOWNLOADS,
    DEFAULT_TEMP_PDF_DIR,
    _covered_paper_ids,
    _extract_record_row,
    _iter_metadata_candidates,
    _resolve_download_candidate,
    _run_gsutil_cp,
    _write_json_atomic,
)


DEFAULT_COORDINATOR_HOST = "0.0.0.0"
DEFAULT_COORDINATOR_PORT = 8787
DEFAULT_LEASE_SIZE = 64
DEFAULT_LEASE_TIMEOUT_SECONDS = 1800
DEFAULT_IDLE_SECONDS = 15
DEFAULT_PROGRESS_FILENAME = "distributed_backfill_progress.json"
DEFAULT_STATS_FILENAME = "distributed_backfill_stats.json"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60
DEFAULT_LOCAL_WORKERS = 1


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_request(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    if handler.headers.get("Content-Encoding", "").lower() == "gzip":
        raw = gzip.decompress(raw)
    if not raw:
        return {}
    parsed = json.loads(raw.decode("utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _require_auth(handler: BaseHTTPRequestHandler, auth_token: str) -> bool:
    header = str(handler.headers.get("Authorization") or "")
    expected = f"Bearer {auth_token}"
    return secrets.compare_digest(header, expected)


def _http_json(
    *,
    method: str,
    url: str,
    auth_token: str,
    payload: Optional[Dict[str, Any]] = None,
    gzip_body: bool = False,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    body: Optional[bytes]
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    if payload is None:
        body = None
    else:
        body = json.dumps(payload).encode("utf-8")
        if gzip_body:
            body = gzip.compress(body)
            headers["Content-Encoding"] = "gzip"
    request = Request(url, data=body, headers=headers, method=method.upper())
    try:
        with urlopen(request, timeout=timeout_seconds) as resp:
            raw = resp.read()
            if not raw:
                return {}
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc


class DistributedBackfillCoordinator:
    def __init__(
        self,
        *,
        auth_token: str,
        existing_structured_dirs: Sequence[str],
        existing_parquet_dirs: Sequence[str],
        existing_parquet_paths: Sequence[str],
        metadata_path: str,
        out_dir: str,
        temp_pdf_dir: str,
        target_total_papers: int,
        max_papers: int,
        download_batch_size: int,
        shard_size: int,
        row_group_rows: int,
        raw_pdf_max_chars: int,
        raw_pdf_timeout_seconds: int,
        parquet_compression: str,
        category_prefix: str,
        min_year: int,
        max_year: int,
        keywords: Sequence[str],
        gcs_prefix: str,
        lease_timeout_seconds: int,
        progress_every: int,
        progress_path: str,
    ) -> None:
        self.auth_token = str(auth_token or "").strip()
        if not self.auth_token:
            raise ValueError("auth_token is required")

        self.lock = threading.Lock()
        self.existing_structured_dirs = [Path(path).resolve() for path in existing_structured_dirs if str(path).strip()]
        self.existing_parquet_dirs = [Path(path).resolve() for path in existing_parquet_dirs if str(path).strip()]
        self.existing_parquet_paths = [Path(path).resolve() for path in existing_parquet_paths if str(path).strip()]
        self.metadata_path = Path(metadata_path).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.temp_pdf_dir = Path(temp_pdf_dir).resolve()
        self.target_total_papers = int(target_total_papers or 0)
        self.max_papers = int(max_papers or 0)
        self.download_batch_size = int(download_batch_size or DEFAULT_DOWNLOAD_BATCH_SIZE)
        self.rows_per_file = max(1, int(shard_size or DEFAULT_SHARD_SIZE))
        self.row_group_rows = max(1, int(row_group_rows or DEFAULT_ROW_GROUP_ROWS))
        self.raw_pdf_max_chars = int(raw_pdf_max_chars or DEFAULT_RAW_PDF_MAX_CHARS)
        self.raw_pdf_timeout_seconds = int(raw_pdf_timeout_seconds or DEFAULT_RAW_PDF_TIMEOUT_SECONDS)
        self.parquet_compression = str(parquet_compression or DEFAULT_PARQUET_COMPRESSION)
        self.category_prefix = str(category_prefix or DEFAULT_CATEGORY_PREFIX)
        self.min_year = int(min_year or 0)
        self.max_year = int(max_year or 0)
        self.keywords = [str(term) for term in keywords if str(term).strip()]
        self.gcs_prefix = str(gcs_prefix or DEFAULT_GCS_PREFIX).rstrip("/")
        self.lease_timeout_seconds = max(60, int(lease_timeout_seconds or DEFAULT_LEASE_TIMEOUT_SECONDS))
        self.progress_every = int(progress_every or DEFAULT_PROGRESS_EVERY)
        self.progress_path = (
            Path(progress_path).resolve()
            if str(progress_path or "").strip()
            else (self.out_dir / DEFAULT_PROGRESS_FILENAME)
        )
        self.stats_path = self.out_dir / DEFAULT_STATS_FILENAME

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.temp_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.stale_temp_shards_removed = _cleanup_temporary_shards(self.out_dir)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata snapshot not found: {self.metadata_path}")

        self.resume_metadata_offset = self._load_resume_metadata_offset()
        self.next_metadata_offset = int(self.resume_metadata_offset)
        self.covered_ids = _covered_paper_ids(
            existing_structured_dirs=self.existing_structured_dirs,
            existing_parquet_dirs=self.existing_parquet_dirs,
            existing_parquet_paths=self.existing_parquet_paths,
            out_dir=self.out_dir,
        )
        self.covered_ids_before = len(self.covered_ids)
        self.output_rows_before = self._count_existing_output_rows()
        self.extracted_rows = self.output_rows_before

        self.metadata_scanned = 0
        self.metadata_matched = 0
        self.skipped_existing = 0
        self.skipped_empty = 0
        self.missing_downloads = 0
        self.download_batches = 0
        self.download_requested = 0
        self.downloaded_pdfs = 0
        self.deleted_temp_pdfs = 0
        self.version_fallback_uses = 0
        self.direct_pdf_fallback_uses = 0
        self.completed_shards = len(list(self.out_dir.glob("paper_text_backfill_*.parquet")))
        self.started_at = time.time()

        self.writer: Optional[pq.ParquetWriter] = None
        self.current_tmp_path: Optional[Path] = None
        self.current_final_path: Optional[Path] = None
        self.shard_idx = _next_shard_index(self.out_dir)
        self.active_shard_rows = 0
        self.rows_buffer: List[Dict[str, Any]] = []
        self.open_shard_offsets: List[int] = []

        self.pending_records: Deque[Dict[str, Any]] = deque()
        self.leases: Dict[str, Dict[str, Any]] = {}
        self.leased_ids: Dict[str, str] = {}
        self.iterator = _iter_metadata_candidates(
            metadata_path=self.metadata_path,
            category_prefix=self.category_prefix,
            min_year=self.min_year,
            max_year=self.max_year,
            keywords=self.keywords,
            gcs_prefix=self.gcs_prefix,
            start_offset=self.resume_metadata_offset,
        )
        self.iterator_exhausted = False
        self.status = "running"
        self._write_progress_locked()

    def _count_existing_output_rows(self) -> int:
        count = 0
        for shard_path in sorted(self.out_dir.glob("paper_text_backfill_*.parquet")):
            try:
                parquet_file = pq.ParquetFile(str(shard_path))
            except Exception:
                continue
            try:
                count += int(parquet_file.metadata.num_rows)
            except Exception:
                continue
        return count

    def _resume_signature(self) -> Dict[str, Any]:
        return {
            "metadata_path": str(self.metadata_path),
            "out_dir": str(self.out_dir),
            "category_prefix": self.category_prefix,
            "min_year": int(self.min_year),
            "max_year": int(self.max_year),
            "keywords": list(self.keywords),
            "gcs_prefix": self.gcs_prefix,
        }

    def _load_resume_metadata_offset(self) -> int:
        if not self.progress_path.is_file():
            return 0
        try:
            payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if not isinstance(payload, dict):
            return 0
        for key, value in self._resume_signature().items():
            if payload.get(key) != value:
                return 0
        try:
            return max(0, int(payload.get("resume_metadata_offset") or 0))
        except Exception:
            return 0

    def _new_rows_this_run(self) -> int:
        return max(0, int(self.extracted_rows) - int(self.output_rows_before))

    def _remaining_target_rows_locked(self) -> Optional[int]:
        remaining: Optional[int] = None
        if self.target_total_papers > 0:
            remaining = max(0, int(self.target_total_papers) - int(len(self.covered_ids)))
        if self.max_papers > 0:
            remaining_new_rows = max(0, int(self.max_papers) - int(self._new_rows_this_run()))
            remaining = remaining_new_rows if remaining is None else min(remaining, remaining_new_rows)
        return remaining

    def _resume_metadata_offset_locked(self) -> int:
        offsets: List[int] = []
        for record in self.pending_records:
            try:
                offsets.append(int(record.get("__metadata_start_offset") or 0))
            except Exception:
                continue
        for lease in self.leases.values():
            for record in (lease.get("records") or []):
                try:
                    offsets.append(int(record.get("__metadata_start_offset") or 0))
                except Exception:
                    continue
        offsets.extend(int(offset) for offset in self.open_shard_offsets if int(offset) >= 0)
        if offsets:
            return min(offsets)
        return int(self.next_metadata_offset)

    def _snapshot_locked(self) -> Dict[str, Any]:
        remaining = self._remaining_target_rows_locked()
        return {
            "status": self.status,
            "metadata_path": str(self.metadata_path),
            "out_dir": str(self.out_dir),
            "temp_pdf_dir": str(self.temp_pdf_dir),
            "covered_ids_before": int(self.covered_ids_before),
            "covered_ids_after": int(len(self.covered_ids)),
            "metadata_scanned": int(self.metadata_scanned),
            "metadata_matched": int(self.metadata_matched),
            "extracted_rows": int(self.extracted_rows),
            "skipped_existing": int(self.skipped_existing),
            "skipped_empty": int(self.skipped_empty),
            "missing_downloads": int(self.missing_downloads),
            "download_batches": int(self.download_batches),
            "download_requested": int(self.download_requested),
            "downloaded_pdfs": int(self.downloaded_pdfs),
            "deleted_temp_pdfs": int(self.deleted_temp_pdfs),
            "version_fallback_uses": int(self.version_fallback_uses),
            "direct_pdf_fallback_uses": int(self.direct_pdf_fallback_uses),
            "resume_metadata_offset": int(self._resume_metadata_offset_locked()),
            "next_metadata_offset": int(self.next_metadata_offset),
            "target_total_papers": int(self.target_total_papers),
            "max_papers": int(self.max_papers),
            "remaining_target_rows": None if remaining is None else int(remaining),
            "category_prefix": self.category_prefix,
            "min_year": int(self.min_year),
            "max_year": int(self.max_year),
            "keywords": list(self.keywords),
            "gcs_prefix": self.gcs_prefix,
            "raw_pdf_max_chars": int(self.raw_pdf_max_chars),
            "raw_pdf_timeout_seconds": int(self.raw_pdf_timeout_seconds),
            "parquet_compression": self.parquet_compression,
            "rows_per_parquet_file": int(self.rows_per_file),
            "row_group_rows": int(self.row_group_rows),
            "lease_timeout_seconds": int(self.lease_timeout_seconds),
            "open_shard_rows_buffered": int(len(self.rows_buffer)),
            "open_shard_rows_written": int(self.active_shard_rows),
            "pending_records": int(len(self.pending_records)),
            "active_leases": int(len(self.leases)),
            "parquet_shards_written": int(self.completed_shards),
            "stale_temp_shards_removed": int(self.stale_temp_shards_removed),
            "elapsed_seconds": time.time() - self.started_at,
        }

    def _write_progress_locked(self) -> None:
        _write_json_atomic(self.progress_path, self._snapshot_locked())

    def _flush_rows_locked(self, *, close_after: bool = False) -> None:
        if self.rows_buffer:
            if self.writer is None:
                self.writer, self.current_tmp_path, self.current_final_path = _open_writer(
                    self.out_dir,
                    self.shard_idx,
                    compression=self.parquet_compression,
                )
            table = pa.Table.from_pylist(list(self.rows_buffer), schema=_backfill_schema())
            self.writer.write_table(table)
            self.active_shard_rows += len(self.rows_buffer)
            self.rows_buffer = []

        should_close = close_after or (
            self.writer is not None and self.active_shard_rows >= self.rows_per_file
        )
        if should_close and self.writer is not None:
            self.writer.close()
            if self.current_tmp_path is None or self.current_final_path is None:
                raise RuntimeError("Parquet shard paths were not initialized.")
            _finalize_writer(self.current_tmp_path, self.current_final_path)
            self.writer = None
            self.current_tmp_path = None
            self.current_final_path = None
            self.active_shard_rows = 0
            self.open_shard_offsets = []
            self.shard_idx += 1
            self.completed_shards += 1

    def _expire_leases_locked(self) -> None:
        now = time.time()
        expired: List[str] = []
        for lease_id, lease in self.leases.items():
            leased_at = float(lease.get("leased_at") or 0.0)
            if now - leased_at > self.lease_timeout_seconds:
                expired.append(lease_id)
        for lease_id in expired:
            lease = self.leases.pop(lease_id, None)
            if not lease:
                continue
            records = lease.get("records") or []
            for record in reversed(records):
                canonical_paper_id = str(record.get("canonical_paper_id") or "")
                self.leased_ids.pop(canonical_paper_id, None)
                self.pending_records.appendleft(record)

    def _refill_pending_locked(self, target_records: int) -> None:
        if self.iterator_exhausted:
            return
        while len(self.pending_records) < target_records:
            if self.target_total_papers > 0 and len(self.covered_ids) >= self.target_total_papers:
                break
            if self.max_papers > 0 and self._new_rows_this_run() >= self.max_papers:
                break
            try:
                record = next(self.iterator)
            except StopIteration:
                self.iterator_exhausted = True
                break
            try:
                self.next_metadata_offset = max(
                    int(self.next_metadata_offset),
                    int(record.get("__metadata_end_offset") or self.next_metadata_offset),
                )
            except Exception:
                pass
            self.metadata_scanned += 1
            canonical_paper_id = str(record.get("canonical_paper_id") or "").strip()
            if not canonical_paper_id:
                continue
            if canonical_paper_id in self.covered_ids or canonical_paper_id in self.leased_ids:
                self.skipped_existing += 1
                continue
            self.metadata_matched += 1
            self.pending_records.append(record)

    def _maybe_complete_locked(self) -> None:
        if self.target_total_papers > 0 and len(self.covered_ids) >= self.target_total_papers:
            self.status = "completed"
        elif self.max_papers > 0 and self._new_rows_this_run() >= self.max_papers:
            self.status = "completed"
        elif self.iterator_exhausted and not self.pending_records and not self.leases:
            self.status = "completed"
        else:
            self.status = "running"

    def lease(self, *, worker_id: str, max_records: int) -> Dict[str, Any]:
        with self.lock:
            self._expire_leases_locked()
            self._refill_pending_locked(max(1, int(max_records)))
            self._maybe_complete_locked()
            if not self.pending_records:
                self._write_progress_locked()
                return {
                    "status": self.status,
                    "lease_id": "",
                    "records": [],
                    "lease_timeout_seconds": int(self.lease_timeout_seconds),
                }

            take = min(max(1, int(max_records)), len(self.pending_records))
            records = [self.pending_records.popleft() for _ in range(take)]
            lease_id = str(uuid.uuid4())
            now = time.time()
            for record in records:
                canonical_paper_id = str(record.get("canonical_paper_id") or "")
                self.leased_ids[canonical_paper_id] = lease_id
            self.leases[lease_id] = {
                "worker_id": str(worker_id or ""),
                "leased_at": now,
                "records": records,
            }
            self._write_progress_locked()
            return {
                "status": self.status,
                "lease_id": lease_id,
                "records": records,
                "lease_timeout_seconds": int(self.lease_timeout_seconds),
            }

    def heartbeat(self, *, worker_id: str, lease_id: str) -> Dict[str, Any]:
        with self.lock:
            self._expire_leases_locked()
            lease = self.leases.get(str(lease_id))
            if not lease:
                raise ValueError(f"Unknown or expired lease_id: {lease_id}")
            expected_worker_id = str(lease.get("worker_id") or "")
            if expected_worker_id and expected_worker_id != str(worker_id or ""):
                raise ValueError("worker_id does not match lease owner")
            lease["leased_at"] = time.time()
            self._maybe_complete_locked()
            self._write_progress_locked()
            return {
                "status": self.status,
                "lease_id": str(lease_id),
                "lease_timeout_seconds": int(self.lease_timeout_seconds),
            }

    def submit(
        self,
        *,
        worker_id: str,
        lease_id: str,
        rows: Sequence[Dict[str, Any]],
        worker_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            self._expire_leases_locked()
            lease = self.leases.pop(str(lease_id), None)
            if not lease:
                raise ValueError(f"Unknown or expired lease_id: {lease_id}")
            expected_worker_id = str(lease.get("worker_id") or "")
            if expected_worker_id and expected_worker_id != str(worker_id or ""):
                raise ValueError("worker_id does not match lease owner")

            assigned_ids = {
                str(record.get("canonical_paper_id") or "").strip()
                for record in (lease.get("records") or [])
                if str(record.get("canonical_paper_id") or "").strip()
            }
            lease_offsets = {
                str(record.get("canonical_paper_id") or "").strip(): int(record.get("__metadata_start_offset") or 0)
                for record in (lease.get("records") or [])
                if str(record.get("canonical_paper_id") or "").strip()
            }
            row_ids = {
                str(row.get("canonical_paper_id") or "").strip()
                for row in rows
                if str(row.get("canonical_paper_id") or "").strip()
            }
            if not row_ids.issubset(assigned_ids):
                raise ValueError("Submitted rows contain canonical ids not present in the lease")

            for canonical_paper_id in assigned_ids:
                self.leased_ids.pop(canonical_paper_id, None)

            worker_stats = dict(worker_stats or {})
            self.skipped_empty += int(worker_stats.get("skipped_empty") or 0)
            self.missing_downloads += int(worker_stats.get("missing_downloads") or 0)
            self.download_batches += int(worker_stats.get("download_batches") or 0)
            self.download_requested += int(worker_stats.get("download_requested") or 0)
            self.downloaded_pdfs += int(worker_stats.get("downloaded_pdfs") or 0)
            self.deleted_temp_pdfs += int(worker_stats.get("deleted_temp_pdfs") or 0)
            self.version_fallback_uses += int(worker_stats.get("version_fallback_uses") or 0)
            self.direct_pdf_fallback_uses += int(worker_stats.get("direct_pdf_fallback_uses") or 0)

            for row in rows:
                canonical_paper_id = str(row.get("canonical_paper_id") or "").strip()
                if not canonical_paper_id or canonical_paper_id in self.covered_ids:
                    continue
                self.rows_buffer.append(dict(row))
                self.covered_ids.add(canonical_paper_id)
                self.extracted_rows += 1
                self.open_shard_offsets.append(int(lease_offsets.get(canonical_paper_id, 0)))
                if len(self.rows_buffer) >= self.row_group_rows:
                    self._flush_rows_locked()

            self._maybe_complete_locked()
            self._write_progress_locked()
            return self._snapshot_locked()

    def finalize(self) -> Dict[str, Any]:
        with self.lock:
            self._flush_rows_locked(close_after=True)
            self._maybe_complete_locked()
            stats = self._snapshot_locked()
            _write_json_atomic(self.stats_path, stats)
            _write_json_atomic(self.progress_path, stats)
            return stats

    def progress(self) -> Dict[str, Any]:
        with self.lock:
            self._expire_leases_locked()
            self._maybe_complete_locked()
            progress = self._snapshot_locked()
            self._write_progress_locked()
            return progress


class _CoordinatorHandler(BaseHTTPRequestHandler):
    coordinator: DistributedBackfillCoordinator

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return
        if not _require_auth(self, self.coordinator.auth_token):
            _json_response(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if path == "/progress":
            _json_response(self, HTTPStatus.OK, self.coordinator.progress())
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not _require_auth(self, self.coordinator.auth_token):
            _json_response(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            payload = _read_json_request(self)
            if path == "/lease":
                worker_id = str(payload.get("worker_id") or "").strip()
                max_records = int(payload.get("max_records") or DEFAULT_LEASE_SIZE)
                result = self.coordinator.lease(worker_id=worker_id, max_records=max_records)
                _json_response(self, HTTPStatus.OK, result)
                return
            if path == "/submit":
                worker_id = str(payload.get("worker_id") or "").strip()
                lease_id = str(payload.get("lease_id") or "").strip()
                rows = payload.get("rows") or []
                worker_stats = payload.get("worker_stats") or {}
                result = self.coordinator.submit(
                    worker_id=worker_id,
                    lease_id=lease_id,
                    rows=rows if isinstance(rows, list) else [],
                    worker_stats=worker_stats if isinstance(worker_stats, dict) else {},
                )
                _json_response(self, HTTPStatus.OK, result)
                return
            if path == "/heartbeat":
                worker_id = str(payload.get("worker_id") or "").strip()
                lease_id = str(payload.get("lease_id") or "").strip()
                result = self.coordinator.heartbeat(worker_id=worker_id, lease_id=lease_id)
                _json_response(self, HTTPStatus.OK, result)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def run_coordinator(
    *,
    auth_token: str,
    bind_host: str,
    port: int,
    existing_structured_dirs: Sequence[str],
    existing_parquet_dirs: Sequence[str],
    existing_parquet_paths: Sequence[str],
    metadata_path: str,
    out_dir: str,
    temp_pdf_dir: str,
    target_total_papers: int,
    max_papers: int,
    download_batch_size: int,
    shard_size: int,
    row_group_rows: int,
    raw_pdf_max_chars: int,
    raw_pdf_timeout_seconds: int,
    parquet_compression: str,
    category_prefix: str,
    min_year: int,
    max_year: int,
    keywords: Sequence[str],
    gcs_prefix: str,
    lease_timeout_seconds: int,
    progress_every: int,
    progress_path: str,
    local_workers: int,
    local_max_records_per_lease: int,
    local_extract_workers: int,
    local_retry_missing_downloads: bool,
    local_direct_arxiv_pdf_fallback: bool = DEFAULT_DIRECT_ARXIV_PDF_FALLBACK,
    local_heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    coordinator = DistributedBackfillCoordinator(
        auth_token=auth_token,
        existing_structured_dirs=existing_structured_dirs,
        existing_parquet_dirs=existing_parquet_dirs,
        existing_parquet_paths=existing_parquet_paths,
        metadata_path=metadata_path,
        out_dir=out_dir,
        temp_pdf_dir=temp_pdf_dir,
        target_total_papers=target_total_papers,
        max_papers=max_papers,
        download_batch_size=download_batch_size,
        shard_size=shard_size,
        row_group_rows=row_group_rows,
        raw_pdf_max_chars=raw_pdf_max_chars,
        raw_pdf_timeout_seconds=raw_pdf_timeout_seconds,
        parquet_compression=parquet_compression,
        category_prefix=category_prefix,
        min_year=min_year,
        max_year=max_year,
        keywords=keywords,
        gcs_prefix=gcs_prefix,
        lease_timeout_seconds=lease_timeout_seconds,
        progress_every=progress_every,
        progress_path=progress_path,
    )
    handler_cls = type(
        "CoordinatorHandler",
        (_CoordinatorHandler,),
        {"coordinator": coordinator},
    )
    server = ThreadingHTTPServer((bind_host, int(port)), handler_cls)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    worker_threads: List[threading.Thread] = []
    coordinator_url = f"http://127.0.0.1:{int(port)}"
    for worker_idx in range(max(0, int(local_workers or 0))):
        worker_thread = threading.Thread(
            target=run_worker,
            kwargs={
                "coordinator_url": coordinator_url,
                "auth_token": auth_token,
                "worker_id": f"coordinator-local-{worker_idx:02d}",
                "temp_pdf_dir": str(Path(temp_pdf_dir).resolve() / f"local_worker_{worker_idx:02d}"),
                "max_records_per_lease": int(local_max_records_per_lease or DEFAULT_LEASE_SIZE),
                "idle_seconds": DEFAULT_IDLE_SECONDS,
                "raw_pdf_max_chars": int(raw_pdf_max_chars),
                "raw_pdf_timeout_seconds": int(raw_pdf_timeout_seconds),
                "extract_workers": int(local_extract_workers or DEFAULT_EXTRACT_WORKERS),
                "retry_missing_downloads": bool(local_retry_missing_downloads),
                "direct_arxiv_pdf_fallback": bool(local_direct_arxiv_pdf_fallback),
                "heartbeat_interval_seconds": int(
                    local_heartbeat_interval_seconds or DEFAULT_HEARTBEAT_INTERVAL_SECONDS
                ),
            },
            daemon=True,
        )
        worker_thread.start()
        worker_threads.append(worker_thread)
    try:
        while True:
            progress = coordinator.progress()
            if str(progress.get("status") or "") == "completed":
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        for worker_thread in worker_threads:
            worker_thread.join(timeout=5)
        stats = coordinator.finalize()
        print(json.dumps({"stats": stats}, indent=2))


def run_worker(
    *,
    coordinator_url: str,
    auth_token: str,
    worker_id: str,
    temp_pdf_dir: str,
    max_records_per_lease: int,
    idle_seconds: int,
    raw_pdf_max_chars: int,
    raw_pdf_timeout_seconds: int,
    extract_workers: int,
    retry_missing_downloads: bool,
    direct_arxiv_pdf_fallback: bool = DEFAULT_DIRECT_ARXIV_PDF_FALLBACK,
    heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    coordinator_url = coordinator_url.rstrip("/") + "/"
    worker_id = str(worker_id or uuid.uuid4())
    temp_pdf_dir_path = Path(temp_pdf_dir).resolve()
    temp_pdf_dir_path.mkdir(parents=True, exist_ok=True)
    extract_workers = max(1, int(extract_workers or DEFAULT_EXTRACT_WORKERS))
    heartbeat_interval_seconds = max(5, int(heartbeat_interval_seconds or DEFAULT_HEARTBEAT_INTERVAL_SECONDS))

    while True:
        lease = _http_json(
            method="POST",
            url=urljoin(coordinator_url, "lease"),
            auth_token=auth_token,
            payload={
                "worker_id": worker_id,
                "max_records": int(max_records_per_lease),
            },
            timeout_seconds=120,
        )
        status = str(lease.get("status") or "")
        records = lease.get("records") or []
        lease_id = str(lease.get("lease_id") or "")
        if not records:
            if status == "completed":
                return
            time.sleep(max(1, int(idle_seconds)))
            continue
        lease_timeout_seconds = int(lease.get("lease_timeout_seconds") or DEFAULT_LEASE_TIMEOUT_SECONDS)

        batch_dir = temp_pdf_dir_path / f"lease_{lease_id}"
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        download_batches = 0
        download_requested = 0
        downloaded_pdfs = 0
        deleted_temp_pdfs = 0
        skipped_empty = 0
        missing_downloads = 0
        version_fallback_uses = 0
        direct_pdf_fallback_uses = 0
        rows: List[Dict[str, Any]] = []

        heartbeat_stop = threading.Event()

        def heartbeat_loop() -> None:
            interval = min(heartbeat_interval_seconds, max(5, lease_timeout_seconds // 3))
            while not heartbeat_stop.wait(interval):
                try:
                    _http_json(
                        method="POST",
                        url=urljoin(coordinator_url, "heartbeat"),
                        auth_token=auth_token,
                        payload={
                            "worker_id": worker_id,
                            "lease_id": lease_id,
                        },
                        timeout_seconds=60,
                    )
                except Exception:
                    continue

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        urls = [str(record.get("gcs_url") or "") for record in records]
        download_requested += len(urls)
        _run_gsutil_cp(urls, batch_dir)
        download_batches += 1
        batch_pdf_paths = sorted(batch_dir.rglob("*.pdf"))
        downloaded_pdfs += len(batch_pdf_paths)

        try:
            resolved_records: List[Tuple[Dict[str, Any], Path]] = []
            for record in records:
                resolved_record, pdf_path, used_fallback, used_direct_pdf_fallback = _resolve_download_candidate(
                    record,
                    batch_dir=batch_dir,
                    retry_missing_downloads=bool(retry_missing_downloads),
                    direct_arxiv_pdf_fallback=bool(direct_arxiv_pdf_fallback),
                )
                if resolved_record is None or pdf_path is None:
                    missing_downloads += 1
                    continue
                if used_fallback:
                    version_fallback_uses += 1
                if used_direct_pdf_fallback:
                    direct_pdf_fallback_uses += 1
                resolved_records.append((resolved_record, pdf_path))

            if extract_workers <= 1 or len(resolved_records) <= 1:
                extraction_results = [
                    _extract_record_row(
                        resolved_record,
                        pdf_path=pdf_path,
                        raw_pdf_max_chars=int(raw_pdf_max_chars),
                        raw_pdf_timeout_seconds=int(raw_pdf_timeout_seconds),
                    )
                    for resolved_record, pdf_path in resolved_records
                ]
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                extraction_results = []
                with ThreadPoolExecutor(max_workers=extract_workers) as executor:
                    futures = [
                        executor.submit(
                            _extract_record_row,
                            resolved_record,
                            pdf_path=pdf_path,
                            raw_pdf_max_chars=int(raw_pdf_max_chars),
                            raw_pdf_timeout_seconds=int(raw_pdf_timeout_seconds),
                        )
                        for resolved_record, pdf_path in resolved_records
                    ]
                    for future in as_completed(futures):
                        extraction_results.append(future.result())

            for row in extraction_results:
                if row is None:
                    skipped_empty += 1
                    continue
                rows.append(row)
        finally:
            deleted_temp_pdfs += len(list(batch_dir.rglob("*.pdf")))
            shutil.rmtree(batch_dir, ignore_errors=True)
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

        _http_json(
            method="POST",
            url=urljoin(coordinator_url, "submit"),
            auth_token=auth_token,
            payload={
                "worker_id": worker_id,
                "lease_id": lease_id,
                "rows": rows,
                "worker_stats": {
                    "download_batches": int(download_batches),
                    "download_requested": int(download_requested),
                    "downloaded_pdfs": int(downloaded_pdfs),
                        "deleted_temp_pdfs": int(deleted_temp_pdfs),
                        "skipped_empty": int(skipped_empty),
                        "missing_downloads": int(missing_downloads),
                        "version_fallback_uses": int(version_fallback_uses),
                        "direct_pdf_fallback_uses": int(direct_pdf_fallback_uses),
                    },
                },
            gzip_body=True,
            timeout_seconds=600,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed authenticated coordinator/worker backfill for temporary "
            "arXiv PDF downloads and central parquet aggregation."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    coordinator = subparsers.add_parser("coordinator")
    coordinator.add_argument("--auth-token", type=str, required=True)
    coordinator.add_argument("--bind-host", type=str, default=DEFAULT_COORDINATOR_HOST)
    coordinator.add_argument("--port", type=int, default=DEFAULT_COORDINATOR_PORT)
    coordinator.add_argument("--existing-structured-dir", action="append", dest="existing_structured_dirs", default=None)
    coordinator.add_argument("--existing-parquet-dir", action="append", dest="existing_parquet_dirs", default=None)
    coordinator.add_argument("--existing-parquet", action="append", dest="existing_parquet_paths", default=None)
    coordinator.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    coordinator.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    coordinator.add_argument("--temp-pdf-dir", type=str, default=str(DEFAULT_TEMP_PDF_DIR))
    coordinator.add_argument("--category-prefix", type=str, default=DEFAULT_CATEGORY_PREFIX)
    coordinator.add_argument("--min-year", type=int, default=0)
    coordinator.add_argument("--max-year", type=int, default=0)
    coordinator.add_argument("--keyword", action="append", dest="keywords", default=None)
    coordinator.add_argument("--gcs-prefix", type=str, default=DEFAULT_GCS_PREFIX)
    coordinator.add_argument("--lease-timeout-seconds", type=int, default=DEFAULT_LEASE_TIMEOUT_SECONDS)
    coordinator.add_argument("--progress-path", type=str, default="")
    coordinator.add_argument("--download-batch-size", type=int, default=DEFAULT_DOWNLOAD_BATCH_SIZE)
    coordinator.add_argument("--target-total-papers", type=int, default=0)
    coordinator.add_argument("--max-papers", type=int, default=0)
    coordinator.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    coordinator.add_argument("--row-group-rows", type=int, default=DEFAULT_ROW_GROUP_ROWS)
    coordinator.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    coordinator.add_argument("--raw-pdf-max-chars", type=int, default=DEFAULT_RAW_PDF_MAX_CHARS)
    coordinator.add_argument("--raw-pdf-timeout-seconds", type=int, default=DEFAULT_RAW_PDF_TIMEOUT_SECONDS)
    coordinator.add_argument("--parquet-compression", type=str, default=DEFAULT_PARQUET_COMPRESSION)
    coordinator.add_argument(
        "--local-workers",
        type=int,
        default=DEFAULT_LOCAL_WORKERS,
        help=(
            "Local worker threads to run on the coordinator host. Default: 1. "
            "Set to 0 for scheduler-only mode."
        ),
    )
    coordinator.add_argument(
        "--local-max-records-per-lease",
        type=int,
        default=DEFAULT_LEASE_SIZE,
        help="Lease size for coordinator-host local workers. Default: 64.",
    )
    coordinator.add_argument(
        "--local-extract-workers",
        type=int,
        default=DEFAULT_EXTRACT_WORKERS,
        help="Per-local-worker concurrent PDF extraction workers. Default follows CPU count.",
    )
    coordinator.add_argument("--local-retry-missing-downloads", action="store_true")
    coordinator.add_argument(
        "--disable-local-direct-arxiv-pdf-fallback",
        action="store_true",
        help="Disable direct https://arxiv.org/pdf fallback for coordinator-host local workers.",
    )
    coordinator.add_argument(
        "--local-heartbeat-interval-seconds",
        type=int,
        default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        help="Heartbeat interval for coordinator-host local workers.",
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument("--coordinator-url", type=str, required=True)
    worker.add_argument("--auth-token", type=str, required=True)
    worker.add_argument("--worker-id", type=str, default="")
    worker.add_argument("--temp-pdf-dir", type=str, default=str(DEFAULT_TEMP_PDF_DIR))
    worker.add_argument("--max-records-per-lease", type=int, default=DEFAULT_LEASE_SIZE)
    worker.add_argument("--idle-seconds", type=int, default=DEFAULT_IDLE_SECONDS)
    worker.add_argument("--raw-pdf-max-chars", type=int, default=DEFAULT_RAW_PDF_MAX_CHARS)
    worker.add_argument("--raw-pdf-timeout-seconds", type=int, default=DEFAULT_RAW_PDF_TIMEOUT_SECONDS)
    worker.add_argument("--extract-workers", type=int, default=DEFAULT_EXTRACT_WORKERS)
    worker.add_argument("--retry-missing-downloads", action="store_true")
    worker.add_argument("--disable-direct-arxiv-pdf-fallback", action="store_true")
    worker.add_argument("--heartbeat-interval-seconds", type=int, default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS)

    args = parser.parse_args()
    if args.mode == "coordinator":
        run_coordinator(
            auth_token=args.auth_token,
            bind_host=args.bind_host,
            port=args.port,
            existing_structured_dirs=args.existing_structured_dirs or [],
            existing_parquet_dirs=args.existing_parquet_dirs or [],
            existing_parquet_paths=args.existing_parquet_paths or [],
            metadata_path=args.metadata_path,
            out_dir=args.out_dir,
            temp_pdf_dir=args.temp_pdf_dir,
            target_total_papers=int(args.target_total_papers),
            max_papers=int(args.max_papers),
            download_batch_size=int(args.download_batch_size),
            shard_size=int(args.shard_size),
            row_group_rows=int(args.row_group_rows),
            raw_pdf_max_chars=int(args.raw_pdf_max_chars),
            raw_pdf_timeout_seconds=int(args.raw_pdf_timeout_seconds),
            parquet_compression=str(args.parquet_compression or DEFAULT_PARQUET_COMPRESSION),
            category_prefix=str(args.category_prefix or DEFAULT_CATEGORY_PREFIX),
            min_year=int(args.min_year or 0),
            max_year=int(args.max_year or 0),
            keywords=args.keywords or [],
            gcs_prefix=str(args.gcs_prefix or DEFAULT_GCS_PREFIX),
            lease_timeout_seconds=int(args.lease_timeout_seconds or DEFAULT_LEASE_TIMEOUT_SECONDS),
            progress_every=int(args.progress_every or DEFAULT_PROGRESS_EVERY),
            progress_path=str(args.progress_path or ""),
            local_workers=int(args.local_workers or 0),
            local_max_records_per_lease=int(args.local_max_records_per_lease or DEFAULT_LEASE_SIZE),
            local_extract_workers=int(args.local_extract_workers or DEFAULT_EXTRACT_WORKERS),
            local_retry_missing_downloads=bool(args.local_retry_missing_downloads),
            local_direct_arxiv_pdf_fallback=not bool(args.disable_local_direct_arxiv_pdf_fallback),
            local_heartbeat_interval_seconds=int(
                args.local_heartbeat_interval_seconds or DEFAULT_HEARTBEAT_INTERVAL_SECONDS
            ),
        )
        return

    run_worker(
        coordinator_url=args.coordinator_url,
        auth_token=args.auth_token,
        worker_id=args.worker_id,
        temp_pdf_dir=args.temp_pdf_dir,
        max_records_per_lease=int(args.max_records_per_lease or DEFAULT_LEASE_SIZE),
        idle_seconds=int(args.idle_seconds or DEFAULT_IDLE_SECONDS),
        raw_pdf_max_chars=int(args.raw_pdf_max_chars),
        raw_pdf_timeout_seconds=int(args.raw_pdf_timeout_seconds),
        extract_workers=int(args.extract_workers or DEFAULT_EXTRACT_WORKERS),
        retry_missing_downloads=bool(args.retry_missing_downloads or DEFAULT_RETRY_MISSING_DOWNLOADS),
        direct_arxiv_pdf_fallback=not bool(args.disable_direct_arxiv_pdf_fallback),
        heartbeat_interval_seconds=int(args.heartbeat_interval_seconds or DEFAULT_HEARTBEAT_INTERVAL_SECONDS),
    )


if __name__ == "__main__":
    main()
