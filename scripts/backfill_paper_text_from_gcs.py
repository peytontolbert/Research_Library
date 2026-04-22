from __future__ import annotations

"""
Grow the paper-text corpus by temporarily downloading missing arXiv PDFs from
GCS, extracting full text, writing parquet backfill shards, and deleting the
temporary PDFs after each batch.

Typical use:

  PYTHONPATH=. python -m scripts.backfill_paper_text_from_gcs \
    --existing-parquet-dir /arxiv/huggingface/paper_text_124k_dedup_v1 \
    --metadata-path /data/arxiv/arxiv-metadata-oai-snapshot.json \
    --out-dir /arxiv/pdfs_structured_gcs \
    --temp-pdf-dir /arxiv/tmp_arxiv_pdf_batches \
    --target-total-papers 200000

This produces `paper_text_backfill_*.parquet` shards under `--out-dir`. Those
shards can then be merged into a new deduped dataset with
`scripts.merge_paper_text_parquets`.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.request import Request, urlopen

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from scripts.backfill_missing_paper_text_shards import (
    BACKFILL_PARQUET_GLOB,
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_PROGRESS_EVERY,
    DEFAULT_RAW_PDF_MAX_CHARS,
    DEFAULT_RAW_PDF_TIMEOUT_SECONDS,
    DEFAULT_ROW_GROUP_ROWS,
    DEFAULT_SHARD_SIZE,
    _cleanup_temporary_shards,
    _backfill_schema,
    _canonical_paper_id,
    _collapse_raw_pdf_text,
    _existing_paper_ids,
    _extract_pdf_text_fast,
    _finalize_writer,
    _next_shard_index,
    _open_writer,
)


DEFAULT_METADATA_PATH = Path("/data/arxiv/arxiv-metadata-oai-snapshot.json")
DEFAULT_OUTPUT_DIR = Path("/arxiv/pdfs_structured_gcs")
DEFAULT_TEMP_PDF_DIR = Path("/arxiv/tmp_arxiv_pdf_batches")
DEFAULT_DOWNLOAD_BATCH_SIZE = 256
DEFAULT_GCS_PREFIX = "gs://arxiv-dataset/arxiv/pdf"
DEFAULT_CATEGORY_PREFIX = ""
DEFAULT_PARTITION_COUNT = 1
DEFAULT_PARTITION_INDEX = 0
DEFAULT_PROGRESS_FILENAME = "gcs_backfill_progress.json"
DEFAULT_EXTRACT_WORKERS = max(1, min(8, int(os.cpu_count() or 1)))
DEFAULT_RETRY_MISSING_DOWNLOADS = False
DEFAULT_DIRECT_ARXIV_PDF_FALLBACK = True
DEFAULT_DIRECT_ARXIV_TIMEOUT_SECONDS = 60
DIRECT_ARXIV_USER_AGENT = "repository-library-paper-backfill/1.0"


def _split_legacy_archive_id(raw_paper_id: str) -> Optional[Tuple[str, str]]:
    paper_id = str(raw_paper_id or "").strip()
    if "/" not in paper_id:
        return None
    archive, local_id = paper_id.split("/", 1)
    archive = archive.strip()
    local_id = local_id.strip()
    if not archive or len(local_id) < 4 or not local_id[:4].isdigit():
        return None
    return archive, local_id


def _extract_year(raw: Dict[str, Any]) -> Optional[int]:
    paper_id = str(raw.get("id") or "").strip()
    legacy_parts = _split_legacy_archive_id(paper_id)
    if legacy_parts is not None:
        _, local_id = legacy_parts
        yy = local_id[:2]
        if yy.isdigit():
            yy_int = int(yy)
            return 1900 + yy_int if yy_int >= 90 else 2000 + yy_int
    if len(paper_id) >= 4 and paper_id[:4].isdigit():
        return 2000 + int(paper_id[:2])

    update_date = str(raw.get("update_date") or "").strip()
    if len(update_date) >= 4 and update_date[:4].isdigit():
        return int(update_date[:4])

    versions = raw.get("versions")
    if isinstance(versions, list) and versions:
        created = str((versions[0] or {}).get("created") or "").strip()
        if len(created) >= 4 and created[:4].isdigit():
            return int(created[:4])
    return None


def _latest_version(raw: Dict[str, Any]) -> str:
    versions = raw.get("versions")
    if not isinstance(versions, list) or not versions:
        return ""
    latest = versions[-1]
    if not isinstance(latest, dict):
        return ""
    version = str(latest.get("version") or "").strip()
    return version if version.startswith("v") else ""


def _download_candidates(raw: Dict[str, Any], *, gcs_prefix: str) -> List[Dict[str, str]]:
    canonical_paper_id = str(raw.get("id") or "").strip()
    if not canonical_paper_id:
        return []
    legacy_parts = _split_legacy_archive_id(canonical_paper_id)
    is_legacy = legacy_parts is not None
    if is_legacy:
        archive_prefix, local_id = legacy_parts or ("", "")
        path_prefix = f"{str(gcs_prefix).rstrip('/').removesuffix('/pdf')}/{archive_prefix}/pdf/{local_id[:4]}"
    elif len(canonical_paper_id) >= 4 and canonical_paper_id[:4].isdigit():
        path_prefix = f"{str(gcs_prefix).rstrip('/')}/{canonical_paper_id[:4]}"
    else:
        return []
    versions = raw.get("versions")
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    if isinstance(versions, list):
        for entry in reversed(versions):
            if not isinstance(entry, dict):
                continue
            paper_version = str(entry.get("version") or "").strip()
            if paper_version and not paper_version.startswith("v"):
                continue
            paper_id = f"{canonical_paper_id}{paper_version}" if paper_version else canonical_paper_id
            if paper_id in seen:
                continue
            seen.add(paper_id)
            stem = f"{local_id}{paper_version}" if is_legacy else paper_id
            remote_name = f"{stem}.pdf"
            pdf_path = f"{path_prefix}/{remote_name}"
            out.append(
                {
                    "paper_id": paper_id,
                    "paper_version": paper_version,
                    "gcs_url": pdf_path,
                    "pdf_path": pdf_path,
                }
            )
    if out:
        return out
    stem = local_id if is_legacy else canonical_paper_id
    pdf_path = f"{path_prefix}/{stem}.pdf"
    return [
        {
            "paper_id": canonical_paper_id,
            "paper_version": "",
            "gcs_url": pdf_path,
            "pdf_path": pdf_path,
        }
    ]


def _metadata_categories_match(categories: str, category_prefix: str) -> bool:
    prefix = str(category_prefix or "").strip()
    if not prefix:
        return True
    return any(token.startswith(prefix) for token in str(categories or "").split())


def _metadata_keywords_match(raw: Dict[str, Any], keywords: Sequence[str]) -> bool:
    terms = [str(term or "").strip().lower() for term in keywords if str(term or "").strip()]
    if not terms:
        return True
    haystack = " ".join(
        [
            str(raw.get("title") or "").lower(),
            str(raw.get("abstract") or "").lower(),
        ]
    )
    return any(term in haystack for term in terms)


def _iter_metadata_candidates(
    *,
    metadata_path: Path,
    category_prefix: str,
    min_year: int,
    max_year: int,
    keywords: Sequence[str],
    gcs_prefix: str,
    start_offset: int = 0,
) -> Iterator[Dict[str, Any]]:
    with metadata_path.open("rb") as fh:
        if int(start_offset or 0) > 0:
            fh.seek(int(start_offset))
        while True:
            line_start_offset = int(fh.tell())
            raw_line = fh.readline()
            if not raw_line:
                break
            line_end_offset = int(fh.tell())
            try:
                line = raw_line.decode("utf-8")
            except Exception:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue

            canonical_paper_id = str(raw.get("id") or "").strip()
            if not canonical_paper_id:
                continue

            categories = str(raw.get("categories") or "").strip()
            if not _metadata_categories_match(categories, category_prefix):
                continue

            year = _extract_year(raw)
            if int(min_year or 0) and (year is None or year < int(min_year)):
                continue
            if int(max_year or 0) and (year is None or year > int(max_year)):
                continue

            if not _metadata_keywords_match(raw, keywords):
                continue

            download_candidates = _download_candidates(raw, gcs_prefix=gcs_prefix)
            if not download_candidates:
                continue
            selected = download_candidates[0]
            yield {
                "paper_id": selected["paper_id"],
                "canonical_paper_id": canonical_paper_id,
                "paper_version": selected["paper_version"],
                "gcs_url": selected["gcs_url"],
                "pdf_path": selected["pdf_path"],
                "download_candidates": download_candidates,
                "__metadata_start_offset": int(line_start_offset),
                "__metadata_end_offset": int(line_end_offset),
                "title": str(raw.get("title") or "").strip(),
                "abstract": str(raw.get("abstract") or "").strip(),
                "authors": str(raw.get("authors") or "").strip(),
                "categories": categories,
                "license": str(raw.get("license") or "").strip(),
                "update_date": str(raw.get("update_date") or "").strip(),
                "version_count": len(raw.get("versions") or []) if isinstance(raw.get("versions"), list) else 0,
                "metadata_found": True,
            }


def _iter_parquet_paths_from_dir(path: Path) -> Iterator[Path]:
    if not path.exists():
        return
    for parquet_path in sorted(path.glob("*.parquet")):
        if parquet_path.is_file():
            yield parquet_path


def _parquet_canonical_ids(paths: Sequence[Path]) -> Set[str]:
    out: Set[str] = set()
    for path in paths:
        try:
            parquet_file = pq.ParquetFile(str(path))
        except Exception:
            continue
        for row_group_idx in range(parquet_file.num_row_groups):
            try:
                table = parquet_file.read_row_group(row_group_idx, columns=["canonical_paper_id"])
            except Exception:
                continue
            values = table.column("canonical_paper_id").to_pylist()
            for value in values:
                canonical_paper_id = _canonical_paper_id(str(value or "").strip())
                if canonical_paper_id:
                    out.add(canonical_paper_id)
    return out


def _covered_paper_ids(
    *,
    existing_structured_dirs: Sequence[Path],
    existing_parquet_dirs: Sequence[Path],
    existing_parquet_paths: Sequence[Path],
    out_dir: Path,
) -> Set[str]:
    covered = _existing_paper_ids([*existing_structured_dirs, out_dir])
    parquet_paths: List[Path] = []
    for parquet_dir in existing_parquet_dirs:
        parquet_paths.extend(list(_iter_parquet_paths_from_dir(parquet_dir)))
    parquet_paths.extend(list(existing_parquet_paths))
    covered.update(_parquet_canonical_ids(parquet_paths))
    return covered


def _run_gsutil_cp(urls: Sequence[str], dest_dir: Path) -> int:
    if not urls:
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["gsutil", "-m", "cp", "-I", str(dest_dir)],
        input=("\n".join(urls) + "\n").encode("utf-8"),
    )
    return int(proc.returncode)


def _run_gsutil_cp_one(url: str, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["gsutil", "cp", str(url), str(dest_dir)],
    )
    return int(proc.returncode) == 0


def _direct_arxiv_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{str(paper_id or '').strip()}.pdf"


def _run_direct_arxiv_pdf_download_one(
    paper_id: str,
    *,
    dest_dir: Path,
    timeout_seconds: int = DEFAULT_DIRECT_ARXIV_TIMEOUT_SECONDS,
) -> bool:
    normalized_paper_id = str(paper_id or "").strip()
    if not normalized_paper_id:
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{Path(normalized_paper_id).name}.pdf"
    request = Request(
        _direct_arxiv_pdf_url(normalized_paper_id),
        headers={
            "User-Agent": DIRECT_ARXIV_USER_AGENT,
            "Accept": "application/pdf",
        },
    )
    try:
        with urlopen(request, timeout=int(timeout_seconds)) as resp:
            content_type = str(resp.headers.get("Content-Type") or "").lower()
            if "pdf" not in content_type and "application/octet-stream" not in content_type:
                return False
            with dest_path.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
    except Exception:
        try:
            if dest_path.exists():
                dest_path.unlink()
        except Exception:
            pass
        return False
    return dest_path.is_file() and dest_path.stat().st_size > 0


def _find_downloaded_pdf(batch_dir: Path, canonical_paper_id: str) -> Optional[Path]:
    filename_stem = Path(str(canonical_paper_id or "").strip()).name
    exact = sorted(batch_dir.rglob(f"{filename_stem}.pdf"))
    if exact:
        return exact[0]
    fuzzy = sorted(batch_dir.rglob(f"{filename_stem}*.pdf"))
    return fuzzy[0] if fuzzy else None


def _resolve_download_candidate(
    record: Dict[str, Any],
    *,
    batch_dir: Path,
    retry_missing_downloads: bool,
    direct_arxiv_pdf_fallback: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], bool, bool]:
    candidates_any = record.get("download_candidates") or []
    candidates: List[Dict[str, str]] = [
        dict(candidate)
        for candidate in candidates_any
        if isinstance(candidate, dict) and str(candidate.get("paper_id") or "").strip()
    ]
    if not candidates:
        candidates = [
            {
                "paper_id": str(record.get("paper_id") or ""),
                "paper_version": str(record.get("paper_version") or ""),
                "gcs_url": str(record.get("gcs_url") or ""),
                "pdf_path": str(record.get("pdf_path") or ""),
            }
        ]

    for idx, candidate in enumerate(candidates):
        pdf_path = _find_downloaded_pdf(batch_dir, str(candidate.get("paper_id") or ""))
        if pdf_path is not None:
            resolved = dict(record)
            resolved.update(candidate)
            return resolved, pdf_path, idx > 0, False

    if retry_missing_downloads:
        for idx, candidate in enumerate(candidates):
            gcs_url = str(candidate.get("gcs_url") or "").strip()
            if not gcs_url:
                continue
            if _run_gsutil_cp_one(gcs_url, batch_dir):
                pdf_path = _find_downloaded_pdf(batch_dir, str(candidate.get("paper_id") or ""))
                if pdf_path is not None:
                    resolved = dict(record)
                    resolved.update(candidate)
                    return resolved, pdf_path, idx > 0, False
    if direct_arxiv_pdf_fallback:
        for idx, candidate in enumerate(candidates):
            paper_id = str(candidate.get("paper_id") or "").strip()
            if not paper_id:
                continue
            if _run_direct_arxiv_pdf_download_one(paper_id, dest_dir=batch_dir):
                pdf_path = _find_downloaded_pdf(batch_dir, paper_id)
                if pdf_path is None:
                    continue
                resolved = dict(record)
                resolved.update(candidate)
                resolved["pdf_path"] = _direct_arxiv_pdf_url(paper_id)
                return resolved, pdf_path, idx > 0, True
    return None, None, False, False


def _token_type_counts_json(line_count: int) -> str:
    return json.dumps(
        {"raw_text_preextracted": int(line_count)},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _extract_record_row(
    record: Dict[str, Any],
    *,
    pdf_path: Path,
    raw_pdf_max_chars: int,
    raw_pdf_timeout_seconds: int,
) -> Optional[Dict[str, Any]]:
    raw_text = _extract_pdf_text_fast(
        pdf_path,
        max_chars=int(raw_pdf_max_chars),
        timeout_seconds=int(raw_pdf_timeout_seconds),
    )
    normalized_text, line_count, page_count = _collapse_raw_pdf_text(raw_text)
    if not normalized_text:
        return None
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
        "text": normalized_text,
        "text_source": "raw_pdf_preextracted",
        "text_is_partial": int(raw_pdf_max_chars) > 0,
        "text_char_count": len(normalized_text),
        "text_line_count": int(line_count),
        "token_count": int(line_count),
        "page_count": int(page_count),
        "token_types": ["raw_text_preextracted"],
        "token_type_counts_json": _token_type_counts_json(line_count),
    }


def _partition_matches(canonical_paper_id: str, *, partition_count: int, partition_index: int) -> bool:
    count = max(1, int(partition_count))
    index = int(partition_index)
    if index < 0 or index >= count:
        raise ValueError(f"partition_index must be in [0, {count - 1}]")
    if count == 1:
        return True
    digest = hashlib.sha1(str(canonical_paper_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % count
    return bucket == index


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def backfill_paper_text_from_gcs(
    *,
    existing_structured_dirs: Sequence[str],
    existing_parquet_dirs: Sequence[str],
    existing_parquet_paths: Sequence[str],
    metadata_path: str,
    out_dir: str,
    temp_pdf_dir: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
    row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    raw_pdf_max_chars: int = DEFAULT_RAW_PDF_MAX_CHARS,
    raw_pdf_timeout_seconds: int = DEFAULT_RAW_PDF_TIMEOUT_SECONDS,
    parquet_compression: str = DEFAULT_PARQUET_COMPRESSION,
    max_papers: int = 0,
    target_total_papers: int = 0,
    download_batch_size: int = DEFAULT_DOWNLOAD_BATCH_SIZE,
    category_prefix: str = DEFAULT_CATEGORY_PREFIX,
    min_year: int = 0,
    max_year: int = 0,
    keywords: Sequence[str] = (),
    gcs_prefix: str = DEFAULT_GCS_PREFIX,
    delete_temp_pdfs: bool = True,
    extract_workers: int = DEFAULT_EXTRACT_WORKERS,
    retry_missing_downloads: bool = DEFAULT_RETRY_MISSING_DOWNLOADS,
    direct_arxiv_pdf_fallback: bool = DEFAULT_DIRECT_ARXIV_PDF_FALLBACK,
    partition_count: int = DEFAULT_PARTITION_COUNT,
    partition_index: int = DEFAULT_PARTITION_INDEX,
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:
    existing_dirs = [Path(path).resolve() for path in existing_structured_dirs if str(path).strip()]
    existing_dataset_dirs = [Path(path).resolve() for path in existing_parquet_dirs if str(path).strip()]
    existing_dataset_paths = [Path(path).resolve() for path in existing_parquet_paths if str(path).strip()]
    metadata_path_obj = Path(metadata_path).resolve()
    out_dir_path = Path(out_dir).resolve()
    temp_pdf_dir_path = Path(temp_pdf_dir).resolve()

    if not metadata_path_obj.is_file():
        raise FileNotFoundError(f"Metadata snapshot not found: {metadata_path_obj}")

    out_dir_path.mkdir(parents=True, exist_ok=True)
    temp_pdf_dir_path.mkdir(parents=True, exist_ok=True)
    stale_temp_shards_removed = _cleanup_temporary_shards(out_dir_path)

    rows_per_file = max(1, int(shard_size))
    rows_per_group = max(1, int(row_group_rows))
    batch_size = max(1, int(download_batch_size))
    extract_workers = max(1, int(extract_workers or DEFAULT_EXTRACT_WORKERS))
    gcs_prefix = str(gcs_prefix or DEFAULT_GCS_PREFIX).rstrip("/")
    partition_count = max(1, int(partition_count))
    partition_index = int(partition_index)
    if partition_index < 0 or partition_index >= partition_count:
        raise ValueError(f"partition_index must be in [0, {partition_count - 1}]")
    progress_path_obj = (
        Path(progress_path).resolve()
        if str(progress_path or "").strip()
        else (out_dir_path / DEFAULT_PROGRESS_FILENAME)
    )

    covered_ids = _covered_paper_ids(
        existing_structured_dirs=existing_dirs,
        existing_parquet_dirs=existing_dataset_dirs,
        existing_parquet_paths=existing_dataset_paths,
        out_dir=out_dir_path,
    )
    covered_ids_before = len(covered_ids)

    requested_new_rows: Optional[int] = None
    if int(max_papers) > 0:
        requested_new_rows = int(max_papers)
    if int(target_total_papers) > 0:
        needed = max(0, int(target_total_papers) - covered_ids_before)
        requested_new_rows = needed if requested_new_rows is None else min(requested_new_rows, needed)
    if requested_new_rows is not None and requested_new_rows <= 0:
        stats = {
            "existing_structured_dirs": [str(path) for path in existing_dirs],
            "existing_parquet_dirs": [str(path) for path in existing_dataset_dirs],
            "existing_parquet_paths": [str(path) for path in existing_dataset_paths],
            "metadata_path": str(metadata_path_obj),
            "out_dir": str(out_dir_path),
            "temp_pdf_dir": str(temp_pdf_dir_path),
            "covered_ids_before": int(covered_ids_before),
            "extracted_rows": 0,
            "covered_ids_after": int(covered_ids_before),
            "target_total_papers": int(target_total_papers),
            "max_papers": int(max_papers),
            "partition_count": int(partition_count),
            "partition_index": int(partition_index),
            "status": "already_at_target",
        }
        stats_path = out_dir_path / "gcs_backfill_stats.json"
        _write_json_atomic(stats_path, stats)
        _write_json_atomic(progress_path_obj, stats)
        return {"stats": stats, "stats_path": str(stats_path)}

    shard_idx = _next_shard_index(out_dir_path)
    writer: Optional[pq.ParquetWriter] = None
    current_tmp_path: Optional[Path] = None
    current_final_path: Optional[Path] = None
    active_shard_rows = 0
    completed_shards = 0
    rows: List[Dict[str, Any]] = []

    metadata_scanned = 0
    metadata_matched = 0
    skipped_existing = 0
    skipped_empty = 0
    missing_downloads = 0
    download_batches = 0
    download_requested = 0
    downloaded_pdfs = 0
    deleted_temp_pdfs = 0
    extracted = 0
    version_fallback_uses = 0
    direct_pdf_fallback_uses = 0
    started_at = time.time()

    def snapshot(*, status: str) -> Dict[str, Any]:
        elapsed = time.time() - started_at
        remaining = None
        if requested_new_rows is not None:
            remaining = max(0, int(requested_new_rows) - int(extracted))
        return {
            "status": str(status),
            "existing_structured_dirs": [str(path) for path in existing_dirs],
            "existing_parquet_dirs": [str(path) for path in existing_dataset_dirs],
            "existing_parquet_paths": [str(path) for path in existing_dataset_paths],
            "metadata_path": str(metadata_path_obj),
            "out_dir": str(out_dir_path),
            "temp_pdf_dir": str(temp_pdf_dir_path),
            "covered_ids_before": int(covered_ids_before),
            "covered_ids_after": int(len(covered_ids)),
            "metadata_scanned": int(metadata_scanned),
            "metadata_matched": int(metadata_matched),
            "extracted_rows": int(extracted),
            "skipped_existing": int(skipped_existing),
            "skipped_empty": int(skipped_empty),
            "missing_downloads": int(missing_downloads),
            "download_batches": int(download_batches),
            "download_requested": int(download_requested),
            "downloaded_pdfs": int(downloaded_pdfs),
            "deleted_temp_pdfs": int(deleted_temp_pdfs),
            "target_total_papers": int(target_total_papers),
            "max_papers": int(max_papers),
            "remaining_target_rows": None if remaining is None else int(remaining),
            "category_prefix": str(category_prefix or ""),
            "min_year": int(min_year or 0),
            "max_year": int(max_year or 0),
            "keywords": [str(term) for term in keywords if str(term).strip()],
            "raw_pdf_max_chars": int(raw_pdf_max_chars),
            "raw_pdf_timeout_seconds": int(raw_pdf_timeout_seconds),
            "parquet_compression": str(parquet_compression),
            "rows_per_parquet_file": int(rows_per_file),
            "row_group_rows": int(rows_per_group),
            "download_batch_size": int(batch_size),
            "extract_workers": int(extract_workers),
            "retry_missing_downloads": bool(retry_missing_downloads),
            "partition_count": int(partition_count),
            "partition_index": int(partition_index),
            "version_fallback_uses": int(version_fallback_uses),
            "direct_pdf_fallback_uses": int(direct_pdf_fallback_uses),
            "open_shard_rows_buffered": int(len(rows)),
            "open_shard_rows_written": int(active_shard_rows),
            "open_temp_pdfs": int(sum(1 for _ in temp_pdf_dir_path.rglob("*.pdf"))),
            "parquet_shards_written": int(completed_shards),
            "stale_temp_shards_removed": int(stale_temp_shards_removed),
            "elapsed_seconds": elapsed,
        }

    def write_progress(status: str) -> None:
        _write_json_atomic(progress_path_obj, snapshot(status=status))

    write_progress("running")

    def flush_rows(*, close_after: bool = False) -> None:
        nonlocal rows, writer, current_tmp_path, current_final_path
        nonlocal active_shard_rows, shard_idx, completed_shards
        if rows:
            if writer is None:
                writer, current_tmp_path, current_final_path = _open_writer(
                    out_dir_path,
                    shard_idx,
                    compression=parquet_compression,
                )
            table = pa.Table.from_pylist(list(rows), schema=_backfill_schema())
            writer.write_table(table)
            active_shard_rows += len(rows)
            rows = []

        should_close = close_after or (
            writer is not None and active_shard_rows >= rows_per_file
        )
        if should_close and writer is not None:
            writer.close()
            if current_tmp_path is None or current_final_path is None:
                raise RuntimeError("Parquet shard paths were not initialized.")
            _finalize_writer(current_tmp_path, current_final_path)
            print(f"[write] shard {shard_idx} ({active_shard_rows} recs)", flush=True)
            writer = None
            current_tmp_path = None
            current_final_path = None
            active_shard_rows = 0
            shard_idx += 1
            completed_shards += 1
            write_progress("running")

    def process_batch(batch_records: List[Dict[str, Any]], batch_idx: int) -> None:
        nonlocal skipped_empty, missing_downloads, download_batches, download_requested
        nonlocal downloaded_pdfs, deleted_temp_pdfs, extracted
        nonlocal version_fallback_uses, direct_pdf_fallback_uses

        if not batch_records:
            return

        batch_dir = temp_pdf_dir_path / f"batch_{batch_idx:05d}"
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        urls = [record["gcs_url"] for record in batch_records]
        download_requested += len(urls)
        _run_gsutil_cp(urls, batch_dir)
        download_batches += 1

        batch_pdf_paths = sorted(batch_dir.rglob("*.pdf"))
        downloaded_pdfs += len(batch_pdf_paths)

        try:
            resolved_records: List[Tuple[Dict[str, Any], Path]] = []
            for record in batch_records:
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
                    (
                        resolved_record,
                        _extract_record_row(
                            resolved_record,
                            pdf_path=pdf_path,
                            raw_pdf_max_chars=int(raw_pdf_max_chars),
                            raw_pdf_timeout_seconds=int(raw_pdf_timeout_seconds),
                        ),
                    )
                    for resolved_record, pdf_path in resolved_records
                ]
            else:
                extraction_results = []
                with ThreadPoolExecutor(max_workers=extract_workers) as executor:
                    future_to_record = {
                        executor.submit(
                            _extract_record_row,
                            resolved_record,
                            pdf_path=pdf_path,
                            raw_pdf_max_chars=int(raw_pdf_max_chars),
                            raw_pdf_timeout_seconds=int(raw_pdf_timeout_seconds),
                        ): resolved_record
                        for resolved_record, pdf_path in resolved_records
                    }
                    for future in as_completed(future_to_record):
                        extraction_results.append((future_to_record[future], future.result()))

            for resolved_record, row in extraction_results:
                if row is None:
                    skipped_empty += 1
                    continue
                rows.append(row)
                covered_ids.add(str(resolved_record["canonical_paper_id"]))
                extracted += 1

                if len(rows) >= rows_per_group:
                    flush_rows()
        finally:
            if delete_temp_pdfs:
                deleted_temp_pdfs += len(list(batch_dir.rglob("*.pdf")))
                shutil.rmtree(batch_dir, ignore_errors=True)

    try:
        pending_batch: List[Dict[str, Any]] = []
        batch_idx = 0
        for record in _iter_metadata_candidates(
            metadata_path=metadata_path_obj,
            category_prefix=category_prefix,
            min_year=int(min_year),
            max_year=int(max_year),
            keywords=keywords,
            gcs_prefix=gcs_prefix,
        ):
            metadata_scanned += 1
            canonical_paper_id = record["canonical_paper_id"]
            if not _partition_matches(
                canonical_paper_id,
                partition_count=partition_count,
                partition_index=partition_index,
            ):
                continue
            if canonical_paper_id in covered_ids:
                skipped_existing += 1
                if progress_every > 0 and metadata_scanned % progress_every == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"[resume] scanned {metadata_scanned} extracted {extracted} "
                        f"skipped_existing {skipped_existing} skipped_empty {skipped_empty} "
                        f"(elapsed {elapsed:.1f}s)",
                        flush=True,
                    )
                    write_progress("running")
                continue

            metadata_matched += 1
            pending_batch.append(record)

            remaining = None
            if requested_new_rows is not None:
                remaining = max(0, requested_new_rows - extracted)
                if remaining <= 0:
                    break

            should_process = len(pending_batch) >= batch_size
            if remaining is not None and len(pending_batch) >= remaining:
                should_process = True

            if should_process:
                if remaining is not None and remaining < len(pending_batch):
                    pending_batch = pending_batch[:remaining]
                process_batch(pending_batch, batch_idx)
                pending_batch = []
                batch_idx += 1

            if progress_every > 0 and metadata_scanned % progress_every == 0:
                elapsed = time.time() - started_at
                print(
                    f"[status] scanned {metadata_scanned} extracted {extracted} "
                    f"skipped_existing {skipped_existing} skipped_empty {skipped_empty} "
                    f"(elapsed {elapsed:.1f}s)",
                    flush=True,
                )
                write_progress("running")

            if requested_new_rows is not None and extracted >= requested_new_rows:
                break

        if pending_batch and (requested_new_rows is None or extracted < requested_new_rows):
            remaining = None if requested_new_rows is None else max(0, requested_new_rows - extracted)
            if remaining is not None:
                pending_batch = pending_batch[:remaining]
            if pending_batch:
                process_batch(pending_batch, batch_idx)
    finally:
        flush_rows(close_after=True)

    stats = snapshot(status="completed")
    stats_path = out_dir_path / "gcs_backfill_stats.json"
    _write_json_atomic(stats_path, stats)
    _write_json_atomic(progress_path_obj, stats)
    return {"stats": stats, "stats_path": str(stats_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily download missing arXiv PDFs from GCS, extract full text "
            "into parquet backfill shards, then delete the temporary PDFs."
        )
    )
    parser.add_argument(
        "--existing-structured-dir",
        action="append",
        dest="existing_structured_dirs",
        default=None,
        help="Existing structured dir to skip against. May be passed multiple times.",
    )
    parser.add_argument(
        "--existing-parquet-dir",
        action="append",
        dest="existing_parquet_dirs",
        default=None,
        help="Existing dataset dir containing one or more parquet files to skip against.",
    )
    parser.add_argument(
        "--existing-parquet",
        action="append",
        dest="existing_parquet_paths",
        default=None,
        help="Existing dataset parquet to skip against. May be passed multiple times.",
    )
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--temp-pdf-dir", type=str, default=str(DEFAULT_TEMP_PDF_DIR))
    parser.add_argument("--category-prefix", type=str, default=DEFAULT_CATEGORY_PREFIX)
    parser.add_argument("--min-year", type=int, default=0)
    parser.add_argument("--max-year", type=int, default=0)
    parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        default=None,
        help="Optional keyword filter over title/abstract. May be passed multiple times.",
    )
    parser.add_argument("--gcs-prefix", type=str, default=DEFAULT_GCS_PREFIX)
    parser.add_argument(
        "--partition-count",
        type=int,
        default=DEFAULT_PARTITION_COUNT,
        help="Total number of deterministic worker partitions. Default: 1.",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=DEFAULT_PARTITION_INDEX,
        help="Zero-based worker partition index. Default: 0.",
    )
    parser.add_argument(
        "--progress-path",
        type=str,
        default="",
        help=(
            "Optional live progress JSON path. Defaults to "
            "<out-dir>/gcs_backfill_progress.json."
        ),
    )
    parser.add_argument(
        "--download-batch-size",
        type=int,
        default=DEFAULT_DOWNLOAD_BATCH_SIZE,
        help="PDFs to download per temporary GCS batch. Default: 256.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=DEFAULT_EXTRACT_WORKERS,
        help=(
            "Concurrent PDF text extraction workers per batch. Defaults to a "
            "bounded CPU-count-derived value."
        ),
    )
    parser.add_argument(
        "--retry-missing-downloads",
        action="store_true",
        help=(
            "Try slower per-paper fallback downloads when a batch misses files. "
            "Disabled by default for fastest corpus-wide passes."
        ),
    )
    parser.add_argument(
        "--disable-direct-arxiv-pdf-fallback",
        action="store_true",
        help=(
            "Disable direct https://arxiv.org/pdf fallback for objects missing from the "
            "GCS mirror."
        ),
    )
    parser.add_argument(
        "--target-total-papers",
        type=int,
        default=0,
        help=(
            "Stop once existing covered papers plus newly extracted rows reaches this "
            "count. Useful for targets like 200k or 500k."
        ),
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Optional cap on newly extracted papers for the current run.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Target number of paper rows per parquet file. Default: 100000.",
    )
    parser.add_argument(
        "--row-group-rows",
        type=int,
        default=DEFAULT_ROW_GROUP_ROWS,
        help="Rows to buffer before flushing one parquet row group. Default: 256.",
    )
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--raw-pdf-max-chars", type=int, default=DEFAULT_RAW_PDF_MAX_CHARS)
    parser.add_argument(
        "--raw-pdf-timeout-seconds",
        type=int,
        default=DEFAULT_RAW_PDF_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--parquet-compression",
        type=str,
        default=DEFAULT_PARQUET_COMPRESSION,
        help="Parquet compression codec to use for backfill shards. Default: zstd.",
    )
    parser.add_argument(
        "--keep-temp-pdfs",
        action="store_true",
        help="Keep the downloaded batch PDFs instead of deleting them after extraction.",
    )
    args = parser.parse_args()

    result = backfill_paper_text_from_gcs(
        existing_structured_dirs=args.existing_structured_dirs or [],
        existing_parquet_dirs=args.existing_parquet_dirs or [],
        existing_parquet_paths=args.existing_parquet_paths or [],
        metadata_path=args.metadata_path,
        out_dir=args.out_dir,
        temp_pdf_dir=args.temp_pdf_dir,
        shard_size=int(args.shard_size),
        row_group_rows=int(args.row_group_rows),
        progress_every=int(args.progress_every),
        raw_pdf_max_chars=int(args.raw_pdf_max_chars),
        raw_pdf_timeout_seconds=int(args.raw_pdf_timeout_seconds),
        parquet_compression=str(args.parquet_compression or DEFAULT_PARQUET_COMPRESSION),
        max_papers=int(args.max_papers),
        target_total_papers=int(args.target_total_papers),
        download_batch_size=int(args.download_batch_size),
        extract_workers=int(args.extract_workers or DEFAULT_EXTRACT_WORKERS),
        retry_missing_downloads=bool(args.retry_missing_downloads),
        direct_arxiv_pdf_fallback=not bool(args.disable_direct_arxiv_pdf_fallback),
        category_prefix=str(args.category_prefix or DEFAULT_CATEGORY_PREFIX),
        min_year=int(args.min_year or 0),
        max_year=int(args.max_year or 0),
        keywords=args.keywords or [],
        gcs_prefix=str(args.gcs_prefix or DEFAULT_GCS_PREFIX),
        delete_temp_pdfs=not bool(args.keep_temp_pdfs),
        partition_count=int(args.partition_count or DEFAULT_PARTITION_COUNT),
        partition_index=int(args.partition_index or DEFAULT_PARTITION_INDEX),
        progress_path=str(args.progress_path or ""),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
