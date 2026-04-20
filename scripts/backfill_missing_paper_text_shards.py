from __future__ import annotations

"""
Backfill full-paper text rows for local arXiv PDFs missing from an existing
structured shard corpus, storing the result as compressed parquet shards.

Design:
- Skip papers already represented in one or more existing structured dirs.
- Extract full text from remaining local PDFs under `/arxiv/pdfs`.
- Persist rows into large parquet files (default: 100k papers per file).
- Flush small parquet row groups incrementally (default: 256 rows) so work is
  durable during long runs without buffering a whole file in RAM.

The resulting parquet rows are directly consumable by
`scripts.export_paper_text_hf_dataset`, which can merge them with legacy
`pdf_structured_*.jsonl` rows without reopening the source PDFs.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore


DEFAULT_EXISTING_STRUCTURED_DIRS = [Path("exports/pdfs_structured")]
DEFAULT_OUTPUT_DIR = Path("/arxiv/pdfs_structured")
DEFAULT_PDF_ROOT = Path("/arxiv/pdfs")
DEFAULT_SHARD_SIZE = 100_000
DEFAULT_ROW_GROUP_ROWS = 256
DEFAULT_PROGRESS_EVERY = 250
DEFAULT_RAW_PDF_MAX_CHARS = 0
DEFAULT_RAW_PDF_TIMEOUT_SECONDS = 20
DEFAULT_PARQUET_COMPRESSION = "zstd"
BACKFILL_PARQUET_GLOB = "paper_text_backfill_*.parquet"
BACKFILL_PARQUET_TMP_GLOB = ".paper_text_backfill_*.parquet.tmp"


def _backfill_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("paper_id", pa.string()),
            pa.field("canonical_paper_id", pa.string()),
            pa.field("paper_version", pa.string()),
            pa.field("pdf_path", pa.string()),
            pa.field("title", pa.string()),
            pa.field("abstract", pa.string()),
            pa.field("authors", pa.string()),
            pa.field("categories", pa.string()),
            pa.field("license", pa.string()),
            pa.field("update_date", pa.string()),
            pa.field("version_count", pa.int64()),
            pa.field("metadata_found", pa.bool_()),
            pa.field("text", pa.string()),
            pa.field("text_source", pa.string()),
            pa.field("text_is_partial", pa.bool_()),
            pa.field("text_char_count", pa.int64()),
            pa.field("text_line_count", pa.int64()),
            pa.field("token_count", pa.int64()),
            pa.field("page_count", pa.int64()),
            pa.field("token_types", pa.list_(pa.string())),
            pa.field("token_type_counts_json", pa.string()),
        ]
    )


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _iter_legacy_rows(structured_dir: Path) -> Iterator[Dict[str, Any]]:
    if not structured_dir.exists():
        return
    for shard in sorted(structured_dir.glob("pdf_structured_*.jsonl")):
        yield from _iter_jsonl(shard)


def _iter_backfill_parquet_rows(structured_dir: Path) -> Iterator[Dict[str, Any]]:
    if not structured_dir.exists():
        return
    for shard in sorted(structured_dir.glob(BACKFILL_PARQUET_GLOB)):
        try:
            parquet_file = pq.ParquetFile(str(shard))
        except Exception:
            continue
        for row_group_idx in range(parquet_file.num_row_groups):
            try:
                table = parquet_file.read_row_group(row_group_idx)
            except Exception:
                continue
            for row in table.to_pylist():
                if isinstance(row, dict):
                    yield row


def _paper_id_from_row(row: Dict[str, Any]) -> str:
    explicit = str(row.get("paper_id") or "").strip()
    if explicit:
        return explicit
    pdf_path = str(row.get("pdf_path") or "").strip()
    if not pdf_path:
        return ""
    return Path(pdf_path).stem


def _canonical_paper_id(raw_paper_id: str) -> str:
    paper_id = str(raw_paper_id or "").strip()
    if not paper_id:
        return ""
    if "v" in paper_id:
        prefix, suffix = paper_id.rsplit("v", 1)
        if suffix.isdigit():
            return prefix
    return paper_id


def _paper_version(raw_paper_id: str) -> str:
    paper_id = str(raw_paper_id or "").strip()
    if "v" in paper_id:
        prefix, suffix = paper_id.rsplit("v", 1)
        if prefix and suffix.isdigit():
            return f"v{suffix}"
    return ""


def _existing_paper_ids(structured_dirs: Sequence[Path]) -> Set[str]:
    paper_ids: Set[str] = set()
    for structured_dir in structured_dirs:
        for row in _iter_legacy_rows(structured_dir):
            paper_id = _canonical_paper_id(_paper_id_from_row(row))
            if paper_id:
                paper_ids.add(paper_id)
        for row in _iter_backfill_parquet_rows(structured_dir):
            paper_id = _canonical_paper_id(_paper_id_from_row(row))
            if paper_id:
                paper_ids.add(paper_id)
    return paper_ids


def _iter_local_pdfs(pdf_root: Path) -> Iterator[Path]:
    if not pdf_root.exists():
        return
    for pdf_path in sorted(pdf_root.rglob("*.pdf")):
        if pdf_path.is_file():
            yield pdf_path


def _collapse_raw_pdf_text(raw_text: str) -> Tuple[str, int, int]:
    text = str(raw_text or "").replace("\x0c", "\n\n")
    lines = [" ".join(str(line or "").split()) for line in text.splitlines()]
    cleaned_lines: List[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if cleaned_lines and blank_run <= 1:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(line)
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    text_out = "\n".join(cleaned_lines).strip()
    line_count = len([line for line in cleaned_lines if line])
    page_count = max(1, raw_text.count("\x0c") + 1) if text_out else 0
    return text_out, line_count, page_count


def _extract_pdf_text_fast(path: Path, *, max_chars: int, timeout_seconds: int) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", "-q", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
    except Exception:
        return ""
    text = proc.stdout if int(max_chars) <= 0 else proc.stdout[: int(max_chars)]
    if not any(ch.isalnum() for ch in text):
        return ""
    return text


def _next_shard_index(out_dir: Path) -> int:
    max_idx = -1
    for shard in out_dir.glob(BACKFILL_PARQUET_GLOB):
        stem = shard.stem
        try:
            idx = int(stem.rsplit("_", 1)[-1])
        except Exception:
            continue
        max_idx = max(max_idx, idx)
    return max_idx + 1


def _shard_path(out_dir: Path, shard_idx: int) -> Path:
    return out_dir / f"paper_text_backfill_{shard_idx:05d}.parquet"


def _shard_tmp_path(out_dir: Path, shard_idx: int) -> Path:
    return out_dir / f".paper_text_backfill_{shard_idx:05d}.parquet.tmp"


def _cleanup_temporary_shards(out_dir: Path) -> int:
    removed = 0
    for tmp_path in sorted(out_dir.glob(BACKFILL_PARQUET_TMP_GLOB)):
        try:
            tmp_path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def _open_writer(
    out_dir: Path,
    shard_idx: int,
    *,
    compression: str,
) -> Tuple[pq.ParquetWriter, Path, Path]:
    tmp_path = _shard_tmp_path(out_dir, shard_idx)
    final_path = _shard_path(out_dir, shard_idx)
    if tmp_path.exists():
        tmp_path.unlink()
    return (
        pq.ParquetWriter(
            str(tmp_path),
            _backfill_schema(),
            compression=compression,
        ),
        tmp_path,
        final_path,
    )


def _finalize_writer(tmp_path: Path, final_path: Path) -> None:
    if final_path.exists():
        final_path.unlink()
    tmp_path.replace(final_path)


def backfill_missing_paper_text_shards(
    *,
    existing_structured_dirs: Sequence[str],
    out_dir: str,
    pdf_root: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
    row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    raw_pdf_max_chars: int = DEFAULT_RAW_PDF_MAX_CHARS,
    raw_pdf_timeout_seconds: int = DEFAULT_RAW_PDF_TIMEOUT_SECONDS,
    parquet_compression: str = DEFAULT_PARQUET_COMPRESSION,
    max_papers: int = 0,
) -> Dict[str, Any]:
    existing_dirs = [Path(path).resolve() for path in existing_structured_dirs if str(path).strip()]
    out_dir_path = Path(out_dir).resolve()
    pdf_root_path = Path(pdf_root).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    stale_temp_shards_removed = _cleanup_temporary_shards(out_dir_path)

    rows_per_file = max(1, int(shard_size))
    rows_per_group = max(1, int(row_group_rows))

    seen_paper_ids = _existing_paper_ids([*existing_dirs, out_dir_path])
    shard_idx = _next_shard_index(out_dir_path)
    rows: List[Dict[str, Any]] = []
    writer: Optional[pq.ParquetWriter] = None
    current_tmp_path: Optional[Path] = None
    current_final_path: Optional[Path] = None
    active_shard_rows = 0
    completed_shards = 0

    scanned = 0
    extracted = 0
    skipped_existing = 0
    skipped_empty = 0
    started_at = time.time()

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

    try:
        for pdf_path in _iter_local_pdfs(pdf_root_path):
            scanned += 1
            raw_paper_id = pdf_path.stem
            canonical_paper_id = _canonical_paper_id(raw_paper_id)
            if canonical_paper_id in seen_paper_ids:
                skipped_existing += 1
                if progress_every > 0 and scanned % progress_every == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"[resume] scanned {scanned} extracted {extracted} "
                        f"skipped_existing {skipped_existing} skipped_empty {skipped_empty} "
                        f"(elapsed {elapsed:.1f}s)",
                        flush=True,
                    )
                continue

            raw_text = _extract_pdf_text_fast(
                pdf_path,
                max_chars=int(raw_pdf_max_chars),
                timeout_seconds=int(raw_pdf_timeout_seconds),
            )
            normalized_text, line_count, page_count = _collapse_raw_pdf_text(raw_text)
            if not normalized_text:
                skipped_empty += 1
                continue

            token_type_counts_json = json.dumps(
                {"raw_text_preextracted": int(line_count)},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            rows.append(
                {
                    "paper_id": raw_paper_id,
                    "canonical_paper_id": canonical_paper_id,
                    "paper_version": _paper_version(raw_paper_id),
                    "pdf_path": str(pdf_path),
                    "title": "",
                    "abstract": "",
                    "authors": "",
                    "categories": "",
                    "license": "",
                    "update_date": "",
                    "version_count": 0,
                    "metadata_found": False,
                    "text": normalized_text,
                    "text_source": "raw_pdf_preextracted",
                    "text_is_partial": int(raw_pdf_max_chars) > 0,
                    "text_char_count": len(normalized_text),
                    "text_line_count": int(line_count),
                    "token_count": int(line_count),
                    "page_count": int(page_count),
                    "token_types": ["raw_text_preextracted"],
                    "token_type_counts_json": token_type_counts_json,
                }
            )
            seen_paper_ids.add(canonical_paper_id)
            extracted += 1

            if len(rows) >= rows_per_group:
                flush_rows()

            if progress_every > 0 and scanned % progress_every == 0:
                elapsed = time.time() - started_at
                print(
                    f"[status] scanned {scanned} extracted {extracted} "
                    f"skipped_existing {skipped_existing} skipped_empty {skipped_empty} "
                    f"(elapsed {elapsed:.1f}s)",
                    flush=True,
                )
            if max_papers > 0 and extracted >= int(max_papers):
                break
    finally:
        flush_rows(close_after=True)

    elapsed = time.time() - started_at
    stats = {
        "existing_structured_dirs": [str(path) for path in existing_dirs],
        "out_dir": str(out_dir_path),
        "pdf_root": str(pdf_root_path),
        "scanned_pdfs": int(scanned),
        "extracted_rows": int(extracted),
        "skipped_existing": int(skipped_existing),
        "skipped_empty": int(skipped_empty),
        "raw_pdf_max_chars": int(raw_pdf_max_chars),
        "raw_pdf_timeout_seconds": int(raw_pdf_timeout_seconds),
        "parquet_compression": str(parquet_compression),
        "rows_per_parquet_file": int(rows_per_file),
        "row_group_rows": int(rows_per_group),
        "parquet_shards_written": int(completed_shards),
        "stale_temp_shards_removed": int(stale_temp_shards_removed),
        "elapsed_seconds": elapsed,
    }
    stats_path = out_dir_path / "backfill_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {"stats": stats, "stats_path": str(stats_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract full text for local arXiv PDFs missing from an existing "
            "structured shard corpus and write compressed parquet shards."
        )
    )
    parser.add_argument(
        "--existing-structured-dir",
        action="append",
        dest="existing_structured_dirs",
        default=None,
        help=(
            "Existing structured shard dir to skip against. May be passed multiple "
            "times. Defaults to exports/pdfs_structured."
        ),
    )
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pdf-root", type=str, default=str(DEFAULT_PDF_ROOT))
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
        "--max-papers",
        type=int,
        default=0,
        help="Optional cap on newly extracted papers for sampling or staged runs.",
    )
    args = parser.parse_args()

    result = backfill_missing_paper_text_shards(
        existing_structured_dirs=args.existing_structured_dirs or [
            str(path) for path in DEFAULT_EXISTING_STRUCTURED_DIRS
        ],
        out_dir=args.out_dir,
        pdf_root=args.pdf_root,
        shard_size=int(args.shard_size),
        row_group_rows=int(args.row_group_rows),
        progress_every=int(args.progress_every),
        raw_pdf_max_chars=int(args.raw_pdf_max_chars),
        raw_pdf_timeout_seconds=int(args.raw_pdf_timeout_seconds),
        parquet_compression=str(args.parquet_compression or DEFAULT_PARQUET_COMPRESSION),
        max_papers=int(args.max_papers),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
