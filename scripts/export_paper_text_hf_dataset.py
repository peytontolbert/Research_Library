from __future__ import annotations

"""
Export one combined text blob per paper from `exports/pdfs_structured` into a
Hugging Face-ready dataset on local disk, with optional Hub push.

Important: this exporter reflects the current structured PDF shards. If those
shards were built with truncated PDF preprocessing (for example `--max-pages 3`)
then the resulting paper text is partial. The dataset card and exported rows
carry that provenance explicitly. To recover full paper text for an existing
paper set, use `--prefer-raw-pdf-text` so each paper row is re-extracted from
the source PDF instead of the truncated structured tokens.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from datasets import Dataset, DatasetDict, Features, Sequence as HFSequence, Value  # type: ignore
from huggingface_hub import HfApi, get_token  # type: ignore
from huggingface_hub.errors import HfHubHTTPError  # type: ignore
from models.shared.pdf_utils import extract_pdf_text


DEFAULT_STRUCTURED_DIR = Path("exports/pdfs_structured")
DEFAULT_METADATA_PATH = Path("/data/arxiv/arxiv-metadata-oai-snapshot.json")
DEFAULT_OUTPUT_DIR = Path("exports/huggingface/paper_text_structured_v1")
DEFAULT_INCLUDE_TYPES = ("text", "heading", "table", "reference")
DEFAULT_RAW_PDF_MAX_CHARS = 1_000_000
DEFAULT_RAW_PDF_TIMEOUT_SECONDS = 8
DEFAULT_PARQUET_BATCH_ROWS = 256
PDF_SEARCH_ROOTS = [Path("exports/arxiv_pdfs"), Path("/arxiv/pdfs")]
BACKFILL_PARQUET_GLOB = "paper_text_backfill_*.parquet"


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


def _iter_structured_rows(structured_dir: Path) -> Iterator[Dict[str, Any]]:
    if not structured_dir.exists():
        return
    for shard in sorted(structured_dir.glob("pdf_structured_*.jsonl")):
        for row in _iter_jsonl(shard):
            yield row
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


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


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
    return re.sub(r"v\d+$", "", paper_id)


def _paper_version(raw_paper_id: str) -> str:
    match = re.search(r"(v\d+)$", str(raw_paper_id or "").strip())
    return match.group(1) if match else ""


def _page_value(token: Dict[str, Any]) -> int:
    try:
        return int(token.get("page") or 0)
    except Exception:
        return 0


def _line_value(token: Dict[str, Any]) -> int:
    try:
        return int(token.get("line_no") or 0)
    except Exception:
        return 0


def _bbox_sort_key(token: Dict[str, Any]) -> Tuple[float, float]:
    bbox = token.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
        try:
            return float(bbox[1]), float(bbox[0])
        except Exception:
            return 0.0, 0.0
    return 0.0, 0.0


def _token_sort_key(token: Dict[str, Any]) -> Tuple[int, int, float, float]:
    y, x = _bbox_sort_key(token)
    return (_page_value(token), _line_value(token), y, x)


def _clean_token_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw or raw == "[IMAGE]" or raw.startswith("PDF_PATH::"):
        return ""
    return raw


def _collapse_tokens(
    tokens: Sequence[Dict[str, Any]],
    *,
    include_types: Set[str],
    dedupe_consecutive: bool,
) -> Tuple[str, Counter[str], int]:
    parts: List[str] = []
    type_counts: Counter[str] = Counter()
    last_text = ""
    last_page: Optional[int] = None
    page_count = 0

    for token in sorted(tokens, key=_token_sort_key):
        token_type = str(token.get("type") or "text").strip().lower() or "text"
        if token_type not in include_types:
            continue
        text = _clean_token_text(token.get("text"))
        if not text:
            continue
        if dedupe_consecutive and text == last_text:
            continue

        page = _page_value(token)
        if page and page != last_page:
            if parts and parts[-1] != "":
                parts.append("")
            last_page = page
            page_count += 1

        if token_type == "heading" and parts and parts[-1] != "":
            parts.append("")
        parts.append(text)
        if token_type == "heading":
            parts.append("")

        type_counts[token_type] += 1
        last_text = text

    text = "\n".join(parts)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip(), type_counts, page_count


def _collect_paper_ids(structured_dir: Path) -> Set[str]:
    ids: Set[str] = set()
    for row in _iter_structured_rows(structured_dir):
        paper_id = _canonical_paper_id(_paper_id_from_row(row))
        if paper_id:
            ids.add(paper_id)
    return ids


def _load_metadata_subset(metadata_path: Path, paper_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    if not metadata_path.is_file() or not paper_ids:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            paper_id = str(obj.get("id") or "").strip()
            if not paper_id or paper_id not in paper_ids:
                continue
            versions = obj.get("versions") or []
            if not isinstance(versions, list):
                versions = []
            out[paper_id] = {
                "title": str(obj.get("title") or "").strip(),
                "abstract": str(obj.get("abstract") or "").strip(),
                "authors": str(obj.get("authors") or "").strip(),
                "categories": str(obj.get("categories") or "").strip(),
                "license": str(obj.get("license") or "").strip(),
                "update_date": str(obj.get("update_date") or "").strip(),
                "version_count": len(versions),
            }
            if len(out) >= len(paper_ids):
                break
    return out


def _without_broken_torch():
    broken_torch = sys.modules.get("torch")
    removed = False
    if broken_torch is not None and not hasattr(broken_torch, "Tensor"):
        removed = True
        sys.modules.pop("torch", None)
    return broken_torch, removed


def _restore_torch(broken_torch: Any, removed: bool) -> None:
    if removed and broken_torch is not None:
        sys.modules["torch"] = broken_torch


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    explicit = str(token or "").strip()
    if explicit:
        return explicit
    cached = get_token()
    if isinstance(cached, str) and cached.strip():
        return cached.strip()
    return None


def _paper_dataset_features() -> Features:
    return Features(
        {
            "paper_id": Value("string"),
            "canonical_paper_id": Value("string"),
            "paper_version": Value("string"),
            "pdf_path": Value("string"),
            "title": Value("string"),
            "abstract": Value("string"),
            "authors": Value("string"),
            "categories": Value("string"),
            "license": Value("string"),
            "update_date": Value("string"),
            "version_count": Value("int64"),
            "metadata_found": Value("bool"),
            "text": Value("string"),
            "text_source": Value("string"),
            "text_is_partial": Value("bool"),
            "text_char_count": Value("int64"),
            "text_line_count": Value("int64"),
            "token_count": Value("int64"),
            "page_count": Value("int64"),
            "token_types": HFSequence(Value("string")),
            "token_type_counts_json": Value("string"),
        }
    )


def _paper_dataset_schema() -> pa.Schema:
    return _paper_dataset_features().arrow_schema


def _write_parquet_batch(
    writer: pq.ParquetWriter,
    rows: Sequence[Dict[str, Any]],
    *,
    schema: pa.Schema,
) -> None:
    if not rows:
        return
    writer.write_table(pa.Table.from_pylist(list(rows), schema=schema))


def _dataset_from_parquet(parquet_path: Path) -> Dataset:
    broken_torch, removed_torch = _without_broken_torch()
    try:
        return Dataset.from_parquet(str(parquet_path))
    finally:
        _restore_torch(broken_torch, removed_torch)


def _local_pdf_candidates(raw_paper_id: str, pdf_path: str) -> List[Path]:
    candidates: List[Path] = []
    if pdf_path:
        candidates.append(Path(pdf_path))

    norm_id = str(raw_paper_id or "").strip()
    if not norm_id:
        return candidates

    yymm = norm_id[:4]
    for root in PDF_SEARCH_ROOTS:
        if len(yymm) == 4 and yymm.isdigit():
            candidates.append(root / yymm / f"{norm_id}.pdf")
        candidates.append(root / f"{norm_id}.pdf")
    return candidates


def _resolve_existing_pdf_path(raw_paper_id: str, pdf_path: str) -> Optional[Path]:
    seen: Set[Path] = set()
    for candidate in _local_pdf_candidates(raw_paper_id, pdf_path):
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _collapse_raw_pdf_text(raw_text: str) -> Tuple[str, int, int]:
    text = str(raw_text or "").replace("\x0c", "\n\n")
    lines = [_normalize_space(line) for line in text.splitlines()]
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
    if path.suffix.lower() != ".pdf":
        return extract_pdf_text(str(path), max_chars=max_chars)
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


def _fallback_raw_pdf_text(
    raw_paper_id: str,
    pdf_path: str,
    *,
    max_chars: int,
    timeout_seconds: int,
) -> Tuple[str, str, int, int]:
    resolved_pdf = _resolve_existing_pdf_path(raw_paper_id, pdf_path)
    if resolved_pdf is None:
        return "", pdf_path, 0, 0
    raw_text = _extract_pdf_text_fast(
        resolved_pdf,
        max_chars=max_chars,
        timeout_seconds=timeout_seconds,
    )
    if not raw_text or raw_text.startswith("PDF_PATH::"):
        return "", str(resolved_pdf), 0, 0
    text, line_count, page_count = _collapse_raw_pdf_text(raw_text)
    return text, str(resolved_pdf), line_count, page_count


def _preextracted_raw_text(
    row: Dict[str, Any],
) -> Tuple[str, int, int]:
    inline_text = str(row.get("raw_text") or row.get("full_text") or "")
    raw_text_path = str(row.get("raw_text_path") or "").strip()
    text = inline_text
    if not text and raw_text_path:
        try:
            text = Path(raw_text_path).read_text(encoding="utf-8")
        except Exception:
            text = ""
    if not text:
        return "", 0, 0

    if "\x0c" in text:
        normalized_text, line_count, page_count = _collapse_raw_pdf_text(text)
    else:
        normalized_text = str(text).strip()
        line_count = len([line for line in normalized_text.splitlines() if line.strip()])
        page_count = max(1, int(row.get("raw_text_page_count") or row.get("page_count") or 1))

    try:
        row_line_count = int(row.get("raw_text_line_count") or 0)
        if row_line_count > 0:
            line_count = row_line_count
    except Exception:
        pass

    try:
        row_page_count = int(row.get("raw_text_page_count") or 0)
        if row_page_count > 0:
            page_count = row_page_count
    except Exception:
        pass

    return normalized_text, int(line_count), int(page_count)


def _counter_from_token_type_counts_json(raw_value: Any) -> Counter[str]:
    text = str(raw_value or "").strip()
    if not text:
        return Counter()
    try:
        obj = json.loads(text)
    except Exception:
        return Counter()
    if not isinstance(obj, dict):
        return Counter()
    out: Counter[str] = Counter()
    for key, value in obj.items():
        try:
            out[str(key)] += int(value)
        except Exception:
            continue
    return out


def _prebuilt_text_row(
    row: Dict[str, Any],
) -> Tuple[str, str, int, int, int, List[str], str, Counter[str]]:
    text = str(row.get("text") or "").strip()
    if not text:
        return "", "", 0, 0, 0, [], "", Counter()

    text_source = str(row.get("text_source") or "raw_pdf_preextracted").strip() or "raw_pdf_preextracted"
    try:
        page_count = int(row.get("page_count") or 0)
    except Exception:
        page_count = 0
    try:
        token_count = int(row.get("token_count") or row.get("text_line_count") or 0)
    except Exception:
        token_count = 0
    token_types_any = row.get("token_types") or []
    token_types = [str(token_type).strip() for token_type in token_types_any if str(token_type).strip()]
    emit_type_counts = _counter_from_token_type_counts_json(row.get("token_type_counts_json"))
    if not emit_type_counts and token_types and token_count > 0:
        if len(token_types) == 1:
            emit_type_counts = Counter({token_types[0]: token_count})
        else:
            emit_type_counts = Counter({token_type: 1 for token_type in token_types})
    token_type_counts_json = str(row.get("token_type_counts_json") or "").strip()
    if not token_type_counts_json:
        token_type_counts_json = json.dumps(
            dict(sorted(emit_type_counts.items())),
            ensure_ascii=True,
            separators=(",", ":"),
        )
    return (
        text,
        text_source,
        int(page_count),
        int(token_count),
        int(row.get("text_line_count") or token_count or len([ln for ln in text.splitlines() if ln.strip()])),
        token_types,
        token_type_counts_json,
        emit_type_counts,
    )


def _write_dataset_card(
    *,
    output_dir: Path,
    rows_written: int,
    metadata_covered: int,
    license_counts: Counter[str],
    text_source_counts: Counter[str],
    public_note: str,
) -> Path:
    top_licenses = license_counts.most_common(10)
    readme = f"""---
pretty_name: Structured Paper Text (combined token export)
viewer: true
tags:
- datasets
- arxiv
- scientific-papers
- text
size_categories:
- 100K<n<1M
---

# Structured Paper Text Dataset

One row per paper, built by combining the existing tokenized PDF exports under `exports/pdfs_structured`.

## Important provenance note

This dataset reflects the **current structured PDF shards**, not necessarily complete paper text.
If the underlying shards were produced with truncated PDF preprocessing, the `text` field is partial.

{public_note}

## Rows

- `train`: `{rows_written}` papers
- with matched arXiv metadata: `{metadata_covered}`
- structured-token rows: `{text_source_counts.get('combined_structured_tokens', 0)}`
- pre-extracted raw-text rows: `{text_source_counts.get('raw_pdf_preextracted', 0)}`
- preferred raw-PDF rows: `{text_source_counts.get('raw_pdf_preferred', 0)}`
- raw-PDF fallback rows: `{text_source_counts.get('raw_pdf_fallback', 0)}`

## Main columns

- `paper_id`
- `canonical_paper_id`
- `paper_version`
- `pdf_path`
- `title`
- `abstract`
- `authors`
- `categories`
- `license`
- `text`
- `text_char_count`
- `token_count`
- `page_count`
- `token_types`

## Top licenses in export

{chr(10).join(f"- `{lic or '(missing)'}`: {count}" for lic, count in top_licenses)}
"""
    path = output_dir / "README.md"
    path.write_text(readme, encoding="utf-8")
    return path


def export_paper_text_hf_dataset(
    *,
    structured_dir: str = str(DEFAULT_STRUCTURED_DIR),
    metadata_path: str = str(DEFAULT_METADATA_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    include_types: Optional[Sequence[str]] = None,
    dedupe_consecutive: bool = True,
    raw_pdf_fallback: bool = True,
    raw_pdf_max_chars: int = DEFAULT_RAW_PDF_MAX_CHARS,
    raw_pdf_timeout_seconds: int = DEFAULT_RAW_PDF_TIMEOUT_SECONDS,
    parquet_batch_rows: int = DEFAULT_PARQUET_BATCH_ROWS,
    write_jsonl: bool = False,
    write_dataset_dict: bool = False,
    prefer_raw_pdf_text: bool = False,
    license_allow: Optional[Sequence[str]] = None,
    license_deny: Optional[Sequence[str]] = None,
    require_metadata: bool = False,
    push_repo_id: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    structured_dir_path = Path(structured_dir).resolve()
    metadata_path_obj = Path(metadata_path).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    allowed_types = {
        str(token_type).strip().lower()
        for token_type in (include_types or DEFAULT_INCLUDE_TYPES)
        if str(token_type).strip()
    }
    allow_licenses = {
        str(lic).strip()
        for lic in (license_allow or [])
        if str(lic).strip()
    }
    deny_licenses = {
        str(lic).strip()
        for lic in (license_deny or [])
        if str(lic).strip()
    }

    paper_ids = _collect_paper_ids(structured_dir_path)
    metadata_by_id = _load_metadata_subset(metadata_path_obj, paper_ids)

    parquet_path = output_dir_path / "train.parquet"
    jsonl_path = output_dir_path / "papers.jsonl"
    rows_written = 0
    metadata_covered = 0
    duplicate_papers = 0
    seen_paper_ids: Set[str] = set()
    license_counts: Counter[str] = Counter()
    token_type_totals: Counter[str] = Counter()
    text_source_counts: Counter[str] = Counter()
    parquet_schema = _paper_dataset_schema()
    parquet_writer = pq.ParquetWriter(str(parquet_path), parquet_schema, compression="snappy")
    row_batch: List[Dict[str, Any]] = []

    jsonl_fh = jsonl_path.open("w", encoding="utf-8") if write_jsonl else None
    try:
        for row in _iter_structured_rows(structured_dir_path):
            raw_paper_id = _paper_id_from_row(row)
            if not raw_paper_id:
                continue
            if raw_paper_id in seen_paper_ids:
                duplicate_papers += 1
                continue
            seen_paper_ids.add(raw_paper_id)
            canonical_paper_id = _canonical_paper_id(raw_paper_id)
            paper_version = _paper_version(raw_paper_id)

            pdf_path = str(row.get("pdf_path") or "")
            tokens = row.get("tokens") or []
            if not isinstance(tokens, list):
                continue

            text = ""
            page_count = 0
            text_source = "combined_structured_tokens"
            token_count = 0
            token_types: List[str] = []
            token_type_counts_json = json.dumps({}, ensure_ascii=True, separators=(",", ":"))
            emit_type_counts: Counter[str] = Counter()

            prebuilt_text, prebuilt_source, prebuilt_page_count, prebuilt_token_count, prebuilt_line_count, prebuilt_token_types, prebuilt_counts_json, prebuilt_counts = _prebuilt_text_row(row)
            if prebuilt_text:
                text = prebuilt_text
                page_count = prebuilt_page_count
                text_source = prebuilt_source
                token_count = prebuilt_token_count
                token_types = prebuilt_token_types
                token_type_counts_json = prebuilt_counts_json
                emit_type_counts = prebuilt_counts
            else:
                preextracted_text, preextracted_line_count, preextracted_page_count = _preextracted_raw_text(row)
                if preextracted_text:
                    text = preextracted_text
                    page_count = preextracted_page_count
                    text_source = "raw_pdf_preextracted"
                    token_count = preextracted_line_count
                    token_types = ["raw_text_preextracted"]
                    token_type_counts_json = json.dumps(
                        {"raw_text_preextracted": preextracted_line_count},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    emit_type_counts = Counter({"raw_text_preextracted": preextracted_line_count})

            if not text and prefer_raw_pdf_text:
                preferred_text, preferred_pdf_path, preferred_line_count, preferred_page_count = _fallback_raw_pdf_text(
                    raw_paper_id,
                    pdf_path,
                    max_chars=int(raw_pdf_max_chars),
                    timeout_seconds=int(raw_pdf_timeout_seconds),
                )
                if preferred_text:
                    text = preferred_text
                    pdf_path = preferred_pdf_path
                    page_count = preferred_page_count
                    text_source = "raw_pdf_preferred"
                    token_count = preferred_line_count
                    token_types = ["raw_text_pdf"]
                    token_type_counts_json = json.dumps(
                        {"raw_text_pdf": preferred_line_count},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    emit_type_counts = Counter({"raw_text_pdf": preferred_line_count})

            if not text:
                text, type_counts, page_count = _collapse_tokens(
                    tokens,
                    include_types=allowed_types,
                    dedupe_consecutive=dedupe_consecutive,
                )
                text_source = "combined_structured_tokens"
                token_count = int(sum(type_counts.values()))
                token_types = sorted(type_counts.keys())
                token_type_counts_json = json.dumps(
                    dict(sorted(type_counts.items())),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                emit_type_counts = Counter(type_counts)
            if not text and raw_pdf_fallback:
                fallback_text, fallback_pdf_path, fallback_line_count, fallback_page_count = _fallback_raw_pdf_text(
                    raw_paper_id,
                    pdf_path,
                    max_chars=int(raw_pdf_max_chars),
                    timeout_seconds=int(raw_pdf_timeout_seconds),
                )
                if fallback_text:
                    text = fallback_text
                    pdf_path = fallback_pdf_path
                    page_count = fallback_page_count
                    text_source = "raw_pdf_fallback"
                    token_count = fallback_line_count
                    token_types = ["raw_text_fallback"]
                    token_type_counts_json = json.dumps(
                        {"raw_text_fallback": fallback_line_count},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    emit_type_counts = Counter({"raw_text_fallback": fallback_line_count})
            if not text:
                continue

            meta = metadata_by_id.get(canonical_paper_id) or {}
            license_url = str(meta.get("license") or "")
            if require_metadata and not meta:
                continue
            if allow_licenses and license_url not in allow_licenses:
                continue
            if deny_licenses and license_url in deny_licenses:
                continue

            if meta:
                metadata_covered += 1
            license_counts[license_url] += 1
            token_type_totals.update(emit_type_counts)
            text_source_counts[text_source] += 1

            out = {
                "paper_id": raw_paper_id,
                "canonical_paper_id": canonical_paper_id,
                "paper_version": paper_version,
                "pdf_path": pdf_path,
                "title": str(meta.get("title") or ""),
                "abstract": str(meta.get("abstract") or ""),
                "authors": str(meta.get("authors") or ""),
                "categories": str(meta.get("categories") or ""),
                "license": license_url,
                "update_date": str(meta.get("update_date") or ""),
                "version_count": int(meta.get("version_count") or 0),
                "metadata_found": bool(meta),
                "text": text,
                "text_source": text_source,
                "text_is_partial": bool(row.get("text_is_partial"))
                if prebuilt_text
                else not (
                    text_source == "raw_pdf_preextracted"
                    or (
                        text_source in {"raw_pdf_preferred", "raw_pdf_fallback"}
                        and int(raw_pdf_max_chars) <= 0
                    )
                ),
                "text_char_count": len(text),
                "text_line_count": prebuilt_line_count
                if prebuilt_text
                else len([ln for ln in text.splitlines() if ln.strip()]),
                "token_count": token_count,
                "page_count": int(page_count),
                "token_types": token_types,
                "token_type_counts_json": token_type_counts_json,
            }
            if jsonl_fh is not None:
                jsonl_fh.write(json.dumps(out, ensure_ascii=True) + "\n")
            row_batch.append(out)
            if len(row_batch) >= max(1, int(parquet_batch_rows)):
                _write_parquet_batch(parquet_writer, row_batch, schema=parquet_schema)
                row_batch.clear()
            rows_written += 1
        if row_batch:
            _write_parquet_batch(parquet_writer, row_batch, schema=parquet_schema)
    finally:
        parquet_writer.close()
        if jsonl_fh is not None:
            jsonl_fh.close()

    dataset_disk_dir = output_dir_path / "dataset_dict"
    if write_dataset_dict and dataset_disk_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing dataset_dict at {dataset_disk_dir}. "
            "Remove it first or choose a new output directory."
        )
    train_dataset: Optional[Dataset] = None
    if write_dataset_dict or push_repo_id:
        train_dataset = _dataset_from_parquet(parquet_path)
    if write_dataset_dict:
        DatasetDict({"train": train_dataset}).save_to_disk(str(dataset_disk_dir))

    public_note = (
        "Use per-paper `license` filtering before publishing broadly; many arXiv records "
        "carry `nonexclusive-distrib` rather than a general reuse license."
    )
    if prefer_raw_pdf_text and int(raw_pdf_max_chars) <= 0:
        public_note += " This export prefers raw PDF text and does not apply a character cap, so rows sourced from PDFs are full-document extracts."
    elif prefer_raw_pdf_text:
        public_note += f" This export prefers raw PDF text, but rows may still be truncated at {int(raw_pdf_max_chars)} characters."
    readme_path = _write_dataset_card(
        output_dir=output_dir_path,
        rows_written=rows_written,
        metadata_covered=metadata_covered,
        license_counts=license_counts,
        text_source_counts=text_source_counts,
        public_note=public_note,
    )

    stats = {
        "structured_dir": str(structured_dir_path),
        "metadata_path": str(metadata_path_obj),
        "rows_written": rows_written,
        "metadata_covered": metadata_covered,
        "duplicate_papers_skipped": duplicate_papers,
        "include_types": sorted(allowed_types),
        "parquet_batch_rows": int(parquet_batch_rows),
        "prefer_raw_pdf_text": bool(prefer_raw_pdf_text),
        "raw_pdf_max_chars": int(raw_pdf_max_chars),
        "jsonl_written": bool(write_jsonl),
        "dataset_dict_written": bool(write_dataset_dict),
        "token_type_totals": dict(sorted(token_type_totals.items())),
        "text_source_counts": dict(sorted(text_source_counts.items())),
        "license_counts_top10": license_counts.most_common(10),
        "jsonl_path": str(jsonl_path) if write_jsonl else "",
        "dataset_disk_dir": str(dataset_disk_dir) if write_dataset_dict else "",
        "parquet_path": str(parquet_path),
        "readme_path": str(readme_path),
    }
    (output_dir_path / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    push_info: Optional[Dict[str, Any]] = None
    if push_repo_id:
        resolved_token = _resolve_hf_token(token)
        if not resolved_token:
            raise RuntimeError(
                "Hugging Face push requested but no token was found. "
                "Set HF_TOKEN, pass --token hf_..., or login first via huggingface_hub."
            )
        api = HfApi(token=resolved_token)
        try:
            api.whoami(token=resolved_token)
            api.create_repo(
                repo_id=push_repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            if train_dataset is None:
                train_dataset = _dataset_from_parquet(parquet_path)
            train_dataset.push_to_hub(
                repo_id=push_repo_id,
                config_name="default",
                split="train",
                set_default=True,
                private=private,
                token=resolved_token,
            )
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=push_repo_id,
                repo_type="dataset",
                token=resolved_token,
            )
            api.upload_file(
                path_or_fileobj=str(output_dir_path / "stats.json"),
                path_in_repo="stats.json",
                repo_id=push_repo_id,
                repo_type="dataset",
                token=resolved_token,
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(
                "Hugging Face push failed during authentication or repo creation. "
                f"Local export is still available at {output_dir_path}. "
                f"Original error: {exc}"
            ) from exc
        push_info = {
            "repo_id": push_repo_id,
            "private": bool(private),
        }

    return {
        "stats": stats,
        "push": push_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine structured paper tokens into one text blob per paper and export a HF-ready dataset."
    )
    parser.add_argument("--structured-dir", type=str, default=str(DEFAULT_STRUCTURED_DIR))
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--include-type",
        action="append",
        dest="include_types",
        help="Token type to include. May be passed multiple times. Defaults to text/heading/table/reference.",
    )
    parser.add_argument(
        "--no-dedupe-consecutive",
        action="store_true",
        help="Do not remove immediately repeated consecutive lines.",
    )
    parser.add_argument(
        "--disable-raw-pdf-fallback",
        action="store_true",
        help="Do not fall back to raw PDF text when the structured row has no usable text.",
    )
    parser.add_argument(
        "--raw-pdf-max-chars",
        type=int,
        default=DEFAULT_RAW_PDF_MAX_CHARS,
        help="Character cap for the raw PDF fallback extractor.",
    )
    parser.add_argument(
        "--raw-pdf-timeout-seconds",
        type=int,
        default=DEFAULT_RAW_PDF_TIMEOUT_SECONDS,
        help="Timeout per raw PDF fallback extraction attempt.",
    )
    parser.add_argument(
        "--parquet-batch-rows",
        type=int,
        default=DEFAULT_PARQUET_BATCH_ROWS,
        help="Number of rows to buffer before flushing a Parquet batch.",
    )
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Also emit papers.jsonl for debugging. Disabled by default to reduce local disk usage.",
    )
    parser.add_argument(
        "--write-dataset-dict",
        action="store_true",
        help="Also save a dataset_dict directory for local datasets.load_from_disk workflows.",
    )
    parser.add_argument(
        "--prefer-raw-pdf-text",
        action="store_true",
        help="Prefer full text re-extracted from the source PDF for each paper instead of the structured-token export.",
    )
    parser.add_argument(
        "--license-allow",
        action="append",
        dest="license_allow",
        help="Optional allowlist of exact license URLs.",
    )
    parser.add_argument(
        "--license-deny",
        action="append",
        dest="license_deny",
        help="Optional denylist of exact license URLs.",
    )
    parser.add_argument(
        "--require-metadata",
        action="store_true",
        help="Only keep papers that matched an arXiv metadata record.",
    )
    parser.add_argument("--push-repo-id", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN, then cached huggingface_hub login if available.",
    )
    args = parser.parse_args()

    result = export_paper_text_hf_dataset(
        structured_dir=args.structured_dir,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        include_types=args.include_types,
        dedupe_consecutive=not bool(args.no_dedupe_consecutive),
        raw_pdf_fallback=not bool(args.disable_raw_pdf_fallback),
        raw_pdf_max_chars=int(args.raw_pdf_max_chars),
        raw_pdf_timeout_seconds=int(args.raw_pdf_timeout_seconds),
        parquet_batch_rows=int(args.parquet_batch_rows),
        write_jsonl=bool(args.write_jsonl),
        write_dataset_dict=bool(args.write_dataset_dict),
        prefer_raw_pdf_text=bool(args.prefer_raw_pdf_text),
        license_allow=args.license_allow,
        license_deny=args.license_deny,
        require_metadata=bool(args.require_metadata),
        push_repo_id=args.push_repo_id,
        private=bool(args.private),
        token=args.token,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
