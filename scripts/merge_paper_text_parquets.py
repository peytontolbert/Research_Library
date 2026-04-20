from __future__ import annotations

"""
Merge paper-text parquet datasets, deduping on canonical_paper_id.

Typical use:

  PYTHONPATH=. python -m scripts.merge_paper_text_parquets \
    --base-parquet /arxiv/huggingface/paper_text_60k_full_v1/train.parquet \
    --backfill-parquet /arxiv/pdfs_structured/paper_text_backfill_00000.parquet \
    --metadata-path /data/arxiv/arxiv-metadata-oai-snapshot.json \
    --output-dir /arxiv/huggingface/paper_text_124k_dedup_v1 \
    --rows-per-output-file 100000 \
    --compression zstd

This writes one or more `train_*.parquet` files plus `README.md` and `stats.json`.
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import duckdb  # type: ignore
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore


DEFAULT_METADATA_PATH = Path("/data/arxiv/arxiv-metadata-oai-snapshot.json")
DEFAULT_OUTPUT_DIR = Path("/arxiv/huggingface/paper_text_124k_dedup_v1")
DEFAULT_ROWS_PER_OUTPUT_FILE = 100_000
DEFAULT_COMPRESSION = "zstd"
DEFAULT_METADATA_BATCH_ROWS = 4096
DEFAULT_MEMORY_LIMIT = "8GB"


def _quote_sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_path_list(paths: Sequence[Path]) -> str:
    return "[" + ", ".join(_quote_sql_string(str(path.resolve())) for path in paths) + "]"


def _expand_parquet_inputs(paths: Sequence[str], dirs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_file():
            out.append(path)
    for raw in dirs:
        path = Path(raw).resolve()
        if not path.is_dir():
            continue
        for parquet_path in sorted(path.glob("*.parquet")):
            if parquet_path.is_file():
                out.append(parquet_path.resolve())
    deduped: List[Path] = []
    seen: Set[Path] = set()
    for path in out:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _metadata_subset_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("canonical_paper_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("abstract", pa.string()),
            pa.field("authors", pa.string()),
            pa.field("categories", pa.string()),
            pa.field("license", pa.string()),
            pa.field("update_date", pa.string()),
            pa.field("version_count", pa.int64()),
        ]
    )


def _load_metadata_map(
    *,
    metadata_path: Path,
    canonical_ids: Set[str],
) -> Dict[str, Dict[str, Any]]:
    metadata_by_id: Dict[str, Dict[str, Any]] = {}
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
            canonical_paper_id = str(obj.get("id") or "").strip()
            if not canonical_paper_id or canonical_paper_id not in canonical_ids:
                continue
            versions_any = obj.get("versions") or []
            version_count = len(versions_any) if isinstance(versions_any, list) else 0
            metadata_by_id[canonical_paper_id] = {
                "title": str(obj.get("title") or "").strip(),
                "abstract": str(obj.get("abstract") or "").strip(),
                "authors": str(obj.get("authors") or "").strip(),
                "categories": str(obj.get("categories") or "").strip(),
                "license": str(obj.get("license") or "").strip(),
                "update_date": str(obj.get("update_date") or "").strip(),
                "version_count": int(version_count),
            }
    return metadata_by_id


def _write_dataset_card(
    *,
    output_dir: Path,
    merged_rows: int,
    base_rows: int,
    backfill_rows: int,
    merged_unique_canonical_ids: int,
    output_files: Sequence[Path],
) -> Path:
    files_block = "\n".join(f"- `{path.name}`" for path in output_files)
    readme = f"""---
pretty_name: Paper Text Deduped Merge
viewer: true
tags:
- datasets
- arxiv
- scientific-papers
- text
---

# Paper Text Deduped Merge

This dataset merges an existing paper-text parquet export with one or more
backfill parquet shards, then keeps exactly one row per `canonical_paper_id`.

## Counts

- merged rows: `{merged_rows}`
- merged unique canonical papers: `{merged_unique_canonical_ids}`
- base input rows: `{base_rows}`
- backfill input rows: `{backfill_rows}`

## Files

{files_block}
"""
    path = output_dir / "README.md"
    path.write_text(readme, encoding="utf-8")
    return path


def _source_union_sql(*, base_list_sql: str, backfill_list_sql: str, slim: bool) -> str:
    if slim:
        columns = """
            source_bucket,
            paper_id,
            canonical_paper_id,
            paper_version,
            text_source,
            text_is_partial,
            text_char_count,
            page_count
        """
        base_select = f"""
            select
              'base' as source_bucket,
              paper_id,
              canonical_paper_id,
              paper_version,
              text_source,
              text_is_partial,
              text_char_count,
              page_count
            from read_parquet({base_list_sql}, union_by_name=true)
        """
        backfill_select = f"""
            select
              'backfill' as source_bucket,
              paper_id,
              canonical_paper_id,
              paper_version,
              text_source,
              text_is_partial,
              text_char_count,
              page_count
            from read_parquet({backfill_list_sql}, union_by_name=true)
        """
    else:
        columns = """
            source_bucket,
            paper_id,
            canonical_paper_id,
            paper_version,
            pdf_path,
            title,
            abstract,
            authors,
            categories,
            license,
            update_date,
            version_count,
            metadata_found,
            text,
            text_source,
            text_is_partial,
            text_char_count,
            text_line_count,
            token_count,
            page_count,
            token_types,
            token_type_counts_json
        """
        base_select = f"""
            select
              'base' as source_bucket,
              paper_id,
              canonical_paper_id,
              paper_version,
              pdf_path,
              title,
              abstract,
              authors,
              categories,
              license,
              update_date,
              version_count,
              metadata_found,
              text,
              text_source,
              text_is_partial,
              text_char_count,
              text_line_count,
              token_count,
              page_count,
              token_types,
              token_type_counts_json
            from read_parquet({base_list_sql}, union_by_name=true)
        """
        backfill_select = f"""
            select
              'backfill' as source_bucket,
              paper_id,
              canonical_paper_id,
              paper_version,
              pdf_path,
              title,
              abstract,
              authors,
              categories,
              license,
              update_date,
              version_count,
              metadata_found,
              text,
              text_source,
              text_is_partial,
              text_char_count,
              text_line_count,
              token_count,
              page_count,
              token_types,
              token_type_counts_json
            from read_parquet({backfill_list_sql}, union_by_name=true)
        """
    return f"""
        select {columns}
        from (
          {base_select}
          union all
          {backfill_select}
        )
    """


def _source_priority(text_source: str) -> int:
    if text_source == "raw_pdf_preextracted":
        return 4
    if text_source == "raw_pdf_preferred":
        return 3
    if text_source == "raw_pdf_fallback":
        return 2
    if text_source == "combined_structured_tokens":
        return 1
    return 0


def _paper_version_num(paper_version: str) -> int:
    match = re.search(r"v([0-9]+)", str(paper_version or ""))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _row_rank(row: Dict[str, Any]) -> Tuple[int, int, int, int, int, str]:
    return (
        0 if bool(row.get("text_is_partial")) else 1,
        _source_priority(str(row.get("text_source") or "")),
        _paper_version_num(str(row.get("paper_version") or "")),
        int(row.get("text_char_count") or 0),
        int(row.get("page_count") or 0),
        str(row.get("paper_id") or ""),
    )


def _enrich_row_from_metadata(row: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    enriched = dict(row)
    if not metadata:
        enriched["metadata_found"] = bool(row.get("metadata_found"))
        return enriched
    for field in ["title", "abstract", "authors", "categories", "license", "update_date"]:
        value = str(metadata.get(field) or "").strip()
        if value:
            enriched[field] = value
    version_count = int(metadata.get("version_count") or 0)
    if version_count > 0:
        enriched["version_count"] = version_count
    enriched["metadata_found"] = True
    return enriched


def _write_output_shard(
    *,
    rows: List[Dict[str, Any]],
    schema: pa.Schema,
    output_dir: Path,
    shard_idx: int,
    compression: str,
) -> Path:
    tmp_path = output_dir / f".train_{shard_idx:05d}.parquet.tmp"
    final_path = output_dir / f"train_{shard_idx:05d}.parquet"
    if tmp_path.exists():
        tmp_path.unlink()
    if final_path.exists():
        final_path.unlink()
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        str(tmp_path),
        compression=compression,
        row_group_size=min(max(1, len(rows)), DEFAULT_METADATA_BATCH_ROWS),
    )
    tmp_path.rename(final_path)
    return final_path


def _stream_winner_rows_to_parquet(
    *,
    base_paths: Sequence[Path],
    backfill_paths: Sequence[Path],
    winners: Dict[str, Tuple[str, str]],
    metadata_by_id: Dict[str, Dict[str, Any]],
    output_dir: Path,
    rows_per_output_file: int,
    compression: str,
) -> List[Path]:
    output_paths: List[Path] = []
    schema = pq.ParquetFile(str(base_paths[0])).schema_arrow
    buffer: List[Dict[str, Any]] = []
    shard_idx = 0
    written_canonical_ids: Set[str] = set()
    written_rows = 0
    current_shard_rows = 0
    writer: Optional[pq.ParquetWriter] = None
    current_tmp_path: Optional[Path] = None
    current_final_path: Optional[Path] = None

    def ensure_writer() -> None:
        nonlocal writer, current_tmp_path, current_final_path
        if writer is not None:
            return
        current_tmp_path = output_dir / f".train_{shard_idx:05d}.parquet.tmp"
        current_final_path = output_dir / f"train_{shard_idx:05d}.parquet"
        if current_tmp_path.exists():
            current_tmp_path.unlink()
        if current_final_path.exists():
            current_final_path.unlink()
        writer = pq.ParquetWriter(str(current_tmp_path), schema, compression=compression)

    def close_writer() -> None:
        nonlocal writer, current_tmp_path, current_final_path, current_shard_rows, shard_idx
        if writer is None or current_tmp_path is None or current_final_path is None:
            return
        writer.close()
        current_tmp_path.rename(current_final_path)
        output_paths.append(current_final_path)
        writer = None
        current_tmp_path = None
        current_final_path = None
        current_shard_rows = 0
        shard_idx += 1

    def flush_buffer(force_close: bool = False) -> None:
        nonlocal buffer, current_shard_rows
        while buffer:
            remaining = rows_per_output_file - current_shard_rows
            if remaining <= 0:
                close_writer()
                remaining = rows_per_output_file
            if not force_close and len(buffer) < DEFAULT_METADATA_BATCH_ROWS and len(buffer) < remaining:
                return
            take = len(buffer) if force_close else min(len(buffer), DEFAULT_METADATA_BATCH_ROWS, remaining)
            if take <= 0:
                close_writer()
                continue
            ensure_writer()
            chunk = buffer[:take]
            del buffer[:take]
            writer.write_table(pa.Table.from_pylist(chunk, schema=schema))
            current_shard_rows += len(chunk)
            if current_shard_rows >= rows_per_output_file:
                close_writer()

    for source_bucket, source_paths in [("base", base_paths), ("backfill", backfill_paths)]:
        for path in source_paths:
            parquet_file = pq.ParquetFile(str(path))
            for batch in parquet_file.iter_batches(batch_size=DEFAULT_METADATA_BATCH_ROWS):
                for row in pa.Table.from_batches([batch]).to_pylist():
                    canonical_paper_id = str(row.get("canonical_paper_id") or "").strip()
                    if not canonical_paper_id:
                        continue
                    winner = winners.get(canonical_paper_id)
                    if winner is None:
                        continue
                    winner_source_bucket, winner_paper_id = winner
                    if winner_source_bucket != source_bucket:
                        continue
                    if str(row.get("paper_id") or "").strip() != winner_paper_id:
                        continue
                    if canonical_paper_id in written_canonical_ids:
                        continue
                    metadata = metadata_by_id.get(canonical_paper_id)
                    buffer.append(_enrich_row_from_metadata(row, metadata))
                    written_canonical_ids.add(canonical_paper_id)
                    written_rows += 1
                    flush_buffer()
    flush_buffer(force_close=True)
    close_writer()

    missing = set(winners.keys()) - written_canonical_ids
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            f"Missing {len(missing)} deduped canonical papers during output write; sample={sample}"
        )
    if written_rows != len(winners):
        raise RuntimeError(
            f"Wrote {written_rows} rows but expected {len(winners)} deduped winners."
        )
    return output_paths


def merge_paper_text_parquets(
    *,
    base_parquets: Sequence[str],
    backfill_parquets: Sequence[str],
    base_parquet_dirs: Sequence[str] = (),
    backfill_parquet_dirs: Sequence[str] = (),
    metadata_path: str = str(DEFAULT_METADATA_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    rows_per_output_file: int = DEFAULT_ROWS_PER_OUTPUT_FILE,
    compression: str = DEFAULT_COMPRESSION,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    keep_temp: bool = False,
) -> Dict[str, Any]:
    base_paths = _expand_parquet_inputs(base_parquets, base_parquet_dirs)
    backfill_paths = _expand_parquet_inputs(backfill_parquets, backfill_parquet_dirs)
    if not base_paths:
        raise ValueError("At least one --base-parquet or --base-parquet-dir is required.")
    if not backfill_paths:
        raise ValueError("At least one --backfill-parquet or --backfill-parquet-dir is required.")
    for path in [*base_paths, *backfill_paths]:
        if not path.is_file():
            raise FileNotFoundError(f"Parquet input not found: {path}")

    metadata_path_obj = Path(metadata_path).resolve()
    if not metadata_path_obj.is_file():
        raise FileNotFoundError(f"Metadata snapshot not found: {metadata_path_obj}")

    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir_path / ".tmp_merge"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        spill_dir = tmp_dir / "duckdb_spill"
        spill_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"set temp_directory={_quote_sql_string(str(spill_dir))}")
        con.execute(f"set memory_limit={_quote_sql_string(str(memory_limit))}")

        base_list_sql = _sql_path_list(base_paths)
        backfill_list_sql = _sql_path_list(backfill_paths)
        slim_source_sql = _source_union_sql(
            base_list_sql=base_list_sql,
            backfill_list_sql=backfill_list_sql,
            slim=True,
        )
        con.execute(
            f"""
            create temp table candidates as
            {slim_source_sql}
            """
        )

        base_rows = int(con.execute(f"select count(*) from read_parquet({base_list_sql}, union_by_name=true)").fetchone()[0])
        backfill_rows = int(con.execute(f"select count(*) from read_parquet({backfill_list_sql}, union_by_name=true)").fetchone()[0])

        con.execute(
            """
            create temp table winners as
            with ranked as (
              select
                source_bucket,
                paper_id,
                canonical_paper_id,
                row_number() over (
                  partition by canonical_paper_id
                  order by
                    case when coalesce(text_is_partial, false) then 0 else 1 end desc,
                    case
                      when text_source = 'raw_pdf_preextracted' then 4
                      when text_source = 'raw_pdf_preferred' then 3
                      when text_source = 'raw_pdf_fallback' then 2
                      when text_source = 'combined_structured_tokens' then 1
                      else 0
                    end desc,
                    coalesce(try_cast(regexp_extract(paper_version, 'v([0-9]+)', 1) as integer), 0) desc,
                    coalesce(text_char_count, 0) desc,
                    coalesce(page_count, 0) desc,
                    paper_id desc
                ) as rn
              from candidates
              where coalesce(canonical_paper_id, '') <> ''
            )
            select
              source_bucket,
              paper_id,
              canonical_paper_id
            from ranked
            where rn = 1
            """
        )

        canonical_ids = {
            str(row[0])
            for row in con.execute("select canonical_paper_id from winners").fetchall()
            if row and str(row[0] or "").strip()
        }
        metadata_by_id = _load_metadata_map(
            metadata_path=metadata_path_obj,
            canonical_ids=canonical_ids,
        )
        metadata_rows = len(metadata_by_id)

        merged_rows = int(con.execute("select count(*) from winners").fetchone()[0])
        merged_unique_canonical_ids = int(
            con.execute("select count(distinct canonical_paper_id) from winners").fetchone()[0]
        )
        rows_per_file = max(1, int(rows_per_output_file))
        winners: Dict[str, Tuple[str, str]] = {
            str(canonical_paper_id): (str(source_bucket), str(paper_id))
            for source_bucket, paper_id, canonical_paper_id in con.execute(
                "select source_bucket, paper_id, canonical_paper_id from winners"
            ).fetchall()
            if str(canonical_paper_id or "").strip()
        }
        output_paths = _stream_winner_rows_to_parquet(
            base_paths=base_paths,
            backfill_paths=backfill_paths,
            winners=winners,
            metadata_by_id=metadata_by_id,
            output_dir=output_dir_path,
            rows_per_output_file=rows_per_file,
            compression=compression,
        )

        readme_path = _write_dataset_card(
            output_dir=output_dir_path,
            merged_rows=merged_rows,
            base_rows=base_rows,
            backfill_rows=backfill_rows,
            merged_unique_canonical_ids=merged_unique_canonical_ids,
            output_files=output_paths,
        )

        stats = {
            "base_parquets": [str(path) for path in base_paths],
            "backfill_parquets": [str(path) for path in backfill_paths],
            "metadata_path": str(metadata_path_obj),
            "output_dir": str(output_dir_path),
            "base_rows": base_rows,
            "backfill_rows": backfill_rows,
            "merged_rows": merged_rows,
            "merged_unique_canonical_ids": merged_unique_canonical_ids,
            "metadata_subset_rows": metadata_rows,
            "rows_per_output_file": rows_per_file,
            "memory_limit": str(memory_limit),
            "output_files": [str(path) for path in output_paths],
            "readme_path": str(readme_path),
        }
        stats_path = output_dir_path / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    finally:
        con.close()
        if not keep_temp and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    return {
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge paper-text parquet datasets and dedupe by canonical_paper_id."
    )
    parser.add_argument(
        "--base-parquet",
        action="append",
        dest="base_parquets",
        default=None,
        help="Base parquet input. May be passed multiple times.",
    )
    parser.add_argument(
        "--base-parquet-dir",
        action="append",
        dest="base_parquet_dirs",
        default=None,
        help="Directory containing base parquet inputs. May be passed multiple times.",
    )
    parser.add_argument(
        "--backfill-parquet",
        action="append",
        dest="backfill_parquets",
        default=None,
        help="Backfill parquet input. May be passed multiple times.",
    )
    parser.add_argument(
        "--backfill-parquet-dir",
        action="append",
        dest="backfill_parquet_dirs",
        default=None,
        help="Directory containing backfill parquet inputs. May be passed multiple times.",
    )
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--rows-per-output-file",
        type=int,
        default=DEFAULT_ROWS_PER_OUTPUT_FILE,
        help="Target number of rows per output parquet file. Default: 100000.",
    )
    parser.add_argument("--compression", type=str, default=DEFAULT_COMPRESSION)
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=DEFAULT_MEMORY_LIMIT,
        help="DuckDB memory limit before spilling to disk. Default: 8GB.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    result = merge_paper_text_parquets(
        base_parquets=args.base_parquets or [],
        backfill_parquets=args.backfill_parquets or [],
        base_parquet_dirs=args.base_parquet_dirs or [],
        backfill_parquet_dirs=args.backfill_parquet_dirs or [],
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        rows_per_output_file=int(args.rows_per_output_file),
        compression=str(args.compression or DEFAULT_COMPRESSION),
        memory_limit=str(args.memory_limit or DEFAULT_MEMORY_LIMIT),
        keep_temp=bool(args.keep_temp),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
