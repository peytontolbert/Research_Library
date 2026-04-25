"""
Build a Hugging Face-ready P1 full-paper LM dataset.

The source corpus is expected to look like PeytonT/1m_papers_text or the local
paper_text parquet mirror. Output is sharded Parquet with the schema expected by
the P1 seq2seq objective:

  text   -> current full-paper chunk
  target -> next full-paper chunk

Example:
  PYTHONPATH=. python -m models.scripts.build_p1_full_paper_lm_dataset \
    --source-dataset-id PeytonT/1m_papers_text \
    --streaming \
    --out-dir /data/tmp/p1_full_paper_lm_hf \
    --chunk-chars 3000 \
    --chunk-overlap 300
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset

from models.shared.data import _chunk_text, _compose_full_paper_text
from models.shared.training import (
    _datasets_fingerprint_compatible_env,
    _normalize_paper_row,
    _paper_parquet_paths,
    _paper_row_year,
)


SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("target", pa.string()),
        ("paper_id", pa.string()),
        ("title", pa.string()),
        ("categories", pa.string()),
        ("year", pa.int32()),
        ("offset", pa.int64()),
        ("chunk_index", pa.int32()),
        ("chunk_chars", pa.int32()),
        ("chunk_overlap", pa.int32()),
        ("source_dataset", pa.string()),
    ]
)


def _load_source(args: argparse.Namespace):
    with _datasets_fingerprint_compatible_env():
        if args.source_dir:
            paths = _paper_parquet_paths(args.source_dir)
            if not paths:
                raise FileNotFoundError(f"no parquet shards found under {args.source_dir}")
            return load_dataset("parquet", data_files=[str(path) for path in paths], split=args.split, streaming=args.streaming)
        return load_dataset(args.source_dataset_id, split=args.split, streaming=args.streaming)


def _matches(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    year = _paper_row_year(row)
    if args.min_year is not None and year is not None and year < args.min_year:
        return False
    if args.max_year is not None and year is not None and year > args.max_year:
        return False
    if args.categories:
        categories = str(row.get("categories") or row.get("primary_category") or "")
        if not any(category in categories for category in args.categories):
            return False
    text = str(row.get("text") or "").strip()
    if not text:
        return False
    text_len = int(row.get("text_char_count") or len(text))
    if text_len < args.min_chars or text_len > args.max_chars:
        return False
    if args.require_full_text and bool(row.get("text_is_partial")):
        return False
    return True


def _examples_for_row(row: Dict[str, Any], args: argparse.Namespace, source_name: str) -> Iterator[Dict[str, Any]]:
    paper_id = str(row.get("canonical_paper_id") or row.get("paper_id") or row.get("id") or "").strip()
    title = str(row.get("title") or "").strip()
    categories = str(row.get("categories") or row.get("primary_category") or "").strip()
    year = _paper_row_year(row)
    chunks: List[tuple[str, int]] = []
    for chunk, offset in _chunk_text(
        _compose_full_paper_text(row),
        chunk_chars=max(512, args.chunk_chars),
        overlap=max(0, args.chunk_overlap),
    ):
        chunk = str(chunk or "").strip()
        if chunk:
            chunks.append((chunk, int(offset)))
        if args.max_chunks_per_paper and len(chunks) >= args.max_chunks_per_paper + 1:
            break

    upper = len(chunks) - 1
    if args.max_chunks_per_paper:
        upper = min(upper, args.max_chunks_per_paper)
    for idx in range(max(0, upper)):
        text, offset = chunks[idx]
        target = chunks[idx + 1][0].strip()
        if not text or not target:
            continue
        yield {
            "id": f"{paper_id}:{offset}:{idx}",
            "text": text,
            "target": target,
            "paper_id": paper_id,
            "title": title,
            "categories": categories,
            "year": int(year) if year is not None else None,
            "offset": offset,
            "chunk_index": idx,
            "chunk_chars": args.chunk_chars,
            "chunk_overlap": args.chunk_overlap,
            "source_dataset": source_name,
        }


def _write_shard(out_dir: Path, shard_idx: int, rows: List[Dict[str, Any]]) -> Path:
    path = out_dir / f"train-{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path


def _write_dataset_card(out_dir: Path, args: argparse.Namespace, stats: Dict[str, Any]) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "---",
                "configs:",
                "- config_name: default",
                "  data_files:",
                "  - split: train",
                "    path: train-*.parquet",
                "---",
                "",
                "# P1 Full-Paper LM Dataset",
                "",
                "Seq2seq next-chunk examples for the P1 full-paper language model.",
                "",
                "Columns:",
                "- `text`: current paper chunk",
                "- `target`: following paper chunk",
                "- `paper_id`, `title`, `categories`, `year`, `offset`, `chunk_index`: metadata",
                "",
                "Build stats:",
                "```json",
                json.dumps(stats, indent=2, sort_keys=True),
                "```",
                "",
                "Builder arguments:",
                "```json",
                json.dumps(vars(args), indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_name = args.source_dataset_id if not args.source_dir else str(args.source_dir)
    raw = _load_source(args)

    rows: List[Dict[str, Any]] = []
    shard_idx = 0
    written_examples = 0
    seen_papers = 0
    kept_papers = 0
    start = time.time()
    shard_paths: List[str] = []

    for raw_row in raw:
        seen_papers += 1
        row = _normalize_paper_row(raw_row)
        if not _matches(row, args):
            if args.max_papers and seen_papers >= args.max_papers:
                break
            continue
        kept_papers += 1
        for example in _examples_for_row(row, args, source_name):
            rows.append(example)
            if len(rows) >= args.shard_rows:
                shard_paths.append(str(_write_shard(out_dir, shard_idx, rows)))
                written_examples += len(rows)
                shard_idx += 1
                rows = []
                if args.max_examples and written_examples >= args.max_examples:
                    break
        if args.max_examples and written_examples >= args.max_examples:
            break
        if args.max_papers and seen_papers >= args.max_papers:
            break
        if args.progress_every and seen_papers % args.progress_every == 0:
            elapsed = time.time() - start
            print(
                f"[build] seen_papers={seen_papers:,} kept_papers={kept_papers:,} "
                f"examples={written_examples + len(rows):,} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if rows:
        shard_paths.append(str(_write_shard(out_dir, shard_idx, rows)))
        written_examples += len(rows)

    stats = {
        "seen_papers": seen_papers,
        "kept_papers": kept_papers,
        "examples": written_examples,
        "shards": len(shard_paths),
        "elapsed_seconds": round(time.time() - start, 3),
        "data_files": [Path(path).name for path in shard_paths],
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_dataset_card(out_dir, args, stats)
    return stats


def push_to_hub(out_dir: str, repo_id: str, private: bool) -> None:
    ds = Dataset.from_parquet(str(Path(out_dir) / "train-*.parquet"))
    ds.push_to_hub(repo_id, private=private)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-id", default="PeytonT/1m_papers_text")
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-chars", type=int, default=3000)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument("--max-chunks-per-paper", type=int, default=0, help="0 means all next-chunk windows")
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--min-chars", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=131072)
    parser.add_argument("--require-full-text", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--push-to-hub", default="")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(args)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    if args.push_to_hub:
        push_to_hub(args.out_dir, args.push_to_hub, args.private)
    if args.streaming:
        os._exit(0)


if __name__ == "__main__":
    main()
