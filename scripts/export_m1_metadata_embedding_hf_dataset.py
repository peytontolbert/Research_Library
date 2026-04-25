from __future__ import annotations

"""
Export the M1 metadata-embedding contrastive training pairs as a Hugging Face
dataset folder.

The M1 trainer builds pairs from the 1M paper-text dataset in streaming mode:

- positive: current paper title/abstract query -> current metadata card
- negative: current paper title/abstract query -> previous paper metadata card

This script mirrors that streaming construction and writes parquet shards
incrementally so it does not materialize the full pair dataset in memory or in
the Hugging Face Arrow cache.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from datasets import load_dataset  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.shared.data import (
    _metadata_embedding_doc,
    _metadata_embedding_query,
    _normalize_paper_row,
    _paper_parquet_paths,
    _paper_row_year,
)


DEFAULT_DATASET_DIR = Path("/arxiv/huggingface/paper_text_1m_dedup_v1")
DEFAULT_DATASET_ID = "PeytonT/1m_papers_text"
DEFAULT_OUTPUT_DIR = Path("exports/huggingface/m1_metadata_embedding_pairs_v1")
DEFAULT_SHARD_ROWS = 100_000
DEFAULT_SHUFFLE_BUFFER = 10_000
DEFAULT_SEED = 42
DEFAULT_YEAR_MIN = 2000
DEFAULT_YEAR_MAX = 2025


def _load_stream(
    *,
    dataset_dir: Path,
    dataset_id: str,
    split: str,
    cache_dir: Optional[str],
    shuffle_buffer: int,
    seed: int,
) -> Iterable[Dict[str, Any]]:
    parquet_paths = _paper_parquet_paths(dataset_dir)
    if parquet_paths:
        raw = load_dataset(
            "parquet",
            data_files=[str(path) for path in parquet_paths],
            split="train",
            streaming=True,
        )
    else:
        kwargs: Dict[str, Any] = {"split": split, "streaming": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        raw = load_dataset(dataset_id, **kwargs)

    if shuffle_buffer > 0 and hasattr(raw, "shuffle"):
        raw = raw.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return raw


def _matches_m1_filters(row: Dict[str, Any], *, year_min: int, year_max: int) -> bool:
    year = _paper_row_year(row)
    if year is not None and not (year_min <= int(year) <= year_max):
        return False
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    return bool(title or abstract)


def _iter_m1_pairs(
    rows: Iterable[Dict[str, Any]],
    *,
    year_min: int,
    year_max: int,
    max_pairs: int,
) -> Iterator[Dict[str, Any]]:
    previous_doc = ""
    previous_paper_id = ""
    pair_index = 0

    for raw_row in rows:
        row = _normalize_paper_row(raw_row)
        if not _matches_m1_filters(row, year_min=year_min, year_max=year_max):
            continue

        query = _metadata_embedding_query(row)
        doc = _metadata_embedding_doc(row)
        query_paper_id = str(row.get("canonical_paper_id") or row.get("paper_id") or "").strip()
        if not query or not doc:
            continue

        current_doc = f"METADATA_CARD:\n{doc}"
        yield {
            "pair_id": f"{pair_index:012d}",
            "text_a": query,
            "text_b": current_doc,
            "label": 1,
            "paper_id": query_paper_id,
            "query_paper_id": query_paper_id,
            "doc_paper_id": query_paper_id,
            "source": "m1_streaming_metadata_positive",
        }
        pair_index += 1
        if max_pairs > 0 and pair_index >= max_pairs:
            return

        if previous_doc and previous_doc != doc:
            yield {
                "pair_id": f"{pair_index:012d}",
                "text_a": query,
                "text_b": f"METADATA_CARD:\n{previous_doc}",
                "label": 0,
                "paper_id": previous_paper_id,
                "query_paper_id": query_paper_id,
                "doc_paper_id": previous_paper_id,
                "source": "m1_streaming_metadata_previous_negative",
            }
            pair_index += 1
            if max_pairs > 0 and pair_index >= max_pairs:
                return

        previous_doc = doc
        previous_paper_id = query_paper_id


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("pair_id", pa.string()),
            ("text_a", pa.string()),
            ("text_b", pa.string()),
            ("label", pa.int8()),
            ("paper_id", pa.string()),
            ("query_paper_id", pa.string()),
            ("doc_paper_id", pa.string()),
            ("source", pa.string()),
        ]
    )


def _write_sharded_parquet(
    pairs: Iterable[Dict[str, Any]],
    output_dir: Path,
    *,
    shard_rows: int,
) -> Dict[str, Any]:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    schema = _schema()
    rows: List[Dict[str, Any]] = []
    shard_index = 0
    total_rows = 0
    label_counts = {0: 0, 1: 0}

    def flush() -> None:
        nonlocal rows, shard_index, total_rows
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        path = data_dir / f"train-{shard_index:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        total_rows += len(rows)
        shard_index += 1
        rows = []

    for pair in pairs:
        label = int(pair.get("label") or 0)
        label_counts[label] = label_counts.get(label, 0) + 1
        rows.append(pair)
        if len(rows) >= shard_rows:
            flush()
    flush()

    return {
        "rows": total_rows,
        "shards": shard_index,
        "label_counts": label_counts,
    }


def _write_dataset_card(output_dir: Path, *, stats: Dict[str, Any], args: argparse.Namespace) -> None:
    card = f"""---
dataset_info:
  features:
  - name: pair_id
    dtype: string
  - name: text_a
    dtype: string
  - name: text_b
    dtype: string
  - name: label
    dtype: int8
  - name: paper_id
    dtype: string
  - name: query_paper_id
    dtype: string
  - name: doc_paper_id
    dtype: string
  - name: source
    dtype: string
  splits:
  - name: train
    num_examples: {stats["rows"]}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
tags:
- scientific-papers
- arxiv
- retrieval
- embeddings
- contrastive-learning
- metadata
- research-library
---

# M1 Metadata Embedding Pairs

This dataset contains the contrastive metadata-retrieval pairs used to train the
Research Library `M1` paper metadata embedding adapter.

It was derived from `{args.dataset_id}` or the local mirror at
`{args.dataset_dir}` using the same streaming construction as the M1 trainer.

## Fields

- `text_a`: paper query text, usually title plus abstract.
- `text_b`: metadata card prefixed with `METADATA_CARD:`.
- `label`: `1` for matching paper metadata, `0` for previous-paper negative.
- `paper_id`: document-side paper id, matching the field emitted during M1 training.
- `query_paper_id`: paper id used to build `text_a`.
- `doc_paper_id`: paper id used to build `text_b`.
- `source`: pair construction source.

## Construction

- Source dataset: `{args.dataset_id}`
- Local source path: `{args.dataset_dir}`
- Streaming shuffle seed: `{args.seed}`
- Streaming shuffle buffer: `{args.shuffle_buffer}`
- Year filter: `{args.year_min}` to `{args.year_max}`
- Max pairs: `{args.max_pairs if args.max_pairs > 0 else "unlimited"}`
- Output rows: `{stats["rows"]}`
- Output shards: `{stats["shards"]}`
- Label counts: `{json.dumps(stats["label_counts"], sort_keys=True)}`

## Intended Use

Use this dataset to reproduce or inspect the M1 contrastive retrieval objective:
encode `text_a`, encode `text_b`, and train matching pairs to score higher than
negative pairs.

This dataset is for retrieval/embedding training and evaluation, not for
generative language modeling.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def _write_metadata(output_dir: Path, *, stats: Dict[str, Any], args: argparse.Namespace) -> None:
    metadata = {
        "dataset_id": args.dataset_id,
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "cache_dir": args.cache_dir,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
        "year_min": args.year_min,
        "year_max": args.year_max,
        "max_pairs": args.max_pairs,
        "shard_rows": args.shard_rows,
        "stats": stats,
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default="/data/checkpoints")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN)
    parser.add_argument("--year-max", type=int, default=DEFAULT_YEAR_MAX)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means stream all available pairs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output already exists: {args.output_dir} (use --overwrite)")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_stream(
        dataset_dir=args.dataset_dir,
        dataset_id=args.dataset_id,
        split=args.split,
        cache_dir=args.cache_dir,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )
    pairs = _iter_m1_pairs(
        rows,
        year_min=args.year_min,
        year_max=args.year_max,
        max_pairs=args.max_pairs,
    )
    stats = _write_sharded_parquet(pairs, args.output_dir, shard_rows=max(1, args.shard_rows))
    _write_dataset_card(args.output_dir, stats=stats, args=args)
    _write_metadata(args.output_dir, stats=stats, args=args)
    print(json.dumps({"output_dir": str(args.output_dir), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
