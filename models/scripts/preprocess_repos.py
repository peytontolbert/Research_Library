"""
Lightweight repo preprocessing script.

Scans /data/repositories, chunks code files, and emits JSONL with code/context
for downstream models (R1–R6, C2, C6) under exports/repos_chunks/.

Usage:
  PYTHONPATH=.. python -m models.scripts.preprocess_repos --max-files 5000 --extensions .py .md .txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

from models.shared.data import _chunk_text


def iter_repo_files(root: Path, extensions: Tuple[str, ...], max_files: int):
    files: List[Path] = []
    for ext in extensions:
        for p in root.rglob(f"*{ext}"):
            try:
                size = p.stat().st_size
            except Exception:
                continue
            if size == 0 or size > 1_000_000:
                continue
            files.append(p)
            if len(files) >= max_files:
                return files
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default="/data/repositories")
    ap.add_argument("--extensions", nargs="+", default=[".py", ".md", ".txt"])
    ap.add_argument("--max-files", type=int, default=5000)
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default="exports/repos_chunks")
    args = ap.parse_args()

    root = Path(args.repo_root)
    paths = iter_repo_files(root, tuple(args.extensions), args.max_files)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size = 1000
    shard = []
    shard_idx = 0
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            try:
                text = p.read_bytes().decode("latin-1", errors="ignore")
            except Exception:
                text = ""
        for chunk, offset in _chunk_text(text, chunk_chars=args.chunk_chars, overlap=args.chunk_overlap):
            shard.append({"path": str(p), "offset": offset, "code": chunk})
            if len(shard) >= shard_size:
                out_path = out_dir / f"repo_chunks_{shard_idx:05d}.jsonl"
                with out_path.open("w", encoding="utf-8") as f:
                    for rec in shard:
                        f.write(json.dumps(rec) + "\n")
                shard = []
                shard_idx += 1
    if shard:
        out_path = out_dir / f"repo_chunks_{shard_idx:05d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in shard:
                f.write(json.dumps(rec) + "\n")

    print(f"[done] wrote shards to {out_dir}")


if __name__ == "__main__":
    main()
