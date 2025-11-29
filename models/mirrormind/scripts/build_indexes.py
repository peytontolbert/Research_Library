"""
Build episodic indexes from a JSONL of episodes.
Each line should match models.mirrormind.memory.Episode fields.
Outputs FAISS (if available), dense, and sparse index files.

Usage:
  python -m models.mirrormind.scripts.build_indexes --episodes path/to/episodes.jsonl --out-dir indexes/
"""

import argparse
from pathlib import Path

from models.mirrormind.memory import EpisodicMemoryStore, Episode


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="Path to episode JSONL")
    ap.add_argument("--out-dir", required=True, help="Directory to write indexes (faiss.index, dense.jsonl, sparse.jsonl)")
    ap.add_argument("--embed-model", default=None, help="Override embedding model name (env MIRRORMIND_EMBED_MODEL also supported)")
    return ap.parse_args()


def main():
    args = parse_args()
    eps_path = Path(args.episodes)
    out_dir = Path(args.out_dir)
    if args.embed_model:
        import os
        os.environ["MIRRORMIND_EMBED_MODEL"] = args.embed_model
    store = EpisodicMemoryStore.load(eps_path)
    store.build_indexes()
    # Save dense/sparse
    dense_path = out_dir / "dense.jsonl"
    sparse_path = out_dir / "sparse.jsonl"
    store.dense_index.save(dense_path) if store.dense_index else None
    store.sparse_index.save(sparse_path) if store.sparse_index else None
    if store.faiss_index:
        store.faiss_index.save(out_dir / "faiss.index")
    print(f"Built indexes to {out_dir}")


if __name__ == "__main__":
    main()
