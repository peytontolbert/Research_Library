"""
Extract paper concepts from an arXiv manifest into models/exports/paper_concepts.jsonl.
Heuristic: use category tags + title keywords as concept IDs/names.

Usage:
  python -m models.mirrormind.scripts.extract_paper_concepts --limit 1000
"""

import argparse
import json
from pathlib import Path
import re

from models.shared.data import load_manifest

OUT_PATH = Path("models/exports/paper_concepts.jsonl")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="Max papers to process")
    ap.add_argument("--out", type=str, default=str(OUT_PATH), help="Output path for paper concepts JSONL")
    return ap.parse_args()


def normalize_token(tok: str) -> str:
    tok = tok.strip().lower()
    tok = re.sub(r"[^a-z0-9_]+", "_", tok)
    tok = re.sub(r"_+", "_", tok).strip("_")
    return tok


def main():
    args = parse_args()
    manifest = load_manifest()
    entries = manifest.get("entries") or manifest.get("papers") or []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            if written >= args.limit:
                break
            pid = entry.get("id") or entry.get("paper_id")
            if not pid:
                continue
            title = entry.get("title") or ""
            cats = entry.get("categories") or entry.get("primary_category") or ""
            if isinstance(cats, str):
                cat_list = cats.split()
            else:
                cat_list = list(cats) if cats else []
            concepts = set()
            for c in cat_list:
                norm = normalize_token(c)
                if norm:
                    concepts.add(norm)
            for tok in title.split():
                norm = normalize_token(tok)
                if len(norm) > 3:
                    concepts.add(norm)
            for c in concepts:
                obj = {
                    "id": f"{pid}:{c}",
                    "paper_id": pid,
                    "name": c,
                    "source": "heuristic_categories_title",
                }
                f.write(json.dumps(obj) + "\n")
            written += 1
    print(f"Wrote concepts for {written} papers to {out_path}")


if __name__ == "__main__":
    main()
