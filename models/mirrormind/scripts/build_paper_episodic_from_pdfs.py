"""
Build episodic episodes for papers from structured PDF exports.

This script is a convenience wrapper that turns the layout-aware PDF tokens
under `exports/pdfs_structured/` into `Episode` objects suitable for use
with `EpisodicMemoryStore` and `SemanticMemoryStore`.

It focuses on closing the data coverage gap described in models/paper.md:
- sections / headings -> body_chunk episodes
- equations / pseudo-code (heuristically detected) -> equation_block / pseudo_code

Usage (example):
    python -m models.mirrormind.scripts.build_paper_episodic_from_pdfs \\
        --structured-dir exports/pdfs_structured \\
        --output models/exports/paper_episodic_from_pdfs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from models.mirrormind.memory import Episode, EpisodicMemoryStore


def _load_structured_tokens(structured_dir: Path) -> Iterable[Dict]:
    if not structured_dir.exists():
        return []
    for shard in sorted(structured_dir.glob("pdf_structured_*.jsonl")):
        try:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except Exception:
            continue


def _detect_block_type(text: str, token_type: str) -> str:
    """
    Heuristic mapping from PDF token to episode.type.

    - headings / short lines -> abstract_chunk or body_chunk
    - lines with math-like symbols -> equation_block
    - lines that look code-ish -> pseudo_code
    """
    t = text.strip()
    lower = t.lower()
    if not t:
        return "body_chunk"
    if token_type == "heading":
        if lower.startswith("abstract"):
            return "abstract_chunk"
        return "body_chunk"
    # Very lightweight code/pseudo-code detection.
    if any(sym in t for sym in ("def ", "class ", "{", "}", "=>", "::")):
        return "pseudo_code"
    # Very lightweight equation detection.
    if any(sym in t for sym in ("=", r"\int", r"\sum", "∑", "∫", "≥", "≤")) and len(t.split()) <= 40:
        return "equation_block"
    return "body_chunk"


def build_paper_episodes_from_pdfs(structured_dir: Path) -> EpisodicMemoryStore:
    """
    Convert structured PDF tokens into paper-level Episode objects.

    Expects each JSONL row to contain at least:
        - pdf_path: path to the PDF on disk
        - tokens: list of {type, text, ...}
        - paper_id (optional): if absent, paper_id is inferred from basename.
    """
    store = EpisodicMemoryStore()
    per_paper_count: Dict[str, int] = {}

    for rec in _load_structured_tokens(structured_dir):
        pdf_path = rec.get("pdf_path") or ""
        # Best-effort paper identifier: caller can override via explicit field.
        paper_id = rec.get("paper_id") or Path(pdf_path).stem or "unknown_paper"
        tokens = rec.get("tokens") or []
        if not isinstance(tokens, list):
            continue
        for tok in tokens:
            text = str(tok.get("text") or "").strip()
            if not text:
                continue
            token_type = str(tok.get("type") or "text")
            ep_type = _detect_block_type(text, token_type)
            per_paper_count[paper_id] = per_paper_count.get(paper_id, 0) + 1
            ep = Episode(
                id=f"{paper_id}:{per_paper_count[paper_id]}",
                entity_id=paper_id,
                time=None,
                type=ep_type,
                text=text,
                graph_context=[],
                dense=[float(len(text))],
                sparse={},
            )
            store.add(ep)
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper episodic episodes from structured PDF exports.")
    parser.add_argument(
        "--structured-dir",
        type=Path,
        default=Path("exports/pdfs_structured"),
        help="Directory containing pdf_structured_*.jsonl shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for paper episodic episodes.",
    )
    args = parser.parse_args()

    store = build_paper_episodes_from_pdfs(args.structured_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entity_id in store.entities():
            for ep in store.episodes_for(entity_id):
                f.write(json.dumps(ep.__dict__, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


