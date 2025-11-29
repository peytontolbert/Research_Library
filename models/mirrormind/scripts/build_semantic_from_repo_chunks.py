"""
Build semantic summaries directly from repo_chunks JSONL exports.

This is a convenience wrapper around `build_semantic_summaries` so we can
generate semantic memory JSONL per repo using an LLM summarizer.

Usage:
    python -m models.mirrormind.scripts.build_semantic_from_repo_chunks \
        --repo-chunks-dir models/exports/repos_chunks \
        --output models/exports/semantic_from_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Dict, Sequence, Optional, Callable

from models.mirrormind.memory import EpisodicMemoryStore, Episode, SemanticMemoryStore, build_semantic_summaries
from models.mirrormind.llm import safe_build_llm


def _repo_id_from_path(path: str) -> str:
    """
    Extract a repo identifier from a repo_chunks path.
    Expected form: /data/repositories/<repo_name>/...
    Falls back to the immediate parent name if the sentinel is absent.
    """
    p = Path(path)
    parts = list(p.parts)
    if "repositories" in parts:
        idx = parts.index("repositories")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # fallback to parent dir name
    return p.parent.name or "unknown"


def _load_episodic_from_chunks(paths: Sequence[Path]) -> EpisodicMemoryStore:
    store = EpisodicMemoryStore()
    per_repo_count: Dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                repo_id = _repo_id_from_path(obj.get("path", ""))
                per_repo_count[repo_id] = per_repo_count.get(repo_id, 0) + 1
                text = obj.get("code") or ""
                if not text:
                    continue
                ep = Episode(
                    id=f"{repo_id}:{per_repo_count[repo_id]}",
                    entity_id=repo_id,
                    time=None,
                    type="doc_paragraph",
                    text=text,
                    graph_context=[],
                    dense=[float(len(text))],
                    sparse={},
                )
                store.add(ep)
    return store


def build_semantic_from_repo_chunks(
    repo_chunks_dir: Path,
    output_path: Path,
    *,
    summarize_fn: Callable[[str], str],
) -> SemanticMemoryStore:
    files = [Path(p) for p in glob(str(repo_chunks_dir / "repo_chunks_*.jsonl"))]
    episodic = _load_episodic_from_chunks(files)
    entities = episodic.entities()
    semantic = SemanticMemoryStore()
    for ent in entities:
        episodes = episodic.episodes_for(ent)
        summaries = build_semantic_summaries(
            ent,
            episodes,
            scope_label="repo_chunks",
            summarize_fn=summarize_fn,
            include_raw=False,
        )
        semantic.bulk_add(summaries)
    semantic.save(output_path)
    return semantic


def _llm_summarizer() -> Optional[callable]:
    """
    Build an LLM-based summarizer. Returns None if the model cannot be loaded.
    """
    model, tokenizer, _ = safe_build_llm()
    if model is None or tokenizer is None:
        return None

    def _summarize(text: str) -> str:
        prompt = (
            "Summarize the following repository code/documentation into 5 bullet points "
            "covering main modules, key functions/classes, and notable behaviors. "
            "Keep it concise but mention specific names.\n\n"
            f"{text}"
        )
        # Guard input length to avoid OOM; chunk if necessary.
        max_chars = 6000
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars]
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return _summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic memory from repo_chunks JSONL exports.")
    parser.add_argument("--repo-chunks-dir", type=Path, required=True, help="Directory containing repo_chunks_*.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path for semantic summaries.")
    args = parser.parse_args()

    summarize_fn = _llm_summarizer()
    if summarize_fn is None:
        raise RuntimeError("LLM summarizer unavailable; ensure weights are present for summarization.")

    semantic = build_semantic_from_repo_chunks(
        args.repo_chunks_dir,
        args.output,
        summarize_fn=summarize_fn,
    )
    total = sum(len(v) for v in semantic._by_entity.values())
    print(f"Wrote {total} semantic summaries to {args.output}")


if __name__ == "__main__":
    main()
