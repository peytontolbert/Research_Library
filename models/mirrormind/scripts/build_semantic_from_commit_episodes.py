"""
Build semantic commit trajectories from commit-level episodic episodes.

This script takes the commit episodes produced by
`build_commit_episodic_from_git.py` and aggregates them into temporal
`SemanticSummary` windows per repo, writing the result to a JSONL file
compatible with the `RepoTwin` loader (see `semantic_commits.jsonl` in
`twins.py`).

Usage (example):
    python -m models.mirrormind.scripts.build_semantic_from_commit_episodes \\
        --episodes models/exports/commit_episodes.jsonl \\
        --output models/exports/semantic_commits.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Callable

from models.mirrormind.memory import (
    EpisodicMemoryStore,
    SemanticMemoryStore,
    build_temporal_semantic_summaries,
)
from models.mirrormind.llm import safe_build_llm


def _llm_summarizer() -> Optional[Callable[[str], str]]:
    """
    Build an LLM-based summarizer. Returns None if the model cannot be loaded.
    """
    model, tokenizer, _ = safe_build_llm()
    if model is None or tokenizer is None:
        return None

    def _summarize(text: str) -> str:
        prompt = (
            "Summarize the following sequence of commit messages into 5 bullet "
            "points that describe the main evolution of this repository over "
            "time (features, refactors, bugfixes, and major architectural "
            "changes). Be concise but mention specific components when "
            "possible.\n\n"
            f"{text}"
        )
        max_chars = 6000
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars]
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return _summarize


def build_semantic_from_commit_episodes(
    episodes_path: Path,
    output_path: Path,
    *,
    summarize_fn: Callable[[str], str],
) -> SemanticMemoryStore:
    """
    Load commit-level episodes from JSONL and aggregate them into temporal
    semantic windows per repo.
    """
    episodic = EpisodicMemoryStore.load(episodes_path)
    semantic = SemanticMemoryStore()
    for entity_id in episodic.entities():
        eps = episodic.episodes_for(entity_id)
        summaries = build_temporal_semantic_summaries(
            entity_id,
            eps,
            summarize_fn=summarize_fn,
            scope_label="commit_trajectory",
        )
        semantic.bulk_add(summaries)
    semantic.save(output_path)
    return semantic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build semantic commit trajectories from commit episodes."
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="Path to commit_episodes.jsonl produced by build_commit_episodic_from_git.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for semantic commit summaries (e.g., semantic_commits.jsonl).",
    )
    args = parser.parse_args()

    summarize_fn = _llm_summarizer()
    if summarize_fn is None:
        raise RuntimeError(
            "LLM summarizer unavailable; ensure weights are present for summarization."
        )

    semantic = build_semantic_from_commit_episodes(
        args.episodes,
        args.output,
        summarize_fn=summarize_fn,
    )
    total = sum(len(v) for v in semantic._by_entity.values())
    print(f"Wrote {total} semantic commit summaries to {args.output}")


if __name__ == "__main__":
    main()


