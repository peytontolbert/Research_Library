"""
Offline helper to build semantic summaries from episodic memory JSONL.

Usage:
    python -m models.mirrormind.scripts.build_semantic_memory \
        --episodes models/exports/episodic.jsonl \
        --output models/exports/semantic.jsonl \
        [--entity-id repo123] [--scope-label generic]

This wires the build_semantic_summaries helper so you can plug in a real
summarizer later without changing the storage layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Callable

from models.mirrormind.memory import (
    EpisodicMemoryStore,
    SemanticMemoryStore,
    build_semantic_summaries,
    build_temporal_semantic_summaries,
    Episode,
    SemanticSummary,
)
from models.mirrormind.llm import safe_build_llm


def _load_episodes(path: Path) -> EpisodicMemoryStore:
    return EpisodicMemoryStore.load(path)


def _llm_summarizer() -> Callable[[str], str]:
    """
    Build an LLM-based summarizer for episodic → semantic aggregation.

    This is intentionally *not* optional for the CLI: if the model cannot be
    loaded, we raise instead of silently falling back to raw concatenation of
    code, so that semantic memory truly reflects LLM-derived summaries as
    described in models/paper.md.
    """
    model, tokenizer, _ = safe_build_llm()
    if model is None or tokenizer is None:
        raise RuntimeError(
            "LLM summarizer unavailable; ensure Llama checkpoints are present "
            "under /data/checkpoints before running build_semantic_memory.py."
        )

    def _summarize(text: str) -> str:
        prompt = (
            "Summarize the following repository- or paper-related episodes "
            "(code chunks, documentation, commit messages, issues, etc.) into "
            "5–8 bullet points capturing:\n"
            "- key modules, functions, and classes\n"
            "- important behaviors, APIs, and invariants\n"
            "- any notable patterns, bugs, or refactors mentioned.\n\n"
            "Keep the summary concise but include specific names where possible.\n\n"
            f"{text}"
        )
        # Guard input length to avoid OOM; truncate long spans.
        max_chars = 6000
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars]
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return _summarize


def build_for_entities(
    episodic: EpisodicMemoryStore,
    entities: Sequence[str],
    scope_label: str,
    summarize_fn: Optional[Callable[[str], str]] = None,
) -> SemanticMemoryStore:
    out = SemanticMemoryStore()
    # If no explicit summarizer is provided (e.g., from a caller using this as
    # a library function), default to the shared LLM summarizer so that
    # summaries are never just raw concatenations of episode text.
    if summarize_fn is None:
        summarize_fn = _llm_summarizer()
    for ent in entities:
        episodes = episodic.episodes_for(ent)
        if not episodes:
            continue
        # For commit-style trajectories, prefer temporal sharding to produce
        # multiple windows per repo/entity as described in the spec.
        if scope_label.lower() == "commits" or all(ep.type == "commit_message" for ep in episodes):
            summaries = build_temporal_semantic_summaries(
                ent,
                episodes,
                summarize_fn=summarize_fn,
                scope_label=scope_label,
            )
        else:
            summaries = build_semantic_summaries(
                ent,
                episodes,
                summarize_fn=summarize_fn,
                scope_label=scope_label,
            )
        out.bulk_add(summaries)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic summaries from episodic memory JSONL.")
    parser.add_argument("--episodes", required=True, type=Path, help="Path to episodic JSONL (EpisodicMemoryStore.save format).")
    parser.add_argument("--output", required=True, type=Path, help="Where to write semantic summaries JSONL.")
    parser.add_argument("--entity-id", type=str, help="Optional single entity_id to process; defaults to all in the store.")
    parser.add_argument("--scope-label", type=str, default="generic", help="Scope label to attach to summaries.")
    args = parser.parse_args()

    episodic = _load_episodes(args.episodes)
    entities = [args.entity_id] if args.entity_id else episodic.entities()
    semantic_store = build_for_entities(
        episodic,
        entities,
        scope_label=args.scope_label,
        # For the CLI entrypoint we always use the default LLM summarizer so
        # that semantic memory is fully LLM-derived. Callers that want a
        # different summarizer can pass one explicitly to build_for_entities().
        summarize_fn=None,
    )
    semantic_store.save(args.output)
    print(f"Built {sum(len(v) for v in semantic_store._by_entity.values())} summaries for {len(entities)} entities -> {args.output}")


if __name__ == "__main__":
    main()
