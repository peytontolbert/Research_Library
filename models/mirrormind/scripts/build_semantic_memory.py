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
    Episode,
    SemanticSummary,
)


def _load_episodes(path: Path) -> EpisodicMemoryStore:
    return EpisodicMemoryStore.load(path)


def build_for_entities(
    episodic: EpisodicMemoryStore,
    entities: Sequence[str],
    scope_label: str,
    summarize_fn: Optional[Callable[[str], str]] = None,
) -> SemanticMemoryStore:
    out = SemanticMemoryStore()
    for ent in entities:
        episodes = episodic.episodes_for(ent)
        summaries = build_semantic_summaries(ent, episodes, summarize_fn=summarize_fn, scope_label=scope_label)
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
    semantic_store = build_for_entities(episodic, entities, scope_label=args.scope_label)
    semantic_store.save(args.output)
    print(f"Built {sum(len(v) for v in semantic_store._by_entity.values())} summaries for {len(entities)} entities -> {args.output}")


if __name__ == "__main__":
    main()
