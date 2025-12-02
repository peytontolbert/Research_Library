"""
RepoTwin and PaperTwin scaffolding.
Each twin bundles episodic memory, semantic memory, and a persona schema.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import uuid
from pathlib import Path

from models.mirrormind.memory import (
    Episode,
    SemanticSummary,
    EpisodicMemoryStore,
    SemanticMemoryStore,
    EpisodeType,
)
from models.mirrormind.persona import PersonaBuilder, PersonaSchema
from models.shared.data import load_manifest


def _make_episode(entity_id: str, text: str, type_name: str) -> Episode:
    return Episode(
        id=str(uuid.uuid4()),
        entity_id=entity_id,
        time=None,
        type=type_name,
        text=text,
        graph_context=[],
        dense=[float(len(text))],
        sparse={},
    )


class BaseTwin:
    """Shared helpers for RepoTwin/PaperTwin."""

    def __init__(
        self,
        entity_id: str,
        persona: PersonaSchema,
        episodic: EpisodicMemoryStore,
        semantic: SemanticMemoryStore,
    ) -> None:
        self.entity_id = entity_id
        self.persona = persona
        self.episodic = episodic
        self.semantic = semantic

    def persona_prompt(self) -> str:
        return self.persona.to_prompt()

    def semantic_scope(self, task_text: str, top_k: int = 3) -> List[SemanticSummary]:
        return self.semantic.query(entity_id=self.entity_id, text=task_text, top_k=top_k)

    def episodic_context(
        self, task_text: str, types: Optional[Sequence[str]] = None, top_k: int = 5
    ) -> List[Episode]:
        return self.episodic.query(entity_id=self.entity_id, text=task_text, types=types, top_k=top_k)

    def episodic_context_with_semantic(
        self,
        task_text: str,
        semantic_summaries: Sequence[SemanticSummary],
        types: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[Episode]:
        """
        Episodic retrieval that is explicitly conditioned on semantic summaries.

        This mirrors the "semantic_context_r → episodic_context_r" path in
        models/paper.md Section 6 by:
        - enriching the query text with top-k semantic summaries,
        - biasing retrieval toward tests/docs via type_weights.
        """
        # Enrich the textual query with a small slice of semantic summaries.
        extra_bits: List[str] = []
        for s in semantic_summaries[:3]:
            if s.summary_text:
                extra_bits.append(s.summary_text)
        enriched_text = task_text
        if extra_bits:
            enriched_text = task_text + " " + " ".join(extra_bits)

        # Prefer tests and docs, as suggested by the spec.
        type_weights: Dict[str, float] = {
            "test_case": 1.5,
            "doc_paragraph": 1.3,
            "commit_message": 1.1,
        }
        return self.episodic.query(
            entity_id=self.entity_id,
            text=enriched_text,
            types=types,
            type_weights=type_weights,
            top_k=top_k,
        )

    def episodic_for_graph_nodes(
        self,
        task_text: str,
        graph_nodes: Sequence[str],
        types: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[Episode]:
        """
        ProgramGraph-aware episodic retrieval:
        restrict candidates to episodes whose graph_context intersects the
        provided ProgramGraph node IDs / file URIs before applying the usual
        text/type/recency scoring. This is a thin wrapper over
        EpisodicMemoryStore.query(graph_nodes=...) to mirror the paper's
        "subsystem-scoped" retrieval story.
        """
        return self.episodic.query(
            entity_id=self.entity_id,
            text=task_text,
            types=types,
            graph_nodes=graph_nodes,
            top_k=top_k,
        )


class RepoTwin(BaseTwin):
    """Digital twin for a repository."""

    def __init__(
        self,
        repo_id: str,
        persona_builder: Optional[PersonaBuilder] = None,
        episodes: Optional[List[Episode]] = None,
        summaries: Optional[List[SemanticSummary]] = None,
    ) -> None:
        persona_builder = persona_builder or PersonaBuilder()
        episodic = EpisodicMemoryStore()
        semantic = SemanticMemoryStore()
        # If explicit episodes/summaries are not provided, attempt to load
        # defaults from the on-disk episodic/semantic exports so that RepoTwin
        # can reflect both repo_chunks and commit trajectories out of the box.
        if episodes is None:
            episodes = []
            # Commit-level episodes (e.g., built by build_commit_episodic_from_git.py).
            commit_path = Path("models/exports/commit_episodes.jsonl")
            if commit_path.exists():
                commit_store = EpisodicMemoryStore.load(commit_path)
                episodes.extend(commit_store.episodes_for(repo_id))
        if summaries is None:
            summaries = []
            # Semantic summaries from repo_chunks.
            chunks_sem_path = Path("models/exports/semantic_from_chunks.jsonl")
            if chunks_sem_path.exists():
                chunks_store = SemanticMemoryStore.load(chunks_sem_path)
                summaries.extend(chunks_store._by_entity.get(repo_id, []))
            # Optional commit-level semantic trajectories, if present.
            commits_sem_path = Path("models/exports/semantic_commits.jsonl")
            if commits_sem_path.exists():
                commits_store = SemanticMemoryStore.load(commits_sem_path)
                summaries.extend(commits_store._by_entity.get(repo_id, []))

        if episodes:
            episodic.bulk_add(episodes)
        if summaries:
            semantic.bulk_add(summaries)
        persona = persona_builder.build(repo_id, persona_type="repo")
        super().__init__(entity_id=repo_id, persona=persona, episodic=episodic, semantic=semantic)


class PaperTwin(BaseTwin):
    """Digital twin for a paper using arXiv metadata as a proxy."""

    def __init__(
        self,
        paper_id: str,
        manifest: Optional[Dict] = None,
        persona_builder: Optional[PersonaBuilder] = None,
    ) -> None:
        persona_builder = persona_builder or PersonaBuilder()
        episodic = EpisodicMemoryStore()
        semantic = SemanticMemoryStore()

        # Very lightweight sampling from the arXiv manifest.
        manifest = manifest or load_manifest()
        entries = manifest.get("entries") or manifest.get("papers") or []
        for entry in entries[:16]:
            pid = entry.get("id") or entry.get("paper_id") or ""
            if pid and pid != paper_id:
                continue
            abstract = entry.get("abstract") or entry.get("summary") or ""
            title = entry.get("title") or ""
            if title:
                episodic.add(_make_episode(paper_id, title, EpisodeType.TITLE))
            if abstract:
                episodic.add(_make_episode(paper_id, abstract, EpisodeType.ABSTRACT_CHUNK))
                semantic.add(
                    SemanticSummary(
                        id=str(uuid.uuid4()),
                        entity_id=paper_id,
                        time_window=entry.get("update_date", ""),
                        scope="abstract",
                        summary_text=abstract,
                        key_concepts=[c.strip() for c in (entry.get("categories") or "").split() if c],
                        dense=[float(len(abstract))],
                    )
                )
            break

        persona = persona_builder.build(paper_id, persona_type="paper")
        super().__init__(entity_id=paper_id, persona=persona, episodic=episodic, semantic=semantic)
