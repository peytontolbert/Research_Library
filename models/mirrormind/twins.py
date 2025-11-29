"""
RepoTwin and PaperTwin scaffolding.
Each twin bundles episodic memory, semantic memory, and a persona schema.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import uuid

from models.mirrormind.memory import Episode, SemanticSummary, EpisodicMemoryStore, SemanticMemoryStore
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
                episodic.add(_make_episode(paper_id, title, "title"))
            if abstract:
                episodic.add(_make_episode(paper_id, abstract, "abstract_chunk"))
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
