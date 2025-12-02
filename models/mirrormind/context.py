"""
Context assembly pipeline mirroring models/paper.md section 6.
Composes persona prompt, semantic scope, episodic retrieval, and user task.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Optional, Tuple

from models.mirrormind.twins import RepoTwin, PaperTwin
from models.mirrormind.memory import Episode, SemanticSummary
from models.mirrormind.domain import DomainAgent


def _format_semantic(summaries: Sequence[SemanticSummary]) -> str:
    parts: List[str] = []
    seen: set = set()
    for s in summaries:
        key = (s.scope, s.summary_text)
        if key in seen:
            continue
        seen.add(key)
        text = s.summary_text.strip().replace("\n", " ")
        if len(text) > 320:
            text = text[:317] + "..."
        parts.append(f"- [{s.scope}] {text}")
    return "\n".join(parts)


def _format_episodic(episodes: Sequence[Episode]) -> str:
    parts: List[str] = []
    seen: set = set()
    for ep in episodes:
        snippet = ep.text.strip().replace("\n", " ")
        if not snippet:
            continue
        key = (ep.type, snippet)
        if key in seen:
            continue
        seen.add(key)
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        parts.append(f"- ({ep.type}) {snippet}")
    return "\n".join(parts)


class ContextAssembler:
    """Build prompt blocks for RepoTwin and PaperTwin calls."""

    def __init__(self, domain_agent: Optional[DomainAgent] = None) -> None:
        # Optional DomainAgent to allow graph-aware ranking of semantic
        # summaries relative to the current task. When absent, we fall back
        # to purely local scoring.
        self.domain_agent = domain_agent

    def _rank_semantic(
        self,
        summaries: Sequence[SemanticSummary],
        task_text: str,
        max_semantic: int,
    ) -> List[SemanticSummary]:
        """
        Lightweight graph-aware ranking of semantic summaries:
        - If a DomainAgent is available, use its concept search to get
          task-relevant concepts and prefer summaries whose scope or
          key_concepts overlap those names.
        - Otherwise, return the summaries as-is.
        """
        if not summaries:
            return []
        if self.domain_agent is None:
            return list(summaries)[:max_semantic]

        concepts = self.domain_agent.search_concepts(task_text, top_k=8)
        concept_names = [str(c.get("name") or "").lower() for c in concepts if c.get("name")]
        concept_tokens = set(t for name in concept_names for t in name.split())

        scored: List[Tuple[float, SemanticSummary]] = []
        for s in summaries:
            scope_tokens = set((s.scope or "").lower().split())
            key_tokens = set(k.lower() for k in (s.key_concepts or []))
            text_tokens = set((s.summary_text or "").lower().split())
            # Overlap with concept tokens at different levels.
            score = 0.0
            if concept_tokens:
                score += 0.6 * (len(scope_tokens & concept_tokens) / float(len(scope_tokens | concept_tokens) or 1))
                score += 0.3 * (len(key_tokens & concept_tokens) / float(len(key_tokens | concept_tokens) or 1))
                score += 0.1 * (len(text_tokens & concept_tokens) / float(len(text_tokens | concept_tokens) or 1))
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:max_semantic]]

    def build_repo_context(self, twin: RepoTwin, task_text: str, *, max_semantic: int = 5, max_episodic: int = 8) -> Dict[str, str]:
        semantic_raw = twin.semantic_scope(task_text, top_k=max_semantic * 2)
        semantic = self._rank_semantic(semantic_raw, task_text, max_semantic=max_semantic)

        # Primary episodic set: condition retrieval on semantic summaries when
        # available, as described in models/paper.md Section 6.
        if hasattr(twin, "episodic_context_with_semantic"):
            episodic_main = twin.episodic_context_with_semantic(  # type: ignore[attr-defined]
                task_text, semantic, top_k=max_episodic
            )
        else:
            episodic_main = twin.episodic_context(task_text, top_k=max_episodic)

        # Optional ProgramGraph-aware episodic retrieval: if a DomainAgent is
        # available, use its concept search to pick ProgramGraph node IDs that
        # both match the task and belong to this repo, then restrict episodic
        # candidates to those nodes. This makes the end-to-end path
        # "task -> DomainAgent -> ProgramGraph nodes -> episodic episodes"
        # concrete, matching the spec's subsystem-scoped retrieval story.
        episodic_pg: List[Episode] = []
        if self.domain_agent is not None and hasattr(twin, "episodic_for_graph_nodes"):
            repo_id = getattr(twin, "entity_id", None)
            graph_nodes: List[str] = []
            if repo_id:
                concepts = self.domain_agent.search_concepts(task_text, top_k=8)
                for c in concepts:
                    cid = c.get("concept_id")
                    tops = c.get("top_repos") or []
                    if cid and isinstance(tops, list) and repo_id in tops:
                        graph_nodes.append(str(cid))
            # De-duplicate node IDs while preserving order.
            if graph_nodes:
                seen_nodes = set()
                uniq_nodes: List[str] = []
                for nid in graph_nodes:
                    if nid in seen_nodes:
                        continue
                    seen_nodes.add(nid)
                    uniq_nodes.append(nid)
                episodic_pg = twin.episodic_for_graph_nodes(  # type: ignore[attr-defined]
                    task_text,
                    graph_nodes=uniq_nodes,
                    top_k=max_episodic,
                )

        # Merge and trim episodic candidates, preferring ProgramGraph-scoped
        # episodes but falling back to the semantic-conditioned set to keep
        # behavior robust when graph coverage is sparse.
        episodic: List[Episode] = []
        seen_ids = set()
        for ep in episodic_pg + episodic_main:
            if ep.id in seen_ids:
                continue
            seen_ids.add(ep.id)
            episodic.append(ep)
            if len(episodic) >= max_episodic:
                break

        return {
            "system": twin.persona_prompt(),
            "semantic_context": _format_semantic(semantic),
            "episodic_context": _format_episodic(episodic),
            "user_task": task_text,
        }

    def build_paper_context(self, twin: PaperTwin, task_text: str, *, max_semantic: int = 5, max_episodic: int = 8) -> Dict[str, str]:
        semantic_raw = twin.semantic_scope(task_text, top_k=max_semantic * 2)
        semantic = self._rank_semantic(semantic_raw, task_text, max_semantic=max_semantic)
        if hasattr(twin, "episodic_context_with_semantic"):
            episodic = twin.episodic_context_with_semantic(task_text, semantic, top_k=max_episodic)  # type: ignore[attr-defined]
        else:
            episodic = twin.episodic_context(task_text, top_k=max_episodic)
        return {
            "system": twin.persona_prompt(),
            "semantic_context": _format_semantic(semantic),
            "episodic_context": _format_episodic(episodic),
            "user_task": task_text,
        }
