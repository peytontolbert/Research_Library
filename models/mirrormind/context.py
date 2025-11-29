"""
Context assembly pipeline mirroring models/paper.md section 6.
Composes persona prompt, semantic scope, episodic retrieval, and user task.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from models.mirrormind.twins import RepoTwin, PaperTwin
from models.mirrormind.memory import Episode, SemanticSummary


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

    def build_repo_context(self, twin: RepoTwin, task_text: str, *, max_semantic: int = 5, max_episodic: int = 8) -> Dict[str, str]:
        semantic = twin.semantic_scope(task_text, top_k=max_semantic)
        episodic = twin.episodic_context(task_text, top_k=max_episodic)
        return {
            "system": twin.persona_prompt(),
            "semantic_context": _format_semantic(semantic),
            "episodic_context": _format_episodic(episodic),
            "user_task": task_text,
        }

    def build_paper_context(self, twin: PaperTwin, task_text: str, *, max_semantic: int = 5, max_episodic: int = 8) -> Dict[str, str]:
        semantic = twin.semantic_scope(task_text, top_k=max_semantic)
        episodic = twin.episodic_context(task_text, top_k=max_episodic)
        return {
            "system": twin.persona_prompt(),
            "semantic_context": _format_semantic(semantic),
            "episodic_context": _format_episodic(episodic),
            "user_task": task_text,
        }
