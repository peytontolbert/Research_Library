"""
Shared AgentState for PIANO-style control.
Holds working notes plus pointers into MirrorMind memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.mirrormind.twins import RepoTwin, PaperTwin
from models.mirrormind.domain import DomainAgent


@dataclass
class AgentState:
    """
    Lightweight agent state for the Code+Paper environment.

    This object explicitly binds the PIANO controller to the MirrorMind memory
    layer by holding:
      - RepoTwin / PaperTwin for entity-level episodic + semantic memory,
      - DomainAgent for concept / domain-level reasoning.
    The `summarize` method exposes a compact view of these long-term memories
    so that CognitiveController/LLM policies operate in a graph-native way.
    """

    task: str
    working_memory: List[str] = field(default_factory=list)
    scratchpad: Dict[str, str] = field(default_factory=dict)
    repo_twin: Optional[RepoTwin] = None
    paper_twin: Optional[PaperTwin] = None
    domain_agent: DomainAgent = field(default_factory=DomainAgent)

    def summarize(self) -> str:
        """Return a compact text summary of the agent's state and LTM."""
        wm = "\n".join(self.working_memory[-8:])

        # Persona / identity hints from twins.
        persona_bits: List[str] = []
        if self.repo_twin:
            persona_bits.append(f"repo={self.repo_twin.entity_id}")
        if self.paper_twin:
            persona_bits.append(f"paper={self.paper_twin.entity_id}")
        persona_str = ", ".join(persona_bits) if persona_bits else "none"

        # Concept-level sketch for the current task.
        concept_names: List[str] = []
        try:
            concepts = self.domain_agent.search_concepts(self.task, top_k=3)
            for c in concepts:
                name = c.get("name")
                if name:
                    concept_names.append(str(name))
        except Exception:
            pass
        concepts_str = ", ".join(concept_names) if concept_names else "none"

        # Very small semantic glimpse from repo/paper twins, if present.
        repo_sem = ""
        if self.repo_twin:
            try:
                sem = self.repo_twin.semantic_scope(self.task, top_k=1)
                if sem:
                    repo_sem = sem[0].summary_text[:160]
            except Exception:
                repo_sem = ""
        paper_sem = ""
        if self.paper_twin:
            try:
                sem = self.paper_twin.semantic_scope(self.task, top_k=1)
                if sem:
                    paper_sem = sem[0].summary_text[:160]
            except Exception:
                paper_sem = ""

        return (
            f"TASK: {self.task}\n"
            f"WM:\n{wm}\n"
            f"Scratchpad keys: {list(self.scratchpad.keys())}\n"
            f"Persona: {persona_str}\n"
            f"Concepts: {concepts_str}\n"
            f"RepoSemantic: {repo_sem}\n"
            f"PaperSemantic: {paper_sem}"
        )
