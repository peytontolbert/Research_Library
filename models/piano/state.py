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
    """Lightweight agent state for the Code+Paper environment."""

    task: str
    working_memory: List[str] = field(default_factory=list)
    scratchpad: Dict[str, str] = field(default_factory=dict)
    repo_twin: Optional[RepoTwin] = None
    paper_twin: Optional[PaperTwin] = None
    domain_agent: DomainAgent = field(default_factory=DomainAgent)

    def summarize(self) -> str:
        wm = "\n".join(self.working_memory[-8:])
        return f"TASK: {self.task}\nWM:\n{wm}\nScratchpad keys: {list(self.scratchpad.keys())}"
