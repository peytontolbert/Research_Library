"""
MirrorMind scaffolding: twins, memories, personas, domain tools, and coordinator.
This module is intentionally lightweight and uses existing exports/shared utils
to provide a callable interface for the architecture described in models/paper.md.
"""

from .memory import Episode, SemanticSummary, EpisodicMemoryStore, SemanticMemoryStore
from .persona import PersonaSchema, PersonaBuilder
from .twins import RepoTwin, PaperTwin
from .domain import DomainGraph, DomainAgent
from .coordinator import Coordinator, ReviewAgent
from .context import ContextAssembler
from .llm import build_llm, safe_build_llm, default_llama1b_config

__all__ = [
    "Episode",
    "SemanticSummary",
    "EpisodicMemoryStore",
    "SemanticMemoryStore",
    "PersonaSchema",
    "PersonaBuilder",
    "RepoTwin",
    "PaperTwin",
    "DomainGraph",
    "DomainAgent",
    "Coordinator",
    "ReviewAgent",
    "ContextAssembler",
    "build_llm",
    "safe_build_llm",
    "default_llama1b_config",
]
