"""
Lightweight shared types and interfaces for program graphs and repositories.

This package currently exposes:
- `program_graph`:
  - ProgramGraph: abstract base class for graph backends
  - Entity, Edge, Artifact, Span, ResolvedAnchor, EntityId
- `repository` (planned/experimental):
  - Repository, RepoIndex, RepoTools, SkillSet: structured views over
    per-repository exports under `/data/repository_library/exports`.

Scripts under `scripts/` use these to represent repositories, datasets,
and other graph-like structures in a uniform way.
"""

from .program_graph import (  # noqa: F401
    ProgramGraph,
    Entity,
    Edge,
    Artifact,
    Span,
    ResolvedAnchor,
    EntityId,
)

