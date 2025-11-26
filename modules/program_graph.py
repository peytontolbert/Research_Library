from __future__ import annotations

"""
Core, minimal program-graph types and interfaces.

This is a lightweight reimplementation tailored for the repository library.
It provides the shapes expected by the existing scripts:
- ProgramGraph: base class that graph backends subclass.
- Entity, Edge, Artifact, Span, ResolvedAnchor, EntityId: dataclasses
  used throughout the codebase for graph data exchange.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


EntityId = str


@dataclass
class Span:
    """A 1-based inclusive line-span within a file or artifact."""

    start_line: int
    end_line: int


@dataclass
class Entity:
    """
    A logical program entity (e.g., module, class, function, dataset item).

    `uri` is a stable, opaque identifier (often `program://...`),
    while `id` is a backend-specific key (e.g., `py:pkg.mod.Class`).
    """

    uri: str
    id: EntityId
    kind: str
    name: str
    owner: Optional[str] = None
    labels: Optional[List[str]] = None


@dataclass
class Edge:
    """
    A directed relationship between two entities.

    The `type` field is an application-defined relation name such as
    `owns`, `imports`, `calls`, `has_reference`, etc.
    """

    src: EntityId
    dst: EntityId
    type: str
    meta: Optional[Dict[str, Any]] = None


@dataclass
class Artifact:
    """
    A concrete file-like artifact, typically source code or dataset JSON.

    `uri` is usually a `program://{program_id}/artifact/...` URI.
    """

    uri: str
    type: str
    hash: str
    span: Optional[Span] = None


@dataclass
class ResolvedAnchor:
    """
    A concrete resolution of a URI to a file-level artifact and span.
    """

    artifact_uri: str
    span: Span
    hash: str


class ProgramGraph:
    """
    Abstract base for graph backends.

    Concrete implementations are expected to override:
    - entities()
    - edges()
    - artifacts(kind)
    - search_refs(token)
    - resolve(uri)
    - subgraph(seeds, radius)
    """

    # --- Primary graph accessors --- #

    def entities(self) -> Iterable[Entity]:
        raise NotImplementedError

    def edges(self) -> Iterable[Edge]:
        raise NotImplementedError

    def artifacts(self, kind: str) -> Iterable[Artifact]:
        raise NotImplementedError

    # --- Search & navigation helpers --- #

    def search_refs(self, token: str) -> Iterable[Tuple[EntityId, Span]]:
        """
        Search for textual or symbolic references to `token`.

        Backends may provide best-effort implementations; the default
        provides an empty result.
        """
        return []

    def resolve(self, uri: str) -> ResolvedAnchor:
        """
        Resolve a program URI (usually `program://...`) to a concrete
        artifact + span.
        """
        raise NotImplementedError

    def subgraph(self, seeds: List[EntityId], radius: int) -> "ProgramGraph":
        """
        Optionally return a restricted view over a neighborhood around `seeds`.

        Implementations may return `self` if they do not support filtered
        views; callers should be prepared for that.
        """
        return self



