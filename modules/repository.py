from __future__ import annotations

"""
Repository-level abstractions built on top of exported program graphs.

These types are intentionally lightweight: they provide a normalized
Python representation of a repository in the library, without committing
to any particular storage backend (graph DB, vector index, adapter
runtime, etc.).

The default implementation is file/manifest-based and is designed to
work with the JSONL exports produced under
`/data/repository_library/exports`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol

from .program_graph import ProgramGraph


class RepoIndex(Protocol):
    """
    Abstract handle for per-repository indices (embeddings / search).

    Concrete implementations might:
    - Keep all state in-memory (e.g., numpy arrays + metadata).
    - Proxy to an external vector store (Qdrant/FAISS/etc.) by key.
    - Be a thin wrapper over JSON/NPY files under the repo's export dir.
    """

    def search(self, query: str, *, top_k: int = 20) -> Any:  # pragma: no cover - interface only
        """Return index-specific search results for `query`."""
        ...


class RepoTools(Protocol):
    """
    Abstract interface for repo-local tools used by higher-level skills.

    Implementations are expected to be thin, side-effect-aware wrappers
    around real tools (pytest, linters, build systems, grep, etc.).
    """

    def grep(self, pattern: str, *, path: Optional[str] = None) -> Any:  # pragma: no cover - interface only
        """Search within the repository's files."""
        ...

    def run_tests(self, *, selector: Optional[str] = None) -> Any:  # pragma: no cover - interface only
        """Run the repository's tests (or a subset) and return a summary."""
        ...

    def build(self, *, target: Optional[str] = None) -> Any:  # pragma: no cover - interface only
        """Invoke the repository's build system for an optional target."""
        ...


class SkillAdapter(Protocol):
    """
    Common protocol for a single skill adapter, such as QA or Edit.

    Each adapter is free to define richer methods; this protocol exists
    mainly to give type-checkers a stable anchor and to document intent.
    """

    def info(self) -> Mapping[str, Any]:  # pragma: no cover - interface only
        """Return metadata describing this adapter (kind, version, etc.)."""
        ...


class AdapterBank(Protocol):
    """
    Registry/lookup interface for skill adapters.

    Concrete implementations are expected to know how to:
    - Materialize repo-local adapters (e.g. qa[repo_id]).
    - Materialize cross-repo/meta adapters (e.g. style_imitation).
    - Optionally consult `scripts/registry.py` or other metadata stores.
    """

    def get_repo_adapter(self, repo_id: str, skill: str) -> Optional[SkillAdapter]:  # pragma: no cover - interface only
        """
        Return an adapter for a specific repo + skill (e.g. skill='qa' or 'edit'),
        or None if not available.
        """
        ...

    def get_meta_adapter(self, task_family: str) -> Optional[SkillAdapter]:  # pragma: no cover - interface only
        """
        Return a meta-level adapter for a task family such as 'style_imitation'
        or 'test_generation', or None if not available.
        """
        ...


@dataclass
class SkillSet:
    """
    Collection of per-repository skills / adapters.

    All fields are optional so that a Repository can expose only a subset
    of skills (e.g., QA-only, or QA+Edit) while keeping the interface
    stable for callers.
    """

    qa: Optional[SkillAdapter] = None
    edit: Optional[SkillAdapter] = None
    meta: Optional[SkillAdapter] = None
    nav: Optional[SkillAdapter] = None
    test: Optional[SkillAdapter] = None
    perf: Optional[SkillAdapter] = None
    security: Optional[SkillAdapter] = None
    api: Optional[SkillAdapter] = None
    style: Optional[SkillAdapter] = None


@dataclass
class Repository:
    """
    Normalized view of a single repository in the library.

    This type is deliberately conservative: it does *not* perform any I/O
    by itself and assumes that:
    - `graph` is an already-constructed ProgramGraph instance.
    - `index`, `tools`, and `skills` are optional and may be attached
      lazily by higher-level code.
    - Additional metadata can be threaded through the `metadata` dict,
      typically seeded from `exports/_manifest.json`.
    """

    repo_id: str
    root_path: Path
    graph: ProgramGraph
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: Optional[RepoIndex] = None
    tools: Optional[RepoTools] = None
    skills: SkillSet = field(default_factory=SkillSet)

    def with_index(self, index: RepoIndex) -> "Repository":
        """Return a shallow copy with `index` attached."""
        self.index = index
        return self

    def with_tools(self, tools: RepoTools) -> "Repository":
        """Return a shallow copy with `tools` attached."""
        self.tools = tools
        return self

    def with_skills(self, skills: SkillSet) -> "Repository":
        """Return a shallow copy with `skills` attached."""
        self.skills = skills
        return self


