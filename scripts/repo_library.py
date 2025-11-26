from __future__ import annotations

"""
Repository Library: in-process helpers and high-level orchestration.

This module sits on top of:
- `scripts/library_repo_graph_export.py` for the on-disk manifest/exports.
- `scripts/python_repo_graph.py` for per-repo ProgramGraph backends.
- `modules.repository` for in-process Repository / SkillSet abstractions.

It provides two layers:

1. Low-level helpers:
   - Manifest loading.
   - Opening individual `Repository` objects from exports.

2. A high-level `RepoLibrary` façade that:
   - Selects repositories for a query or task.
   - Looks up suitable adapters via an `AdapterBank`.
   - Returns a structured plan describing which repos, skills, and
     context to use for downstream LLM calls or trainers.

The LLM runtime itself is intentionally out of scope for this module.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.repository import AdapterBank, Repository, SkillSet  # type: ignore
from scripts.library_repo_graph_export import (  # type: ignore
    DEFAULT_EXPORT_ROOT,
)
from scripts.python_repo_graph import PythonRepoGraph  # type: ignore


_MANIFEST_FILENAME = "_manifest.json"


def _manifest_path(export_root: str) -> str:
    return os.path.join(export_root, _MANIFEST_FILENAME)


def load_manifest(export_root: str = DEFAULT_EXPORT_ROOT) -> Dict[str, Any]:
    """
    Load the library manifest from `export_root`.

    If the manifest is missing or malformed, an empty dictionary is
    returned. Callers should treat the result as read-only.
    """
    path = _manifest_path(export_root)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.loads(fh.read())
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def list_repo_ids(export_root: str = DEFAULT_EXPORT_ROOT) -> List[str]:
    """
    Return a list of known repo_ids from the manifest.
    """
    manifest = load_manifest(export_root)
    repos = manifest.get("repos") or {}
    if not isinstance(repos, dict):
        return []
    return sorted(repos.keys())


def _repo_entry(manifest: Dict[str, Any], repo_id: str) -> Optional[Dict[str, Any]]:
    repos = manifest.get("repos") or {}
    if not isinstance(repos, dict):
        return None
    entry = repos.get(repo_id)
    return entry if isinstance(entry, dict) else None


def compute_repo_context_key(repo_id: str, entry: Dict[str, Any]) -> str:
    """
    Compute a stable context key c_repo for a repository.

    This follows the high-level idea:
        c_repo = hash(repo_id || branch || language_profile)

    At present we derive:
    - repo_id from the manifest key.
    - branch from `entry["repo_state"].get("branch")` when available.
    - language_profile from an optional `languages` field (if present).
    """
    repo_state = entry.get("repo_state") or {}
    if not isinstance(repo_state, dict):
        repo_state = {}
    branch = str(repo_state.get("branch") or "")
    langs = entry.get("languages") or []
    if isinstance(langs, (list, tuple)):
        lang_profile = ",".join(sorted(str(x) for x in langs))
    else:
        lang_profile = ""
    raw = f"{repo_id}|{branch}|{lang_profile}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def open_repository(
    repo_id: str,
    *,
    export_root: str = DEFAULT_EXPORT_ROOT,
) -> Repository:
    """
    Construct a Repository object for `repo_id` using the current manifest.

    This performs no embedding/index loading and does not instantiate any
    real skill adapters; it simply wires together:
    - repo_id
    - root_path (from manifest)
    - a fresh PythonRepoGraph
    - metadata (manifest entry)
    - an empty SkillSet placeholder
    """
    manifest = load_manifest(export_root)
    entry = _repo_entry(manifest, repo_id)
    if not entry:
        raise KeyError(f"repo_id not found in manifest: {repo_id!r}")

    repo_root = entry.get("repo_root")
    if not isinstance(repo_root, str) or not repo_root:
        raise ValueError(f"invalid or missing repo_root for repo_id={repo_id!r}")

    root_path = Path(os.path.abspath(repo_root))
    graph = PythonRepoGraph(str(root_path))

    # Shallow copy of manifest entry as metadata; callers may add more.
    metadata: Dict[str, Any] = dict(entry)

    # Skills are not materialized here; we only preserve any metadata
    # that may have been recorded under `skills` in the manifest.
    skills_meta = metadata.get("skills") or {}
    if not isinstance(skills_meta, dict):
        skills_meta = {}
    skills = SkillSet()  # actual adapters to be attached elsewhere
    # For convenience, attach skills metadata under Repository.metadata.
    metadata["skills"] = skills_meta

    return Repository(
        repo_id=repo_id,
        root_path=root_path,
        graph=graph,
        metadata=metadata,
        index=None,
        tools=None,
        skills=skills,
    )


def iter_repositories(
    *,
    export_root: str = DEFAULT_EXPORT_ROOT,
    repo_ids: Optional[Iterable[str]] = None,
) -> Iterable[Repository]:
    """
    Convenience generator that yields Repository objects for all (or a
    selected subset of) repo_ids present in the manifest.
    """
    manifest = load_manifest(export_root)
    repos = manifest.get("repos") or {}
    if not isinstance(repos, dict):
        return []

    ids: Iterable[str]
    if repo_ids is not None:
        ids = repo_ids
    else:
        ids = repos.keys()

    for rid in ids:
        try:
            yield open_repository(rid, export_root=export_root)
        except Exception:
            # Skip repos that cannot be opened; callers can choose to log if needed.
            continue


class QueryMode(str, Enum):
    """High-level query modes supported by RepoLibrary."""

    QA = "qa"
    QA_COMPARATIVE = "qa_comparative"


class TaskMode(str, Enum):
    """High-level task modes supported by RepoLibrary."""

    META_SKILL = "meta_skill"
    AGENT_EDIT = "agent_edit"


@dataclass
class RepoSelection:
    """Selection result for a query: which repos and context keys to use."""

    repo_ids: List[str]
    context_keys: Dict[str, str]


class RepoLibrary:
    """
    High-level façade over the repository library.

    Responsibilities:
    - Load manifest and `Repository` objects.
    - Select relevant repositories for a query or task.
    - Look up appropriate adapters via an `AdapterBank`.
    - Return a structured *plan* that downstream components (LLM services,
      trainers, agents) can execute.
    """

    def __init__(
        self,
        *,
        base_model: Any,
        adapter_bank: Optional[AdapterBank] = None,
        export_root: str = DEFAULT_EXPORT_ROOT,
    ) -> None:
        self._base_model = base_model
        self._adapter_bank = adapter_bank
        self._export_root = export_root

    # --- Public API --- #

    def query(
        self,
        *,
        question: str,
        mode: QueryMode = QueryMode.QA,
        repo_hint: Optional[str] = None,
        repo_hints: Optional[Sequence[str]] = None,
        qa_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Plan a query over the repository library.

        This method does not call the LLM itself; instead, it returns a
        structured plan specifying:
        - Which repos are involved.
        - Which skills / adapters to use.
        - Per-repo context keys for conditioning.
        - (Optionally) a QA sub-mode hint (`qa_mode`) for QA-style queries.

        Downstream code is expected to use this plan to build prompts and
        invoke the actual model.
        """
        manifest = load_manifest(self._export_root)
        if mode == QueryMode.QA:
            if not repo_hint:
                raise ValueError("QueryMode.QA requires `repo_hint` (single repo_id).")
            selection = self._select_repos_single(manifest, repo_hint)
        elif mode == QueryMode.QA_COMPARATIVE:
            hints = list(repo_hints or [])
            if not hints:
                raise ValueError("QueryMode.QA_COMPARATIVE requires non-empty `repo_hints`.")
            selection = self._select_repos_multi(manifest, hints)
        else:
            raise ValueError(f"unsupported QueryMode: {mode!r}")

        # Attach skill/adapters where available
        skills: Dict[str, Dict[str, Any]] = {}
        if self._adapter_bank is not None:
            for rid in selection.repo_ids:
                qa_adapter = self._adapter_bank.get_repo_adapter(rid, "qa")
                if qa_adapter is not None:
                    skills[rid] = {"qa": qa_adapter.info()}

        plan: Dict[str, Any] = {
            "type": "query_plan",
            "mode": mode.value,
            "question": question,
            "repos": selection.repo_ids,
            "repo_context_keys": selection.context_keys,
            "skills": skills,
        }
        if qa_mode is not None:
            plan["qa_mode"] = qa_mode
        return plan

    def run_task(
        self,
        *,
        mode: TaskMode,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Plan a library-level task such as meta-skill training.

        For TaskMode.META_SKILL, the config is expected to contain:
        - task_family: e.g. "style_imitation", "test_generation"
        - target_repos: list of repo_ids
        - num_tasks: approximate task count to generate
        """
        manifest = load_manifest(self._export_root)

        if mode is TaskMode.META_SKILL:
            task_family = str(config.get("task_family") or "").strip()
            target_repos = list(config.get("target_repos") or [])
            if not task_family:
                raise ValueError("TaskMode.META_SKILL requires `task_family` in config.")
            if not target_repos:
                raise ValueError("TaskMode.META_SKILL requires `target_repos` in config.")
            selection = self._select_repos_multi(manifest, target_repos)

            meta_adapter_info: Optional[Dict[str, Any]] = None
            if self._adapter_bank is not None:
                meta_adapter = self._adapter_bank.get_meta_adapter(task_family)
                if meta_adapter is not None:
                    meta_adapter_info = dict(meta_adapter.info())

            plan: Dict[str, Any] = {
                "type": "task_plan",
                "mode": mode.value,
                "task_family": task_family,
                "target_repos": selection.repo_ids,
                "repo_context_keys": selection.context_keys,
                "num_tasks": int(config.get("num_tasks") or 0),
                "meta_adapter": meta_adapter_info,
            }
            return plan

        if mode is TaskMode.AGENT_EDIT:
            # Skeleton placeholder for future agentic editing tasks.
            plan = {
                "type": "task_plan",
                "mode": mode.value,
                "config": dict(config),
            }
            return plan

        raise ValueError(f"unsupported TaskMode: {mode!r}")

    # --- Internal helpers --- #

    def _select_repos_single(self, manifest: Dict[str, Any], repo_id: str) -> RepoSelection:
        entry = _repo_entry(manifest, repo_id)
        if not entry:
            raise KeyError(f"repo_id not found in manifest: {repo_id!r}")
        c_repo = compute_repo_context_key(repo_id, entry)
        return RepoSelection(repo_ids=[repo_id], context_keys={repo_id: c_repo})

    def _select_repos_multi(self, manifest: Dict[str, Any], repo_ids: Sequence[str]) -> RepoSelection:
        repo_ids_norm: List[str] = []
        context: Dict[str, str] = {}
        for rid in repo_ids:
            entry = _repo_entry(manifest, rid)
            if not entry:
                continue
            c_repo = compute_repo_context_key(rid, entry)
            repo_ids_norm.append(rid)
            context[rid] = c_repo
        if not repo_ids_norm:
            raise KeyError("no valid repo_ids found in manifest for selection")
        return RepoSelection(repo_ids=repo_ids_norm, context_keys=context)


