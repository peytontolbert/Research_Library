from __future__ import annotations

"""
Skill build and status utilities for per-repository SkillSets.

This module provides:
- Status computation for a given (repo_id, skill) pair:
    - not_built
    - up_to_date
    - needs_update  (repo changed or skill schema changed)
- A lightweight "build" function that:
    - Registers a repo-local adapter in the adapter registry.
    - Updates `exports/_manifest.json` under the `skills` key
      for the corresponding repo.

The actual training logic is intentionally stubbed out for now; this
simulates the presence of a built adapter so that higher-level plumbing
and the UI can be exercised.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from scripts.library_repo_graph_export import DEFAULT_EXPORT_ROOT  # type: ignore
from scripts.registry import register_adapter  # type: ignore
from scripts.repo_library import load_manifest, open_repository  # type: ignore
from modules.model_registry import get_model_config  # type: ignore
from modules.vector_index import build_repo_qa_index  # type: ignore


SKILL_STATUS_NOT_BUILT = "not_built"
SKILL_STATUS_UP_TO_DATE = "up_to_date"
SKILL_STATUS_NEEDS_UPDATE = "needs_update"


# Per-skill schema versions (bump when training/config changes in a
# way that should force rebuild). QA was bumped to 2 when model-based
# adapter configuration was introduced and to 3 when LoRA-based
# repo/task adapters and the swarm orchestration were added so that
# older adapters are treated as stale and rebuilt.
SKILL_SCHEMA_VERSIONS: Dict[str, int] = {
    "qa": 3,
    "edit": 1,
    "meta": 1,
    "nav": 1,
    "test": 1,
    "perf": 1,
    "security": 1,
    "api": 1,
    "style": 1,
}


_MANIFEST_FILENAME = "_manifest.json"


def _manifest_path(export_root: str) -> str:
    return os.path.join(export_root, _MANIFEST_FILENAME)


def _save_manifest(export_root: str, manifest: Dict[str, Any]) -> str:
    os.makedirs(export_root, exist_ok=True)
    path = _manifest_path(export_root)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2))
    return path


def _ensure_repo_entry(manifest: Dict[str, Any], repo_id: str) -> Dict[str, Any]:
    repos = manifest.get("repos") or {}
    if not isinstance(repos, dict):
        raise KeyError("manifest has invalid 'repos' structure")
    entry_any = repos.get(repo_id)
    if not isinstance(entry_any, dict):
        raise KeyError(f"repo_id not found in manifest: {repo_id!r}")
    return entry_any


def _skill_entry_for_repo(entry: Dict[str, Any], skill: str) -> Dict[str, Any]:
    skills = entry.get("skills") or {}
    if not isinstance(skills, dict):
        skills = {}
    rec_any = skills.get(skill)
    return rec_any if isinstance(rec_any, dict) else {}


def _compute_status_for(entry: Dict[str, Any], skill: str) -> Dict[str, Any]:
    repo_state = entry.get("repo_state") or {}
    if not isinstance(repo_state, dict):
        repo_state = {}
    skill_rec = _skill_entry_for_repo(entry, skill)
    target_version = int(SKILL_SCHEMA_VERSIONS.get(skill, 1))

    if not skill_rec:
        return {
            "status": SKILL_STATUS_NOT_BUILT,
            "skill_schema_version": target_version,
        }

    prev_state = skill_rec.get("repo_state") or {}
    if not isinstance(prev_state, dict):
        prev_state = {}
    prev_ver = int(skill_rec.get("skill_schema_version") or 0)

    if prev_state != repo_state or prev_ver != target_version:
        return {
            "status": SKILL_STATUS_NEEDS_UPDATE,
            "skill_schema_version": target_version,
            "last_built_at": skill_rec.get("last_built_at"),
            "adapter_id": skill_rec.get("adapter_id"),
        }

    return {
        "status": SKILL_STATUS_UP_TO_DATE,
        "skill_schema_version": target_version,
        "last_built_at": skill_rec.get("last_built_at"),
        "adapter_id": skill_rec.get("adapter_id"),
    }


def skill_status_for_repo(
    repo_id: str,
    skill: str,
    *,
    export_root: str = DEFAULT_EXPORT_ROOT,
) -> Dict[str, Any]:
    """
    Compute status for a single (repo_id, skill) pair.
    """
    manifest = load_manifest(export_root)
    entry = _ensure_repo_entry(manifest, repo_id)
    info = _compute_status_for(entry, skill)
    out = dict(info)
    out["repo_id"] = repo_id
    out["skill"] = skill
    return out


def all_skill_statuses_for_repo(
    repo_id: str,
    *,
    export_root: str = DEFAULT_EXPORT_ROOT,
    skills: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute status for a set of skills for a given repo.

    If `skills` is omitted, a default SkillSet-like list is used.
    """
    if skills is None:
        skills = list(SKILL_SCHEMA_VERSIONS.keys())
    return [skill_status_for_repo(repo_id, s, export_root=export_root) for s in skills]


def build_skill(
    repo_id: str,
    skill: str,
    *,
    export_root: str = DEFAULT_EXPORT_ROOT,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Lightweight build for a (repo_id, skill) pair.

    For now this:
    - Checks status; if already up_to_date and not force, returns without changes.
    - Registers/updates a repo-local adapter in the adapter registry.
    - Updates the manifest's `skills` entry for this repo/skill.

    Returns a summary dictionary including the new status.
    """
    manifest = load_manifest(export_root)
    repos = manifest.get("repos") or {}
    if not isinstance(repos, dict):
        raise KeyError("manifest has invalid 'repos' structure")
    entry_any = repos.get(repo_id)
    if not isinstance(entry_any, dict):
        raise KeyError(f"repo_id not found in manifest: {repo_id!r}")
    entry: Dict[str, Any] = entry_any

    current_status = _compute_status_for(entry, skill)
    if current_status.get("status") == SKILL_STATUS_UP_TO_DATE and not force:
        summary = dict(current_status)
        summary["repo_id"] = repo_id
        summary["skill"] = skill
        summary["changed"] = False
        return summary

    repo_state = entry.get("repo_state") or {}
    if not isinstance(repo_state, dict):
        repo_state = {}

    # Optionally build / refresh a retrieval index for skills that need it.
    index_meta: Dict[str, Any] = {}
    if skill == "qa":
        try:
            repo_obj = open_repository(repo_id, export_root=export_root)
            index_out_dir = os.path.join(export_root, repo_id, "indices", "qa")
            index_meta = build_repo_qa_index(repo_obj, out_dir=index_out_dir)
        except Exception as exc:
            # Preserve a minimal error marker; callers can still fall back
            # to graph-only QA if index construction fails.
            index_meta = {"type": "simple_numpy", "error": str(exc)}

    # Simulate adapter training by registering/refreshing a record
    # in the adapter registry, now enriched with retrieval metadata
    # and model configuration.
    version = int(SKILL_SCHEMA_VERSIONS.get(skill, 1))
    adapter_id = f"{skill}:{repo_id}"
    built_at = int(time.time())

    adapter_info: Dict[str, Any] = {
        "adapter_id": adapter_id,
        "repo_id": repo_id,
        "skill": skill,
        "built_at": built_at,
        "skill_schema_version": version,
    }

    # Attach model configuration for QA adapters so that runtime can
    # load the appropriate model/quantization/LoRA without hardcoding.
    if skill == "qa":
        # Default to the "llama" model from model.yml, if present.
        base_model_cfg = get_model_config("llama")
        if base_model_cfg is not None:
            adapter_info.setdefault("model_name", base_model_cfg.name)
            adapter_info.setdefault("model_id", base_model_cfg.model_id)
            if base_model_cfg.cache_dir:
                adapter_info.setdefault("cache_dir", base_model_cfg.cache_dir)
        else:
            # Fallback to the HF id even if model.yml is missing/malformed.
            adapter_info.setdefault("model_name", "llama")
            adapter_info.setdefault("model_id", "meta-llama/Meta-Llama-3-8B-Instruct")

        # Prefer 4-bit quantization for QA efficiency by default.
        adapter_info.setdefault("quantization", "4bit")
        # LoRA paths are optional and can be filled in by future training
        # flows. For QA we conceptually support composing a repo adapter
        # and a task adapter at runtime.
        adapter_info.setdefault("repo_lora_path", None)
        adapter_info.setdefault("task_lora_path", None)
        # Generation defaults tuned for short, chat-style QA answers.
        adapter_info.setdefault("max_new_tokens", 256)
        adapter_info.setdefault("temperature", 0.1)
        adapter_info.setdefault("top_p", 0.95)
        # Prefer the two larger GPUs (e.g., 2x3090) for inference.
        adapter_info.setdefault("infer_devices", [0, 1])

        if index_meta and not index_meta.get("error"):
            adapter_info["index"] = index_meta
    else:
        if index_meta and not index_meta.get("error"):
            adapter_info["index"] = index_meta

    adapter_meta: Dict[str, Any] = {
        "type": "repo",
        "repo_id": repo_id,
        "skill": skill,
        "info": adapter_info,
    }
    register_adapter(adapter_id, adapter_meta)

    # Update manifest
    skills_meta = entry.get("skills") or {}
    if not isinstance(skills_meta, dict):
        skills_meta = {}

    skills_meta[skill] = {
        "status": SKILL_STATUS_UP_TO_DATE,
        "skill_schema_version": version,
        "last_built_at": built_at,
        "adapter_id": adapter_id,
        "repo_state": repo_state,
    }

    # Record index metadata under the repo's `indices` section when present.
    indices_meta = entry.get("indices") or {}
    if not isinstance(indices_meta, dict):
        indices_meta = {}
    if index_meta and not index_meta.get("error"):
        indices_meta[skill] = index_meta

    entry["skills"] = skills_meta
    entry["indices"] = indices_meta
    repos[repo_id] = entry
    manifest["repos"] = repos
    _save_manifest(export_root, manifest)

    summary = skill_status_for_repo(repo_id, skill, export_root=export_root)
    summary["changed"] = True
    return summary



