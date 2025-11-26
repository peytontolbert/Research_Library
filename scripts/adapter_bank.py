from __future__ import annotations

"""
File-backed AdapterBank implementation for the repository library.

This module provides a minimal concrete implementation of the
`AdapterBank` protocol defined in `modules.repository` by using the
JSON registry utilities from `scripts.registry`.

Registry schema (convention):
    {
      "<adapter_id>": {
        "type": "repo" | "meta",
        "repo_id": "<repo_id>",          # for type == "repo"
        "skill": "qa" | "edit" | "...",  # for type == "repo"
        "task_family": "style_imitation" # for type == "meta"
        "info": { ... arbitrary metadata ... },
        "created_ts": ...,
        "updated_ts": ...
      },
      ...
    }

This is intentionally simple and can be evolved as training pipelines
start writing richer metadata.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from modules.repository import AdapterBank, SkillAdapter  # type: ignore
from scripts.registry import load_registry  # type: ignore


@dataclass
class _RegistrySkillAdapter:
    """Lightweight SkillAdapter backed directly by a registry record."""

    meta: Dict[str, Any]

    def info(self) -> Dict[str, Any]:
        # Expose a shallow copy to avoid accidental in-place modification
        return dict(self.meta)


class FileAdapterBank(AdapterBank):
    """
    AdapterBank implementation backed by a JSON registry file.

    The registry path is resolved by `scripts.registry._default_registry_path`
    when not explicitly provided.
    """

    def __init__(self, registry_path: Optional[str] = None) -> None:
        self._registry_path = registry_path

    @property
    def _data(self) -> Dict[str, Any]:
        return load_registry(self._registry_path)

    def get_repo_adapter(self, repo_id: str, skill: str) -> Optional[SkillAdapter]:
        data = self._data
        for _aid, rec_any in data.items():
            rec = rec_any if isinstance(rec_any, dict) else {}
            if rec.get("type") != "repo":
                continue
            if rec.get("repo_id") != repo_id:
                continue
            if str(rec.get("skill") or "") != skill:
                continue
            # Use everything under "info" as adapter metadata, falling back
            # to the whole record when "info" is absent.
            info_meta = rec.get("info")
            if not isinstance(info_meta, dict):
                info_meta = dict(rec)
            return _RegistrySkillAdapter(meta=info_meta)
        return None

    def get_meta_adapter(self, task_family: str) -> Optional[SkillAdapter]:
        data = self._data
        for _aid, rec_any in data.items():
            rec = rec_any if isinstance(rec_any, dict) else {}
            if rec.get("type") != "meta":
                continue
            if str(rec.get("task_family") or "") != task_family:
                continue
            info_meta = rec.get("info")
            if not isinstance(info_meta, dict):
                info_meta = dict(rec)
            return _RegistrySkillAdapter(meta=info_meta)
        return None


