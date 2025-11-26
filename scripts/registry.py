from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_registry_path() -> str:
	"""
	Return the default on-disk location for the adapter registry.

	For the repository library, we keep this alongside the exports under:
	    /data/repository_library/exports/_adapters/adapter_registry.json

	The path is derived relative to this file so that the code works even
	if the project root is relocated.
	"""
	root = Path(__file__).resolve().parent.parent
	artifacts_dir = root / "exports" / "_adapters"
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	return str(artifacts_dir / "adapter_registry.json")


def _read_json(path: str) -> Dict[str, Any]:
	if not os.path.isfile(path):
		return {}
	try:
		with open(path, "r", encoding="utf-8") as fh:
			return json.loads(fh.read()) or {}
	except Exception:
		return {}


def _write_json(path: str, obj: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as fh:
		fh.write(json.dumps(obj, indent=2))


def load_registry(registry_path: Optional[str] = None) -> Dict[str, Any]:
	rp = registry_path or _default_registry_path()
	data = _read_json(rp)
	if not isinstance(data, dict):
		return {}
	return data


def save_registry(data: Dict[str, Any], registry_path: Optional[str] = None) -> str:
	rp = registry_path or _default_registry_path()
	_write_json(rp, data)
	return rp


def register_adapter(adapter_id: str, meta: Dict[str, Any], registry_path: Optional[str] = None) -> str:
	data = load_registry(registry_path)
	now = int(time.time())
	rec = dict(meta)
	rec.setdefault("created_ts", now)
	rec["updated_ts"] = now
	data[adapter_id] = rec
	return save_registry(data, registry_path)


def remove_adapter(adapter_id: str, registry_path: Optional[str] = None) -> str:
	data = load_registry(registry_path)
	if adapter_id in data:
		del data[adapter_id]
	return save_registry(data, registry_path)


def list_adapters(registry_path: Optional[str] = None) -> List[Dict[str, Any]]:
	data = load_registry(registry_path)
	out: List[Dict[str, Any]] = []
	for aid, rec in data.items():
		entry = dict(rec)
		entry["adapter_id"] = aid
		out.append(entry)
	return out



