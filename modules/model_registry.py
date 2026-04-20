from __future__ import annotations

"""
Lightweight model registry backed by `model.yml`.

This module exposes a small typed interface for looking up models by
logical name (e.g. "llama", "T5", "bert") so that skills/adapters can
refer to models in a stable, config-driven way.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    model_id: str
    model_path: Optional[str]
    model_type: str
    cache_dir: Optional[str]
    model_description: Optional[str]
    model_tags: List[str]
    raw: Dict[str, Any]


_MODEL_REGISTRY: Optional[Dict[str, ModelConfig]] = None


def _default_model_yaml_path() -> Path:
    """
    Return the default location of `model.yml` relative to the project root.
    """
    # modules/ -> project root is parent
    root = Path(__file__).resolve().parent.parent
    return root / "model.yml"


def _load_model_yaml(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _default_model_yaml_path()
    if not p.is_file():
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _build_registry() -> Dict[str, ModelConfig]:
    data = _load_model_yaml()
    models_any = data.get("models")

    # Support both:
    # - mapping: { "llama": {...}, "t5": {...} }
    # - list: [ { name: "llama", ... }, ... ]
    registry: Dict[str, ModelConfig] = {}

    if isinstance(models_any, dict):
        items = []
        for key, val in models_any.items():
            if isinstance(val, dict):
                v = dict(val)
                v.setdefault("name", key)
                items.append(v)
    elif isinstance(models_any, list):
        items = [m for m in models_any if isinstance(m, dict)]
    else:
        items = []

    for rec in items:
        name_any = rec.get("name")
        if not isinstance(name_any, str):
            continue
        name = name_any.strip()
        if not name:
            continue
        model_id = str(rec.get("model_id") or "").strip()
        if not model_id:
            continue
        model_path_any = rec.get("model_path")
        if isinstance(model_path_any, str):
            model_path = model_path_any.strip() or None
        else:
            model_path = None
        model_type = str(rec.get("model_type") or "").strip() or "causal-lm"
        cache_dir_val = rec.get("cache_dir")
        if isinstance(cache_dir_val, str):
            cache_dir = cache_dir_val.strip() or None
        else:
            cache_dir = None
        desc = rec.get("model_description")
        if isinstance(desc, str):
            desc = desc.strip() or None
        else:
            desc = None
        tags_any = rec.get("model_tags") or []
        tags = [str(t) for t in tags_any] if isinstance(tags_any, list) else []
        cfg = ModelConfig(
            name=name,
            model_id=model_id,
            model_path=model_path,
            model_type=model_type,
            cache_dir=cache_dir,
            model_description=desc,
            model_tags=tags,
            raw=dict(rec),
        )
        registry[name] = cfg

    return registry


def _ensure_registry() -> Dict[str, ModelConfig]:
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        _MODEL_REGISTRY = _build_registry()
    return _MODEL_REGISTRY


def get_model_config(name: str) -> Optional[ModelConfig]:
    """
    Look up a model configuration by logical name.

    Returns None if the model is unknown or config is missing.
    """
    reg = _ensure_registry()
    key = str(name or "").strip()
    if not key:
        return None
    return reg.get(key)


def list_models() -> Dict[str, ModelConfig]:
    """
    Return all known models keyed by logical name.
    """
    return dict(_ensure_registry())


