"""
Configuration helpers for pipeline command templates.
Allows overriding default HF/CI commands via environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence, Dict


def _parse_env_command(env_var: str) -> List[str]:
    val = os.getenv(env_var)
    if not val:
        return []
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            return parsed.split()
    except Exception:
        return val.split()
    return []


def resolve_command(env_var: str, default: Sequence[str], overrides: Optional[List[str]] = None) -> List[str]:
    """Return command list from config overrides or ENV or default."""
    if overrides:
        return list(overrides)
    cmd = _parse_env_command(env_var)
    if cmd:
        return cmd
    return list(default)


def load_pipeline_config(env_var: str = "PIANO_PIPELINE_CONFIG") -> Dict[str, List[str]]:
    """Load pipeline config from a JSON file if specified in env."""
    path = os.getenv(env_var)
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if isinstance(v, list)}
    except Exception:
        return {}
    return {}
