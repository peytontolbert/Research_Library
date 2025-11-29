"""
Lightweight config loader for model experiments.
Reads JSON experiment specs under models/experiments/.
"""

import json
from pathlib import Path


def load_experiment(path: str) -> dict:
    """Load an experiment JSON spec."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with p.open("r", encoding="ascii") as f:
        return json.load(f)


def validate_cache_dirs(config: dict, default_cache: str = "/data/checkpoints") -> dict:
    """Ensure cache/checkpoint dirs are set; fill defaults if missing."""
    config.setdefault("training", {})
    config.setdefault("dataset", {})
    config.setdefault("backbone", {})
    config["training"].setdefault("cache_dir", default_cache)
    config["training"].setdefault("checkpoint_dir", default_cache)
    config["dataset"].setdefault("cache_dir", default_cache)
    config["backbone"].setdefault("cache_dir", default_cache)
    return config
