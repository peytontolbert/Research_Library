"""
Lightweight config loader for model experiments.
Reads JSON experiment specs under models/experiments/.
"""

import json
import os
from pathlib import Path

DEFAULT_HF_HOME = Path("/data/.cache/huggingface")
DEFAULT_TMP_DIR = Path("/data/tmp")


def load_experiment(path: str) -> dict:
    """Load an experiment JSON spec."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with p.open("r", encoding="ascii") as f:
        return json.load(f)


def ensure_runtime_cache_env(
    hf_home: str = str(DEFAULT_HF_HOME),
    tmp_dir: str = str(DEFAULT_TMP_DIR),
) -> dict:
    """
    Pin HF and temp caches to /data by default so large dataset/materialization
    work does not spill onto the root filesystem under /home.
    """
    hf_root = Path(os.environ.setdefault("HF_HOME", hf_home)).expanduser()
    hub_root = Path(os.environ.setdefault("HF_HUB_CACHE", str(hf_root / "hub"))).expanduser()
    datasets_root = Path(os.environ.setdefault("HF_DATASETS_CACHE", str(hf_root / "datasets"))).expanduser()
    xet_root = Path(os.environ.setdefault("HF_XET_CACHE", str(hf_root / "xet"))).expanduser()
    tmp_root = Path(os.environ.setdefault("TMPDIR", tmp_dir)).expanduser()
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_root))
    os.environ.setdefault("TMP", str(tmp_root))
    os.environ.setdefault("TEMP", str(tmp_root))

    for path in (hf_root, hub_root, datasets_root, xet_root, tmp_root):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "HF_HOME": str(hf_root),
        "HF_HUB_CACHE": str(hub_root),
        "HF_DATASETS_CACHE": str(datasets_root),
        "HF_XET_CACHE": str(xet_root),
        "TMPDIR": str(tmp_root),
    }


def validate_cache_dirs(config: dict, default_cache: str = "/data/checkpoints") -> dict:
    """Ensure cache/checkpoint dirs are set; fill defaults if missing."""
    ensure_runtime_cache_env()
    cache_override = os.environ.get("RESEARCH_LIBRARY_CACHE_DIR")
    config.setdefault("training", {})
    config.setdefault("dataset", {})
    config.setdefault("backbone", {})
    config["training"].setdefault("cache_dir", default_cache)
    config["training"].setdefault("checkpoint_dir", default_cache)
    config["dataset"].setdefault("cache_dir", default_cache)
    config["backbone"].setdefault("cache_dir", default_cache)
    if cache_override:
        config["training"]["cache_dir"] = cache_override
        config["dataset"]["cache_dir"] = cache_override
        config["backbone"]["cache_dir"] = cache_override
    return config
