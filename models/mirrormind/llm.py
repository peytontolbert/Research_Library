"""
LLM loader for MirrorMind/PIANO using the default Llama 1B checkpoint.
Always uses cache_dir=/data/checkpoints and avoids network fetches when offline.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import logging

from models.shared.config import validate_cache_dirs
from models.shared.training import build_tokenizer, build_backbone, apply_peft_if_needed
from models.shared.modeling import build_hf_model
from modules.model_registry import get_model_config

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_CACHE_DIR = "/data/checkpoints"

logger = logging.getLogger(__name__)


def default_llama1b_config(model_id: str = "U1", logical_name: str = "llama") -> Dict[str, Any]:
    """
    Return a minimal config for building an LLM, optionally backed by model.yml.

    If a model with `logical_name` is present in modules/model_registry.py, we
    use its `model_id` and `cache_dir` as the backing HF model; otherwise we
    fall back to DEFAULT_BASE_MODEL / DEFAULT_CACHE_DIR.
    """
    base_model = DEFAULT_BASE_MODEL
    cache_dir = DEFAULT_CACHE_DIR
    try:
        cfg = get_model_config(logical_name)
        if cfg is not None and cfg.model_id:
            base_model = cfg.model_id
        if cfg is not None and cfg.cache_dir:
            cache_dir = cfg.cache_dir
    except Exception:
        # Registry is best-effort; keep hard-coded defaults when unavailable.
        pass

    return {
        "model_id": model_id,
        "backbone": {
            "base_model": base_model,
            "cache_dir": cache_dir,
            "type": "decoder",
            "adapter_type": "none",
            "load_in_8bit": True,
            "device_map": "cuda:0",
        },
        "training": {
            "use_hf_trainer": True,
            "finetune_strategy": "none",
            "cache_dir": cache_dir,
            "checkpoint_dir": cache_dir,
        },
        "dataset": {
            "tokenization": {"tokenizer_name": base_model, "max_source_tokens": 512},
            "cache_dir": cache_dir,
        },
    }


def build_llm(model_id: str = "U1", logical_name: str = "llama") -> Tuple[Any, Any, Any]:
    """
    Build tokenizer, backbone, and HF wrapper for the given model_id using
    the model specified in model.yml (logical_name), falling back to the
    default Llama 1B config when necessary.

    Returns (model_wrapper, tokenizer, backbone).
    """
    logger.info("Building LLM for %s using logical_name=%s", model_id, logical_name)
    config = validate_cache_dirs(default_llama1b_config(model_id=model_id, logical_name=logical_name), default_cache=DEFAULT_CACHE_DIR)
    tokenizer = build_tokenizer(config)
    backbone = build_backbone(config)
    backbone = apply_peft_if_needed(backbone, config)
    model = build_hf_model(model_id, tokenizer=tokenizer, backbone=backbone, config=config)
    return model, tokenizer, backbone


def safe_build_llm(model_id: str = "U1", logical_name: str = "llama") -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Non-raising variant for environments without HF weights."""
    try:
        llm_tuple = build_llm(model_id=model_id, logical_name=logical_name)
        logger.info("LLM ready for %s", model_id)
        return llm_tuple
    except Exception as exc:
        logger.warning("LLM unavailable for %s (fallback to stub): %s", model_id, exc)
        return None, None, None
