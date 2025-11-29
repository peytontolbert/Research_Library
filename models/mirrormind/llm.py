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

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_CACHE_DIR = "/data/checkpoints"

logger = logging.getLogger(__name__)


def default_llama1b_config(model_id: str = "U1") -> Dict[str, Any]:
    """Return a minimal config for Llama 1B cached under /data/checkpoints."""
    return {
        "model_id": model_id,
        "backbone": {
            "base_model": DEFAULT_BASE_MODEL,
            "cache_dir": DEFAULT_CACHE_DIR,
            "type": "decoder",
            "adapter_type": "none",
            "load_in_8bit": True,
            "device_map": "cuda:0",
        },
        "training": {
            "use_hf_trainer": True,
            "finetune_strategy": "none",
            "cache_dir": DEFAULT_CACHE_DIR,
            "checkpoint_dir": DEFAULT_CACHE_DIR,
        },
        "dataset": {
            "tokenization": {"tokenizer_name": DEFAULT_BASE_MODEL, "max_source_tokens": 512},
            "cache_dir": DEFAULT_CACHE_DIR,
        },
    }


def build_llm(model_id: str = "U1") -> Tuple[Any, Any, Any]:
    """
    Build tokenizer, backbone, and HF wrapper for the given model_id using Llama 1B.
    Returns (model_wrapper, tokenizer, backbone).
    """
    logger.info("Building LLM for %s using %s (cache=%s)", model_id, DEFAULT_BASE_MODEL, DEFAULT_CACHE_DIR)
    config = validate_cache_dirs(default_llama1b_config(model_id=model_id), default_cache=DEFAULT_CACHE_DIR)
    tokenizer = build_tokenizer(config)
    backbone = build_backbone(config)
    backbone = apply_peft_if_needed(backbone, config)
    model = build_hf_model(model_id, tokenizer=tokenizer, backbone=backbone, config=config)
    return model, tokenizer, backbone


def safe_build_llm(model_id: str = "U1") -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Non-raising variant for environments without HF weights."""
    try:
        llm_tuple = build_llm(model_id=model_id)
        logger.info("LLM ready for %s", model_id)
        return llm_tuple
    except Exception as exc:
        logger.warning("LLM unavailable for %s (fallback to stub): %s", model_id, exc)
        return None, None, None
