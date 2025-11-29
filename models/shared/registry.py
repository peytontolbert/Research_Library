"""
Registry mapping model IDs to HF-backed archetype wrappers.
All model IDs resolve to the shared HF wrapper based on `shared/archetypes.py`,
ensuring every tier is hooked into the same tokenizer/backbone/PEFT path.
"""

from typing import Any

from models.shared.modeling import build_hf_model
from models.shared.archetypes import get_archetype


def build_model(model_id: str, tokenizer: Any = None, backbone: Any = None, config: Any = None):
    """Instantiate HF-backed wrapper for a model_id."""
    archetype = get_archetype(model_id)
    hf_model = None
    try:
        hf_model = build_hf_model(model_id, tokenizer=tokenizer, backbone=backbone, config=config)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize HF-backed model for {model_id}: {exc}") from exc
    if hf_model is None or archetype is None:
        raise KeyError(f"Unknown or unsupported model_id: {model_id}")
    return hf_model
