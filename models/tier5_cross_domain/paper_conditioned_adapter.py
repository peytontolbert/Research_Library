"""
Model 23: Paper-Conditioned Adapter (C3).
HF-backed adapter generator conditioned on paper embeddings.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import AdapterModel


class PaperConditionedAdapter(AdapterModel):
    """HF-backed PCA generator (C3)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
