"""
Model 25: Adapter Fusion Model (C5).
HF-backed adapter fusion over PCA/RCA deltas.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import AdapterModel


class AdapterFusionModel(AdapterModel):
    """HF-backed adapter fusion model (C5)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
