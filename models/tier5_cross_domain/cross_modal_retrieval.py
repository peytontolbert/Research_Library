"""
Model 26: Cross-Modal Retrieval (PDF ↔ Repo) (C6).
HF-backed contrastive encoder for shared paper/code embedding space.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class CrossModalRetrievalModel(ContrastiveModel):
    """HF-backed cross-modal retrieval model (C6)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
