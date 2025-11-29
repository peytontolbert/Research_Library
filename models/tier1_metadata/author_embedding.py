"""
Model 5: Author Embedding / Community Model.
HF-backed contrastive encoder over author nodes and neighborhoods.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class AuthorEmbeddingModel(ContrastiveModel):
    """HF-backed author embedding model (M5)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
