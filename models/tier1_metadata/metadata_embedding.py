"""
Model 1: Metadata Embedding Model (contrastive encoder).
Uses HF backbone + PEFT to produce embeddings over (title, abstract, categories, authors).
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class MetadataEmbeddingModel(ContrastiveModel):
    """HF-backed contrastive encoder for metadata (M1)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
