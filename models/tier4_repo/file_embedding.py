"""
Model 16: File-Level Embedding Model (R2).
HF-backed contrastive encoder over individual source files.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class FileEmbeddingModel(ContrastiveModel):
    """HF-backed file embedding model (R2)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
