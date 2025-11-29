"""
Model 15: Repo Embedding Model (R1, RCA base).
HF-backed contrastive encoder over repository content.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class RepoEmbeddingModel(ContrastiveModel):
    """HF-backed repo embedding model (R1)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
