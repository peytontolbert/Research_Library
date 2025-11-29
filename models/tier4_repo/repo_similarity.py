"""
Model 20: Repo-to-Repo Similarity Model (R6).
HF-backed contrastive scorer between repositories.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class RepoSimilarityModel(ContrastiveModel):
    """HF-backed repo similarity model (R6)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
