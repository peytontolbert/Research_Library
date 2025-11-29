"""
Model 24: Repo-Conditioned Adapter (C4).
HF-backed adapter generator conditioned on repo embeddings.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import AdapterModel


class RepoConditionedAdapter(AdapterModel):
    """HF-backed RCA generator (C4)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
