"""
Model 31: Self-Play Reward Model (S2).
HF-backed classifier to score trajectories/outputs.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ClassifierModel


class SelfPlayRewardModel(ClassifierModel):
    """HF-backed reward classifier (S2)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
