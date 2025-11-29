"""
Model 28: World/Planner Adapter (U2).
HF-backed generative planner producing goal-directed action sequences.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GenerativeModel


class WorldPlannerAdapter(GenerativeModel):
    """HF-backed planner model (U2)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
