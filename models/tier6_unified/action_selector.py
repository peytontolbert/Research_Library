"""
Model 29: Action Selector (PAVU Loop) (U3).
HF-backed policy model that chooses actions given system state.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import PolicyModel


class ActionSelectorModel(PolicyModel):
    """HF-backed action selector (U3)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
