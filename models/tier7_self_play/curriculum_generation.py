"""
Model 32: Curriculum Generation Model (S3).
HF-backed policy/generative model to propose new tasks and curricula.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import PolicyModel


class CurriculumGenerationModel(PolicyModel):
    """HF-backed curriculum generator (S3)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
