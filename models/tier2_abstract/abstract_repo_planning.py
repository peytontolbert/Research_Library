"""
Model 9: Abstract → Repo-Planning (A4).
HF-backed classifier to select repo modules/classes given an abstract.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ClassifierModel


class AbstractRepoPlanningModel(ClassifierModel):
    """HF-backed repo planning classifier (A4) with overlap fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def predict(self, batch):
        if self._ensure_ready():
            texts = [f"ABSTRACT:\n{ex.get('abstract')}\nCOMPONENTS:\n{ex.get('components')}" for ex in batch]
            return super().predict(texts)

        outputs = []
        for ex in batch:
            abstract = str(ex.get("abstract") or ex.get("text") or "")
            components = ex.get("components") or ex.get("repo_components") or []
            if not isinstance(components, list):
                components = list(components)
            abs_tokens = set(abstract.lower().replace("\n", " ").split())
            scored = []
            for comp in components:
                comp_tokens = set(str(comp).lower().replace("\n", " ").split())
                union = len(abs_tokens | comp_tokens) or 1
                score = len(abs_tokens & comp_tokens) / union
                scored.append((comp, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            outputs.append([c for c, _ in scored[:5]])
        return outputs
