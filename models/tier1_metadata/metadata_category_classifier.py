"""
Model 2: Metadata Category Classifier (multi-label).
HF-backed classifier head over (title + abstract), aligned to the plan.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ClassifierModel


class MetadataCategoryClassifier(ClassifierModel):
    """HF-backed classifier for arXiv categories (M2) with keyword fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    @staticmethod
    def _keyword_rules(text: str) -> str:
        lower = text.lower()
        rules = [
            (("reinforcement learning", "policy", "q-learning"), "cs.LG"),
            (("neural network", "gradient descent", "deep learning"), "cs.LG"),
            (("language model", "nlp", "translation", "tokenization"), "cs.CL"),
            (("vision", "image", "segmentation", "detection"), "cs.CV"),
            (("graph", "gnn", "node", "edge"), "cs.SI"),
            (("bayesian", "posterior", "prior", "likelihood"), "stat.ML"),
        ]
        for kws, cat in rules:
            if any(kw in lower for kw in kws):
                return cat
        return "cs.AI"

    def predict(self, texts):
        if self._ensure_ready():
            return super().predict(texts)
        return [self._keyword_rules(str(t)) for t in texts]
