"""
Model 22: Repo-Paper Alignment Model (C2).
HF-backed contrastive scorer aligning paper and repo embeddings.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class RepoPaperAlignmentModel(ContrastiveModel):
    """HF-backed repo-paper alignment model (C2) with text overlap fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def predict(self, batch):
        if self._ensure_ready():
            pairs = [{"text_a": str(ex.get("paper") or ex.get("abstract") or ""), "text_b": str(ex.get("repo") or ex.get("code") or "")} for ex in batch]
            return self.score_pairs(pairs)
        scores = []
        for ex in batch:
            a = set(str(ex.get("paper") or ex.get("abstract") or "").lower().split())
            b = set(str(ex.get("repo") or ex.get("code") or "").lower().split())
            score = len(a & b) / float(len(a | b) or 1)
            scores.append(score)
        return scores
