"""
Model 6: Abstract → Code-Relevance (A1).
HF-backed contrastive encoder for abstract ↔ repo relevance.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ContrastiveModel


class AbstractCodeRelevanceModel(ContrastiveModel):
    """HF-backed relevance scorer (A1) with lexical fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def predict(self, batch):
        """Return relevance scores 0–1 for abstract↔code pairs."""
        if self._ensure_ready():
            pairs = [{"text_a": str(ex.get("abstract") or ex.get("text") or ""), "text_b": str(ex.get("code") or ex.get("snippet") or ex.get("docstring") or "")} for ex in batch]
            return self.score_pairs(pairs)

        scores = []
        for ex in batch:
            abstract = str(ex.get("abstract") or ex.get("text") or "")
            code = str(ex.get("code") or ex.get("snippet") or ex.get("docstring") or "")
            a_tokens = set(abstract.lower().replace("\n", " ").split())
            c_tokens = set(code.lower().replace("\n", " ").split())
            if not a_tokens or not c_tokens:
                scores.append(0.0)
                continue
            jaccard = len(a_tokens & c_tokens) / float(len(a_tokens | c_tokens) or 1)
            scores.append(max(0.0, min(1.0, jaccard)))
        return scores
