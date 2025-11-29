"""
Model 8: Abstract → Paper Keywords (A3).
HF-backed classifier/generative tagger for keywords.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ClassifierModel


class AbstractKeywordsModel(ClassifierModel):
    """HF-backed keyword predictor (A3) with frequency fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def predict(self, batch):
        if self._ensure_ready():
            texts = [str(ex.get("abstract") or ex.get("text") or "") for ex in batch]
            return super().predict(texts)

        import re
        outputs = []
        token_re = re.compile(r"[A-Za-z][A-Za-z0-9_\\-]+")
        for ex in batch:
            text = str(ex.get("abstract") or ex.get("text") or "")
            tokens = [t.lower() for t in token_re.findall(text) if len(t) >= 4]
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            keywords = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]]
            outputs.append(keywords)
        return outputs
