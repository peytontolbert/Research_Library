"""
Model 7: Abstract → Method Summary (A2).
HF-backed generative summarizer over abstracts.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GenerativeModel


class AbstractMethodSummaryModel(GenerativeModel):
    """HF-backed method summarizer (A2) with sentence-truncation fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def generate(self, batch):
        if self._ensure_ready():
            prompts = [f"ABSTRACT:\n{str(ex.get('abstract') or ex.get('text') or '')}\nSummarize the method in 2-5 sentences." for ex in batch]
            return super().generate(prompts, max_new_tokens=200, temperature=0.1)

        outputs = []
        for ex in batch:
            abstract = str(ex.get("abstract") or ex.get("text") or "")
            sents = [p.strip() for p in abstract.replace("\n", " ").split(".") if p.strip()]
            summary = ". ".join(sents[:3])
            if summary and not summary.endswith("."):
                summary += "."
            outputs.append(summary or abstract[:280])
        return outputs
