"""
Supplemental paper-QA model over full-text papers.
HF-backed generative QA over paper context + question.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GenerativeModel


class PaperQAModel(GenerativeModel):
    """HF-backed paper QA model (P5) with extractive-style fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def generate(self, batch):
        if self._ensure_ready():
            prompts = []
            for ex in batch:
                question = ex.get("question") or ex.get("query") or ""
                context = ex.get("context") or ex.get("paper_context") or ex.get("text") or ""
                prompts.append(f"QUESTION:\n{question}\nCONTEXT:\n{context}\nAnswer grounded in the paper.")
            return super().generate(prompts, max_new_tokens=256, temperature=0.0)

        outputs = []
        for ex in batch:
            question = str(ex.get("question") or ex.get("query") or "")
            context = str(ex.get("context") or ex.get("paper_context") or ex.get("text") or "")
            answer = context[:400].strip()
            outputs.append(f"Q: {question}\nA: {answer}")
        return outputs
