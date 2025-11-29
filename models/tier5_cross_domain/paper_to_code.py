"""
Model 21: Paper → Code Generator (C1).
HF-backed generative model translating methods/pseudocode to runnable code.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GenerativeModel


class PaperToCodeModel(GenerativeModel):
    """HF-backed paper-to-code generator (C1) with echo fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def generate(self, batch):
        if self._ensure_ready():
            prompts = []
            for ex in batch:
                method = ex.get("method_section_text") or ex.get("text") or ""
                prompts.append(f"METHOD/PSEUDOCODE:\n{method}\nGenerate runnable Python/PyTorch code.")
            return super().generate(prompts, max_new_tokens=400, temperature=0.2)
        return [f"# code sketch for:\n{ex.get('method_section_text') or ex.get('text') or ''}" for ex in batch]
