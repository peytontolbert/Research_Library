"""
Model 12: Section → Algorithm (P2).
Hybrid: HF generation when available; deterministic step list fallback for PLAN.md alignment.
"""

from typing import Any, Dict, Optional, Sequence, List

from models.shared.modeling import GenerativeModel


class SectionToAlgorithmModel(GenerativeModel):
    """HF-backed pseudo-code generator with rule-based fallback (P2)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None, max_steps: int = 10) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
        self.max_steps = max_steps

    @staticmethod
    def _extract_text(example: Dict[str, Any]) -> str:
        return str(
            example.get("methods")
            or example.get("section")
            or example.get("text")
            or ""
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        spaced = text.replace("\n", " ")
        parts = [p.strip() for p in spaced.split(".") if p.strip()]
        return parts

    def generate(self, batch: Sequence[Dict[str, Any]]) -> List[str]:
        # Use HF path when available via parent.
        if self._ensure_ready():
            prompts = []
            for ex in batch:
                text = self._extract_text(ex)
                prompts.append(f"METHOD SECTION:\n{text}\nProduce pseudo-code steps.")
            return super().generate(prompts, max_new_tokens=256, temperature=0.0)

        # Fallback: numbered step list.
        results: List[str] = []
        for ex in batch:
            text = self._extract_text(ex)
            sentences = self._split_sentences(text)
            if not sentences:
                results.append("")
                continue
            lines: List[str] = []
            for i, sent in enumerate(sentences[: self.max_steps], start=1):
                lines.append(f"step_{i}: {sent}")
            results.append("\n".join(lines))
        return results
