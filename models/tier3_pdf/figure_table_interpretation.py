"""
Model 14: Figure/Table Interpretation (P4).
Hybrid: HF generation when available; text-only figure/table summarizer fallback.
"""

from typing import Any, Dict, Optional, Sequence, List

from models.shared.modeling import GenerativeModel


class FigureTableInterpretationModel(GenerativeModel):
    """HF-backed figure/table interpreter with heuristic fallback (P4)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    @staticmethod
    def _extract_text(example: Dict[str, Any]) -> str:
        return str(
            example.get("caption")
            or example.get("table")
            or example.get("text")
            or example.get("figure")
            or ""
        )

    @staticmethod
    def _guess_kind(text: str) -> str:
        lower = text.lower()
        if "table" in lower or "|" in text or "\t" in text:
            return "table"
        if "figure" in lower or "fig." in lower:
            return "figure"
        digits = sum(ch.isdigit() for ch in text)
        letters = sum(ch.isalpha() for ch in text)
        if digits > 0 and digits >= letters:
            return "table"
        return "figure"

    @staticmethod
    def _summarize(text: str) -> str:
        normalized = " ".join(text.strip().split())
        if len(normalized) <= 240:
            return normalized
        snippet = normalized[:237]
        last_space = snippet.rfind(" ")
        if last_space > 0:
            snippet = snippet[:last_space]
        return snippet + "..."

    def predict(self, batch: Sequence[Dict[str, Any]]) -> List[str]:
        if self._ensure_ready():
            prompts = []
            for ex in batch:
                text = self._extract_text(ex)
                prompts.append(f"FIGURE/TABLE CONTENT:\n{text}\nDescribe the content and any notable values.")
            return super().generate(prompts, max_new_tokens=200, temperature=0.2)

        outputs: List[str] = []
        for ex in batch:
            raw = self._extract_text(ex)
            if not raw.strip():
                outputs.append("No figure or table text provided.")
                continue
            kind = self._guess_kind(raw)
            summary = self._summarize(raw)
            outputs.append(f"This {kind} appears to show: {summary}")
        return outputs
