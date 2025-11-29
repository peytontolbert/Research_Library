"""
Model 11: Full-Paper Language Model (P1).
Hybrid: uses HF generation when backbone/tokenizer are present; otherwise
returns a deterministic continuation to keep behavior aligned with PLAN.md.
"""

from typing import Any, Dict, Optional, Sequence, List
import textwrap

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from models.shared.modeling import GenerativeModel


class FullPaperLanguageModel(GenerativeModel):
    """HF-backed full-paper LM with heuristic fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None, max_new_tokens: int = 64) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _extract_text(example: Any) -> str:
        if isinstance(example, str):
            return example
        if isinstance(example, dict):
            if example.get("text") is not None:
                return str(example.get("text") or "")
            tokens = example.get("tokens")
            if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
                pieces: List[str] = []
                for tok in tokens:
                    if isinstance(tok, str):
                        pieces.append(tok)
                    elif isinstance(tok, dict) and "text" in tok:
                        pieces.append(str(tok["text"]))
                if pieces:
                    return " ".join(pieces)
        return ""

    def generate(self, batch: Sequence[Any]) -> List[str]:
        outputs: List[str] = []
        if self._ensure_ready() and torch is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.backbone.to(device)
            self.backbone.eval()
            with torch.no_grad():
                for ex in batch:
                    text = self._extract_text(ex)
                    prompt = textwrap.shorten(text, width=4096, placeholder=" ...")
                    enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    gen = self.backbone.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    out = self.tokenizer.decode(gen[0], skip_special_tokens=True)
                    outputs.append(out[len(prompt):].strip() or out)
                return outputs

        # Fallback deterministic continuation.
        for ex in batch:
            text = self._extract_text(ex)
            words = [w for w in text.split() if w]
            tail = words[-self.max_new_tokens :] if words else []
            outputs.append(" ".join(tail))
        return outputs
