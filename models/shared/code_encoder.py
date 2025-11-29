"""
Lightweight code/text encoder helper for repo/file models.

This is intentionally optional: it can run with a provided HF tokenizer+model
or fall back to a simple bag-of-words embedding when those are absent.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _mean_pool(hidden_states):
    if hidden_states is None or hidden_states.ndim != 3:
        return None
    return hidden_states.mean(dim=1)


class CodeEncoder:
    """Small wrapper to get embeddings from an HF model or fallback."""

    def __init__(self, tokenizer: Any = None, model: Any = None, device: Optional[str] = None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def encode(self, texts: List[str]):
        """Return a list of embeddings (torch tensors or simple lists)."""
        if self.tokenizer is None or self.model is None or torch is None:
            # Fallback: simple length-based vector.
            return [[float(len(t))] for t in texts]

        self.model.eval()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None
            pooled = _mean_pool(hidden)
            if pooled is None:
                return [[float(len(t))] for t in texts]
            return pooled.cpu()
