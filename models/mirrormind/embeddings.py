"""
Embedding utilities for MirrorMind episodic indexing.
Uses sentence-transformers or HF AutoModel as available; falls back to length.
"""

from __future__ import annotations

import os
from typing import Sequence, Optional, Any, List


class TextEmbedder:
    """Optional text embedder using sentence-transformers or HF AutoModel."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "/data/checkpoints") -> None:
        env_model = os.getenv("MIRRORMIND_EMBED_MODEL")
        if env_model:
            model_name = env_model
        self.model = None
        self.tokenizer = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
            return
        except Exception:
            pass
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.eval()
            self._torch = torch
        except Exception:
            self.model = None
            self.tokenizer = None
            self._torch = None

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self.model is None:
            return [[float(len(t))] for t in texts]
        # sentence-transformers path
        if hasattr(self.model, "encode"):
            try:
                return self.model.encode(list(texts), normalize_embeddings=True).tolist()
            except Exception:
                return [[float(len(t))] for t in texts]
        # HF AutoModel path
        if self.tokenizer is not None and getattr(self, "_torch", None) is not None:
            torch = self._torch  # type: ignore
            with torch.no_grad():
                enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt")
                out = self.model(**enc)
                hidden = out.last_hidden_state
                pooled = hidden.mean(dim=1)
                norms = torch.norm(pooled, dim=1, keepdim=True) + 1e-8
                pooled = pooled / norms
                return pooled.cpu().tolist()
        return [[float(len(t))] for t in texts]
