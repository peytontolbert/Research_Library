from __future__ import annotations

"""
Lightweight embedding utilities for the repository library.

This module intentionally keeps a very small surface area:

    - It exposes a single `embed_texts` helper that:
      - Lazily loads an embedding model via `transformers`.
      - Uses mean-pooled last hidden states as sentence embeddings.
      - Optionally batches inputs to avoid OOM for large corpora.
- The concrete model is configurable via the `EMBED_MODEL_PATH`
  environment variable and defaults to a compact sentence-transformers
  style model.

The goal is to provide a simple, local embedding runtime that can be
used both by:

- Offline index builders (e.g., QA skill build).
- Online retrieval (e.g., query-time similarity search).
"""

import os
from typing import Iterable, List, Optional

try:  # Optional heavy deps; callers get a clear error if missing.
    import torch  # type: ignore
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore

import numpy as np

from .model_registry import get_model_config  # type: ignore


_EMBED_MODEL = None
_EMBED_TOKENIZER = None
_EMBED_MODEL_DEVICE: Optional[str] = None


def _resolve_embed_device(explicit_device: str | None = None) -> str:
    device = str(explicit_device or os.environ.get("EMBED_DEVICE") or "").strip()
    if device:
        return device
    if torch is not None and torch.cuda.is_available():  # pragma: no cover - runtime env
        return "cuda"
    return "cpu"


def _load_embed_model(*, device: str | None = None):
    """
    Load (and cache) the embedding model.

    By default this uses a small sentence-transformers style model which
    should be significantly lighter than the main QA LLM:

        EMBED_MODEL_PATH (env) or
        "sentence-transformers/all-MiniLM-L6-v2"
    """
    global _EMBED_MODEL, _EMBED_TOKENIZER, _EMBED_MODEL_DEVICE

    resolved_device = _resolve_embed_device(device)

    if _EMBED_MODEL is not None and _EMBED_TOKENIZER is not None:
        if _EMBED_MODEL_DEVICE != resolved_device:
            _EMBED_MODEL = _EMBED_MODEL.to(resolved_device)  # type: ignore[assignment]
            _EMBED_MODEL_DEVICE = resolved_device
        return _EMBED_MODEL, _EMBED_TOKENIZER

    if AutoModel is None or AutoTokenizer is None:
        raise RuntimeError(
            "transformers/torch are not installed; cannot load embedding model."
        )

    # Prefer a logical model name from model.yml so that embedding models
    # are configured centrally. Fall back to the previous EMBED_MODEL_PATH
    # behaviour if the registry entry is missing.
    model_name = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    base_cfg = get_model_config(model_name)

    if base_cfg is not None:
        model_id = base_cfg.model_id
        cache_dir = base_cfg.cache_dir
    else:
        model_id = os.environ.get(
            "EMBED_MODEL_PATH",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        cache_dir = None

    tok_kwargs = {}
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

    model_kwargs = {}
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    model = AutoModel.from_pretrained(model_id, **model_kwargs)

    model = model.to(resolved_device)  # type: ignore[assignment]

    model.eval()
    _EMBED_MODEL = model
    _EMBED_TOKENIZER = tokenizer
    _EMBED_MODEL_DEVICE = resolved_device
    return _EMBED_MODEL, _EMBED_TOKENIZER


def embed_texts(
    texts: Iterable[str],
    batch_size: int | None = None,
    *,
    device: str | None = None,
) -> np.ndarray:
    """
    Compute vector embeddings for a sequence of texts.

    Returns:
        A numpy array of shape (N, D) with float32 embeddings.
    """
    texts_list: List[str] = [str(t) for t in texts]
    if not texts_list:
        return np.zeros((0, 0), dtype="float32")

    resolved_device = _resolve_embed_device(device)
    model, tokenizer = _load_embed_model(device=resolved_device)
    if torch is None:
        raise RuntimeError("torch is not available; cannot run embedding inference.")

    bs = int(batch_size) if batch_size else None
    outputs_all: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts_list), bs or len(texts_list)):
            chunk = texts_list[start : (start + (bs or len(texts_list)))]
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(resolved_device) for k, v in inputs.items()}  # type: ignore[assignment]
            model = model.to(resolved_device)  # type: ignore[assignment]

            outputs = model(**inputs)
            # Mean-pool last_hidden_state using the attention mask.
            last_hidden = outputs.last_hidden_state  # type: ignore[attr-defined]
            mask = inputs.get("attention_mask")
            if mask is None:
                pooled = last_hidden.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1).type_as(last_hidden)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            outputs_all.append(pooled.cpu().numpy().astype("float32"))

    if not outputs_all:
        return np.zeros((0, 0), dtype="float32")
    return np.concatenate(outputs_all, axis=0)
