"""
HF-backed model wrappers aligned with the 32-model plan.

Each archetype (contrastive, classifier, generative, adapter, policy)
uses a shared interface so we can swap in HF/PEFT backbones while
keeping lightweight heuristics as a fallback when dependencies are
missing. These classes are intentionally minimal; the heavy lifting
is delegated to `build_backbone` and HF Trainer in `training.py`.
"""

from typing import Any, Dict, List, Optional, Sequence
import textwrap

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


class BaseHFModel:
    """Base wrapper that holds tokenizer/backbone references."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.config = config or {}

    def _ensure_ready(self) -> bool:
        return self.tokenizer is not None and self.backbone is not None


class ContrastiveModel(BaseHFModel):
    """Encode text pairs and compute similarities."""

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        if not self._ensure_ready() or torch is None:
            # Fallback: length-based embedding for determinism.
            return [[float(len(t))] for t in texts]
        self.backbone.eval()
        max_len = self.config.get("dataset", {}).get("tokenization", {}).get("max_source_tokens", 256)
        with torch.no_grad():
            enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            enc = {k: v.to(device) for k, v in enc.items()}
            self.backbone.to(device)
            out = self.backbone(**enc)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else None
            if hidden is None:
                return [[float(len(t))] for t in texts]
            pooled = hidden.mean(dim=1)
            return pooled.detach().cpu().tolist()

    def score_pairs(self, pairs: Sequence[Dict[str, str]]) -> List[float]:
        """Compute cosine similarity for text pairs."""
        sims: List[float] = []
        if not pairs:
            return sims
        a_texts = [p.get("text_a", "") for p in pairs]
        b_texts = [p.get("text_b", "") for p in pairs]
        embeds_a = self.encode(a_texts)
        embeds_b = self.encode(b_texts)
        if torch is None or not embeds_a or not embeds_b:
            return [0.0 for _ in pairs]
        a = torch.tensor(embeds_a)
        b = torch.tensor(embeds_b)
        sims_tensor = torch.nn.functional.cosine_similarity(a, b)
        return sims_tensor.tolist()


class GraphModel(ContrastiveModel):
    """Graph/link-prediction friendly wrapper using text projections as nodes."""

    def score_edges(self, edges: Sequence[Dict[str, str]]) -> List[float]:
        """Edges are dicts with 'src' and 'dst' textual descriptions."""
        if not edges:
            return []
        pairs = [{"text_a": e.get("src", ""), "text_b": e.get("dst", "")} for e in edges]
        return self.score_pairs(pairs)


class ClassifierModel(BaseHFModel):
    """Simple classifier wrapper using an HF backbone with a classification head."""

    def predict(self, texts: Sequence[str]) -> List[str]:
        if not self._ensure_ready() or torch is None:
            return ["label_0" for _ in texts]
        max_len = self.config.get("dataset", {}).get("tokenization", {}).get("max_source_tokens", 256)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone.to(device)
        self.backbone.eval()
        id2label = getattr(getattr(self.backbone, "config", None), "id2label", None) or {}
        outputs: List[str] = []
        with torch.no_grad():
            for chunk_start in range(0, len(texts), 8):
                chunk = texts[chunk_start : chunk_start + 8]
                enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = self.backbone(**enc).logits
                preds = logits.argmax(dim=-1).cpu().tolist()
                for p in preds:
                    outputs.append(id2label.get(p, f"label_{p}"))
        return outputs


class GenerativeModel(BaseHFModel):
    """Autoregressive/seq2seq generation wrapper."""

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 64, temperature: float = 0.7) -> List[str]:
        if not self._ensure_ready() or torch is None:
            return [textwrap.shorten(p, width=120, placeholder=" ...") for p in prompts]
        # Handle 8-bit/accelerate device maps: avoid calling .to on quantized modules.
        device_map = getattr(self.backbone, "hf_device_map", None)
        is_8bit = bool(getattr(self.backbone, "is_loaded_in_8bit", False))
        device = None
        if device_map or is_8bit:
            # Pick the first non-cpu device from the device map if present.
            if device_map:
                devs = [d for d in device_map.values() if isinstance(d, str)]
                # Preserve order but de-dup.
                seen = set()
                ordered = []
                for d in devs:
                    if d in seen:
                        continue
                    seen.add(d)
                    ordered.append(d)
                for d in ordered:
                    if d != "cpu":
                        device = d
                        break
                if device is None and ordered:
                    device = ordered[0]
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.backbone.eval()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.backbone.to(device)
            self.backbone.eval()
        outputs: List[str] = []
        with torch.no_grad():
            for chunk_start in range(0, len(prompts), 4):
                chunk = prompts[chunk_start : chunk_start + 4]
                enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()} if device else enc
                gen = self.backbone.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
                outputs.extend(texts)
        return outputs


class AdapterModel(GenerativeModel):
    """Adapter-focused model. Behaves generatively but marks adapter metadata."""

    def generate_adapter(self, prompts: Sequence[str], max_new_tokens: int = 64) -> List[str]:
        outputs = super().generate(prompts, max_new_tokens=max_new_tokens, temperature=0.1)
        return [f"[ADAPTER_DELTA]\n{o}" for o in outputs]


class PolicyModel(GenerativeModel):
    """Policy head wrapper for self-play / action selection."""

    def act(self, states: Sequence[Dict[str, Any]]) -> List[str]:
        prompts = []
        for st in states:
            parts = [f"STATE::{k}={v}" for k, v in st.items() if v]
            prompts.append("\n".join(parts))
        actions = super().generate(prompts, max_new_tokens=32, temperature=0.2)
        return [textwrap.shorten(a.strip(), width=160, placeholder=" ...") for a in actions]


def build_hf_model(model_id: str, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None):
    """Factory returning the HF-backed model wrapper per archetype."""
    from models.shared.archetypes import get_archetype

    archetype = (get_archetype(model_id) or {}).get("archetype")
    model_kwargs = {"tokenizer": tokenizer, "backbone": backbone, "config": config}
    if archetype == "contrastive":
        return ContrastiveModel(**model_kwargs)
    if archetype == "graph":
        return GraphModel(**model_kwargs)
    if archetype == "classifier":
        return ClassifierModel(**model_kwargs)
    if archetype in {"generative", "adapter"}:
        return AdapterModel(**model_kwargs) if archetype == "adapter" else GenerativeModel(**model_kwargs)
    if archetype in {"policy"}:
        return PolicyModel(**model_kwargs)
    return None
