from __future__ import annotations

"""
QA model runtime
----------------

This module owns loading and caching of LLMs (and optional LoRA adapters)
for QA skills. Adapters provide model configuration; this runtime turns
that into concrete model/tokenizer instances and runs generation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    try:
        # Optional LoRA support via PEFT; if unavailable, we fall back to
        # using the base model without adapters.
        from peft import PeftModel  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        PeftModel = None  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore
    PeftModel = None  # type: ignore

from .model_registry import get_model_config


@dataclass
class QAModelConfig:
    """
    Normalized QA model configuration derived from adapter metadata.
    """

    model_name: str
    model_id: str
    model_path: Optional[str]
    cache_dir: Optional[str]
    quantization: str
    # Optional LoRA adapters applied on top of the base model. For now we
    # support composing a "repo" adapter and a "task" adapter sequentially.
    repo_lora_path: Optional[str]
    task_lora_path: Optional[str]
    max_new_tokens: int
    temperature: float
    top_p: float
    infer_devices: List[int]


_MODEL_CACHE: Dict[Tuple[str, str, Optional[str]], Tuple[Any, Any]] = {}


def _normalize_optional_path(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _hf_cache_repo_dir(model_id: str, cache_dir: Optional[str]) -> Optional[Path]:
    model_id = str(model_id or "").strip()
    cache_dir = _normalize_optional_path(cache_dir)
    if not model_id or not cache_dir:
        return None
    return Path(cache_dir) / f"models--{model_id.replace('/', '--')}"


def _resolve_hf_cache_snapshot(model_id: str, cache_dir: Optional[str]) -> Optional[str]:
    repo_dir = _hf_cache_repo_dir(model_id, cache_dir)
    if repo_dir is None or not repo_dir.is_dir():
        return None

    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        try:
            ref = refs_main.read_text(encoding="utf-8").strip()
        except Exception:
            ref = ""
        if ref:
            snapshot = snapshots_dir / ref
            if snapshot.is_dir():
                return str(snapshot)

    snapshots = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if snapshots:
        return str(snapshots[0])
    return None


def _resolve_model_source(cfg: QAModelConfig) -> Tuple[str, bool]:
    explicit_model_path = _normalize_optional_path(cfg.model_path)
    if explicit_model_path:
        explicit_path = Path(explicit_model_path)
        if explicit_path.exists():
            return str(explicit_path), True
        # Support explicitly configured HF-style ids as a model_path override.
        return explicit_model_path, False

    cached_snapshot = _resolve_hf_cache_snapshot(cfg.model_id, cfg.cache_dir)
    if cached_snapshot:
        return cached_snapshot, True

    return cfg.model_id, False


def _infer_devices_from_cfg(cfg_devices: Optional[List[int]]) -> List[int]:
    if cfg_devices:
        return [int(d) for d in cfg_devices if isinstance(d, int)]
    # Default: prefer the first two GPUs (e.g., 2x3090) for QA inference.
    return [0, 1]


def get_model_config_from_adapter(qa_meta: Dict[str, Any]) -> QAModelConfig:
    """
    Build a QAModelConfig from adapter metadata, falling back to
    `model.yml` defaults (llama) where fields are missing.
    """
    # Adapter-provided fields (all optional)
    model_name = str(qa_meta.get("model_name") or "llama").strip()
    quantization = str(qa_meta.get("quantization") or "4bit").strip().lower()
    repo_lora_any = qa_meta.get("repo_lora_path")
    repo_lora_path = (
        str(repo_lora_any).strip() or None if isinstance(repo_lora_any, str) else None
    )
    task_lora_any = qa_meta.get("task_lora_path")
    task_lora_path = (
        str(task_lora_any).strip() or None if isinstance(task_lora_any, str) else None
    )
    max_new_tokens = int(qa_meta.get("max_new_tokens") or 256)
    temperature = float(qa_meta.get("temperature") or 0.1)
    top_p = float(qa_meta.get("top_p") or 0.95)
    infer_devices_any = qa_meta.get("infer_devices")
    infer_devices = _infer_devices_from_cfg(
        infer_devices_any if isinstance(infer_devices_any, list) else None
    )
    model_path = _normalize_optional_path(qa_meta.get("model_path"))

    # Merge with model.yml config.
    base_cfg = get_model_config(model_name)
    base_model_id = str(base_cfg.model_id or "").strip() if base_cfg is not None else ""
    model_id = qa_meta.get("model_id") or base_model_id
    model_id = str(model_id or "").strip()
    if model_path is None and base_cfg is not None:
        model_path = _normalize_optional_path(base_cfg.model_path)
    if model_path is None and model_name == "llama":
        model_path = _normalize_optional_path(os.environ.get("LLAMA_MODEL_PATH"))
    cache_dir_any = qa_meta.get("cache_dir") or (base_cfg.cache_dir if base_cfg else None)
    cache_dir = (
        str(cache_dir_any).strip() or None if isinstance(cache_dir_any, str) else None
    )

    # Older adapter metadata may still reference an earlier HF id even when a
    # newer canonical base model is already cached locally. Prefer the base
    # registry id when the adapter's id has no local snapshot but the base id does.
    if (
        model_path is None
        and cache_dir
        and base_model_id
        and model_id
        and model_id != base_model_id
    ):
        adapter_snapshot = _resolve_hf_cache_snapshot(model_id, cache_dir)
        base_snapshot = _resolve_hf_cache_snapshot(base_model_id, cache_dir)
        if adapter_snapshot is None and base_snapshot is not None:
            model_id = base_model_id

    if not model_id:
        raise RuntimeError("QA adapter does not specify a valid model_id and no default was found.")

    return QAModelConfig(
        model_name=model_name,
        model_id=model_id,
        model_path=model_path,
        cache_dir=cache_dir,
        quantization=quantization,
        repo_lora_path=repo_lora_path,
        task_lora_path=task_lora_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        infer_devices=infer_devices,
    )


def get_default_qa_base_config() -> QAModelConfig:
    """
    Build a default QA base-model config (for global preload, etc.).

    This uses the "llama" entry from model.yml when available and
    prefers 4-bit quantization by default (overridable via the
    LLM_QUANTIZATION environment variable).
    """
    base_cfg = get_model_config("llama")
    model_path = _normalize_optional_path(os.environ.get("LLAMA_MODEL_PATH"))
    if base_cfg is None:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        if model_path is None:
            model_path = None
        cache_dir = None
        model_name = "llama"
    else:
        model_id = base_cfg.model_id
        if model_path is None:
            model_path = _normalize_optional_path(base_cfg.model_path)
        cache_dir = base_cfg.cache_dir
        model_name = base_cfg.name

    quant_env = os.environ.get("LLM_QUANTIZATION", "").strip().lower()
    quantization = quant_env or "4bit"

    return QAModelConfig(
        model_name=model_name,
        model_id=model_id,
        model_path=model_path,
        cache_dir=cache_dir,
        quantization=quantization,
        repo_lora_path=None,
        task_lora_path=None,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        infer_devices=[0, 1],
    )


def _device_map_for(cfg: QAModelConfig) -> Optional[Dict[int, List[int]]]:
    """
    Compute a simple device map hint based on `infer_devices`.

    For now we let transformers/accelerate decide exact sharding via
    `device_map="auto"` and only influence which CUDA devices are
    visible via environment configuration. This function is a hook
    for future explicit mappings if needed.
    """
    # We intentionally return None here; callers can still set CUDA_VISIBLE_DEVICES
    # externally to restrict available GPUs. A more complex mapping can be added
    # later without changing the public interface.
    _ = cfg
    return None


def get_or_load_model(cfg: QAModelConfig) -> Tuple[Any, Any]:
    """
    Load (or retrieve from cache) the model/tokenizer pair for a QA adapter.
    """
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(
            "transformers/torch are not installed; cannot load QA LLM runtime."
        )

    model_source, use_local_files_only = _resolve_model_source(cfg)
    cache_key = (
        model_source,
        cfg.quantization,
        cfg.repo_lora_path,
        cfg.task_lora_path,
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Base tokenizer
    tok_kwargs: Dict[str, Any] = {}
    if cfg.cache_dir:
        tok_kwargs["cache_dir"] = cfg.cache_dir
    if use_local_files_only:
        tok_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_source, **tok_kwargs)

    # Base model with optional 4-bit quantization.
    model_kwargs: Dict[str, Any] = {}
    if cfg.cache_dir:
        model_kwargs["cache_dir"] = cfg.cache_dir
    if use_local_files_only:
        model_kwargs["local_files_only"] = True

    use_4bit = cfg.quantization in ("4bit", "bnb_4bit", "bnb-4bit")
    if use_4bit and BitsAndBytesConfig is not None and torch is not None:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            # Let transformers/accelerate decide placement across visible GPUs.
            model_kwargs["device_map"] = "auto"
        except Exception:
            # Fall back to non-quantized loading.
            pass

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

    # Optionally apply LoRA adapters on top of the base model. We support a
    # simple composition of up to two adapters: a repo-conditioned adapter
    # and a task-conditioned adapter. Both are treated as PEFT LoRA weights.
    if PeftModel is not None and (cfg.repo_lora_path or cfg.task_lora_path):
        try:
            for lora_path in (cfg.repo_lora_path, cfg.task_lora_path):
                if not lora_path:
                    continue
                model = PeftModel.from_pretrained(model, lora_path)  # type: ignore[assignment]
        except Exception:
            # If LoRA loading fails for any reason, we continue with the
            # base model; callers still get a working QA path.
            pass

    # If we're not using device_map (e.g., CPU or non-quantized single-GPU),
    # respect the first infer_device when CUDA is available.
    if "device_map" not in model_kwargs and torch is not None:
        if torch.cuda.is_available() and cfg.infer_devices:
            primary = int(cfg.infer_devices[0])
            model = model.to(f"cuda:{primary}")  # type: ignore[assignment]

    model.eval()
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def run_qa_generation(
    cfg: QAModelConfig,
    model: Any,
    tokenizer: Any,
    prompt: str,
) -> str:
    """
    Run a single-turn QA generation using the provided model/tokenizer.
    """
    if torch is None:
        raise RuntimeError("torch is not available; cannot run QA LLM inference.")

    inputs = tokenizer(prompt, return_tensors="pt")

    # If the model is on CUDA, move inputs to the same device.
    try:
        device = next(model.parameters()).device  # type: ignore[attr-defined]
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}  # type: ignore[assignment]
    except Exception:
        # Best-effort; if we cannot inspect parameters, rely on default device.
        pass

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(cfg.max_new_tokens),
        "do_sample": cfg.temperature > 0.0,
        "temperature": float(cfg.temperature),
        "top_p": float(cfg.top_p),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.strip()

