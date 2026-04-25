"""
Training scaffolding that can be wired to HF/PEFT/bitsandbytes.
This remains lightweight and dependency-optional; actual HF imports are lazy.
"""

import os
import sys
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from models.shared.data import (
    _attach_teacher_embeddings,
    _build_metadata_embedding_samples,
    build_dataset,
    _build_fulltext_samples,
    _build_paper_keyword_samples,
    _build_paper_qa_samples,
    _build_paper_fulltext_embedding_samples,
    _build_paper_retrieval_samples,
    _build_paper_sentence_embedding_samples,
    _chunk_text,
    _compose_full_paper_text,
    _format_text_from_entry,
    _load_teacher_embedding_lookup,
    _normalize_paper_row,
    _paper_keyword_target,
    _paper_method_summary_target,
    _paper_parquet_paths,
    _paper_row_year,
)
from models.shared.config import validate_cache_dirs
from models.shared.archetypes import get_archetype
from models.shared.code_encoder import CodeEncoder

BitsAndBytesConfig = None  # lazy import to avoid bitsandbytes errors when unavailable

# Prefer the less-fragmenting CUDA allocator before torch initializes. This is
# particularly important for quantized + PEFT training where eval can otherwise
# fail despite several GiB being reserved but unallocated.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

if torch is not None and not hasattr(torch, "Tensor"):  # pragma: no cover - broken optional dependency
    sys.modules.pop("torch", None)
    torch = None


@contextmanager
def _datasets_fingerprint_compatible_env():
    """
    The `datasets` package inspects `sys.modules["torch"]` while fingerprinting
    configs and map functions. Some environments expose a partial torch module
    without `Tensor`, which crashes that inspection path. Hide that module for
    the duration of dataset construction.
    """
    removed_torch = None
    torch_module = sys.modules.get("torch")
    if torch_module is not None and not hasattr(torch_module, "Tensor"):
        removed_torch = torch_module
        sys.modules.pop("torch", None)
    try:
        yield
    finally:
        if removed_torch is not None:
            sys.modules["torch"] = removed_torch

try:
    from transformers import (
        Trainer as HFTrainer,
        TrainingArguments,
        default_data_collator,
    )
except Exception:  # pragma: no cover - optional dependency
    HFTrainer = None
    TrainingArguments = None
    default_data_collator = None

try:
    from datasets import Dataset as HFDataset
except Exception:  # pragma: no cover - optional dependency
    HFDataset = None
try:
    from datasets import load_dataset, concatenate_datasets
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None
    concatenate_datasets = None
try:
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - optional dependency
    DataLoader = None
try:
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    F = None

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None


def _maybe_import(transformer_name: str):
    """Lazy import helper to keep this module usable without heavy deps."""
    try:
        module = __import__("transformers", fromlist=[transformer_name])
        return getattr(module, transformer_name)
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError(
            f"Missing transformers dependency for {transformer_name}: {exc}"
        )


def _load_paper_text_dataset(construction: Dict[str, Any], ds_cfg: Dict[str, Any], training_cfg: Dict[str, Any]):
    dataset_dir = construction.get("paper_dataset_dir") or "/arxiv/huggingface/paper_text_1m_dedup_v1"
    parquet_paths = _paper_parquet_paths(dataset_dir)
    if parquet_paths:
        return load_dataset("parquet", data_files=[str(p) for p in parquet_paths], split="train")

    dataset_id = (
        construction.get("paper_dataset_id")
        or construction.get("paper_dataset_name")
        or ds_cfg.get("paper_dataset_id")
        or "PeytonT/1m_papers_text"
    )
    split = construction.get("paper_dataset_split") or ds_cfg.get("split") or "train"
    cache_dir = construction.get("cache_dir") or ds_cfg.get("cache_dir") or training_cfg.get("cache_dir")
    kwargs: Dict[str, Any] = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(str(dataset_id), **kwargs)


def build_tokenizer(config: Dict[str, Any]):
    """Construct a tokenizer from config; requires transformers."""
    name = config.get("dataset", {}).get(
        "tokenization", {}
    ).get("tokenizer_name", config.get("backbone", {}).get("base_model"))
    cache_dir = config.get("backbone", {}).get("cache_dir")
    AutoTokenizer = _maybe_import("AutoTokenizer")
    try:
        tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    except Exception:
        # Offline fallback: local cache only.
        tok = AutoTokenizer.from_pretrained(
            name, cache_dir=cache_dir, local_files_only=True
        )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _iter_backbone_configs(backbone: Any) -> list[Any]:
    if backbone is None:
        return []
    candidates = [backbone]
    seen_ids = set()
    configs = []
    while candidates:
        current = candidates.pop(0)
        if current is None:
            continue
        ident = id(current)
        if ident in seen_ids:
            continue
        seen_ids.add(ident)
        cfg = getattr(current, "config", None)
        if cfg is not None:
            configs.append(cfg)
        for attr in ("base_model", "model"):
            nested = getattr(current, attr, None)
            if nested is not None:
                candidates.append(nested)
        getter = getattr(current, "get_base_model", None)
        if callable(getter):
            try:
                nested = getter()
            except Exception:
                nested = None
            if nested is not None:
                candidates.append(nested)
    return configs


def _effective_model_max_length(tokenizer: Any = None, backbone: Any = None) -> Optional[int]:
    limits = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    try:
        tokenizer_limit_int = int(tokenizer_limit)
    except Exception:
        tokenizer_limit_int = None
    if tokenizer_limit_int and 0 < tokenizer_limit_int < 1_000_000:
        limits.append(tokenizer_limit_int)

    for backbone_cfg in _iter_backbone_configs(backbone):
        for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            value = getattr(backbone_cfg, attr, None)
            try:
                value_int = int(value)
            except Exception:
                value_int = None
            if value_int and 0 < value_int < 1_000_000:
                limits.append(value_int)
    if not limits:
        return None
    return int(min(limits))


def _effective_source_token_limit(config: Dict[str, Any], tokenizer: Any = None, backbone: Any = None) -> int:
    configured = ((config.get("dataset") or {}).get("tokenization") or {}).get("max_source_tokens", 2048)
    try:
        configured_limit = int(configured)
    except Exception:
        configured_limit = 2048
    model_limit = _effective_model_max_length(tokenizer=tokenizer, backbone=backbone)
    if model_limit is None:
        return max(8, configured_limit)
    return max(8, min(configured_limit, model_limit))


_SEQ2SEQ_MODEL_IDS = {"A2", "A3", "P1", "P2", "P3", "P4", "P5", "R3", "R5", "C1"}


def _resolve_backbone_type(config: Dict[str, Any]) -> Optional[str]:
    backbone_cfg = config.get("backbone", {}) or {}
    model_type = backbone_cfg.get("type")
    base_model = str(backbone_cfg.get("base_model") or "")
    # Llama-family checkpoints are decoder-only; treat mislabeled configs as such.
    if model_type == "encoder" and "llama" in base_model.lower():
        return "decoder"
    return model_type


def _prefer_seq2seq_config(config: Dict[str, Any]) -> bool:
    training_cfg = config.get("training", {}) or {}
    if training_cfg.get("model_type") == "seq2seq":
        return True
    if _resolve_backbone_type(config) == "encoder_decoder":
        return True
    return config.get("model_id") in _SEQ2SEQ_MODEL_IDS


def _is_classifier_config(config: Dict[str, Any]) -> bool:
    training_cfg = config.get("training", {}) or {}
    archetype = get_archetype(config.get("model_id", "")) or {}
    return archetype.get("archetype") == "classifier" or training_cfg.get("model_type") == "classifier"


def _infer_peft_task_type(config: Dict[str, Any]) -> str:
    model_type = _resolve_backbone_type(config)
    archetype = get_archetype(config.get("model_id", "")) or {}
    if model_type == "encoder_decoder" or _prefer_seq2seq_config(config):
        return "SEQ_2_SEQ_LM"
    if _is_classifier_config(config):
        return "SEQ_CLS"
    if model_type == "encoder" or archetype.get("archetype") == "contrastive":
        return "FEATURE_EXTRACTION"
    return "CAUSAL_LM"


def _resolve_eval_split(config: Dict[str, Any]) -> float:
    training_cfg = config.get("training", {}) or {}
    if training_cfg.get("eval_split") is not None:
        try:
            return max(0.0, min(0.5, float(training_cfg.get("eval_split") or 0.0)))
        except Exception:
            return 0.0
    construction = ((config.get("dataset") or {}).get("construction") or {})
    split = construction.get("train_val_test_split") or []
    if isinstance(split, (list, tuple)) and len(split) >= 2:
        try:
            return max(0.0, min(0.5, float(split[1] or 0.0)))
        except Exception:
            return 0.0
    return 0.0


def _normalize_dataset_splits(ds: Any) -> Any:
    if not isinstance(ds, dict):
        return ds
    train_ds = ds.get("train")
    eval_ds = ds.get("eval")
    if eval_ds is None:
        eval_ds = ds.get("validation")
    if eval_ds is None:
        eval_ds = ds.get("test")
    if train_ds is None:
        train_ds = ds.get("train") or eval_ds
    if train_ds is None:
        return ds
    out = {"train": train_ds}
    if eval_ds is not None:
        out["eval"] = eval_ds
    return out


def _resolve_eval_max_samples(training_cfg: Dict[str, Any]) -> int:
    raw_value = training_cfg.get("eval_max_samples", 256)
    if raw_value is None:
        return 0
    try:
        value = int(raw_value)
    except Exception:
        return 256
    return max(0, value)


def _maybe_limit_eval_dataset(eval_ds: Any, training_cfg: Dict[str, Any]) -> Any:
    max_samples = _resolve_eval_max_samples(training_cfg)
    if not max_samples:
        return eval_ds
    try:
        length = len(eval_ds)
    except Exception:
        return eval_ds
    if length <= max_samples:
        return eval_ds
    if hasattr(eval_ds, "select"):
        return eval_ds.select(range(max_samples))
    try:
        return eval_ds[:max_samples]
    except Exception:
        return eval_ds


def _latest_checkpoint_dir(output_dir: str) -> Optional[str]:
    root = Path(str(output_dir or ""))
    if not root.is_dir():
        return None
    candidates = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.rsplit("-", 1)[-1]
        try:
            step = int(suffix)
        except Exception:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def _resolve_resume_checkpoint(output_dir: str, training_cfg: Dict[str, Any]) -> Optional[str]:
    raw_value = training_cfg.get("resume_from_checkpoint", "auto")
    if raw_value is False:
        return None
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"", "false", "no", "none", "off", "0"}:
            return None
        if normalized not in {"true", "yes", "auto", "1"}:
            if not Path(raw_value).exists():
                print(f"[train] resume checkpoint not found, starting fresh: {raw_value}")
                return None
            return str(raw_value)
    return _latest_checkpoint_dir(output_dir)


def _training_arguments_kwargs(
    training_cfg: Dict[str, Any],
    *,
    output_dir: str,
    has_eval: bool,
    use_cuda: bool,
) -> Dict[str, Any]:
    if TrainingArguments is None:
        return {}
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_strategy = training_cfg.get("evaluation_strategy", training_cfg.get("eval_strategy", "steps" if has_eval else "no"))
    if not has_eval:
        eval_strategy = "no"
    save_strategy = training_cfg.get("save_strategy", "steps")
    load_best_model = bool(has_eval and eval_strategy != "no" and training_cfg.get("load_best_model_at_end", False))

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": training_cfg.get("num_epochs", 1),
        "max_steps": training_cfg.get("max_steps", -1),
        "per_device_train_batch_size": training_cfg.get("batch_size", 1),
        # Eval has a much larger activation/logit footprint for causal LM
        # losses. Default it to 1 unless explicitly overridden.
        "per_device_eval_batch_size": training_cfg.get("eval_batch_size", 1),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": training_cfg.get("learning_rate", 5e-5),
        "weight_decay": training_cfg.get("weight_decay", 0.0),
        "warmup_steps": training_cfg.get("warmup_steps", 0),
        "fp16": use_cuda and training_cfg.get("precision", "fp16") == "fp16",
        "bf16": use_cuda and training_cfg.get("precision", "fp16") == "bf16",
        "logging_steps": training_cfg.get("logging_steps", 50),
        "save_steps": training_cfg.get("save_steps", 200),
        "save_total_limit": 1,
        "remove_unused_columns": False,
        "prediction_loss_only": training_cfg.get("prediction_loss_only", True),
        "report_to": [],
        "no_cuda": not use_cuda,
    }
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strategy
    if "save_strategy" in params:
        kwargs["save_strategy"] = save_strategy
    if has_eval and eval_strategy != "no" and "eval_steps" in params:
        kwargs["eval_steps"] = training_cfg.get("eval_steps", 200)
    if has_eval and eval_strategy != "no" and "eval_accumulation_steps" in params:
        kwargs["eval_accumulation_steps"] = training_cfg.get("eval_accumulation_steps", 1)
    if use_cuda and has_eval and eval_strategy != "no" and "torch_empty_cache_steps" in params:
        kwargs["torch_empty_cache_steps"] = training_cfg.get("torch_empty_cache_steps", training_cfg.get("eval_steps", 200))
    if "load_best_model_at_end" in params:
        kwargs["load_best_model_at_end"] = load_best_model
    if load_best_model and "metric_for_best_model" in params:
        metric_name = training_cfg.get("metric_for_best_model")
        if metric_name:
            kwargs["metric_for_best_model"] = metric_name
    return {key: value for key, value in kwargs.items() if key in params}


def build_backbone(config: Dict[str, Any]):
    """Construct a backbone model per config; supports encoder/decoder/encoder_decoder."""
    backbone_cfg = config.get("backbone", {})
    model_type = _resolve_backbone_type(config)
    base_model = backbone_cfg.get("base_model")
    training_cfg = config.get("training", {})
    prefer_seq2seq = _prefer_seq2seq_config(config)
    cache_dir = backbone_cfg.get("cache_dir")
    load_in_8bit = backbone_cfg.get("load_in_8bit", True)
    load_in_4bit = backbone_cfg.get("load_in_4bit", False)
    device_map = backbone_cfg.get("device_map", training_cfg.get("device_map"))
    if (
        device_map is None
        and torch is not None
        and torch.cuda.is_available()
        and not training_cfg.get("force_cpu", False)
        and torch.cuda.device_count() > 1
    ):
        device_map = "auto"

    cuda_available = torch is not None and torch.cuda.is_available()
    quantization_enabled = False
    if cuda_available and (load_in_4bit or load_in_8bit):
        try:
            from transformers import BitsAndBytesConfig as _BitsAndBytesConfig  # type: ignore
            global BitsAndBytesConfig
            BitsAndBytesConfig = _BitsAndBytesConfig
            quantization_enabled = True
        except Exception:
            quantization_enabled = False

    quantization_config = None
    if quantization_enabled:
        # Prefer 4-bit when requested; fall back to 8-bit. Use bf16 compute for better stability.
        q_kwargs: Dict[str, Any] = {
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit and not load_in_4bit,
        }
        if load_in_4bit and torch is not None:
            # QLoRA-style defaults.
            q_kwargs.update(
                {
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                }
            )
        quantization_config = BitsAndBytesConfig(**q_kwargs)
    else:
        # Disable quantization when bitsandbytes/cuda are unavailable.
        load_in_4bit = False
        load_in_8bit = False

    def _load(cls, name, **kwargs):
        try:
            return cls.from_pretrained(name, **kwargs)
        except Exception:
            return cls.from_pretrained(name, local_files_only=True, **kwargs)

    common_kwargs = {"cache_dir": cache_dir}
    if device_map:
        common_kwargs["device_map"] = device_map
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config
    elif load_in_4bit:
        common_kwargs["load_in_4bit"] = True
    elif load_in_8bit:
        common_kwargs["load_in_8bit"] = True

    # Optionally switch to classification head for classifier archetypes.
    is_classifier = _is_classifier_config(config)
    num_labels = int(training_cfg.get("num_labels", 10))

    if model_type == "encoder":
        if is_classifier:
            AutoModelForSequenceClassification = _maybe_import("AutoModelForSequenceClassification")
            model = _load(AutoModelForSequenceClassification, base_model, num_labels=num_labels, **common_kwargs)
        else:
            AutoModel = _maybe_import("AutoModel")
            model = _load(AutoModel, base_model, **common_kwargs)
    elif model_type == "decoder":
        if is_classifier:
            AutoModelForSequenceClassification = _maybe_import("AutoModelForSequenceClassification")
            model = _load(AutoModelForSequenceClassification, base_model, num_labels=num_labels, **common_kwargs)
        elif prefer_seq2seq:
            AutoModelForSeq2SeqLM = _maybe_import("AutoModelForSeq2SeqLM")
            model = _load(AutoModelForSeq2SeqLM, base_model, **common_kwargs)
        else:
            AutoModelForCausalLM = _maybe_import("AutoModelForCausalLM")
            model = _load(AutoModelForCausalLM, base_model, **common_kwargs)
    elif model_type == "encoder_decoder":
        AutoModelForSeq2SeqLM = _maybe_import("AutoModelForSeq2SeqLM")
        model = _load(AutoModelForSeq2SeqLM, base_model, **common_kwargs)
    else:
        raise NotImplementedError(f"Unsupported backbone type: {model_type}")

    # For 8/4-bit PEFT, prepare inputs/embeddings so gradients can flow.
    if quantization_enabled or load_in_8bit or load_in_4bit:
        try:
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(model)
        except Exception:
            # If peft is unavailable or preparation fails, continue without blocking.
            pass

    # Ensure input embeddings allow grad when checkpointing/LoRA.
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    # Apply optional gradient checkpointing to cut activation memory when requested.
    if training_cfg.get("gradient_checkpointing", False) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        try:
            model.gradient_checkpointing_enable()
            # Disable KV cache for decoder models when using checkpointing to save memory.
            if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not None:
                model.config.use_cache = False
        except Exception:
            # If a specific backbone doesn't support this, continue without failing.
            pass

    return model


def apply_peft_if_needed(model: Any, config: Dict[str, Any]) -> Any:
    """Attach LoRA adapters when requested."""
    adapter_type = config.get("backbone", {}).get("adapter_type", "none")
    strategy = config.get("training", {}).get("finetune_strategy", "peft_lora")
    if torch is None or not torch.cuda.is_available():
        return model
    if adapter_type != "lora" and strategy != "peft_lora":
        return model
    try:
        peft = __import__("peft")
        get_peft_model = getattr(peft, "get_peft_model")
        LoraConfig = getattr(peft, "LoraConfig")
        TaskType = getattr(peft, "TaskType", None)
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError(f"Missing PEFT dependency: {exc}")
    task_type = _infer_peft_task_type(config)
    if TaskType is not None and hasattr(TaskType, task_type):
        task_type = getattr(TaskType, task_type)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=None,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
    )
    return get_peft_model(model, lora_cfg)


class Trainer:
    """Minimal trainer scaffold; swap internals with HF Trainer as needed."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_stub: Any,
        tokenizer: Optional[Any] = None,
        backbone: Optional[Any] = None,
    ):
        self.config = config
        self.model_stub = model_stub
        self.strategy = config.get("training", {}).get("finetune_strategy", "peft_lora")
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.label_to_id = {}

    def _label_to_id(self, label: Any) -> int:
        key = str(label or "unknown")
        if key not in self.label_to_id:
            self.label_to_id[key] = len(self.label_to_id)
        return int(self.label_to_id[key])

    def _move_to_device(self, batch: Any, device: str) -> Any:
        if hasattr(batch, "to"):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        return batch

    def _contrastive_collate(self, batch):
        texts_a = []
        texts_b = []
        labels = []
        for pair, label in batch:
            if isinstance(pair, dict):
                text_a = pair.get("text_a") or pair.get("text") or ""
                text_b = pair.get("text_b") or ""
            elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                text_a = pair[0]
                text_b = pair[1]
            else:
                text_a = pair
                text_b = ""
            texts_a.append(str(text_a or ""))
            texts_b.append(str(text_b or ""))
            labels.append(int(label))
        return texts_a, texts_b, labels

    def _should_use_hf_trainer(self) -> bool:
        """Decide if we should attempt HF Trainer (CPU or CUDA)."""
        use_flag = self.config.get("training", {}).get("use_hf_trainer", False)
        allow_cpu = self.config.get("training", {}).get("allow_cpu_trainer", True)
        if not use_flag:
            return False
        if torch is None:
            return False
        if not allow_cpu and not torch.cuda.is_available():
            return False
        return all(
            [
                HFTrainer,
                TrainingArguments,
                default_data_collator,
                HFDataset,
                self.tokenizer,
                self.backbone,
            ]
        )

    def _build_corpus_hf_dataset(self):
        """
        Build an HF Dataset (or DatasetDict) directly from the sharded corpus
        under exports/corpus when dataset.sources includes corpus_* entries.

        This bypasses the in-memory Python list construction in build_dataset
        so we can train over the full corpus (repos, papers, and/or pairs)
        backed by Arrow/JSONL on disk.
        """
        if load_dataset is None or HFDataset is None:
            return None

        ds_cfg = self.config.get("dataset", {}) or {}
        sources = ds_cfg.get("sources") or []
        wants_corpus = any(s in {"corpus_repos", "corpus_papers", "corpus_pairs"} for s in sources)
        if not wants_corpus:
            return None

        corpus_root = Path(ds_cfg.get("corpus_dir") or "exports/corpus")
        if not corpus_root.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            candidates = [corpus_root, repo_root / corpus_root]
            if not str(corpus_root).startswith("models/"):
                candidates.append(repo_root / "models" / corpus_root)
            for candidate in candidates:
                if candidate.exists():
                    corpus_root = candidate
                    break
        data_files: Dict[str, Any] = {}

        def _glob(subdir: str, pattern: str) -> Any:
            path = corpus_root / subdir
            if not path.exists():
                return []
            return sorted(str(p) for p in path.glob(pattern))

        if "corpus_repos" in sources:
            repo_files = _glob("repos", "repo_*.jsonl")
            if repo_files:
                data_files["repos"] = repo_files
        if "corpus_papers" in sources:
            paper_files = _glob("papers", "paper_*.jsonl")
            if paper_files:
                data_files["papers"] = paper_files
        if "corpus_pairs" in sources:
            pair_files = _glob("pairs", "pair_*.jsonl")
            if pair_files:
                data_files["pairs"] = pair_files

        if not data_files:
            return None

        with _datasets_fingerprint_compatible_env():
            raw = load_dataset("json", data_files=data_files)
            archetype = get_archetype(self.config.get("model_id", "")) or {}
            archetype_name = archetype.get("archetype", "generative")
            ds_quality = ds_cfg.get("quality_filters") or {}
            ds_domains = set(ds_cfg.get("domains") or [])
            training_cfg = self.config.get("training", {}) or {}
            eval_split = _resolve_eval_split(self.config)

            # --- Quality filtering heuristics --------------------------------- #

            def _len_bounds(kind: str) -> tuple[int, int]:
                if kind == "repos":
                    return int(ds_quality.get("repos_min_chars", 64)), int(ds_quality.get("repos_max_chars", 65536))
                if kind == "papers":
                    return int(ds_quality.get("papers_min_chars", 64)), int(ds_quality.get("papers_max_chars", 65536))
                if kind == "pairs":
                    return int(ds_quality.get("pairs_min_chars", 64)), int(ds_quality.get("pairs_max_chars", 65536))
                return 0, 10**9

            def _filter_repos(ex: Dict[str, Any]) -> bool:
                text = str(ex.get("text") or "").strip()
                mn, mx = _len_bounds("repos")
                if not (mn <= len(text) <= mx):
                    return False
                meta = ex.get("meta") or {}
                path = str(meta.get("path") or "")
                # Basic path-based noise filtering (virtualenvs, caches, etc.).
                noisy_fragments = ds_quality.get(
                    "repos_exclude_path_fragments",
                    ["site-packages", ".venv", "__pycache__", "node_modules", "dist-packages"],
                )
                for frag in noisy_fragments:
                    if frag and frag in path:
                        return False
                return True

            def _filter_papers(ex: Dict[str, Any]) -> bool:
                text = str(ex.get("text") or "").strip()
                mn, mx = _len_bounds("papers")
                if not (mn <= len(text) <= mx):
                    return False
                return True

            def _filter_pairs(ex: Dict[str, Any]) -> bool:
                paper = str(ex.get("paper_text") or "").strip()
                repo = str(ex.get("repo_text") or "").strip()
                mn, mx = _len_bounds("pairs")
                total_len = len(paper) + len(repo)
                if not paper or not repo:
                    return False
                if not (mn <= total_len <= mx):
                    return False
                return True

            if "repos" in raw:
                raw["repos"] = raw["repos"].filter(_filter_repos)
            if "papers" in raw:
                raw["papers"] = raw["papers"].filter(_filter_papers)
            if "pairs" in raw:
                raw["pairs"] = raw["pairs"].filter(_filter_pairs)

            # --- Domain-level filtering ---------------------------------------- #

            if ds_domains:
                def _has_domain(ex: Dict[str, Any]) -> bool:
                    meta = ex.get("meta") or {}
                    doms = meta.get("domains") or []
                    if isinstance(doms, str):
                        doms_list = [doms]
                    else:
                        try:
                            doms_list = list(doms)
                        except Exception:
                            doms_list = []
                    return any(d in ds_domains for d in doms_list)

                if "repos" in raw:
                    raw["repos"] = raw["repos"].filter(_has_domain)
                if "papers" in raw:
                    raw["papers"] = raw["papers"].filter(_has_domain)
                if "pairs" in raw:
                    raw["pairs"] = raw["pairs"].filter(_has_domain)

            # --- Static corpus mixing / re-weighting -------------------------- #

            mix_cfg = ds_cfg.get("corpus_mix") or {}
            if mix_cfg and any(k in raw for k in ("repos", "papers")) and archetype_name != "contrastive":
                parts = []

                def _reweight(split_name: str, ds_split):
                    weight = float(mix_cfg.get(split_name, 1.0))
                    if weight <= 0.0:
                        return None
                    # For weight < 1.0, subsample; for >1.0, oversample up to 3x.
                    n = len(ds_split)
                    if n == 0:
                        return None
                    target = int(max(1, min(n * max(weight, 0.0), n * 3.0)))
                    if target == n:
                        return ds_split
                    return ds_split.shuffle(seed=int(training_cfg.get("shuffle_seed", 42))).select(range(target))

                if "repos" in raw:
                    ds_r = _reweight("repos", raw["repos"])
                    if ds_r is not None:
                        raw["repos"] = ds_r
                if "papers" in raw:
                    ds_p = _reweight("papers", raw["papers"])
                    if ds_p is not None:
                        raw["papers"] = ds_p

            # Contrastive models (e.g., C2/C6) should train only on pair records.
            if archetype_name == "contrastive" and "pairs" in raw:
                base = raw["pairs"]
                if eval_split > 0.0:
                    return _normalize_dataset_splits(base.train_test_split(test_size=eval_split))
                return base

            # Generative/classifier models: mix repo and paper corpora if available.
            parts = []
            if "repos" in raw:
                parts.append(raw["repos"])
            if "papers" in raw:
                parts.append(raw["papers"])
            if not parts:
                return None
            mixed = parts[0] if len(parts) == 1 or concatenate_datasets is None else concatenate_datasets(parts)
            if eval_split > 0.0:
                return _normalize_dataset_splits(mixed.train_test_split(test_size=eval_split))
            return mixed

    def _build_paper_text_hf_dataset(self):
        """
        Build an HF Dataset directly from the parquet-backed paper-text corpus.

        This avoids materializing large Python lists for paper-only models like
        P1/C3 and keeps the Arrow/parquet path available for abstract and
        metadata style models over the same paper rows.
        """
        if load_dataset is None or HFDataset is None:
            return None

        ds_cfg = self.config.get("dataset", {}) or {}
        sources = ds_cfg.get("sources") or []
        if "paper_text_parquet" not in sources:
            return None

        construction = ds_cfg.get("construction") or {}
        with _datasets_fingerprint_compatible_env():
            filters = ds_cfg.get("filters") or {}
            training_cfg = self.config.get("training", {}) or {}
            raw = _load_paper_text_dataset(construction, ds_cfg, training_cfg)
            years_filter = filters.get("years")
            categories_filter = set(filters.get("categories") or [])
            ds_quality = ds_cfg.get("quality_filters") or {}
            archetype = get_archetype(self.config.get("model_id", "")) or {}
            archetype_name = archetype.get("archetype", "generative")
            model_id = self.config.get("model_id", "")
            eval_split = _resolve_eval_split(self.config)
            max_samples = int(construction.get("max_samples", 0) or 0)
            shuffle_seed = int(training_cfg.get("shuffle_seed", construction.get("shuffling_seed", 42)))
            prefer_seq2seq = _prefer_seq2seq_config(self.config)

            def _match_filters(ex: Dict[str, Any]) -> bool:
                normalized = _normalize_paper_row(ex)
                if years_filter and len(years_filter) == 2:
                    year = _paper_row_year(normalized)
                    if year is not None and not (int(years_filter[0]) <= year <= int(years_filter[1])):
                        return False
                if categories_filter:
                    categories = normalized.get("categories") or normalized.get("primary_category") or ""
                    if not any(cat in str(categories) for cat in categories_filter):
                        return False
                if model_id in {"P1", "C3", "M6", "M7"}:
                    text = str(normalized.get("text") or "").strip()
                    if not text:
                        return False
                    min_chars = int(ds_quality.get("papers_min_chars", 256))
                    max_chars = int(ds_quality.get("papers_max_chars", 131072))
                    text_len = int(normalized.get("text_char_count") or len(text))
                    if not (min_chars <= text_len <= max_chars):
                        return False
                else:
                    title = str(normalized.get("title") or "").strip()
                    abstract = str(normalized.get("abstract") or "").strip()
                    if not title and not abstract:
                        return False
                if ds_quality.get("require_full_text", False) and bool(normalized.get("text_is_partial")):
                    return False
                return True

            raw = raw.filter(_match_filters)
            if len(raw) == 0:
                return None

            raw = raw.shuffle(seed=shuffle_seed)
            if max_samples > 0:
                raw = raw.select(range(min(max_samples, len(raw))))
            remove_columns = list(raw.features)

            def _map_classifier(batch: Dict[str, Any]) -> Dict[str, Any]:
                texts = []
                labels = []
                size = len(next(iter(batch.values()))) if batch else 0
                for idx in range(size):
                    row = _normalize_paper_row({k: batch[k][idx] for k in batch})
                    texts.append(_format_text_from_entry(row))
                    labels.append(str(row.get("primary_category") or row.get("categories") or "unknown"))
                return {"text": texts, "label": labels}

            def _map_method_summary(batch: Dict[str, Any]) -> Dict[str, Any]:
                texts = []
                targets = []
                size = len(next(iter(batch.values()))) if batch else 0
                for idx in range(size):
                    row = _normalize_paper_row({k: batch[k][idx] for k in batch})
                    texts.append(str(row.get("abstract") or "").strip() or _format_text_from_entry(row))
                    targets.append(_paper_method_summary_target(row))
                return {"text": texts, "target": targets}

            def _map_keywords(batch: Dict[str, Any]) -> Dict[str, Any]:
                texts = []
                targets = []
                size = len(next(iter(batch.values()))) if batch else 0
                for idx in range(size):
                    row = _normalize_paper_row({k: batch[k][idx] for k in batch})
                    texts.append(str(row.get("abstract") or "").strip() or _format_text_from_entry(row))
                    targets.append(_paper_keyword_target(row))
                return {"text": texts, "target": targets}

            def _map_paper_qa(batch: Dict[str, Any]) -> Dict[str, Any]:
                size = len(next(iter(batch.values()))) if batch else 0
                rows = []
                for idx in range(size):
                    rows.append(_normalize_paper_row({k: batch[k][idx] for k in batch}))
                samples = _build_paper_qa_samples(rows, max_samples=max(1, size * 3))
                return {
                    "question": [sample["question"] for sample in samples],
                    "context": [sample["context"] for sample in samples],
                    "target": [sample["target"] for sample in samples],
                    "paper_id": [sample.get("paper_id") for sample in samples],
                    "pdf_path": [sample.get("pdf_path") for sample in samples],
                }

            def _map_fulltext(batch: Dict[str, Any]) -> Dict[str, Any]:
                samples = []
                chunk_chars = int(construction.get("chunk_chars_text", construction.get("chunk_chars_pdf", 8000)) or 8000)
                chunk_overlap = int(construction.get("chunk_overlap_text", construction.get("chunk_overlap_pdf", 400)) or 400)
                size = len(next(iter(batch.values()))) if batch else 0
                rows = []
                for idx in range(size):
                    rows.append(_normalize_paper_row({k: batch[k][idx] for k in batch}))
                samples = _build_fulltext_samples(
                    rows,
                    max_samples=max(1, size * 64),
                    chunk_chars=chunk_chars,
                    overlap=chunk_overlap,
                    target_mode="next_chunk" if model_id == "P1" and prefer_seq2seq else "none",
                )
                result = {
                    "text": [sample["text"] for sample in samples],
                    "paper_id": [sample.get("paper_id") for sample in samples],
                    "pdf_path": [sample.get("pdf_path") for sample in samples],
                    "offset": [sample.get("offset") for sample in samples],
                }
                if any(sample.get("target") is not None for sample in samples):
                    result["target"] = [sample.get("target") for sample in samples]
                return result

            def _map_contrastive(batch: Dict[str, Any]) -> Dict[str, Any]:
                size = len(next(iter(batch.values()))) if batch else 0
                rows = []
                for idx in range(size):
                    rows.append(_normalize_paper_row({k: batch[k][idx] for k in batch}))
                chunk_chars = int(construction.get("chunk_chars_text", construction.get("chunk_chars_pdf", 8000)) or 8000)
                chunk_overlap = int(construction.get("chunk_overlap_text", construction.get("chunk_overlap_pdf", 400)) or 400)
                if model_id == "M6":
                    pairs = _build_paper_fulltext_embedding_samples(
                        rows,
                        max_samples=max(1, size * 8),
                        chunk_chars=chunk_chars,
                        overlap=chunk_overlap,
                        max_chunks_per_paper=int(construction.get("max_chunks_per_paper", 2) or 2),
                        document_chars=int(construction.get("document_chars", 6000) or 6000),
                    )
                elif model_id == "M7":
                    pairs = _build_paper_sentence_embedding_samples(
                        rows,
                        max_samples=max(1, size * 6),
                        max_query_sentences=int(construction.get("max_query_sentences", 3) or 3),
                        max_body_sentences=int(construction.get("max_body_sentences", 16) or 16),
                    )
                elif model_id == "M1":
                    pairs = _build_metadata_embedding_samples(rows, max_samples=max(1, size * 4))
                else:
                    pairs = _build_paper_retrieval_samples(
                        rows,
                        max_samples=max(1, size * 6),
                        chunk_chars=chunk_chars,
                        overlap=chunk_overlap,
                    )
                return {
                    "text_a": [sample["text_a"] for sample in pairs],
                    "text_b": [sample["text_b"] for sample in pairs],
                    "label": [sample["label"] for sample in pairs],
                    "paper_id": [sample.get("paper_id") for sample in pairs],
                }

            if archetype_name == "contrastive":
                processed = raw.map(_map_contrastive, batched=True, batch_size=1024, remove_columns=remove_columns)
            elif archetype_name == "classifier":
                processed = raw.map(_map_classifier, batched=True, batch_size=1024, remove_columns=remove_columns)
            elif model_id == "A2":
                processed = raw.map(_map_method_summary, batched=True, batch_size=256, remove_columns=remove_columns)
            elif model_id == "A3":
                processed = raw.map(_map_keywords, batched=True, batch_size=256, remove_columns=remove_columns)
            elif model_id == "P5":
                processed = raw.map(_map_paper_qa, batched=True, batch_size=128, remove_columns=remove_columns)
            elif model_id in {"P1", "C3"}:
                processed = raw.map(_map_fulltext, batched=True, batch_size=64, remove_columns=remove_columns)
            else:
                return None

            if len(processed) == 0:
                return None
            if archetype_name == "contrastive" and "paper_id" in processed.features:
                processed = self._augment_dataset_with_teacher_embeddings(processed, construction)
            if eval_split > 0.0:
                return _normalize_dataset_splits(processed.train_test_split(test_size=eval_split))
            return processed

    def _augment_dataset_with_teacher_embeddings(self, dataset: Any, construction: Dict[str, Any]):
        if HFDataset is None or dataset is None:
            return dataset
        feature_names = set(getattr(dataset, "column_names", []) or [])
        if not feature_names and getattr(dataset, "features", None) is not None:
            try:
                feature_names = set(dataset.features.keys())
            except Exception:
                feature_names = set()
        if "paper_id" not in feature_names:
            return dataset
        training_cfg = self.config.get("training", {}) or {}
        if float(training_cfg.get("distillation_weight", 0.0) or 0.0) <= 0.0:
            return dataset
        try:
            paper_ids = {
                str(paper_id or "").strip()
                for paper_id in dataset["paper_id"]
                if str(paper_id or "").strip()
            }
        except Exception:
            return dataset
        lookup = _load_teacher_embedding_lookup(paper_ids, construction)
        if not lookup:
            return dataset
        default_embedding = [0.0 for _ in range(len(next(iter(lookup.values()))))]

        def _attach(batch: Dict[str, Any]) -> Dict[str, Any]:
            embeddings = []
            masks = []
            for paper_id in batch.get("paper_id") or []:
                key = str(paper_id or "").strip()
                teacher = lookup.get(key)
                embeddings.append(list(teacher) if teacher is not None else list(default_embedding))
                masks.append(1 if teacher is not None else 0)
            return {"teacher_embedding": embeddings, "teacher_mask": masks}

        return dataset.map(_attach, batched=True, batch_size=1024)

    def _maybe_prepare_teacher_projection(self, train_ds: Any) -> None:
        if torch is None or self.backbone is None or train_ds is None:
            return
        training_cfg = self.config.get("training", {}) or {}
        if float(training_cfg.get("distillation_weight", 0.0) or 0.0) <= 0.0:
            return
        sample = None
        try:
            if len(train_ds) > 0:
                sample = train_ds[0]
        except Exception:
            sample = None
        teacher_embedding = sample.get("teacher_embedding") if isinstance(sample, dict) else None
        if not isinstance(teacher_embedding, list) or not teacher_embedding:
            return
        teacher_dim = int(len(teacher_embedding))
        cfg = getattr(self.backbone, "config", None)
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if not hidden_size:
            return
        if teacher_dim != int(hidden_size) and not hasattr(self.backbone, "teacher_projection"):
            self.backbone.add_module(
                "teacher_projection",
                torch.nn.Linear(int(hidden_size), teacher_dim, bias=False),
            )

    def _build_hf_dataset(self):
        # Prefer full-corpus HF datasets when corpus_* sources are configured.
        corpus_ds = self._build_corpus_hf_dataset()
        if corpus_ds is not None:
            return corpus_ds
        paper_ds = self._build_paper_text_hf_dataset()
        if paper_ds is not None:
            return paper_ds

        dataset = build_dataset(self.config)
        if not dataset:
            return None
        eval_split = _resolve_eval_split(self.config)
        if eval_split > 0 and len(dataset) > 4:
            cut = max(1, int(len(dataset) * eval_split))
            eval_list = dataset[:cut]
            train_list = dataset[cut:]
            return _normalize_dataset_splits({
                "train": HFDataset.from_list(train_list),
                "eval": HFDataset.from_list(eval_list),
            })
        return HFDataset.from_list(dataset)

    def _mean_pool(self, hidden_states):
        if hidden_states is None or hidden_states.ndim != 3:
            return None
        return hidden_states.mean(dim=1)

    def _contrastive_loss(self, embeds_a, embeds_b, labels, margin: float = 0.2):
        if torch is None:
            return None
        # Normalize
        embeds_a = torch.nn.functional.normalize(embeds_a, dim=-1)
        embeds_b = torch.nn.functional.normalize(embeds_b, dim=-1)
        sims = (embeds_a * embeds_b).sum(dim=-1)
        pos_mask = torch.tensor([1 if l else 0 for l in labels], device=sims.device, dtype=torch.float)
        neg_mask = 1 - pos_mask
        pos_loss = torch.clamp(1 - sims, min=0) * pos_mask
        neg_loss = torch.clamp(sims - margin, min=0) * neg_mask
        loss = (pos_loss + neg_loss).mean()
        return loss

    def _contrastive_train(self):
        """Lightweight contrastive training loop when HF Trainer is not used."""
        if torch is None or DataLoader is None or self.tokenizer is None or self.backbone is None:
            return False
        training_cfg = self.config.get("training", {})
        ds_list = build_dataset(self.config)
        if not ds_list:
            return False
        pairs = []
        labels = []
        for ex in ds_list:
            a = ex.get("text_a") or ex.get("text") or ""
            b = ex.get("text_b") or ""
            pairs.append((a, b))
            labels.append(int(ex.get("label", 0)))
        device = "cuda" if torch.cuda.is_available() and not training_cfg.get("force_cpu", False) else "cpu"
        self.backbone.to(device)
        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=training_cfg.get("learning_rate", 1e-4))
        batch_size = max(1, int(training_cfg.get("batch_size", 4)))
        max_steps = training_cfg.get("max_steps", 50)

        class ContrastiveDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(pairs)

            def __getitem__(self, idx):
                return pairs[idx], labels[idx]

        dl = DataLoader(ContrastiveDataset(), batch_size=batch_size, shuffle=True, collate_fn=self._contrastive_collate)
        steps = 0
        self.backbone.train()
        for batch in dl:
            if steps >= max_steps:
                break
            texts_a, texts_b, lbls = batch
            enc_a = self._move_to_device(
                self.tokenizer(texts_a, padding=True, truncation=True, max_length=256, return_tensors="pt"),
                device,
            )
            enc_b = self._move_to_device(
                self.tokenizer(texts_b, padding=True, truncation=True, max_length=256, return_tensors="pt"),
                device,
            )
            out_a = self.backbone(**enc_a)
            out_b = self.backbone(**enc_b)
            hidden_a = out_a.last_hidden_state if hasattr(out_a, "last_hidden_state") else None
            hidden_b = out_b.last_hidden_state if hasattr(out_b, "last_hidden_state") else None
            pooled_a = self._mean_pool(hidden_a)
            pooled_b = self._mean_pool(hidden_b)
            if pooled_a is None or pooled_b is None:
                continue
            loss = self._contrastive_loss(pooled_a, pooled_b, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
        print(f"[train] contrastive loop completed in {steps} steps.")
        return True

    def _tokenize_fn(self, examples: Dict[str, Any]):
        tok_cfg = self.config.get("dataset", {}).get("tokenization", {})
        max_src = _effective_source_token_limit(self.config, tokenizer=self.tokenizer, backbone=self.backbone)
        training_cfg = self.config.get("training", {})
        archetype = get_archetype(self.config.get("model_id", "")) or {}
        objective = training_cfg.get("objective")
        prefer_seq2seq = _prefer_seq2seq_config(self.config)
        classifier_like = _is_classifier_config(self.config) or archetype.get("archetype") == "graph" or objective == "link_prediction"
        text = examples.get("text", "")
        tokens = examples.get("tokens")
        if (not text) and tokens:
            if isinstance(tokens, list):
                # Structured tokens: flatten token text fields; fallback to str(token)
                parts = []
                for t in tokens:
                    if isinstance(t, dict):
                        parts.append(str(t.get("text") or t.get("content") or t.get("type") or ""))
                    else:
                        parts.append(str(t))
                text = "\n".join([p for p in parts if p])
            else:
                text = str(tokens)
        target = examples.get("target")
        text_a = examples.get("text_a")
        text_b = examples.get("text_b")
        label = examples.get("label")
        question = examples.get("question") or examples.get("query")
        context = examples.get("context") or examples.get("code") or examples.get("snippet")

        # For contrastive, return dual encodings
        if archetype.get("archetype") == "contrastive" and text_a and text_b:
            enc_a = self.tokenizer(
                text_a,
                truncation=True,
                max_length=max_src,
                padding="max_length",
            )
            enc_b = self.tokenizer(
                text_b,
                truncation=True,
                max_length=max_src,
                padding="max_length",
            )
            enc_a = {f"{k}_a": v for k, v in enc_a.items()}
            enc_b = {f"{k}_b": v for k, v in enc_b.items()}
            enc_a.update(enc_b)
            enc_a["labels"] = int(label) if label is not None else 0
            teacher_embedding = examples.get("teacher_embedding")
            teacher_mask = examples.get("teacher_mask")
            if teacher_embedding is not None:
                enc_a["teacher_embedding"] = teacher_embedding
            if teacher_mask is not None:
                enc_a["teacher_mask"] = int(teacher_mask)
            return enc_a

        # Otherwise, single prompt path (seq2seq or decoder LM)
        if text_a and text_b:
            prompt = f"QUERY:\n{text_a}\nDOC:\n{text_b}"
        else:
            prompt = text

        # QA/mutation shaping
        if question or context:
            prompt = ""
            if question:
                prompt += f"QUESTION:\n{question}\n"
            if context:
                prompt += f"CONTEXT:\n{context}\n"
            if text:
                prompt += f"TEXT:\n{text}\n"
            if target and not prefer_seq2seq:
                prompt += f"TARGET:\n{target}"

        if target is not None and not prefer_seq2seq:
            prompt = f"INPUT:\n{prompt}\nTARGET:\n{target}"

        if prefer_seq2seq and target is not None:
            target_kwargs = {"truncation": True, "max_length": tok_cfg.get("max_target_tokens", 512), "padding": "max_length"}
            try:
                model_inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_src,
                    padding="max_length",
                )
                target_inputs = self.tokenizer(text_target=str(target), **target_kwargs)
                model_inputs["labels"] = target_inputs["input_ids"]
            except TypeError:
                model_inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_src,
                    padding="max_length",
                )
                with self.tokenizer.as_target_tokenizer():
                    model_inputs["labels"] = self.tokenizer(str(target), **target_kwargs)["input_ids"]
            return model_inputs

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_src,
            padding="max_length",
        )
        # Labels handling
        if classifier_like:
            if isinstance(label, str):
                # Map string labels to ids
                label_id = self._label_to_id(label)
                encoded["labels"] = label_id
            elif isinstance(label, (int, float)):
                encoded["labels"] = int(label)
            else:
                encoded["labels"] = -100
        elif target is not None:
            # Generative target as labels
            target_ids = self.tokenizer(
                str(target),
                truncation=True,
                max_length=tok_cfg.get("max_target_tokens", 512),
                padding="max_length",
            )["input_ids"]
            encoded["labels"] = target_ids
        else:
            # Align labels with input_ids to avoid batch shape mismatches.
            encoded["labels"] = encoded["input_ids"]
        return encoded

    def _train_with_hf_trainer(self):
        ds = self._build_hf_dataset()
        if ds is None:
            return False
        ds = _normalize_dataset_splits(ds)
        has_eval = isinstance(ds, dict)
        train_ds = ds["train"] if has_eval else ds
        eval_ds = ds["eval"] if has_eval else None
        # Collect label space for classifiers
        training_cfg = self.config.get("training", {})
        archetype = get_archetype(self.config.get("model_id", "")) or {}
        objective = training_cfg.get("objective")
        if objective == "cross_entropy" and archetype.get("archetype") == "classifier":
            labels = set()
            for ex in (train_ds if not has_eval else train_ds):
                if ex.get("label") is not None:
                    labels.add(str(ex["label"]))
            self.label_to_id = {lbl: idx for idx, lbl in enumerate(sorted(labels))}
            training_cfg["num_labels"] = max(len(self.label_to_id), 2)
            if self.backbone is not None and getattr(self.backbone, "config", None) is not None:
                self.backbone.config.label2id = dict(self.label_to_id)
                self.backbone.config.id2label = {idx: lbl for lbl, idx in self.label_to_id.items()}

        self._maybe_prepare_teacher_projection(train_ds)
        tokenized_train = train_ds.map(self._tokenize_fn, remove_columns=list(train_ds.features))
        tokenized_eval = None
        if eval_ds is not None:
            eval_ds = _maybe_limit_eval_dataset(eval_ds, training_cfg)
            tokenized_eval = eval_ds.map(self._tokenize_fn, remove_columns=list(eval_ds.features))
            tokenized_eval = _maybe_limit_eval_dataset(tokenized_eval, training_cfg)
            try:
                print(
                    "[train] eval dataset capped at "
                    f"{len(tokenized_eval):,} samples "
                    "(training.eval_max_samples; set 0 to disable cap)."
                )
            except Exception:
                pass
        training_cfg = self.config.get("training", {})
        cache_cfg = validate_cache_dirs(self.config)
        preferred_out = training_cfg.get("checkpoint_dir", cache_cfg["training"]["checkpoint_dir"])
        out_dir = preferred_out
        # Fallback to a local writable path if preferred is not writable or sandboxed.
        if (preferred_out.startswith("/data/checkpoints")) or (not os.access(preferred_out, os.W_OK)):
            out_dir = os.path.join(
                os.getcwd(),
                "models",
                "checkpoints",
                self.config.get("model_id", "model"),
            )
            os.makedirs(out_dir, exist_ok=True)
        force_cpu = training_cfg.get("force_cpu", False)
        use_cuda = torch is not None and torch.cuda.is_available() and not force_cpu
        args = TrainingArguments(
            **_training_arguments_kwargs(
                training_cfg,
                output_dir=out_dir,
                has_eval=bool(tokenized_eval is not None),
                use_cuda=use_cuda,
            )
        )
        # Choose trainer
        if archetype.get("archetype") == "contrastive":
            if torch is None:
                return False

            class ContrastiveTrainer(HFTrainer):
                @staticmethod
                def _clone_eval_value(value):
                    if hasattr(value, "clone"):
                        try:
                            return value.clone()
                        except Exception:
                            return value
                    if isinstance(value, list):
                        return list(value)
                    if isinstance(value, tuple):
                        return tuple(value)
                    if isinstance(value, dict):
                        return dict(value)
                    return value

                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    teacher_embedding = inputs.pop("teacher_embedding", None)
                    teacher_mask = inputs.pop("teacher_mask", None)
                    # Split inputs
                    def split_inputs(prefix):
                        return {
                            k.replace(f"_{prefix}", ""): v
                            for k, v in inputs.items()
                            if k.endswith(f"_{prefix}")
                        }
                    inp_a = split_inputs("a")
                    inp_b = split_inputs("b")
                    out_a = model(**inp_a)
                    out_b = model(**inp_b)
                    hidden_a = out_a.last_hidden_state if hasattr(out_a, "last_hidden_state") else None
                    hidden_b = out_b.last_hidden_state if hasattr(out_b, "last_hidden_state") else None
                    if hidden_a is None or hidden_b is None:
                        loss = model(**inp_a, labels=labels).loss
                        return (loss, out_a) if return_outputs else loss
                    pooled_a = hidden_a.mean(dim=1)
                    pooled_b = hidden_b.mean(dim=1)
                    sims = F.cosine_similarity(pooled_a, pooled_b)
                    lbl = labels.to(sims.device).float()
                    pos = (1 - sims) * lbl
                    margin = training_cfg.get("contrastive_margin", 0.2)
                    neg = torch.clamp(sims - margin, min=0.0) * (1 - lbl)
                    loss = (pos + neg).mean()
                    distillation_weight = float(training_cfg.get("distillation_weight", 0.0) or 0.0)
                    if (
                        distillation_weight > 0.0
                        and teacher_embedding is not None
                        and teacher_mask is not None
                    ):
                        teacher = teacher_embedding.to(pooled_b.device).float()
                        teacher_active = teacher_mask.to(pooled_b.device).float().view(-1)
                        if teacher.ndim == 2 and teacher.shape[-1] > 0 and torch.count_nonzero(teacher_active) > 0:
                            student_teacher = pooled_b
                            teacher_projection = getattr(model, "teacher_projection", None)
                            if teacher_projection is not None:
                                student_teacher = teacher_projection(student_teacher)
                            student_teacher = F.normalize(student_teacher, dim=-1)
                            teacher = F.normalize(teacher, dim=-1)
                            distill_loss = 1 - (student_teacher * teacher).sum(dim=-1)
                            distill_loss = (distill_loss * teacher_active).sum() / teacher_active.sum().clamp_min(1.0)
                            loss = loss + (distillation_weight * distill_loss)
                    return (loss, (out_a, out_b)) if return_outputs else loss

                def prediction_step(
                    self,
                    model,
                    inputs,
                    prediction_loss_only,
                    ignore_keys=None,
                ):
                    eval_inputs = {
                        key: self._clone_eval_value(value)
                        for key, value in inputs.items()
                    }
                    with torch.no_grad():
                        loss = self.compute_loss(
                            model,
                            eval_inputs,
                            return_outputs=False,
                        )
                    if not hasattr(loss, "detach"):
                        return (loss, None, None)
                    loss = loss.detach()
                    if prediction_loss_only:
                        return (loss, None, None)
                    labels = inputs.get("labels")
                    if hasattr(labels, "detach"):
                        labels = labels.detach()
                    return (loss, None, labels)

            trainer = ContrastiveTrainer(
                model=self.backbone,
                args=args,
                train_dataset=tokenized_train,
                tokenizer=self.tokenizer,
                data_collator=default_data_collator,
                eval_dataset=tokenized_eval,
            )
        else:
            trainer_kwargs = {
                "model": self.backbone,
                "args": args,
                "train_dataset": tokenized_train,
                "tokenizer": self.tokenizer,
                "data_collator": default_data_collator,
            }
            if tokenized_eval is not None:
                trainer_kwargs["eval_dataset"] = tokenized_eval
            trainer = HFTrainer(**trainer_kwargs)
        resume_checkpoint = _resolve_resume_checkpoint(out_dir, training_cfg)
        if resume_checkpoint:
            print(f"[train] resuming HF Trainer from {resume_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()
        return True

    def train(self):
        """Mock train loop to keep CLI end-to-end runnable."""
        archetype = get_archetype(self.config.get("model_id", "")) or {}
        training_cfg = self.config.get("training", {}) or {}
        use_hf_trainer = self._should_use_hf_trainer()

        # Prefer HF Trainer when configured so Arrow-backed datasets and eval
        # splits are used. Fall back to the lightweight contrastive loop if
        # trainer initialization or runtime fails.
        if archetype.get("archetype") == "contrastive" and not use_hf_trainer:
            try:
                if self._contrastive_train():
                    print("[train] completed contrastive loop.")
                    return None
            except Exception as exc:
                print(f"[warn] contrastive loop failed, falling back: {exc}")

        # Optional multi-phase curriculum over the full corpus: if
        # training.curriculum is present, iterate phases, adjusting
        # dataset.corpus_mix and basic training knobs (e.g., max_steps)
        # per phase while reusing the same backbone/checkpoints.
        curriculum = training_cfg.get("curriculum") or []
        if use_hf_trainer and curriculum:
            ds_cfg = self.config.setdefault("dataset", {})
            orig_mix = dict(ds_cfg.get("corpus_mix") or {})
            orig_max_steps = training_cfg.get("max_steps", -1)
            for idx, phase in enumerate(curriculum):
                phase_name = phase.get("name") or f"phase_{idx}"
                phase_mix = phase.get("corpus_mix")
                if phase_mix is not None:
                    ds_cfg["corpus_mix"] = phase_mix
                phase_max_steps = phase.get("max_steps")
                if phase_max_steps is not None:
                    training_cfg["max_steps"] = phase_max_steps
                print(f"[train] curriculum {phase_name}: corpus_mix={ds_cfg.get('corpus_mix')} max_steps={training_cfg.get('max_steps')}")
                # Run one HF-Trainer pass for this phase; subsequent phases
                # will continue from the latest checkpoint on disk.
                try:
                    if self._train_with_hf_trainer():
                        print(f"[train] completed curriculum phase {phase_name}.")
                except Exception as exc:
                    print(f"[warn] curriculum phase {phase_name} failed, continuing: {exc}")
            # Restore original config knobs for any follow-up calls.
            ds_cfg["corpus_mix"] = orig_mix
            training_cfg["max_steps"] = orig_max_steps
            return None

        if use_hf_trainer:
            try:
                if self._train_with_hf_trainer():
                    print("[train] completed via HF Trainer.")
                    return None
            except Exception as exc:  # pragma: no cover - fallback
                print(f"[warn] HF Trainer path failed, falling back to mock: {exc}")
                if not training_cfg.get("allow_mock_fallback", False):
                    raise RuntimeError(
                        "HF Trainer failed and mock fallback is disabled for this run. "
                        "Set training.allow_mock_fallback=true only if you explicitly want a dry-run style fallback."
                    ) from exc
        if archetype.get("archetype") == "contrastive":
            try:
                if self._contrastive_train():
                    print("[train] completed contrastive loop.")
                    return None
            except Exception as exc:
                print(f"[warn] contrastive loop failed after trainer fallback: {exc}")
        dataset = build_dataset(self.config)
        print(
            f"[train] strategy={self.strategy} samples={len(dataset)} "
            "(mock loop; replace with HF Trainer)."
        )
        for idx, sample in enumerate(dataset):
            if idx >= 3:
                break
            print(f"[train] sample[{idx}]: {sample}")
        print("[train] completed mock loop.")
        return None

    def evaluate(self):
        """Mock eval loop."""
        dataset = build_dataset(self.config)
        print(f"[eval] samples={len(dataset)} (mock eval).")
        return None
