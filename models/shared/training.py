"""
Training scaffolding that can be wired to HF/PEFT/bitsandbytes.
This remains lightweight and dependency-optional; actual HF imports are lazy.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from models.shared.data import build_dataset
from models.shared.config import validate_cache_dirs
from models.shared.archetypes import get_archetype
from models.shared.code_encoder import CodeEncoder

BitsAndBytesConfig = None  # lazy import to avoid bitsandbytes errors when unavailable

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

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


def build_backbone(config: Dict[str, Any]):
    """Construct a backbone model per config; supports encoder/decoder/encoder_decoder."""
    backbone_cfg = config.get("backbone", {})
    model_type = backbone_cfg.get("type")
    base_model = backbone_cfg.get("base_model")
    training_cfg = config.get("training", {})
    from models.shared.archetypes import get_archetype
    archetype = get_archetype(config.get("model_id", "")) or {}
    # Prefer seq2seq head for certain generative IDs.
    seq2seq_ids = {"A2", "P1", "P2", "P3", "P4", "R3", "R5", "C1"}
    prefer_seq2seq = training_cfg.get("model_type") == "seq2seq" or config.get("model_id") in seq2seq_ids
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

    # Default llama-family to causal LM even if type is mislabeled.
    if model_type == "encoder" and "llama" in base_model.lower():
        model_type = "decoder"

    # Optionally switch to classification head for classifier archetypes.
    is_classifier = archetype.get("archetype") == "classifier" or training_cfg.get("model_type") == "classifier"
    num_labels = int(training_cfg.get("num_labels", 10))

    if model_type == "encoder":
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
    except Exception as exc:  # pragma: no cover - dependency optional
        raise ImportError(f"Missing PEFT dependency: {exc}")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=None,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
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

        raw = load_dataset("json", data_files=data_files)
        archetype = get_archetype(self.config.get("model_id", "")) or {}
        archetype_name = archetype.get("archetype", "generative")
        ds_quality = ds_cfg.get("quality_filters") or {}
        ds_domains = set(ds_cfg.get("domains") or [])
        training_cfg = self.config.get("training", {}) or {}
        eval_split = float(training_cfg.get("eval_split", 0.0) or 0.0)

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
                return base.train_test_split(test_size=eval_split)
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
            return mixed.train_test_split(test_size=eval_split)
        return mixed

    def _build_hf_dataset(self):
        # Prefer full-corpus HF datasets when corpus_* sources are configured.
        corpus_ds = self._build_corpus_hf_dataset()
        if corpus_ds is not None:
            return corpus_ds

        dataset = build_dataset(self.config)
        if not dataset:
            return None
        training_cfg = self.config.get("training", {})
        eval_split = float(training_cfg.get("eval_split", 0.1) or 0.0)
        if eval_split > 0 and len(dataset) > 4:
            cut = max(1, int(len(dataset) * eval_split))
            eval_list = dataset[:cut]
            train_list = dataset[cut:]
            return {
                "train": HFDataset.from_list(train_list),
                "eval": HFDataset.from_list(eval_list),
            }
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

        dl = DataLoader(ContrastiveDataset(), batch_size=batch_size, shuffle=True)
        steps = 0
        self.backbone.train()
        for batch in dl:
            if steps >= max_steps:
                break
            (text_pairs, lbls) = batch
            texts_a = [p[0] for p in text_pairs]
            texts_b = [p[1] for p in text_pairs]
            enc_a = self.tokenizer(texts_a, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            enc_b = self.tokenizer(texts_b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
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
        max_src = tok_cfg.get("max_source_tokens", 2048)
        training_cfg = self.config.get("training", {})
        archetype = get_archetype(self.config.get("model_id", "")) or {}
        objective = training_cfg.get("objective")
        prefer_seq2seq = training_cfg.get("model_type") == "seq2seq" or self.config.get("model_id") in {"A2", "P1", "P2", "P3", "P4", "R3", "R5", "C1"}
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
            if target:
                prompt += f"TARGET:\n{target}"

        if target:
            prompt = f"INPUT:\n{prompt}\nTARGET:\n{target}"

        if prefer_seq2seq and target is not None:
            model_inputs = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_src,
                padding="max_length",
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    str(target),
                    truncation=True,
                    max_length=tok_cfg.get("max_target_tokens", 512),
                    padding="max_length",
                )["input_ids"]
            model_inputs["labels"] = labels
            return model_inputs

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_src,
            padding="max_length",
        )
        # Labels handling
        if objective == "cross_entropy" and archetype.get("archetype") == "classifier":
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

        tokenized_train = train_ds.map(self._tokenize_fn, remove_columns=list(train_ds.features))
        tokenized_eval = None
        if eval_ds is not None:
            tokenized_eval = eval_ds.map(self._tokenize_fn, remove_columns=list(eval_ds.features))
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
        use_cuda = torch is not None and torch.cuda.is_available() and not force_cpu
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=training_cfg.get("num_epochs", 1),
            max_steps=training_cfg.get("max_steps", -1),
            per_device_train_batch_size=training_cfg.get("batch_size", 1),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            learning_rate=training_cfg.get("learning_rate", 5e-5),
            weight_decay=training_cfg.get("weight_decay", 0.0),
            warmup_steps=training_cfg.get("warmup_steps", 0),
            fp16=use_cuda and training_cfg.get("precision", "fp16") == "fp16",
            bf16=use_cuda and training_cfg.get("precision", "fp16") == "bf16",
            logging_steps=training_cfg.get("logging_steps", 50),
            save_steps=training_cfg.get("save_steps", 200),
            save_total_limit=1,
            remove_unused_columns=False,
            report_to=[],
            no_cuda=not use_cuda,
            evaluation_strategy=training_cfg.get("evaluation_strategy", "steps" if has_eval else "no"),
            eval_steps=training_cfg.get("eval_steps", 200),
            load_best_model_at_end=training_cfg.get("load_best_model_at_end", False),
            metric_for_best_model=training_cfg.get("metric_for_best_model", None),
        )
        # Choose trainer
        if archetype.get("archetype") == "contrastive":
            if torch is None:
                return False

            class ContrastiveTrainer(HFTrainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.pop("labels")
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
                    return (loss, (out_a, out_b)) if return_outputs else loss

            trainer = ContrastiveTrainer(
                model=self.backbone,
                args=args,
                train_dataset=tokenized_train,
                tokenizer=self.tokenizer,
                data_collator=default_data_collator,
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
        trainer.train()
        return True

    def train(self):
        """Mock train loop to keep CLI end-to-end runnable."""
        archetype = get_archetype(self.config.get("model_id", "")) or {}
        training_cfg = self.config.get("training", {}) or {}

        # Prefer contrastive loop for contrastive archetypes.
        if archetype.get("archetype") == "contrastive":
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
        if self._should_use_hf_trainer() and curriculum:
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

        if self._should_use_hf_trainer():
            try:
                if self._train_with_hf_trainer():
                    print("[train] completed via HF Trainer.")
                    return None
            except Exception as exc:  # pragma: no cover - fallback
                print(f"[warn] HF Trainer path failed, falling back to mock: {exc}")
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
