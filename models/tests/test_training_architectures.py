from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.shared.training as training
from models.shared.training import Trainer, apply_peft_if_needed, build_backbone


def _config(model_id: str, backbone_type: str, base_model: str, **training_overrides):
    return {
        "model_id": model_id,
        "backbone": {
            "type": backbone_type,
            "base_model": base_model,
            "adapter_type": "lora",
            "load_in_8bit": False,
            "load_in_4bit": False,
        },
        "dataset": {"tokenization": {"max_source_tokens": 128, "max_target_tokens": 64}},
        "training": training_overrides,
    }


def test_build_backbone_uses_seq2seq_loader_for_encoder_decoder(monkeypatch) -> None:
    def _loader(name: str):
        class _FakeLoader:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                return {"loader": name, "model_name": model_name, "kwargs": kwargs}

        return _FakeLoader

    monkeypatch.setattr(training, "_maybe_import", lambda name: _loader(name))
    monkeypatch.setattr(training, "torch", None)

    model = build_backbone(
        _config(
            "P1",
            "encoder_decoder",
            "google/flan-t5-base",
            model_type="seq2seq",
        )
    )

    assert model["loader"] == "AutoModelForSeq2SeqLM"
    assert model["model_name"] == "google/flan-t5-base"


def test_build_backbone_uses_classifier_head_for_encoder_classifier(monkeypatch) -> None:
    def _loader(name: str):
        class _FakeLoader:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                return {"loader": name, "model_name": model_name, "kwargs": kwargs}

        return _FakeLoader

    monkeypatch.setattr(training, "_maybe_import", lambda name: _loader(name))
    monkeypatch.setattr(training, "torch", None)

    model = build_backbone(
        _config(
            "M4",
            "encoder",
            "allenai/scibert_scivocab_uncased",
            model_type="classifier",
            num_labels=2,
        )
    )

    assert model["loader"] == "AutoModelForSequenceClassification"
    assert model["kwargs"]["num_labels"] == 2


def test_effective_source_token_limit_clamps_to_model_capacity() -> None:
    limit = training._effective_source_token_limit(
        _config("M1", "encoder", "allenai/scibert_scivocab_uncased"),
        tokenizer=SimpleNamespace(model_max_length=512),
        backbone=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)),
    )

    assert limit == 128

    cfg = _config("M1", "encoder", "allenai/scibert_scivocab_uncased")
    cfg["dataset"]["tokenization"]["max_source_tokens"] = 2048
    limit = training._effective_source_token_limit(
        cfg,
        tokenizer=SimpleNamespace(model_max_length=512),
        backbone=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)),
    )

    assert limit == 512


def test_effective_source_token_limit_unwraps_wrapped_backbone_configs() -> None:
    cfg = _config("M6", "encoder", "allenai/scibert_scivocab_uncased")
    cfg["dataset"]["tokenization"]["max_source_tokens"] = 2048
    wrapped = SimpleNamespace(
        config=SimpleNamespace(),
        base_model=SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=512)
            )
        ),
    )

    limit = training._effective_source_token_limit(
        cfg,
        tokenizer=SimpleNamespace(model_max_length=10**30),
        backbone=wrapped,
    )

    assert limit == 512


def test_infer_peft_task_type_tracks_architecture() -> None:
    assert training._infer_peft_task_type(
        _config("A2", "encoder_decoder", "google/flan-t5-base", model_type="seq2seq")
    ) == "SEQ_2_SEQ_LM"
    assert training._infer_peft_task_type(
        _config("M1", "encoder", "allenai/scibert_scivocab_uncased", model_type="contrastive")
    ) == "FEATURE_EXTRACTION"
    assert training._infer_peft_task_type(
        _config("A3", "encoder_decoder", "google/flan-t5-base", model_type="seq2seq")
    ) == "SEQ_2_SEQ_LM"
    assert training._infer_peft_task_type(
        _config("M4", "encoder", "allenai/scibert_scivocab_uncased", model_type="classifier")
    ) == "SEQ_CLS"
    assert training._infer_peft_task_type(
        _config("C3", "decoder", "meta-llama/Llama-3.2-1B", model_type="causal_lm")
    ) == "CAUSAL_LM"


def test_apply_peft_uses_seq2seq_task_type(monkeypatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

    fake_torch = SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setattr(training, "torch", fake_torch)

    class _FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_peft = SimpleNamespace(
        LoraConfig=_FakeLoraConfig,
        TaskType=SimpleNamespace(
            SEQ_2_SEQ_LM="SEQ_2_SEQ_LM",
            FEATURE_EXTRACTION="FEATURE_EXTRACTION",
            SEQ_CLS="SEQ_CLS",
            CAUSAL_LM="CAUSAL_LM",
        ),
        get_peft_model=lambda model, cfg: {"model": model, "config": cfg},
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    wrapped = apply_peft_if_needed(
        object(),
        _config("A2", "encoder_decoder", "google/flan-t5-base", model_type="seq2seq"),
    )

    assert wrapped["config"].kwargs["task_type"] == "SEQ_2_SEQ_LM"


def test_seq2seq_tokenizer_does_not_leak_target_into_source_prompt() -> None:
    class _RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, text=None, text_target=None, **kwargs):
            self.calls.append(text_target if text_target is not None else text)
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        def as_target_tokenizer(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    tokenizer = _RecordingTokenizer()
    trainer = Trainer(
        config=_config("A2", "encoder_decoder", "google/flan-t5-base", model_type="seq2seq"),
        model_stub=None,
        tokenizer=tokenizer,
    )

    tokenized = trainer._tokenize_fn({"text": "abstract only", "target": "method summary"})

    assert tokenizer.calls[0] == "abstract only"
    assert tokenizer.calls[1] == "method summary"
    assert tokenized["labels"] == [1, 2, 3]


def test_training_arguments_kwargs_uses_compatible_eval_argument(monkeypatch) -> None:
    class _ArgsEvalStrategy:
        def __init__(self, output_dir=None, eval_strategy=None, save_strategy=None, load_best_model_at_end=None, **kwargs):
            pass

    monkeypatch.setattr(training, "TrainingArguments", _ArgsEvalStrategy)

    kwargs = training._training_arguments_kwargs(
        {"eval_strategy": "steps", "load_best_model_at_end": True, "metric_for_best_model": "eval_loss"},
        output_dir="out",
        has_eval=True,
        use_cuda=False,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert "evaluation_strategy" not in kwargs
    assert kwargs["load_best_model_at_end"] is True


def test_training_arguments_kwargs_disables_best_model_without_eval(monkeypatch) -> None:
    class _ArgsEvaluationStrategy:
        def __init__(self, output_dir=None, evaluation_strategy=None, load_best_model_at_end=None, metric_for_best_model=None, **kwargs):
            pass

    monkeypatch.setattr(training, "TrainingArguments", _ArgsEvaluationStrategy)

    kwargs = training._training_arguments_kwargs(
        {"evaluation_strategy": "steps", "load_best_model_at_end": True, "metric_for_best_model": "eval_loss"},
        output_dir="out",
        has_eval=False,
        use_cuda=False,
    )

    assert kwargs["evaluation_strategy"] == "no"
    assert kwargs["load_best_model_at_end"] is False
    assert "metric_for_best_model" not in kwargs


def test_trainer_label_to_id_is_available() -> None:
    trainer = Trainer(config=_config("M4", "encoder", "allenai/scibert_scivocab_uncased", model_type="classifier"), model_stub=None)

    assert trainer._label_to_id("cs.AI") == 0
    assert trainer._label_to_id("cs.AI") == 0
    assert trainer._label_to_id("cs.LG") == 1


def test_contrastive_collate_preserves_batch_lengths() -> None:
    trainer = Trainer(config=_config("M1", "encoder", "allenai/scibert_scivocab_uncased", model_type="contrastive"), model_stub=None)

    texts_a, texts_b, labels = trainer._contrastive_collate(
        [
            (("paper a", "paper b"), 1),
            (("paper c", "paper d"), 0),
            (("paper e", "paper f"), 1),
        ]
    )

    assert texts_a == ["paper a", "paper c", "paper e"]
    assert texts_b == ["paper b", "paper d", "paper f"]
    assert labels == [1, 0, 1]


def test_resolve_eval_split_uses_dataset_split_when_training_split_missing() -> None:
    cfg = {
        "model_id": "P1",
        "dataset": {"construction": {"train_val_test_split": [0.8, 0.1, 0.1]}},
        "training": {},
    }

    assert training._resolve_eval_split(cfg) == 0.1


def test_normalize_dataset_splits_accepts_test_key() -> None:
    normalized = training._normalize_dataset_splits({"train": "train_ds", "test": "test_ds"})

    assert normalized == {"train": "train_ds", "eval": "test_ds"}


def test_normalize_dataset_splits_prefers_eval_key() -> None:
    normalized = training._normalize_dataset_splits({"train": "train_ds", "eval": "eval_ds", "test": "test_ds"})

    assert normalized == {"train": "train_ds", "eval": "eval_ds"}


def test_tokenize_fn_uses_numeric_labels_for_graph_link_prediction() -> None:
    class _Tokenizer:
        def __call__(self, text=None, text_target=None, **kwargs):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    trainer = Trainer(
        config=_config("M4", "encoder", "allenai/scibert_scivocab_uncased", model_type="classifier", objective="link_prediction"),
        model_stub=None,
        tokenizer=_Tokenizer(),
    )

    encoded = trainer._tokenize_fn({"text_a": "paper one", "text_b": "paper two", "label": 1})

    assert encoded["labels"] == 1


def test_contrastive_tokenize_fn_preserves_teacher_features() -> None:
    class _Tokenizer:
        def __call__(self, text=None, text_target=None, **kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    trainer = Trainer(
        config=_config("M6", "encoder", "allenai/scibert_scivocab_uncased", model_type="contrastive", distillation_weight=0.2),
        model_stub=None,
        tokenizer=_Tokenizer(),
    )

    encoded = trainer._tokenize_fn(
        {
            "text_a": "query text",
            "text_b": "paper span",
            "label": 1,
            "teacher_embedding": [0.1, 0.2, 0.3, 0.4],
            "teacher_mask": 1,
        }
    )

    assert encoded["labels"] == 1
    assert encoded["teacher_embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert encoded["teacher_mask"] == 1


def test_maybe_prepare_teacher_projection_adds_projection_for_mismatched_dims(monkeypatch) -> None:
    class _FakeLinear:
        def __init__(self, in_features, out_features, bias=False):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Linear=_FakeLinear),
    )
    monkeypatch.setattr(training, "torch", fake_torch)

    class _Backbone:
        def __init__(self):
            self.config = SimpleNamespace(hidden_size=8)

        def add_module(self, name, module):
            setattr(self, name, module)

    trainer = Trainer(
        config=_config("M6", "encoder", "allenai/scibert_scivocab_uncased", model_type="contrastive", distillation_weight=0.2),
        model_stub=None,
        backbone=_Backbone(),
    )

    trainer._maybe_prepare_teacher_projection([{"teacher_embedding": [0.1, 0.2, 0.3, 0.4]}])

    assert hasattr(trainer.backbone, "teacher_projection")
    assert trainer.backbone.teacher_projection.in_features == 8
    assert trainer.backbone.teacher_projection.out_features == 4


def test_contrastive_hf_trainer_receives_eval_dataset(monkeypatch) -> None:
    recorded = {}

    class _Args:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeHFTrainer:
        def __init__(self, *args, **kwargs):
            recorded["eval_dataset"] = kwargs.get("eval_dataset")
            recorded["train_dataset"] = kwargs.get("train_dataset")

        def train(self):
            recorded["trained"] = True

    class _Tokenizer:
        def __call__(self, text=None, text_target=None, **kwargs):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    class _FakeDataset:
        def __init__(self, rows):
            self.rows = list(rows)
            self.features = {key: None for key in self.rows[0].keys()} if self.rows else {}

        def map(self, fn, remove_columns=None):
            mapped = [fn(row) for row in self.rows]
            return _FakeDataset(mapped)

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    monkeypatch.setattr(training, "TrainingArguments", _Args)
    monkeypatch.setattr(training, "HFTrainer", _FakeHFTrainer)
    monkeypatch.setattr(training, "default_data_collator", object())
    monkeypatch.setattr(training, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))

    train_ds = _FakeDataset(
        [
            {"text_a": "query one", "text_b": "doc one", "label": 1},
            {"text_a": "query two", "text_b": "doc two", "label": 0},
        ]
    )
    eval_ds = _FakeDataset(
        [
            {"text_a": "query eval", "text_b": "doc eval", "label": 1},
        ]
    )

    trainer = Trainer(
        config=_config("M6", "encoder", "allenai/scibert_scivocab_uncased", model_type="contrastive", eval_strategy="steps"),
        model_stub=None,
        tokenizer=_Tokenizer(),
        backbone=object(),
    )
    monkeypatch.setattr(trainer, "_build_hf_dataset", lambda: {"train": train_ds, "eval": eval_ds})

    assert trainer._train_with_hf_trainer() is True
    assert recorded["train_dataset"] is not None
    assert recorded["eval_dataset"] is not None
    assert recorded["trained"] is True


def test_contrastive_hf_trainer_prediction_step_uses_custom_loss(monkeypatch) -> None:
    if training.torch is None:
        return

    recorded = {}

    class _Args:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeHFTrainer:
        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("model")
            self.eval_dataset = kwargs.get("eval_dataset")
            recorded["eval_dataset"] = self.eval_dataset

        def train(self):
            row = self.eval_dataset[0]
            batch = {}
            for key, value in row.items():
                if isinstance(value, list):
                    batch[key] = training.torch.tensor([value])
                elif isinstance(value, (int, float)):
                    batch[key] = training.torch.tensor([value])
                else:
                    batch[key] = value
            loss, logits, labels = self.prediction_step(
                self.model,
                batch,
                prediction_loss_only=False,
            )
            recorded["loss"] = loss
            recorded["logits"] = logits
            recorded["labels"] = labels

    class _Tokenizer:
        def __call__(self, text=None, text_target=None, **kwargs):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    class _FakeDataset:
        def __init__(self, rows):
            self.rows = list(rows)
            self.features = {key: None for key in self.rows[0].keys()} if self.rows else {}

        def map(self, fn, remove_columns=None):
            mapped = [fn(row) for row in self.rows]
            return _FakeDataset(mapped)

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    class _FakeModel:
        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            if "input_ids_a" in kwargs or "input_ids_b" in kwargs:
                raise AssertionError("contrastive eval should not call the model with raw dual-input batch keys")
            hidden = training.torch.tensor(input_ids, dtype=training.torch.float32).unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(training, "TrainingArguments", _Args)
    monkeypatch.setattr(training, "HFTrainer", _FakeHFTrainer)
    monkeypatch.setattr(training, "default_data_collator", object())
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)

    train_ds = _FakeDataset(
        [
            {"text_a": "query one", "text_b": "doc one", "label": 1},
            {"text_a": "query two", "text_b": "doc two", "label": 0},
        ]
    )
    eval_ds = _FakeDataset(
        [
            {"text_a": "query eval", "text_b": "doc eval", "label": 1},
        ]
    )

    trainer = Trainer(
        config=_config("M6", "encoder", "allenai/scibert_scivocab_uncased", model_type="contrastive", eval_strategy="steps"),
        model_stub=None,
        tokenizer=_Tokenizer(),
        backbone=_FakeModel(),
    )
    monkeypatch.setattr(trainer, "_build_hf_dataset", lambda: {"train": train_ds, "eval": eval_ds})

    assert trainer._train_with_hf_trainer() is True
    assert recorded["eval_dataset"] is not None
    assert recorded["logits"] is None
    assert recorded["labels"].tolist() == [1]
    assert float(recorded["loss"]) >= 0.0
