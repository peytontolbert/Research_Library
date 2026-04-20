from __future__ import annotations

from pathlib import Path

from modules import qa_runtime


class _DummyTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, dict(kwargs)))
        return cls()


class _DummyModel:
    calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, dict(kwargs)))
        return cls()

    def eval(self):
        return self


class _DummyTorch:
    float16 = "float16"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


def test_get_or_load_model_prefers_local_hf_snapshot(monkeypatch, tmp_path: Path) -> None:
    repo_dir = tmp_path / "models--meta-llama--Llama-3.1-8B-Instruct"
    snapshot_id = "abc123"
    snapshot_dir = repo_dir / "snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(snapshot_id, encoding="utf-8")

    _DummyTokenizer.calls = []
    _DummyModel.calls = []
    monkeypatch.setattr(qa_runtime, "AutoTokenizer", _DummyTokenizer)
    monkeypatch.setattr(qa_runtime, "AutoModelForCausalLM", _DummyModel)
    monkeypatch.setattr(qa_runtime, "BitsAndBytesConfig", None)
    monkeypatch.setattr(qa_runtime, "PeftModel", None)
    monkeypatch.setattr(qa_runtime, "torch", _DummyTorch())
    monkeypatch.setattr(qa_runtime, "_MODEL_CACHE", {})

    cfg = qa_runtime.QAModelConfig(
        model_name="llama",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_path=None,
        cache_dir=str(tmp_path),
        quantization="4bit",
        repo_lora_path=None,
        task_lora_path=None,
        max_new_tokens=32,
        temperature=0.1,
        top_p=0.95,
        infer_devices=[0, 1],
    )

    model, tokenizer = qa_runtime.get_or_load_model(cfg)

    assert isinstance(model, _DummyModel)
    assert isinstance(tokenizer, _DummyTokenizer)
    assert _DummyTokenizer.calls[0][0] == str(snapshot_dir)
    assert _DummyModel.calls[0][0] == str(snapshot_dir)
    assert _DummyTokenizer.calls[0][1]["local_files_only"] is True
    assert _DummyModel.calls[0][1]["local_files_only"] is True
    assert _DummyTokenizer.calls[0][1]["cache_dir"] == str(tmp_path)
    assert _DummyModel.calls[0][1]["cache_dir"] == str(tmp_path)


def test_get_default_qa_base_config_honors_llama_model_path_env(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_MODEL_PATH", "/data/checkpoints/local-llama")
    cfg = qa_runtime.get_default_qa_base_config()
    assert cfg.model_path == "/data/checkpoints/local-llama"


def test_get_model_config_from_adapter_prefers_base_model_id_when_only_base_snapshot_exists(
    monkeypatch, tmp_path: Path
) -> None:
    repo_dir = tmp_path / "models--meta-llama--Llama-3.1-8B-Instruct"
    snapshot_id = "snap123"
    (repo_dir / "snapshots" / snapshot_id).mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(snapshot_id, encoding="utf-8")

    qa_meta = {
        "model_name": "llama",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "cache_dir": str(tmp_path),
    }

    cfg = qa_runtime.get_model_config_from_adapter(qa_meta)
    assert cfg.model_id == "meta-llama/Llama-3.1-8B-Instruct"
