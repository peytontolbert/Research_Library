from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.shared.config import ensure_runtime_cache_env


def test_ensure_runtime_cache_env_sets_data_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("TMP", raising=False)
    monkeypatch.delenv("TEMP", raising=False)

    hf_home = tmp_path / "hf"
    tmp_dir = tmp_path / "tmp"
    result = ensure_runtime_cache_env(str(hf_home), str(tmp_dir))

    assert result["HF_HOME"] == str(hf_home)
    assert os.environ["HF_HOME"] == str(hf_home)
    assert os.environ["HF_HUB_CACHE"] == str(hf_home / "hub")
    assert os.environ["HF_DATASETS_CACHE"] == str(hf_home / "datasets")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(hf_home / "hub")
    assert os.environ["TRANSFORMERS_CACHE"] == str(hf_home / "hub")
    assert os.environ["HF_XET_CACHE"] == str(hf_home / "xet")
    assert os.environ["TMPDIR"] == str(tmp_dir)
    assert os.environ["TMP"] == str(tmp_dir)
    assert os.environ["TEMP"] == str(tmp_dir)
    assert (hf_home / "hub").is_dir()
    assert (hf_home / "datasets").is_dir()
    assert (hf_home / "xet").is_dir()
    assert tmp_dir.is_dir()
