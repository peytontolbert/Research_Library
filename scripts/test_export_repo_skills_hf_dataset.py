from __future__ import annotations

import json
from pathlib import Path

from scripts.export_repo_skills_hf_dataset import export_repo_skills_hf_dataset


def test_export_repo_skills_hf_dataset_writes_dataset_and_parquet(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    repo_dir = export_root / "demo" / "structured"
    repo_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "repo_id": "demo",
        "imported_at": 123,
        "source": {
            "miner_repo_id": "miner-demo",
            "miner_repo_name": "demo",
            "miner_revision_id": "rev-demo",
            "miner_revision": "deadbeef",
        },
        "counts": {
            "skills": 1,
            "annotations": 1,
            "annotated_skills": 1,
            "signals": 1,
            "skill_kinds": {"function": 1},
        },
        "annotation_models": ["demo-model"],
        "signal_kinds": ["repo_signals"],
        "top_annotation_summaries": [{"summary": "Runs demo"}],
        "signal_summaries": [{"kind": "repo_signals", "summary": "pytest"}],
    }
    (repo_dir / "repo_skills_miner.summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (repo_dir / "repo_skills_miner.skills.jsonl").write_text(
        json.dumps(
            {
                "repo_id": "demo",
                "source": "repo_skills_miner",
                "miner_repo_id": "miner-demo",
                "miner_revision_id": "rev-demo",
                "skill_id": "skill-demo",
                "kind": "function",
                "module": "demo.module",
                "qualname": "run_demo",
                "signature": "(x)",
                "file_path": "demo/module.py",
                "line_start": 10,
                "line_end": 20,
                "doc_text": "doc text",
                "snippet": "def run_demo(x): return x",
                "has_annotation": True,
                "annotation_id": "ann-demo",
                "annotation_summary": "Runs demo",
                "annotation_confidence": 0.9,
                "annotation_model_id": "demo-model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_dir / "repo_skills_miner.annotations.jsonl").write_text(
        json.dumps(
            {
                "repo_id": "demo",
                "source": "repo_skills_miner",
                "miner_repo_id": "miner-demo",
                "miner_revision_id": "rev-demo",
                "skill_id": "skill-demo",
                "annotation_id": "ann-demo",
                "model_id": "demo-model",
                "summary": "Runs demo",
                "confidence": 0.9,
                "created_ms": 456,
                "kind": "function",
                "module": "demo.module",
                "qualname": "run_demo",
                "signature": "(x)",
                "file_path": "demo/module.py",
                "line_start": 10,
                "line_end": 20,
                "annotation": {"summary": "Runs demo", "when_to_use": ["testing"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_dir / "repo_skills_miner.signals.jsonl").write_text(
        json.dumps(
            {
                "repo_id": "demo",
                "source": "repo_skills_miner",
                "miner_repo_id": "miner-demo",
                "miner_revision_id": "rev-demo",
                "signals_id": "sig-demo",
                "revision_id": "rev-demo",
                "kind": "repo_signals",
                "summary": "pytest",
                "created_ms": 789,
                "signals": {"kind": "repo_signals", "detected": {"test_tools": ["pytest"]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "repos": {
            "demo": {
                "repo_root": "/tmp/demo",
                "extensions": {
                    "repo_skills_miner": {
                        "source": "repo_skills_miner",
                        "imported_at": 123,
                        "miner_repo_id": "miner-demo",
                        "miner_revision_id": "rev-demo",
                        "paths": {
                            "summary": "demo/structured/repo_skills_miner.summary.json",
                            "skills": "demo/structured/repo_skills_miner.skills.jsonl",
                            "annotations": "demo/structured/repo_skills_miner.annotations.jsonl",
                            "signals": "demo/structured/repo_skills_miner.signals.jsonl",
                        },
                    }
                },
            }
        }
    }
    (export_root / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out_dir = tmp_path / "hf_out"
    result = export_repo_skills_hf_dataset(
        export_root=str(export_root),
        output_dir=str(out_dir),
    )

    stats = result["stats"]
    assert stats["skill_count"] == 1
    assert stats["annotation_count"] == 1
    assert stats["signal_count"] == 1
    assert stats["public_safe"] is True

    assert (out_dir / "dataset_dict").is_dir()
    assert (out_dir / "parquet" / "train.parquet").is_file()
    assert (out_dir / "parquet" / "repos.parquet").is_file()
    assert (out_dir / "parquet" / "annotations.parquet").is_file()
    assert (out_dir / "parquet" / "signals.parquet").is_file()
    assert (out_dir / "README.md").is_file()
    assert (out_dir / "stats.json").is_file()

    from datasets import load_from_disk  # type: ignore

    ds = load_from_disk(str(out_dir / "dataset_dict"))
    train_row = ds["train"][0]
    assert train_row["repo_id"] == "demo"
    assert train_row["annotation_summary"] == "Runs demo"
    assert "snippet" not in train_row
    assert "doc_text" not in train_row

    ann_row = ds["annotations"][0]
    assert ann_row["annotation_json"] == "{\"summary\":\"Runs demo\",\"when_to_use\":[\"testing\"]}"

    signal_row = ds["signals"][0]
    assert signal_row["signals_json"] == "{\"detected\":{\"test_tools\":[\"pytest\"]},\"kind\":\"repo_signals\"}"
