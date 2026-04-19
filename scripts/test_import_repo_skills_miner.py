from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.import_repo_skills_miner import import_repo_skills_miner


def _write_blob(store_root: Path, sha256: str, payload: dict) -> None:
    path = store_root / "blobs" / sha256[:2] / sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_import_repo_skills_miner_writes_structured_exports_and_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repos" / "demo"
    repo_root.mkdir(parents=True, exist_ok=True)

    export_root = tmp_path / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repos": {
            "demo": {
                "repo_root": str(repo_root.resolve()),
                "repo_state": {"vcs": "git", "head": "abc"},
                "indices": {},
                "skills": {},
            }
        }
    }
    (export_root / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    miner_root = tmp_path / "miner"
    miner_store = miner_root / "skill_engine_store"
    (miner_store / "bundles").mkdir(parents=True, exist_ok=True)
    db_path = miner_root / "skill_engine.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE repos (
            repo_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            created_ms INTEGER NOT NULL
        );
        CREATE TABLE revisions (
            revision_id TEXT PRIMARY KEY,
            repo_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            created_ms INTEGER NOT NULL
        );
        CREATE TABLE skills (
            skill_id TEXT PRIMARY KEY,
            revision_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            module TEXT NOT NULL,
            qualname TEXT NOT NULL,
            signature TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            doc_blob_sha256 TEXT NOT NULL,
            snippet_blob_sha256 TEXT NOT NULL,
            created_ms INTEGER NOT NULL
        );
        CREATE TABLE annotations (
            annotation_id TEXT PRIMARY KEY,
            skill_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            cache_dir TEXT NOT NULL,
            offline INTEGER NOT NULL,
            prompt_blob_sha256 TEXT NOT NULL,
            response_blob_sha256 TEXT NOT NULL,
            annotation_blob_sha256 TEXT NOT NULL,
            summary TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_ms INTEGER NOT NULL
        );
        CREATE TABLE revision_signals (
            signals_id TEXT PRIMARY KEY,
            revision_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            signals_blob_sha256 TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_ms INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO repos(repo_id, path, name, created_ms) VALUES (?, ?, ?, ?)",
        ("miner-demo", str(repo_root.resolve()), "demo", 1),
    )
    conn.execute(
        "INSERT INTO revisions(revision_id, repo_id, revision, content_sha256, created_ms) VALUES (?, ?, ?, ?, ?)",
        ("rev-demo", "miner-demo", "deadbeef", "f" * 64, 2),
    )
    conn.execute(
        """
        INSERT INTO skills(
            skill_id, revision_id, kind, module, qualname, signature,
            file_path, line_start, line_end, doc_blob_sha256, snippet_blob_sha256, created_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "skill-demo",
            "rev-demo",
            "function",
            "demo.module",
            "run_demo",
            "(x)",
            "demo/module.py",
            10,
            20,
            "0" * 64,
            "1" * 64,
            3,
        ),
    )
    conn.execute(
        """
        INSERT INTO annotations(
            annotation_id, skill_id, model_id, cache_dir, offline,
            prompt_blob_sha256, response_blob_sha256, annotation_blob_sha256,
            summary, confidence, created_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ann-demo",
            "skill-demo",
            "demo-model",
            "/tmp/cache",
            1,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "Runs the demo pipeline.",
            0.95,
            4,
        ),
    )
    conn.execute(
        """
        INSERT INTO revision_signals(
            signals_id, revision_id, kind, signals_blob_sha256, summary, created_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "sig-demo",
            "rev-demo",
            "repo_signals",
            "5" * 64,
            "build=pyproject tests=pytest",
            5,
        ),
    )
    conn.commit()
    conn.close()

    _write_blob(
        miner_store,
        "4" * 64,
        {
            "summary": "Runs the demo pipeline.",
            "when_to_use": ["When testing the importer."],
            "inputs": {"x": "demo input"},
        },
    )
    _write_blob(
        miner_store,
        "5" * 64,
        {
            "kind": "repo_signals",
            "repo_name": "demo",
            "detected": {"test_tools": ["pytest"]},
        },
    )

    bundle_path = miner_store / "bundles" / "rev-demo.jsonl"
    bundle_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "repo",
                        "repo_id": "miner-demo",
                        "repo_path": str(repo_root.resolve()),
                        "repo_name": "demo",
                        "revision": "deadbeef",
                        "revision_id": "rev-demo",
                        "content_sha256": "f" * 64,
                        "created_ms": 2,
                    }
                ),
                json.dumps(
                    {
                        "type": "skill",
                        "skill_id": "skill-demo",
                        "kind": "function",
                        "module": "demo.module",
                        "qualname": "run_demo",
                        "signature": "(x)",
                        "file_path": "demo/module.py",
                        "line_start": 10,
                        "line_end": 20,
                        "doc_text": "doc text",
                        "snippet": "def run_demo(x):\n    return x\n",
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    results = import_repo_skills_miner(
        miner_db=str(db_path),
        miner_store=str(miner_store),
        export_root=str(export_root),
    )

    assert any(row.get("repo_id") == "demo" and row.get("status") == "imported" for row in results)

    manifest_after = json.loads((export_root / "_manifest.json").read_text(encoding="utf-8"))
    ext = manifest_after["repos"]["demo"]["extensions"]["repo_skills_miner"]
    assert ext["miner_revision_id"] == "rev-demo"
    assert ext["counts"]["skills"] == 1
    assert ext["counts"]["annotations"] == 1
    assert ext["counts"]["signals"] == 1

    summary_path = export_root / ext["paths"]["summary"]
    skills_path = export_root / ext["paths"]["skills"]
    annotations_path = export_root / ext["paths"]["annotations"]
    signals_path = export_root / ext["paths"]["signals"]
    assert summary_path.is_file()
    assert skills_path.is_file()
    assert annotations_path.is_file()
    assert signals_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["repo_id"] == "demo"
    assert summary["counts"]["skills"] == 1
    assert summary["annotation_models"] == ["demo-model"]
    assert summary["top_annotation_summaries"][0]["summary"] == "Runs the demo pipeline."

    skill_row = json.loads(skills_path.read_text(encoding="utf-8").splitlines()[0])
    assert skill_row["repo_id"] == "demo"
    assert skill_row["snippet"].startswith("def run_demo")
    assert skill_row["annotation_summary"] == "Runs the demo pipeline."

    annotation_row = json.loads(annotations_path.read_text(encoding="utf-8").splitlines()[0])
    assert annotation_row["annotation"]["when_to_use"] == ["When testing the importer."]
    assert annotation_row["confidence"] == 0.95

    signal_row = json.loads(signals_path.read_text(encoding="utf-8").splitlines()[0])
    assert signal_row["signals"]["detected"]["test_tools"] == ["pytest"]
