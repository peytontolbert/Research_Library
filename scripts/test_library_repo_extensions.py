from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.library_repo_graph_export import export_library
from scripts.library_repo_scanner import discover_repositories


def _create_repo(root: Path, name: str, *, module_body: str = "def value() -> int:\n    return 1\n") -> Path:
    repo = root / name
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text("[project]\nname = \"demo\"\n", encoding="utf-8")
    (repo / "module.py").write_text(module_body, encoding="utf-8")
    return repo


def _load_manifest(export_root: Path) -> dict:
    return json.loads((export_root / "_manifest.json").read_text(encoding="utf-8"))


def test_discover_repositories_prefers_primary_root_for_duplicate_repo_ids(tmp_path: Path) -> None:
    primary_root = tmp_path / "primary"
    extra_root = tmp_path / "extra"
    primary_root.mkdir()
    extra_root.mkdir()

    shared_primary = _create_repo(primary_root, "shared")
    _create_repo(extra_root, "shared", module_body="def value() -> int:\n    return 2\n")
    extra_only = _create_repo(extra_root, "extra_only")

    repos = discover_repositories(root=str(primary_root), roots=[str(extra_root)])
    by_id = {repo.repo_id: repo for repo in repos}

    assert set(by_id) == {"extra_only", "shared"}
    assert by_id["shared"].root == os.path.abspath(shared_primary)
    assert by_id["shared"].library_root == os.path.abspath(primary_root)
    assert by_id["extra_only"].root == os.path.abspath(extra_only)
    assert by_id["extra_only"].library_root == os.path.abspath(extra_root)


def test_export_library_persists_extensions_and_skips_duplicate_repo_ids(tmp_path: Path) -> None:
    primary_root = tmp_path / "primary"
    extra_root = tmp_path / "extra"
    export_root = tmp_path / "exports"
    primary_root.mkdir()
    extra_root.mkdir()

    primary_only = _create_repo(primary_root, "primary_only")
    shared_primary = _create_repo(primary_root, "shared")
    extra_only = _create_repo(extra_root, "extra_only")
    _create_repo(extra_root, "shared", module_body="def value() -> int:\n    return 3\n")

    export_library(
        library_root=str(primary_root),
        extra_library_roots=[str(extra_root)],
        export_root=str(export_root),
    )
    manifest = _load_manifest(export_root)

    assert manifest["library_roots"] == {
        "default": os.path.abspath(primary_root),
        "extensions": [os.path.abspath(extra_root)],
    }
    assert set(manifest["repos"]) == {"extra_only", "primary_only", "shared"}
    assert manifest["repos"]["primary_only"]["repo_root"] == os.path.abspath(primary_only)
    assert manifest["repos"]["shared"]["repo_root"] == os.path.abspath(shared_primary)
    assert manifest["repos"]["shared"]["library_root"] == os.path.abspath(primary_root)
    assert manifest["repos"]["extra_only"]["repo_root"] == os.path.abspath(extra_only)
    assert manifest["repos"]["extra_only"]["library_root"] == os.path.abspath(extra_root)
    manifest["repos"]["shared"]["extensions"] = {
        "demo_extension": {
            "source": "test",
            "paths": {"summary": "shared/structured/demo.summary.json"},
        }
    }

    # Simulate a legacy manifest entry that predates the new field.
    del manifest["repos"]["primary_only"]["library_root"]
    (export_root / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summaries = export_library(
        library_root=str(primary_root),
        export_root=str(export_root),
    )
    summary_by_id = {entry["repo_id"]: entry for entry in summaries}
    refreshed_manifest = _load_manifest(export_root)

    assert set(summary_by_id) == {"extra_only", "primary_only", "shared"}
    assert summary_by_id["extra_only"]["library_root"] == os.path.abspath(extra_root)
    assert summary_by_id["shared"]["root"] == os.path.abspath(shared_primary)
    assert refreshed_manifest["repos"]["primary_only"]["library_root"] == os.path.abspath(primary_root)
    assert refreshed_manifest["repos"]["shared"]["extensions"] == {
        "demo_extension": {
            "source": "test",
            "paths": {"summary": "shared/structured/demo.summary.json"},
        }
    }
