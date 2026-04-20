from __future__ import annotations

"""
Export repository graph artifacts from the library exports root as a
Hugging Face-ready parquet dataset with optional Hub push.

This exporter covers two related data products when available:

- per-repo raw graph exports under `exports/<repo_id>/`
  - `<repo_id>.entities.jsonl`
  - `<repo_id>.edges.jsonl`
  - `<repo_id>.artifacts.jsonl`
- the aggregated universe under `exports/_universe/`
  - `nodes.jsonl`
  - `edges.jsonl`
  - `repo_knn_edges.jsonl`
  - optional coordinates / embeddings arrays

To keep memory bounded, all splits are streamed directly to parquet.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from datasets import Dataset, Features, Sequence as HFSequence, Value  # type: ignore
from huggingface_hub import HfApi, get_token  # type: ignore
from huggingface_hub.errors import HfHubHTTPError  # type: ignore

from scripts.library_repo_graph_export import DEFAULT_EXPORT_ROOT


DEFAULT_OUTPUT_DIR = Path("exports/huggingface/library_repo_graph_universe_v1")
DEFAULT_PARQUET_BATCH_ROWS = 4096
UNIVERSE_DIRNAME = "_universe"
_MANIFEST_FILENAME = "_manifest.json"
_UNIVERSE_ASSET_FILENAMES = (
    "universe_3d.png",
    "universe_3d_detailed.png",
    "nodes_3d_sample.html",
    "universe_3d_hover.html",
)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _sanitize_nested_paths(value: Any, export_root: Path) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_nested_paths(v, export_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nested_paths(v, export_root) for v in value]
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        path = Path(text)
    except Exception:
        return value
    if not path.is_absolute():
        return value
    try:
        resolved = path.resolve()
        export_root_resolved = export_root.resolve()
    except Exception:
        return ""
    if resolved == export_root_resolved or export_root_resolved in resolved.parents:
        return str(resolved.relative_to(export_root_resolved))
    return ""


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _without_broken_torch() -> Tuple[Any, bool]:
    broken_torch = sys.modules.get("torch")
    removed = False
    if broken_torch is not None and not hasattr(broken_torch, "Tensor"):
        removed = True
        sys.modules.pop("torch", None)
    return broken_torch, removed


def _restore_torch(broken_torch: Any, removed: bool) -> None:
    if removed and broken_torch is not None:
        sys.modules["torch"] = broken_torch


def _dataset_from_parquet(parquet_path: Path) -> Dataset:
    broken_torch, removed_torch = _without_broken_torch()
    try:
        return Dataset.from_parquet(str(parquet_path))
    finally:
        _restore_torch(broken_torch, removed_torch)


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    explicit = str(token or "").strip()
    if explicit:
        return explicit
    cached = get_token()
    if isinstance(cached, str) and cached.strip():
        return cached.strip()
    return None


def _size_category(total_rows: int) -> str:
    if total_rows < 1_000:
        return "n<1K"
    if total_rows < 10_000:
        return "1K<n<10K"
    if total_rows < 100_000:
        return "10K<n<100K"
    if total_rows < 1_000_000:
        return "100K<n<1M"
    if total_rows < 10_000_000:
        return "1M<n<10M"
    return "10M<n<100M"


def _repo_features() -> Features:
    return Features(
        {
            "repo_id": Value("string"),
            "export_schema_version": Value("int64"),
            "repo_vcs": Value("string"),
            "repo_head": Value("string"),
            "repo_branch": Value("string"),
            "snapshot_mtime": Value("int64"),
            "last_indexed_at": Value("int64"),
            "languages": HFSequence(Value("string")),
            "index_names": HFSequence(Value("string")),
            "skill_names": HFSequence(Value("string")),
            "extension_names": HFSequence(Value("string")),
            "indices_json": Value("string"),
            "skills_json": Value("string"),
            "extensions_json": Value("string"),
            "has_entities": Value("bool"),
            "has_edges": Value("bool"),
            "has_artifacts": Value("bool"),
            "in_universe": Value("bool"),
            "universe_repo_x": Value("float32"),
            "universe_repo_y": Value("float32"),
            "universe_repo_z": Value("float32"),
        }
    )


def _entity_features() -> Features:
    return Features(
        {
            "repo_id": Value("string"),
            "id": Value("string"),
            "uri": Value("string"),
            "kind": Value("string"),
            "name": Value("string"),
            "owner": Value("string"),
        }
    )


def _edge_features() -> Features:
    return Features(
        {
            "repo_id": Value("string"),
            "src": Value("string"),
            "dst": Value("string"),
            "type": Value("string"),
        }
    )


def _artifact_features() -> Features:
    return Features(
        {
            "repo_id": Value("string"),
            "uri": Value("string"),
            "type": Value("string"),
            "hash": Value("string"),
        }
    )


def _universe_node_features() -> Features:
    return Features(
        {
            "node_id": Value("string"),
            "repo_id": Value("string"),
            "kind": Value("string"),
            "name": Value("string"),
            "uri": Value("string"),
            "labels": HFSequence(Value("string")),
            "x": Value("float32"),
            "y": Value("float32"),
            "z": Value("float32"),
        }
    )


def _repo_knn_features() -> Features:
    return Features(
        {
            "src_repo": Value("string"),
            "dst_repo": Value("string"),
            "weight": Value("float32"),
        }
    )


def _node_embedding_features() -> Features:
    return Features(
        {
            "node_id": Value("string"),
            "repo_id": Value("string"),
            "embedding": HFSequence(Value("float32")),
        }
    )


class _SplitWriter:
    def __init__(self, path: Path, features: Features, *, batch_rows: int) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = features.arrow_schema
        self.writer = pq.ParquetWriter(str(path), self.schema, compression="snappy")
        self.batch_rows = max(1, int(batch_rows))
        self.rows: List[Dict[str, Any]] = []
        self.count = 0

    def write(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def _discover_universe_assets(universe_root: Path) -> List[Path]:
    paths: List[Path] = []
    for name in _UNIVERSE_ASSET_FILENAMES:
        path = universe_root / name
        if path.is_file():
            paths.append(path)
    return paths


def _load_universe_repo_coords(universe_root: Path, universe_manifest: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    repo_ids = [str(repo_id) for repo_id in universe_manifest.get("repo_ids") or [] if str(repo_id).strip()]
    if not repo_ids:
        return {}
    repo_coords_path = universe_root / "repo_coords.npy"
    if not repo_coords_path.is_file():
        return {}
    repo_coords = np.load(repo_coords_path, mmap_mode="r")
    if repo_coords.shape[0] != len(repo_ids):
        raise RuntimeError(
            f"Universe repo_coords.npy row count {repo_coords.shape[0]} does not match repo_ids length {len(repo_ids)}."
        )
    coords_by_id: Dict[str, Tuple[float, float, float]] = {}
    for idx, repo_id in enumerate(repo_ids):
        row = repo_coords[idx]
        coords_by_id[repo_id] = (
            float(row[0]) if row.shape[0] > 0 else 0.0,
            float(row[1]) if row.shape[0] > 1 else 0.0,
            float(row[2]) if row.shape[0] > 2 else 0.0,
        )
    return coords_by_id


def _write_dataset_card(
    *,
    output_dir: Path,
    split_counts: Dict[str, int],
    repo_count: int,
    universe_repo_count: int,
    include_node_embeddings: bool,
    asset_names: Sequence[str],
) -> Path:
    total_rows = int(sum(split_counts.values()))
    default_config = "universe_nodes" if "universe_nodes" in split_counts else next(iter(split_counts.keys()), "repos")
    yaml_configs = "\n".join(
        [
            "configs:",
            *[
                "\n".join(
                    [
                        f"- config_name: {name}",
                        f"  default: {str(name == default_config).lower()}",
                        "  data_files:",
                        "  - split: train",
                        f"    path: \"{name}/*.parquet\"",
                    ]
                )
                for name in split_counts
            ],
        ]
    )
    configs = "\n".join(f"- `{name}`: `{count}` rows" for name, count in split_counts.items())
    loading = "\n".join(
        f'{name} = load_dataset("YOUR_NAMESPACE/YOUR_DATASET", "{name}")'
        for name in split_counts
    )
    image_blocks: List[str] = []
    if "universe_3d.png" in asset_names:
        image_blocks.append("### Universe 3D Overview\n\n![Universe 3D overview](./universe_3d.png)")
    if "universe_3d_detailed.png" in asset_names:
        image_blocks.append("### Universe 3D Detailed View\n\n![Universe 3D detailed view](./universe_3d_detailed.png)")
    if "nodes_3d_sample.html" in asset_names:
        image_blocks.append("- [Open the sampled 3D HTML view](./nodes_3d_sample.html)")
    if "universe_3d_hover.html" in asset_names:
        image_blocks.append("- [Open the interactive hover view](./universe_3d_hover.html)")
    asset_note = ""
    if asset_names:
        listed_assets = "\n".join(f"- `{name}`" for name in asset_names)
        rendered_assets = "\n\n".join(image_blocks).strip()
        if rendered_assets:
            rendered_assets = f"\n\n{rendered_assets}\n"
        asset_note = (
            "\n## Universe Assets\n\n"
            "The local export also includes these visualization assets when present:\n\n"
            f"{listed_assets}"
            f"{rendered_assets}"
        )
    readme = f"""---
pretty_name: Repository Graph Universe
viewer: true
tags:
- datasets
- code
- graph
- software-engineering
- repository-graphs
size_categories:
- {_size_category(total_rows)}
{yaml_configs}
---

# Repository Graph Universe Dataset

Parquet-first export of the repository graph artifacts already built under the Repository Library.

This dataset preserves:

- per-repo raw graph exports (`entities`, `edges`, `artifacts`)
- repo manifest metadata (`repos`)
- the aggregated universe graph when available (`universe_nodes`, `universe_edges`, `repo_knn`)

## Coverage

- repos in library manifest: `{repo_count}`
- repos represented in universe: `{universe_repo_count}`
- node embeddings split included: `{str(include_node_embeddings).lower()}`

## Configs

- default viewer/config: `{default_config}`
{configs}

## Loading

```python
from datasets import load_dataset

{loading}
```

## Notes

- `repos` joins universe repo coordinates into the repo metadata rows when available.
- `universe_nodes` includes `x`, `y`, `z` coordinates from the existing 3D projection.
- `artifacts` only contains program URIs and hashes, not file contents.
{asset_note}
"""
    path = output_dir / "README.md"
    path.write_text(readme, encoding="utf-8")
    return path


def export_library_repo_graph_hf_dataset(
    *,
    export_root: str = DEFAULT_EXPORT_ROOT,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    include_universe: bool = True,
    include_node_embeddings: bool = False,
    parquet_batch_rows: int = DEFAULT_PARQUET_BATCH_ROWS,
    push_repo_id: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    export_root_path = Path(export_root).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    parquet_dir = output_dir_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(export_root_path / _MANIFEST_FILENAME)
    repos_meta = manifest.get("repos") or {}
    if not isinstance(repos_meta, dict) or not repos_meta:
        raise RuntimeError(f"No repository manifest entries found under {export_root_path}.")

    universe_root = export_root_path / UNIVERSE_DIRNAME
    universe_manifest = _load_json(universe_root / "manifest.json") if include_universe else {}
    universe_available = bool(include_universe and universe_manifest)
    universe_repo_coords = _load_universe_repo_coords(universe_root, universe_manifest) if universe_available else {}
    universe_asset_paths = _discover_universe_assets(universe_root) if universe_available else []

    split_writers: Dict[str, _SplitWriter] = {
        "repos": _SplitWriter(parquet_dir / "repos.parquet", _repo_features(), batch_rows=parquet_batch_rows),
        "entities": _SplitWriter(parquet_dir / "entities.parquet", _entity_features(), batch_rows=parquet_batch_rows),
        "edges": _SplitWriter(parquet_dir / "edges.parquet", _edge_features(), batch_rows=parquet_batch_rows),
        "artifacts": _SplitWriter(parquet_dir / "artifacts.parquet", _artifact_features(), batch_rows=parquet_batch_rows),
    }
    if universe_available:
        split_writers["universe_nodes"] = _SplitWriter(
            parquet_dir / "universe_nodes.parquet",
            _universe_node_features(),
            batch_rows=parquet_batch_rows,
        )
        split_writers["universe_edges"] = _SplitWriter(
            parquet_dir / "universe_edges.parquet",
            _edge_features(),
            batch_rows=parquet_batch_rows,
        )
        split_writers["repo_knn"] = _SplitWriter(
            parquet_dir / "repo_knn.parquet",
            _repo_knn_features(),
            batch_rows=parquet_batch_rows,
        )
        if include_node_embeddings:
            split_writers["universe_node_embeddings"] = _SplitWriter(
                parquet_dir / "universe_node_embeddings.parquet",
                _node_embedding_features(),
                batch_rows=max(16, min(parquet_batch_rows, 256)),
            )

    try:
        for repo_id in sorted(str(repo) for repo in repos_meta.keys()):
            entry_any = repos_meta.get(repo_id) or {}
            entry = entry_any if isinstance(entry_any, dict) else {}
            repo_state = entry.get("repo_state") or {}
            repo_state = repo_state if isinstance(repo_state, dict) else {}
            indices = _sanitize_nested_paths(entry.get("indices") or {}, export_root_path)
            skills = _sanitize_nested_paths(entry.get("skills") or {}, export_root_path)
            extensions = _sanitize_nested_paths(entry.get("extensions") or {}, export_root_path)

            repo_dir = export_root_path / repo_id
            entities_path = repo_dir / f"{repo_id}.entities.jsonl"
            edges_path = repo_dir / f"{repo_id}.edges.jsonl"
            artifacts_path = repo_dir / f"{repo_id}.artifacts.jsonl"
            coords = universe_repo_coords.get(repo_id)

            split_writers["repos"].write(
                {
                    "repo_id": repo_id,
                    "export_schema_version": int(entry.get("export_schema_version") or 0),
                    "repo_vcs": str(repo_state.get("vcs") or ""),
                    "repo_head": str(repo_state.get("head") or ""),
                    "repo_branch": str(repo_state.get("branch") or ""),
                    "snapshot_mtime": int(repo_state.get("snapshot_mtime") or 0),
                    "last_indexed_at": int(entry.get("last_indexed_at") or 0),
                    "languages": _string_list(entry.get("languages") or []),
                    "index_names": sorted(str(k) for k in indices.keys()),
                    "skill_names": sorted(str(k) for k in skills.keys()),
                    "extension_names": sorted(str(k) for k in extensions.keys()),
                    "indices_json": _compact_json(indices),
                    "skills_json": _compact_json(skills),
                    "extensions_json": _compact_json(extensions),
                    "has_entities": entities_path.is_file(),
                    "has_edges": edges_path.is_file(),
                    "has_artifacts": artifacts_path.is_file(),
                    "in_universe": coords is not None,
                    "universe_repo_x": float(coords[0]) if coords is not None else None,
                    "universe_repo_y": float(coords[1]) if coords is not None else None,
                    "universe_repo_z": float(coords[2]) if coords is not None else None,
                }
            )

            if entities_path.is_file():
                for row in _iter_jsonl(entities_path):
                    split_writers["entities"].write(
                        {
                            "repo_id": str(row.get("repo_id") or repo_id),
                            "id": str(row.get("id") or ""),
                            "uri": str(row.get("uri") or ""),
                            "kind": str(row.get("kind") or ""),
                            "name": str(row.get("name") or ""),
                            "owner": str(row.get("owner") or ""),
                        }
                    )
            if edges_path.is_file():
                for row in _iter_jsonl(edges_path):
                    split_writers["edges"].write(
                        {
                            "repo_id": str(row.get("repo_id") or repo_id),
                            "src": str(row.get("src") or ""),
                            "dst": str(row.get("dst") or ""),
                            "type": str(row.get("type") or ""),
                        }
                    )
            if artifacts_path.is_file():
                for row in _iter_jsonl(artifacts_path):
                    split_writers["artifacts"].write(
                        {
                            "repo_id": str(row.get("repo_id") or repo_id),
                            "uri": str(row.get("uri") or ""),
                            "type": str(row.get("type") or ""),
                            "hash": str(row.get("hash") or ""),
                        }
                    )

        if universe_available:
            node_coords_path = universe_root / "node_coords.npy"
            node_coords = np.load(node_coords_path, mmap_mode="r") if node_coords_path.is_file() else None
            node_embeddings_path = universe_root / "node_embeddings.npy"
            node_embeddings = (
                np.load(node_embeddings_path, mmap_mode="r")
                if include_node_embeddings and node_embeddings_path.is_file()
                else None
            )
            expected_nodes = int(universe_manifest.get("node_count") or 0)
            if node_coords is not None and expected_nodes and node_coords.shape[0] != expected_nodes:
                raise RuntimeError(
                    f"Universe node_coords.npy row count {node_coords.shape[0]} does not match manifest node_count {expected_nodes}."
                )
            if node_embeddings is not None and expected_nodes and node_embeddings.shape[0] != expected_nodes:
                raise RuntimeError(
                    f"Universe node_embeddings.npy row count {node_embeddings.shape[0]} does not match manifest node_count {expected_nodes}."
                )

            node_count = 0
            nodes_path = universe_root / "nodes.jsonl"
            for row in _iter_jsonl(nodes_path):
                coords = None
                if node_coords is not None:
                    coords = node_coords[node_count]
                split_writers["universe_nodes"].write(
                    {
                        "node_id": str(row.get("node_id") or ""),
                        "repo_id": str(row.get("repo_id") or ""),
                        "kind": str(row.get("kind") or ""),
                        "name": str(row.get("name") or ""),
                        "uri": str(row.get("uri") or ""),
                        "labels": _string_list(row.get("labels")),
                        "x": float(coords[0]) if coords is not None and coords.shape[0] > 0 else None,
                        "y": float(coords[1]) if coords is not None and coords.shape[0] > 1 else None,
                        "z": float(coords[2]) if coords is not None and coords.shape[0] > 2 else None,
                    }
                )
                if include_node_embeddings and node_embeddings is not None:
                    vec = node_embeddings[node_count]
                    split_writers["universe_node_embeddings"].write(
                        {
                            "node_id": str(row.get("node_id") or ""),
                            "repo_id": str(row.get("repo_id") or ""),
                            "embedding": [float(v) for v in vec.tolist()],
                        }
                    )
                node_count += 1
            if expected_nodes and node_count != expected_nodes:
                raise RuntimeError(
                    f"Universe nodes.jsonl row count {node_count} does not match manifest node_count {expected_nodes}."
                )

            universe_edges_path = universe_root / "edges.jsonl"
            for row in _iter_jsonl(universe_edges_path):
                split_writers["universe_edges"].write(
                    {
                        "repo_id": str(row.get("repo_id") or ""),
                        "src": str(row.get("src") or ""),
                        "dst": str(row.get("dst") or ""),
                        "type": str(row.get("type") or ""),
                    }
                )

            repo_knn_path = universe_root / "repo_knn_edges.jsonl"
            for row in _iter_jsonl(repo_knn_path):
                split_writers["repo_knn"].write(
                    {
                        "src_repo": str(row.get("src_repo") or ""),
                        "dst_repo": str(row.get("dst_repo") or ""),
                        "weight": float(row.get("weight") or 0.0),
                    }
                )
    finally:
        for writer in split_writers.values():
            writer.close()

    split_counts = {name: int(writer.count) for name, writer in split_writers.items()}
    split_parquet_paths = {name: str(writer.path) for name, writer in split_writers.items()}
    asset_names = [path.name for path in universe_asset_paths]
    readme_path = _write_dataset_card(
        output_dir=output_dir_path,
        split_counts=split_counts,
        repo_count=len(repos_meta),
        universe_repo_count=len(universe_repo_coords),
        include_node_embeddings=bool(include_node_embeddings),
        asset_names=asset_names,
    )

    stats = {
        "export_root": str(export_root_path),
        "output_dir": str(output_dir_path),
        "repo_count_manifest": len(repos_meta),
        "universe_available": bool(universe_available),
        "universe_repo_count": len(universe_repo_coords),
        "include_node_embeddings": bool(include_node_embeddings),
        "parquet_batch_rows": int(parquet_batch_rows),
        "splits": split_counts,
        "parquet_paths": split_parquet_paths,
        "universe_assets": asset_names,
        "readme_path": str(readme_path),
    }
    stats_path = output_dir_path / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    push_info: Optional[Dict[str, Any]] = None
    if push_repo_id:
        resolved_token = _resolve_hf_token(token)
        if not resolved_token:
            raise RuntimeError(
                "Hugging Face push requested but no token was found. "
                "Set HF_TOKEN, pass --token hf_..., or login first via huggingface_hub."
            )
        api = HfApi(token=resolved_token)
        default_config = "universe_nodes" if "universe_nodes" in split_parquet_paths else "entities"
        pushed_configs: List[Dict[str, Any]] = []
        uploaded_assets: List[str] = []
        try:
            api.whoami(token=resolved_token)
            api.create_repo(
                repo_id=push_repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            split_order = [default_config] + [name for name in split_parquet_paths.keys() if name != default_config]
            for split_name in split_order:
                dataset = _dataset_from_parquet(Path(split_parquet_paths[split_name]))
                dataset.push_to_hub(
                    repo_id=push_repo_id,
                    config_name=split_name,
                    split="train",
                    set_default=bool(split_name == default_config),
                    private=private,
                    token=resolved_token,
                )
                pushed_configs.append(
                    {
                        "config_name": split_name,
                        "split": "train",
                        "rows": int(dataset.num_rows),
                    }
                )
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=push_repo_id,
                repo_type="dataset",
                token=resolved_token,
            )
            api.upload_file(
                path_or_fileobj=str(stats_path),
                path_in_repo="stats.json",
                repo_id=push_repo_id,
                repo_type="dataset",
                token=resolved_token,
            )
            for asset_path in universe_asset_paths:
                api.upload_file(
                    path_or_fileobj=str(asset_path),
                    path_in_repo=asset_path.name,
                    repo_id=push_repo_id,
                    repo_type="dataset",
                    token=resolved_token,
                )
                uploaded_assets.append(asset_path.name)
        except HfHubHTTPError as exc:
            raise RuntimeError(
                "Hugging Face push failed during authentication or repo creation. "
                f"Local export is still available at {output_dir_path}. "
                f"Original error: {exc}"
            ) from exc
        push_info = {
            "repo_id": push_repo_id,
            "private": bool(private),
            "default_config": default_config,
            "configs": pushed_configs,
            "uploaded_assets": uploaded_assets,
        }

    return {
        "stats": stats,
        "push": push_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export library repository graphs and universe artifacts as a Hugging Face-ready parquet dataset."
    )
    parser.add_argument("--export-root", type=str, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--skip-universe",
        action="store_true",
        help="Do not export the aggregated _universe graph even if it exists.",
    )
    parser.add_argument(
        "--include-node-embeddings",
        action="store_true",
        help="Also export the universe node embedding matrix as a separate split.",
    )
    parser.add_argument(
        "--parquet-batch-rows",
        type=int,
        default=DEFAULT_PARQUET_BATCH_ROWS,
        help="Number of rows to buffer before flushing each parquet batch.",
    )
    parser.add_argument("--push-repo-id", type=str, default=None, help="Optional Hugging Face dataset repo id (e.g. org/name).")
    parser.add_argument("--private", action="store_true", help="Push as a private dataset repo.")
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN, then cached huggingface_hub login if available.",
    )
    args = parser.parse_args()

    result = export_library_repo_graph_hf_dataset(
        export_root=args.export_root,
        output_dir=args.output_dir,
        include_universe=not bool(args.skip_universe),
        include_node_embeddings=bool(args.include_node_embeddings),
        parquet_batch_rows=int(args.parquet_batch_rows),
        push_repo_id=args.push_repo_id,
        private=bool(args.private),
        token=args.token,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
