from __future__ import annotations

"""
Export the paper-universe graph as a Hugging Face-ready parquet dataset with
optional Hub push.

Unlike the repository-graph exporter, the paper universe is already stored as
parquet splits. This exporter packages those existing artifacts into a
Hugging Face dataset layout, copies or hard-links the visualization assets, and
writes a dataset card plus stats manifest.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq  # type: ignore
from datasets import Dataset  # type: ignore
from huggingface_hub import HfApi, get_token  # type: ignore
from huggingface_hub.errors import HfHubHTTPError  # type: ignore


DEFAULT_UNIVERSE_DIR = Path("exports/_paper_universe")
DEFAULT_OUTPUT_DIR = Path("exports/huggingface/paper_universe_graph_v1")
_DEFAULT_SPLIT_MAP: Tuple[Tuple[str, str], ...] = (
    ("paper_nodes", "paper_nodes.parquet"),
    ("paper_category_edges", "edges.parquet"),
    ("paper_knn", "paper_knn_edges.parquet"),
    ("category_nodes", "category_nodes.parquet"),
    ("category_knn", "category_knn_edges.parquet"),
    ("topic_nodes", "topic_nodes.parquet"),
    ("paper_topic_edges", "paper_topic_edges.parquet"),
    ("year_nodes", "year_nodes.parquet"),
    ("paper_year_edges", "paper_year_edges.parquet"),
    ("paper_embeddings", "paper_embeddings.parquet"),
    ("paper_fulltext_embeddings", "paper_fulltext_embeddings.parquet"),
)
_ASSET_FILENAMES: Tuple[str, ...] = (
    "manifest.json",
    "progress.json",
    "render_manifest.json",
    "viewer_manifest.json",
    "universe_3d.png",
    "universe_3d_detailed.png",
    "nodes_3d_sample.html",
    "universe_3d_hover.html",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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
    if total_rows < 100_000_000:
        return "10M<n<100M"
    return "100M<n<1B"


def _path_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


def _materialize_file(src: Path, dst: Path, *, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    materialized_mode = mode
    if mode == "auto":
        try:
            os.link(src, dst)
            return "hardlink"
        except Exception:
            materialized_mode = "copy"
    if materialized_mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    shutil.copy2(src, dst)
    return "copy"


def _materialize_directory(src: Path, dst: Path, *, mode: str) -> List[str]:
    if not src.is_dir():
        return []
    if dst.exists():
        shutil.rmtree(dst)
    copied: List[str] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(src)
        _materialize_file(path, dst / relative, mode=mode)
        copied.append(str(relative))
    return copied


def _discover_assets(universe_dir: Path) -> List[Path]:
    assets: List[Path] = []
    for name in _ASSET_FILENAMES:
        path = universe_dir / name
        if path.is_file():
            assets.append(path)
    return assets


def _render_dataset_card(
    *,
    split_counts: Dict[str, int],
    manifest: Dict[str, Any],
    asset_names: Sequence[str],
    interactive_files: Sequence[str],
    path_template: str,
) -> str:
    total_rows = int(sum(split_counts.values()))
    default_config = "paper_nodes" if "paper_nodes" in split_counts else next(iter(split_counts.keys()), "paper_nodes")
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
                        f"    path: \"{path_template.format(name=name)}\"",
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
    asset_blocks: List[str] = []
    if "universe_3d.png" in asset_names:
        asset_blocks.append("### Universe 3D Overview\n\n![Paper universe 3D overview](./universe_3d.png)")
    if "universe_3d_detailed.png" in asset_names:
        asset_blocks.append("### Universe 3D Detailed View\n\n![Paper universe 3D detailed view](./universe_3d_detailed.png)")
    if "nodes_3d_sample.html" in asset_names:
        asset_blocks.append("- [Open the sampled 3D HTML view](./nodes_3d_sample.html)")
    if "universe_3d_hover.html" in asset_names:
        asset_blocks.append("- [Open the interactive hover view](./universe_3d_hover.html)")
    if interactive_files:
        asset_blocks.append("- Interactive viewer payload is included under `interactive/`.")
    asset_section = ""
    if asset_names or interactive_files:
        listed_assets = "\n".join(f"- `{name}`" for name in [*asset_names, *(f"interactive/{name}" for name in interactive_files)])
        rendered_assets = "\n\n".join(block for block in asset_blocks if block).strip()
        if rendered_assets:
            rendered_assets = f"\n\n{rendered_assets}\n"
        asset_section = (
            "\n## Visualization Assets\n\n"
            "The export includes the local paper-universe visualizations and viewer payload when present:\n\n"
            f"{listed_assets}"
            f"{rendered_assets}"
        )

    return f"""---
pretty_name: Paper Universe Graph
viewer: true
tags:
- datasets
- graph
- scientific-papers
- arxiv
- retrieval
- embeddings
size_categories:
- {_size_category(total_rows)}
{yaml_configs}
---

# Paper Universe Graph Dataset

Parquet-first export of the paper-universe graph already built under the Repository Library.

This dataset preserves:

- paper nodes with metadata references and 3D coordinates
- paper/category/year/topic graph layers
- optional paper-to-paper and category-to-category similarity edges
- metadata and full-text paper embedding splits

## Coverage

- papers: `{int(manifest.get("paper_count") or 0)}`
- categories: `{int(manifest.get("category_count") or 0)}`
- years: `{int(manifest.get("year_count") or 0)}`
- topics: `{int(manifest.get("topic_count") or 0)}`
- embedding dimension: `{int(manifest.get("embedding_dim") or 0)}`
- full-text embeddings included: `{str(bool((manifest.get("paper_fulltext_embeddings") or {}).get("enabled"))).lower()}`

## Configs

- default viewer/config: `{default_config}`
{configs}

## Loading

```python
from datasets import load_dataset

{loading}
```

## Notes

- `paper_nodes` stores metadata references and coordinates, not the full paper body.
- The original full text remains in the source paper dataset referenced by the manifest.
- `paper_embeddings` is the metadata/title+abstract embedding split.
- `paper_fulltext_embeddings` is the aggregated full-body embedding split when available.
{asset_section}
"""


def _write_dataset_card(
    *,
    output_dir: Path,
    split_counts: Dict[str, int],
    manifest: Dict[str, Any],
    asset_names: Sequence[str],
    interactive_files: Sequence[str],
    path_template: str,
) -> Path:
    readme = _render_dataset_card(
        split_counts=split_counts,
        manifest=manifest,
        asset_names=asset_names,
        interactive_files=interactive_files,
        path_template=path_template,
    )
    path = output_dir / "README.md"
    path.write_text(readme, encoding="utf-8")
    return path


def export_paper_universe_hf_dataset(
    *,
    universe_dir: str = str(DEFAULT_UNIVERSE_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    materialize_mode: str = "auto",
    push_repo_id: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    universe_dir_path = Path(universe_dir).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    parquet_dir = output_dir_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(universe_dir_path / "manifest.json")
    if not manifest:
        raise RuntimeError(f"No paper universe manifest found under {universe_dir_path}.")

    split_counts: Dict[str, int] = {}
    split_source_paths: Dict[str, str] = {}
    split_output_paths: Dict[str, str] = {}
    split_materialization: Dict[str, str] = {}
    for split_name, filename in _DEFAULT_SPLIT_MAP:
        src = universe_dir_path / filename
        if not src.is_file():
            continue
        dst = parquet_dir / f"{split_name}.parquet"
        split_materialization[split_name] = _materialize_file(src, dst, mode=materialize_mode)
        split_counts[split_name] = _path_row_count(src)
        split_source_paths[split_name] = str(src)
        split_output_paths[split_name] = str(dst)

    if not split_output_paths:
        raise RuntimeError(f"No parquet universe splits were found under {universe_dir_path}.")

    copied_assets: List[str] = []
    for asset_path in _discover_assets(universe_dir_path):
        _materialize_file(asset_path, output_dir_path / asset_path.name, mode=materialize_mode)
        copied_assets.append(asset_path.name)

    interactive_src = universe_dir_path / "interactive"
    interactive_dst = output_dir_path / "interactive"
    interactive_files = _materialize_directory(interactive_src, interactive_dst, mode=materialize_mode)

    readme_path = _write_dataset_card(
        output_dir=output_dir_path,
        split_counts=split_counts,
        manifest=manifest,
        asset_names=copied_assets,
        interactive_files=interactive_files,
        path_template="parquet/{name}.parquet",
    )
    remote_readme_path = output_dir_path / "README.remote.md"
    remote_readme_path.write_text(
        _render_dataset_card(
            split_counts=split_counts,
            manifest=manifest,
            asset_names=copied_assets,
            interactive_files=interactive_files,
            path_template="{name}/*.parquet",
        ),
        encoding="utf-8",
    )

    stats = {
        "universe_dir": str(universe_dir_path),
        "output_dir": str(output_dir_path),
        "paper_count": int(manifest.get("paper_count") or 0),
        "category_count": int(manifest.get("category_count") or 0),
        "year_count": int(manifest.get("year_count") or 0),
        "topic_count": int(manifest.get("topic_count") or 0),
        "embedding_dim": int(manifest.get("embedding_dim") or 0),
        "materialize_mode": materialize_mode,
        "splits": split_counts,
        "source_paths": split_source_paths,
        "parquet_paths": split_output_paths,
        "materialization": split_materialization,
        "assets": copied_assets,
        "interactive_files": interactive_files,
        "readme_path": str(readme_path),
        "remote_readme_path": str(remote_readme_path),
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
        default_config = "paper_nodes" if "paper_nodes" in split_output_paths else next(iter(split_output_paths.keys()))
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
            split_order = [default_config] + [name for name in split_output_paths.keys() if name != default_config]
            for split_name in split_order:
                dataset = _dataset_from_parquet(Path(split_output_paths[split_name]))
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
            for path, path_in_repo in [(remote_readme_path, "README.md"), (stats_path, "stats.json")]:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=push_repo_id,
                    repo_type="dataset",
                    token=resolved_token,
                )
            for asset_name in copied_assets:
                asset_path = output_dir_path / asset_name
                api.upload_file(
                    path_or_fileobj=str(asset_path),
                    path_in_repo=asset_name,
                    repo_id=push_repo_id,
                    repo_type="dataset",
                    token=resolved_token,
                )
                uploaded_assets.append(asset_name)
            if interactive_dst.is_dir():
                api.upload_folder(
                    folder_path=str(interactive_dst),
                    path_in_repo="interactive",
                    repo_id=push_repo_id,
                    repo_type="dataset",
                    token=resolved_token,
                )
                uploaded_assets.extend(f"interactive/{name}" for name in interactive_files)
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
        description="Export the paper universe as a Hugging Face-ready parquet dataset."
    )
    parser.add_argument("--universe-dir", type=str, default=str(DEFAULT_UNIVERSE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--materialize-mode",
        type=str,
        default="auto",
        choices=["auto", "hardlink", "copy"],
        help="How to place parquet and asset files into the output directory.",
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

    result = export_paper_universe_hf_dataset(
        universe_dir=args.universe_dir,
        output_dir=args.output_dir,
        materialize_mode=args.materialize_mode,
        push_repo_id=args.push_repo_id,
        private=bool(args.private),
        token=args.token,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
