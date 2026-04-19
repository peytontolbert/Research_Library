from __future__ import annotations

"""
Export the imported `repo_skills_miner` library extension as a Hugging Face
dataset on local disk, and optionally push it to the Hub.

Default behavior is conservative for public release:
- strips raw code/doc text from the primary skill rows
- strips local cache paths and CAS blob references
- keeps metadata, line spans, and annotation summaries

Use `--include-code-text` only when exporting for private/local use and after
reviewing upstream repo licenses.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict  # type: ignore
from huggingface_hub import HfApi, get_token  # type: ignore
from huggingface_hub.errors import HfHubHTTPError  # type: ignore

from scripts.library_repo_graph_export import DEFAULT_EXPORT_ROOT


_MANIFEST_FILENAME = "_manifest.json"
_EXTENSION_NAME = "repo_skills_miner"


def _load_manifest(export_root: Path) -> Dict[str, Any]:
    path = export_root / _MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_export_relative_path(export_root: Path, raw_path: str) -> Optional[Path]:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return export_root / path


def _jsonl_rows(path: Optional[Path]) -> Iterator[Dict[str, Any]]:
    if path is None or not path.is_file():
        return
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


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _iter_extension_entries(
    manifest: Dict[str, Any],
) -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    repos = manifest.get("repos") or {}
    if not isinstance(repos, dict):
        return
    for repo_id, entry_any in sorted(repos.items()):
        entry = entry_any if isinstance(entry_any, dict) else {}
        extensions = entry.get("extensions") or {}
        if not isinstance(extensions, dict):
            continue
        ext = extensions.get(_EXTENSION_NAME)
        if not isinstance(ext, dict):
            continue
        yield str(repo_id), entry, ext


def _build_repo_row(repo_id: str, summary: Dict[str, Any], ext: Dict[str, Any]) -> Dict[str, Any]:
    counts = summary.get("counts") or ext.get("counts") or {}
    source = summary.get("source") or {}
    skill_kinds = counts.get("skill_kinds") or {}
    top_annotation_summaries = summary.get("top_annotation_summaries") or []
    return {
        "repo_id": repo_id,
        "source": _EXTENSION_NAME,
        "miner_repo_id": str(source.get("miner_repo_id") or ext.get("miner_repo_id") or ""),
        "miner_repo_name": str(source.get("miner_repo_name") or ext.get("miner_repo_name") or ""),
        "miner_revision_id": str(source.get("miner_revision_id") or ext.get("miner_revision_id") or ""),
        "miner_revision": str(source.get("miner_revision") or ext.get("miner_revision") or ""),
        "imported_at": int(summary.get("imported_at") or ext.get("imported_at") or 0),
        "skill_count": int(counts.get("skills") or 0),
        "annotation_count": int(counts.get("annotations") or 0),
        "annotated_skill_count": int(counts.get("annotated_skills") or 0),
        "signal_count": int(counts.get("signals") or 0),
        "skill_kinds_json": _compact_json(skill_kinds),
        "annotation_models": _string_list(summary.get("annotation_models") or ext.get("annotation_models") or []),
        "signal_kinds": _string_list(summary.get("signal_kinds") or ext.get("signal_kinds") or []),
        "top_annotation_summaries_json": _compact_json(top_annotation_summaries),
        "signal_summaries_json": _compact_json(summary.get("signal_summaries") or []),
    }


def _build_skill_row(
    row: Dict[str, Any],
    *,
    include_code_text: bool,
) -> Dict[str, Any]:
    out = {
        "repo_id": str(row.get("repo_id") or ""),
        "source": _EXTENSION_NAME,
        "miner_repo_id": str(row.get("miner_repo_id") or ""),
        "miner_revision_id": str(row.get("miner_revision_id") or ""),
        "skill_id": str(row.get("skill_id") or ""),
        "kind": str(row.get("kind") or ""),
        "module": str(row.get("module") or ""),
        "qualname": str(row.get("qualname") or ""),
        "signature": str(row.get("signature") or ""),
        "file_path": str(row.get("file_path") or ""),
        "line_start": int(row.get("line_start") or 0),
        "line_end": int(row.get("line_end") or 0),
        "has_annotation": bool(row.get("has_annotation")),
        "annotation_id": str(row.get("annotation_id") or ""),
        "annotation_summary": str(row.get("annotation_summary") or ""),
        "annotation_confidence": float(row.get("annotation_confidence") or 0.0),
        "annotation_model_id": str(row.get("annotation_model_id") or ""),
    }
    if include_code_text:
        out["doc_text"] = str(row.get("doc_text") or "")
        out["snippet"] = str(row.get("snippet") or "")
    return out


def _build_annotation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "repo_id": str(row.get("repo_id") or ""),
        "source": _EXTENSION_NAME,
        "miner_repo_id": str(row.get("miner_repo_id") or ""),
        "miner_revision_id": str(row.get("miner_revision_id") or ""),
        "skill_id": str(row.get("skill_id") or ""),
        "annotation_id": str(row.get("annotation_id") or ""),
        "model_id": str(row.get("model_id") or ""),
        "summary": str(row.get("summary") or ""),
        "confidence": float(row.get("confidence") or 0.0),
        "created_ms": int(row.get("created_ms") or 0),
        "kind": str(row.get("kind") or ""),
        "module": str(row.get("module") or ""),
        "qualname": str(row.get("qualname") or ""),
        "signature": str(row.get("signature") or ""),
        "file_path": str(row.get("file_path") or ""),
        "line_start": int(row.get("line_start") or 0),
        "line_end": int(row.get("line_end") or 0),
        "annotation_json": _compact_json(row.get("annotation") or {}),
    }


def _build_signal_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "repo_id": str(row.get("repo_id") or ""),
        "source": _EXTENSION_NAME,
        "miner_repo_id": str(row.get("miner_repo_id") or ""),
        "miner_revision_id": str(row.get("miner_revision_id") or ""),
        "signals_id": str(row.get("signals_id") or ""),
        "revision_id": str(row.get("revision_id") or ""),
        "kind": str(row.get("kind") or ""),
        "summary": str(row.get("summary") or ""),
        "created_ms": int(row.get("created_ms") or 0),
        "signals_json": _compact_json(row.get("signals") or {}),
    }


def _dataset_from_rows(rows: List[Dict[str, Any]]) -> Dataset:
    broken_torch = sys.modules.get("torch")
    removed_torch = False
    if broken_torch is not None and not hasattr(broken_torch, "Tensor"):
        removed_torch = True
        sys.modules.pop("torch", None)
    try:
        if rows:
            return Dataset.from_list(rows)
        return Dataset.from_dict({})
    finally:
        if removed_torch and broken_torch is not None:
            sys.modules["torch"] = broken_torch


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    explicit = str(token or "").strip()
    if explicit:
        return explicit
    cached = get_token()
    if isinstance(cached, str) and cached.strip():
        return cached.strip()
    return None


def _push_multiconfig_dataset(
    *,
    api: HfApi,
    repo_id: str,
    token: str,
    private: bool,
    primary_config_name: str,
    dataset_dict: DatasetDict,
) -> List[Dict[str, Any]]:
    """
    Push heterogeneous tables as separate dataset configs.

    `datasets` requires equal features across splits inside a single
    `DatasetDict` push on older versions. Our export intentionally emits
    different schemas for skills, repos, annotations, and signals, so the
    Hub representation is:

    - config `<primary_config_name>` for the main skills table
    - config `repos`
    - config `annotations`
    - config `signals`

    Each config uses a single `train` split.
    """
    pushed: List[Dict[str, Any]] = []

    if "train" in dataset_dict:
        dataset_dict["train"].push_to_hub(
            repo_id=repo_id,
            config_name=primary_config_name,
            set_default=True,
            split="train",
            private=private,
            token=token,
        )
        pushed.append(
            {
                "config_name": primary_config_name,
                "split": "train",
                "rows": int(dataset_dict["train"].num_rows),
                "source_split": "train",
            }
        )

    for split_name in ("repos", "annotations", "signals"):
        if split_name not in dataset_dict:
            continue
        dataset = dataset_dict[split_name]
        dataset.push_to_hub(
            repo_id=repo_id,
            config_name=split_name,
            split="train",
            private=private,
            token=token,
        )
        pushed.append(
            {
                "config_name": split_name,
                "split": "train",
                "rows": int(dataset.num_rows),
                "source_split": split_name,
            }
        )
    return pushed


def export_repo_skills_hf_dataset(
    *,
    export_root: str = DEFAULT_EXPORT_ROOT,
    output_dir: str,
    include_code_text: bool = False,
    include_annotations_split: bool = True,
    include_signals_split: bool = True,
    repo_ids: Optional[Sequence[str]] = None,
    push_repo_id: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    config_name: str = "default",
) -> Dict[str, Any]:
    export_root_path = Path(export_root).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(export_root_path)
    requested_repo_ids = {str(repo_id) for repo_id in (repo_ids or []) if str(repo_id).strip()}

    repo_rows: List[Dict[str, Any]] = []
    skill_rows: List[Dict[str, Any]] = []
    annotation_rows: List[Dict[str, Any]] = []
    signal_rows: List[Dict[str, Any]] = []

    for repo_id, _entry, ext in _iter_extension_entries(manifest):
        if requested_repo_ids and repo_id not in requested_repo_ids:
            continue
        paths = ext.get("paths") or {}
        if not isinstance(paths, dict):
            continue
        summary_path = _resolve_export_relative_path(export_root_path, str(paths.get("summary") or ""))
        skills_path = _resolve_export_relative_path(export_root_path, str(paths.get("skills") or ""))
        annotations_path = _resolve_export_relative_path(export_root_path, str(paths.get("annotations") or ""))
        signals_path = _resolve_export_relative_path(export_root_path, str(paths.get("signals") or ""))

        if summary_path is None or not summary_path.is_file() or skills_path is None or not skills_path.is_file():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        repo_rows.append(_build_repo_row(repo_id, summary, ext))

        for row in _jsonl_rows(skills_path):
            skill_rows.append(_build_skill_row(row, include_code_text=include_code_text))
        if include_annotations_split:
            for row in _jsonl_rows(annotations_path):
                annotation_rows.append(_build_annotation_row(row))
        if include_signals_split:
            for row in _jsonl_rows(signals_path):
                signal_rows.append(_build_signal_row(row))

    splits: Dict[str, Dataset] = {
        "train": _dataset_from_rows(skill_rows),
        "repos": _dataset_from_rows(repo_rows),
    }
    if include_annotations_split:
        splits["annotations"] = _dataset_from_rows(annotation_rows)
    if include_signals_split:
        splits["signals"] = _dataset_from_rows(signal_rows)
    dataset_dict = DatasetDict(splits)

    parquet_dir = output_dir_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths: Dict[str, str] = {}
    for split_name, dataset in dataset_dict.items():
        path = parquet_dir / f"{split_name}.parquet"
        dataset.to_parquet(str(path))
        parquet_paths[split_name] = str(path)

    dataset_disk_dir = output_dir_path / "dataset_dict"
    dataset_dict.save_to_disk(str(dataset_disk_dir))

    stats = {
        "source_extension": _EXTENSION_NAME,
        "export_root": str(export_root_path),
        "output_dir": str(output_dir_path),
        "public_safe": not include_code_text,
        "splits": {name: int(dataset.num_rows) for name, dataset in dataset_dict.items()},
        "repo_count": len(repo_rows),
        "skill_count": len(skill_rows),
        "annotation_count": len(annotation_rows),
        "signal_count": len(signal_rows),
        "parquet_paths": parquet_paths,
    }
    (output_dir_path / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    readme = f"""---
pretty_name: Repository Skills Miner (72k public-safe)
viewer: true
tags:
- datasets
- text
- code
- software-engineering
size_categories:
- 10K<n<100K
---

# Repository Skills Miner Dataset

Structured repository skill metadata exported from `/data/repo_skills_miner` and normalized through the Repository Library.

## Splits

- `train`: `{len(skill_rows)}` skill rows
- `repos`: `{len(repo_rows)}` repo summary rows
- `annotations`: `{len(annotation_rows)}` annotation rows
- `signals`: `{len(signal_rows)}` revision signal rows

## Hugging Face configs

This dataset is pushed as multiple Hub configs:

- `default`: the main skill table
- `repos`: repo-level summary rows
- `annotations`: structured LLM annotation rows
- `signals`: revision-level signal rows

Typical loading patterns:

```python
from datasets import load_dataset

skills = load_dataset("YOUR_NAMESPACE/YOUR_DATASET", "default")
repos = load_dataset("YOUR_NAMESPACE/YOUR_DATASET", "repos")
annotations = load_dataset("YOUR_NAMESPACE/YOUR_DATASET", "annotations")
signals = load_dataset("YOUR_NAMESPACE/YOUR_DATASET", "signals")
```

## Safety Mode

- `public_safe`: `{str(not include_code_text).lower()}`
- Raw `snippet` and `doc_text` fields are {"included" if include_code_text else "excluded"}.

## Primary Columns

### train

- `repo_id`
- `miner_revision_id`
- `skill_id`
- `kind`
- `module`
- `qualname`
- `signature`
- `file_path`
- `line_start`
- `line_end`
- `annotation_summary`

### repos

- `repo_id`
- `miner_repo_name`
- `skill_count`
- `annotation_count`
- `signal_count`
- `annotation_models`

## Notes

- Default export strips raw code/doc text to reduce licensing risk for public release.
- If you need the exact code/doc strings for private workflows, rerun with `--include-code-text`.
"""
    (output_dir_path / "README.md").write_text(readme, encoding="utf-8")

    push_info: Optional[Dict[str, Any]] = None
    if push_repo_id:
        resolved_token = _resolve_hf_token(token)
        if not resolved_token:
            raise RuntimeError(
                "Hugging Face push requested but no token was found. "
                "Set HF_TOKEN, pass --token hf_..., or login first via huggingface_hub."
            )
        api = HfApi(token=resolved_token)
        pushed_configs: List[Dict[str, Any]] = []
        try:
            api.whoami(token=resolved_token)
            api.create_repo(
                repo_id=push_repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            pushed_configs = _push_multiconfig_dataset(
                api=api,
                repo_id=push_repo_id,
                token=resolved_token,
                private=private,
                primary_config_name=config_name,
                dataset_dict=dataset_dict,
            )
            api.upload_file(
                path_or_fileobj=str(output_dir_path / "README.md"),
                path_in_repo="README.md",
                repo_id=push_repo_id,
                repo_type="dataset",
                token=resolved_token,
            )
            api.upload_file(
                path_or_fileobj=str(output_dir_path / "stats.json"),
                path_in_repo="stats.json",
                repo_id=push_repo_id,
                repo_type="dataset",
                token=resolved_token,
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(
                "Hugging Face push failed during authentication or repo creation. "
                f"Local export is still available at {output_dir_path}. "
                "Verify your token has permission to create dataset repos and retry "
                f"for {push_repo_id!r}. Original error: {exc}"
            ) from exc
        push_info = {
            "repo_id": push_repo_id,
            "private": bool(private),
            "config_name": config_name,
            "configs": pushed_configs,
        }

    return {
        "stats": stats,
        "dataset_disk_dir": str(dataset_disk_dir),
        "parquet_dir": str(parquet_dir),
        "push": push_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export repo_skills_miner structured exports as a Hugging Face dataset."
    )
    parser.add_argument("--export-root", type=str, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/huggingface/repo_skills_miner_public",
        help="Local directory for parquet files, dataset_dict, and README.",
    )
    parser.add_argument(
        "--include-code-text",
        action="store_true",
        help="Include raw doc_text/snippet columns. Avoid for public release unless licenses are reviewed.",
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        help="Restrict export to one or more library repo_ids.",
    )
    parser.add_argument(
        "--no-annotations-split",
        action="store_true",
        help="Omit the separate annotations split.",
    )
    parser.add_argument(
        "--no-signals-split",
        action="store_true",
        help="Omit the separate signals split.",
    )
    parser.add_argument("--push-repo-id", type=str, default=None, help="Optional Hugging Face dataset repo id (e.g. org/name).")
    parser.add_argument("--private", action="store_true", help="Push as a private dataset repo.")
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN, then cached huggingface_hub login if available.",
    )
    parser.add_argument("--config-name", type=str, default="default", help="Hub config name when pushing.")
    args = parser.parse_args()

    result = export_repo_skills_hf_dataset(
        export_root=args.export_root,
        output_dir=args.output_dir,
        include_code_text=bool(args.include_code_text),
        include_annotations_split=not bool(args.no_annotations_split),
        include_signals_split=not bool(args.no_signals_split),
        repo_ids=args.repo_ids,
        push_repo_id=args.push_repo_id,
        private=bool(args.private),
        token=args.token,
        config_name=args.config_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
