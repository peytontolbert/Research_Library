from __future__ import annotations

"""
Import structured repo metadata from `/data/repo_skills_miner` into the
Repository Library exports tree.

This importer treats repo_skills_miner as a per-repository extension:

- Writes structured files under:
    exports/<repo_id>/structured/
      - repo_skills_miner.summary.json
      - repo_skills_miner.skills.jsonl
      - repo_skills_miner.annotations.jsonl
      - repo_skills_miner.signals.jsonl
- Registers those files under:
    manifest["repos"][repo_id]["extensions"]["repo_skills_miner"]

The library manifest already flows through `Repository.metadata` and the
runtime `/api/repos/{repo_id}` endpoint, so once this importer runs the
extension becomes part of the normal library surface area.
"""

import argparse
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from scripts.library_repo_graph_export import DEFAULT_EXPORT_ROOT


DEFAULT_MINER_DB = "/data/repo_skills_miner/skill_engine.db"
DEFAULT_MINER_STORE = "/data/repo_skills_miner/skill_engine_store"
_MANIFEST_FILENAME = "_manifest.json"
_EXTENSION_NAME = "repo_skills_miner"


def _manifest_path(export_root: Path) -> Path:
    return export_root / _MANIFEST_FILENAME


def _load_manifest(export_root: Path) -> Dict[str, Any]:
    path = _manifest_path(export_root)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_manifest(export_root: Path, manifest: Dict[str, Any]) -> Path:
    export_root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(export_root)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True))
            fh.write("\n")
            count += 1
    return count


def _relpath(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _resolve_export_path(export_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return export_root / path


def _jsonl_rows(path: Path) -> Iterator[Dict[str, Any]]:
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


def _blob_path(store_root: Path, sha256: str) -> Path:
    return store_root / "blobs" / sha256[:2] / sha256


def _load_blob_json(store_root: Path, sha256: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not sha256:
        return None, None
    path = _blob_path(store_root, sha256)
    if not path.is_file():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
    except Exception:
        return None, text
    return data if isinstance(data, dict) else None, text


def _build_manifest_repo_index(manifest: Dict[str, Any]) -> Dict[str, str]:
    repos = manifest.get("repos") or {}
    if not isinstance(repos, dict):
        return {}
    index: Dict[str, str] = {}
    for repo_id, entry_any in repos.items():
        entry = entry_any if isinstance(entry_any, dict) else {}
        repo_root = entry.get("repo_root")
        if isinstance(repo_root, str) and repo_root:
            index[os.path.abspath(repo_root)] = str(repo_id)
    return index


def _select_latest_revisions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            repos.repo_id AS miner_repo_id,
            repos.path AS repo_path,
            repos.name AS repo_name,
            revisions.revision_id,
            revisions.revision,
            revisions.content_sha256,
            revisions.created_ms
        FROM repos
        JOIN revisions ON revisions.repo_id = repos.repo_id
        ORDER BY repos.repo_id ASC, revisions.created_ms DESC
        """
    ).fetchall()

    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        record = dict(row)
        miner_repo_id = str(record.get("miner_repo_id") or "")
        if not miner_repo_id or miner_repo_id in latest:
            continue
        latest[miner_repo_id] = record
    return list(latest.values())


def _load_bundle_skills(bundle_path: Path) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    repo_header: Optional[Dict[str, Any]] = None
    skills: List[Dict[str, Any]] = []
    for row in _jsonl_rows(bundle_path):
        row_type = str(row.get("type") or "")
        if row_type == "repo" and repo_header is None:
            repo_header = row
            continue
        if row_type == "skill":
            skills.append(row)
    skills.sort(key=lambda row: (str(row.get("file_path") or ""), int(row.get("line_start") or 0)))
    return repo_header, skills


def _load_annotations_for_revision(
    conn: sqlite3.Connection,
    revision_id: str,
    store_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            a.annotation_id,
            a.skill_id,
            a.model_id,
            a.cache_dir,
            a.offline,
            a.prompt_blob_sha256,
            a.response_blob_sha256,
            a.annotation_blob_sha256,
            a.summary,
            a.confidence,
            a.created_ms,
            s.kind,
            s.module,
            s.qualname,
            s.signature,
            s.file_path,
            s.line_start,
            s.line_end
        FROM annotations AS a
        JOIN skills AS s ON s.skill_id = a.skill_id
        WHERE s.revision_id = ?
        ORDER BY a.confidence DESC, a.created_ms DESC, a.skill_id ASC
        """,
        (revision_id,),
    ).fetchall()

    annotations: List[Dict[str, Any]] = []
    by_skill: Dict[str, Dict[str, Any]] = {}
    model_ids: List[str] = []

    for row in rows:
        record = dict(row)
        skill_id = str(record.get("skill_id") or "")
        blob_json, blob_text = _load_blob_json(
            store_root, str(record.get("annotation_blob_sha256") or "")
        )
        normalized = {
            "skill_id": skill_id,
            "annotation_id": str(record.get("annotation_id") or ""),
            "model_id": str(record.get("model_id") or ""),
            "cache_dir": str(record.get("cache_dir") or ""),
            "offline": bool(record.get("offline")),
            "summary": str(record.get("summary") or ""),
            "confidence": float(record.get("confidence") or 0.0),
            "created_ms": int(record.get("created_ms") or 0),
            "kind": str(record.get("kind") or ""),
            "module": str(record.get("module") or ""),
            "qualname": str(record.get("qualname") or ""),
            "signature": str(record.get("signature") or ""),
            "file_path": str(record.get("file_path") or ""),
            "line_start": int(record.get("line_start") or 0),
            "line_end": int(record.get("line_end") or 0),
            "blob_refs": {
                "prompt": str(record.get("prompt_blob_sha256") or ""),
                "response": str(record.get("response_blob_sha256") or ""),
                "annotation": str(record.get("annotation_blob_sha256") or ""),
            },
        }
        if blob_json is not None:
            normalized["annotation"] = blob_json
        elif blob_text:
            normalized["annotation_text"] = blob_text
        annotations.append(normalized)
        by_skill[skill_id] = normalized
        model_id = normalized["model_id"]
        if model_id:
            model_ids.append(model_id)

    return annotations, by_skill, sorted(set(model_ids))


def _load_signals_for_revision(
    conn: sqlite3.Connection,
    revision_id: str,
    store_root: Path,
) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            signals_id,
            revision_id,
            kind,
            signals_blob_sha256,
            summary,
            created_ms
        FROM revision_signals
        WHERE revision_id = ?
        ORDER BY created_ms DESC, kind ASC
        """,
        (revision_id,),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        blob_json, blob_text = _load_blob_json(
            store_root, str(record.get("signals_blob_sha256") or "")
        )
        normalized = {
            "signals_id": str(record.get("signals_id") or ""),
            "revision_id": str(record.get("revision_id") or ""),
            "kind": str(record.get("kind") or ""),
            "summary": str(record.get("summary") or ""),
            "created_ms": int(record.get("created_ms") or 0),
            "signals_blob_sha256": str(record.get("signals_blob_sha256") or ""),
        }
        if blob_json is not None:
            normalized["signals"] = blob_json
        elif blob_text:
            normalized["signals_text"] = blob_text
        out.append(normalized)
    return out


def _top_annotation_summaries(
    annotations: Sequence[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    ranked = sorted(
        (
            row
            for row in annotations
            if str(row.get("summary") or "").strip()
        ),
        key=lambda row: (
            -float(row.get("confidence") or 0.0),
            str(row.get("file_path") or ""),
            str(row.get("qualname") or ""),
        ),
    )
    out: List[Dict[str, Any]] = []
    for row in ranked[:limit]:
        out.append(
            {
                "skill_id": str(row.get("skill_id") or ""),
                "qualname": str(row.get("qualname") or ""),
                "file_path": str(row.get("file_path") or ""),
                "summary": str(row.get("summary") or ""),
                "confidence": float(row.get("confidence") or 0.0),
            }
        )
    return out


def _normalize_skill_rows(
    bundle_skills: Sequence[Dict[str, Any]],
    *,
    library_repo_id: str,
    miner_repo_id: str,
    revision_id: str,
    annotations_by_skill: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in bundle_skills:
        skill_id = str(row.get("skill_id") or "")
        ann = annotations_by_skill.get(skill_id)
        normalized = {
            "repo_id": library_repo_id,
            "source": _EXTENSION_NAME,
            "miner_repo_id": miner_repo_id,
            "miner_revision_id": revision_id,
            "skill_id": skill_id,
            "kind": str(row.get("kind") or ""),
            "module": str(row.get("module") or ""),
            "qualname": str(row.get("qualname") or ""),
            "signature": str(row.get("signature") or ""),
            "file_path": str(row.get("file_path") or ""),
            "line_start": int(row.get("line_start") or 0),
            "line_end": int(row.get("line_end") or 0),
            "doc_text": str(row.get("doc_text") or ""),
            "snippet": str(row.get("snippet") or ""),
            "has_annotation": ann is not None,
        }
        if ann is not None:
            normalized["annotation_id"] = str(ann.get("annotation_id") or "")
            normalized["annotation_summary"] = str(ann.get("summary") or "")
            normalized["annotation_confidence"] = float(ann.get("confidence") or 0.0)
            normalized["annotation_model_id"] = str(ann.get("model_id") or "")
        out.append(normalized)
    return out


def _normalize_annotation_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    library_repo_id: str,
    miner_repo_id: str,
    revision_id: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        normalized = {
            "repo_id": library_repo_id,
            "source": _EXTENSION_NAME,
            "miner_repo_id": miner_repo_id,
            "miner_revision_id": revision_id,
            **row,
        }
        out.append(normalized)
    return out


def _normalize_signal_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    library_repo_id: str,
    miner_repo_id: str,
    revision_id: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        normalized = {
            "repo_id": library_repo_id,
            "source": _EXTENSION_NAME,
            "miner_repo_id": miner_repo_id,
            "miner_revision_id": revision_id,
            **row,
        }
        out.append(normalized)
    return out


def _is_up_to_date(
    repo_entry: Dict[str, Any],
    *,
    export_root: Path,
    revision_id: str,
) -> bool:
    extensions = repo_entry.get("extensions") or {}
    if not isinstance(extensions, dict):
        return False
    ext = extensions.get(_EXTENSION_NAME) or {}
    if not isinstance(ext, dict):
        return False
    if str(ext.get("miner_revision_id") or "") != revision_id:
        return False
    paths = ext.get("paths") or {}
    if not isinstance(paths, dict):
        return False
    for key in ("summary", "skills", "annotations", "signals"):
        raw = str(paths.get(key) or "")
        if not raw:
            continue
        if not _resolve_export_path(export_root, raw).is_file():
            return False
    return True


def import_repo_skills_miner(
    *,
    miner_db: str = DEFAULT_MINER_DB,
    miner_store: str = DEFAULT_MINER_STORE,
    export_root: str = DEFAULT_EXPORT_ROOT,
    repo_ids: Optional[Sequence[str]] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    export_root_path = Path(export_root).resolve()
    miner_db_path = Path(miner_db).resolve()
    miner_store_path = Path(miner_store).resolve()

    manifest = _load_manifest(export_root_path)
    if not isinstance(manifest.get("repos"), dict):
        manifest["repos"] = {}
    repos_meta: Dict[str, Any] = manifest["repos"]

    requested_repo_ids = {str(repo_id) for repo_id in (repo_ids or []) if str(repo_id).strip()}
    repo_root_to_library_id = _build_manifest_repo_index(manifest)

    conn = sqlite3.connect(str(miner_db_path))
    latest_revisions = _select_latest_revisions(conn)

    results: List[Dict[str, Any]] = []
    skipped_unmapped = 0
    skipped_filtered = 0
    skipped_cached = 0

    for revision_meta in latest_revisions:
        repo_path = os.path.abspath(str(revision_meta.get("repo_path") or ""))
        library_repo_id = repo_root_to_library_id.get(repo_path)
        if not library_repo_id:
            skipped_unmapped += 1
            continue
        if requested_repo_ids and library_repo_id not in requested_repo_ids:
            skipped_filtered += 1
            continue

        repo_entry_any = repos_meta.get(library_repo_id)
        repo_entry = repo_entry_any if isinstance(repo_entry_any, dict) else {}
        revision_id = str(revision_meta.get("revision_id") or "")
        if (not force) and _is_up_to_date(repo_entry, export_root=export_root_path, revision_id=revision_id):
            ext = ((repo_entry.get("extensions") or {}).get(_EXTENSION_NAME) or {})
            results.append(
                {
                    "repo_id": library_repo_id,
                    "status": "skipped",
                    "reason": "up_to_date",
                    "miner_revision_id": revision_id,
                    "counts": dict(ext.get("counts") or {}),
                }
            )
            skipped_cached += 1
            continue

        bundle_path = miner_store_path / "bundles" / f"{revision_id}.jsonl"
        if not bundle_path.is_file():
            results.append(
                {
                    "repo_id": library_repo_id,
                    "status": "skipped",
                    "reason": "missing_bundle",
                    "miner_revision_id": revision_id,
                    "bundle_path": str(bundle_path),
                }
            )
            continue

        bundle_repo, bundle_skills = _load_bundle_skills(bundle_path)
        annotations, annotations_by_skill, annotation_models = _load_annotations_for_revision(
            conn, revision_id, miner_store_path
        )
        signals = _load_signals_for_revision(conn, revision_id, miner_store_path)

        skills_rows = _normalize_skill_rows(
            bundle_skills,
            library_repo_id=library_repo_id,
            miner_repo_id=str(revision_meta.get("miner_repo_id") or ""),
            revision_id=revision_id,
            annotations_by_skill=annotations_by_skill,
        )
        annotation_rows = _normalize_annotation_rows(
            annotations,
            library_repo_id=library_repo_id,
            miner_repo_id=str(revision_meta.get("miner_repo_id") or ""),
            revision_id=revision_id,
        )
        signal_rows = _normalize_signal_rows(
            signals,
            library_repo_id=library_repo_id,
            miner_repo_id=str(revision_meta.get("miner_repo_id") or ""),
            revision_id=revision_id,
        )

        out_dir = export_root_path / library_repo_id / "structured"
        summary_path = out_dir / "repo_skills_miner.summary.json"
        skills_path = out_dir / "repo_skills_miner.skills.jsonl"
        annotations_path = out_dir / "repo_skills_miner.annotations.jsonl"
        signals_path = out_dir / "repo_skills_miner.signals.jsonl"

        kind_counts = Counter(str(row.get("kind") or "") for row in skills_rows if str(row.get("kind") or ""))
        signal_kinds = sorted(
            {str(row.get("kind") or "") for row in signal_rows if str(row.get("kind") or "")}
        )

        summary = {
            "extension": _EXTENSION_NAME,
            "repo_id": library_repo_id,
            "imported_at": int(time.time()),
            "source": {
                "miner_db": str(miner_db_path),
                "miner_store": str(miner_store_path),
                "miner_repo_id": str(revision_meta.get("miner_repo_id") or ""),
                "miner_repo_name": str(revision_meta.get("repo_name") or ""),
                "miner_repo_path": repo_path,
                "miner_revision_id": revision_id,
                "miner_revision": str(revision_meta.get("revision") or ""),
                "content_sha256": str(revision_meta.get("content_sha256") or ""),
                "bundle_path": str(bundle_path),
            },
            "counts": {
                "skills": len(skills_rows),
                "annotations": len(annotation_rows),
                "annotated_skills": sum(1 for row in skills_rows if row.get("has_annotation")),
                "signals": len(signal_rows),
                "skill_kinds": dict(kind_counts),
            },
            "annotation_models": annotation_models,
            "signal_kinds": signal_kinds,
            "signal_summaries": [
                {
                    "kind": str(row.get("kind") or ""),
                    "summary": str(row.get("summary") or ""),
                }
                for row in signal_rows
            ],
            "top_annotation_summaries": _top_annotation_summaries(annotation_rows),
            "bundle_repo_header": bundle_repo or {},
        }

        _write_json(summary_path, summary)
        _write_jsonl(skills_path, skills_rows)
        _write_jsonl(annotations_path, annotation_rows)
        _write_jsonl(signals_path, signal_rows)

        extensions = repo_entry.get("extensions") or {}
        if not isinstance(extensions, dict):
            extensions = {}
        extensions[_EXTENSION_NAME] = {
            "source": _EXTENSION_NAME,
            "imported_at": summary["imported_at"],
            "miner_repo_id": str(revision_meta.get("miner_repo_id") or ""),
            "miner_repo_name": str(revision_meta.get("repo_name") or ""),
            "miner_revision_id": revision_id,
            "miner_revision": str(revision_meta.get("revision") or ""),
            "counts": summary["counts"],
            "annotation_models": annotation_models,
            "signal_kinds": signal_kinds,
            "paths": {
                "summary": _relpath(summary_path, export_root_path),
                "skills": _relpath(skills_path, export_root_path),
                "annotations": _relpath(annotations_path, export_root_path),
                "signals": _relpath(signals_path, export_root_path),
            },
        }
        repo_entry["extensions"] = extensions
        repos_meta[library_repo_id] = repo_entry

        results.append(
            {
                "repo_id": library_repo_id,
                "status": "imported",
                "miner_revision_id": revision_id,
                "counts": summary["counts"],
            }
        )

    _save_manifest(export_root_path, manifest)
    results.append(
        {
            "status": "summary",
            "imported_repos": sum(1 for row in results if row.get("status") == "imported"),
            "skipped_cached": skipped_cached,
            "skipped_unmapped": skipped_unmapped,
            "skipped_filtered": skipped_filtered,
        }
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import repo_skills_miner metadata as a structured library extension."
    )
    parser.add_argument("--miner-db", type=str, default=DEFAULT_MINER_DB)
    parser.add_argument("--miner-store", type=str, default=DEFAULT_MINER_STORE)
    parser.add_argument("--export-root", type=str, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        help="Restrict import to one or more library repo_ids.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite extension exports even when the manifest shows the same miner revision.",
    )
    args = parser.parse_args()

    results = import_repo_skills_miner(
        miner_db=args.miner_db,
        miner_store=args.miner_store,
        export_root=args.export_root,
        repo_ids=args.repo_ids,
        force=bool(args.force),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
