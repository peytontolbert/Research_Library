from __future__ import annotations

"""
FastAPI server for the Repository Library.

This exposes:
- JSON APIs for:
  - Listing repositories and their basic metadata.
  - Inspecting a single repository entry.
  - Planning queries (QA / comparative QA) via `RepoLibrary.query`.
  - Planning meta-skill / agentic tasks via `RepoLibrary.run_task`.
- A lightweight HTML UI for interactively browsing the library and
  sending queries to the JSON endpoints.

Usage (from the project root):

    uvicorn run:app --reload --host 0.0.0.0 --port 8000

Dependencies:
    pip install fastapi uvicorn transformers torch
"""

import json
import logging
import hashlib
import math
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:  # LLM runtime (meta-llama/Llama-3.1-8B-Instruct) dependencies
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    try:
        # Optional quantization support (4-bit via bitsandbytes).
        from transformers import BitsAndBytesConfig  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        BitsAndBytesConfig = None  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore

from scripts.adapter_bank import FileAdapterBank  # type: ignore
from scripts.library_repo_graph_export import DEFAULT_EXPORT_ROOT  # type: ignore
from scripts.repo_graph import parse_program_uri  # type: ignore
from scripts.repo_library import (  # type: ignore
    QueryMode,
    RepoLibrary,
    TaskMode,
    compute_repo_context_key,
    load_manifest,
    open_repository,
)
from modules.vector_index import load_simple_repo_index  # type: ignore
from modules.arxiv_library import iter_metadata as arxiv_iter_metadata, search_keyword as arxiv_search_keyword  # type: ignore
from modules.algorithms_library import (  # type: ignore
    iter_algorithms,
    iter_problems,
    iter_implementations,
    search_algorithms as algo_search_algorithms,
)
from modules.qa_runtime import (  # type: ignore
    QAModelConfig,
    get_default_qa_base_config,
    get_model_config_from_adapter,
    get_or_load_model,
    run_qa_generation,
)
from modules.qa_swarm import (  # type: ignore
    QASwarmController,
    RetrieverAgent,
    SemanticRouter,
    SkillAdapterManager,
)
from scripts.skill_build import (  # type: ignore
    all_skill_statuses_for_repo,
    build_skill,
)


app = FastAPI(title="Research Library", version="0.1.0")

logger = logging.getLogger("repository_library.server")


_LLM_MODEL = None
_LLM_TOKENIZER = None
_QA_INDEX_CACHE: Dict[str, Any] = {}
_COARSE_LANE_RETRIEVER = None


REPO_ROOT = Path(__file__).resolve().parent
ARXIV_PDF_ROOT = Path("/arxiv/pdfs")
ARXIV_PDF_CACHE_ROOT = REPO_ROOT / "exports" / "arxiv_pdfs"
PAPER_UNIVERSE_ROOT = REPO_ROOT / "exports" / "_paper_universe"
PAPER_UNIVERSE_URL_ROOT = "/paper-universe"
PAPER_TEXT_DATASET_ROOTS = (
    Path("/arxiv/huggingface/paper_text_1m_dedup_v1"),
    REPO_ROOT / "exports" / "huggingface" / "paper_text_60k_full_v1",
)
PAPER_TEXT_LOOKUP_COLUMNS = (
    "paper_id",
    "canonical_paper_id",
    "paper_version",
    "pdf_path",
    "title",
    "abstract",
    "authors",
    "categories",
    "license",
    "update_date",
    "version_count",
    "metadata_found",
    "text",
    "text_source",
    "text_is_partial",
    "text_char_count",
    "text_line_count",
    "token_count",
    "page_count",
    "token_types",
    "token_type_counts_json",
)

app.mount(
    PAPER_UNIVERSE_URL_ROOT,
    StaticFiles(directory=str(PAPER_UNIVERSE_ROOT), check_dir=False),
    name="paper-universe",
)


def _all_local_arxiv_pdf_roots(preferred_root: Optional[Path] = None) -> List[Path]:
    roots: List[Path] = []
    if preferred_root is not None:
        roots.append(preferred_root)
    roots.extend([ARXIV_PDF_ROOT, ARXIV_PDF_CACHE_ROOT])
    deduped: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _can_write_arxiv_pdf_root(root: Path) -> bool:
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".codex_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def _choose_arxiv_pdf_download_root(preferred_root: Optional[Path] = None) -> Path:
    for root in _all_local_arxiv_pdf_roots(preferred_root):
        if _can_write_arxiv_pdf_root(root):
            return root
    return preferred_root or ARXIV_PDF_CACHE_ROOT


def _local_arxiv_pdf_candidates(pdf_id: str) -> List[Path]:
    """
    Candidate local PDF paths for a normalized arXiv id.

    We support both legacy root-level storage:
      /arxiv/pdfs/<id>.pdf
    and bucketed storage:
      /arxiv/pdfs/YYMM/<id>.pdf
    """
    norm_id = str(pdf_id or "").strip().split("/")[-1]
    if not norm_id:
        return []

    candidates: List[Path] = []
    yymm = norm_id[:4]
    for root in _all_local_arxiv_pdf_roots():
        if len(yymm) == 4 and yymm.isdigit():
            candidates.append(root / yymm / f"{norm_id}.pdf")
        candidates.append(root / f"{norm_id}.pdf")
    return candidates


def _find_local_arxiv_pdf(pdf_id: str) -> Optional[Path]:
    for candidate in _local_arxiv_pdf_candidates(pdf_id):
        if candidate.is_file():
            return candidate
    return None


def _arxiv_pdf_url(paper_id: str) -> str:
    pdf_id = str(paper_id or "").strip().split("/")[-1]
    if not pdf_id:
        return ""
    return f"https://arxiv.org/pdf/{pdf_id}.pdf"


def _normalize_arxiv_paper_id(paper_id: str) -> str:
    paper_text = str(paper_id or "").strip().split("/")[-1]
    if not paper_text:
        return ""
    lower = paper_text.lower()
    if "v" in lower:
        head, tail = lower.rsplit("v", 1)
        if tail.isdigit() and head:
            return paper_text[: len(head)]
    return paper_text


def _enrich_arxiv_record(rec: Dict[str, object]) -> Dict[str, object]:
    rec_id = str(rec.get("id") or "").strip()
    pdf_id = _normalize_arxiv_paper_id(rec_id)
    enriched = dict(rec)
    enriched["has_pdf"] = bool(pdf_id and _find_local_arxiv_pdf(pdf_id) is not None)
    return enriched


def _find_local_arxiv_record_by_id(paper_id: str) -> Optional[Dict[str, object]]:
    normalized = _normalize_arxiv_paper_id(paper_id)
    if not normalized:
        return None
    for rec in arxiv_iter_metadata():
        rec_id = _normalize_arxiv_paper_id(str(rec.id or ""))
        if rec_id != normalized:
            continue
        enriched = _enrich_arxiv_record(
            {
                "id": rec.id,
                "title": rec.title,
                "abstract": rec.abstract,
                "authors": rec.authors,
                "categories": rec.categories,
            }
        )
        universe_record = _paper_universe_find_node_by_id(normalized)
        if universe_record is not None:
            for key in ("year", "primary_category", "paper_version", "update_date"):
                value = universe_record.get(key)
                if value not in (None, "", []):
                    enriched[key] = value
            if not enriched.get("categories") and universe_record.get("categories"):
                enriched["categories"] = universe_record["categories"]
        return enriched
    return None


def _download_arxiv_pdf(arxiv_id: str, *, timeout: int = 60) -> bool:
    """
    Download a single Arxiv PDF by id into ARXIV_PDF_ROOT.

    Returns True if a new file was created, False if it already existed.

    PDFs are stored primarily under `/arxiv/pdfs/YYMM/<id>.pdf`, where `YYMM`
    comes from the first 4 digits of the Arxiv ID (e.g. `2101.00001` ->
    `/arxiv/pdfs/2101/2101.00001.pdf`). If an ID does not start with 4 digits,
    it falls back to `/arxiv/pdfs/<id>.pdf`.
    """

    # Match the downloader script behavior: use the trailing segment.
    norm_id = str(arxiv_id or "").strip().split("/")[-1]
    if not norm_id:
        raise ValueError("invalid arxiv_id")

    download_root = _choose_arxiv_pdf_download_root()

    # Decide on output directory: prefer /arxiv/pdfs/YYMM/<id>.pdf when possible.
    yymm = norm_id[:4]
    if len(yymm) == 4 and yymm.isdigit():
        out_dir = download_root / yymm
    else:
        out_dir = download_root
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_url = f"https://export.arxiv.org/pdf/{norm_id}.pdf"
    out_path = out_dir / f"{norm_id}.pdf"

    if _find_local_arxiv_pdf(norm_id) is not None:
        return False

    resp = requests.get(pdf_url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with out_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            fh.write(chunk)
    return True


def _compute_repo_commit_count(repo_root: str, *, timeout: float = 15.0) -> Optional[int]:
    """
    Return the total number of commits reachable from HEAD for the given repo.

    This is computed on demand using `git rev-list --count HEAD`. If the path
    is not a Git repository or the command fails for any reason, this returns
    None so callers can gracefully omit the field from responses or mark it as
    unknown in the UI.
    """
    repo_root = os.path.abspath(repo_root)
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "rev-list", "--count", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=True,
        )
    except Exception:
        return None

    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return int(out)
    except ValueError:
        return None


def _resolve_export_relative_path(raw_path: str) -> Optional[Path]:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(DEFAULT_EXPORT_ROOT) / path


def _read_json_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl_rows(path: Optional[Path], *, limit: int = 20) -> List[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


@lru_cache(maxsize=4096)
def _count_jsonl_lines_cached(path_text: str, mtime_ns: int, size: int) -> int:
    del mtime_ns, size
    path = Path(path_text)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        stat = path.stat()
    except Exception:
        return 0
    return _count_jsonl_lines_cached(str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _repo_name_tokens(repo_id: str) -> Set[str]:
    tokens = re.split(r"[^A-Za-z0-9]+", str(repo_id or "").lower())
    return {token for token in tokens if len(token) >= 3}


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _repo_universe_repo_position(
    repo: Dict[str, Any],
    *,
    language_index: Dict[str, int],
    root_index: Dict[str, int],
    language_count: int,
    root_count: int,
) -> Tuple[float, float, float]:
    repo_id = str(repo.get("repo_id") or "")
    primary_language = str(repo.get("primary_language") or "unknown")
    lang_idx = language_index.get(primary_language, 0)
    root_idx = root_index.get(str(repo.get("library_root") or ""), 0)
    lang_base = (2.0 * math.pi * lang_idx) / max(1, language_count)
    root_bias = (2.0 * math.pi * root_idx) / max(1, root_count)
    angle = lang_base + (_stable_unit_interval(repo_id + ":angle") - 0.5) * 0.75 + math.sin(root_bias) * 0.16
    entity_count = max(0, int(repo.get("entity_count") or 0))
    radius = 1.15 + min(1.65, math.log10(entity_count + 10) * 0.42) + _stable_unit_interval(repo_id + ":radius") * 0.55
    x = math.cos(angle) * radius
    y = math.sin(angle) * radius
    z = (
        (_stable_unit_interval(repo_id + ":z") - 0.5) * 2.4
        + (0.45 if repo.get("qa_ready") else -0.15)
        + min(0.7, math.log10(max(1, int(repo.get("miner_skill_count") or 0)) + 1) * 0.18)
    )
    return (round(x, 5), round(y, 5), round(z, 5))


def _repo_universe_payload(
    *,
    manifest: Optional[Dict[str, Any]] = None,
    export_root: Optional[Path] = None,
    max_similarity_edges: int = 350,
) -> Dict[str, Any]:
    manifest_data = manifest if manifest is not None else load_manifest()
    repos_meta = manifest_data.get("repos") or {}
    if not isinstance(repos_meta, dict):
        repos_meta = {}

    root = Path(export_root or DEFAULT_EXPORT_ROOT)
    repo_rows: List[Dict[str, Any]] = []
    language_counts: Dict[str, int] = {}
    library_root_counts: Dict[str, int] = {}
    qa_count = 0

    for repo_id_raw, entry_any in sorted(repos_meta.items()):
        repo_id = str(repo_id_raw or "").strip()
        if not repo_id or not isinstance(entry_any, dict):
            continue
        entry = entry_any
        state = entry.get("repo_state") or {}
        if not isinstance(state, dict):
            state = {}
        languages_any = entry.get("languages") or []
        languages = sorted(
            {
                str(lang or "").strip().lower()
                for lang in languages_any
                if str(lang or "").strip()
            }
        )
        for lang in languages:
            language_counts[lang] = language_counts.get(lang, 0) + 1

        library_root = str(entry.get("library_root") or "").strip()
        library_label = Path(library_root).name if library_root else "unknown-root"
        library_root_counts[library_label] = library_root_counts.get(library_label, 0) + 1

        skills = entry.get("skills") or {}
        if not isinstance(skills, dict):
            skills = {}
        qa_ready = bool(isinstance(skills.get("qa"), dict) and skills["qa"].get("status") == "up_to_date")
        if qa_ready:
            qa_count += 1

        extensions = entry.get("extensions") or {}
        if not isinstance(extensions, dict):
            extensions = {}
        miner = extensions.get("repo_skills_miner") or {}
        if not isinstance(miner, dict):
            miner = {}
        miner_counts = miner.get("counts") or {}
        if not isinstance(miner_counts, dict):
            miner_counts = {}

        repo_dir = root / repo_id
        entity_count = _count_jsonl_lines(repo_dir / f"{repo_id}.entities.jsonl")
        edge_count = _count_jsonl_lines(repo_dir / f"{repo_id}.edges.jsonl")
        artifact_count = _count_jsonl_lines(repo_dir / f"{repo_id}.artifacts.jsonl")

        repo_rows.append(
            {
                "id": f"repo:{repo_id}",
                "kind": "repo",
                "label": repo_id,
                "repo_id": repo_id,
                "repo_root": entry.get("repo_root") or "",
                "library_root": library_label,
                "branch": state.get("branch") or "",
                "head": state.get("head") or "",
                "languages": languages,
                "primary_language": languages[0] if languages else "",
                "entity_count": entity_count,
                "edge_count": edge_count,
                "artifact_count": artifact_count,
                "qa_ready": qa_ready,
                "has_indices": bool(entry.get("indices")),
                "has_extensions": bool(extensions),
                "miner_skill_count": int(miner_counts.get("skills") or 0),
                "name_tokens": sorted(_repo_name_tokens(repo_id)),
            }
        )

    language_order = [
        lang for lang, _count in sorted(language_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    root_order = [
        label for label, _count in sorted(library_root_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    language_index = {lang: idx for idx, lang in enumerate(language_order)}
    root_index = {label: idx for idx, label in enumerate(root_order)}
    for repo in repo_rows:
        x, y, z = _repo_universe_repo_position(
            repo,
            language_index=language_index,
            root_index=root_index,
            language_count=len(language_index),
            root_count=len(root_index),
        )
        repo["x"] = x
        repo["y"] = y
        repo["z"] = z

    nodes: List[Dict[str, Any]] = list(repo_rows)
    for idx, lang in enumerate(language_order):
        count = language_counts[lang]
        angle = (2.0 * math.pi * idx) / max(1, len(language_order))
        nodes.append(
            {
                "id": f"language:{lang}",
                "kind": "language",
                "label": lang,
                "repo_count": count,
                "x": round(math.cos(angle) * 4.1, 5),
                "y": round(math.sin(angle) * 4.1, 5),
                "z": 0.0,
            }
        )
    for idx, library_label in enumerate(root_order):
        count = library_root_counts[library_label]
        angle = (2.0 * math.pi * idx) / max(1, len(root_order))
        nodes.append(
            {
                "id": f"root:{library_label}",
                "kind": "library_root",
                "label": library_label,
                "repo_count": count,
                "x": round(math.cos(angle) * 1.35, 5),
                "y": round(math.sin(angle) * 1.35, 5),
                "z": -2.65,
            }
        )
    if qa_count:
        nodes.append(
            {
                "id": "skill:qa",
                "kind": "skill",
                "label": "QA ready",
                "repo_count": qa_count,
                "x": 0.0,
                "y": 0.0,
                "z": 3.15,
            }
        )

    edges: List[Dict[str, Any]] = []
    for repo in repo_rows:
        repo_node = repo["id"]
        for lang in repo.get("languages") or []:
            edges.append(
                {
                    "id": f"{repo_node}->language:{lang}",
                    "source": repo_node,
                    "target": f"language:{lang}",
                    "type": "uses_language",
                    "weight": 0.35,
                }
            )
        library_label = str(repo.get("library_root") or "")
        if library_label:
            edges.append(
                {
                    "id": f"{repo_node}->root:{library_label}",
                    "source": repo_node,
                    "target": f"root:{library_label}",
                    "type": "in_library_root",
                    "weight": 0.2,
                }
            )
        if repo.get("qa_ready"):
            edges.append(
                {
                    "id": f"{repo_node}->skill:qa",
                    "source": repo_node,
                    "target": "skill:qa",
                    "type": "has_skill",
                    "weight": 0.25,
                }
            )

    similarity_edges: List[Dict[str, Any]] = []
    for idx, left in enumerate(repo_rows):
        left_langs = set(left.get("languages") or [])
        left_tokens = set(left.get("name_tokens") or [])
        for right in repo_rows[idx + 1 :]:
            right_langs = set(right.get("languages") or [])
            right_tokens = set(right.get("name_tokens") or [])
            lang_union = left_langs | right_langs
            lang_score = (len(left_langs & right_langs) / len(lang_union)) if lang_union else 0.0
            token_union = left_tokens | right_tokens
            token_score = (len(left_tokens & right_tokens) / len(token_union)) if token_union else 0.0
            root_score = 1.0 if left.get("library_root") and left.get("library_root") == right.get("library_root") else 0.0
            qa_score = 1.0 if left.get("qa_ready") and right.get("qa_ready") else 0.0
            score = (0.68 * lang_score) + (0.16 * token_score) + (0.08 * root_score) + (0.08 * qa_score)
            if score < 0.34:
                continue
            similarity_edges.append(
                {
                    "id": f"{left['id']}--{right['id']}",
                    "source": left["id"],
                    "target": right["id"],
                    "type": "similar_repo",
                    "weight": round(score, 4),
                }
            )

    similarity_edges.sort(key=lambda edge: float(edge.get("weight") or 0.0), reverse=True)
    edges.extend(similarity_edges[: max(0, int(max_similarity_edges or 0))])

    return {
        "type": "repo_universe",
        "repo_count": len(repo_rows),
        "language_count": len(language_counts),
        "library_root_count": len(library_root_counts),
        "qa_ready_count": qa_count,
        "nodes": nodes,
        "edges": edges,
        "similarity_edge_count": min(len(similarity_edges), max(0, int(max_similarity_edges or 0))),
        "similarity_edge_total": len(similarity_edges),
    }


def _paper_universe_asset_url(path: Path, *, universe_root: Optional[Path] = None) -> Optional[str]:
    if not path.is_file():
        return None
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    try:
        rel_path = path.resolve().relative_to(root)
    except Exception:
        return None
    return f"{PAPER_UNIVERSE_URL_ROOT}/{'/'.join(rel_path.parts)}"


def _paper_universe_assets_payload(universe_root: Optional[Path] = None) -> Dict[str, Any]:
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    render_manifest = _read_json_file(root / "render_manifest.json")
    viewer_manifest = _read_json_file(root / "viewer_manifest.json")
    progress = _read_json_file(root / "progress.json")

    overview_path = root / "universe_3d.png"
    detailed_path = root / "universe_3d_detailed.png"
    viewer_path = root / "universe_3d_hover.html"
    interactive_manifest_path = root / "interactive" / "manifest.json"

    return {
        "type": "paper_universe_assets",
        "available": bool(root.is_dir()),
        "root_dir": str(root),
        "root_url": PAPER_UNIVERSE_URL_ROOT,
        "overview_image_url": _paper_universe_asset_url(overview_path, universe_root=root),
        "detailed_image_url": _paper_universe_asset_url(detailed_path, universe_root=root),
        "interactive_viewer_url": _paper_universe_asset_url(viewer_path, universe_root=root),
        "interactive_manifest_url": _paper_universe_asset_url(interactive_manifest_path, universe_root=root),
        "render_manifest": render_manifest,
        "viewer_manifest": viewer_manifest,
        "progress": progress,
    }


def _paper_text_dataset_root() -> Optional[Path]:
    for root in PAPER_TEXT_DATASET_ROOTS:
        if root.is_dir() and any(root.glob("train*.parquet")):
            return root
    return None


@lru_cache(maxsize=1)
def _paper_text_dataset_files() -> Tuple[str, ...]:
    root = _paper_text_dataset_root()
    if root is None:
        return ()
    files = sorted(root.glob("train*.parquet"))
    return tuple(str(path) for path in files if path.suffix == ".parquet")


@lru_cache(maxsize=1)
def _paper_text_dataset() -> Any:
    files = list(_paper_text_dataset_files())
    if not files:
        return None
    try:
        import pyarrow.dataset as ds  # type: ignore

        return ds.dataset(files, format="parquet")
    except Exception:
        return None


def _paper_text_row_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize_arxiv_paper_id(paper_id)
    dataset = _paper_text_dataset()
    if not normalized or dataset is None:
        return None

    try:
        import pyarrow.dataset as ds  # type: ignore
    except Exception:
        return None

    raw_paper_id = str(paper_id or "").strip().split("/")[-1]
    filters = [ds.field("canonical_paper_id") == normalized]
    if raw_paper_id and raw_paper_id != normalized:
        filters.insert(0, ds.field("paper_id") == raw_paper_id)

    for filter_expr in filters:
        try:
            rows = (
                dataset.scanner(
                    columns=list(PAPER_TEXT_LOOKUP_COLUMNS),
                    filter=filter_expr,
                )
                .to_table()
                .to_pylist()
            )
        except Exception:
            rows = []
        if rows:
            return rows[0]
    return None


def _paper_text_existing_pdf_path(row: Dict[str, Any]) -> Optional[Path]:
    row_pdf_path = str(row.get("pdf_path") or "").strip()
    if row_pdf_path:
        candidate = Path(row_pdf_path)
        if candidate.is_file():
            return candidate

    for candidate_id in (
        str(row.get("paper_id") or "").strip(),
        str(row.get("canonical_paper_id") or "").strip(),
    ):
        normalized = _normalize_arxiv_paper_id(candidate_id)
        if not normalized:
            continue
        found = _find_local_arxiv_pdf(normalized)
        if found is not None and found.is_file():
            return found
    return None


def _normalize_page_text(raw_text: str) -> str:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    cleaned: List[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if cleaned and blank_run <= 1:
                cleaned.append("")
            continue
        blank_run = 0
        cleaned.append(line.strip())
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _looks_like_heading(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    single = " ".join(normalized.split())
    if len(single) > 120:
        return False
    if sum(1 for ch in single if ch.isalpha()) < 3:
        return False
    if re.search(r"[-=]{2,}|\\|/|[∑∫√≤≥≠≈±]", single):
        return False
    if re.search(r"\b[A-Za-z]\s*[=<>]\s*[-+0-9A-Za-z]", single):
        return False
    if single.count("-") >= 3:
        return False
    lower = single.lower().rstrip(".:")
    if lower in {
        "abstract",
        "introduction",
        "references",
        "acknowledgements",
        "acknowledgments",
        "appendix",
        "conclusion",
        "conclusions",
        "discussion",
        "results",
        "method",
        "methods",
        "related work",
        "experiments",
    }:
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", single):
        return True
    if re.match(r"^(Appendix|Section)\b", single):
        return True
    letters = [ch for ch in single if ch.isalpha()]
    uppercase_ratio = (
        sum(1 for ch in letters if ch.isupper()) / float(len(letters))
        if letters
        else 0.0
    )
    if uppercase_ratio > 0.72 and len(single.split()) <= 8:
        return True
    return False


def _reflow_paragraph_lines(lines: List[str]) -> str:
    out = ""
    for raw_line in lines:
        line = " ".join(str(raw_line or "").split())
        if not line:
            continue
        if not out:
            out = line
            continue
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
            continue
        if _looks_like_heading(out) or _looks_like_heading(line):
            out += "\n" + line
            continue
        out += " " + line
    return out.strip()


def _page_blocks_from_text(text: str) -> List[Dict[str, Any]]:
    normalized = _normalize_page_text(text)
    if not normalized:
        return []

    blocks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    for raw_line in normalized.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            if current_lines:
                paragraph = _reflow_paragraph_lines(current_lines)
                if paragraph:
                    blocks.append(
                        {
                            "kind": "heading" if _looks_like_heading(paragraph) else "paragraph",
                            "text": paragraph,
                        }
                    )
                current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        paragraph = _reflow_paragraph_lines(current_lines)
        if paragraph:
            blocks.append(
                {
                    "kind": "heading" if _looks_like_heading(paragraph) else "paragraph",
                    "text": paragraph,
                }
            )
    return blocks


def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(str(block.get("text") or "").strip() for block in blocks if str(block.get("text") or "").strip()).strip()


def _split_blocks_into_pages(blocks: List[Dict[str, Any]], page_count: int) -> List[List[Dict[str, Any]]]:
    if not blocks:
        return []
    total_pages = max(1, int(page_count or 1))
    if total_pages <= 1 or len(blocks) <= 1:
        return [blocks]

    weights = [max(1, len(str(block.get("text") or ""))) for block in blocks]
    total_weight = sum(weights)
    target_weight = max(1, total_weight // total_pages)

    pages: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_weight = 0
    for idx, (block, weight) in enumerate(zip(blocks, weights)):
        remaining_blocks = len(blocks) - idx
        remaining_pages = total_pages - len(pages)
        must_break = remaining_blocks <= remaining_pages
        if current and current_weight >= target_weight and len(pages) < total_pages - 1 and not must_break:
            pages.append(current)
            current = []
            current_weight = 0
        current.append(block)
        current_weight += weight

    if current:
        pages.append(current)

    if len(pages) > total_pages:
        collapsed = pages[: total_pages - 1]
        tail: List[Dict[str, Any]] = []
        for page_blocks in pages[total_pages - 1 :]:
            tail.extend(page_blocks)
        collapsed.append(tail)
        pages = collapsed

    while len(pages) < total_pages and pages:
        last = pages[-1]
        if len(last) <= 1:
            break
        split_at = max(1, len(last) // 2)
        pages[-1] = last[:split_at]
        pages.append(last[split_at:])

    return [page for page in pages if page]


def _split_text_lines_exact(text: str, page_count: int) -> List[str]:
    normalized = _normalize_page_text(text)
    if not normalized:
        return []
    total_pages = max(1, int(page_count or 1))
    if total_pages <= 1:
        return [normalized]
    lines = normalized.splitlines()
    if len(lines) <= 1:
        return [normalized]

    pages: List[str] = []
    for idx in range(total_pages):
        start = round(idx * len(lines) / total_pages)
        end = round((idx + 1) * len(lines) / total_pages)
        if end <= start:
            continue
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            pages.append(chunk)
    return pages or [normalized]


def _extract_pdf_pages(path: Path) -> List[str]:
    if not path.is_file():
        return []
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", "-q", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=12,
        )
    except Exception:
        return []
    raw_text = str(proc.stdout or "")
    if not raw_text:
        return []
    pages = []
    for raw_page in raw_text.split("\x0c"):
        page_text = _normalize_page_text(raw_page)
        if page_text:
            pages.append(page_text)
    return pages


def _paper_text_pages_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    text = str(row.get("text") or "").strip()
    reported_page_count = max(1, int(row.get("page_count") or 1)) if text else 0
    exact_pages = _extract_pdf_pages(_paper_text_existing_pdf_path(row) or Path(""))
    page_mode = "inferred"

    if exact_pages:
        page_mode = "exact_pdf"
        pages = exact_pages
    else:
        blocks = _page_blocks_from_text(text)
        split_pages = _split_blocks_into_pages(blocks, reported_page_count)
        if reported_page_count > 1 and len(split_pages) != reported_page_count:
            pages = _split_text_lines_exact(text, reported_page_count)
        else:
            pages = [_blocks_to_text(page_blocks) for page_blocks in split_pages] if split_pages else ([text] if text else [])
        if len(pages) == 1:
            page_mode = "single_page"

    page_payloads: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = []
    for idx, page_text in enumerate(pages, start=1):
        blocks = _page_blocks_from_text(page_text)
        headings = [str(block.get("text") or "").strip() for block in blocks if block.get("kind") == "heading"]
        for heading in headings:
            sections.append({"title": heading, "page": idx})
        page_payloads.append(
            {
                "page": idx,
                "text": page_text,
                "blocks": blocks,
                "heading_titles": headings,
                "char_count": len(page_text),
                "line_count": len([line for line in page_text.splitlines() if line.strip()]),
            }
        )

    return {
        "page_mode": page_mode,
        "reported_page_count": reported_page_count,
        "pages": page_payloads,
        "sections": sections,
    }


def _paper_text_payload(paper_id: str) -> Optional[Dict[str, Any]]:
    row = _paper_text_row_by_id(paper_id)
    if row is None:
        return None

    normalized = _normalize_arxiv_paper_id(paper_id)
    row_paper_id = str(row.get("paper_id") or normalized).strip()
    canonical_id = _normalize_arxiv_paper_id(str(row.get("canonical_paper_id") or row_paper_id))
    page_info = _paper_text_pages_payload(row)
    text = str(row.get("text") or "")
    local_pdf = _paper_text_existing_pdf_path(row)
    source_pdf_url = _arxiv_pdf_url(row_paper_id or canonical_id)
    return {
        "type": "paper_text_result",
        "paper_id": row_paper_id,
        "canonical_paper_id": canonical_id or row_paper_id,
        "paper_version": str(row.get("paper_version") or "").strip(),
        "dataset_root": str(_paper_text_dataset_root() or ""),
        "title": str(row.get("title") or "").strip(),
        "authors": str(row.get("authors") or "").strip(),
        "categories": str(row.get("categories") or "").strip(),
        "license": str(row.get("license") or "").strip(),
        "update_date": str(row.get("update_date") or "").strip(),
        "text_source": str(row.get("text_source") or "").strip(),
        "text_is_partial": bool(row.get("text_is_partial")),
        "text_char_count": int(row.get("text_char_count") or len(text)),
        "text_line_count": int(row.get("text_line_count") or len([line for line in text.splitlines() if line.strip()])),
        "token_count": int(row.get("token_count") or 0),
        "page_count": len(page_info["pages"]),
        "reported_page_count": int(page_info["reported_page_count"] or 0),
        "page_mode": str(page_info["page_mode"]),
        "sections": page_info["sections"],
        "pages": page_info["pages"],
        "has_local_pdf": bool(local_pdf is not None),
        "source_pdf_url": source_pdf_url,
        "token_types": [str(item).strip() for item in (row.get("token_types") or []) if str(item).strip()],
        "token_type_counts_json": str(row.get("token_type_counts_json") or "").strip(),
    }


def _paper_universe_table_rows(
    path: Path,
    *,
    columns: Optional[List[str]] = None,
    filters: Optional[List[tuple[str, str, object]]] = None,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(
            path,
            columns=columns,
            filters=filters,
        )
    except Exception:
        return []
    try:
        rows = table.to_pylist()
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def _paper_universe_has_pdf(row: Dict[str, Any]) -> bool:
    paper_id = _normalize_arxiv_paper_id(str(row.get("paper_id") or row.get("canonical_paper_id") or ""))
    if paper_id and _find_local_arxiv_pdf(paper_id) is not None:
        return True
    pdf_path = str(row.get("pdf_path") or "").strip()
    return bool(pdf_path and Path(pdf_path).is_file())


def _paper_universe_node_record(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_paper_id = str(row.get("paper_id") or row.get("canonical_paper_id") or "").strip()
    canonical_id = _normalize_arxiv_paper_id(str(row.get("canonical_paper_id") or raw_paper_id))
    category_list_raw = row.get("categories")
    category_list: List[str]
    if isinstance(category_list_raw, list):
        category_list = [str(item).strip() for item in category_list_raw if str(item).strip()]
    elif isinstance(category_list_raw, str) and category_list_raw.strip():
        category_list = [part.strip() for part in category_list_raw.split(",") if part.strip()]
    else:
        primary = str(row.get("primary_category") or "").strip()
        category_list = [primary] if primary else []

    year_raw = row.get("year")
    try:
        year = int(year_raw) if year_raw is not None else None
    except Exception:
        year = None

    return {
        "paper_idx": int(row.get("paper_idx") or 0),
        "id": raw_paper_id or canonical_id,
        "paper_id": raw_paper_id or canonical_id,
        "canonical_paper_id": canonical_id or raw_paper_id,
        "paper_version": str(row.get("paper_version") or "").strip(),
        "title": str(row.get("title") or "").strip(),
        "authors": str(row.get("authors") or "").strip(),
        "categories": ", ".join(category_list),
        "category_list": category_list,
        "primary_category": str(row.get("primary_category") or "").strip(),
        "update_date": str(row.get("update_date") or "").strip(),
        "year": year,
        "pdf_path": str(row.get("pdf_path") or "").strip(),
        "has_pdf": _paper_universe_has_pdf(row),
        "x": float(row.get("x") or 0.0),
        "y": float(row.get("y") or 0.0),
        "z": float(row.get("z") or 0.0),
    }


def _paper_universe_find_node_by_id(
    paper_id: str,
    *,
    universe_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    paper_nodes_path = root / "paper_nodes.parquet"
    if not paper_nodes_path.is_file():
        return None

    normalized = _normalize_arxiv_paper_id(paper_id)
    columns = [
        "paper_idx",
        "paper_id",
        "canonical_paper_id",
        "paper_version",
        "title",
        "authors",
        "categories",
        "primary_category",
        "update_date",
        "year",
        "pdf_path",
        "x",
        "y",
        "z",
    ]

    rows: List[Dict[str, Any]] = []
    if normalized:
        rows = _paper_universe_table_rows(
            paper_nodes_path,
            columns=columns,
            filters=[("canonical_paper_id", "==", normalized)],
        )
    if not rows:
        raw_paper_id = str(paper_id or "").strip().split("/")[-1]
        if raw_paper_id:
            rows = _paper_universe_table_rows(
                paper_nodes_path,
                columns=columns,
                filters=[("paper_id", "==", raw_paper_id)],
            )
    if not rows:
        return None
    return _paper_universe_node_record(rows[0])


def _paper_universe_nodes_by_idx(
    paper_indices: List[int],
    *,
    universe_root: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    paper_nodes_path = root / "paper_nodes.parquet"
    idx_values = sorted({int(idx) for idx in paper_indices})
    if not idx_values or not paper_nodes_path.is_file():
        return {}

    rows = _paper_universe_table_rows(
        paper_nodes_path,
        columns=[
            "paper_idx",
            "paper_id",
            "canonical_paper_id",
            "paper_version",
            "title",
            "authors",
            "categories",
            "primary_category",
            "update_date",
            "year",
            "pdf_path",
            "x",
            "y",
            "z",
        ],
        filters=[("paper_idx", "in", idx_values)],
    )
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        record = _paper_universe_node_record(row)
        out[int(record["paper_idx"])] = record
    return out


def _paper_universe_category_records(
    category_ids: List[str],
    *,
    universe_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    wanted = sorted({str(category_id).strip() for category_id in category_ids if str(category_id).strip()})
    if not wanted:
        return {}
    rows = _paper_universe_table_rows(
        root / "category_nodes.parquet",
        columns=["category_id", "name", "paper_count"],
        filters=[("category_id", "in", wanted)],
    )
    return {str(row.get("category_id") or ""): row for row in rows}


def _paper_universe_topic_records(
    topic_ids: List[str],
    *,
    universe_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    wanted = sorted({str(topic_id).strip() for topic_id in topic_ids if str(topic_id).strip()})
    if not wanted:
        return {}
    rows = _paper_universe_table_rows(
        root / "topic_nodes.parquet",
        columns=["topic_id", "name", "paper_count"],
        filters=[("topic_id", "in", wanted)],
    )
    return {str(row.get("topic_id") or ""): row for row in rows}


def _paper_universe_selected_metadata_edges(
    selected_idx: int,
    *,
    universe_root: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    return {
        "categories": _paper_universe_table_rows(
            root / "edges.parquet",
            columns=["src_paper_idx", "dst_category_id", "type"],
            filters=[("src_paper_idx", "==", int(selected_idx))],
        ),
        "topics": _paper_universe_table_rows(
            root / "paper_topic_edges.parquet",
            columns=["src_paper_idx", "dst_topic_id", "type"],
            filters=[("src_paper_idx", "==", int(selected_idx))],
        ),
        "years": _paper_universe_table_rows(
            root / "paper_year_edges.parquet",
            columns=["src_paper_idx", "dst_year", "type"],
            filters=[("src_paper_idx", "==", int(selected_idx))],
        ),
    }


def _selected_paper_section_nodes(paper_id: str, *, max_sections: int = 10) -> List[Dict[str, Any]]:
    payload = _paper_text_payload(paper_id)
    if not payload:
        return []
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int]] = set()
    for section in payload.get("sections") or []:
        title = str(section.get("title") or "").strip()
        if not title or not _looks_like_heading(title):
            continue
        try:
            page = int(section.get("page") or 1)
        except Exception:
            page = 1
        key = (title.lower(), page)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "title": title,
                "page": page,
                "id": f"section:{page}:{len(out)}",
            }
        )
        if len(out) >= max(1, int(max_sections or 1)):
            break
    return out


def _paper_universe_neighborhood_payload(
    paper_id: str,
    *,
    neighbor_limit: int = 12,
    universe_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = (universe_root or PAPER_UNIVERSE_ROOT).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"paper universe not found under {root}")

    selected = _paper_universe_find_node_by_id(paper_id, universe_root=root)
    if selected is None:
        raise KeyError(f"paper not found in paper universe: {paper_id}")

    edges_path = root / "paper_knn_edges.parquet"
    if not edges_path.is_file():
        raise FileNotFoundError(f"paper KNN edges not found under {root}")

    limit = max(1, min(int(neighbor_limit or 12), 32))
    edge_rows = _paper_universe_table_rows(
        edges_path,
        columns=["src_paper_idx", "dst_paper_idx", "type", "weight"],
        filters=[("src_paper_idx", "==", int(selected["paper_idx"]))],
    )
    edge_rows.sort(key=lambda row: float(row.get("weight") or 0.0), reverse=True)

    trimmed_edges: List[Dict[str, Any]] = []
    seen_neighbor_idx: Set[int] = set()
    for row in edge_rows:
        dst_idx = int(row.get("dst_paper_idx") or -1)
        if dst_idx < 0 or dst_idx in seen_neighbor_idx:
            continue
        seen_neighbor_idx.add(dst_idx)
        trimmed_edges.append(row)
        if len(trimmed_edges) >= limit:
            break

    neighbor_records = _paper_universe_nodes_by_idx(
        [int(row.get("dst_paper_idx") or -1) for row in trimmed_edges if int(row.get("dst_paper_idx") or -1) >= 0],
        universe_root=root,
    )

    selected_id = str(selected.get("canonical_paper_id") or selected.get("paper_id") or selected.get("paper_idx"))
    nodes: List[Dict[str, Any]] = [
        {
            **selected,
            "id": selected_id,
            "label": selected.get("title") or selected_id,
            "role": "selected",
            "similarity": 1.0,
        }
    ]
    graph_edges: List[Dict[str, Any]] = []
    selected_graph_id = f"paper:{selected_id}"
    for row in trimmed_edges:
        neighbor_idx = int(row.get("dst_paper_idx") or -1)
        neighbor = neighbor_records.get(neighbor_idx)
        if not neighbor:
            continue
        neighbor_id = str(neighbor.get("canonical_paper_id") or neighbor.get("paper_id") or neighbor.get("paper_idx"))
        similarity = float(row.get("weight") or 0.0)
        nodes.append(
            {
                **neighbor,
                "id": neighbor_id,
                "label": neighbor.get("title") or neighbor_id,
                "role": "neighbor",
                "similarity": similarity,
            }
        )
        graph_edges.append(
            {
                "source": selected_graph_id,
                "target": f"paper:{neighbor_id}",
                "weight": similarity,
                "type": str(row.get("type") or "paper_knn"),
            }
        )

    paper_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("canonical_paper_id") or node.get("paper_id") or node.get("paper_idx"))
        paper_nodes.append(
            {
                **node,
                "id": f"paper:{node_id}",
                "paper_node_id": node_id,
                "label": node.get("title") or node_id,
                "kind": "paper",
            }
        )

    metadata_edges = _paper_universe_selected_metadata_edges(int(selected["paper_idx"]), universe_root=root)
    category_ids = [
        str(row.get("dst_category_id") or "").strip()
        for row in metadata_edges["categories"]
        if str(row.get("dst_category_id") or "").strip()
    ]
    if not category_ids:
        category_ids = [str(item).strip() for item in selected.get("category_list", []) if str(item).strip()]
    category_records = _paper_universe_category_records(category_ids, universe_root=root)

    topic_ids = [
        str(row.get("dst_topic_id") or "").strip()
        for row in metadata_edges["topics"]
        if str(row.get("dst_topic_id") or "").strip()
    ][:6]
    topic_records = _paper_universe_topic_records(topic_ids, universe_root=root)

    graph_nodes: List[Dict[str, Any]] = list(paper_nodes)
    for category_id in category_ids:
        record = category_records.get(category_id) or {}
        graph_nodes.append(
            {
                "id": f"category:{category_id}",
                "label": category_id,
                "kind": "category",
                "category_id": category_id,
                "paper_count": int(record.get("paper_count") or 0),
            }
        )
        graph_edges.append(
            {
                "source": selected_graph_id,
                "target": f"category:{category_id}",
                "type": "has_category",
                "weight": 1.0,
            }
        )

    selected_year = selected.get("year")
    year_values = [
        row.get("dst_year")
        for row in metadata_edges["years"]
        if row.get("dst_year") is not None
    ] or ([selected_year] if selected_year is not None else [])
    for year_value in year_values[:1]:
        try:
            year = int(year_value)
        except Exception:
            continue
        graph_nodes.append(
            {
                "id": f"year:{year}",
                "label": str(year),
                "kind": "year",
                "year": year,
            }
        )
        graph_edges.append(
            {
                "source": selected_graph_id,
                "target": f"year:{year}",
                "type": "published_in",
                "weight": 1.0,
            }
        )

    for topic_id in topic_ids:
        record = topic_records.get(topic_id) or {}
        graph_nodes.append(
            {
                "id": f"topic:{topic_id}",
                "label": str(record.get("name") or topic_id),
                "kind": "topic",
                "topic_id": topic_id,
                "paper_count": int(record.get("paper_count") or 0),
            }
        )
        graph_edges.append(
            {
                "source": selected_graph_id,
                "target": f"topic:{topic_id}",
                "type": "has_topic",
                "weight": 1.0,
            }
        )

    section_nodes = _selected_paper_section_nodes(selected_id, max_sections=10)
    for section in section_nodes:
        graph_nodes.append(
            {
                "id": section["id"],
                "label": section["title"],
                "kind": "section",
                "page": int(section["page"]),
            }
        )
        graph_edges.append(
            {
                "source": selected_graph_id,
                "target": section["id"],
                "type": "has_section",
                "weight": 1.0,
            }
        )

    # Connect similar papers back to selected paper categories when they share
    # a category. This gives the graph a useful local structure instead of a
    # flat star centered only on the selected paper.
    selected_categories = set(category_ids)
    for node in paper_nodes:
        if node.get("role") == "selected":
            continue
        node_id = str(node.get("paper_node_id") or "")
        for category_id in selected_categories.intersection(set(node.get("category_list") or [])):
            graph_edges.append(
                {
                    "source": node["id"],
                    "target": f"category:{category_id}",
                    "type": "shares_category",
                    "weight": 0.45,
                }
            )

    return {
        "type": "selected_paper_graph",
        "paper": selected,
        "nodes": graph_nodes,
        "edges": graph_edges,
        "neighbor_limit": limit,
        "neighbor_count": len(trimmed_edges),
        "section_count": len(section_nodes),
        "topic_count": len(topic_ids),
        "category_count": len(category_ids),
        "available": True,
    }


def _load_base_llm() -> Any:
    """
    Load the base LLM used for all skills.

    By default this loads `meta-llama/Llama-3.1-8B-Instruct` (or a compatible
    checkpoint) from `/data/checkpoints`, but this can be overridden via:

    - `LLAMA_MODEL_PATH`: absolute or HF-style model path.

    This is loaded once per process and reused for all requests.
    """
    global _LLM_MODEL, _LLM_TOKENIZER

    if _LLM_MODEL is not None and _LLM_TOKENIZER is not None:
        return _LLM_MODEL, _LLM_TOKENIZER

    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(
            "transformers/torch are not installed; cannot load base QA LLM."
        )

    # Delegate base model loading to the shared QA runtime so that
    # server startup and QA skills share a single underlying model
    # instance (and cache entry).
    cfg: QAModelConfig = get_default_qa_base_config()
    model, tokenizer = get_or_load_model(cfg)

    _LLM_MODEL = model
    _LLM_TOKENIZER = tokenizer
    return _LLM_MODEL, _LLM_TOKENIZER


def _llm_generate_answer(prompt: str, *, max_new_tokens: int = 512) -> str:
    """
    Run a single-turn completion against the base LLM.
    """
    model, tokenizer = _load_base_llm()
    if torch is None:
        raise RuntimeError("torch is not available; cannot run LLM inference.")

    # Use the same device policy as `_load_base_llm`: default to CPU unless
    # LLM_DEVICE=cuda is explicitly set and CUDA is available.
    device = os.environ.get("LLM_DEVICE", "cpu").lower()

    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda" and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}  # type: ignore[assignment]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.strip()


def _make_repo_library() -> RepoLibrary:
    """
    Construct a RepoLibrary instance.

    To keep server startup lightweight and robust, we *do not* load the
    LLM at import time. The RepoLibrary itself only needs the manifest
    and graph exports; LLM-backed flows call `_load_base_llm()` lazily
    via `_llm_generate_answer`.
    """
    base_model = None  # LLM is optional for RepoLibrary planning.
    adapter_bank = FileAdapterBank()
    return RepoLibrary(base_model=base_model, adapter_bank=adapter_bank)


repo_lib = _make_repo_library()


def _get_coarse_lane_retriever():
    """
    Lazily build the coarse lane-based retriever over:
    - repo semantic summaries
    - paper↔repo alignments
    - coarse paper-span ↔ repo-chunk bridge rows
    """
    global _COARSE_LANE_RETRIEVER
    if _COARSE_LANE_RETRIEVER is not None:
        return _COARSE_LANE_RETRIEVER

    from models.mirrormind import CoarseLaneRetriever

    _COARSE_LANE_RETRIEVER = CoarseLaneRetriever(
        repo_semantic_path="models/exports/semantic_from_chunks.jsonl",
        paper_repo_align_path="exports/paper_repo_align.jsonl",
        paper_repo_span_path="exports/paper_repo_span_align.jsonl",
    )
    return _COARSE_LANE_RETRIEVER


def _execute_coarse_retrieval(
    *,
    question: str,
    top_k_repos: int = 5,
    top_k_papers: int = 5,
    top_k_spans: int = 6,
) -> Dict[str, Any]:
    retriever = _get_coarse_lane_retriever()
    return retriever.retrieve(
        question,
        top_k_repos=max(1, min(int(top_k_repos), 20)),
        top_k_papers=max(1, min(int(top_k_papers), 20)),
        top_k_spans=max(1, min(int(top_k_spans), 30)),
    )


def _shorten_for_answer(text: str, *, limit: int = 260) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return (cut or text[:limit]).rstrip(".,;:") + "..."


def _format_coarse_retrieval_answer(question: str, result: Dict[str, Any]) -> str:
    fused_repos = result.get("fused_repos") or []
    paper_hits = result.get("paper_hits") or []
    support_spans = result.get("support_spans") or []

    if not fused_repos:
        return (
            "I could not connect that query to any repo in the current "
            "paper-repo alignment set."
        )

    best_repo = fused_repos[0]
    repo_id = str(best_repo.get("repo_id") or "")
    repo_score = float(best_repo.get("score") or 0.0)
    summary_text = _shorten_for_answer(str(best_repo.get("summary_text") or ""), limit=320)
    key_concepts = [str(x) for x in (best_repo.get("key_concepts") or []) if str(x)]

    repo_papers = [hit for hit in paper_hits if str(hit.get("repo_id") or "") == repo_id]
    top_paper = repo_papers[0] if repo_papers else (paper_hits[0] if paper_hits else None)

    repo_spans = [hit for hit in support_spans if str(hit.get("repo_id") or "") == repo_id]
    if not repo_spans:
        repo_spans = support_spans[:]

    lines: List[str] = []
    header = f"Top coarse match: {repo_id}"
    if top_paper:
        header += (
            f", aligned to paper "
            f"{str(top_paper.get('paper_title') or '')} "
            f"({str(top_paper.get('paper_id') or '')})"
        )
    header += f" [score {repo_score:.3f}]"
    lines.append(header)

    if summary_text:
        lines.append("")
        lines.append(summary_text)

    if key_concepts:
        lines.append("")
        lines.append("Key concepts: " + ", ".join(key_concepts[:8]))

    evidence_lines: List[str] = []
    seen_paths: Set[str] = set()
    for span in repo_spans:
        repo_path = str(span.get("repo_path") or "")
        if repo_path in seen_paths:
            continue
        seen_paths.add(repo_path)
        paper_excerpt = _shorten_for_answer(str(span.get("paper_text") or ""), limit=220)
        matched_terms = [str(x) for x in (span.get("matched_terms") or span.get("shared_terms") or []) if str(x)]
        page_start = span.get("page_start")
        page_end = span.get("page_end")
        page_label = ""
        if page_start is not None and page_end is not None:
            if int(page_start) == int(page_end):
                page_label = f"page {int(page_start)}"
            else:
                page_label = f"pages {int(page_start)}-{int(page_end)}"
        ev = f"- {repo_path}"
        if page_label:
            ev += f" <-> {page_label}"
        if matched_terms:
            ev += f" | terms: {', '.join(matched_terms[:6])}"
        if paper_excerpt:
            ev += f"\n  paper span: {paper_excerpt}"
        evidence_lines.append(ev)
        if len(evidence_lines) >= 3:
            break

    if evidence_lines:
        lines.append("")
        lines.append("Grounding:")
        lines.extend(evidence_lines)

    alternates = [
        hit for hit in fused_repos[1:4]
        if float(hit.get("score") or 0.0) >= 0.45 * repo_score
    ]
    if alternates:
        alt_text = ", ".join(
            f"{str(hit.get('repo_id') or '')} ({float(hit.get('score') or 0.0):.3f})"
            for hit in alternates
        )
        lines.append("")
        lines.append("Other plausible repo matches: " + alt_text)

    return "\n".join(line for line in lines if line is not None).strip()


@app.on_event("startup")
async def _preload_llm_on_startup() -> None:
    """
    Optionally preload the base LLM at process startup so the first
    user request does not pay model load latency.

    Controlled via the `PRELOAD_LLM` environment variable:
    - If unset or set to a truthy value ("1", "true", "yes", "on"),
      the LLM will be loaded during startup.
    - Any other value disables preloading and keeps the previous
      lazy-load behavior.
    """
    preload = os.environ.get("PRELOAD_LLM", "1").lower()
    if preload not in ("1", "true", "yes", "y", "on"):
        return

    if AutoModelForCausalLM is None or AutoTokenizer is None:
        # Dependencies are missing; do not fail startup.
        logger.warning(
            "PRELOAD_LLM is enabled but transformers/torch are not available; "
            "skipping LLM preloading."
        )
        return

    try:
        logger.info("Preloading base LLM model at startup...")
        _load_base_llm()
        logger.info("Base LLM model loaded successfully.")
    except Exception as exc:  # pragma: no cover - defensive
        # Do not crash the server if preloading fails; runtime calls will
        # still attempt to load and surface detailed errors.
        logger.warning("Failed to preload LLM model at startup: %s", exc)


def _format_qa_answer_stub(
    plan: Dict[str, Any],
    qa_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    LLM-backed QA executor that uses the program graph to surface relevant
    code and then answer with `meta-llama/Llama-3.1-8B-Instruct`.

    Given a single-repo QA plan, this will:
    - Open the corresponding Repository.
    - Extract simple tokens from the question.
    - Use the repo's ProgramGraph search to find matching entities.
    - Resolve those entities to artifacts and line spans.
    - Build a compact textual context summary.
    - Call the shared base LLM to produce a natural-language answer.

    If the LLM runtime is unavailable (e.g. missing checkpoints or
    dependencies), it falls back to returning the graph-based context
    summary so that the system remains debuggable.
    """
    question = str(plan.get("question") or "").strip()
    repos = plan.get("repos") or []

    if not question:
        return "Unable to execute QA: question is empty after normalization."
    if not isinstance(repos, list) or not repos:
        return (
            "Unable to execute QA: the query plan does not contain any target "
            "repositories."
        )

    # For QueryMode.QA we enforce a single repo upstream, but we guard
    # defensively here and just pick the first one.
    repo_id = str(repos[0])

    try:
        repo = open_repository(repo_id)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Unable to open repository {repo_id!r}: {exc}"

    graph = repo.graph

    # Helper to load a small, read-only code snippet for a given match. This is
    # used to *ground* the QA answer in the actual source corresponding to
    # entities like functions, tests, or variables (e.g., dictionaries defined
    # inside a test), so the model can answer questions about their contents
    # without guessing.
    def _load_snippet_for_match(
        rel_path: str,
        start_line: Optional[int],
        end_line: Optional[int],
        *,
        context: int = 3,
        max_lines: int = 40,
    ) -> Optional[str]:
        try:
            root_path = getattr(repo, "root_path", None)
        except Exception:
            root_path = None
        if not rel_path or root_path is None:
            return None
        try:
            root_resolved = root_path.resolve()
        except Exception:
            return None
        try:
            abs_path = (root_path / rel_path).resolve()
        except Exception:
            return None
        # Defensive: ensure we stay within the repo.
        if not str(abs_path).startswith(str(root_resolved)):
            return None
        if not abs_path.is_file():
            return None

        try:
            a = int(start_line) if start_line is not None else 1
            b = int(end_line) if end_line is not None else max(a, a + 1)
        except Exception:
            a, b = 1, 1

        a = max(1, a)
        b = max(a, b)
        # Expand slightly around the span for local context, but cap total lines.
        s = max(1, a - context)
        e = b + context
        lines: List[str] = []
        try:
            with abs_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for idx, line in enumerate(fh, start=1):
                    if idx < s:
                        continue
                    if len(lines) >= max_lines or idx > e:
                        break
                    # Keep original indentation; strip only the trailing newline.
                    lines.append(line.rstrip("\n"))
        except Exception:
            return None
        if not lines:
            return None
        return "\n".join(lines)

    # Build an id → Entity map so we can interpret both graph and
    # vector-index-based hits.
    try:
        entities = list(graph.entities())
    except Exception as exc:  # pragma: no cover - defensive
        return f"Failed to enumerate entities for repo {repo_id!r}: {exc}"

    entities_by_id: Dict[str, Any] = {e.id: e for e in entities}

    matches: List[Dict[str, Any]] = []

    # --- First, try vector-index-backed retrieval if a QA index is present. --- #
    index_meta: Optional[Dict[str, Any]] = None
    if qa_meta and isinstance(qa_meta, dict):
        idx_any = qa_meta.get("index")
        if isinstance(idx_any, dict):
            index_meta = idx_any

    index_key = None
    if index_meta:
        # Use the embeddings path as a stable cache key when available.
        index_key = str(index_meta.get("embeddings_path") or repo_id)

    index = None
    if index_key:
        cached = _QA_INDEX_CACHE.get(index_key)
        if cached is not None:
            index = cached
        else:
            index = load_simple_repo_index(index_meta or {})
            if index is not None:
                _QA_INDEX_CACHE[index_key] = index

    if index is not None:
        # Use similarity search as the primary signal. We query both with the
        # full natural-language question and with any "code-like" tokens
        # (e.g., `update_screenshots`) to better match function/class names.
        import re

        hit_by_id: Dict[str, Dict[str, Any]] = {}

        def _accumulate_hits(q: str, *, top_k: int = 10) -> None:
            nonlocal hit_by_id
            try:
                local_hits = index.search(q, top_k=top_k)
            except Exception:
                return
            for h in local_hits:
                ent_id = h.get("entity_id")
                if not ent_id:
                    continue
                prev = hit_by_id.get(ent_id)
                if prev is None or float(h.get("score", 0.0)) > float(
                    prev.get("score", 0.0)
                ):
                    hit_by_id[ent_id] = h

        # 1) Full-question semantic search.
        _accumulate_hits(question, top_k=15)

        # 2) Extra passes for code-like tokens (identifiers) extracted from the question.
        raw_tokens = re.findall(r"[A-Za-z0-9_]+", question)
        codey_tokens = [t for t in raw_tokens if "_" in t or t[0].isupper()]
        for tok in codey_tokens[:5]:
            _accumulate_hits(tok, top_k=5)

        for ent_id, hit in hit_by_id.items():
            ent = entities_by_id.get(ent_id)
            if not ent:
                continue
            try:
                anchor = graph.resolve(ent.uri)
                _pid, _kind, resource, _span = parse_program_uri(anchor.artifact_uri)
                path = resource
            except Exception:
                anchor = None
                path = ""

            start_line = getattr(anchor.span, "start_line", None) if anchor else None
            end_line = getattr(anchor.span, "end_line", None) if anchor else None

            matches.append(
                {
                    "token": hit.get("entity_name") or "",
                    "entity_name": getattr(ent, "name", ""),
                    "entity_kind": getattr(ent, "kind", ""),
                    "path": path,
                    "start_line": int(start_line) if start_line is not None else None,
                    "end_line": int(end_line) if end_line is not None else None,
                }
            )

    # --- Fallback / augmentation: simple token-based + fuzzy graph search. --- #
    if not matches:
        # Best-effort tokenization of the question; bias towards longer tokens
        # first so we get more specific matches.
        import re

        raw_tokens = re.findall(r"[A-Za-z0-9_]+", question)
        tokens = [t.lower() for t in raw_tokens if len(t) >= 3]
        if not tokens:
            tokens = [t.lower() for t in raw_tokens]
        # Soft limit on how many distinct tokens we will search for.
        tokens = tokens[:8]

        # 1) Exact token search via `graph.search_refs`.
        for tok in tokens:
            try:
                hits = list(graph.search_refs(tok))
            except Exception:
                hits = []
            for ent_id, span in hits:
                ent = entities_by_id.get(ent_id)
                if not ent:
                    continue
                try:
                    anchor = graph.resolve(ent.uri)
                    # Derive a repository-relative path from the artifact URI.
                    _pid, _kind, resource, _span = parse_program_uri(
                        anchor.artifact_uri
                    )
                    path = resource
                except Exception:
                    # If resolution fails, still record the entity without path info.
                    anchor = None
                    path = ""

                start_line = getattr(span, "start_line", None)
                end_line = getattr(span, "end_line", None)
                if anchor is not None:
                    if start_line is None:
                        start_line = getattr(anchor.span, "start_line", None)
                    if end_line is None:
                        end_line = getattr(anchor.span, "end_line", None)

                matches.append(
                    {
                        "token": tok,
                        "entity_name": getattr(ent, "name", ""),
                        "entity_kind": getattr(ent, "kind", ""),
                        "path": path,
                        "start_line": int(start_line)
                        if start_line is not None
                        else None,
                        "end_line": int(end_line) if end_line is not None else None,
                    }
                )
                if len(matches) >= 20:
                    break
            if len(matches) >= 20:
                break

        # 2) If we still have no matches, fall back to fuzzy entity-name matching
        #    to handle small naming variations like pluralization.
        if not matches and tokens:
            for tok in tokens:
                base_tok = tok.rstrip("s")
                if not base_tok:
                    continue
                for ent in entities:
                    name = getattr(ent, "name", "").lower()
                    if not name:
                        continue
                    if not (
                        name == base_tok
                        or name.startswith(base_tok)
                        or base_tok in name
                    ):
                        continue

                    try:
                        anchor = graph.resolve(ent.uri)
                        _pid, _kind, resource, _span = parse_program_uri(
                            anchor.artifact_uri
                        )
                        path = resource
                    except Exception:
                        anchor = None
                        path = ""

                    start_line = getattr(anchor.span, "start_line", None) if anchor else None
                    end_line = getattr(anchor.span, "end_line", None) if anchor else None

                    matches.append(
                        {
                            "token": tok,
                            "entity_name": getattr(ent, "name", ""),
                            "entity_kind": getattr(ent, "kind", ""),
                            "path": path,
                            "start_line": int(start_line)
                            if start_line is not None
                            else None,
                            "end_line": int(end_line) if end_line is not None else None,
                        }
                    )
                    if len(matches) >= 10:
                        break
                if len(matches) >= 10:
                    break

    # Build a compact, deterministic context summary derived from the graph,
    # augmented with short code snippets so that answers can be grounded in
    # the *actual source*, not just file paths.
    if not matches:
        context_summary = (
            f"I searched the program graph for repository '{repo_id}' but did not "
            f"find any obvious matches for the question."
        )
    else:
        lines: List[str] = []
        lines.append(f"Repository: {repo_id}")
        lines.append("Question:")
        lines.append(question)
        lines.append("")
        lines.append("Relevant code locations (based on simple token search):")

        # Limit how many locations we expand with snippets to keep the prompt
        # compact while still giving the model concrete grounding.
        MAX_SNIPPET_LOCS = 5

        for idx, m in enumerate(matches, 1):
            loc = m["path"] or "<unknown path>"
            if m["start_line"] is not None and m["end_line"] is not None:
                loc = f"{loc} (L{m['start_line']}-L{m['end_line']})"
            lines.append(
                f"{idx}. [{m['entity_kind']}] {m['entity_name']} — {loc} "
                f"(matched on '{m['token']}')"
            )
            # Attach a small snippet for the first few, so questions like
            # "what's in this dictionary or function body?" can be answered
            # from exact code instead of guesswork.
            if idx <= MAX_SNIPPET_LOCS and m.get("path"):
                snippet = _load_snippet_for_match(
                    m["path"],
                    m.get("start_line"),
                    m.get("end_line"),
                )
                if snippet:
                    lines.append("   code snippet:")
                    for ln in snippet.splitlines():
                        # Indent snippet lines for readability.
                        lines.append("     " + ln)
        context_summary = "\n".join(lines)

    # Use the adapter-specified QA model to answer, falling back to the
    # context summary on error. The prompt is designed for a *user-facing*
    # answer: short, clear, and without echoing the full prompt or dumping
    # large code blocks.
    prompt = (
        "You are a senior software engineer helping a user understand a codebase.\n"
        "Answer their question in a friendly, concise way suitable for a chat UI.\n"
        "- Do NOT repeat the full question or any system instructions.\n"
        "- Do NOT restate the repository ID or headings.\n"
        "- Do NOT include large code blocks unless the user explicitly asks for code.\n"
        "- When the question asks about the contents of a specific variable, dict, or\n"
        "  function (e.g., \"what's in this dictionary?\"), base your answer\n"
        "  *strictly* on the provided code snippets and describe the concrete\n"
        "  keys/values or arguments rather than speculating.\n"
        "- Prefer 1–3 short paragraphs or a few bullet points.\n"
        "- If the context seems insufficient, say so briefly.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Static analysis context from the repository (for your reference):\n"
        f"{context_summary}\n\n"
        "Now write only the final answer you would show to the user:"
    )

    # Extract just the final user-facing answer in case the model echoes
    # the entire prompt (system instructions + context).
    def _extract_final_answer(raw: str) -> str:
        text = (raw or "").strip()
        marker = "Now write only the final answer you would show to the user:"
        idx = text.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker) :].strip()
        return text if text else context_summary

    # Primary path: adapter-driven QA runtime.
    if qa_meta is not None:
        try:
            qa_cfg: QAModelConfig = get_model_config_from_adapter(qa_meta)
            model, tokenizer = get_or_load_model(qa_cfg)
            answer = run_qa_generation(qa_cfg, model, tokenizer, prompt)
            return _extract_final_answer(answer)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("QA adapter runtime failed; falling back to base LLM: %s", exc)

    # Fallback: legacy global base LLM.
    try:
        answer = _llm_generate_answer(prompt)
        return _extract_final_answer(answer)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return context_summary + f"\n\n[LLM error: {exc}]"


# High-level QA swarm components: router, adapter manager, and retriever.
_qa_semantic_router = SemanticRouter()
_qa_adapter_manager = SkillAdapterManager(repo_library=repo_lib)
_qa_retriever_agent = RetrieverAgent(retrieve_fn=_format_qa_answer_stub)
_qa_swarm_controller = QASwarmController(
    router=_qa_semantic_router,
    adapter_manager=_qa_adapter_manager,
    retriever=_qa_retriever_agent,
)


def _execute_skill_chat(
    *,
    skill: str,
    question: str,
    repo_hint: str,
    qa_mode: Optional[str],
) -> Dict[str, Any]:
    """
    Internal helper to execute a per-repo, per-skill interaction.

    For the "qa" skill, this will:
    - Plan a single-repo QA query via `RepoLibrary.query`.
    - Validate that a repo-local QA adapter is registered (i.e., the skill
      is built and present in the adapter registry).
    - Execute a lightweight, non-LLM QA routine over the repository's
      program graph via `_format_qa_answer_stub`.

    Other skills still return a clearly-marked stub response documenting
    where additional runtimes should be wired in.
    """
    skill_norm = skill.strip()
    if not skill_norm:
        raise HTTPException(status_code=400, detail="`skill` is required.")

    if skill_norm == "qa":
        try:
            result = _qa_swarm_controller.run_qa(
                question=question,
                repo_hint=repo_hint,
                qa_mode=qa_mode,
            )
            return result
        except ValueError as exc:
            # Semantic/plan/adapter errors from the swarm layer are exposed
            # to clients as 400s with a clear message.
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # Default stub for non-QA skills.
    return {
        "type": "skill_chat_result",
        "skill": skill_norm,
        "plan": None,
        "answer": (
            f"This is a stub answer for skill={skill_norm!r}. "
            "Planning/execution for this skill has not been implemented yet. "
            "Wire your runtime into `_execute_skill_chat` to enable it."
        ),
    }


@app.get("/api/paper-universe")
async def api_paper_universe_assets() -> Dict[str, Any]:
    """
    Describe the locally generated paper-universe assets exposed by the server.

    The assets themselves are served under `/paper-universe/...` so the HTML
    viewer can load its sibling JSON files directly.
    """
    return _paper_universe_assets_payload()


@app.get("/api/paper-universe/neighborhood/{paper_id}")
async def api_paper_universe_neighborhood(
    paper_id: str,
    limit: int = 12,
) -> Dict[str, Any]:
    """
    Return a small local graph centered on a selected paper using the
    precomputed paper KNN edges from the paper universe.
    """
    try:
        return _paper_universe_neighborhood_payload(
            paper_id,
            neighbor_limit=limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"failed to load paper neighborhood: {exc}",
        )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """
    Simple HTML UI shell.
    """
    html_path = Path(__file__).with_name("ui.html")
    if html_path.is_file():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    # Fallback inline UI if `ui.html` is not present.
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Research Library</title>
  <style>
    * { box-sizing: border-box; }
    html, body { max-width: 100%; overflow-x: hidden; }
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e5e7eb; }
    header { padding: 1rem 2rem; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
    header h1 { margin: 0; font-size: 1.2rem; }
    main { display: grid; grid-template-columns: 320px minmax(0, 1fr); height: calc(100vh - 64px); min-width: 0; overflow: hidden; }
    aside { border-right: 1px solid #1f2937; overflow-y: auto; overflow-x: hidden; padding: 1rem; min-width: 0; }
    section { padding: 1rem 1.5rem; overflow-y: auto; overflow-x: hidden; min-width: 0; max-width: 100%; }
    .repo-item { padding: 0.35rem 0.5rem; border-radius: 0.25rem; cursor: pointer; margin-bottom: 0.15rem; }
    .repo-item:hover { background: #111827; }
    .repo-item.active { background: #1e293b; }
    .repo-id { font-size: 0.85rem; font-weight: 600; }
    .repo-meta { font-size: 0.75rem; color: #9ca3af; }
    .pill { display: inline-flex; align-items: center; padding: 0.1rem 0.45rem; border-radius: 999px; font-size: 0.7rem; background: #1e293b; color: #9ca3af; margin-left: 0.25rem; }
    label { display: block; font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 0.1rem; color: #9ca3af; }
    input[type="text"], textarea, select { width: 100%; background: #020617; border: 1px solid #1f2937; border-radius: 0.25rem; color: #e5e7eb; padding: 0.35rem 0.5rem; font-size: 0.85rem; }
    textarea { min-height: 72px; resize: vertical; }
    button, .button-link { margin-top: 0.5rem; padding: 0.35rem 0.75rem; border-radius: 0.25rem; border: none; cursor: pointer; font-size: 0.8rem; background: #2563eb; color: #e5e7eb; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; box-sizing: border-box; }
    button.secondary, .button-link.secondary { background: #111827; border: 1px solid #1f2937; margin-left: 0.5rem; }
    pre { background: #020617; border-radius: 0.25rem; padding: 0.5rem 0.75rem; font-size: 0.78rem; overflow-x: auto; max-width: 100%; white-space: pre-wrap; overflow-wrap: anywhere; }
    .row { display: flex; gap: 0.75rem; }
    .row > div { flex: 1; min-width: 0; }
    #repo-main-section, #arxiv-panel-root, #algorithms-panel-root { min-width: 0; max-width: 100%; overflow-x: hidden; }
    #graph-container { height: 420px; background: radial-gradient(circle at top left, #020617 0, #020617 40%, #020814 100%); border-radius: 0.5rem; border: 1px solid #111827; box-shadow: 0 10px 25px rgba(0,0,0,0.5); min-width: 0; max-width: 100%; }
    #repo-universe-container { position: relative; height: 420px; background: radial-gradient(circle at top left, #03152a 0, #020617 42%, #030712 100%); border-radius: 0.65rem; border: 1px solid #1f2937; box-shadow: 0 16px 42px rgba(0,0,0,0.45); min-width: 0; max-width: 100%; overflow: hidden; }
    #repo-universe-container canvas { position: absolute !important; inset: 0 !important; width: 100% !important; height: 100% !important; max-width: 100% !important; max-height: 100% !important; }
    #paper-graph-container { height: 340px; background: radial-gradient(circle at top left, #020617 0, #020617 40%, #020814 100%); border-radius: 0.5rem; border: 1px solid #111827; box-shadow: 0 10px 25px rgba(0,0,0,0.35); margin-top: 0.5rem; min-width: 0; max-width: 100%; }
    #graph-meta { margin-top: 0.35rem; font-size: 0.75rem; color: #9ca3af; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #repo-universe-meta { margin-top: 0.35rem; font-size: 0.75rem; color: #9ca3af; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #paper-graph-meta { margin-top: 0.35rem; font-size: 0.75rem; color: #9ca3af; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #graph-controls { display:flex; align-items:center; gap:0.35rem; font-size:0.75rem; }
    #graph-controls select { background:#020617; border:1px solid #1f2937; border-radius:0.25rem; color:#e5e7eb; padding:0.2rem 0.4rem; font-size:0.75rem; }
    #repo-universe-controls { display:flex; align-items:center; gap:0.35rem; font-size:0.75rem; flex-wrap:wrap; }
    #repo-universe-controls select { width:auto; background:#020617; border:1px solid #1f2937; border-radius:0.25rem; color:#e5e7eb; padding:0.2rem 0.4rem; font-size:0.75rem; }
    #graph-filter { max-width: 220px; }
    #repo-universe-filter { max-width: 240px; }
    .faded { opacity: 0.15; transition opacity 0.15s ease-out; }
    .asset-card { border:1px solid #1f2937; border-radius:0.5rem; padding:0.75rem; background:#0b1220; min-width:0; max-width:100%; overflow:hidden; }
    .asset-card h3 { font-size:0.85rem; margin:0 0 0.4rem 0; color:#e5e7eb; }
    .asset-frame { display:block; width:100%; max-width:100%; height:720px; border:1px solid #1f2937; border-radius:0.5rem; background:#020617; }
    #paper-text-viewer { margin-top:0.5rem; border:1px solid #1f2937; border-radius:0.5rem; background:#020617; padding:1rem; min-height:240px; overflow:hidden; }
    .paper-text-page { max-width:860px; margin:0 auto; min-width:0; }
    .paper-text-page-meta { font-size:0.75rem; color:#9ca3af; margin-bottom:0.75rem; }
    .paper-text-block { font-family: Georgia, "Times New Roman", serif; font-size:1rem; line-height:1.75; color:#e5e7eb; margin:0 0 1rem 0; white-space:normal; overflow-wrap:anywhere; word-break:normal; hyphens:auto; }
    .paper-text-block.heading { font-size:1rem; font-weight:700; color:#f8fafc; margin-top:1.4rem; }
    .paper-text-sections { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem; }
    .paper-text-chip { border:1px solid #334155; background:#0b1220; color:#cbd5e1; border-radius:999px; padding:0.25rem 0.6rem; font-size:0.72rem; cursor:pointer; margin-top:0; }
    .paper-text-toolbar { display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; }
    .panel-card { border:1px solid #1f2937; border-radius:0.65rem; background:linear-gradient(180deg,#0b1220 0%,#070d19 100%); padding:0.9rem; min-width:0; max-width:100%; overflow:hidden; }
    .paper-detail-card { border:1px solid #1f2937; border-radius:0.65rem; background:linear-gradient(135deg,#101827 0%,#07111f 56%,#061018 100%); padding:1rem; min-height:160px; overflow:hidden; }
    .paper-empty { color:#9ca3af; font-size:0.85rem; line-height:1.5; }
    .paper-title { font-size:1.15rem; line-height:1.3; font-weight:720; color:#f8fafc; margin:0 0 0.35rem 0; overflow-wrap:anywhere; }
    .paper-subtitle { font-size:0.82rem; color:#94a3b8; margin-bottom:0.7rem; overflow-wrap:anywhere; }
    .paper-chips { display:flex; flex-wrap:wrap; gap:0.35rem; margin:0.55rem 0; }
    .paper-chip { display:inline-flex; align-items:center; border:1px solid #334155; background:#0f172a; color:#cbd5e1; border-radius:999px; padding:0.18rem 0.55rem; font-size:0.72rem; max-width:100%; overflow-wrap:anywhere; }
    .paper-chip.accent { border-color:#1d4ed8; color:#bfdbfe; background:#0b1f3f; }
    .paper-chip.local { border-color:#15803d; color:#bbf7d0; background:#052e16; }
    .paper-chip.remote { border-color:#475569; color:#cbd5e1; background:#111827; }
    .paper-abstract { font-family:Georgia, "Times New Roman", serif; color:#dbe4ef; line-height:1.65; font-size:0.96rem; margin-top:0.75rem; max-width:980px; overflow-wrap:anywhere; }
    .paper-meta-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:0.55rem; margin-top:0.8rem; }
    .paper-kv { border:1px solid #1f2937; border-radius:0.5rem; background:#020617; padding:0.55rem 0.65rem; min-width:0; }
    .paper-kv span { display:block; font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.18rem; }
    .paper-kv strong { display:block; font-size:0.82rem; color:#e5e7eb; font-weight:600; overflow-wrap:anywhere; }
    .paper-universe-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:0.55rem; margin-top:0.65rem; }
    .paper-universe-card { border:1px solid #1f2937; border-radius:0.55rem; background:#020617; padding:0.65rem; min-width:0; }
    .paper-universe-card span { display:block; font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.2rem; }
    .paper-universe-card strong { display:block; color:#f8fafc; font-size:1rem; overflow-wrap:anywhere; }
    .paper-universe-card small { display:block; color:#94a3b8; font-size:0.72rem; margin-top:0.2rem; overflow-wrap:anywhere; }
  </style>
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
</head>
<body>
  <header>
    <div style="display:flex;flex-direction:column;gap:0.2rem;">
      <h1>Research Library</h1>
      <span style="font-size:0.8rem;color:#9ca3af;">Repositories · Papers · Algorithms</span>
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;">
      <span style="color:#9ca3af;">Library</span>
      <select id="library-kind" style="width:auto;" onchange="onLibraryChange()">
        <option value="repositories" selected>Repositories</option>
        <option value="papers">Papers</option>
        <option value="algorithms">Algorithms</option>
      </select>
      <span style="color:#9ca3af;">Skills</span>
      <span style="color:#9ca3af;">Library-wide skills</span>
      <select id="global-skill" style="width:auto;">
        <option value="qa">qa</option>
        <option value="edit">edit</option>
        <option value="meta">meta</option>
        <option value="nav">nav</option>
        <option value="test">test</option>
        <option value="perf">perf</option>
        <option value="security">security</option>
        <option value="api">api</option>
        <option value="style">style</option>
      </select>
      <button class="secondary" onclick="buildAllSkills()">Build all</button>
    </div>
  </header>
  <main>
    <aside>
      <div style="margin-bottom:0.5rem; display:flex; justify-content:space-between; align-items:center;">
        <span id="library-list-label" style="font-size:0.8rem;color:#9ca3af;">Repositories</span>
        <button id="library-list-refresh" class="secondary" onclick="refreshActiveLibrary()">Refresh</button>
      </div>
      <div id="repo-list"></div>
    </aside>
    <section id="repo-main-section">
      <div id="repo-panel" style="margin-bottom:1rem;">
        <div id="repo-universe-panel" style="margin-bottom:1rem;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin:0 0 0.35rem 0;gap:0.75rem;flex-wrap:wrap;">
            <div>
              <h2 style="font-size:0.95rem;margin:0;">Repository Universe</h2>
              <div style="font-size:0.75rem;color:#6b7280;margin-top:0.15rem;">
                Library-wide map of repositories, languages, roots, QA readiness, and similar repositories. Click a repository node to load it.
              </div>
            </div>
            <div id="repo-universe-controls">
              <label for="repo-universe-layout" style="color:#9ca3af;margin:0;">Layout</label>
              <select id="repo-universe-layout" onchange="renderRepoUniverse(repoUniverseData)">
                <option value="all" selected>All</option>
                <option value="repos">Repos only</option>
                <option value="hubs">Hubs only</option>
              </select>
              <button class="secondary" type="button" onclick="resetRepoUniverseView()">Fit</button>
              <button id="repo-universe-clear-filter" class="secondary" type="button" onclick="clearRepoUniverseHubFilter()" style="display:none;">Clear hub filter</button>
              <button class="secondary" type="button" onclick="loadRepoUniverse(true)">Reload</button>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;gap:0.5rem;flex-wrap:wrap;">
            <div style="font-size:0.75rem;color:#6b7280;">
              Blue nodes are repositories; green nodes are languages; cyan nodes are library roots; amber nodes are ready skills.
            </div>
            <input id="repo-universe-filter" type="text" placeholder="Filter repositories…" oninput="filterRepoUniverseNodes()" />
          </div>
          <div id="repo-universe-container"></div>
          <div id="repo-universe-meta">Repository universe has not been loaded yet.</div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin:0 0 0.5rem 0;">
          <h2 style="font-size:0.95rem;margin:0;">Repo details</h2>
          <span id="repo-commit-count" style="font-size:0.8rem;color:#9ca3af;"></span>
        </div>
        <pre id="repo-details">{}</pre>
        <div style="margin-top:0.5rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
            <span style="font-size:0.8rem;color:#9ca3af;">Skills</span>
            <div>
              <button class="secondary" onclick="loadSkills()">Refresh skills</button>
              <button class="secondary" onclick="buildQaSkill()">Build QA</button>
            </div>
          </div>
          <pre id="skill-statuses">[]</pre>
        </div>
        <div id="graph-panel" style="margin-top:0.75rem;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;gap:0.75rem;">
            <div style="display:flex;flex-direction:column;gap:0.15rem;">
              <span style="font-size:0.8rem;color:#9ca3af;">Program graph</span>
              <span style="font-size:0.7rem;color:#6b7280;">
                <span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.5rem;">
                  <span style="width:8px;height:8px;border-radius:999px;background:#f59e0b;"></span><span>modules</span>
                </span>
                <span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.5rem;">
                  <span style="width:8px;height:8px;border-radius:999px;background:#10b981;"></span><span>classes</span>
                </span>
                <span style="display:inline-flex;align-items:center;gap:0.25rem;">
                  <span style="width:8px;height:8px;border-radius:999px;background:#6366f1;"></span><span>functions</span>
                </span>
              </span>
            </div>
            <div id="graph-controls">
              <label for="graph-layout" style="color:#9ca3af;">Layout</label>
              <select id="graph-layout">
                <option value="cose" selected>Force</option>
                <option value="concentric">Concentric</option>
                <option value="breadthfirst">Layered</option>
              </select>
              <button class="secondary" onclick="resetGraphView()">Fit</button>
              <button class="secondary" onclick="loadGraph()">Reload</button>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;gap:0.5rem;">
            <div style="font-size:0.75rem;color:#6b7280;">
              Pan with drag, zoom with scroll, click a node to highlight its neighborhood and show source.
            </div>
            <input id="graph-filter" type="text" placeholder="Filter nodes by name…" oninput="filterGraphNodes()" />
          </div>
          <div id="graph-container"></div>
          <div id="graph-meta"></div>
        </div>
        <div id="source-panel" style="margin-top:0.75rem; display:none;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;">
            <span style="font-size:0.8rem;color:#9ca3af;">Source view</span>
            <button class="secondary" onclick="toggleSourceCollapse()" id="source-toggle-btn">Hide</button>
          </div>
          <pre id="source-meta" style="font-size:0.75rem;color:#9ca3af;margin-bottom:0.25rem;"></pre>
          <pre id="source-content"></pre>
        </div>
      </div>
      <div class="row">
        <div>
          <h3 style="font-size:0.9rem;margin:0 0 0.35rem 0;">Interaction</h3>
          <label for="chat-input">Message</label>
          <textarea id="chat-input" placeholder="Ask a question or describe a task..."></textarea>
          <label for="interaction-mode">Mode</label>
          <select id="interaction-mode">
            <option value="repo_skill_chat">Chat (selected repo · per-skill)</option>
            <option value="comparative_qa">QA (library comparative)</option>
            <option value="coarse_retrieval">Coarse Retrieval (paper + repo lanes)</option>
            <option value="meta_skill">Meta Skill (library)</option>
          </select>
          <label for="interaction-skill">Skill (for selected repo)</label>
          <select id="interaction-skill">
            <!-- Options are populated dynamically based on built skills -->
          </select>
          <label for="qa-mode">QA sub-mode</label>
          <select id="qa-mode">
            <option value="">auto (let adapter decide)</option>
            <option value="docs">docs</option>
            <option value="symbol">symbol</option>
            <option value="code_region">code_region</option>
            <option value="usage">usage</option>
            <option value="change">change</option>
          </select>
          <label for="meta-num">Num tasks (for Meta Skill)</label>
          <input id="meta-num" type="text" value="100" />
          <button onclick="runInteraction()">Run</button>
          <pre id="interaction-output"></pre>
        </div>
      </div>
    </section>
    <section id="arxiv-panel-root" style="display:none; padding:1rem 1.5rem; overflow-y:auto;">
      <h2 style="font-size:0.95rem;margin:0 0 0.5rem 0;">Papers</h2>
      <p style="font-size:0.8rem;color:#9ca3af;margin:0 0 0.5rem 0;">
        Search the local paper metadata snapshot, browse results in the left sidebar, inspect paper details, and explore the paper-universe graphs from the same workspace.
      </p>
      <div style="display:flex;gap:0.75rem;align-items:flex-end;flex-wrap:wrap;">
        <div style="flex:1;min-width:220px;">
          <label for="arxiv-query">Search Papers</label>
          <input id="arxiv-query" type="text" placeholder="Search titles, abstracts, or authors…" />
        </div>
        <div style="flex:0.6;min-width:180px;">
          <label for="arxiv-category">Category prefix (optional)</label>
          <input id="arxiv-category" type="text" placeholder="e.g. cs.LG or cs." />
        </div>
        <div style="display:flex;gap:0.5rem;margin-top:0.35rem;">
          <button onclick="runArxivSearch()">Search papers</button>
          <button class="secondary" onclick="downloadAllArxivPdfs()">Download PDFs for results</button>
        </div>
      </div>
      <div id="arxiv-status" style="margin-top:0.4rem;font-size:0.75rem;color:#9ca3af;">
        No search run yet.
      </div>
      <div id="arxiv-page-controls" style="margin-top:0.25rem;font-size:0.75rem;color:#9ca3af;display:flex;gap:0.5rem;align-items:center;">
        <button class="secondary" type="button" onclick="prevArxivPage()">Prev</button>
        <span id="arxiv-page-meta">Page 1</span>
        <button class="secondary" type="button" onclick="nextArxivPage()">Next</button>
      </div>
      <div style="display:flex;gap:0.75rem;margin-top:0.75rem;">
        <div style="flex:1;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.75rem;flex-wrap:wrap;">
            <div>
              <h3 style="font-size:0.9rem;margin:0 0 0.4rem 0;">Interactive Paper Universe</h3>
              <p style="font-size:0.8rem;color:#9ca3af;margin:0;">
                Use the 3D viewer first. Double-clicking a paper there should select it in the library and load its details below.
              </p>
            </div>
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
              <button class="secondary" onclick="loadPaperUniverse(true)">Refresh assets</button>
              <a id="paper-universe-open-interactive" class="button-link secondary" href="#" target="_blank" rel="noopener noreferrer">Open interactive 3D</a>
            </div>
          </div>
          <div id="paper-universe-status" style="margin-top:0.5rem;font-size:0.8rem;color:#9ca3af;">
            Paper-universe assets have not been loaded yet.
          </div>
          <div id="paper-universe-meta" class="paper-universe-grid"></div>
          <div class="asset-card" style="margin-top:0.75rem;">
            <h3>Interactive 3D Viewer</h3>
            <iframe id="paper-universe-viewer-frame" class="asset-frame" src="" style="display:none;"></iframe>
          </div>
          <div style="margin-top:1rem;">
            <h3 style="font-size:0.9rem;margin:0 0 0.35rem 0;">Paper details</h3>
            <div style="font-size:0.75rem;color:#6b7280;margin-bottom:0.35rem;">
              Search papers, click the sidebar, or double-click a point in the 3D viewer. Selection loads details, full text, and the selected-paper graph together.
            </div>
            <div style="display:flex;gap:0.5rem;margin-bottom:0.35rem;align-items:center;flex-wrap:wrap;">
              <button class="secondary" onclick="downloadSingleArxivPdf()">Download selected PDF</button>
              <button id="arxiv-open-pdf" class="secondary" type="button" style="font-size:0.8rem;align-self:center;">
                Open PDF in viewer
              </button>
              <a id="arxiv-open-source-pdf" class="button-link secondary" href="#" target="_blank" rel="noopener noreferrer">
                Open source PDF
              </a>
            </div>
            <div id="arxiv-detail" class="paper-detail-card">
              <div class="paper-empty">Select a paper to load its library record.</div>
            </div>
            <div id="arxiv-pdf-viewer-container" style="margin-top:0.5rem;border:1px solid #1f2937;border-radius:0.25rem;overflow:hidden;display:none;height:360px;">
              <iframe id="arxiv-pdf-viewer" src="" style="width:100%;height:100%;border:0;background:#030712;"></iframe>
            </div>
            <div style="margin-top:1rem;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.75rem;flex-wrap:wrap;">
                <div>
                  <h3 style="font-size:0.9rem;margin:0 0 0.35rem 0;">Paper Text</h3>
                  <div style="font-size:0.75rem;color:#6b7280;">
                    Full text comes from the merged 1M-paper dataset. Pages are exact when a local PDF is available and inferred otherwise.
                  </div>
                </div>
                <div class="paper-text-toolbar">
                  <button class="secondary" type="button" onclick="prevPaperTextPage()">Prev page</button>
                  <span id="paper-text-page-meta" style="font-size:0.75rem;color:#9ca3af;">No paper text loaded.</span>
                  <button class="secondary" type="button" onclick="nextPaperTextPage()">Next page</button>
                  <label for="paper-text-view-mode" style="margin:0;color:#9ca3af;">View</label>
                  <select id="paper-text-view-mode" style="width:auto;" onchange="renderPaperTextPage()">
                    <option value="formatted" selected>Formatted</option>
                    <option value="raw">Raw</option>
                  </select>
                </div>
              </div>
              <div id="paper-text-status" style="margin-top:0.5rem;font-size:0.75rem;color:#9ca3af;">
                Select a paper to load full text.
              </div>
              <div id="paper-text-sections" class="paper-text-sections"></div>
              <div id="paper-text-viewer"></div>
            </div>
            <div style="margin-top:1rem;">
              <h3 style="font-size:0.9rem;margin:0 0 0.35rem 0;">Selected Paper Graph</h3>
              <div style="font-size:0.75rem;color:#6b7280;margin-bottom:0.35rem;">
                This graph breaks the selected paper into papers, categories, topics, year, and text sections. Click paper nodes to load them; click section nodes to jump the text viewer.
              </div>
              <div style="font-size:0.7rem;color:#9ca3af;display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.35rem;">
                <span>paper</span><span>category</span><span>topic</span><span>year</span><span>section</span>
              </div>
              <div id="paper-graph-status" style="font-size:0.75rem;color:#9ca3af;">
                Select a paper to load its graph.
              </div>
              <div id="paper-graph-container"></div>
              <div id="paper-graph-meta"></div>
            </div>
          </div>
        </div>
      </div>
    </section>
    <div id="algorithms-panel-root" style="display:none; padding:1rem 1.5rem; grid-column:1 / span 2;">
      <h2 style="font-size:0.95rem;margin:0 0 0.5rem 0;">Algorithms Library</h2>
      <p style="font-size:0.8rem;color:#9ca3af;margin:0 0 0.5rem 0;">
        Browse and search canonical algorithms imported from your local repositories.
      </p>
      <label for="algorithms-query">Filter by text (optional)</label>
      <input id="algorithms-query" type="text" placeholder="Filter by id, name, or notes..." oninput="runAlgorithmsSearch()" />
      <label for="algorithms-topic">Topic/tag (optional)</label>
      <input id="algorithms-topic" type="text" placeholder="e.g. graphs, dp, shortest_path" oninput="runAlgorithmsSearch()" />
      <div style="display:flex;gap:0.75rem;margin-top:0.75rem;">
        <div style="flex:1;max-height:420px;overflow-y:auto;border:1px solid #1f2937;border-radius:0.25rem;">
          <ul id="algorithms-list" style="list-style:none;margin:0;padding:0;"></ul>
        </div>
        <div style="flex:1;">
          <h3 style="font-size:0.9rem;margin:0 0 0.35rem 0;">Algorithm details</h3>
          <div id="algorithms-detail"></div>
        </div>
      </div>
    </div>
  </main>
  <script>
    let repos = [];
    let activeRepo = null;
    let cy = null;
    let repoUniverseDeck = null;
    let repoUniverseViewState = null;
    let repoUniverseSelectedId = null;
    let repoUniverseHubFilter = null;
    let repoUniverseLoaded = false;
    let repoUniverseData = null;
    let sourceCollapsed = false;
    // Cache per-repo skill statuses so we can restrict interactions
    // to skills that are actually built (e.g., QA only when the per-repo
    // QA skill is up_to_date).
    let skillsByRepo = {};
    // Cached Arxiv search results for the current query.
    let arxivResults = [];
    // Client-side pagination state for Arxiv results.
    let arxivPage = 1;
    const ARXIV_PAGE_SIZE = 50;
    // Cached Algorithms list loaded from /api/algorithms for browsing/filtering.
    let algorithmsCache = [];
    let algorithmsLoaded = false;
    // Currently selected Arxiv paper id.
    let activeArxivPaperId = null;
    let activePaperRecord = null;
    let paperUniverseLoaded = false;
    let paperUniverseMeta = null;
    let paperNeighborhoodCy = null;
    let activePaperText = null;
    let activePaperTextPage = 1;

    function getActiveLibraryKind() {
      const sel = document.getElementById('library-kind');
      return sel ? (sel.value || 'repositories') : 'repositories';
    }

    function normalizePaperId(raw) {
      const text = String(raw || '').trim().split('/').pop();
      if (!text) return '';
      return text.replace(/v\\d+$/i, '');
    }

    function escapeHtml(raw) {
      return String(raw || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function formatCount(raw) {
      const n = Number(raw || 0);
      if (!Number.isFinite(n) || n <= 0) return '0';
      return Math.round(n).toLocaleString();
    }

    function compactText(raw, maxLen = 160) {
      const text = String(raw || '').replace(/\\s+/g, ' ').trim();
      if (!text || text.length <= maxLen) return text;
      return text.slice(0, Math.max(0, maxLen - 1)).trimEnd() + '…';
    }

    function splitTokens(raw) {
      if (Array.isArray(raw)) {
        return raw.map(x => String(x || '').trim()).filter(Boolean);
      }
      return String(raw || '')
        .split(/[,;]\\s*|\\s{2,}/)
        .map(x => x.trim())
        .filter(Boolean);
    }

    function paperCategoryList(paper) {
      const cats = splitTokens(paper && paper.categories);
      const primary = String((paper && paper.primary_category) || '').trim();
      if (primary && !cats.includes(primary)) cats.unshift(primary);
      return cats;
    }

    function renderPaperChips(tokens, className = '') {
      return (tokens || []).filter(Boolean).map(token => (
        '<span class="paper-chip ' + className + '">' + escapeHtml(token) + '</span>'
      )).join('');
    }

    function clearSelectedPaperDetail(message) {
      activePaperRecord = null;
      activeArxivPaperId = null;
      const detailEl = document.getElementById('arxiv-detail');
      const pdfButton = document.getElementById('arxiv-open-pdf');
      const pdfViewer = document.getElementById('arxiv-pdf-viewer');
      const pdfViewerContainer = document.getElementById('arxiv-pdf-viewer-container');
      if (detailEl) {
        detailEl.innerHTML = '<div class="paper-empty">' + escapeHtml(message || 'Select a paper to load its library record.') + '</div>';
      }
      if (pdfButton) {
        pdfButton.disabled = true;
        pdfButton.style.opacity = '0.4';
      }
      if (pdfViewer && pdfViewerContainer) {
        pdfViewer.src = '';
        pdfViewerContainer.style.display = 'none';
      }
      setSourcePdfLink('');
    }

    function renderSelectedPaperDetail(paper) {
      const detailEl = document.getElementById('arxiv-detail');
      if (!detailEl || !paper) return;
      const pid = paper.id || paper.paper_id || paper.canonical_paper_id || '';
      const canonical = normalizePaperId(pid);
      const title = paper.title || '(untitled paper)';
      const authors = compactText(paper.authors || '', 260);
      const categories = paperCategoryList(paper);
      const primary = paper.primary_category || categories[0] || '';
      const year = paper.year || '';
      const abstract = paper.abstract || '';
      const hasPdf = !!paper.has_pdf;
      const sourceUrl = pid ? ('https://arxiv.org/pdf/' + encodeURIComponent(pid) + '.pdf') : '';
      const version = paper.paper_version || (String(pid).match(/v\\d+$/i) || [''])[0];

      let html = '';
      html += '<div class="paper-title">' + escapeHtml(title) + '</div>';
      html += '<div class="paper-subtitle">';
      html += pid ? '<code>' + escapeHtml(pid) + '</code>' : 'No paper id';
      if (authors) html += ' · ' + escapeHtml(authors);
      html += '</div>';
      html += '<div class="paper-chips">';
      html += hasPdf
        ? '<span class="paper-chip local">Local PDF available</span>'
        : '<span class="paper-chip remote">Remote PDF only</span>';
      if (primary) html += '<span class="paper-chip accent">' + escapeHtml(primary) + '</span>';
      if (year) html += '<span class="paper-chip">' + escapeHtml(String(year)) + '</span>';
      if (version) html += '<span class="paper-chip">' + escapeHtml(String(version)) + '</span>';
      html += renderPaperChips(categories.filter(cat => cat !== primary).slice(0, 8));
      html += '</div>';
      html += '<div class="paper-meta-grid">';
      html += '<div class="paper-kv"><span>Canonical ID</span><strong>' + escapeHtml(canonical || pid || '-') + '</strong></div>';
      html += '<div class="paper-kv"><span>Primary Category</span><strong>' + escapeHtml(primary || '-') + '</strong></div>';
      html += '<div class="paper-kv"><span>Year</span><strong>' + escapeHtml(String(year || '-')) + '</strong></div>';
      html += '<div class="paper-kv"><span>Source</span><strong>' + (sourceUrl ? 'arxiv.org PDF' : '-') + '</strong></div>';
      html += '</div>';
      if (abstract) {
        html += '<div class="paper-abstract">' + escapeHtml(abstract) + '</div>';
      } else {
        html += '<div class="paper-empty" style="margin-top:0.75rem;">No abstract is available for this paper record.</div>';
      }
      detailEl.innerHTML = html;
    }

    function renderPaperSelectionPending(paper) {
      const detailEl = document.getElementById('arxiv-detail');
      const arxivStatusEl = document.getElementById('arxiv-status');
      const universeStatusEl = document.getElementById('paper-universe-status');
      const title = compactText((paper && paper.title) || '', 180) || '(untitled paper)';
      const pid = (paper && (paper.paper_id || paper.id || paper.canonical_paper_id)) || '';
      const category = (paper && (paper.primary_category || paper.categories)) || '';
      activeArxivPaperId = pid || activeArxivPaperId;
      activePaperRecord = paper || null;
      if (detailEl) {
        detailEl.innerHTML =
          '<div class="paper-title">' + escapeHtml(title) + '</div>' +
          '<div class="paper-subtitle">' +
            (pid ? '<code>' + escapeHtml(pid) + '</code>' : 'Paper selected from universe') +
            (category ? ' · ' + escapeHtml(category) : '') +
          '</div>' +
          '<div class="paper-chips">' +
            '<span class="paper-chip accent">Click received</span>' +
            '<span class="paper-chip">Loading library record</span>' +
            '<span class="paper-chip">Loading full text</span>' +
            '<span class="paper-chip">Loading graph</span>' +
          '</div>' +
          '<div class="paper-empty" style="margin-top:0.75rem;">The paper universe selection was received. Metadata, full text, and the selected-paper graph are loading.</div>';
      }
      if (arxivStatusEl) {
        arxivStatusEl.textContent = 'Paper selected from universe · loading ' + (pid || title) + '…';
      }
      if (universeStatusEl) {
        universeStatusEl.textContent = 'Paper click received · loading ' + (pid || title) + ' into the library…';
      }
      clearPaperText('Paper selected. Loading full text…');
      clearPaperNeighborhood('Paper selected. Loading selected-paper graph…');
      Array.from(document.getElementsByClassName('repo-item')).forEach(el => {
        const paperId = el.getAttribute('data-paper-id');
        if (paperId !== null && pid) {
          el.classList.toggle('active', normalizePaperId(paperId) === normalizePaperId(pid));
        }
      });
    }

    function upsertPaperResult(paper) {
      if (!paper) return null;
      const incomingId = normalizePaperId(paper.id || paper.paper_id || paper.canonical_paper_id);
      if (!incomingId) return null;
      const existingIdx = arxivResults.findIndex(item => {
        const itemId = normalizePaperId(item && (item.id || item.paper_id || item.canonical_paper_id));
        return itemId === incomingId;
      });
      const normalized = {
        id: paper.id || paper.paper_id || paper.canonical_paper_id || '',
        title: paper.title || '',
        abstract: paper.abstract || '',
        authors: paper.authors || '',
        categories: paper.categories || paper.primary_category || '',
        primary_category: paper.primary_category || '',
        year: paper.year || null,
        paper_version: paper.paper_version || '',
        has_pdf: !!paper.has_pdf,
      };
      if (existingIdx >= 0) {
        arxivResults[existingIdx] = { ...arxivResults[existingIdx], ...normalized };
        return arxivResults[existingIdx];
      }
      arxivResults.unshift(normalized);
      return normalized;
    }

    function updateSidebarChrome(kind) {
      const labelEl = document.getElementById('library-list-label');
      const refreshEl = document.getElementById('library-list-refresh');
      const listEl = document.getElementById('repo-list');
      if (!labelEl || !refreshEl || !listEl) return;
      if (kind === 'papers') {
        labelEl.textContent = 'Papers';
        refreshEl.textContent = 'Refresh results';
        if (!arxivResults.length) {
          listEl.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Search papers to populate the sidebar.</div>';
        }
      } else {
        labelEl.textContent = 'Repositories';
        refreshEl.textContent = 'Refresh';
      }
    }

    function setActiveLibrary(kind) {
      const aside = document.querySelector('main > aside');
      const repoSection = document.getElementById('repo-main-section');
      const arxivPanel = document.getElementById('arxiv-panel-root');
      const algoPanel = document.getElementById('algorithms-panel-root');
      if (!aside || !repoSection || !arxivPanel || !algoPanel) return;
      if (kind === 'papers') {
        aside.style.display = 'block';
        repoSection.style.display = 'none';
        arxivPanel.style.display = 'block';
        algoPanel.style.display = 'none';
        updateSidebarChrome('papers');
        if (!arxivResults.length) {
          clearSelectedPaperDetail('Search papers or double-click a point in the 3D universe to load a paper.');
          clearPaperText('Select a paper to load full text.');
          clearPaperNeighborhood('Select a paper to load its graph.');
        } else {
          renderArxivList(arxivResults);
        }
        loadPaperUniverse().catch(console.error);
        // Clear any stale details prompt when switching in.
      } else if (kind === 'algorithms') {
        aside.style.display = 'none';
        repoSection.style.display = 'none';
        arxivPanel.style.display = 'none';
        algoPanel.style.display = 'block';
        // Lazily load algorithms list on first entry.
        ensureAlgorithmsLoaded().catch(console.error);
      } else {
        // repositories
        aside.style.display = 'block';
        repoSection.style.display = 'block';
        arxivPanel.style.display = 'none';
        algoPanel.style.display = 'none';
        updateSidebarChrome('repositories');
        loadRepos().catch(console.error);
      }
    }

    function onLibraryChange() {
      setActiveLibrary(getActiveLibraryKind());
    }

    function refreshActiveLibrary() {
      const kind = getActiveLibraryKind();
      if (kind === 'papers') {
        const qEl = document.getElementById('arxiv-query');
        const query = qEl ? (qEl.value || '').trim() : '';
        if (query) {
          runArxivSearch().catch(console.error);
        } else {
          updateSidebarChrome('papers');
        }
      } else {
        loadRepos().catch(console.error);
      }
    }

    function setPaperUniverseLink(linkEl, url) {
      if (!linkEl) return;
      if (url) {
        linkEl.href = url;
        linkEl.style.pointerEvents = 'auto';
        linkEl.style.opacity = '1';
      } else {
        linkEl.href = '#';
        linkEl.style.pointerEvents = 'none';
        linkEl.style.opacity = '0.4';
      }
    }

    function renderPaperUniverseMeta(data) {
      const metaEl = document.getElementById('paper-universe-meta');
      if (!metaEl) return;
      if (!data || !data.available) {
        metaEl.innerHTML = '<div class="paper-universe-card"><span>Status</span><strong>Unavailable</strong><small>Build exports/_paper_universe to enable the viewer.</small></div>';
        return;
      }
      const progress = data.progress || {};
      const viewer = data.viewer_manifest || {};
      const render = data.render_manifest || {};
      const lods = Array.isArray(viewer.paper_levels) ? viewer.paper_levels : [];
      const allLod = lods.length ? lods[lods.length - 1] : null;
      const overview = render.overview || {};
      const elapsedHours = progress.elapsed_seconds ? (Number(progress.elapsed_seconds) / 3600).toFixed(1) + 'h' : '';
      const cards = [
        {
          label: 'Universe Papers',
          value: formatCount(progress.processed_papers || progress.total_rows || (allLod && allLod.rows)),
          detail: progress.status ? 'status: ' + progress.status : 'ready',
        },
        {
          label: 'Interactive LOD',
          value: allLod ? String(allLod.label || formatCount(allLod.rows)) : 'Not built',
          detail: lods.length ? lods.map(x => x.label || formatCount(x.rows)).join(' · ') : 'no levels found',
        },
        {
          label: 'Categories',
          value: formatCount(viewer.category_rows || progress.final_category_nodes_written || overview.categories_represented),
          detail: (viewer.year_rows || overview.years_represented) ? formatCount(viewer.year_rows || overview.years_represented) + ' years represented' : '',
        },
        {
          label: 'KNN Edges',
          value: formatCount(progress.knn_edge_count),
          detail: progress.paper_knn ? 'k=' + progress.paper_knn + (elapsedHours ? ' · build ' + elapsedHours : '') : elapsedHours,
        },
      ];
      metaEl.innerHTML = cards.map(card => (
        '<div class="paper-universe-card">' +
          '<span>' + escapeHtml(card.label) + '</span>' +
          '<strong>' + escapeHtml(card.value || '-') + '</strong>' +
          '<small>' + escapeHtml(card.detail || '') + '</small>' +
        '</div>'
      )).join('');
    }

    async function loadPaperUniverse(force = false) {
      if (paperUniverseLoaded && paperUniverseMeta && !force) {
        return;
      }
      const statusEl = document.getElementById('paper-universe-status');
      const metaEl = document.getElementById('paper-universe-meta');
      const frameEl = document.getElementById('paper-universe-viewer-frame');
      const interactiveLink = document.getElementById('paper-universe-open-interactive');
      if (statusEl) statusEl.textContent = 'Loading paper-universe assets…';
      try {
        const res = await fetch('/api/paper-universe');
        const data = await res.json();
        paperUniverseMeta = data;
        paperUniverseLoaded = true;

        const cacheBust = force ? ('?ts=' + Date.now()) : '';
        const interactiveUrl = data.interactive_viewer_url ? (data.interactive_viewer_url + cacheBust) : null;

        if (frameEl) {
          if (interactiveUrl) {
            if (force || frameEl.src !== interactiveUrl) {
              frameEl.src = interactiveUrl;
            }
            frameEl.style.display = 'block';
          } else {
            frameEl.src = '';
            frameEl.style.display = 'none';
          }
        }

        setPaperUniverseLink(interactiveLink, data.interactive_viewer_url);

        const summaryParts = [];
        if (data.progress && typeof data.progress.processed_papers === 'number') {
          summaryParts.push('papers: ' + data.progress.processed_papers.toLocaleString());
        }
        if (data.viewer_manifest && Array.isArray(data.viewer_manifest.paper_levels) && data.viewer_manifest.paper_levels.length) {
          summaryParts.push('interactive LODs: ' + data.viewer_manifest.paper_levels.map(x => x.label || x.rows).join(', '));
        }
        const missing = [];
        if (!data.interactive_viewer_url) missing.push('interactive viewer');
        if (statusEl) {
          if (!data.available) {
            statusEl.textContent = 'Paper-universe export directory was not found under exports/_paper_universe.';
          } else if (missing.length) {
            statusEl.textContent = 'Paper-universe assets are partially available. Missing: ' + missing.join(', ') + '.';
          } else {
            statusEl.textContent = 'Paper-universe assets are available. ' + (summaryParts.join(' · ') || 'Interactive 3D viewer is ready.');
          }
        }
        renderPaperUniverseMeta(data);
      } catch (err) {
        console.error('paper universe load error', err);
        if (statusEl) statusEl.textContent = 'Failed to load paper-universe assets.';
        if (metaEl) metaEl.innerHTML = '<div class="paper-universe-card"><span>Status</span><strong>Load failed</strong><small>See browser console for details.</small></div>';
      }
    }

    window.loadPaperUniverse = loadPaperUniverse;

    function clearPaperText(message) {
      activePaperText = null;
      activePaperTextPage = 1;
      const statusEl = document.getElementById('paper-text-status');
      const metaEl = document.getElementById('paper-text-page-meta');
      const viewerEl = document.getElementById('paper-text-viewer');
      const sectionsEl = document.getElementById('paper-text-sections');
      if (statusEl) {
        statusEl.textContent = message || 'Select a paper to load full text.';
      }
      if (metaEl) {
        metaEl.textContent = 'No paper text loaded.';
      }
      if (viewerEl) {
        viewerEl.innerHTML = '';
      }
      if (sectionsEl) {
        sectionsEl.innerHTML = '';
      }
    }

    function setSourcePdfLink(url) {
      const sourceLink = document.getElementById('arxiv-open-source-pdf');
      if (!sourceLink) return;
      if (url) {
        sourceLink.href = url;
        sourceLink.style.opacity = '1';
        sourceLink.style.pointerEvents = 'auto';
      } else {
        sourceLink.href = '#';
        sourceLink.style.opacity = '0.4';
        sourceLink.style.pointerEvents = 'none';
      }
    }

    function renderPaperTextSections() {
      const sectionsEl = document.getElementById('paper-text-sections');
      if (!sectionsEl) return;
      sectionsEl.innerHTML = '';
      if (!activePaperText || !Array.isArray(activePaperText.sections) || !activePaperText.sections.length) {
        return;
      }
      activePaperText.sections.slice(0, 24).forEach((section) => {
        const btn = document.createElement('button');
        btn.className = 'paper-text-chip';
        btn.type = 'button';
        btn.textContent = (section.title || 'Section') + ' · p.' + (section.page || 1);
        btn.onclick = () => {
          activePaperTextPage = Math.max(1, Number(section.page || 1));
          renderPaperTextPage();
        };
        sectionsEl.appendChild(btn);
      });
    }

    function renderPaperTextPage() {
      const statusEl = document.getElementById('paper-text-status');
      const metaEl = document.getElementById('paper-text-page-meta');
      const viewerEl = document.getElementById('paper-text-viewer');
      const modeEl = document.getElementById('paper-text-view-mode');
      if (!statusEl || !metaEl || !viewerEl || !modeEl) return;
      if (!activePaperText || !Array.isArray(activePaperText.pages) || !activePaperText.pages.length) {
        clearPaperText('Select a paper to load full text.');
        return;
      }

      const pages = activePaperText.pages;
      if (activePaperTextPage < 1) activePaperTextPage = 1;
      if (activePaperTextPage > pages.length) activePaperTextPage = pages.length;
      const page = pages[activePaperTextPage - 1];
      const mode = modeEl.value || 'formatted';
      const labels = {
        exact_pdf: 'exact pages from PDF',
        inferred: 'inferred pages from full text',
        single_page: 'single-page text view',
      };
      statusEl.textContent =
        'Paper text ready · ' +
        (labels[activePaperText.page_mode] || activePaperText.page_mode || 'full text') +
        ' · source: ' + (activePaperText.text_source || 'unknown') +
        (activePaperText.text_is_partial ? ' · partial text' : '') +
        (activePaperText.has_local_pdf ? ' · local PDF available' : '');
      metaEl.textContent =
        'Page ' + activePaperTextPage + ' of ' + pages.length +
        ' · ' + (page.line_count || 0) + ' lines' +
        ' · ' + (page.char_count || 0).toLocaleString() + ' chars';

      if (mode === 'raw') {
        viewerEl.innerHTML =
          '<div class=\"paper-text-page\">' +
            '<div class=\"paper-text-page-meta\">Raw text view</div>' +
            '<pre class=\"paper-text-block\" style=\"margin:0;white-space:pre-wrap;\">' + escapeHtml(page.text || '') + '</pre>' +
          '</div>';
        return;
      }

      const blockHtml = (page.blocks || []).map((block) => {
        const kind = block.kind === 'heading' ? 'heading' : 'paragraph';
        return '<div class=\"paper-text-block ' + kind + '\">' + escapeHtml(block.text || '') + '</div>';
      }).join('');
      viewerEl.innerHTML =
        '<div class=\"paper-text-page\">' +
          '<div class=\"paper-text-page-meta\">Formatted text view</div>' +
          blockHtml +
        '</div>';
    }

    function nextPaperTextPage() {
      if (!activePaperText || !Array.isArray(activePaperText.pages) || !activePaperText.pages.length) return;
      if (activePaperTextPage < activePaperText.pages.length) {
        activePaperTextPage += 1;
        renderPaperTextPage();
      }
    }

    function prevPaperTextPage() {
      if (!activePaperText || !Array.isArray(activePaperText.pages) || !activePaperText.pages.length) return;
      if (activePaperTextPage > 1) {
        activePaperTextPage -= 1;
        renderPaperTextPage();
      }
    }

    async function loadPaperText(paperId) {
      if (!paperId) {
        clearPaperText('Select a paper to load full text.');
        return;
      }
      const requestedId = normalizePaperId(paperId);
      const statusEl = document.getElementById('paper-text-status');
      if (statusEl) {
        statusEl.textContent = 'Loading full text from the merged paper dataset…';
      }
      try {
        const res = await fetch('/api/paper-text/' + encodeURIComponent(paperId));
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || ('paper text failed: ' + res.status));
        }
        if (requestedId && requestedId !== normalizePaperId(activeArxivPaperId)) {
          return;
        }
        activePaperText = data;
        activePaperTextPage = 1;
        setSourcePdfLink(data.source_pdf_url || '');
        renderPaperTextSections();
        renderPaperTextPage();
      } catch (err) {
        if (requestedId && requestedId !== normalizePaperId(activeArxivPaperId)) {
          return;
        }
        console.error('paper text load error', err);
        clearPaperText('Full text is not available for this paper in the local merged dataset.');
      }
    }

    window.nextPaperTextPage = nextPaperTextPage;
    window.prevPaperTextPage = prevPaperTextPage;
    window.renderPaperTextPage = renderPaperTextPage;

    function clearPaperNeighborhood(message) {
      const statusEl = document.getElementById('paper-graph-status');
      const metaEl = document.getElementById('paper-graph-meta');
      const container = document.getElementById('paper-graph-container');
      if (paperNeighborhoodCy) {
        paperNeighborhoodCy.destroy();
        paperNeighborhoodCy = null;
      }
      if (container) {
        container.innerHTML = '';
      }
      if (statusEl) {
        statusEl.textContent = message || 'Select a paper to load its graph.';
      }
      if (metaEl) {
        metaEl.textContent = '';
      }
    }

    function renderPaperNeighborhoodGraph(data) {
      const container = document.getElementById('paper-graph-container');
      const statusEl = document.getElementById('paper-graph-status');
      const metaEl = document.getElementById('paper-graph-meta');
      if (!container || !statusEl || !metaEl) return;

      const nodes = Array.isArray(data && data.nodes) ? data.nodes : [];
      const edges = Array.isArray(data && data.edges) ? data.edges : [];
      if (!nodes.length) {
        clearPaperNeighborhood('No selected-paper graph is available for this selection.');
        return;
      }

      const elements = [];
      nodes.forEach((node) => {
        elements.push({
          data: {
            id: node.id,
            label: node.title || node.label || node.id,
            paper_id: node.paper_id || node.canonical_paper_id || node.id,
            canonical_paper_id: node.canonical_paper_id || '',
            title: node.title || node.label || node.id,
            authors: node.authors || '',
            primary_category: node.primary_category || '',
            category_id: node.category_id || '',
            topic_id: node.topic_id || '',
            paper_count: typeof node.paper_count === 'number' ? node.paper_count : 0,
            year: node.year || '',
            role: node.role || 'neighbor',
            kind: node.kind || 'paper',
            page: typeof node.page === 'number' ? node.page : 0,
            has_pdf: node.has_pdf ? 1 : 0,
            similarity: typeof node.similarity === 'number' ? node.similarity : 0,
          }
        });
      });
      edges.forEach((edge, idx) => {
        elements.push({
          data: {
            id: edge.source + '->' + edge.target + ':' + idx,
            source: edge.source,
            target: edge.target,
            weight: typeof edge.weight === 'number' ? edge.weight : 0,
            type: edge.type || '',
          }
        });
      });

      if (paperNeighborhoodCy) {
        paperNeighborhoodCy.destroy();
      }
      paperNeighborhoodCy = cytoscape({
        container,
        elements,
        style: [
          {
            selector: 'node',
            style: {
              'label': 'data(label)',
              'background-color': '#60a5fa',
              'color': '#e5e7eb',
              'text-wrap': 'wrap',
              'text-max-width': 132,
              'font-size': 10,
              'text-valign': 'center',
              'text-halign': 'center',
              'width': 28,
              'height': 28,
              'border-width': 1,
              'border-color': '#0f172a',
              'text-outline-color': '#020617',
              'text-outline-width': 2,
            }
          },
          {
            selector: 'node[role = "selected"]',
            style: {
              'background-color': '#f59e0b',
              'color': '#111827',
              'width': 44,
              'height': 44,
              'font-size': 11,
              'font-weight': 700,
            }
          },
          {
            selector: 'node[kind = "category"]',
            style: {
              'background-color': '#10b981',
              'shape': 'round-rectangle',
              'width': 38,
              'height': 24,
            }
          },
          {
            selector: 'node[kind = "topic"]',
            style: {
              'background-color': '#a855f7',
              'shape': 'round-rectangle',
              'width': 34,
              'height': 22,
            }
          },
          {
            selector: 'node[kind = "year"]',
            style: {
              'background-color': '#06b6d4',
              'shape': 'diamond',
              'width': 32,
              'height': 32,
            }
          },
          {
            selector: 'node[kind = "section"]',
            style: {
              'background-color': '#f97316',
              'shape': 'tag',
              'width': 34,
              'height': 22,
              'font-size': 9,
            }
          },
          {
            selector: 'node[has_pdf > 0]',
            style: {
              'border-color': '#22c55e',
              'border-width': 2,
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 'mapData(weight, 0, 1, 1.5, 5)',
              'line-color': '#475569',
              'target-arrow-color': '#475569',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'opacity': 0.8,
            }
          },
          {
            selector: 'edge[type = "has_section"]',
            style: { 'line-color': '#f97316', 'target-arrow-color': '#f97316', 'line-style': 'dashed', 'width': 1.5 }
          },
          {
            selector: 'edge[type = "has_topic"]',
            style: { 'line-color': '#a855f7', 'target-arrow-color': '#a855f7', 'line-style': 'dotted', 'width': 1.5 }
          },
          {
            selector: 'edge[type = "has_category"], edge[type = "shares_category"]',
            style: { 'line-color': '#10b981', 'target-arrow-color': '#10b981', 'width': 1.5 }
          },
          {
            selector: 'edge[type = "published_in"]',
            style: { 'line-color': '#06b6d4', 'target-arrow-color': '#06b6d4', 'width': 1.5 }
          }
        ],
        layout: {
          name: 'cose',
          animate: false,
          fit: true,
          padding: 40,
          idealEdgeLength: 100,
          nodeRepulsion: 6500,
          edgeElasticity: 80,
        },
        wheelSensitivity: 0.2,
      });

      const center = nodes.find(node => node.role === 'selected') || nodes[0];
      statusEl.textContent =
        'Selected paper graph: ' +
        (data.category_count || 0) + ' categories · ' +
        (data.topic_count || 0) + ' topics · ' +
        (data.section_count || 0) + ' sections · ' +
        (data.neighbor_count || 0) + ' similar papers.';
      metaEl.textContent =
        (center && center.primary_category ? center.primary_category : 'unknown category') +
        (center && center.year ? ' · ' + center.year : '') +
        (center && center.has_pdf ? ' · Local PDF' : '');

      paperNeighborhoodCy.on('tap', 'node', (evt) => {
        const node = evt.target.data();
        const kind = node.kind || 'paper';
        if (kind === 'section') {
          metaEl.textContent = (node.label || 'Section') + ' · page ' + (node.page || 1);
          if (node.page) {
            activePaperTextPage = Number(node.page || 1);
            renderPaperTextPage();
          }
          return;
        }
        if (kind === 'category') {
          metaEl.textContent = (node.category_id || node.label || 'Category') + ' · ' + (node.paper_count || 0).toLocaleString() + ' papers';
          return;
        }
        if (kind === 'topic') {
          metaEl.textContent = (node.topic_id || node.label || 'Topic') + ' · ' + (node.paper_count || 0).toLocaleString() + ' papers';
          return;
        }
        if (kind === 'year') {
          metaEl.textContent = 'Published in ' + (node.year || node.label || '');
          return;
        }
        metaEl.textContent =
          (node.title || node.paper_id || node.id) +
          (node.primary_category ? ' · ' + node.primary_category : '') +
          (node.year ? ' · ' + node.year : '') +
          (node.similarity ? ' · similarity ' + Number(node.similarity || 0).toFixed(3) : '') +
          (node.has_pdf ? ' · Local PDF' : '');
        if (node.paper_id || node.canonical_paper_id) {
          selectPaperFromViewer({
            paper_id: node.paper_id || node.canonical_paper_id,
            canonical_paper_id: node.canonical_paper_id || node.paper_id,
            title: node.title,
            primary_category: node.primary_category,
            year: node.year,
            has_pdf: node.has_pdf,
          }).catch(console.error);
        }
      });
    }

    async function loadPaperNeighborhood(paperId) {
      if (!paperId) {
        clearPaperNeighborhood('Select a paper to load its graph.');
        return;
      }
      const requestedId = normalizePaperId(paperId);
      const statusEl = document.getElementById('paper-graph-status');
      if (statusEl) {
        statusEl.textContent = 'Loading selected-paper graph...';
      }
      try {
        const params = new URLSearchParams({ limit: '12' });
        const res = await fetch('/api/paper-universe/neighborhood/' + encodeURIComponent(paperId) + '?' + params.toString());
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || ('paper graph failed: ' + res.status));
        }
        if (requestedId && requestedId !== normalizePaperId(activeArxivPaperId)) {
          return;
        }
        renderPaperNeighborhoodGraph(data);
      } catch (err) {
        if (requestedId && requestedId !== normalizePaperId(activeArxivPaperId)) {
          return;
        }
        console.error('paper graph load error', err);
        clearPaperNeighborhood('Failed to load selected-paper graph.');
      }
    }

    function renderRepoUniverse(data) {
      const container = document.getElementById('repo-universe-container');
      const metaEl = document.getElementById('repo-universe-meta');
      if (!container || !metaEl) return;
      if (typeof deck === 'undefined') {
        container.innerHTML = '<div style="padding:0.75rem;font-size:0.85rem;color:#f97316;">3D rendering library did not load.</div>';
        metaEl.textContent = 'deck.gl is unavailable.';
        return;
      }
      const nodes = Array.isArray(data && data.nodes) ? data.nodes : [];
      const edges = Array.isArray(data && data.edges) ? data.edges : [];
      if (!nodes.length) {
        container.innerHTML = '<div style="padding:0.75rem;font-size:0.85rem;color:#9ca3af;">No repository universe data is available.</div>';
        metaEl.textContent = 'No repositories found.';
        return;
      }
      if (!repoUniverseDeck) {
        container.innerHTML = '';
      }
      const {Deck, ScatterplotLayer, LineLayer, TextLayer, OrbitView} = deck;
      const maxEntity = Math.max(1, ...nodes.filter(n => n.kind === 'repo').map(n => Number(n.entity_count || 0)));
      const maxRepoCount = Math.max(1, ...nodes.filter(n => n.kind !== 'repo').map(n => Number(n.repo_count || 0)));
      const modeEl = document.getElementById('repo-universe-layout');
      const mode = modeEl ? (modeEl.value || 'all') : 'all';
      const filterEl = document.getElementById('repo-universe-filter');
      const filter = filterEl ? (filterEl.value || '').toLowerCase().trim() : '';
      const allNodeRows = nodes.map(node => {
        const isRepo = node.kind === 'repo';
        const rawSize = isRepo
          ? 6 + Math.log10(Number(node.entity_count || 1) + 1) / Math.log10(maxEntity + 1) * 10
          : 8 + Math.log10(Number(node.repo_count || 1) + 1) / Math.log10(maxRepoCount + 1) * 8;
        const languages = Array.isArray(node.languages) ? node.languages.join(', ') : '';
        return {
          ...node,
          position: [Number(node.x || 0), Number(node.y || 0), Number(node.z || 0)],
          languages_text: languages,
          radius: Math.max(5, Math.min(18, rawSize)),
          selected: node.id === repoUniverseSelectedId,
          hub_filtered: repoUniverseHubFilter && node.id === repoUniverseHubFilter.id,
        };
      });
      const connectedRepoIds = repoUniverseHubFilter
        ? new Set(edges.flatMap(edge => {
            if (edge.source === repoUniverseHubFilter.id && String(edge.target || '').startsWith('repo:')) return [edge.target];
            if (edge.target === repoUniverseHubFilter.id && String(edge.source || '').startsWith('repo:')) return [edge.source];
            return [];
          }))
        : null;
      const visibleNodes = allNodeRows.filter(node => {
        if (mode === 'repos' && node.kind !== 'repo') return false;
        if (mode === 'hubs' && node.kind === 'repo') return false;
        if (repoUniverseHubFilter) {
          if (node.id === repoUniverseHubFilter.id) {
            // Keep the active hub visible even in "Repos only" mode.
          } else if (node.kind === 'repo') {
            if (!connectedRepoIds || !connectedRepoIds.has(node.id)) return false;
          } else {
            return false;
          }
        }
        if (!filter) return true;
        const haystack = [
          node.label || '',
          node.repo_id || '',
          node.kind || '',
          node.languages_text || '',
          node.library_root || '',
        ].join(' ').toLowerCase();
        return haystack.includes(filter);
      });
      const visibleIds = new Set(visibleNodes.map(node => node.id));
      const nodeById = new Map(allNodeRows.map(node => [node.id, node]));
      const clearHubEl = document.getElementById('repo-universe-clear-filter');
      if (clearHubEl) {
        clearHubEl.style.display = repoUniverseHubFilter ? 'inline-flex' : 'none';
        clearHubEl.textContent = repoUniverseHubFilter
          ? 'Clear ' + (repoUniverseHubFilter.label || 'hub') + ' filter'
          : 'Clear hub filter';
      }
      const lineRows = edges
        .filter(edge => visibleIds.has(edge.source) && visibleIds.has(edge.target))
        .map(edge => {
          const source = nodeById.get(edge.source);
          const target = nodeById.get(edge.target);
          return {
            ...edge,
            sourcePosition: source ? source.position : [0, 0, 0],
            targetPosition: target ? target.position : [0, 0, 0],
          };
        });
      const labels = visibleNodes
        .filter(node => node.kind !== 'repo' || node.selected || Number(node.entity_count || 0) > 900)
        .slice(0, 90);

      const colorForNode = node => {
        if (node.selected) return [249, 115, 22, 255];
        if (node.hub_filtered) return [251, 191, 36, 255];
        if (node.kind === 'language') return [16, 185, 129, 235];
        if (node.kind === 'library_root') return [6, 182, 212, 235];
        if (node.kind === 'skill') return [245, 158, 11, 245];
        if (node.qa_ready) return [96, 165, 250, 245];
        return [59, 130, 246, 210];
      };
      const colorForLine = edge => {
        if (edge.type === 'similar_repo') return [96, 165, 250, 80];
        if (edge.type === 'uses_language') return [16, 185, 129, 105];
        if (edge.type === 'in_library_root') return [6, 182, 212, 105];
        if (edge.type === 'has_skill') return [245, 158, 11, 130];
        return [148, 163, 184, 70];
      };

      const initialViewState = repoUniverseViewState || {
        target: [0, 0, 0],
        rotationX: 28,
        rotationOrbit: 35,
        zoom: 1.15,
      };
      const layers = [
        new LineLayer({
          id: 'repo-universe-lines',
          data: lineRows,
          getSourcePosition: d => d.sourcePosition,
          getTargetPosition: d => d.targetPosition,
          getColor: colorForLine,
          getWidth: d => Math.max(0.8, Math.min(4, Number(d.weight || 0.1) * 3.2)),
          widthUnits: 'pixels',
          pickable: false,
        }),
        new ScatterplotLayer({
          id: 'repo-universe-nodes',
          data: visibleNodes,
          pickable: true,
          radiusUnits: 'pixels',
          getPosition: d => d.position,
          getRadius: d => d.selected ? d.radius + 5 : d.radius,
          getFillColor: colorForNode,
          opacity: 0.92,
          stroked: true,
          getLineColor: d => d.selected ? [255, 237, 213, 255] : [15, 23, 42, 230],
          lineWidthUnits: 'pixels',
          getLineWidth: d => d.selected ? 3 : 1,
          onClick: info => {
            const node = info && info.object;
            if (!node) return;
            repoUniverseSelectedId = node.id;
            updateRepoUniverseMeta(node);
            if (node.kind !== 'repo') {
              repoUniverseHubFilter = {id: node.id, label: node.label || node.id, kind: node.kind || 'hub'};
            }
            renderRepoUniverse(repoUniverseData);
            if (node.kind === 'repo' && node.repo_id) {
              selectRepo(node.repo_id).catch(console.error);
            }
          },
        }),
        new TextLayer({
          id: 'repo-universe-labels',
          data: labels,
          getPosition: d => d.position,
          getText: d => d.label || d.repo_id || d.id,
          getColor: d => d.selected ? [255, 237, 213, 255] : [226, 232, 240, 220],
          getSize: d => d.kind === 'repo' ? 11 : 13,
          sizeUnits: 'pixels',
          getTextAnchor: _ => 'middle',
          getAlignmentBaseline: _ => 'bottom',
          billboard: true,
          pickable: false,
        }),
      ];

      if (repoUniverseDeck) {
        repoUniverseDeck.setProps({layers});
      } else {
        repoUniverseDeck = new Deck({
          parent: container,
          views: [new OrbitView({orbitAxis: 'Z', controller: true})],
          initialViewState,
          controller: true,
          layers,
          getTooltip: ({object}) => {
            if (!object) return null;
            if (object.kind === 'repo') {
              return {
                html:
                  '<b>' + escapeHtml(object.repo_id || object.label || '') + '</b><br/>' +
                  escapeHtml(object.languages_text || 'unknown language') + '<br/>' +
                  'entities: ' + Number(object.entity_count || 0).toLocaleString() +
                  (object.qa_ready ? '<br/>QA ready' : ''),
                style: {backgroundColor: 'rgba(15, 23, 42, 0.94)', color: '#f8fafc'}
              };
            }
            return {
              html: '<b>' + escapeHtml(object.label || object.id || '') + '</b><br/>' + Number(object.repo_count || 0).toLocaleString() + ' repos',
              style: {backgroundColor: 'rgba(15, 23, 42, 0.94)', color: '#f8fafc'}
            };
          },
          onViewStateChange: ({viewState}) => {
            repoUniverseViewState = viewState;
          },
        });
      }
      metaEl.textContent =
        '3D repository universe · ' +
        (data.repo_count || 0).toLocaleString() + ' repositories · ' +
        (data.language_count || 0).toLocaleString() + ' languages · ' +
        (data.qa_ready_count || 0).toLocaleString() + ' QA-ready · ' +
        (data.similarity_edge_count || 0).toLocaleString() + ' similarity edges · ' +
        (repoUniverseHubFilter ? 'filtered by ' + repoUniverseHubFilter.label + ' · ' : '') +
        visibleNodes.length.toLocaleString() + ' visible nodes.';
      highlightRepoUniverse(activeRepo);
    }

    function updateRepoUniverseMeta(node) {
      const metaEl = document.getElementById('repo-universe-meta');
      if (!metaEl || !node) return;
      if (node.kind === 'repo') {
        metaEl.textContent =
          'Loading repository ' + (node.repo_id || node.label || '') +
          (node.languages_text ? ' · ' + node.languages_text : '') +
          (node.entity_count ? ' · ' + Number(node.entity_count).toLocaleString() + ' entities' : '') +
          (node.qa_ready ? ' · QA ready' : '');
        return;
      }
      if (node.kind === 'language') {
        metaEl.textContent = 'Filtering by language: ' + node.label + ' · ' + Number(node.repo_count || 0).toLocaleString() + ' repositories. Click a repo node to load it.';
        return;
      }
      if (node.kind === 'library_root') {
        metaEl.textContent = 'Filtering by library root: ' + node.label + ' · ' + Number(node.repo_count || 0).toLocaleString() + ' repositories. Click a repo node to load it.';
        return;
      }
      if (node.kind === 'skill') {
        metaEl.textContent = 'Filtering by skill: ' + node.label + ' · ' + Number(node.repo_count || 0).toLocaleString() + ' repositories. Click a repo node to load it.';
      }
    }

    async function loadRepoUniverse(force = false) {
      if (repoUniverseLoaded && repoUniverseData && !force) {
        renderRepoUniverse(repoUniverseData);
        return;
      }
      const container = document.getElementById('repo-universe-container');
      const metaEl = document.getElementById('repo-universe-meta');
      if (container) {
        container.innerHTML = '<div style="padding:0.75rem;font-size:0.85rem;color:#9ca3af;">Loading repository universe…</div>';
      }
      if (metaEl) {
        metaEl.textContent = 'Loading repository universe…';
      }
      try {
        const res = await fetch('/api/repo-universe');
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || ('repo universe failed: ' + res.status));
        }
        repoUniverseLoaded = true;
        repoUniverseData = data;
        renderRepoUniverse(data);
      } catch (err) {
        console.error('repo universe load error', err);
        if (container) {
          container.innerHTML = '<div style="padding:0.75rem;font-size:0.85rem;color:#f97316;">Failed to load repository universe.</div>';
        }
        if (metaEl) {
          metaEl.textContent = 'Repository universe failed to load.';
        }
      }
    }

    function resetRepoUniverseView() {
      const filterEl = document.getElementById('repo-universe-filter');
      if (filterEl) {
        filterEl.value = '';
      }
      repoUniverseHubFilter = null;
      repoUniverseSelectedId = activeRepo ? ('repo:' + activeRepo) : null;
      repoUniverseViewState = null;
      if (repoUniverseDeck) {
        repoUniverseDeck.setProps({
          initialViewState: {target: [0, 0, 0], rotationX: 28, rotationOrbit: 35, zoom: 1.15}
        });
      }
      renderRepoUniverse(repoUniverseData);
      highlightRepoUniverse(activeRepo);
    }

    function filterRepoUniverseNodes() {
      renderRepoUniverse(repoUniverseData);
    }

    function clearRepoUniverseHubFilter() {
      repoUniverseHubFilter = null;
      renderRepoUniverse(repoUniverseData);
      highlightRepoUniverse(activeRepo);
    }

    function highlightRepoUniverse(repoId) {
      if (!repoUniverseData || !repoId) return;
      const nextId = 'repo:' + repoId;
      if (repoUniverseSelectedId === nextId) return;
      repoUniverseSelectedId = nextId;
      renderRepoUniverse(repoUniverseData);
    }

    window.loadRepoUniverse = loadRepoUniverse;
    window.resetRepoUniverseView = resetRepoUniverseView;
    window.filterRepoUniverseNodes = filterRepoUniverseNodes;
    window.clearRepoUniverseHubFilter = clearRepoUniverseHubFilter;

    async function loadRepos() {
      const res = await fetch('/api/repos');
      const data = await res.json();
      repos = data.repos || [];
      const listEl = document.getElementById('repo-list');
      if (!listEl) return;
      updateSidebarChrome('repositories');
      listEl.innerHTML = '';
      repos.forEach(r => {
        const div = document.createElement('div');
        div.className = 'repo-item' + (r.repo_id === activeRepo ? ' active' : '');
        div.setAttribute('data-repo-id', r.repo_id || '');
        div.onclick = () => selectRepo(r.repo_id);
        div.innerHTML = '<div class="repo-id">' + r.repo_id + '</div>' +
                        '<div class="repo-meta">' + (r.branch || '-') + ' · ' + (r.head || '').slice(0,7) + '</div>';
        listEl.appendChild(div);
      });
      loadRepoUniverse().catch(console.error);
    }

    // Expose for inline onclick handlers.
    window.loadRepos = loadRepos;

    async function runArxivSearch() {
      const qEl = document.getElementById('arxiv-query');
      const cEl = document.getElementById('arxiv-category');
      const listEl = document.getElementById('repo-list');
      const detailEl = document.getElementById('arxiv-detail');
      const statusEl = document.getElementById('arxiv-status');
      const query = qEl ? (qEl.value || '').trim() : '';
      if (!query) {
        alert('Enter a query for paper search.');
        return;
      }
      const category_prefix = cEl ? (cEl.value || '').trim() : '';
      const payload = {
        query,
        category_prefix: category_prefix || null,
      };
      if (listEl) {
        listEl.innerHTML = '<li style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Searching…</li>';
      }
      clearSelectedPaperDetail('Searching papers…');
      clearPaperText('Select a paper from the results to load full text.');
      clearPaperNeighborhood('Select a paper from the results to load its graph.');
      if (statusEl) {
        statusEl.textContent = 'Searching papers…';
      }
      try {
        const res = await fetch('/api/arxiv/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        arxivResults = data.results || [];
        arxivPage = 1;
        activeArxivPaperId = null;
        const total = typeof data.count === 'number' ? data.count : (arxivResults.length || 0);
        const withPdf = arxivResults.filter(p => p && p.has_pdf).length;
        updateSidebarChrome('papers');
        if (statusEl) {
          statusEl.textContent =
            'Found ' + total + ' paper' + (total === 1 ? '' : 's') +
            (total ? ' · ' + withPdf + ' with local PDFs' : '');
        }
        renderArxivList(arxivResults);
      } catch (err) {
        console.error('arxiv search error', err);
        if (listEl) {
          listEl.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#f97316;">Error running paper search.</div>';
        }
        if (statusEl) {
          statusEl.textContent = 'Paper search failed – see console for details.';
        }
      }
    }

    async function fetchPaperById(paperId) {
      const res = await fetch('/api/arxiv/paper/' + encodeURIComponent(paperId));
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || ('paper lookup failed: ' + res.status));
      }
      return data.paper || null;
    }

    async function selectPaperFromViewer(paper) {
      if (!paper) return;
      const sel = document.getElementById('library-kind');
      if (sel) {
        sel.value = 'papers';
      }
      setActiveLibrary('papers');

      const normalizedTarget = normalizePaperId(paper.paper_id || paper.id || paper.canonical_paper_id);
      if (!normalizedTarget) return;
      renderPaperSelectionPending(paper);

      let matched = arxivResults.find(item => normalizePaperId(item && item.id) === normalizedTarget) || null;
      if (!matched) {
        try {
          matched = await fetchPaperById(paper.paper_id || paper.canonical_paper_id || normalizedTarget);
        } catch (err) {
          console.error('viewer paper lookup failed', err);
          matched = {
            id: paper.paper_id || paper.canonical_paper_id || normalizedTarget,
            title: paper.title || '',
            abstract: '',
            authors: '',
            categories: paper.primary_category || '',
            primary_category: paper.primary_category || '',
            year: paper.year || null,
            has_pdf: false,
          };
        }
        matched = upsertPaperResult(matched);
        arxivPage = 1;
        renderArxivList(arxivResults);
      }
      if (matched) {
        showArxivDetail(matched);
      }
    }

    function renderArxivList(items) {
      const listEl = document.getElementById('repo-list');
      const detailEl = document.getElementById('arxiv-detail');
      const pageMetaEl = document.getElementById('arxiv-page-meta');
      if (!listEl) return;
      listEl.innerHTML = '';
      if (!items.length) {
        listEl.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">No results.</div>';
        clearSelectedPaperDetail('No papers match the current search.');
        clearPaperText('Select a paper to load full text.');
        clearPaperNeighborhood('Select a paper to load its graph.');
        if (pageMetaEl) {
          pageMetaEl.textContent = 'No results';
        }
        return;
      }
      const total = items.length;
      const pageCount = Math.max(1, Math.ceil(total / ARXIV_PAGE_SIZE));
      if (arxivPage < 1) arxivPage = 1;
      if (arxivPage > pageCount) arxivPage = pageCount;
      const startIdx = (arxivPage - 1) * ARXIV_PAGE_SIZE;
      const endIdx = Math.min(startIdx + ARXIV_PAGE_SIZE, total);
      const pageItems = items.slice(startIdx, endIdx);

      pageItems.forEach((paper) => {
        const li = document.createElement('div');
        li.className = 'repo-item' + ((paper.id || '') === activeArxivPaperId ? ' active' : '');
        li.setAttribute('data-paper-id', paper.id || '');
        li.onclick = () => showArxivDetail(paper);
        const title = paper.title || '(untitled)';
        const pid = paper.id || '';
        const cats = paper.categories || '';
        const hasPdf = !!paper.has_pdf;
        const badgeColor = hasPdf ? '#22c55e' : '#6b7280';
        const badgeLabel = hasPdf ? 'Local PDF' : 'Not downloaded';
        li.innerHTML =
          '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.5rem;">' +
            '<div>' +
              '<div style="font-size:0.8rem;font-weight:500;">' + escapeHtml(title) + '</div>' +
              '<div style="font-size:0.75rem;color:#9ca3af;">' + escapeHtml(pid + (cats ? ' · ' + cats : '')) + '</div>' +
            '</div>' +
            '<div style="font-size:0.7rem;color:' + badgeColor + ';white-space:nowrap;">' +
              '<span style="display:inline-flex;align-items:center;gap:0.25rem;">' +
                '<span style="width:7px;height:7px;border-radius:999px;background:' + badgeColor + ';"></span>' +
                escapeHtml(badgeLabel) +
              '</span>' +
            '</div>' +
          '</div>';
        listEl.appendChild(li);
      });
      if (detailEl) {
        if (!activeArxivPaperId && !detailEl.innerHTML.trim()) {
          clearSelectedPaperDetail('Select a paper from the left sidebar to load details, text, and graph.');
        }
      }
      if (!activeArxivPaperId) {
        clearSelectedPaperDetail('Select a paper from the left sidebar to load details, text, and graph.');
        clearPaperText('Select a paper to load full text.');
        clearPaperNeighborhood('Select a paper to load its graph.');
      }
      if (pageMetaEl) {
        pageMetaEl.textContent =
          'Page ' + arxivPage + ' of ' + pageCount +
          ' · showing ' + (startIdx + 1) + '–' + endIdx +
          ' of ' + total;
      }
    }

    function _updateArxivPage(delta) {
      if (!arxivResults || !arxivResults.length) return;
      const total = arxivResults.length;
      const pageCount = Math.max(1, Math.ceil(total / ARXIV_PAGE_SIZE));
      let next = arxivPage + delta;
      if (next < 1) next = 1;
      if (next > pageCount) next = pageCount;
      if (next === arxivPage) return;
      arxivPage = next;
      renderArxivList(arxivResults);
    }

    function nextArxivPage() {
      _updateArxivPage(1);
    }

    function prevArxivPage() {
      _updateArxivPage(-1);
    }

    function showArxivDetail(paper) {
      const detailEl = document.getElementById('arxiv-detail');
      const pdfButton = document.getElementById('arxiv-open-pdf');
      const pdfViewer = document.getElementById('arxiv-pdf-viewer');
      const pdfViewerContainer = document.getElementById('arxiv-pdf-viewer-container');
      if (!detailEl) return;
      const pid = paper.id || paper.paper_id || paper.canonical_paper_id || '';
      const pdfUrl = pid ? ('/api/arxiv/pdf/' + encodeURIComponent(pid)) : null;
      const sourcePdfUrl = pid ? ('https://arxiv.org/pdf/' + encodeURIComponent(pid) + '.pdf') : '';
      const hasPdf = !!paper.has_pdf;
      activeArxivPaperId = pid || null;
      activePaperRecord = paper;
      setSourcePdfLink(sourcePdfUrl);
      if (pdfViewer && pdfViewerContainer) {
        pdfViewer.src = '';
        pdfViewerContainer.style.display = 'none';
      }
      if (pdfButton) {
        if (pdfUrl && hasPdf) {
          pdfButton.disabled = false;
          pdfButton.style.opacity = '1';
          pdfButton.onclick = () => {
            if (pdfViewer && pdfViewerContainer) {
              pdfViewer.src = pdfUrl;
              pdfViewerContainer.style.display = 'block';
            }
          };
        } else {
          pdfButton.disabled = true;
          pdfButton.style.opacity = '0.4';
        }
      }
      if (pdfViewer && pdfViewerContainer && !pdfUrl) {
        pdfViewer.src = '';
        pdfViewerContainer.style.display = 'none';
      }
      renderSelectedPaperDetail(paper);
      const arxivStatusEl = document.getElementById('arxiv-status');
      const universeStatusEl = document.getElementById('paper-universe-status');
      if (arxivStatusEl) {
        arxivStatusEl.textContent = 'Selected paper · ' + (pid || paper.title || 'paper') + ' · loading text and graph…';
      }
      if (universeStatusEl) {
        universeStatusEl.textContent = 'Selected paper loaded into library · ' + (pid || paper.title || 'paper') + '.';
      }
      Array.from(document.getElementsByClassName('repo-item')).forEach(el => {
        const paperId = el.getAttribute('data-paper-id');
        if (paperId !== null) {
          el.classList.toggle('active', normalizePaperId(paperId) === normalizePaperId(activeArxivPaperId));
        }
      });
      loadPaperText(pid).catch(console.error);
      loadPaperNeighborhood(pid).catch(console.error);
    }

    async function downloadSingleArxivPdf() {
      if (!activeArxivPaperId) {
        alert('Select a paper first.');
        return;
      }
      try {
        const res = await fetch('/api/arxiv/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: activeArxivPaperId }),
        });
        const data = await res.json();
        if (!res.ok) {
          alert('Library download failed: ' + (data.detail || res.status));
          return;
        }
        alert('Library download completed: ' + data.downloaded + ' new, ' + data.skipped_existing + ' already present. PDFs are stored under /arxiv/pdfs.');
        // Refresh the PDF link.
        const paper = (arxivResults || []).find(p => p.id === activeArxivPaperId);
        if (paper) {
          paper.has_pdf = true;
          renderArxivList(arxivResults);
          showArxivDetail(paper);
        }
      } catch (err) {
        console.error('arxiv download error', err);
        alert('Download failed due to a network or server error.');
      }
    }

    async function downloadAllArxivPdfs() {
      if (!arxivResults || !arxivResults.length) {
        alert('Run a paper search first.');
        return;
      }
      const ids = arxivResults
        .map(p => p && p.id)
        .filter(id => typeof id === 'string' && id.trim().length > 0);
      if (!ids.length) {
        alert('No valid paper ids found in the current results.');
        return;
      }
      const ok = confirm('Download PDFs for up to ' + ids.length + ' results?\\nThis may take some time.');
      if (!ok) return;
      try {
        const res = await fetch('/api/arxiv/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ids }),
        });
        const data = await res.json();
        if (!res.ok) {
          alert('Bulk library download failed: ' + (data.detail || res.status));
          return;
        }
        alert(
          'Bulk library download completed.\\n' +
          'Requested: ' + data.requested + '\\n' +
          'Downloaded: ' + data.downloaded + '\\n' +
          'Already present: ' + data.skipped_existing + '\\n' +
          (data.errors && data.errors.length ? 'Errors: ' + data.errors.length : '')
        );
        const qEl = document.getElementById('arxiv-query');
        const query = qEl ? (qEl.value || '').trim() : '';
        if (query) {
          await runArxivSearch();
        }
      } catch (err) {
        console.error('arxiv bulk download error', err);
        alert('Bulk download failed due to a network or server error.');
      }
    }

    async function ensureAlgorithmsLoaded() {
      if (algorithmsLoaded) return;
      const listEl = document.getElementById('algorithms-list');
      const detailEl = document.getElementById('algorithms-detail');
      if (listEl) {
        listEl.innerHTML = '<li style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Loading algorithms…</li>';
      }
      if (detailEl) {
        detailEl.innerHTML = '<div style="font-size:0.8rem;color:#9ca3af;">Loading…</div>';
      }
      try {
        const params = new URLSearchParams({ max_results: '1000' });
        const res = await fetch('/api/algorithms?' + params.toString());
        const data = await res.json();
        algorithmsCache = data.algorithms || [];
        algorithmsLoaded = true;
        renderAlgorithmsList(algorithmsCache);
      } catch (err) {
        console.error('algorithms load error', err);
        if (listEl) {
          listEl.innerHTML = '<li style="padding:0.5rem;font-size:0.8rem;color:#f97316;">Error loading algorithms.</li>';
        }
      }
    }

    function renderAlgorithmsList(items) {
      const listEl = document.getElementById('algorithms-list');
      const detailEl = document.getElementById('algorithms-detail');
      if (!listEl) return;
      listEl.innerHTML = '';
      if (!items.length) {
        listEl.innerHTML = '<li style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">No algorithms match the current filters.</li>';
        if (detailEl) {
          detailEl.innerHTML = '<div style="font-size:0.8rem;color:#9ca3af;">No algorithms to show. Try adjusting filters.</div>';
        }
        return;
      }
      items.forEach(algo => {
        const li = document.createElement('li');
        li.style.padding = '0.4rem 0.55rem';
        li.style.cursor = 'pointer';
        li.style.borderBottom = '1px solid #111827';
        li.onmouseenter = () => { li.style.background = '#111827'; };
        li.onmouseleave = () => { li.style.background = 'transparent'; };
        li.onclick = () => showAlgorithmDetail(algo);
        const id = algo.algo_id || '';
        const name = (algo.names && algo.names.length) ? algo.names[0] : '';
        const cat = algo.category || '';
        li.innerHTML =
          '<div style="font-size:0.8rem;font-weight:500;">' + (name || id) + '</div>' +
          '<div style="font-size:0.75rem;color:#9ca3af;">' + id + (cat ? ' · ' + cat : '') + '</div>';
        listEl.appendChild(li);
      });
      if (detailEl && !detailEl.innerHTML.trim()) {
        detailEl.innerHTML = '<div style="font-size:0.8rem;color:#9ca3af;">Select an algorithm from the list to view details.</div>';
      }
    }

    function showAlgorithmDetail(algo) {
      const detailEl = document.getElementById('algorithms-detail');
      if (!detailEl) return;

      const algoId = algo.algo_id || '';
      const names = Array.isArray(algo.names) ? algo.names : [];
      const primaryName = names.length ? names[0] : algoId || '(unnamed)';
      const problems = Array.isArray(algo.problems) ? algo.problems : [];
      const topics = Array.isArray(algo.topics) ? algo.topics : [];
      const tags = Array.isArray(algo.tags) ? algo.tags : [];

      const timeComplexity = algo.time_complexity || {};
      const spaceComplexity = algo.space_complexity || {};
      const properties = algo.properties || {};
      const constraints = algo.constraints || {};
      const notes = algo.notes || '';

      // Track which algorithm is currently rendered to avoid race conditions
      // when async detail fetches complete.
      detailEl.setAttribute('data-algo-id', algoId);

      let html = '';
      html += '<div style="font-size:0.9rem;font-weight:600;margin-bottom:0.25rem;">'
        + primaryName + '</div>';
      html += '<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:0.5rem;">'
        + (algoId ? 'ID: <code style="font-size:0.75rem;">' + algoId + '</code>' : '');
      if (algo.category) {
        html += (algoId ? ' · ' : '') + 'Category: ' + algo.category;
      }
      html += '</div>';

      if (notes) {
        html += '<div style="margin-bottom:0.5rem;">'
          + '<div style="font-size:0.8rem;font-weight:500;margin-bottom:0.15rem;">Notes</div>'
          + '<div style="font-size:0.8rem;white-space:pre-wrap;">'
          + escapeHtml(String(notes))
          + '</div></div>';
      }

      // Placeholder containers for linked problems and implementations; these
      // are hydrated asynchronously by loadAlgorithmLinkedData.
      html += '<div style="margin-bottom:0.5rem;">'
        + '<div style="font-size:0.8rem;font-weight:500;margin-bottom:0.15rem;">Linked problems</div>'
        + '<div id="algorithms-linked-problems" style="font-size:0.8rem;color:#9ca3af;">'
        + (problems.length ? 'Loading problem details…' : 'No linked problems recorded for this algorithm.')
        + '</div>'
        + '</div>';

      html += '<div style="margin-bottom:0.5rem;">'
        + '<div style="font-size:0.8rem;font-weight:500;margin-bottom:0.15rem;">Implementations</div>'
        + '<div id="algorithms-implementations" style="font-size:0.8rem;color:#9ca3af;">'
        + (algoId ? 'Loading implementations…' : 'No algorithm id available.')
        + '</div>'
        + '</div>';

      if (topics.length || tags.length) {
        const chips = topics.concat(tags).map(x => String(x));
        if (chips.length) {
          html += '<div style="margin-bottom:0.5rem;">'
            + '<div style="font-size:0.8rem;font-weight:500;margin-bottom:0.15rem;">Topics / tags</div>'
            + '<div style="display:flex;flex-wrap:wrap;gap:0.25rem;">';
          chips.forEach(token => {
            html += '<span style="font-size:0.7rem;padding:0.1rem 0.3rem;border-radius:999px;'
              + 'background:#111827;border:1px solid #1f2937;">'
              + escapeHtml(token)
              + '</span>';
          });
          html += '</div></div>';
        }
      }

      const summarizeComplexity = (obj) => {
        if (!obj || typeof obj !== 'object') return null;
        const entries = Object.entries(obj);
        if (!entries.length) return null;
        return entries.map(([k, v]) => `${k}: ${v}`).join(', ');
      };

      const timeSummary = summarizeComplexity(timeComplexity);
      const spaceSummary = summarizeComplexity(spaceComplexity);
      if (timeSummary || spaceSummary) {
        html += '<div style="margin-bottom:0.5rem;">'
          + '<div style="font-size:0.8rem;font-weight:500;margin-bottom:0.15rem;">Complexity</div>';
        if (timeSummary) {
          html += '<div style="font-size:0.8rem;margin-bottom:0.1rem;">Time: '
            + escapeHtml(timeSummary) + '</div>';
        }
        if (spaceSummary) {
          html += '<div style="font-size:0.8rem;">Space: '
            + escapeHtml(spaceSummary) + '</div>';
        }
        html += '</div>';
      }

      const kvSection = (title, obj) => {
        if (!obj || typeof obj !== 'object') return '';
        const entries = Object.entries(obj);
        if (!entries.length) return '';
        let block = '<div style="margin-bottom:0.5rem;">'
          + '<div style="font-size:0.8rem;font-weight:500;margin-bottom:0.15rem;">'
          + title + '</div>'
          + '<ul style="list-style:none;margin:0;padding:0;">';
        entries.forEach(([k, v]) => {
          block += '<li style="font-size:0.8rem;color:#e5e7eb;margin-bottom:0.1rem;">'
            + '<span style="color:#9ca3af;">' + escapeHtml(String(k)) + ':</span> '
            + escapeHtml(typeof v === 'string' ? v : JSON.stringify(v))
            + '</li>';
        });
        block += '</ul></div>';
        return block;
      };

      html += kvSection('Properties', properties);
      html += kvSection('Constraints', constraints);

      detailEl.innerHTML = html || '<div style="font-size:0.8rem;color:#9ca3af;">No details available.</div>';

      // Kick off async fetch of linked problems and implementations.
      loadAlgorithmLinkedData(algo).catch(console.error);
    }

    async function loadAlgorithmLinkedData(algo) {
      const detailEl = document.getElementById('algorithms-detail');
      if (!detailEl) return;

      const algoId = algo.algo_id || '';
      const currentAlgoId = detailEl.getAttribute('data-algo-id') || '';
      if (algoId !== currentAlgoId) {
        // User has clicked on a different algorithm since this request started.
        return;
      }

      const problems = Array.isArray(algo.problems) ? algo.problems : [];
      const problemsEl = document.getElementById('algorithms-linked-problems');
      const implsEl = document.getElementById('algorithms-implementations');

      // Load problem details
      if (problemsEl) {
        if (!problems.length) {
          problemsEl.textContent = 'No linked problems recorded for this algorithm.';
        } else {
          try {
            let html = '<ul style="list-style:none;margin:0;padding:0;">';
            for (const pid of problems) {
              const pidStr = String(pid);
              const res = await fetch('/api/algorithms/problems/' + encodeURIComponent(pidStr));
              const data = await res.json();

              // Abort if user navigated away.
              const nowAlgoId = detailEl.getAttribute('data-algo-id') || '';
              if (nowAlgoId !== algoId) return;

              const problem = data && data.problem ? data.problem : null;
              if (!problem) {
                html += '<li style="font-size:0.8rem;color:#e5e7eb;margin-bottom:0.35rem;">'
                  + '<div><code style="font-size:0.75rem;">' + escapeHtml(pidStr) + '</code></div>'
                  + '<div style="font-size:0.75rem;color:#9ca3af;">(problem metadata not found)</div>'
                  + '</li>';
                continue;
              }

              const names = Array.isArray(problem.names) ? problem.names : [];
              const title = names.length ? names[0] : pidStr;
              const desc = problem.description || '';

              html += '<li style="font-size:0.8rem;color:#e5e7eb;margin-bottom:0.35rem;">'
                + '<div style="font-weight:500;">' + escapeHtml(title) + '</div>'
                + '<div style="font-size:0.75rem;color:#9ca3af;margin-bottom:0.1rem;">'
                + '<code style="font-size:0.7rem;">' + escapeHtml(pidStr) + '</code>'
                + '</div>';
              if (desc) {
                html += '<div style="font-size:0.8rem;white-space:pre-wrap;">'
                  + escapeHtml(String(desc))
                  + '</div>';
              }
              html += '</li>';
            }
            html += '</ul>';
            problemsEl.innerHTML = html;
          } catch (err) {
            console.error('failed to load problem details', err);
            problemsEl.textContent = 'Error loading problem details.';
          }
        }
      }

      // Load implementations
      if (implsEl) {
        if (!algoId) {
          implsEl.textContent = 'No algorithm id available.';
        } else {
          try {
            const params = new URLSearchParams({ algo_id: algoId, max_results: '50' });
            const res = await fetch('/api/algorithms/implementations?' + params.toString());
            const data = await res.json();

            // Abort if user navigated away.
            const nowAlgoId = detailEl.getAttribute('data-algo-id') || '';
            if (nowAlgoId !== algoId) return;

            const items = (data && data.results) || [];
            if (!items.length) {
              implsEl.textContent = 'No concrete implementations recorded for this algorithm.';
              return;
            }

            let html = '<ul style="list-style:none;margin:0;padding:0;">';
            items.forEach(impl => {
              const implId = impl.impl_id || '';
              const lang = impl.language || '';
              const repoId = impl.repo_id || '';
              const filePath = impl.file_path || '';
              const entrySymbol = impl.entry_symbol || '';

              html += '<li style="font-size:0.8rem;color:#e5e7eb;margin-bottom:0.35rem;">';
              html += '<div style="font-weight:500;">'
                + (lang ? '[' + escapeHtml(lang) + '] ' : '')
                + (repoId ? escapeHtml(String(repoId)) : '(unknown repo)')
                + '</div>';
              if (filePath) {
                html += '<div style="font-size:0.75rem;color:#9ca3af;">'
                  + escapeHtml(String(filePath));
                if (entrySymbol) {
                  html += ' · <code style="font-size:0.7rem;">'
                    + escapeHtml(String(entrySymbol)) + '</code>';
                }
                html += '</div>';
              }
              if (implId) {
                html += '<div style="font-size:0.7rem;color:#4b5563;">impl_id: '
                  + '<code style="font-size:0.7rem;">' + escapeHtml(String(implId)) + '</code>'
                  + '</div>';
              }
              if (impl.notes) {
                html += '<div style="font-size:0.75rem;color:#9ca3af;white-space:pre-wrap;">'
                  + escapeHtml(String(impl.notes))
                  + '</div>';
              }
              html += '</li>';
            });
            html += '</ul>';
            implsEl.innerHTML = html;
          } catch (err) {
            console.error('failed to load implementations', err);
            implsEl.textContent = 'Error loading implementations.';
          }
        }
      }
    }

    function escapeHtml(str) {
      if (str == null) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    async function runAlgorithmsSearch() {
      await ensureAlgorithmsLoaded();
      const qEl = document.getElementById('algorithms-query');
      const tEl = document.getElementById('algorithms-topic');
      const query = qEl ? (qEl.value || '').trim().toLowerCase() : '';
      const topic = tEl ? (tEl.value || '').trim().toLowerCase() : '';

      let items = algorithmsCache.slice();
      if (query) {
        items = items.filter(algo => {
          const parts = [];
          if (algo.algo_id) parts.push(String(algo.algo_id));
          if (Array.isArray(algo.names)) parts.push(algo.names.join(' '));
          if (typeof algo.notes === 'string') parts.push(algo.notes);
          const haystack = parts.join(' ').toLowerCase();
          return haystack.includes(query);
        });
      }
      if (topic) {
        items = items.filter(algo => {
          const topics = Array.isArray(algo.topics) ? algo.topics : [];
          const tags = Array.isArray(algo.tags) ? algo.tags : [];
          const all = topics.concat(tags).map(x => String(x).toLowerCase());
          return all.includes(topic);
        });
      }
      renderAlgorithmsList(items);
    }

    async function selectRepo(repoId) {
      activeRepo = repoId;
      const res = await fetch('/api/repos/' + encodeURIComponent(repoId));
      const data = await res.json();
      const detailsEl = document.getElementById('repo-details');
      if (detailsEl) {
        detailsEl.textContent = JSON.stringify(data, null, 2);
      }
      const commitMetaEl = document.getElementById('repo-commit-count');
      if (commitMetaEl) {
        const n = (typeof data.commit_count === 'number') ? data.commit_count : null;
        if (n !== null) {
          // Use a localized string so large repositories remain readable.
          commitMetaEl.textContent = 'Commits: ' + n.toLocaleString();
        } else {
          commitMetaEl.textContent = 'Commits: unknown';
        }
      }
      Array.from(document.getElementsByClassName('repo-item')).forEach(el => {
        const repoAttr = el.getAttribute('data-repo-id');
        if (repoAttr !== null) {
          el.classList.toggle('active', repoAttr === repoId);
        }
      });
      highlightRepoUniverse(repoId);
      await loadSkills();
      await loadGraph();
    }

    function refreshInteractionSkillOptions() {
      const sel = document.getElementById('interaction-skill');
      if (!sel) return;
      const skills = skillsByRepo[activeRepo] || [];
      const built = skills.filter(s => s && s.status === 'up_to_date');
      sel.innerHTML = '';
      if (!built.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '(no built skills for this repo)';
        sel.appendChild(opt);
        sel.disabled = true;
        return;
      }
      sel.disabled = false;
      built.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.skill;
        opt.textContent = s.skill;
        sel.appendChild(opt);
      });
    }

    async function runInteraction() {
      const msg = document.getElementById('chat-input').value || '';
      const modeSel = document.getElementById('interaction-mode');
      const skillSel = document.getElementById('interaction-skill');
      const qaModeSel = document.getElementById('qa-mode');
      const metaNumEl = document.getElementById('meta-num');
      const mode = modeSel ? (modeSel.value || 'repo_skill_chat') : 'repo_skill_chat';
      const skill = skillSel ? (skillSel.value || '').trim() : '';
      const qaMode = qaModeSel ? (qaModeSel.value || '') : '';
      const numTasks = metaNumEl ? (parseInt(metaNumEl.value || '0', 10) || 0) : 0;

      if (mode === 'repo_skill_chat') {
        if (!activeRepo) {
          alert('Select a repository first.');
          return;
        }
        if (!skill) {
          alert('Select a built skill for this repo.');
          return;
        }
        // Enforce that the selected per-repo skill is actually built
        // before allowing an LLM-backed interaction. This mirrors the
        // backend constraint that we require a repo-local adapter.
        const skills = skillsByRepo[activeRepo] || [];
        const rec = skills.find(s => s && s.skill === skill);
        if (!rec || rec.status !== 'up_to_date') {
          const status = rec ? rec.status : 'not_built';
          alert('Skill "' + skill + '" for this repo is not ready (status: ' + status + '). Build this skill first.');
          return;
        }
        const response = await fetch('/api/skill_chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: msg,
            skill,
            repo_hint: activeRepo,
            qa_mode: qaMode || null
          })
        });
        const data = await response.json();
        document.getElementById('interaction-output').textContent = JSON.stringify(data, null, 2);
        return;
      } else if (mode === 'comparative_qa') {
        const hints = repos.slice(0, 3).map(r => r.repo_id);
        const response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: msg,
            mode: 'qa_comparative',
            repo_hints: hints,
            qa_mode: qaMode || null
          })
        });
      } else if (mode === 'coarse_retrieval') {
        const response = await fetch('/api/coarse_retrieve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: msg,
            top_k_repos: 5,
            top_k_papers: 5,
            top_k_spans: 6
          })
        });
      } else if (mode === 'meta_skill') {
        const target = repos.slice(0, 3).map(r => r.repo_id);
        const taskFamily = (msg || '').trim() || 'style_imitation';
        const response = await fetch('/api/task', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mode: 'meta_skill',
            config: {
              task_family: taskFamily,
              target_repos: target,
              num_tasks: numTasks
            }
          })
        });
      } else {
        alert('Unsupported interaction mode: ' + mode);
        return;
      }

      const data = await response.json();
      const outEl = document.getElementById('interaction-output');
      if (outEl) {
        if (mode === 'coarse_retrieval' && data && typeof data.answer === 'string') {
          const raw = (data.result != null) ? JSON.stringify(data.result, null, 2) : JSON.stringify(data, null, 2);
          outEl.textContent = data.answer + '\\n\\n---\\n\\n' + raw;
        } else {
          outEl.textContent = JSON.stringify(data, null, 2);
        }
      }
    }

    async function loadSkills() {
      if (!activeRepo) {
        document.getElementById('skill-statuses').textContent = '[]';
        return;
      }
      const res = await fetch('/api/skills/' + encodeURIComponent(activeRepo));
      const data = await res.json();
      const skills = data.skills || [];
      skillsByRepo[activeRepo] = skills;
      document.getElementById('skill-statuses').textContent = JSON.stringify(skills, null, 2);
      refreshInteractionSkillOptions();
    }

    async function buildQaSkill() {
      if (!activeRepo) {
        alert('Select a repository first.');
        return;
      }
      const res = await fetch('/api/skills/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_id: activeRepo, skill: 'qa' })
      });
      const data = await res.json();
      await loadSkills();
      console.log('build result', data);
    }

    async function buildAllSkills() {
      const skillSel = document.getElementById('global-skill');
      const skill = skillSel ? (skillSel.value || 'qa') : 'qa';
      const ok = confirm('Build "' + skill + '" skill for all repositories?');
      if (!ok) return;
      const res = await fetch('/api/skills/build_all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skill })
      });
      const data = await res.json();
      console.log('build_all result', data);
      await loadSkills();
    }

    async function loadGraph() {
      const container = document.getElementById('graph-container');
      if (!container) return;
      if (!activeRepo) {
        container.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Select a repository to view its graph.</div>';
        return;
      }
      try {
        container.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Loading graph…</div>';
        // Use conservative defaults for large repositories; the backend will cap further.
        const params = new URLSearchParams({ max_nodes: '800', max_edges: '1600' });
        const res = await fetch('/api/graph/' + encodeURIComponent(activeRepo) + '?' + params.toString());
        if (!res.ok) {
          const msg = await res.text();
          container.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">No graph available for this repo (' + res.status + ').</div>';
          console.error('graph load failed', msg);
          if (cy) {
            cy.destroy();
            cy = null;
          }
          return;
        }
        const data = await res.json();
        renderGraph(data);
      } catch (err) {
        console.error('graph load error', err);
        container.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Failed to load graph.</div>';
      }
    }

    function getGraphLayout(name, nodeCount) {
      const n = typeof nodeCount === 'number' && nodeCount > 0 ? nodeCount : 0;
      if (name === 'concentric') {
        return {
          name: 'concentric',
          animate: false,
          fit: true,
          padding: 30,
          startAngle: (3 * Math.PI) / 2,
          sweep: 2 * Math.PI,
          minNodeSpacing: 20,
          equidistant: false
        };
      }
      if (name === 'breadthfirst') {
        return {
          name: 'breadthfirst',
          animate: false,
          fit: true,
          padding: 30,
          directed: true,
          spacingFactor: 1.4
        };
      }
      // Default to COSE-style force-directed layout tuned for medium-sized graphs.
      return {
        name: 'cose',
        animate: false,
        fit: true,
        padding: 30,
        spacingFactor: n > 600 ? 1.2 : 1.4,
        idealEdgeLength: n > 600 ? 40 : 55,
        nodeRepulsion: n > 600 ? 250000 : 400000,
        gravity: 80,
        numIter: 2500,
        randomize: true,
        componentSpacing: 80
      };
    }

    function renderGraph(data) {
      const container = document.getElementById('graph-container');
      if (!container) return;
      if (typeof cytoscape === 'undefined') {
        container.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Graph library not loaded.</div>';
        return;
      }
      const nodes = (data && data.nodes) || [];
      const edges = (data && data.edges) || [];
      if (!nodes.length) {
        container.innerHTML = '<div style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">Graph is empty for this repo.</div>';
        if (cy) {
          cy.destroy();
          cy = null;
        }
        return;
      }
      if (cy) {
        cy.destroy();
        cy = null;
      }
      const layoutSelect = document.getElementById('graph-layout');
      const layoutName = layoutSelect ? (layoutSelect.value || 'cose') : 'cose';
      const elements = [];
      nodes.forEach(n => {
        elements.push({
          data: {
            id: String(n.id),
            label: n.label || String(n.id),
            kind: n.kind || '',
            uri: n.uri || '',
            owner: n.owner || ''
          }
        });
      });
      edges.forEach(e => {
        const src = String(e.source);
        const dst = String(e.target);
        if (!src || !dst) return;
        const etype = e.type || '';
        const id = e.id || (src + '->' + dst + (etype ? ':' + etype : ''));
        elements.push({
          data: {
            id,
            source: src,
            target: dst,
            type: etype
          }
        });
      });

      cy = cytoscape({
        container,
        elements,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#3b82f6',
              'label': 'data(label)',
              'font-size': 9,
              'color': '#e5e7eb',
              'text-valign': 'center',
              'text-halign': 'center',
              'text-wrap': 'wrap',
              'text-max-width': 80,
              'text-outline-color': '#020617',
              'text-outline-width': 2,
              'width': 18,
              'height': 18,
              'border-width': 1,
              'border-color': '#020617'
            }
          },
          {
            selector: 'node[kind = "module"]',
            style: { 'background-color': '#f59e0b', 'shape': 'round-rectangle' }
          },
          {
            selector: 'node[kind = "class"]',
            style: { 'background-color': '#10b981', 'shape': 'round-rectangle' }
          },
          {
            selector: 'node[kind = "function"]',
            style: { 'background-color': '#6366f1', 'shape': 'ellipse' }
          },
          {
            selector: 'edge',
            style: {
              'width': 1,
              'line-color': '#4b5563',
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'target-arrow-color': '#4b5563',
              'opacity': 0.65
            }
          },
          {
            selector: 'node:selected',
            style: {
              'border-color': '#f97316',
              'border-width': 2,
              'width': 22,
              'height': 22
            }
          },
          {
            selector: 'edge:selected',
            style: {
              'line-color': '#f97316',
              'target-arrow-color': '#f97316',
              'width': 2
            }
          },
          {
            selector: '.faded',
            style: { 'opacity': 0.15 }
          }
        ],
        layout: getGraphLayout(layoutName, nodes.length),
        wheelSensitivity: 0.2
      });

      cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        const neigh = node.closedNeighborhood();
        cy.elements().removeClass('faded');
        cy.elements().difference(neigh).addClass('faded');
        const meta = document.getElementById('graph-meta');
        if (meta) {
          const label = node.data('label') || String(node.id());
          const kind = node.data('kind') || 'unknown';
          const owner = node.data('owner') || '';
          const uri = node.data('uri') || '';
          const parts = [];
          parts.push(label);
          if (kind) parts.push('[' + kind + ']');
          if (owner) parts.push('· ' + owner);
          if (uri) parts.push('— ' + uri);
          meta.textContent = parts.join('  ');
        }
        const uri = node.data('uri');
        if (uri) {
          loadSourceForNode(uri).catch(console.error);
        }
      });

      cy.on('tap', function(evt) {
        if (evt.target === cy) {
          cy.elements().removeClass('faded');
          const meta = document.getElementById('graph-meta');
          if (meta) {
            meta.textContent = '';
          }
        }
      });
    }

    function resetGraphView() {
      const filterEl = document.getElementById('graph-filter');
      if (filterEl) {
        filterEl.value = '';
      }
      if (!cy) return;
      cy.elements().removeClass('faded');
      cy.fit();
    }

    function filterGraphNodes() {
      const input = document.getElementById('graph-filter');
      if (!cy || !input) return;
      const q = (input.value || '').toLowerCase().trim();
      cy.elements().removeClass('faded');
      if (!q) return;
      const matched = cy.nodes().filter(n => {
        const label = (n.data('label') || '').toLowerCase();
        return label.includes(q);
      });
      const others = cy.nodes().difference(matched);
      others.addClass('faded');
      cy.edges().addClass('faded');
      matched.connectedEdges().removeClass('faded');
    }

    function toggleSourceCollapse() {
      const panel = document.getElementById('source-panel');
      const content = document.getElementById('source-content');
      const btn = document.getElementById('source-toggle-btn');
      if (!panel || !content || !btn) return;
      if (content.style.display === 'none') {
        content.style.display = 'block';
        btn.textContent = 'Hide';
      } else {
        content.style.display = 'none';
        btn.textContent = 'Show';
      }
    }

    async function loadSourceForNode(uri) {
      const panel = document.getElementById('source-panel');
      const metaEl = document.getElementById('source-meta');
      const contentEl = document.getElementById('source-content');
      const btn = document.getElementById('source-toggle-btn');
      if (!panel || !metaEl || !contentEl || !btn) return;
      if (!activeRepo) {
        return;
      }
      panel.style.display = 'block';
      contentEl.style.display = 'block';
      btn.textContent = 'Hide';
      metaEl.textContent = 'Loading source…';
      contentEl.textContent = '';
      try {
        const params = new URLSearchParams({ uri: uri });
        const res = await fetch('/api/source/' + encodeURIComponent(activeRepo) + '?' + params.toString());
        if (!res.ok) {
          const msg = await res.text();
          metaEl.textContent = 'Failed to load source (' + res.status + ').';
          console.error('source load failed', msg);
          return;
        }
        const data = await res.json();
        const span = data.span || {};
        const snippet = data.snippet || {};
        const lines = snippet.lines || [];
        const path = data.path || '';
        const hash = data.hash || '';
        const spanText = (span.start_line && span.end_line)
          ? 'L' + span.start_line + '-L' + span.end_line
          : '';
        const hashText = hash ? ' sha256:' + String(hash).slice(0, 8) : '';
        metaEl.textContent = path + (spanText ? '  [' + spanText + ']' : '') + hashText;
        if (!lines.length) {
          contentEl.textContent = '(no source lines available)';
          return;
        }
        const buf = [];
        for (const ln of lines) {
          const n = typeof ln.line_no === 'number' ? ln.line_no : null;
          const numStr = n !== null ? String(n).padStart(6, ' ') + ' ' : '';
          buf.push(numStr + (ln.text || ''));
        }
        contentEl.textContent = buf.join('\\n');
      } catch (err) {
        console.error('source load error', err);
        metaEl.textContent = 'Failed to load source.';
        contentEl.textContent = '';
      }
    }

    window.addEventListener('message', (event) => {
      if (event.origin !== window.location.origin) return;
      const data = event.data || {};
      if (data.type === 'paper_universe_select_paper' && data.paper) {
        selectPaperFromViewer(data.paper).catch(console.error);
      }
    });

    loadRepos().catch(console.error);
    setActiveLibrary('repositories');
  </script>
</body>
</html>
    """
    return HTMLResponse(html)


@app.get("/api/repos")
async def api_list_repos() -> Dict[str, Any]:
    """
    List repositories known to the library, along with basic metadata and
    context keys.
    """
    manifest = load_manifest()
    repos_meta = manifest.get("repos") or {}
    if not isinstance(repos_meta, dict):
        repos_meta = {}

    out: List[Dict[str, Any]] = []
    for rid, entry_any in repos_meta.items():
        entry = entry_any if isinstance(entry_any, dict) else {}
        state = entry.get("repo_state") or {}
        if not isinstance(state, dict):
            state = {}
        out.append(
            {
                "repo_id": rid,
                "repo_root": entry.get("repo_root"),
                "last_indexed_at": entry.get("last_indexed_at"),
                "branch": state.get("branch"),
                "head": state.get("head"),
                "context_key": compute_repo_context_key(rid, entry),
                "has_indices": bool(entry.get("indices")),
                "has_skills": bool(entry.get("skills")),
                "has_extensions": bool(entry.get("extensions")),
                "has_repo_skills_miner": bool(
                    isinstance(entry.get("extensions"), dict)
                    and (entry.get("extensions") or {}).get("repo_skills_miner")
                ),
            }
        )
    return {"repos": out}


@app.get("/api/repo-universe")
async def api_repo_universe(max_similarity_edges: int = 350) -> Dict[str, Any]:
    """
    Return a library-wide interactive graph where repository nodes are connected
    to language/root/skill hubs and to nearby repositories by simple similarity
    signals derived from manifest metadata.
    """
    max_edges = max(0, min(int(max_similarity_edges or 0), 1000))
    return _repo_universe_payload(max_similarity_edges=max_edges)


@app.get("/api/repos/{repo_id}")
async def api_get_repo(repo_id: str) -> Dict[str, Any]:
    """
    Return manifest metadata for a single repository.
    """
    manifest = load_manifest()
    repos_meta = manifest.get("repos") or {}
    if not isinstance(repos_meta, dict):
        repos_meta = {}
    entry_any = repos_meta.get(repo_id)
    if not isinstance(entry_any, dict):
        raise HTTPException(status_code=404, detail=f"repo_id not found: {repo_id!r}")
    entry = dict(entry_any)
    entry["repo_id"] = repo_id
    entry["context_key"] = compute_repo_context_key(repo_id, entry)

    # Attach a best-effort commit count based on the current Git history.
    repo_root = entry.get("repo_root")
    commit_count: Optional[int] = None
    if isinstance(repo_root, str) and repo_root:
        commit_count = _compute_repo_commit_count(repo_root)
    entry["commit_count"] = commit_count

    return entry


@app.get("/api/repos/{repo_id}/extensions/repo_skills_miner")
async def api_get_repo_skills_miner_extension(
    repo_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Return structured repo_skills_miner data for a single repository.
    """
    manifest = load_manifest()
    repos_meta = manifest.get("repos") or {}
    if not isinstance(repos_meta, dict):
        repos_meta = {}
    entry_any = repos_meta.get(repo_id)
    if not isinstance(entry_any, dict):
        raise HTTPException(status_code=404, detail=f"repo_id not found: {repo_id!r}")

    entry = dict(entry_any)
    extensions = entry.get("extensions") or {}
    if not isinstance(extensions, dict):
        extensions = {}
    ext_meta_any = extensions.get("repo_skills_miner")
    if not isinstance(ext_meta_any, dict):
        raise HTTPException(
            status_code=404,
            detail=f"repo_skills_miner extension not found for repo_id={repo_id!r}",
        )

    ext_meta = dict(ext_meta_any)
    paths = ext_meta.get("paths") or {}
    if not isinstance(paths, dict):
        paths = {}
    summary_path = _resolve_export_relative_path(str(paths.get("summary") or ""))
    skills_path = _resolve_export_relative_path(str(paths.get("skills") or ""))
    annotations_path = _resolve_export_relative_path(str(paths.get("annotations") or ""))
    signals_path = _resolve_export_relative_path(str(paths.get("signals") or ""))
    limit = max(1, min(int(limit), 100))

    return {
        "repo_id": repo_id,
        "extension": "repo_skills_miner",
        "metadata": ext_meta,
        "summary": _read_json_file(summary_path),
        "skills": _read_jsonl_rows(skills_path, limit=limit),
        "annotations": _read_jsonl_rows(annotations_path, limit=limit),
        "signals": _read_jsonl_rows(signals_path, limit=limit),
    }


@app.post("/api/query")
async def api_query(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Plan a query over the repository library.

    Expected payload:
        {
          "question": "...",
          "mode": "qa" | "qa_comparative",
          "repo_hint": "repo_id?",           # for mode == "qa"
          "repo_hints": ["id1", "id2", ...], # for mode == "qa_comparative"
          "qa_mode": "docs" | "symbol" | "code_region" | "usage" | "change" | null?
        }
    """
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` is required.")

    mode_raw = str(payload.get("mode") or QueryMode.QA.value)
    try:
        mode = QueryMode(mode_raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid mode: {mode_raw!r}")

    repo_hint: Optional[str] = payload.get("repo_hint")
    repo_hints: Optional[List[str]] = payload.get("repo_hints")
    qa_mode_raw = payload.get("qa_mode", None)
    qa_mode: Optional[str] = None
    if isinstance(qa_mode_raw, str):
        qa_mode = qa_mode_raw or None

    try:
        plan = repo_lib.query(
            question=question,
            mode=mode,
            repo_hint=repo_hint,
            repo_hints=repo_hints,
            qa_mode=qa_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return plan


@app.post("/api/skill_chat")
async def api_skill_chat(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Execute a per-repo, per-skill chat interaction.

    This endpoint is designed around the concept of *built skills*:
    - The frontend only allows selecting skills whose status is
      "up_to_date" for the active repo.
    - The backend then plans and/or executes the interaction based
      on the requested `skill`.

    For now:
    - The "qa" skill performs a real library query, validates that a
      per-repo QA adapter is present, and returns both a structured
      plan and a graph-backed textual answer.
    - Other skills return a clearly-marked stub response until their
      runtimes are implemented.
    """
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` is required.")

    skill_raw = payload.get("skill")
    skill = str(skill_raw) if skill_raw is not None else ""
    if not skill:
        raise HTTPException(status_code=400, detail="`skill` is required.")

    repo_hint_raw = payload.get("repo_hint")
    repo_hint: Optional[str] = str(repo_hint_raw) if repo_hint_raw is not None else None
    if not repo_hint:
        raise HTTPException(
            status_code=400,
            detail="`repo_hint` (single repo_id) is required for skill chat.",
        )

    qa_mode_raw = payload.get("qa_mode", None)
    qa_mode: Optional[str] = None
    if isinstance(qa_mode_raw, str):
        qa_mode = qa_mode_raw or None

    result = _execute_skill_chat(
        skill=skill,
        question=question,
        repo_hint=repo_hint,
        qa_mode=qa_mode,
    )
    return result


@app.post("/api/qa_execute")
async def api_qa_execute(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Execute a QA-style interaction by first planning via RepoLibrary,
    then (conceptually) routing through a per-repo QA skill / adapter.

    This endpoint enforces that the target repo has a built QA skill,
    i.e., that a repo-local QA adapter is present in the adapter
    registry. It returns:

        {
          "type": "qa_result",
          "plan": { ... query_plan ... },
          "answer": "<model answer text>"
        }

    The default implementation uses a lightweight, non-LLM QA routine
    over the repository's program graph to produce `answer`. To hook up
    a real model, replace the body of `_format_qa_answer_stub` with a
    call into your LLM/adapter runtime.
    """
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` is required.")

    repo_hint_raw = payload.get("repo_hint")
    repo_hint: Optional[str] = str(repo_hint_raw) if repo_hint_raw is not None else None
    if not repo_hint:
        raise HTTPException(
            status_code=400,
            detail="`repo_hint` (single repo_id) is required for QA execution.",
        )

    qa_mode_raw = payload.get("qa_mode", None)
    qa_mode: Optional[str] = None
    if isinstance(qa_mode_raw, str):
        qa_mode = qa_mode_raw or None

    # Delegate to the generic skill-chat executor for skill="qa".
    result = _execute_skill_chat(
        skill="qa",
        question=question,
        repo_hint=repo_hint,
        qa_mode=qa_mode,
    )
    # Preserve a stable type field for QA-specific callers.
    plan = result.get("plan")
    answer_text = result.get("answer")
    return {
        "type": "qa_result",
        "status": "completed",
        "plan": plan,
        "answer": answer_text,
    }


@app.post("/api/coarse_retrieve")
async def api_coarse_retrieve(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Execute lane-based coarse retrieval across:
    - repo semantic summaries
    - aligned paper texts
    - coarse paper-span ↔ repo-chunk bridge rows

    Expected payload:
        {
          "question": "...",
          "top_k_repos": 5?,
          "top_k_papers": 5?,
          "top_k_spans": 6?
        }
    """
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` is required.")

    def _as_optional_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            raise HTTPException(status_code=400, detail="top-k values must be integers.")

    top_k_repos = _as_optional_int(payload.get("top_k_repos"), 5)
    top_k_papers = _as_optional_int(payload.get("top_k_papers"), 5)
    top_k_spans = _as_optional_int(payload.get("top_k_spans"), 6)

    try:
        result = _execute_coarse_retrieval(
            question=question,
            top_k_repos=top_k_repos,
            top_k_papers=top_k_papers,
            top_k_spans=top_k_spans,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"coarse retrieval failed: {exc}")

    answer = _format_coarse_retrieval_answer(question, result)

    return {
        "type": "coarse_retrieval_result",
        "status": "completed",
        "answer": answer,
        "result": result,
    }


@app.post("/api/task")
async def api_task(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Plan a library-level task such as meta-skill training or agentic edit.

    Expected payload:
        {
          "mode": "meta_skill" | "agent_edit",
          "config": { ... mode-specific keys ... }
        }
    """
    mode_raw = str(payload.get("mode") or "").strip()
    if not mode_raw:
        raise HTTPException(status_code=400, detail="`mode` is required.")
    try:
        mode = TaskMode(mode_raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid mode: {mode_raw!r}")

    config_any = payload.get("config") or {}
    if not isinstance(config_any, dict):
        raise HTTPException(status_code=400, detail="`config` must be an object.")
    config: Dict[str, Any] = dict(config_any)

    try:
        plan = repo_lib.run_task(mode=mode, config=config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return plan


@app.get("/api/skills/{repo_id}")
async def api_skills_for_repo(repo_id: str) -> Dict[str, Any]:
    """
    Return status for the known SkillSet skills for a given repo.
    """
    try:
        skills = all_skill_statuses_for_repo(repo_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"repo_id": repo_id, "skills": skills}


@app.post("/api/skills/build")
async def api_skill_build(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Trigger a skill build for a given (repo_id, skill) pair.

    Expected payload:
        {
          "repo_id": "...",
          "skill": "qa" | "edit" | "meta" | ...
          "force": bool?   # optional, default False
        }
    """
    repo_id = str(payload.get("repo_id") or "").strip()
    skill = str(payload.get("skill") or "").strip()
    if not repo_id or not skill:
        raise HTTPException(status_code=400, detail="`repo_id` and `skill` are required.")
    force = bool(payload.get("force") or False)
    try:
        summary = build_skill(repo_id, skill, force=force)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return summary


@app.post("/api/skills/build_all")
async def api_skill_build_all(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Trigger a skill build for all repositories in the library.

    Expected payload:
        {
          "skill": "qa" | "edit" | "meta" | ...,
          "force": bool?   # optional, default False
        }
    """
    skill = str(payload.get("skill") or "").strip()
    if not skill:
        raise HTTPException(status_code=400, detail="`skill` is required.")
    force = bool(payload.get("force") or False)

    manifest = load_manifest()
    repos_meta = manifest.get("repos") or {}
    if not isinstance(repos_meta, dict):
        repos_meta = {}

    results: List[Dict[str, Any]] = []
    for rid in repos_meta.keys():
        try:
            res = build_skill(rid, skill, force=force)
            results.append(res)
        except Exception:
            # Skip repos that fail to build; they can be inspected individually.
            continue

    changed = sum(1 for r in results if r.get("changed"))
    return {"skill": skill, "changed": changed, "results": results}


@app.post("/api/arxiv/search")
async def api_arxiv_search(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Search the local ArXiv metadata snapshot under `/data/arxiv`.

    This is a lightweight, metadata-only search helper designed to make
    it easy to explore papers without hitting external services.

    Expected payload:
        {
          "query": "...",                # required; keyword, case-insensitive
          "max_results": int?,           # optional hard limit; if omitted,
                                         # the server returns all matches
                                         # and the UI is expected to paginate.
          "fields": ["title","abstract","authors"]?,  # optional subset
          "category_prefix": "cs.CL"?   # optional arXiv category prefix
        }
    """
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="`query` is required.")

    # By default, return *all* matches for the query and let the client/UI
    # paginate (e.g., 50 per page) purely on the front-end. Callers that
    # want a hard cap can still pass an explicit `max_results` value.
    max_results_raw = payload.get("max_results", None)
    if max_results_raw is None:
        # Effectively "no limit" for the local snapshot; the underlying
        # search helper will stop at EOF.
        max_results = 10**9
    else:
        try:
            max_results = int(max_results_raw)
        except Exception:
            max_results = 50
        max_results = max(1, max_results)

    fields_any = payload.get("fields")
    fields = None
    if isinstance(fields_any, list):
        fields = [str(f) for f in fields_any]

    category_prefix_any = payload.get("category_prefix")
    category_prefix: Optional[str] = None
    if isinstance(category_prefix_any, str):
        category_prefix = category_prefix_any or None

    try:
        results = arxiv_search_keyword(
            query,
            max_results=max_results,
            fields=fields,
            category_prefix=category_prefix,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"failed to search local ArXiv metadata: {exc}",
        )

    # Annotate each result with whether a local PDF is already present.
    results_with_pdf: List[Dict[str, object]] = [_enrich_arxiv_record(rec) for rec in results]

    return {
        "type": "arxiv_search_result",
        "query": query,
        "count": len(results_with_pdf),
        "results": results_with_pdf,
    }


@app.get("/api/arxiv/pdf/{paper_id}")
async def api_arxiv_pdf(paper_id: str):
    """
    Return a locally downloaded Arxiv PDF for the given paper_id, if present.

    PDFs may live either directly under `/arxiv/pdfs/` or under
    `/arxiv/pdfs/YYMM/`, depending on which downloader wrote them.
    """
    # Normalize to the trailing segment to match the downloader's convention.
    pdf_id = str(paper_id or "").strip().split("/")[-1]
    if not pdf_id:
        raise HTTPException(status_code=400, detail="invalid paper_id")
    pdf_path = _find_local_arxiv_pdf(pdf_id)
    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"PDF not found for paper_id={paper_id!r}. "
                "Ensure it has been downloaded under /arxiv/pdfs."
            ),
        )
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename=\"{pdf_id}.pdf\"'},
    )


@app.get("/api/arxiv/paper/{paper_id}")
async def api_arxiv_paper(paper_id: str) -> Dict[str, object]:
    """
    Return one exact paper metadata record by arXiv id.

    Accepts either a canonical id (e.g. `2401.00001`) or a versioned id
    (e.g. `2401.00001v1`) and normalizes both to the same local lookup key.
    """
    record = _find_local_arxiv_record_by_id(paper_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"paper not found for paper_id={paper_id!r}")
    return {
        "type": "arxiv_paper_result",
        "paper": record,
    }


@app.get("/api/paper-text/{paper_id}")
async def api_paper_text(paper_id: str) -> Dict[str, object]:
    """
    Return full-text display payload for one paper from the local merged
    paper-text dataset, with pages rendered exactly when a local PDF is
    available and inferred from page_count otherwise.
    """
    payload = _paper_text_payload(paper_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"paper text not found for paper_id={paper_id!r}",
        )
    return payload


@app.post("/api/arxiv/download")
async def api_arxiv_download(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Download one or more Arxiv PDFs by id into `/arxiv/pdfs`.

    Expected payload:
        {
          "id": "..."?          # single id
          "ids": ["...", ...]?  # or list of ids
        }

    Returns a summary:
        {
          "type": "arxiv_download_result",
          "requested": N,
          "downloaded": K,
          "skipped_existing": M,
          "errors": [{"id": "...", "error": "..."}]
        }
    """
    single_id_any = payload.get("id")
    ids_any = payload.get("ids")

    ids: List[str] = []
    if isinstance(ids_any, list):
        ids.extend(str(x) for x in ids_any if x)
    if isinstance(single_id_any, str) and single_id_any:
        ids.append(single_id_any)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    norm_ids: List[str] = []
    for raw in ids:
        s = str(raw or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        norm_ids.append(s)

    if not norm_ids:
        raise HTTPException(status_code=400, detail="`id` or `ids` is required.")

    # Soft cap to avoid accidental huge batches from the UI.
    if len(norm_ids) > 100000:
        raise HTTPException(
            status_code=400,
            detail="too many ids requested for download; cap is 100000",
        )

    # Enforce a hard cap on *new* downloads per request so that a single
    # bulk operation cannot attempt to fetch an unbounded number of PDFs.
    # Existing PDFs do not count against this cap.
    MAX_NEW_DOWNLOADS = 1000

    downloaded = 0
    skipped_existing = 0
    skipped_due_to_cap = 0
    errors: List[Dict[str, str]] = []

    for pid in norm_ids:
        # Normalize id similarly to the downloader helper so we can
        # cheaply detect already-present PDFs without counting them
        # against the new-download cap.
        norm_id = str(pid or "").strip().split("/")[-1]
        if not norm_id:
            errors.append({"id": pid, "error": "invalid arxiv id"})
            continue

        if _find_local_arxiv_pdf(norm_id) is not None:
            skipped_existing += 1
            continue

        if downloaded >= MAX_NEW_DOWNLOADS:
            skipped_due_to_cap += 1
            continue

        try:
            changed = _download_arxiv_pdf(pid)
        except Exception as exc:
            errors.append({"id": pid, "error": str(exc)})
            continue
        if changed:
            downloaded += 1
        else:
            # Should be rare given the existence check above, but treat
            # non-changing downloads as "existing" for accounting.
            skipped_existing += 1

    return {
        "type": "arxiv_download_result",
        "requested": len(norm_ids),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "max_new_downloads": MAX_NEW_DOWNLOADS,
        "skipped_due_to_cap": skipped_due_to_cap,
        "errors": errors,
    }


@app.get("/api/algorithms")
async def api_list_algorithms(
    problem_id: Optional[str] = None,
    topic: Optional[str] = None,
    max_results: int = 200,
) -> Dict[str, Any]:
    """
    List algorithms from the local Algorithms Library under `/data/algorithms`.

    Behavior:
        - With no filters: return the first N algorithms from the library.
        - With filters: return algorithms matching the given problem_id/topic.

    Optional filters:
        - problem_id: only algorithms that list this problem_id.
        - topic: only algorithms whose topics/tags include this value.
    """
    # Clamp max_results into a safe range.
    try:
        max_results_int = int(max_results)
    except Exception:
        max_results_int = 200
    max_results_int = max(1, min(max_results_int, 200))

    pid = problem_id or None
    t = topic or None

    # No filters: stream the first N algorithms directly from the snapshot.
    if not pid and not t:
        results: List[Dict[str, Any]] = []
        for algo in iter_algorithms():
            results.append(
                {
                    "algo_id": algo.algo_id,
                    "names": algo.names,
                    "category": algo.category,
                    "problems": algo.problems,
                    "topics": algo.topics,
                    "time_complexity": algo.time_complexity,
                    "space_complexity": algo.space_complexity,
                    "properties": algo.properties,
                    "constraints": algo.constraints,
                    "notes": algo.notes,
                    "tags": algo.tags,
                }
            )
            if len(results) >= max_results_int:
                break
        return {"algorithms": results}

    # With filters: delegate to the shared search helper.
    results = algo_search_algorithms(
        "",
        problem_id=pid,
        topic=t,
        max_results=max_results_int,
    )
    return {"algorithms": results}


@app.get("/api/algorithms/problems/{problem_id}")
async def api_algorithm_problem(problem_id: str) -> Dict[str, Any]:
    """
    Return metadata for a single problem from the Algorithms Library.
    """
    pid = str(problem_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="`problem_id` is required.")

    for prob in iter_problems():
        if prob.problem_id == pid:
            return {
                "type": "algorithm_problem",
                "problem": {
                    "problem_id": prob.problem_id,
                    "names": prob.names,
                    "description": prob.description,
                    "topics": prob.topics,
                    "constraints": prob.constraints,
                    "notes": prob.notes,
                },
            }

    raise HTTPException(
        status_code=404,
        detail=f"problem_id={pid!r} not found in Algorithms Library",
    )


@app.get("/api/algorithms/implementations")
async def api_algorithm_implementations(
    algo_id: str,
    max_results: int = 50,
) -> Dict[str, Any]:
    """
    List concrete implementations for a given algorithm from the Algorithms Library.

    Query params:
        - algo_id: required algorithm identifier.
        - max_results: soft cap on number of implementations returned (default 50).
    """
    aid = str(algo_id or "").strip()
    if not aid:
        raise HTTPException(status_code=400, detail="`algo_id` is required.")

    try:
        max_results_int = int(max_results)
    except Exception:
        max_results_int = 50
    max_results_int = max(1, min(max_results_int, 200))

    results: List[Dict[str, Any]] = []
    for impl in iter_implementations():
        if impl.algo_id != aid:
            continue
        results.append(
            {
                "impl_id": impl.impl_id,
                "algo_id": impl.algo_id,
                "language": impl.language,
                "repo_id": impl.repo_id,
                "repo_root": impl.repo_root,
                "file_path": impl.file_path,
                "entry_symbol": impl.entry_symbol,
                "constraints": impl.constraints,
                "environment": impl.environment,
                "notes": impl.notes,
            }
        )
        if len(results) >= max_results_int:
            break

    return {
        "type": "algorithm_implementations",
        "algo_id": aid,
        "count": len(results),
        "results": results,
    }


@app.post("/api/algorithms/search")
async def api_algorithms_search(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Search the local Algorithms Library under `/data/algorithms`.

    Expected payload:
        {
          "query": "...",                # optional keyword, case-insensitive
          "problem_id": "sssp"?,         # optional problem_id filter
          "topic": "graphs"?,            # optional topic/tag filter
          "page": 1?,                    # optional 1-based page index (default 1)
          "page_size": 50?               # optional, default 50, max 50
          # `max_results` (legacy): if provided and page/page_size are omitted,
          #                         treated as an alias for page_size.
        }
    """
    query = str(payload.get("query") or "").strip()

    problem_id_any = payload.get("problem_id")
    problem_id: Optional[str] = None
    if isinstance(problem_id_any, str):
        problem_id = problem_id_any or None

    topic_any = payload.get("topic")
    topic: Optional[str] = None
    if isinstance(topic_any, str):
        topic = topic_any or None

    # Pagination: limit page size to 50, but allow arbitrarily many pages.
    page_raw = payload.get("page", 1)
    try:
        page = int(page_raw)
    except Exception:
        page = 1
    if page < 1:
        page = 1

    # For backward compatibility, fall back to legacy `max_results` when
    # page_size is not supplied.
    page_size_raw = (
        payload.get("page_size")
        if "page_size" in payload
        else payload.get("max_results", 50)
    )
    try:
        page_size = int(page_size_raw)
    except Exception:
        page_size = 50
    # Hard cap at 50 results per page.
    page_size = max(1, min(page_size, 50))

    # The underlying search helper does not support offsets, so we ask it
    # for results up to the end of the requested page and then slice.
    internal_max = page * page_size
    # Keep a defensive upper bound to avoid scanning excessively.
    internal_max = max(1, min(internal_max, 1000))

    all_results = algo_search_algorithms(
        query,
        problem_id=problem_id,
        topic=topic,
        max_results=internal_max,
    )

    start = (page - 1) * page_size
    end = start + page_size
    page_results = all_results[start:end]

    return {
        "type": "algorithm_search_result",
        "query": query,
        "page": page,
        "page_size": page_size,
        "count": len(page_results),
        "results": page_results,
    }


@app.get("/api/graph/{repo_id}")
async def api_graph(
    repo_id: str,
    max_nodes: int = 2000,
    max_edges: int = 4000,
) -> Dict[str, Any]:
    """
    Return a lightweight view of the exported program graph for a repo.

    This loads entities/edges from the JSONL exports under DEFAULT_EXPORT_ROOT
    and returns them in a shape suitable for client-side visualization.
    """
    # Clamp limits to stay within a safe, interactive range.
    max_nodes = max(1, min(int(max_nodes or 0), 5000))
    max_edges = max(1, min(int(max_edges or 0), 10000))

    export_root = Path(DEFAULT_EXPORT_ROOT)
    ent_path = export_root / repo_id / f"{repo_id}.entities.jsonl"
    edge_path = export_root / repo_id / f"{repo_id}.edges.jsonl"
    if not ent_path.is_file() or not edge_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"graph exports not found for repo_id={repo_id!r}",
        )

    # First pass: stream edges with a hard cap and collect the set of node ids
    # that appear in the sampled edges. This avoids materializing millions of
    # edges/nodes in memory or sending them to the client.
    node_ids: Set[str] = set()
    edges: List[Dict[str, Any]] = []
    try:
        with edge_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(edges) >= max_edges:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                src = str(row.get("src") or "").strip()
                dst = str(row.get("dst") or "").strip()
                if not src or not dst:
                    continue
                etype = row.get("type")
                eid = f"{src}->{dst}:{etype}" if etype else f"{src}->{dst}"
                node_ids.add(src)
                node_ids.add(dst)
                edges.append(
                    {
                        "id": eid,
                        "source": src,
                        "target": dst,
                        "type": etype,
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"failed to load edges for repo_id={repo_id!r}: {exc}",
        )

    # If we still have no nodes from edges (e.g., edge file empty), we will
    # fall back to sampling nodes directly from the entities file.
    if node_ids and len(node_ids) > max_nodes:
        # Trim to a stable subset of nodes and drop edges that fall outside.
        limited_ids: Set[str] = set(list(node_ids)[:max_nodes])
        node_ids = limited_ids
        edges = [
            e
            for e in edges
            if e["source"] in node_ids and e["target"] in node_ids
        ]

    # Second pass: stream entities and only materialize at most `max_nodes`,
    # either the specific ids we saw in the edge sample or, if there were no
    # edges, the first `max_nodes` entities.
    nodes: Dict[str, Dict[str, Any]] = {}
    try:
        with ent_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if node_ids and len(nodes) >= len(node_ids):
                    break
                if not node_ids and len(nodes) >= max_nodes:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                eid = str(row.get("id") or "").strip()
                if not eid:
                    continue
                if node_ids and eid not in node_ids:
                    continue
                if eid in nodes:
                    continue
                nodes[eid] = {
                    "id": eid,
                    "label": row.get("name") or eid,
                    "kind": row.get("kind"),
                    "uri": row.get("uri"),
                    "owner": row.get("owner"),
                }
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500,
            detail=f"failed to load entities for repo_id={repo_id!r}: {exc}",
        )

    # Ensure we only return edges that touch known nodes (defensive if exports diverge)
    realized_ids = set(nodes.keys())
    edges = [
        e for e in edges if e["source"] in realized_ids and e["target"] in realized_ids
    ]

    return {
        "repo_id": repo_id,
        "nodes": list(nodes.values()),
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_limit": max_nodes,
        "edge_limit": max_edges,
    }


@app.get("/api/source/{repo_id}")
async def api_source(
    repo_id: str,
    uri: str,
    context: int = 20,
    max_lines: int = 400,
) -> Dict[str, Any]:
    """
    Resolve an entity/program URI to its underlying artifact and return a
    code snippet from the repository on disk.

    This uses the in-process PythonRepoGraph backend via `open_repository`,
    so it does not rely on any extra data stored in the JSONL exports.
    """
    try:
        repo = open_repository(repo_id)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"repo_id not found or invalid: {repo_id!r} ({exc})",
        )

    graph = repo.graph
    try:
        anchor = graph.resolve(uri)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"failed to resolve uri: {exc}",
        )

    artifact_uri = anchor.artifact_uri
    span = anchor.span

    try:
        _pid, kind, resource, _ = parse_program_uri(artifact_uri)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"invalid artifact uri from graph: {artifact_uri!r} ({exc})",
        )
    if kind != "artifact":
        raise HTTPException(
            status_code=500,
            detail=f"resolved artifact kind is not 'artifact': {kind!r}",
        )

    root_path = repo.root_path
    abs_path = (root_path / resource).resolve()
    try:
        root_resolved = root_path.resolve()
        if not str(abs_path).startswith(str(root_resolved)):
            raise HTTPException(
                status_code=400,
                detail="resolved artifact path escapes repository root",
            )
    except HTTPException:
        raise
    except Exception:
        # Best-effort; if resolution fails we still proceed with abs_path check
        pass

    if not abs_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"artifact file not found: {abs_path}",
        )

    # Normalize context and max_lines to sane bounds.
    context = max(0, min(int(context or 0), 200))
    max_lines = max(1, min(int(max_lines or 0), 800))

    # Determine the primary span; if missing, default to whole file.
    if span is not None:
        a = int(span.start_line)
        b = int(span.end_line)
    else:
        # Count lines once to set bounds.
        total = 0
        try:
            with abs_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for _ in fh:
                    total += 1
        except Exception:
            total = 1
        a, b = 1, max(1, total)

    # Compute snippet window with context and max_lines cap.
    start = max(1, a - context)
    end = b + context
    # If we don't know file length yet, we'll clamp by max_lines during read.
    if end < start:
        end = start

    snippet_lines: List[Dict[str, Any]] = []
    try:
        with abs_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for idx, line in enumerate(fh, start=1):
                if idx < start:
                    continue
                if len(snippet_lines) >= max_lines:
                    break
                if idx > end:
                    break
                snippet_lines.append(
                    {
                        "line_no": idx,
                        "text": line.rstrip("\n"),
                    }
                )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"failed to read artifact file: {exc}",
        )

    rel_path = abs_path.relative_to(root_path)

    return {
        "repo_id": repo_id,
        "uri": uri,
        "artifact_uri": artifact_uri,
        "path": str(rel_path),
        "hash": anchor.hash,
        "span": {
            "start_line": a,
            "end_line": b,
        },
        "snippet": {
            "start_line": start,
            "end_line": start + len(snippet_lines) - 1 if snippet_lines else start,
            "lines": snippet_lines,
        },
    }
