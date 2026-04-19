"""
Dataset scaffolding for the 32-model substrate.
Supports lightweight sampling from:
- arXiv metadata (/data/arxiv/arxiv-metadata-oai-snapshot.json)
- PDFs (/arxiv/pdfs/{year}/) or structured shards under exports/pdfs_structured/
- repositories (/data/repositories)
Falls back to placeholders if sources are missing.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Iterator
import json
import random
import glob
import re
import gzip
import itertools
import hashlib

from models.shared.archetypes import get_archetype
from models.shared.graph_data import (
    load_graph_samples,
    graph_sample_to_text,
    load_paper_graph_samples,
    paper_sample_to_text,
)
from models.shared.pdf_utils import extract_pdf_text


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_SEARCH_ROOTS = [REPO_ROOT / "exports/arxiv_pdfs", Path("/arxiv/pdfs")]


def _candidate_data_paths(path_like: Path | str) -> List[Path]:
    raw = Path(path_like)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(REPO_ROOT / raw)
        if not str(raw).startswith("models/"):
            candidates.append(REPO_ROOT / "models" / raw)
    seen: set[Path] = set()
    out: List[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _resolve_data_dir(path_like: Path | str, pattern: str) -> Path:
    for candidate in _candidate_data_paths(path_like):
        if candidate.is_dir() and any(candidate.glob(pattern)):
            return candidate
    for candidate in _candidate_data_paths(path_like):
        if candidate.exists():
            return candidate
    return _candidate_data_paths(path_like)[0]


def _resolve_data_file(path_like: Path | str) -> Path:
    for candidate in _candidate_data_paths(path_like):
        if candidate.is_file():
            return candidate
    for candidate in _candidate_data_paths(path_like):
        if candidate.exists():
            return candidate
    return _candidate_data_paths(path_like)[0]


def load_manifest(path: str = "/data/arxiv/arxiv-metadata-oai-snapshot.json") -> Dict[str, Any]:
    """Load the arXiv manifest if present; fall back to the snapshot."""
    manifest_path = Path(path)
    if manifest_path.exists():
        # Try JSON or JSONL gzip
        if manifest_path.suffix == ".gz":
            with gzip.open(manifest_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with manifest_path.open("r", encoding="utf-8") as f:
            # If this is JSONL, wrap into entries
            try:
                obj = json.load(f)
                return obj
            except Exception:
                pass
    snapshot = Path("/data/arxiv/arxiv-metadata-oai-snapshot.json")
    if not snapshot.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path} and snapshot missing at {snapshot}")

    entries: List[Dict[str, Any]] = []
    with snapshot.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            entries.append(obj)
            if idx >= 50000:  # cap for speed
                break
    return {"entries": entries}


def _sample_metadata(manifest: Dict[str, Any], max_samples: int, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = manifest.get("entries") or manifest.get("papers") or []
    if not entries:
        return []
    years_filter = filters.get("years")
    categories_filter = set(filters.get("categories") or [])
    random.seed(42)
    if years_filter and len(years_filter) == 2:
        y0, y1 = years_filter

        def _in_year(entry):
            year = None
            if "update_date" in entry:
                try:
                    year = int(str(entry["update_date"])[:4])
                except Exception:
                    year = None
            if year is None and "versions" in entry and entry["versions"]:
                try:
                    year = int(str(entry["versions"][-1].get("created", ""))[:4])
                except Exception:
                    year = None
            return year is None or (y0 <= year <= y1)

        entries = [e for e in entries if _in_year(e)]
    if categories_filter:

        def _has_cat(entry):
            cats = entry.get("categories") or entry.get("primary_category") or ""
            if isinstance(cats, str):
                return any(cat in cats for cat in categories_filter)
            try:
                return any(cat in categories_filter for cat in cats)
            except Exception:
                return False

        entries = [e for e in entries if _has_cat(e)]
    random.shuffle(entries)
    return entries[:max_samples]


def _iter_pdf_paths(filters: Dict[str, Any]) -> Iterable[str]:
    if not any(base.exists() for base in PDF_SEARCH_ROOTS):
        return []

    def _infer_year(path: Path) -> int | None:
        stem = path.stem
        if len(stem) >= 4 and stem[:4].isdigit():
            yy = int(stem[:2])
            return 1900 + yy if yy >= 90 else 2000 + yy
        return None

    years = filters.get("years")
    if years and len(years) == 2:
        y0, y1 = int(years[0]), int(years[1])
        seen: set[Path] = set()
        for base in PDF_SEARCH_ROOTS:
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.pdf")):
                if not p.is_file() or p in seen:
                    continue
                year = _infer_year(p)
                if year is None or year < y0 or year > y1:
                    continue
                seen.add(p)
                yield str(p)
    else:
        seen: set[Path] = set()
        for base in PDF_SEARCH_ROOTS:
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.pdf")):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    yield str(p)


def _iter_structured_pdfs(shard_dir: Path, max_samples: int) -> Iterable[Dict[str, Any]]:
    """Yield records from structured PDF shards if present."""
    shard_dir = _resolve_data_dir(shard_dir, "pdf_structured_*.jsonl")
    if not shard_dir.exists():
        return []
    count = 0
    shard_paths = sorted(shard_dir.glob("pdf_structured_*.jsonl"))
    for shard in shard_paths:
        try:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    if max_samples and count >= max_samples:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    yield obj
                    count += 1
        except Exception:
            continue


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> Iterator[Tuple[str, int]]:
    if chunk_chars <= 0:
        yield text, 0
        return
    step = max(1, chunk_chars - overlap)
    for start in range(0, len(text), step):
        yield text[start : start + chunk_chars], start


def _sample_pdfs(max_samples: int, filters: Dict[str, Any], chunk_chars: int = 8000, overlap: int = 400) -> List[Dict[str, Any]]:
    paths = list(_iter_pdf_paths(filters))
    random.seed(42)
    random.shuffle(paths)
    samples: List[Dict[str, Any]] = []
    for p in paths:
        if len(samples) >= max_samples:
            break
        text = extract_pdf_text(p, max_chars=chunk_chars * 4)
        for chunk, offset in _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap):
            if len(samples) >= max_samples:
                break
            samples.append({"text": chunk, "pdf_path": p, "offset": offset, "label": 0})
    return samples


def _load_structured_pdfs(max_samples: int) -> List[Dict[str, Any]]:
    """Load structured PDF tokens from exports/pdfs_structured shards."""
    base = _resolve_data_dir("exports/pdfs_structured", "pdf_structured_*.jsonl")
    if not base.exists():
        return []
    shards = sorted(base.glob("pdf_structured_*.jsonl"))
    samples: List[Dict[str, Any]] = []
    random.seed(42)
    random.shuffle(shards)
    for shard in shards:
        try:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    samples.append(obj)
                    if len(samples) >= max_samples:
                        return samples
        except Exception:
            continue
    return samples[:max_samples]


def _load_paper_repo_alignment(max_samples: int, path: Path = Path("exports/paper_repo_align.jsonl")) -> List[Dict[str, Any]]:
    """Load precomputed paper↔repo alignment pairs."""
    path = _resolve_data_file(path)
    if not path.exists():
        return []
    samples: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    samples.append(obj)
    except Exception:
        return []
    return samples[:max_samples]


def _load_corpus_split(
    split: str,
    max_samples: int,
    base_dir: Path = Path("exports/corpus"),
) -> List[Dict[str, Any]]:
    """
    Load prebuilt corpus shards from exports/corpus created by models.scripts.build_corpus.

    Layout:
      exports/corpus/
        repos/repo_*.jsonl
        papers/paper_*.jsonl
        pairs/pair_*.jsonl

    Returns a list of simple dict records shaped to match the rest of build_dataset.
    """
    base_dir = _resolve_data_dir(base_dir, "*.jsonl")
    split_dir = base_dir / split
    if not split_dir.exists():
        return []
    records: List[Dict[str, Any]] = []
    # Use a lightweight per-split hash set to avoid trivial duplicates.
    seen_hashes: set[str] = set()

    def _hash_payload(payload: str) -> str:
        h = hashlib.sha1()
        h.update(payload.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    pattern = {
        "repos": "repo_*.jsonl",
        "papers": "paper_*.jsonl",
        "pairs": "pair_*.jsonl",
    }.get(split, "*.jsonl")

    for shard in sorted(split_dir.glob(pattern)):
        try:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    if len(records) >= max_samples:
                        return records
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    source = obj.get("source") or ""
                    if split == "repos" or source == "repo_chunk":
                        text = str(obj.get("text") or "").strip()
                        if not text:
                            continue
                        hval = _hash_payload(text)
                        if hval in seen_hashes:
                            continue
                        seen_hashes.add(hval)
                        meta = obj.get("meta") or {}
                        records.append(
                            {
                                "text": text,
                                "label": 0,
                                "repo_id": meta.get("repo_id"),
                                "path": meta.get("path"),
                                "offset": meta.get("offset", 0),
                            }
                        )
                    elif split == "papers" or source == "paper_chunk":
                        text = str(obj.get("text") or "").strip()
                        if not text:
                            continue
                        hval = _hash_payload(text)
                        if hval in seen_hashes:
                            continue
                        seen_hashes.add(hval)
                        meta = obj.get("meta") or {}
                        records.append(
                            {
                                "text": text,
                                "label": 0,
                                "pdf_path": meta.get("pdf_path"),
                            }
                        )
                    elif split == "pairs" or source == "paper_repo_pair":
                        paper = str(obj.get("paper_text") or "").strip()
                        repo = str(obj.get("repo_text") or "").strip()
                        if not paper or not repo:
                            continue
                        label = int(obj.get("label") or 0)
                        score = float(obj.get("score") or 0.0)
                        records.append(
                            {
                                "text_a": paper,
                                "text_b": repo,
                                "label": label,
                                "score": score,
                            }
                        )
        except Exception:
            continue

    return records[:max_samples]


def _sample_repos(max_samples: int) -> List[Dict[str, Any]]:
    repo_root = Path("/data/repositories")
    if not repo_root.exists():
        return []
    repo_dirs = [p for p in repo_root.iterdir() if p.is_dir()]
    random.seed(42)
    random.shuffle(repo_dirs)
    repo_dirs = repo_dirs[:max_samples]
    samples = []
    for repo in repo_dirs:
        samples.append({"text": f"REPO_PATH::{repo}", "label": 0})
    return samples


def _sample_repo_files(
    max_samples: int,
    extensions: Tuple[str, ...] = (".py", ".md", ".txt"),
    chunk_chars: int = 4000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Sample source files from repos for QA/mutation tasks with chunking."""
    repo_root = Path("/data/repositories")
    if not repo_root.exists():
        return []
    export_manifest = Path("/data/repository_library/exports/_manifest.json")
    repo_dirs: List[Path] = []
    if export_manifest.exists():
        try:
            manifest = json.load(export_manifest.open())
            repos = manifest.get("repos") or manifest
            for meta in repos.values():
                root = meta.get("repo_root")
                if root:
                    repo_dirs.append(Path(root))
        except Exception:
            repo_dirs = [p for p in repo_root.iterdir() if p.is_dir()]
    else:
        repo_dirs = [p for p in repo_root.iterdir() if p.is_dir()]

    random.seed(42)
    random.shuffle(repo_dirs)
    files: List[Path] = []
    for repo_dir in repo_dirs:
        if len(files) >= max_samples:
            break
        if not repo_dir.is_dir():
            continue
        for ext in extensions:
            for p in repo_dir.rglob(f"*{ext}"):
                if len(files) >= max_samples:
                    break
                try:
                    size = p.stat().st_size
                except Exception:
                    continue
                if size == 0 or size > 1_000_000:  # skip empty or huge
                    continue
                files.append(p)
            if len(files) >= max_samples:
                break

    samples: List[Dict[str, Any]] = []
    for f in files:
        if len(samples) >= max_samples:
            break
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            try:
                text = f.read_bytes().decode("latin-1", errors="ignore")
            except Exception:
                text = ""
        for chunk, offset in _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap):
            if len(samples) >= max_samples:
                break
            samples.append({"path": str(f), "code": chunk, "offset": offset})
    return samples


def _summarize_code(code: str) -> str:
    """Crude code summary: pull docstring or first function/class signatures."""
    doc_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if doc_match:
        doc = doc_match.group(1).strip()
        if doc:
            return doc[:400]
    sigs = re.findall(r"^(def\s+[a-zA-Z_][a-zA-Z0-9_]*\(.*\):)", code, re.MULTILINE)
    if not sigs:
        sigs = re.findall(r"^(class\s+[A-Z][a-zA-Z0-9_]*\(?.*\)?:)", code, re.MULTILINE)
    if sigs:
        return " ".join(sigs[:5])[:400]
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    return " ".join(lines[:5])[:400]


def _format_text_from_entry(entry: Dict[str, Any]) -> str:
    title = entry.get("title") or ""
    abstract = entry.get("abstract") or ""
    categories = entry.get("categories") or entry.get("primary_category") or ""
    authors = ", ".join(entry.get("authors", [])) if isinstance(entry.get("authors"), list) else entry.get("authors", "")
    return "\n".join([title, abstract, categories, authors])


def _build_contrastive(manifest_entries: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if not manifest_entries:
        return []
    random.seed(42)
    samples = []
    for i in range(0, min(len(manifest_entries), max_samples)):
        a = manifest_entries[i % len(manifest_entries)]
        b = manifest_entries[(i + 1) % len(manifest_entries)]
        samples.append(
            {
                "text_a": _format_text_from_entry(a),
                "text_b": _format_text_from_entry(b),
                "label": 1,
            }
        )
    return samples


def _build_classifier(manifest_entries: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if not manifest_entries:
        return []
    samples = []
    for e in manifest_entries[:max_samples]:
        text = _format_text_from_entry(e)
        cat = e.get("categories") or e.get("primary_category") or "unknown"
        samples.append({"text": text, "label": cat})
    return samples


def _build_generative(manifest_entries: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if not manifest_entries:
        return []
    samples = []
    for e in manifest_entries[:max_samples]:
        text = _format_text_from_entry(e)
        target = e.get("abstract") or ""
        samples.append({"text": text, "target": target})
    return samples


def _build_qa_pairs(manifest_entries: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    """Construct simple QA pairs from metadata."""
    if not manifest_entries:
        return []
    samples = []
    for e in manifest_entries[:max_samples]:
        q = f"What is the main topic of the paper titled '{e.get('title','')}'?"
        a = e.get("abstract") or e.get("categories") or "Not specified."
        samples.append({"question": q, "context": e.get("abstract") or "", "target": a})
    return samples


def _build_mutation_targets(code_samples: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    """Create simple mutation targets by echoing code with TODO markers."""
    out: List[Dict[str, Any]] = []
    for cs in code_samples[:max_samples]:
        ctx = cs.get("code", "")
        failure = "Refactor or fix potential issues."
        summary = _summarize_code(ctx)
        target = (
            f"# Suggested patch for {cs.get('path','')}\n"
            f"# Summary: {summary}\n"
            "# Identify potential bugs or anti-patterns and refactor:\n"
            "# 1) Add missing error handling\n"
            "# 2) Fix any obvious typos or logic issues\n"
            "# 3) Ensure functions are pure and side-effect free where applicable\n"
            "# 4) Add concise docstring/tests if missing\n"
            "# 5) Keep changes minimal and scoped\n"
            "# Return the patched code below.\n"
            f"{ctx}\n"
        )
        out.append({"context": ctx, "failure": failure, "target": target})
    return out


def build_dataset(config: Dict[str, Any]):
    """
    Minimal dataset builder keyed off dataset.sources.
    Returns mixed samples; objective-specific shaping is left to downstream collators.
    """
    construction = config.get("dataset", {}).get("construction", {}) if isinstance(config.get("dataset", {}), dict) else {}
    max_samples = construction.get("max_samples", 32)
    chunk_chars_pdf = construction.get("chunk_chars_pdf", 8000)
    chunk_overlap_pdf = construction.get("chunk_overlap_pdf", 400)
    chunk_chars_code = construction.get("chunk_chars_code", 4000)
    chunk_overlap_code = construction.get("chunk_overlap_code", 200)
    structured_pdf_dir = _resolve_data_dir(
        construction.get("structured_pdf_dir") or "exports/pdfs_structured",
        "pdf_structured_*.jsonl",
    )
    alignment_path = _resolve_data_file(construction.get("alignment_path") or "exports/paper_repo_align.jsonl")
    sources = config.get("dataset", {}).get("sources", [])
    archetype = get_archetype(config.get("model_id", "UNK")) or {}
    archetype_name = archetype.get("archetype", "generative")
    model_id = config.get("model_id", "UNK")

    manifest_entries: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    if archetype_name == "graph":
        try:
            graph_samples = load_graph_samples(max_samples=max_samples // 2)
            paper_samples = load_paper_graph_samples(max_samples=max_samples - len(graph_samples))
            samples.extend([graph_sample_to_text(gs) for gs in graph_samples])
            samples.extend([paper_sample_to_text(ps) for ps in paper_samples])
        except Exception:
            pass
        return samples[:max_samples] if samples else [{"text": "graph placeholder", "label": 0}]

    try:
        filters = config.get("dataset", {}).get("filters", {}) if isinstance(config.get("dataset", {}), dict) else {}
        if "arxiv_metadata" in sources:
            manifest = load_manifest()
            manifest_entries = _sample_metadata(manifest, max_samples, filters)
        if archetype_name == "contrastive" and manifest_entries:
            samples.extend(_build_contrastive(manifest_entries, max_samples))
        elif archetype_name == "classifier" and manifest_entries:
            samples.extend(_build_classifier(manifest_entries, max_samples))
        elif archetype_name == "generative" and manifest_entries:
            if model_id in {"R3", "R5", "C1"}:
                samples.extend(_build_qa_pairs(manifest_entries, max_samples))
            else:
                samples.extend(_build_generative(manifest_entries, max_samples))
        # Alignment-aware path for paper↔repo fusion models.
        if model_id in {"C1", "C2", "C6"}:
            align_pairs = _load_paper_repo_alignment(max_samples, alignment_path)
            if align_pairs:
                for pair in align_pairs:
                    paper = pair.get("paper_text") or pair.get("paper") or pair.get("text_a") or ""
                    repo = pair.get("repo_text") or pair.get("repo") or pair.get("text_b") or ""
                    label = pair.get("label", 1)
                    if model_id == "C1":
                        # Generative: paper -> code/repo text
                        target = pair.get("target") or repo
                        samples.append({"text": paper, "target": target, "label": label})
                    else:
                        samples.append({"text_a": paper, "text_b": repo, "label": label})
                # If alignment pairs exist, skip the generic PDF/repo mixing for these models.
                return samples[:max_samples]
        # Supplemental sources: prefer structured shards if available.
        # Prefer explicit corpus shards when requested; they are the unified
        # view built by models.scripts.build_corpus and scale better than
        # on-the-fly sampling for large runs.
        if "corpus_repos" in sources:
            samples.extend(_load_corpus_split("repos", max_samples))
        if "corpus_papers" in sources:
            samples.extend(_load_corpus_split("papers", max_samples))
        if "corpus_pairs" in sources:
            samples.extend(_load_corpus_split("pairs", max_samples))
        # Fallback to raw structured PDFs / direct PDF extraction when corpus
        # shards are not used.
        if "arxiv_pdfs_structured" in sources or "arxiv_pdfs" in sources:
            structured_records = list(_iter_structured_pdfs(structured_pdf_dir, max_samples))
            if structured_records:
                for rec in structured_records:
                    tokens = rec.get("tokens")
                    if tokens:
                        samples.append({"tokens": tokens, "pdf_path": rec.get("pdf_path"), "label": 0})
                    else:
                        text = rec.get("text") or rec.get("content") or ""
                        samples.append({"text": text, "pdf_path": rec.get("pdf_path"), "label": 0})
            else:
                pdf_samples = _sample_pdfs(max_samples, filters, chunk_chars=chunk_chars_pdf, overlap=chunk_overlap_pdf)
                samples.extend(pdf_samples)
        if "github_repos" in sources:
            samples.extend(_sample_repos(max_samples))
        if model_id in {"R3", "R5"}:
            code_samples = _sample_repo_files(max_samples, chunk_chars=chunk_chars_code, overlap=chunk_overlap_code)
            if model_id == "R3":
                for cs in code_samples:
                    question = f"What does the code in {cs['path']} do?"
                    summary = _summarize_code(cs["code"])
                    answer = f"File: {cs['path']}\nSummary: {summary}"
                    samples.append({"question": question, "context": cs["code"], "target": answer})
            elif model_id == "R5":
                samples.extend(_build_mutation_targets(code_samples, max_samples))
    except Exception:
        pass

    if not samples:
        model_id = config.get("model_id", "UNK")
        samples = [{"text": f"placeholder sample for {model_id}", "label": 0} for _ in range(min(max_samples, 32))]

    return samples[:max_samples]
