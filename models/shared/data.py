"""
Dataset scaffolding for the 32-model substrate.
Supports lightweight sampling from:
- arXiv metadata (/data/arxiv/arxiv-metadata-oai-snapshot.json)
- parquet-backed paper text datasets (/arxiv/huggingface/paper_text_1m_dedup_v1)
- PDFs (/arxiv/pdfs/{year}/) or structured shards under exports/pdfs_structured/
- repositories (/data/repositories)
Falls back to placeholders if sources are missing.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable, Iterator
import json
import random
import glob
import re
import gzip
import itertools
import hashlib

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pq = None

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
DEFAULT_PAPER_DATASET_DIR = Path("/arxiv/huggingface/paper_text_1m_dedup_v1")
DEFAULT_PAPER_UNIVERSE_DIR = REPO_ROOT / "exports/_paper_universe"
UNLIMITED_SAMPLES = 10**18


def _resolve_max_samples(raw_value: Any, default: int = 32) -> int:
    try:
        value = int(raw_value if raw_value is not None else default)
    except Exception:
        value = default
    return UNLIMITED_SAMPLES if value <= 0 else value


def _limit_samples(samples: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if max_samples >= UNLIMITED_SAMPLES:
        return samples
    return samples[:max_samples]


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


def _paper_parquet_paths(dataset_dir: Path | str) -> List[Path]:
    dataset_dir = _resolve_data_dir(dataset_dir, "*.parquet")
    if dataset_dir.is_file():
        return [dataset_dir]
    paths = sorted(dataset_dir.glob("train_*.parquet"))
    if not paths:
        paths = sorted(dataset_dir.glob("*.parquet"))
    return [p for p in paths if p.is_file()]


def _paper_row_year(row: Dict[str, Any]) -> int | None:
    year = row.get("year")
    try:
        if year is not None:
            return int(year)
    except Exception:
        pass
    update_date = str(row.get("update_date") or "").strip()
    if len(update_date) >= 4 and update_date[:4].isdigit():
        return int(update_date[:4])
    paper_id = str(row.get("canonical_paper_id") or row.get("paper_id") or "").strip()
    if len(paper_id) >= 4 and paper_id[:4].isdigit():
        return 2000 + int(paper_id[:2])
    return None


def _normalize_paper_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    categories = row.get("categories") or row.get("primary_category") or ""
    if isinstance(categories, (list, tuple)):
        categories_text = " ".join(str(item or "").strip() for item in categories if str(item or "").strip())
    else:
        categories_text = str(categories or "").strip()
    authors = row.get("authors") or ""
    if isinstance(authors, (list, tuple)):
        authors_text = ", ".join(str(item or "").strip() for item in authors if str(item or "").strip())
    else:
        authors_text = str(authors or "").strip()
    normalized["title"] = str(row.get("title") or "").strip()
    normalized["abstract"] = str(row.get("abstract") or "").strip()
    normalized["text"] = str(row.get("text") or "").strip()
    normalized["authors"] = authors_text
    normalized["categories"] = categories_text
    if not normalized.get("primary_category") and categories_text:
        normalized["primary_category"] = categories_text.split()[0]
    return normalized


def _sample_paper_text_parquet(
    max_samples: int,
    filters: Dict[str, Any],
    dataset_dir: Path | str = DEFAULT_PAPER_DATASET_DIR,
) -> List[Dict[str, Any]]:
    if pq is None:
        return []
    paths = _paper_parquet_paths(dataset_dir)
    if not paths:
        return []
    years_filter = filters.get("years")
    categories_filter = set(filters.get("categories") or [])
    requested_columns = [
        "paper_id",
        "canonical_paper_id",
        "paper_version",
        "pdf_path",
        "title",
        "abstract",
        "authors",
        "categories",
        "primary_category",
        "license",
        "update_date",
        "year",
        "metadata_found",
        "text",
        "text_source",
        "text_is_partial",
        "text_char_count",
        "token_count",
        "page_count",
    ]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception:
            continue
        available_columns = [col for col in requested_columns if col in parquet_file.schema.names]
        for batch in parquet_file.iter_batches(columns=available_columns, batch_size=1024):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    continue
                normalized = _normalize_paper_row(row)
                if years_filter and len(years_filter) == 2:
                    year = _paper_row_year(normalized)
                    if year is not None and not (int(years_filter[0]) <= year <= int(years_filter[1])):
                        continue
                if categories_filter:
                    categories = normalized.get("categories") or normalized.get("primary_category") or ""
                    if not any(cat in str(categories) for cat in categories_filter):
                        continue
                rows.append(normalized)
                if len(rows) >= max_samples:
                    return rows
    return rows[:max_samples]


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


def _entry_paper_id(entry: Dict[str, Any]) -> str:
    return str(
        entry.get("canonical_paper_id")
        or entry.get("paper_id")
        or entry.get("id")
        or entry.get("entry_id")
        or ""
    ).strip()


def _compose_full_paper_text(entry: Dict[str, Any]) -> str:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstract") or "").strip()
    body = str(entry.get("text") or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip() or _format_text_from_entry(entry)


_KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./+-]{2,}")
_KEYWORD_STOPWORDS = {
    "abstract",
    "analysis",
    "approach",
    "based",
    "between",
    "data",
    "demonstrate",
    "evaluate",
    "experiments",
    "figure",
    "framework",
    "improve",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "present",
    "problem",
    "propose",
    "results",
    "section",
    "show",
    "study",
    "system",
    "that",
    "their",
    "table",
    "task",
    "tasks",
    "technique",
    "this",
    "using",
    "approaches",
    "with",
}

_CATEGORY_LABELS = {
    "cs.AI": "artificial intelligence",
    "cs.CL": "natural language processing",
    "cs.CV": "computer vision",
    "cs.CY": "computers and society",
    "cs.FL": "formal languages",
    "cs.HC": "human-computer interaction",
    "cs.LG": "machine learning",
    "cs.MA": "multi-agent systems",
    "cs.RO": "robotics",
    "cond-mat.mes-hall": "mesoscopic condensed matter",
    "physics.ins-det": "instrumentation",
    "math.CO": "combinatorics",
    "math.AP": "analysis of PDEs",
    "stat.ML": "statistical machine learning",
}


def _paper_body_text(entry: Dict[str, Any]) -> str:
    body = str(entry.get("text") or "").strip()
    if body:
        return body
    abstract = str(entry.get("abstract") or "").strip()
    title = str(entry.get("title") or "").strip()
    return "\n\n".join(part for part in [title, abstract] if part).strip() or _format_text_from_entry(entry)


def _paper_body_chunks(
    entry: Dict[str, Any],
    *,
    chunk_chars: int,
    overlap: int,
    max_chunks: int | None = None,
) -> List[Tuple[str, int]]:
    chunks: List[Tuple[str, int]] = []
    for chunk, offset in _chunk_text(
        _paper_body_text(entry),
        chunk_chars=max(512, int(chunk_chars or 0)),
        overlap=max(0, int(overlap or 0)),
    ):
        normalized = str(chunk or "").strip()
        if not normalized:
            continue
        chunks.append((normalized, offset))
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks


def _split_sentences(text: str, max_sentences: int) -> List[str]:
    sentences = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", text or "") if seg.strip()]
    return sentences[:max(1, max_sentences)]


def _paper_keyword_target(row: Dict[str, Any], max_keywords: int = 8) -> str:
    weighted_terms: Counter[str] = Counter()

    def _add_terms(text: str, weight: float, *, include_bigrams: bool) -> None:
        tokens = [
            token.lower()
            for token in _KEYWORD_TOKEN_RE.findall(text or "")
            if len(token) >= 4 and token.lower() not in _KEYWORD_STOPWORDS and not token.lower().startswith("cs.")
        ]
        if not tokens:
            return
        for token in tokens:
            weighted_terms[token] += weight
        if include_bigrams:
            for left, right in zip(tokens, tokens[1:]):
                if left == right:
                    continue
                phrase = f"{left} {right}"
                weighted_terms[phrase] += weight + 0.35

    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    body = str(row.get("text") or "").strip()
    _add_terms(title, 3.0, include_bigrams=True)
    _add_terms(abstract, 2.0, include_bigrams=True)
    _add_terms(body[:4000], 1.0, include_bigrams=False)

    selected: List[str] = []
    seen_tokens: set[str] = set()
    for term, _score in weighted_terms.most_common(max(max_keywords * 4, 16)):
        parts = term.split()
        if any(part in seen_tokens for part in parts):
            continue
        selected.append(term)
        seen_tokens.update(parts)
        if len(selected) >= max_keywords:
            break

    if not selected:
        fallback_text = title or abstract or body
        fallback_tokens = [token.lower() for token in _KEYWORD_TOKEN_RE.findall(fallback_text or "") if len(token) >= 4]
        selected = fallback_tokens[:max_keywords]
    return ", ".join(selected[:max_keywords]).strip() or (title or abstract or "keyword")


def _paper_domain_label(row: Dict[str, Any]) -> str:
    raw_categories = str(row.get("categories") or row.get("primary_category") or "").strip()
    categories = [cat.strip() for cat in re.split(r"[\s,]+", raw_categories) if cat.strip()]
    labels = [_CATEGORY_LABELS.get(cat, cat.replace(".", " ")) for cat in categories[:2]]
    if labels:
        return " and ".join(labels)
    return "research"


def _paper_method_summary_prompt(row: Dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    categories = str(row.get("categories") or row.get("primary_category") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    parts = [
        "Create a concise research-library card for this paper.",
        "Use the format: Problem / Method / Evidence / Library use / Tags.",
    ]
    if title:
        parts.append(f"TITLE:\n{title}")
    if categories:
        parts.append(f"CATEGORIES:\n{categories}")
    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}")
    if len(parts) <= 2:
        parts.append(_format_text_from_entry(row))
    return "\n\n".join(parts).strip()


def _clean_library_sentence(sentence: str, *, max_words: int = 34) -> str:
    text = re.sub(r"\s+", " ", str(sentence or "")).strip()
    text = re.sub(r"^(in|this)\s+paper\s*,?\s*", "", text, flags=re.IGNORECASE)
    leading_verb = re.match(r"^we\s+(propose|present|introduce|develop|describe)\s+(.+)$", text, flags=re.IGNORECASE)
    if leading_verb:
        verb = {
            "propose": "Proposes",
            "present": "Presents",
            "introduce": "Introduces",
            "develop": "Develops",
            "describe": "Describes",
        }.get(leading_verb.group(1).lower(), leading_verb.group(1))
        text = f"{verb} {leading_verb.group(2)}"
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:") + "..."
    return text.rstrip()


def _select_sentence_by_terms(sentences: List[str], terms: Tuple[str, ...], fallback_index: int = 0) -> str:
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in terms):
            return sentence
    if sentences:
        if fallback_index < 0:
            idx = max(0, len(sentences) + fallback_index)
        else:
            idx = min(fallback_index, len(sentences) - 1)
        return sentences[idx]
    return ""


def _paper_method_summary_target(row: Dict[str, Any]) -> str:
    abstract = str(row.get("abstract") or "").strip()
    full_text = str(row.get("text") or "").strip()
    title = str(row.get("title") or "").strip()
    source = "\n".join(part for part in [abstract, full_text[:8000]] if part).strip()
    sentences = _split_sentences(source, max_sentences=18)
    keywords = [kw.strip() for kw in _paper_keyword_target(row, max_keywords=6).split(",") if kw.strip()]
    domain = _paper_domain_label(row)

    problem_sentence = _select_sentence_by_terms(
        sentences,
        (
            "challenge",
            "problem",
            "objective",
            "aim",
            "goal",
            "need",
            "difficult",
            "limitation",
            "address",
        ),
        fallback_index=0,
    )
    method_sentence = _select_sentence_by_terms(
        sentences,
        (
            "we propose",
            "we present",
            "our method",
            "this paper introduces",
            "we introduce",
            "we develop",
            "we describe",
            "based on",
            "uses",
            "using",
        ),
        fallback_index=1,
    )
    evidence_sentence = _select_sentence_by_terms(
        sentences,
        (
            "results show",
            "we show",
            "experiments show",
            "we achieve",
            "outperform",
            "demonstrate",
            "improves",
            "evaluation",
            "validated",
        ),
        fallback_index=-1,
    )

    problem = _clean_library_sentence(problem_sentence, max_words=28) or (title or "Research problem not specified")
    method = _clean_library_sentence(method_sentence, max_words=34) or "Method details are not specified in the available text"
    evidence = _clean_library_sentence(evidence_sentence, max_words=30)
    if evidence and evidence == problem:
        evidence = "Evidence is not specified in the available abstract"
    evidence = evidence or "Evidence is not specified in the available abstract"
    keyword_text = ", ".join(keywords[:6]) or domain
    library_use = f"Useful for researchers browsing {domain} work related to {', '.join(keywords[:3]) or title or 'this topic'}."

    return "\n".join(
        [
            f"Problem: {problem}",
            f"Method: {method}",
            f"Evidence: {evidence}",
            f"Library use: {library_use}",
            f"Tags: {keyword_text}",
        ]
    ).strip()


def _paper_retrieval_query(row: Dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    if title and abstract:
        return f"TITLE:\n{title}\nABSTRACT:\n{abstract}"
    if title:
        return f"TITLE:\n{title}"
    if abstract:
        return f"ABSTRACT:\n{abstract}"
    return _format_text_from_entry(row)


def _metadata_embedding_query(entry: Dict[str, Any]) -> str:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstract") or "").strip()
    if title and abstract:
        return f"TITLE:\n{title}\nABSTRACT:\n{abstract}"
    if title:
        return f"TITLE:\n{title}"
    if abstract:
        return f"ABSTRACT:\n{abstract}"
    return _format_text_from_entry(entry)


def _metadata_embedding_doc(entry: Dict[str, Any]) -> str:
    title = str(entry.get("title") or "").strip()
    abstract = str(entry.get("abstract") or "").strip()
    categories = str(entry.get("categories") or entry.get("primary_category") or "").strip()
    authors = entry.get("authors") or ""
    if isinstance(authors, (list, tuple)):
        authors_text = ", ".join(str(item or "").strip() for item in authors if str(item or "").strip())
    else:
        authors_text = str(authors or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if categories:
        parts.append(f"Categories: {categories}")
    if authors_text:
        parts.append(f"Authors: {authors_text}")
    return "\n".join(parts).strip() or _format_text_from_entry(entry)


def _paper_document_view(row: Dict[str, Any], *, max_chars: int = 6000) -> str:
    text = _compose_full_paper_text(row)
    return text[: max(512, int(max_chars or 0))].strip()


def _build_paper_method_summary_samples(rows: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in rows[:max_samples]:
        target = _paper_method_summary_target(row)
        prompt = _paper_method_summary_prompt(row)
        samples.append({"text": prompt, "target": target})
    return samples

def _paper_problem_target(row: Dict[str, Any]) -> str:
    abstract = str(row.get("abstract") or "").strip()
    full_text = str(row.get("text") or "").strip()
    sentences = _split_sentences(abstract or full_text, max_sentences=2)
    return " ".join(sentences).strip() or abstract or _format_text_from_entry(row)


def _paper_results_target(row: Dict[str, Any]) -> str:
    hint_terms = (
        "results show",
        "we show",
        "our experiments",
        "experiments show",
        "we achieve",
        "outperform",
        "demonstrate",
        "improves",
        "improvement",
        "state-of-the-art",
    )
    abstract = str(row.get("abstract") or "").strip()
    full_text = str(row.get("text") or "").strip()
    target_sentences: List[str] = []
    for sentence in _split_sentences(full_text or abstract, max_sentences=12):
        if any(term in sentence.lower() for term in hint_terms):
            target_sentences.append(sentence)
        if len(target_sentences) >= 2:
            break
    if not target_sentences:
        abstract_sentences = _split_sentences(abstract or full_text, max_sentences=4)
        target_sentences = abstract_sentences[-2:] if len(abstract_sentences) >= 2 else abstract_sentences
    return " ".join(target_sentences).strip() or abstract or _format_text_from_entry(row)


def _paper_qa_context(row: Dict[str, Any], max_chars: int = 2400) -> str:
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    body = str(row.get("text") or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if body:
        parts.append(f"Paper Body: {body[:max_chars]}")
    return "\n\n".join(parts).strip() or _format_text_from_entry(row)


def _build_paper_keyword_samples(rows: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in rows[:max_samples]:
        prompt = str(row.get("abstract") or "").strip() or _format_text_from_entry(row)
        samples.append({"text": prompt, "target": _paper_keyword_target(row)})
    return samples


def _build_paper_qa_samples(rows: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in rows:
        if len(samples) >= max_samples:
            break
        title = str(row.get("title") or "").strip() or "this paper"
        context = _paper_qa_context(row)
        qa_pairs = [
            {
                "question": f"What problem does the paper '{title}' address?",
                "target": _paper_problem_target(row),
            },
            {
                "question": f"What method does the paper '{title}' propose?",
                "target": _paper_method_summary_target(row),
            },
            {
                "question": f"What results or findings does the paper '{title}' report?",
                "target": _paper_results_target(row),
            },
        ]
        for qa in qa_pairs:
            if len(samples) >= max_samples:
                break
            if not str(qa["target"] or "").strip():
                continue
            samples.append(
                {
                    "question": qa["question"],
                    "context": context,
                    "target": qa["target"],
                    "paper_id": row.get("canonical_paper_id") or row.get("paper_id"),
                    "pdf_path": row.get("pdf_path"),
                }
            )
    return samples[:max_samples]


def _build_paper_retrieval_samples(
    rows: List[Dict[str, Any]],
    max_samples: int,
    *,
    chunk_chars: int,
    overlap: int,
    max_chunks_per_paper: int = 2,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not rows:
        return samples

    chunk_records: List[Dict[str, Any]] = []
    for row in rows:
        paper_id = row.get("canonical_paper_id") or row.get("paper_id")
        if not paper_id:
            continue
        chunks = _paper_body_chunks(
            row,
            chunk_chars=chunk_chars,
            overlap=overlap,
            max_chunks=max_chunks_per_paper,
        )
        if not chunks:
            continue
        chunk_records.append(
            {
                "paper_id": paper_id,
                "query": _paper_retrieval_query(row),
                "category": str(row.get("primary_category") or row.get("categories") or ""),
                "chunks": chunks,
            }
        )

    if len(chunk_records) < 2:
        return samples

    for idx, record in enumerate(chunk_records):
        if len(samples) >= max_samples:
            break
        other = chunk_records[(idx + 1) % len(chunk_records)]
        if other["paper_id"] == record["paper_id"] and len(chunk_records) > 1:
            other = chunk_records[(idx + 2) % len(chunk_records)]
        for chunk_text, offset in record["chunks"]:
            samples.append(
                {
                    "text_a": record["query"],
                    "text_b": f"PAPER_SPAN:\n{chunk_text}",
                    "label": 1,
                    "paper_id": record["paper_id"],
                    "offset": offset,
                }
            )
            if len(samples) >= max_samples:
                break
            negative_chunk, negative_offset = other["chunks"][0]
            samples.append(
                {
                    "text_a": record["query"],
                    "text_b": f"PAPER_SPAN:\n{negative_chunk}",
                    "label": 0,
                    "paper_id": other["paper_id"],
                    "offset": negative_offset,
                }
            )
            if len(samples) >= max_samples:
                break
    return samples[:max_samples]


def _build_metadata_embedding_samples(manifest_entries: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not manifest_entries:
        return samples

    records: List[Dict[str, Any]] = []
    for entry in manifest_entries:
        paper_id = _entry_paper_id(entry)
        query = _metadata_embedding_query(entry)
        doc = _metadata_embedding_doc(entry)
        if not query or not doc:
            continue
        records.append({"paper_id": paper_id, "query": query, "doc": doc})

    if len(records) < 2:
        return samples

    for idx, record in enumerate(records):
        if len(samples) >= max_samples:
            break
        other = records[(idx + 1) % len(records)]
        if other["doc"] == record["doc"] and len(records) > 1:
            other = records[(idx + 2) % len(records)]
        samples.append(
            {
                "text_a": record["query"],
                "text_b": f"METADATA_CARD:\n{record['doc']}",
                "label": 1,
                "paper_id": record["paper_id"],
            }
        )
        if len(samples) >= max_samples:
            break
        samples.append(
            {
                "text_a": record["query"],
                "text_b": f"METADATA_CARD:\n{other['doc']}",
                "label": 0,
                "paper_id": other["paper_id"],
            }
        )
    return samples[:max_samples]


def _build_paper_fulltext_embedding_samples(
    rows: List[Dict[str, Any]],
    max_samples: int,
    *,
    chunk_chars: int,
    overlap: int,
    max_chunks_per_paper: int = 2,
    document_chars: int = 6000,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not rows:
        return samples

    records: List[Dict[str, Any]] = []
    for row in rows:
        paper_id = _entry_paper_id(row)
        if not paper_id:
            continue
        document = _paper_document_view(row, max_chars=document_chars)
        chunks = _paper_body_chunks(
            row,
            chunk_chars=chunk_chars,
            overlap=overlap,
            max_chunks=max_chunks_per_paper,
        )
        if not document or not chunks:
            continue
        records.append(
            {
                "paper_id": paper_id,
                "query": _paper_retrieval_query(row),
                "document": document,
                "chunks": chunks,
            }
        )

    if len(records) < 2:
        return samples

    for idx, record in enumerate(records):
        if len(samples) >= max_samples:
            break
        other = records[(idx + 1) % len(records)]
        if other["paper_id"] == record["paper_id"] and len(records) > 1:
            other = records[(idx + 2) % len(records)]
        samples.append(
            {
                "text_a": record["query"],
                "text_b": f"PAPER_DOCUMENT:\n{record['document']}",
                "label": 1,
                "paper_id": record["paper_id"],
                "retrieval_level": "document",
            }
        )
        if len(samples) >= max_samples:
            break
        samples.append(
            {
                "text_a": record["query"],
                "text_b": f"PAPER_DOCUMENT:\n{other['document']}",
                "label": 0,
                "paper_id": other["paper_id"],
                "retrieval_level": "document",
            }
        )
        if len(samples) >= max_samples:
            break
        for chunk_text, offset in record["chunks"]:
            samples.append(
                {
                    "text_a": record["query"],
                    "text_b": f"PAPER_SPAN:\n{chunk_text}",
                    "label": 1,
                    "paper_id": record["paper_id"],
                    "offset": offset,
                    "retrieval_level": "chunk",
                }
            )
            if len(samples) >= max_samples:
                break
            negative_chunk, negative_offset = other["chunks"][0]
            samples.append(
                {
                    "text_a": record["query"],
                    "text_b": f"PAPER_SPAN:\n{negative_chunk}",
                    "label": 0,
                    "paper_id": other["paper_id"],
                    "offset": negative_offset,
                    "retrieval_level": "chunk",
                }
            )
            if len(samples) >= max_samples:
                break
    return samples[:max_samples]


_SENTENCE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SENTENCE_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "between",
    "from",
    "into",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "result",
    "results",
    "study",
    "their",
    "these",
    "this",
    "using",
    "with",
}


def _sentence_tokens(text: str) -> List[str]:
    return [
        token.lower()
        for token in _SENTENCE_TOKEN_RE.findall(text or "")
        if token.lower() not in _SENTENCE_STOPWORDS
    ]


def _sentence_overlap_score(left: str, right: str) -> float:
    left_tokens = set(_sentence_tokens(left))
    right_tokens = set(_sentence_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    denom = max(1, min(len(left_tokens), len(right_tokens)))
    return float(len(overlap)) / float(denom)


def _build_paper_sentence_embedding_samples(
    rows: List[Dict[str, Any]],
    max_samples: int,
    *,
    max_query_sentences: int = 3,
    max_body_sentences: int = 16,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not rows:
        return samples

    records: List[Dict[str, Any]] = []
    for row in rows:
        paper_id = _entry_paper_id(row)
        if not paper_id:
            continue
        title = str(row.get("title") or "").strip()
        abstract_sentences = _split_sentences(str(row.get("abstract") or "").strip(), max_sentences=max_query_sentences * 2)
        body_sentences = _split_sentences(_paper_body_text(row), max_sentences=max_body_sentences)
        if not abstract_sentences or not body_sentences:
            continue
        sentence_pairs: List[Tuple[str, str]] = []
        for query_sentence in abstract_sentences[:max_query_sentences]:
            best_body = max(body_sentences, key=lambda candidate: (_sentence_overlap_score(query_sentence, candidate), -abs(len(candidate) - len(query_sentence))))
            sentence_pairs.append((query_sentence, best_body))
        records.append({"paper_id": paper_id, "title": title, "pairs": sentence_pairs})

    if len(records) < 2:
        return samples

    for idx, record in enumerate(records):
        if len(samples) >= max_samples:
            break
        other = records[(idx + 1) % len(records)]
        if other["paper_id"] == record["paper_id"] and len(records) > 1:
            other = records[(idx + 2) % len(records)]
        for sentence_idx, (query_sentence, body_sentence) in enumerate(record["pairs"]):
            query_parts = []
            if record["title"]:
                query_parts.append(f"TITLE:\n{record['title']}")
            query_parts.append(f"ABSTRACT_SENTENCE:\n{query_sentence}")
            samples.append(
                {
                    "text_a": "\n".join(query_parts),
                    "text_b": f"PAPER_SENTENCE:\n{body_sentence}",
                    "label": 1,
                    "paper_id": record["paper_id"],
                    "sentence_idx": sentence_idx,
                    "retrieval_level": "sentence",
                }
            )
            if len(samples) >= max_samples:
                break
            negative_sentence = other["pairs"][sentence_idx % len(other["pairs"])][1]
            samples.append(
                {
                    "text_a": "\n".join(query_parts),
                    "text_b": f"PAPER_SENTENCE:\n{negative_sentence}",
                    "label": 0,
                    "paper_id": other["paper_id"],
                    "sentence_idx": sentence_idx,
                    "retrieval_level": "sentence",
                }
            )
            if len(samples) >= max_samples:
                break
    return samples[:max_samples]


def _build_fulltext_samples(
    rows: List[Dict[str, Any]],
    max_samples: int,
    *,
    chunk_chars: int,
    overlap: int,
    target_mode: str = "none",
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in rows:
        if len(samples) >= max_samples:
            break
        chunks = list(
            _chunk_text(
                _compose_full_paper_text(row),
                chunk_chars=max(512, int(chunk_chars or 0)),
                overlap=max(0, int(overlap or 0)),
            )
        )
        if not chunks:
            continue
        for idx, (chunk, offset) in enumerate(chunks):
            chunk = str(chunk or "").strip()
            if not chunk:
                continue
            target = None
            if target_mode == "next_chunk":
                if idx + 1 >= len(chunks):
                    continue
                target = str(chunks[idx + 1][0] or "").strip()
                if not target:
                    continue
            elif target_mode == "same_chunk":
                target = chunk
            samples.append(
                {
                    "text": chunk,
                    "target": target,
                    "paper_id": row.get("canonical_paper_id") or row.get("paper_id"),
                    "pdf_path": row.get("pdf_path"),
                    "offset": offset,
                    "label": 0,
                }
            )
            if len(samples) >= max_samples:
                break
    return samples[:max_samples]


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


def _resolve_teacher_embedding_path(construction: Dict[str, Any]) -> Path | None:
    teacher_kind = str(
        construction.get("teacher_embeddings")
        or construction.get("teacher_embedding_source")
        or ""
    ).strip().lower()
    explicit_path = construction.get("teacher_embedding_path")
    if explicit_path:
        resolved = _resolve_data_file(explicit_path)
        return resolved if resolved.exists() else None
    if not teacher_kind or teacher_kind in {"none", "disabled"}:
        return None
    universe_dir = _resolve_data_dir(
        construction.get("paper_universe_dir") or DEFAULT_PAPER_UNIVERSE_DIR,
        "*.parquet",
    )
    if teacher_kind in {"fulltext", "paper_fulltext", "paper_fulltext_embeddings"}:
        candidate = universe_dir / "paper_fulltext_embeddings.parquet"
        return candidate if candidate.exists() else None
    if teacher_kind in {"metadata", "paper_embeddings"}:
        candidate = universe_dir / "paper_embeddings.parquet"
        return candidate if candidate.exists() else None
    return None


def _load_teacher_embedding_lookup(paper_ids: Iterable[str], construction: Dict[str, Any]) -> Dict[str, List[float]]:
    if pq is None:
        return {}
    wanted = {str(paper_id or "").strip() for paper_id in paper_ids if str(paper_id or "").strip()}
    if not wanted:
        return {}
    embedding_path = _resolve_teacher_embedding_path(construction)
    if embedding_path is None or not embedding_path.exists():
        return {}

    lookup: Dict[str, List[float]] = {}
    try:
        parquet_file = pq.ParquetFile(embedding_path)
    except Exception:
        return {}
    schema_names = set(getattr(getattr(parquet_file, "schema_arrow", None), "names", []) or [])
    if not schema_names:
        try:
            schema_names = set(parquet_file.schema.names)
        except Exception:
            schema_names = set()
    available_columns = [col for col in ["paper_id", "canonical_paper_id", "embedding"] if col in schema_names]
    if "embedding" not in available_columns:
        return {}
    for batch in parquet_file.iter_batches(columns=available_columns, batch_size=2048):
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                continue
            paper_id = str(row.get("canonical_paper_id") or row.get("paper_id") or "").strip()
            if not paper_id or paper_id not in wanted:
                continue
            embedding = row.get("embedding")
            if isinstance(embedding, list) and embedding:
                lookup[paper_id] = [float(value) for value in embedding]
        if len(lookup) >= len(wanted):
            break
    return lookup


def _attach_teacher_embeddings(samples: List[Dict[str, Any]], construction: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not samples:
        return samples
    lookup = _load_teacher_embedding_lookup(
        (sample.get("paper_id") for sample in samples),
        construction,
    )
    if not lookup:
        return samples
    default_embedding = [0.0 for _ in range(len(next(iter(lookup.values()))))]
    for sample in samples:
        paper_id = str(sample.get("paper_id") or "").strip()
        embedding = lookup.get(paper_id)
        sample["teacher_embedding"] = list(embedding) if embedding is not None else list(default_embedding)
        sample["teacher_mask"] = 1 if embedding is not None else 0
    return samples


def _build_classifier(manifest_entries: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if not manifest_entries:
        return []
    samples = []
    for e in manifest_entries[:max_samples]:
        text = _format_text_from_entry(e)
        cat = e.get("primary_category") or e.get("categories") or "unknown"
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
    max_samples = _resolve_max_samples(construction.get("max_samples"), default=32)
    chunk_chars_pdf = construction.get("chunk_chars_pdf", 8000)
    chunk_overlap_pdf = construction.get("chunk_overlap_pdf", 400)
    chunk_chars_text = construction.get("chunk_chars_text", chunk_chars_pdf)
    chunk_overlap_text = construction.get("chunk_overlap_text", chunk_overlap_pdf)
    chunk_chars_code = construction.get("chunk_chars_code", 4000)
    chunk_overlap_code = construction.get("chunk_overlap_code", 200)
    structured_pdf_dir = _resolve_data_dir(
        construction.get("structured_pdf_dir") or "exports/pdfs_structured",
        "pdf_structured_*.jsonl",
    )
    paper_dataset_dir = _resolve_data_dir(
        construction.get("paper_dataset_dir") or DEFAULT_PAPER_DATASET_DIR,
        "*.parquet",
    )
    paper_universe_dir = _resolve_data_dir(
        construction.get("paper_universe_dir") or DEFAULT_PAPER_UNIVERSE_DIR,
        "*.parquet",
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
            source_set = set(sources)
            wants_repo_graph = (not source_set) or ("github_repos" in source_set)
            wants_paper_graph = (not source_set) or bool(source_set & {"arxiv_metadata", "paper_text_parquet", "paper_universe", "paper_universe_graph"})
            if wants_repo_graph:
                repo_budget = max_samples if not wants_paper_graph else max(1, max_samples // 2)
                graph_samples = load_graph_samples(max_samples=repo_budget)
                samples.extend([graph_sample_to_text(gs) for gs in graph_samples])
            if wants_paper_graph and len(samples) < max_samples:
                prefer_universe = bool(source_set & {"paper_universe", "paper_universe_graph"}) or paper_universe_dir.exists()
                paper_samples = load_paper_graph_samples(
                    max_samples=max_samples - len(samples),
                    universe_dir=paper_universe_dir,
                    prefer_universe=prefer_universe,
                )
                samples.extend([paper_sample_to_text(ps) for ps in paper_samples])
        except Exception:
            pass
        return _limit_samples(samples, max_samples) if samples else [{"text": "graph placeholder", "label": 0}]

    try:
        filters = config.get("dataset", {}).get("filters", {}) if isinstance(config.get("dataset", {}), dict) else {}
        if "arxiv_metadata" in sources:
            manifest = load_manifest()
            manifest_entries = _sample_metadata(manifest, max_samples, filters)
            if model_id == "M1":
                samples.extend(_build_metadata_embedding_samples(manifest_entries, max_samples))
                manifest_entries = []
        if "paper_text_parquet" in sources:
            paper_rows = _sample_paper_text_parquet(max_samples, filters, paper_dataset_dir)
            if model_id == "M6":
                samples.extend(
                    _build_paper_fulltext_embedding_samples(
                        paper_rows,
                        max_samples,
                        chunk_chars=chunk_chars_text,
                        overlap=chunk_overlap_text,
                        max_chunks_per_paper=int(construction.get("max_chunks_per_paper", 2) or 2),
                        document_chars=int(construction.get("document_chars", 6000) or 6000),
                    )
                )
            elif model_id == "M7":
                samples.extend(
                    _build_paper_sentence_embedding_samples(
                        paper_rows,
                        max_samples,
                        max_query_sentences=int(construction.get("max_query_sentences", 3) or 3),
                        max_body_sentences=int(construction.get("max_body_sentences", 16) or 16),
                    )
                )
            elif model_id == "M1":
                samples.extend(
                    _build_metadata_embedding_samples(paper_rows, max_samples)
                )
            elif model_id in {"P1", "C3"}:
                samples.extend(
                    _build_fulltext_samples(
                        paper_rows,
                        max_samples,
                        chunk_chars=chunk_chars_text,
                        overlap=chunk_overlap_text,
                        target_mode=(
                            "next_chunk"
                            if model_id == "P1"
                            and (
                                config.get("training", {}).get("model_type") == "seq2seq"
                                or config.get("backbone", {}).get("type") == "encoder_decoder"
                            )
                            else "none"
                        ),
                    )
                )
            elif model_id == "A2":
                samples.extend(_build_paper_method_summary_samples(paper_rows, max_samples))
            elif model_id == "A3":
                samples.extend(_build_paper_keyword_samples(paper_rows, max_samples))
            elif model_id == "P5":
                samples.extend(_build_paper_qa_samples(paper_rows, max_samples))
            else:
                manifest_entries.extend(paper_rows)
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
                return _limit_samples(samples, max_samples)
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

    if samples and model_id in {"M6", "M7"}:
        samples = _attach_teacher_embeddings(samples, construction)

    return _limit_samples(samples, max_samples)
