"""
Paper ↔ Repo alignment preprocessing.

Builds heuristic paper↔repo pairs from structured PDF shards and repo chunk
shards, with three goals:

1. Use focused paper text (title/abstract/snippet), not flattened 20k-char blobs.
2. Use IDF-weighted lexical retrieval instead of naive best-global Jaccard.
3. Prevent a handful of generic repo chunks from dominating the positives.

Output is JSONL with backwards-compatible fields:
  {
    "paper_text": "...",
    "repo_text": "...",
    "label": 1 or 0,
    "score": float,

    # extra metadata for grouped splits / debugging
    "paper_id": "...",
    "paper_title": "...",
    "paper_abstract": "...",
    "pdf_path": "...",
    "repo_id": "...",
    "repo_path": "...",
    "repo_offset": 0,
    "shared_terms": ["...", ...],
    "candidate_rank": 1,
    "negative_type": "hard" | "random" | null
  }

Usage:
  PYTHONPATH=.. python -m models.scripts.preprocess_alignment \
      --max-papers 5000 \
      --max-repos 5000 \
      --top-k 5000 \
      --negatives 5000 \
      --out exports/paper_repo_align.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests

from models.shared.pdf_utils import extract_pdf_text


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_ROOT = Path("/arxiv/pdfs")
PDF_CACHE_ROOT = REPO_ROOT / "exports/arxiv_pdfs"
DEFAULT_DOWNLOAD_DELAY = 3.0
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./+-]{2,}")
CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
SECTION_RE = re.compile(r"^(?:\d+[\.\)]?|[IVXLC]+[\.\)]?)\s+[A-Z][A-Z0-9\s:-]{2,}$")
STOPWORDS = {
    "a",
    "an",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "among",
    "and",
    "any",
    "are",
    "as",
    "at",
    "around",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "case",
    "common",
    "could",
    "current",
    "data",
    "dataset",
    "datasets",
    "details",
    "different",
    "does",
    "each",
    "efficiency",
    "field",
    "for",
    "from",
    "given",
    "generate",
    "have",
    "help",
    "here",
    "however",
    "how",
    "improving",
    "into",
    "is",
    "it",
    "its",
    "key",
    "keys",
    "large",
    "logic",
    "may",
    "method",
    "methods",
    "model",
    "models",
    "more",
    "most",
    "much",
    "new",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "paper",
    "papers",
    "performance",
    "propose",
    "proposed",
    "provide",
    "provides",
    "quality",
    "results",
    "secure",
    "security",
    "show",
    "shows",
    "shown",
    "specific",
    "state",
    "states",
    "such",
    "system",
    "systems",
    "than",
    "that",
    "their",
    "there",
    "the",
    "these",
    "this",
    "those",
    "to",
    "through",
    "under",
    "use",
    "used",
    "uses",
    "user",
    "using",
    "vulnerabilities",
    "very",
    "way",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
    "without",
    "words",
}
GENERIC_REPO_FRAGMENTS = (
    "/example/",
    "/examples/",
    "/demo/",
    "/demos/",
    "/tutorial/",
    "/tutorials/",
    "/docs/",
    "/doc/",
    "/tests/",
    "/test/",
    "/benchmark/",
    "/benchmarks/",
    "/prompts/",
)
GENERIC_FILENAMES = {
    "readme.md",
    "changelog.md",
    "news.md",
    "setup.py",
    "versioneer.py",
    "conftest.py",
}
ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})", re.IGNORECASE)


@dataclass
class PaperRecord:
    paper_id: str
    pdf_path: str
    title: str
    abstract: str
    snippet: str
    paper_text: str
    query_counts: Counter[str]
    title_terms: set[str]
    normalized_title: str
    query_weights: Dict[str, float] = field(default_factory=dict)
    title_weights: Dict[str, float] = field(default_factory=dict)
    total_query_weight: float = 0.0
    total_title_weight: float = 0.0


@dataclass
class RepoChunk:
    repo_id: str
    path: str
    offset: int
    text: str
    terms: set[str]
    path_terms: set[str]
    quality: float
    quality_flags: List[str]
    term_weight: float = 1.0


@dataclass
class RepoProfile:
    repo_id: str
    text: str
    terms: set[str]
    name_terms: set[str]
    normalized_text: str
    term_weight: float = 1.0


@dataclass
class RepoHintIndex:
    by_paper_id: Dict[str, set[str]]
    by_term: Dict[str, set[str]]


def _unique_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _candidate_paths(path_str: str) -> List[Path]:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(REPO_ROOT / raw)
        if not str(raw).startswith("models/"):
            candidates.append(REPO_ROOT / "models" / raw)
    return _unique_paths(candidates)


def _pdf_roots(preferred_root: Optional[Path] = None) -> List[Path]:
    roots = []
    if preferred_root is not None:
        roots.append(preferred_root)
    roots.extend([PDF_ROOT, PDF_CACHE_ROOT])
    return _unique_paths(roots)


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".codex_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def _download_root(preferred_root: Optional[Path] = None) -> Path:
    for root in _pdf_roots(preferred_root):
        if _is_writable_dir(root):
            return root
    return preferred_root or PDF_CACHE_ROOT


def _resolve_data_dir(path_str: str, pattern: str) -> Path:
    for candidate in _candidate_paths(path_str):
        if candidate.is_dir() and any(candidate.glob(pattern)):
            return candidate
    for candidate in _candidate_paths(path_str):
        if candidate.exists():
            return candidate
    return _candidate_paths(path_str)[0]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_phrase(text: str) -> str:
    return _normalize_space(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()))


def _paper_id_from_pdf_path(pdf_path: str) -> str:
    return Path(pdf_path).stem


def _repo_id_from_path(path: str) -> str:
    parts = list(Path(path).parts)
    if "repositories" in parts:
        idx = parts.index("repositories")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).parent.name or "unknown"


def _text_from_tokens(tokens: Sequence[dict]) -> str:
    parts: List[str] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = _normalize_space(str(token.get("text") or ""))
        if text:
            parts.append(text)
    return "\n".join(parts)


def _lines_from_record(obj: Dict[str, object]) -> List[str]:
    tokens = obj.get("tokens")
    if isinstance(tokens, list) and tokens:
        raw_lines = [_normalize_space(str(t.get("text") or "")) for t in tokens if isinstance(t, dict)]
    else:
        raw = obj.get("text") or obj.get("content") or ""
        raw_lines = [_normalize_space(line) for line in str(raw).splitlines()]
    lines = [line for line in raw_lines if line]
    return lines


def _is_authorish(line: str) -> bool:
    lower = line.lower()
    if line.startswith("arXiv:"):
        return True
    if any(token in lower for token in ("university", "department", "institute", "school", "laboratory", "laboratoire")):
        return True
    if "@" in line:
        return True
    if lower.count(",") >= 2 and len(line.split()) <= 18:
        return True
    return False


def _extract_title_and_abstract(lines: Sequence[str]) -> Tuple[str, str]:
    title = ""
    for line in lines[:12]:
        if len(line) < 12:
            continue
        if _is_authorish(line):
            continue
        if line.lower().startswith("abstract"):
            break
        title = line
        break
    if not title and lines:
        title = lines[0]

    abstract_parts: List[str] = []
    collecting = False
    for line in lines[:160]:
        lower = line.lower()
        if not collecting:
            if lower.startswith("abstract"):
                collecting = True
                tail = re.sub(r"^abstract[:.\s-]*", "", line, flags=re.IGNORECASE).strip()
                if tail:
                    abstract_parts.append(tail)
                continue
        else:
            if SECTION_RE.match(line):
                break
            if lower.startswith(("keywords", "index terms", "introduction", "1 introduction", "i. introduction")):
                break
            abstract_parts.append(line)
            if len(" ".join(abstract_parts)) >= 1600:
                break
    abstract = _normalize_space(" ".join(abstract_parts))
    return _normalize_space(title), abstract


def _paper_snippet(lines: Sequence[str], abstract: str) -> str:
    if abstract:
        return abstract[:1600]
    body_lines: List[str] = []
    for line in lines[1:80]:
        if SECTION_RE.match(line):
            continue
        body_lines.append(line)
        if len(" ".join(body_lines)) >= 1200:
            break
    return _normalize_space(" ".join(body_lines))[:1200]


def _compose_paper_text(title: str, abstract: str, snippet: str) -> str:
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    elif snippet:
        parts.append(f"Excerpt: {snippet}")
    return "\n".join(parts).strip()


def _build_paper_record(
    *,
    paper_id: str,
    pdf_path: str,
    lines: Sequence[str],
    repo_profiles: Optional[Dict[str, RepoProfile]] = None,
    repo_hint_index: Optional[RepoHintIndex] = None,
) -> Optional[PaperRecord]:
    if not lines:
        return None
    title, abstract = _extract_title_and_abstract(lines)
    snippet = _paper_snippet(lines, abstract)
    paper_text = _compose_paper_text(title, abstract, snippet)
    if len(paper_text) < 40:
        return None
    query_counts = _term_counts("\n".join(filter(None, [title, abstract, snippet])), max_terms=256)
    if not query_counts:
        return None
    title_terms = set(_term_counts(title, max_terms=48))
    record = PaperRecord(
        paper_id=paper_id,
        pdf_path=pdf_path,
        title=title,
        abstract=abstract,
        snippet=snippet,
        paper_text=paper_text,
        query_counts=query_counts,
        title_terms=title_terms,
        normalized_title=_normalize_phrase(title),
    )
    if repo_profiles is not None and repo_hint_index is not None:
        if not _paper_candidate_repo_ids(record, repo_profiles, repo_hint_index):
            return None
    elif repo_profiles is not None:
        return None
    return record


def _split_terms(text: str) -> Iterator[str]:
    prepared = CAMEL_RE.sub(r"\1 \2", text or "")
    for match in WORD_RE.finditer(prepared):
        raw = match.group(0).lower().strip("._/-+")
        for piece in re.split(r"[._/\-+]+", raw):
            if len(piece) < 3:
                continue
            if piece.isdigit():
                continue
            if piece in STOPWORDS:
                continue
            yield piece


def _term_counts(text: str, *, max_terms: int = 256) -> Counter[str]:
    counts: Counter[str] = Counter()
    for term in _split_terms(text):
        counts[term] += 1
        if sum(counts.values()) >= max_terms:
            break
    return counts


def _path_terms(path: str) -> set[str]:
    return set(_split_terms(Path(path).as_posix()))


def _repo_quality(path: str, text: str) -> Tuple[float, List[str]]:
    flags: List[str] = []
    score = 1.0
    lower_path = path.lower()
    basename = Path(path).name.lower()
    suffix = Path(path).suffix.lower()

    if any(fragment in lower_path for fragment in GENERIC_REPO_FRAGMENTS):
        score *= 0.35
        flags.append("generic_path")
    if basename in GENERIC_FILENAMES:
        score *= 0.2
        flags.append("generic_file")
    if "prompt" in lower_path:
        score *= 0.2
        flags.append("prompt")
    if suffix in {".md", ".rst", ".txt"}:
        score *= 0.5
        flags.append("docs")
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        score *= 0.65
        flags.append("config")

    lower_text = text.lower()
    code_markers = ("def ", "class ", "import ", "from ", "return ", "function ", "{", "};")
    if any(marker in lower_text for marker in code_markers):
        score *= 1.05
    elif len(text) < 800:
        score *= 0.75
        flags.append("short_text")

    return max(0.05, min(score, 1.25)), flags


def _load_structured_papers(
    shard_dir: Path,
    max_records: int,
    seed: int,
    repo_profiles: Optional[Dict[str, RepoProfile]] = None,
    repo_hint_index: Optional[RepoHintIndex] = None,
    existing_ids: Optional[set[str]] = None,
) -> List[PaperRecord]:
    papers: List[PaperRecord] = []
    seen_ids: set[str] = set(existing_ids or set())
    rng = random.Random(seed)
    seen_eligible = 0
    shard_paths = sorted(shard_dir.glob("pdf_structured_*.jsonl"))
    rng.shuffle(shard_paths)
    scan_limit = None if repo_profiles else (max_records * 8 if max_records else None)
    for shard in shard_paths:
        if repo_profiles is not None and max_records and len(papers) >= max_records:
            break
        if scan_limit and seen_eligible >= scan_limit:
            break
        try:
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if repo_profiles is not None and max_records and len(papers) >= max_records:
                        break
                    if scan_limit and seen_eligible >= scan_limit:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    pdf_path = str(obj.get("pdf_path") or "")
                    if not pdf_path:
                        continue
                    paper_id = _paper_id_from_pdf_path(pdf_path)
                    if paper_id in seen_ids:
                        continue
                    lines = _lines_from_record(obj)
                    record = _build_paper_record(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        lines=lines,
                        repo_profiles=repo_profiles,
                        repo_hint_index=repo_hint_index,
                    )
                    if record is None:
                        seen_ids.add(paper_id)
                        continue
                    seen_eligible += 1
                    if not max_records or len(papers) < max_records:
                        papers.append(record)
                    else:
                        replace_idx = rng.randrange(seen_eligible)
                        if replace_idx < max_records:
                            papers[replace_idx] = record
                    seen_ids.add(paper_id)
        except Exception:
            continue
    return papers


def _candidate_pdf_paths_for_id(paper_id: str, pdf_root: Optional[Path] = None) -> List[Path]:
    paths = []
    stem = paper_id.split("v", 1)[0]
    for root in _pdf_roots(pdf_root):
        if len(stem) >= 4 and stem[:4].isdigit():
            paths.append(root / stem[:4] / f"{stem}.pdf")
        paths.append(root / f"{stem}.pdf")
    return _unique_paths(paths)


def _pdf_storage_path_for_id(paper_id: str, pdf_root: Optional[Path] = None) -> Path:
    stem = paper_id.split("v", 1)[0]
    root = _download_root(pdf_root)
    if len(stem) >= 4 and stem[:4].isdigit():
        return root / stem[:4] / f"{stem}.pdf"
    return root / f"{stem}.pdf"


def _ensure_pdf_for_id(
    paper_id: str,
    *,
    pdf_root: Optional[Path] = None,
    timeout: int = 60,
) -> Tuple[Optional[Path], bool]:
    stem = paper_id.split("v", 1)[0]
    for candidate in _candidate_pdf_paths_for_id(stem, pdf_root):
        if candidate.is_file():
            return candidate, False

    target = _pdf_storage_path_for_id(stem, pdf_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    pdf_url = f"https://export.arxiv.org/pdf/{stem}.pdf"

    try:
        resp = requests.get(pdf_url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with target.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
    except Exception:
        try:
            target.unlink()
        except Exception:
            pass
        raise
    return target, True


def _load_raw_repo_linked_papers(
    repo_hint_index: RepoHintIndex,
    repo_profiles: Dict[str, RepoProfile],
    existing_ids: set[str],
    limit: int,
    *,
    pdf_root: Optional[Path] = None,
    download_missing: bool = False,
    download_timeout: int = 60,
    download_delay: float = DEFAULT_DOWNLOAD_DELAY,
) -> List[PaperRecord]:
    records: List[PaperRecord] = []
    downloaded = 0
    failed = 0
    for paper_id in sorted(repo_hint_index.by_paper_id):
        if limit and len(records) >= limit:
            break
        if paper_id in existing_ids:
            continue
        pdf_path = None
        for candidate in _candidate_pdf_paths_for_id(paper_id, pdf_root):
            if candidate.is_file():
                pdf_path = candidate
                break
        if pdf_path is None:
            if not download_missing:
                continue
            try:
                pdf_path, changed = _ensure_pdf_for_id(paper_id, pdf_root=pdf_root, timeout=download_timeout)
            except Exception as exc:
                failed += 1
                print(f"[warn] failed to download linked paper {paper_id}: {exc}")
                continue
            if pdf_path is None:
                continue
            if changed:
                downloaded += 1
                print(f"[download] linked paper {paper_id} -> {pdf_path}")
                if download_delay > 0:
                    time.sleep(download_delay)
        text = extract_pdf_text(str(pdf_path), max_chars=16_000)
        lines = [_normalize_space(line) for line in text.splitlines() if _normalize_space(line)]
        record = _build_paper_record(
            paper_id=paper_id,
            pdf_path=str(pdf_path),
            lines=lines,
            repo_profiles=repo_profiles,
            repo_hint_index=repo_hint_index,
        )
        if record is None:
            continue
        records.append(record)
        existing_ids.add(paper_id)
    if download_missing and (downloaded or failed):
        print(f"[download] repo-linked PDFs: downloaded={downloaded} failed={failed}")
    return records


def _load_repo_chunks(chunks_dir: Path, max_records: int, seed: int) -> List[RepoChunk]:
    chunks: List[RepoChunk] = []
    seen_payloads: set[Tuple[str, int, str]] = set()
    rng = random.Random(seed)
    seen_eligible = 0
    shard_paths = sorted(chunks_dir.glob("repo_chunks_*.jsonl"))
    rng.shuffle(shard_paths)
    scan_limit = max_records * 8 if max_records else None
    for shard in shard_paths:
        if scan_limit and seen_eligible >= scan_limit:
            break
        try:
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if scan_limit and seen_eligible >= scan_limit:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    path = str(obj.get("path") or "")
                    text = _normalize_space(str(obj.get("code") or obj.get("text") or ""))
                    if not path or not text:
                        continue
                    offset = int(obj.get("offset") or 0)
                    dedupe_key = (path, offset, text[:256])
                    if dedupe_key in seen_payloads:
                        continue
                    seen_payloads.add(dedupe_key)
                    counts = _term_counts(text, max_terms=256)
                    if len(counts) < 3:
                        continue
                    quality, quality_flags = _repo_quality(path, text)
                    record = RepoChunk(
                        repo_id=_repo_id_from_path(path),
                        path=path,
                        offset=offset,
                        text=text,
                        terms=set(counts),
                        path_terms=_path_terms(path),
                        quality=quality,
                        quality_flags=quality_flags,
                    )
                    seen_eligible += 1
                    if not max_records or len(chunks) < max_records:
                        chunks.append(record)
                    else:
                        replace_idx = rng.randrange(seen_eligible)
                        if replace_idx < max_records:
                            chunks[replace_idx] = record
        except Exception:
            continue
    return chunks


def _read_repo_profile_text(repo_id: str, chunks: Sequence[RepoChunk]) -> str:
    repo_root = Path("/data/repositories") / repo_id
    parts = [repo_id.replace("-", " ").replace("_", " ")]
    if repo_root.is_dir():
        readmes = sorted(
            p
            for p in repo_root.iterdir()
            if p.is_file() and p.name.lower().startswith("readme")
        )
        for readme in readmes[:1]:
            try:
                text = readme.read_text(encoding="utf-8")
            except Exception:
                try:
                    text = readme.read_bytes().decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
            if text:
                parts.append(text[:12_000])
                break
    if len(parts) == 1:
        sample_paths = [Path(chunk.path).name for chunk in chunks[:8]]
        if sample_paths:
            parts.append(" ".join(sample_paths))
    return "\n".join(part for part in parts if part).strip()


def _build_repo_profiles(chunks: Sequence[RepoChunk]) -> Dict[str, RepoProfile]:
    grouped: Dict[str, List[RepoChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.repo_id].append(chunk)

    profiles: Dict[str, RepoProfile] = {}
    for repo_id, repo_chunks in grouped.items():
        profile_text = _read_repo_profile_text(repo_id, repo_chunks)
        counts = _term_counts(profile_text, max_terms=512)
        profiles[repo_id] = RepoProfile(
            repo_id=repo_id,
            text=profile_text,
            terms=set(counts),
            name_terms=_path_terms(repo_id),
            normalized_text=_normalize_phrase(profile_text),
        )
    return profiles


def _build_repo_hint_index(repo_profiles: Dict[str, RepoProfile]) -> RepoHintIndex:
    by_paper_id: Dict[str, set[str]] = defaultdict(set)
    by_term: Dict[str, set[str]] = defaultdict(set)
    for repo_id, profile in repo_profiles.items():
        for match in ARXIV_ID_RE.finditer(profile.text):
            by_paper_id[match.group(1)].add(repo_id)
        for term in profile.terms:
            by_term[term].add(repo_id)
    return RepoHintIndex(by_paper_id=dict(by_paper_id), by_term=dict(by_term))


def _paper_candidate_repo_ids(
    paper: PaperRecord,
    repo_profiles: Dict[str, RepoProfile],
    repo_hint_index: RepoHintIndex,
) -> List[str]:
    candidate_repo_ids: set[str] = set(repo_hint_index.by_paper_id.get(paper.paper_id, set()))
    for term in paper.title_terms:
        candidate_repo_ids.update(repo_hint_index.by_term.get(term, set()))

    if not candidate_repo_ids:
        return []

    matches: List[str] = []
    for repo_id in candidate_repo_ids:
        profile = repo_profiles[repo_id]
        if paper.paper_id and paper.paper_id in profile.text:
            matches.append(repo_id)
            continue
        if paper.normalized_title and len(paper.normalized_title) >= 16 and paper.normalized_title in profile.normalized_text:
            matches.append(repo_id)
    return matches


def _build_idf(chunks: Sequence[RepoChunk]) -> Dict[str, float]:
    df: Counter[str] = Counter()
    for chunk in chunks:
        df.update(chunk.terms)
    total = max(1, len(chunks))
    idf: Dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((1.0 + total) / (1.0 + freq)) + 1.0
    return idf


def _build_repo_profile_idf(profiles: Dict[str, RepoProfile]) -> Dict[str, float]:
    df: Counter[str] = Counter()
    for profile in profiles.values():
        df.update(profile.terms)
    total = max(1, len(profiles))
    idf: Dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((1.0 + total) / (1.0 + freq)) + 1.0
    return idf


def _attach_paper_weights(papers: Sequence[PaperRecord], idf: Dict[str, float]) -> None:
    for paper in papers:
        query_weights: Dict[str, float] = {}
        title_weights: Dict[str, float] = {}
        for term, count in paper.query_counts.items():
            idf_value = idf.get(term, 1.0)
            if idf_value < 1.2:
                continue
            if idf_value < 1.6 and term not in paper.title_terms:
                continue
            weight = float(count) * idf_value
            if term in paper.title_terms:
                weight *= 1.5
                title_weights[term] = float(count) * idf_value
            query_weights[term] = weight
        paper.query_weights = query_weights
        paper.title_weights = title_weights
        paper.total_query_weight = sum(query_weights.values()) or 1.0
        paper.total_title_weight = sum(title_weights.values()) or 1.0


def _attach_chunk_weights(chunks: Sequence[RepoChunk], idf: Dict[str, float]) -> None:
    for chunk in chunks:
        chunk.term_weight = sum(idf.get(term, 1.0) for term in chunk.terms if idf.get(term, 1.0) >= 1.4) or 1.0


def _attach_repo_profile_weights(profiles: Dict[str, RepoProfile], idf: Dict[str, float]) -> None:
    for profile in profiles.values():
        profile.term_weight = sum(idf.get(term, 1.0) for term in profile.terms if idf.get(term, 1.0) >= 1.2) or 1.0


def _build_inverted_index(chunks: Sequence[RepoChunk]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        for term in chunk.terms:
            index[term].append(idx)
    return index


def _shared_terms(paper: PaperRecord, chunk: RepoChunk) -> List[str]:
    shared = [term for term in paper.query_weights if term in chunk.terms]
    shared.sort(key=lambda term: paper.query_weights.get(term, 0.0), reverse=True)
    return shared


def _repo_candidate_score(
    paper: PaperRecord,
    profile: RepoProfile,
    *,
    idf: Dict[str, float],
) -> float:
    title_norm = _normalize_phrase(paper.title)
    phrase_match = 1.0 if title_norm and len(title_norm) >= 16 and title_norm in profile.normalized_text else 0.0
    paper_id_match = 1.0 if paper.paper_id and paper.paper_id in profile.text else 0.0
    title_overlap_terms = paper.title_terms & profile.terms
    if phrase_match <= 0 and paper_id_match <= 0 and len(title_overlap_terms) < 2:
        return 0.0

    shared = [term for term in paper.query_weights if term in profile.terms]
    if not shared and phrase_match <= 0 and paper_id_match <= 0:
        return 0.0

    overlap_weight = sum(paper.query_weights.get(term, 0.0) for term in shared)
    lexical = overlap_weight / paper.total_query_weight
    shared_idf = sum(idf.get(term, 1.0) for term in shared)
    precision = shared_idf / profile.term_weight
    title_term_coverage = len(title_overlap_terms) / float(max(1, len(paper.title_terms)))
    name_overlap = len(paper.title_terms & profile.name_terms) / float(max(1, len(paper.title_terms)))

    return (
        0.40 * lexical
        + 0.20 * precision
        + 0.20 * title_term_coverage
        + 0.10 * name_overlap
        + 0.60 * phrase_match
        + 0.90 * paper_id_match
    )


def _candidate_score(
    paper: PaperRecord,
    chunk: RepoChunk,
    *,
    min_shared_terms: int,
    idf: Dict[str, float],
) -> Tuple[float, List[str]]:
    shared = _shared_terms(paper, chunk)
    if len(shared) < min_shared_terms:
        return 0.0, []

    overlap_weight = sum(paper.query_weights.get(term, 0.0) for term in shared)
    lexical = overlap_weight / paper.total_query_weight

    shared_idf = sum(idf.get(term, 1.0) for term in shared)
    precision = shared_idf / chunk.term_weight

    title_overlap = sum(paper.title_weights.get(term, 0.0) for term in shared)
    title_score = title_overlap / paper.total_title_weight if paper.title_terms else 0.0

    path_hits = len(paper.title_terms & chunk.path_terms)
    path_score = path_hits / float(max(1, len(paper.title_terms))) if paper.title_terms else 0.0

    specific_hits = sum(1 for term in shared if idf.get(term, 0.0) >= 3.5)
    specificity = specific_hits / float(len(shared))
    if specific_hits == 0 and title_score <= 0 and path_score <= 0:
        return 0.0, []

    shared_bonus = min(0.20, 0.03 * len(shared))
    score = (0.48 * lexical + 0.32 * precision + 0.15 * title_score + 0.05 * path_score + 0.08 * specificity + shared_bonus) * chunk.quality
    return score, shared[:16]


def _retrieve_candidates(
    paper: PaperRecord,
    index: Dict[str, List[int]],
    chunks: Sequence[RepoChunk],
    *,
    allowed_repos: Optional[set[str]],
    candidate_pool: int,
    max_postings: int,
    min_shared_terms: int,
    idf: Dict[str, float],
) -> List[Tuple[float, int, List[str]]]:
    prelim: Dict[int, float] = defaultdict(float)
    for term, weight in paper.query_weights.items():
        postings = index.get(term)
        if not postings or len(postings) > max_postings:
            continue
        boost = 1.2 if term in paper.title_terms else 1.0
        for idx in postings:
            if allowed_repos is not None and chunks[idx].repo_id not in allowed_repos:
                continue
            prelim[idx] += weight * boost
    if not prelim:
        return []

    ranked = sorted(prelim.items(), key=lambda item: item[1], reverse=True)[: candidate_pool * 4]
    rescored: List[Tuple[float, int, List[str]]] = []
    for idx, _ in ranked:
        score, shared = _candidate_score(paper, chunks[idx], min_shared_terms=min_shared_terms, idf=idf)
        if score <= 0:
            continue
        rescored.append((score, idx, shared))
    rescored.sort(key=lambda item: item[0], reverse=True)
    return rescored[:candidate_pool]


def _positive_record(
    paper: PaperRecord,
    chunk: RepoChunk,
    *,
    score: float,
    shared_terms: Sequence[str],
    candidate_rank: int,
) -> Dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "paper_title": paper.title,
        "paper_abstract": paper.abstract,
        "pdf_path": paper.pdf_path,
        "paper_text": paper.paper_text,
        "repo_id": chunk.repo_id,
        "repo_path": chunk.path,
        "repo_offset": chunk.offset,
        "repo_text": chunk.text,
        "label": 1,
        "score": round(score, 6),
        "shared_terms": list(shared_terms),
        "candidate_rank": candidate_rank,
        "negative_type": None,
    }


def _negative_record(
    paper: PaperRecord,
    chunk: RepoChunk,
    *,
    score: float,
    shared_terms: Sequence[str],
    candidate_rank: int,
    negative_type: str,
) -> Dict[str, object]:
    return {
        "paper_id": paper.paper_id,
        "paper_title": paper.title,
        "paper_abstract": paper.abstract,
        "pdf_path": paper.pdf_path,
        "paper_text": paper.paper_text,
        "repo_id": chunk.repo_id,
        "repo_path": chunk.path,
        "repo_offset": chunk.offset,
        "repo_text": chunk.text,
        "label": 0,
        "score": round(score, 6),
        "shared_terms": list(shared_terms),
        "candidate_rank": candidate_rank,
        "negative_type": negative_type,
    }


def build_alignment(
    papers: Sequence[PaperRecord],
    chunks: Sequence[RepoChunk],
    *,
    top_k: int,
    negatives: int,
    min_score: float,
    candidate_pool: int,
    max_postings: int,
    min_shared_terms: int,
    max_positives_per_repo_chunk: int,
    max_positives_per_repo: int,
    seed: int,
    repo_profiles: Optional[Dict[str, RepoProfile]] = None,
) -> List[Dict[str, object]]:
    random.seed(seed)
    idf = _build_idf(chunks)
    _attach_paper_weights(papers, idf)
    _attach_chunk_weights(chunks, idf)
    repo_profiles = repo_profiles or _build_repo_profiles(chunks)
    repo_hint_index = _build_repo_hint_index(repo_profiles)
    repo_profile_idf = _build_repo_profile_idf(repo_profiles)
    _attach_repo_profile_weights(repo_profiles, repo_profile_idf)
    index = _build_inverted_index(chunks)

    paper_candidates: List[Tuple[PaperRecord, List[Tuple[float, int, List[str]]]]] = []
    for paper in papers:
        candidate_repo_ids = _paper_candidate_repo_ids(paper, repo_profiles, repo_hint_index)
        if not candidate_repo_ids:
            continue
        repo_scores: List[Tuple[float, str]] = []
        for repo_id in candidate_repo_ids:
            profile = repo_profiles[repo_id]
            repo_score = _repo_candidate_score(paper, profile, idf=repo_profile_idf)
            if repo_score > 0:
                repo_scores.append((repo_score, repo_id))
        repo_scores.sort(reverse=True)
        allowed_repos = {repo_id for score, repo_id in repo_scores[:4] if score >= 0.35}
        if not allowed_repos:
            continue
        candidates = _retrieve_candidates(
            paper,
            index,
            chunks,
            allowed_repos=allowed_repos,
            candidate_pool=candidate_pool,
            max_postings=max_postings,
            min_shared_terms=min_shared_terms,
            idf=idf,
        )
        if candidates:
            paper_candidates.append((paper, candidates))

    paper_candidates.sort(key=lambda item: item[1][0][0], reverse=True)

    chosen_pairs: List[Tuple[PaperRecord, int, float, List[str], int, List[Tuple[float, int, List[str]]]]] = []
    chunk_positive_counts: Counter[int] = Counter()
    repo_positive_counts: Counter[str] = Counter()
    seen_papers: set[str] = set()

    for paper, candidates in paper_candidates:
        if len(chosen_pairs) >= top_k:
            break
        if paper.paper_id in seen_papers:
            continue
        for rank, (score, idx, shared_terms) in enumerate(candidates, start=1):
            chunk = chunks[idx]
            if score < min_score:
                break
            if chunk_positive_counts[idx] >= max_positives_per_repo_chunk:
                continue
            if repo_positive_counts[chunk.repo_id] >= max_positives_per_repo:
                continue
            chosen_pairs.append((paper, idx, score, shared_terms, rank, candidates))
            chunk_positive_counts[idx] += 1
            repo_positive_counts[chunk.repo_id] += 1
            seen_papers.add(paper.paper_id)
            break

    if negatives <= 0:
        negatives = len(chosen_pairs)

    samples: List[Dict[str, object]] = []
    used_pairs: set[Tuple[str, str, int]] = set()
    hard_negative_pool: List[Tuple[PaperRecord, int, float, List[str], int]] = []

    for paper, idx, score, shared_terms, rank, candidates in chosen_pairs:
        chunk = chunks[idx]
        pair_key = (paper.paper_id, chunk.path, chunk.offset)
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        samples.append(_positive_record(paper, chunk, score=score, shared_terms=shared_terms, candidate_rank=rank))
        for cand_rank, (cand_score, cand_idx, cand_shared_terms) in enumerate(candidates, start=1):
            if cand_idx == idx:
                continue
            hard_negative_pool.append((paper, cand_idx, cand_score, cand_shared_terms, cand_rank))

    hard_negative_pool.sort(key=lambda item: item[2], reverse=True)

    negative_count = 0
    used_negative_keys: set[Tuple[str, str, int]] = set()
    for paper, idx, score, shared_terms, rank in hard_negative_pool:
        if negative_count >= negatives:
            break
        chunk = chunks[idx]
        key = (paper.paper_id, chunk.path, chunk.offset)
        if key in used_pairs or key in used_negative_keys:
            continue
        used_negative_keys.add(key)
        samples.append(
            _negative_record(
                paper,
                chunk,
                score=score,
                shared_terms=shared_terms,
                candidate_rank=rank,
                negative_type="hard",
            )
        )
        negative_count += 1

    positive_papers = [paper for paper, _, _, _, _, _ in chosen_pairs]
    while negative_count < negatives and positive_papers:
        paper = random.choice(positive_papers)
        chunk = random.choice(chunks)
        key = (paper.paper_id, chunk.path, chunk.offset)
        if key in used_pairs or key in used_negative_keys:
            continue
        used_negative_keys.add(key)
        samples.append(
            _negative_record(
                paper,
                chunk,
                score=0.0,
                shared_terms=[],
                candidate_rank=0,
                negative_type="random",
            )
        )
        negative_count += 1

    random.shuffle(samples)
    return samples


def _print_summary(samples: Sequence[Dict[str, object]]) -> None:
    positives = [sample for sample in samples if int(sample.get("label") or 0) == 1]
    negatives = [sample for sample in samples if int(sample.get("label") or 0) == 0]
    unique_papers = len({str(sample.get("paper_id") or "") for sample in samples})
    unique_repos = len({str(sample.get("repo_id") or "") for sample in positives})
    unique_chunks = len({(str(sample.get("repo_path") or ""), int(sample.get("repo_offset") or 0)) for sample in positives})
    hard_negatives = sum(1 for sample in negatives if sample.get("negative_type") == "hard")
    print(
        "[summary] "
        f"pairs={len(samples)} positives={len(positives)} negatives={len(negatives)} "
        f"unique_papers={unique_papers} unique_positive_repos={unique_repos} "
        f"unique_positive_chunks={unique_chunks} hard_negatives={hard_negatives}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=str, default="exports/pdfs_structured", help="Structured PDF shards dir")
    ap.add_argument("--repos-dir", type=str, default="exports/repos_chunks", help="Repo chunks dir")
    ap.add_argument("--out", type=str, default="exports/paper_repo_align.jsonl", help="Output JSONL path")
    ap.add_argument("--max-papers", type=int, default=5000, help="Maximum matched papers to align (0 = all matched papers)")
    ap.add_argument("--max-repos", type=int, default=0, help="Maximum repo chunks to consider (0 = all chunks)")
    ap.add_argument("--top-k", type=int, default=5000, help="Maximum positive pairs")
    ap.add_argument("--negatives", type=int, default=0, help="Number of negative pairs (0 = match positive count)")
    ap.add_argument("--candidate-pool", type=int, default=24, help="Top candidate chunks retained per paper before balancing")
    ap.add_argument("--max-postings", type=int, default=400, help="Ignore extremely common terms with posting lists above this size")
    ap.add_argument("--min-score", type=float, default=0.22, help="Minimum score required for a positive pair")
    ap.add_argument("--min-shared-terms", type=int, default=2, help="Minimum shared normalized terms")
    ap.add_argument("--max-positives-per-repo-chunk", type=int, default=24)
    ap.add_argument("--max-positives-per-repo", type=int, default=96)
    ap.add_argument(
        "--pdf-dir",
        type=str,
        default=str(PDF_ROOT),
        help="Local PDF root directory used for lookup and downloads.",
    )
    ap.add_argument(
        "--download-missing-pdfs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download repo-linked PDFs from arXiv when they are not already local.",
    )
    ap.add_argument("--download-timeout", type=int, default=60, help="HTTP timeout in seconds for missing PDF downloads.")
    ap.add_argument(
        "--download-delay",
        type=float,
        default=DEFAULT_DOWNLOAD_DELAY,
        help="Delay in seconds between newly downloaded PDFs.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    papers_dir = _resolve_data_dir(args.papers_dir, "pdf_structured_*.jsonl")
    repos_dir = _resolve_data_dir(args.repos_dir, "repo_chunks_*.jsonl")
    pdf_root = Path(args.pdf_dir)
    active_pdf_root = _download_root(pdf_root) if args.download_missing_pdfs else pdf_root
    if args.download_missing_pdfs and active_pdf_root != pdf_root:
        print(f"[status] pdf download root {pdf_root} is not writable; falling back to {active_pdf_root}")
    chunks = _load_repo_chunks(repos_dir, args.max_repos, seed=args.seed + 1)
    repo_profiles = _build_repo_profiles(chunks)
    repo_hint_index = _build_repo_hint_index(repo_profiles)
    if args.max_papers:
        papers = _load_raw_repo_linked_papers(
            repo_hint_index,
            repo_profiles,
            set(),
            args.max_papers,
            pdf_root=pdf_root,
            download_missing=args.download_missing_pdfs,
            download_timeout=args.download_timeout,
            download_delay=args.download_delay,
        )
    else:
        papers = _load_raw_repo_linked_papers(
            repo_hint_index,
            repo_profiles,
            set(),
            0,
            pdf_root=pdf_root,
            download_missing=args.download_missing_pdfs,
            download_timeout=args.download_timeout,
            download_delay=args.download_delay,
        )
    target_structured_records = 0 if not args.max_papers else max(0, args.max_papers - len(papers))
    if target_structured_records > 0:
        papers.extend(
            _load_structured_papers(
                papers_dir,
                target_structured_records,
                seed=args.seed,
                repo_profiles=repo_profiles,
                repo_hint_index=repo_hint_index,
                existing_ids={paper.paper_id for paper in papers},
            )
        )
    if args.max_papers:
        papers = papers[: args.max_papers]
    print(f"[status] loaded papers={len(papers)} from {papers_dir}, repos={len(chunks)} from {repos_dir}")
    if not papers or not chunks:
        print("[warn] missing papers or repos; run preprocess_pdfs and preprocess_repos first, or check paths.")
        return

    samples = build_alignment(
        papers,
        chunks,
        top_k=args.top_k,
        negatives=args.negatives,
        min_score=args.min_score,
        candidate_pool=args.candidate_pool,
        max_postings=args.max_postings,
        min_shared_terms=args.min_shared_terms,
        max_positives_per_repo_chunk=args.max_positives_per_repo_chunk,
        max_positives_per_repo=args.max_positives_per_repo,
        seed=args.seed,
        repo_profiles=repo_profiles,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in samples:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    _print_summary(samples)
    print(f"[done] wrote {len(samples)} pairs to {out_path}")


if __name__ == "__main__":
    main()
