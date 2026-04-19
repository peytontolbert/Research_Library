"""
Build coarse paper-span ↔ repo-chunk correlations from the repo-level alignment file.

This stage keeps the existing paper↔repo retrieval dataset unchanged and derives
paragraph-like paper spans from structured PDF shards, then correlates each span
against code/doc chunks inside the already-matched repo.

Output schema:
  {
    "paper_id": "...",
    "paper_title": "...",
    "pdf_path": "...",
    "repo_id": "...",
    "repo_path": "...",
    "repo_offset": 0,
    "paragraph_id": 3,
    "page_start": 1,
    "page_end": 2,
    "line_start": 28,
    "line_end": 35,
    "paper_text": "...",
    "repo_text": "...",
    "label": 1 or 0,
    "score": 0.42,
    "shared_terms": ["...", ...],
    "negative_type": "hard" | null
  }

Usage:
  PYTHONPATH=.. python -m models.scripts.preprocess_alignment_spans \
      --alignment-path exports/paper_repo_align.jsonl \
      --structured-dir exports/pdfs_structured \
      --repos-dir exports/repos_chunks \
      --out exports/paper_repo_span_align.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from models.shared.pdf_utils import extract_pdf_text


REPO_ROOT = Path(__file__).resolve().parents[2]
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./+-]{2,}")
CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
SECTION_RE = re.compile(r"^(?:\d+[\.\)]?|[IVXLC]+[\.\)]?)\s+[A-Z][A-Z0-9\s:-]{2,}$")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "not",
    "our",
    "paper",
    "their",
    "the",
    "this",
    "that",
    "these",
    "those",
    "use",
    "using",
    "with",
}
GENERIC_PATH_FRAGMENTS = (
    "/example/",
    "/examples/",
    "/demo/",
    "/demos/",
    "/docs/",
    "/doc/",
    "/tests/",
    "/test/",
    "/benchmark/",
    "/benchmarks/",
)
GENERIC_FILENAMES = {
    "__init__.py",
    "_version.py",
    "conftest.py",
    "requirements.txt",
    "setup.py",
    "versioneer.py",
}


@dataclass
class RepoChunk:
    repo_id: str
    path: str
    offset: int
    text: str
    terms: set[str]
    path_terms: set[str]
    term_weight: float = 1.0


@dataclass
class PaperParagraph:
    paper_id: str
    paper_title: str
    pdf_path: str
    paragraph_id: int
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    text: str
    title_terms: set[str]
    query_counts: Counter[str]
    query_weights: Dict[str, float] = field(default_factory=dict)
    total_query_weight: float = 0.0


def _candidate_paths(path_str: str) -> List[Path]:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(REPO_ROOT / raw)
        if not str(raw).startswith("models/"):
            candidates.append(REPO_ROOT / "models" / raw)
    out: List[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _resolve_data_dir(path_str: str, pattern: str) -> Path:
    for candidate in _candidate_paths(path_str):
        if candidate.is_dir() and any(candidate.glob(pattern)):
            return candidate
    for candidate in _candidate_paths(path_str):
        if candidate.exists():
            return candidate
    return _candidate_paths(path_str)[0]


def _resolve_data_file(path_str: str) -> Path:
    for candidate in _candidate_paths(path_str):
        if candidate.is_file():
            return candidate
    for candidate in _candidate_paths(path_str):
        if candidate.exists():
            return candidate
    return _candidate_paths(path_str)[0]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _split_terms(text: str) -> Iterator[str]:
    prepared = CAMEL_RE.sub(r"\1 \2", text or "")
    for match in WORD_RE.finditer(prepared):
        raw = match.group(0).lower().strip("._/-+")
        for piece in re.split(r"[._/\-+]+", raw):
            if len(piece) < 3 or piece.isdigit() or piece in STOPWORDS:
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


def _load_alignment_positives(path: Path) -> Dict[str, Dict[str, str]]:
    positives: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("label") or 0) != 1:
                continue
            paper_id = str(obj.get("paper_id") or "")
            repo_id = str(obj.get("repo_id") or "")
            if not paper_id or not repo_id or paper_id in positives:
                continue
            positives[paper_id] = {
                "paper_title": str(obj.get("paper_title") or ""),
                "pdf_path": str(obj.get("pdf_path") or ""),
                "repo_id": repo_id,
                "repo_path": str(obj.get("repo_path") or ""),
                "repo_offset": str(obj.get("repo_offset") or 0),
                "coarse_score": str(obj.get("score") or 0.0),
            }
    return positives


def _load_repo_chunks(chunks_dir: Path, repo_ids: set[str]) -> Dict[str, List[RepoChunk]]:
    grouped: Dict[str, List[RepoChunk]] = defaultdict(list)
    for shard in sorted(chunks_dir.glob("repo_chunks_*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                path = str(obj.get("path") or "")
                if not path:
                    continue
                parts = list(Path(path).parts)
                repo_id = ""
                if "repositories" in parts:
                    idx = parts.index("repositories")
                    if idx + 1 < len(parts):
                        repo_id = parts[idx + 1]
                if repo_id not in repo_ids:
                    continue
                text = _normalize_space(str(obj.get("code") or obj.get("text") or ""))
                if len(text) < 40:
                    continue
                grouped[repo_id].append(
                    RepoChunk(
                        repo_id=repo_id,
                        path=path,
                        offset=int(obj.get("offset") or 0),
                        text=text,
                        terms=set(_term_counts(text)),
                        path_terms=_path_terms(path),
                    )
                )
    return grouped


def _build_idf(chunks_by_repo: Dict[str, List[RepoChunk]]) -> Dict[str, float]:
    df: Counter[str] = Counter()
    total = 0
    for chunks in chunks_by_repo.values():
        for chunk in chunks:
            df.update(chunk.terms)
            total += 1
    total = max(1, total)
    return {term: math.log((1.0 + total) / (1.0 + freq)) + 1.0 for term, freq in df.items()}


def _attach_chunk_weights(chunks_by_repo: Dict[str, List[RepoChunk]], idf: Dict[str, float]) -> None:
    for chunks in chunks_by_repo.values():
        for chunk in chunks:
            chunk.term_weight = sum(idf.get(term, 1.0) for term in chunk.terms if idf.get(term, 1.0) >= 1.4) or 1.0


def _looks_like_heading(line: str) -> bool:
    stripped = _normalize_space(line)
    if not stripped:
        return False
    lower = stripped.lower()
    if lower in {"abstract", "introduction", "references", "related work", "conclusion"}:
        return True
    return bool(SECTION_RE.match(stripped))


def _flush_paragraph(
    paper_id: str,
    paper_title: str,
    pdf_path: str,
    paragraph_id: int,
    current_lines: List[Tuple[int, int, str]],
    current_heading: Optional[str],
) -> Optional[PaperParagraph]:
    if not current_lines:
        return None
    pieces: List[str] = []
    for _page, _line_no, text in current_lines:
        text = _normalize_space(text)
        if not text:
            continue
        if pieces and pieces[-1].endswith("-") and text and text[0].islower():
            pieces[-1] = pieces[-1][:-1] + text
        else:
            pieces.append(text)
    text = _normalize_space(" ".join(pieces))
    if current_heading:
        text = f"{current_heading} {text}".strip()
    if len(text) < 80:
        return None
    query_counts = _term_counts(text, max_terms=256)
    if len(query_counts) < 4:
        return None
    pages = [page for page, _, _ in current_lines]
    line_nos = [line_no for _, line_no, _ in current_lines if line_no > 0]
    return PaperParagraph(
        paper_id=paper_id,
        paper_title=paper_title,
        pdf_path=pdf_path,
        paragraph_id=paragraph_id,
        page_start=min(pages),
        page_end=max(pages),
        line_start=min(line_nos) if line_nos else 0,
        line_end=max(line_nos) if line_nos else 0,
        text=text[:1600],
        title_terms=set(_term_counts(paper_title, max_terms=48)),
        query_counts=query_counts,
    )


def _paragraphs_from_tokens(
    *,
    paper_id: str,
    paper_title: str,
    pdf_path: str,
    tokens: Sequence[dict],
) -> List[PaperParagraph]:
    paragraphs: List[PaperParagraph] = []
    current_lines: List[Tuple[int, int, str]] = []
    current_heading: Optional[str] = None
    paragraph_id = 0

    def flush() -> None:
        nonlocal paragraph_id, current_lines
        paragraph = _flush_paragraph(paper_id, paper_title, pdf_path, paragraph_id, current_lines, current_heading)
        if paragraph is not None:
            paragraphs.append(paragraph)
            paragraph_id += 1
        current_lines = []

    for token in tokens:
        if not isinstance(token, dict):
            continue
        text = str(token.get("text") or "")
        page = int(token.get("page") or 1)
        line_no = int(token.get("line_no") or 0)
        if not _normalize_space(text):
            flush()
            continue
        if _looks_like_heading(text):
            flush()
            current_heading = _normalize_space(text)
            continue
        current_lines.append((page, line_no, text))
        joined = " ".join(part for _, _, part in current_lines)
        if len(joined) >= 1200:
            flush()
    flush()
    return paragraphs


def _load_paper_paragraphs(structured_dir: Path, positives: Dict[str, Dict[str, str]]) -> Dict[str, List[PaperParagraph]]:
    paper_ids = set(positives)
    paragraphs_by_paper: Dict[str, List[PaperParagraph]] = {}
    for shard in sorted(structured_dir.glob("pdf_structured_*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pdf_path = str(obj.get("pdf_path") or "")
                if not pdf_path:
                    continue
                paper_id = Path(pdf_path).stem.split("v", 1)[0]
                if paper_id not in paper_ids or paper_id in paragraphs_by_paper:
                    continue
                tokens = obj.get("tokens") or []
                paragraphs: List[PaperParagraph] = []
                if isinstance(tokens, list) and tokens:
                    paragraphs = _paragraphs_from_tokens(
                        paper_id=paper_id,
                        paper_title=positives[paper_id]["paper_title"],
                        pdf_path=pdf_path,
                        tokens=tokens,
                    )
                if not paragraphs:
                    raw = extract_pdf_text(pdf_path, max_chars=30000)
                    raw_tokens = [
                        {"type": "text", "text": line, "page": 1, "line_no": idx}
                        for idx, line in enumerate(raw.splitlines(), start=1)
                    ]
                    paragraphs = _paragraphs_from_tokens(
                        paper_id=paper_id,
                        paper_title=positives[paper_id]["paper_title"],
                        pdf_path=pdf_path,
                        tokens=raw_tokens,
                    )
                if paragraphs:
                    paragraphs_by_paper[paper_id] = paragraphs
    return paragraphs_by_paper


def _attach_paragraph_weights(paragraphs_by_paper: Dict[str, List[PaperParagraph]], idf: Dict[str, float]) -> None:
    for paragraphs in paragraphs_by_paper.values():
        for paragraph in paragraphs:
            query_weights: Dict[str, float] = {}
            for term, count in paragraph.query_counts.items():
                idf_value = idf.get(term, 1.0)
                if idf_value < 1.2 and term not in paragraph.title_terms:
                    continue
                weight = float(count) * idf_value
                if term in paragraph.title_terms:
                    weight *= 1.5
                query_weights[term] = weight
            paragraph.query_weights = query_weights
            paragraph.total_query_weight = sum(query_weights.values()) or 1.0


def _build_repo_indices(chunks_by_repo: Dict[str, List[RepoChunk]]) -> Dict[str, Dict[str, List[int]]]:
    indices: Dict[str, Dict[str, List[int]]] = {}
    for repo_id, chunks in chunks_by_repo.items():
        index: Dict[str, List[int]] = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            for term in (chunk.terms | chunk.path_terms):
                index[term].append(idx)
        indices[repo_id] = index
    return indices


def _chunk_quality_multiplier(chunk: RepoChunk) -> float:
    path_l = chunk.path.lower()
    mult = 1.0
    if any(fragment in path_l for fragment in GENERIC_PATH_FRAGMENTS):
        mult *= 0.72
    if Path(path_l).name in GENERIC_FILENAMES:
        mult *= 0.42
    return mult


def _score_span_chunk(paragraph: PaperParagraph, chunk: RepoChunk, idf: Dict[str, float], min_shared_terms: int) -> Tuple[float, List[str]]:
    shared = [term for term in paragraph.query_weights if term in chunk.terms or term in chunk.path_terms]
    if len(shared) < min_shared_terms:
        return 0.0, []
    shared.sort(key=lambda term: paragraph.query_weights.get(term, 0.0), reverse=True)
    overlap_weight = sum(paragraph.query_weights.get(term, 0.0) for term in shared)
    lexical = overlap_weight / paragraph.total_query_weight
    shared_idf = sum(idf.get(term, 1.0) for term in shared)
    precision = shared_idf / chunk.term_weight
    title_hits = len(paragraph.title_terms & (chunk.terms | chunk.path_terms))
    title_score = title_hits / float(max(1, len(paragraph.title_terms))) if paragraph.title_terms else 0.0
    path_hits = len(set(shared) & chunk.path_terms)
    path_score = path_hits / float(max(1, len(paragraph.title_terms))) if paragraph.title_terms else 0.0
    specific_hits = sum(1 for term in shared if idf.get(term, 1.0) >= 3.0)
    specificity = specific_hits / float(len(shared))
    score = 0.50 * lexical + 0.23 * precision + 0.12 * title_score + 0.10 * path_score + 0.05 * specificity
    score += min(0.12, 0.02 * len(shared))
    score *= _chunk_quality_multiplier(chunk)
    return score, shared[:16]


def _retrieve_chunk_candidates(
    paragraph: PaperParagraph,
    repo_chunks: Sequence[RepoChunk],
    repo_index: Dict[str, List[int]],
    *,
    idf: Dict[str, float],
    candidate_pool: int,
    max_postings: int,
    min_shared_terms: int,
) -> List[Tuple[float, int, List[str]]]:
    prelim: Dict[int, float] = defaultdict(float)
    for term, weight in paragraph.query_weights.items():
        postings = repo_index.get(term)
        if not postings or len(postings) > max_postings:
            continue
        boost = 1.15 if term in paragraph.title_terms else 1.0
        for idx in postings:
            prelim[idx] += weight * boost
    ranked = sorted(prelim.items(), key=lambda item: item[1], reverse=True)[: candidate_pool * 4]
    rescored: List[Tuple[float, int, List[str]]] = []
    for idx, _ in ranked:
        score, shared = _score_span_chunk(paragraph, repo_chunks[idx], idf=idf, min_shared_terms=min_shared_terms)
        if score <= 0:
            continue
        rescored.append((score, idx, shared))
    rescored.sort(key=lambda item: item[0], reverse=True)
    return rescored[:candidate_pool]


def build_span_alignment(
    positives: Dict[str, Dict[str, str]],
    paragraphs_by_paper: Dict[str, List[PaperParagraph]],
    chunks_by_repo: Dict[str, List[RepoChunk]],
    *,
    idf: Dict[str, float],
    candidate_pool: int,
    max_postings: int,
    min_shared_terms: int,
    min_score: float,
    max_spans_per_paper: int,
    negatives_per_positive: int,
) -> List[Dict[str, object]]:
    repo_indices = _build_repo_indices(chunks_by_repo)
    rows: List[Dict[str, object]] = []

    for paper_id, meta in sorted(positives.items()):
        repo_id = meta["repo_id"]
        repo_chunks = chunks_by_repo.get(repo_id) or []
        repo_index = repo_indices.get(repo_id) or {}
        paragraphs = paragraphs_by_paper.get(paper_id) or []
        if not repo_chunks or not paragraphs:
            continue

        positives_for_paper: List[Tuple[PaperParagraph, float, int, List[str], List[Tuple[float, int, List[str]]]]] = []
        fallback_for_paper: List[Tuple[PaperParagraph, float, int, List[str], List[Tuple[float, int, List[str]]]]] = []
        for paragraph in paragraphs:
            candidates = _retrieve_chunk_candidates(
                paragraph,
                repo_chunks,
                repo_index,
                idf=idf,
                candidate_pool=candidate_pool,
                max_postings=max_postings,
                min_shared_terms=min_shared_terms,
            )
            if not candidates:
                continue
            score, idx, shared_terms = candidates[0]
            if score < min_score:
                if score > 0:
                    fallback_for_paper.append((paragraph, score, idx, shared_terms, candidates))
                continue
            positives_for_paper.append((paragraph, score, idx, shared_terms, candidates))

        if not positives_for_paper and fallback_for_paper:
            fallback_for_paper.sort(key=lambda item: item[1], reverse=True)
            positives_for_paper = fallback_for_paper[: min(3, max_spans_per_paper)]

        positives_for_paper.sort(key=lambda item: item[1], reverse=True)
        positives_for_paper = positives_for_paper[:max_spans_per_paper]

        for paragraph, score, idx, shared_terms, candidates in positives_for_paper:
            chunk = repo_chunks[idx]
            rows.append(
                {
                    "paper_id": paragraph.paper_id,
                    "paper_title": paragraph.paper_title,
                    "pdf_path": paragraph.pdf_path,
                    "repo_id": repo_id,
                    "repo_path": chunk.path,
                    "repo_offset": chunk.offset,
                    "paragraph_id": paragraph.paragraph_id,
                    "page_start": paragraph.page_start,
                    "page_end": paragraph.page_end,
                    "line_start": paragraph.line_start,
                    "line_end": paragraph.line_end,
                    "paper_text": paragraph.text,
                    "repo_text": chunk.text,
                    "label": 1,
                    "score": round(score, 6),
                    "shared_terms": shared_terms,
                    "negative_type": None,
                }
            )

            negatives_written = 0
            for cand_score, cand_idx, cand_shared in candidates[1:]:
                if negatives_written >= negatives_per_positive:
                    break
                neg_chunk = repo_chunks[cand_idx]
                rows.append(
                    {
                        "paper_id": paragraph.paper_id,
                        "paper_title": paragraph.paper_title,
                        "pdf_path": paragraph.pdf_path,
                        "repo_id": repo_id,
                        "repo_path": neg_chunk.path,
                        "repo_offset": neg_chunk.offset,
                        "paragraph_id": paragraph.paragraph_id,
                        "page_start": paragraph.page_start,
                        "page_end": paragraph.page_end,
                        "line_start": paragraph.line_start,
                        "line_end": paragraph.line_end,
                        "paper_text": paragraph.text,
                        "repo_text": neg_chunk.text,
                        "label": 0,
                        "score": round(cand_score, 6),
                        "shared_terms": cand_shared,
                        "negative_type": "hard",
                    }
                )
                negatives_written += 1

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build coarse paper-span ↔ repo-chunk correlations.")
    ap.add_argument("--alignment-path", type=str, default="exports/paper_repo_align.jsonl")
    ap.add_argument("--structured-dir", type=str, default="exports/pdfs_structured")
    ap.add_argument("--repos-dir", type=str, default="exports/repos_chunks")
    ap.add_argument("--out", type=str, default="exports/paper_repo_span_align.jsonl")
    ap.add_argument("--candidate-pool", type=int, default=8)
    ap.add_argument("--max-postings", type=int, default=300)
    ap.add_argument("--min-shared-terms", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.14)
    ap.add_argument("--max-spans-per-paper", type=int, default=12)
    ap.add_argument("--negatives-per-positive", type=int, default=1)
    args = ap.parse_args()

    alignment_path = _resolve_data_file(args.alignment_path)
    structured_dir = _resolve_data_dir(args.structured_dir, "pdf_structured_*.jsonl")
    repos_dir = _resolve_data_dir(args.repos_dir, "repo_chunks_*.jsonl")

    positives = _load_alignment_positives(alignment_path)
    repo_ids = {meta["repo_id"] for meta in positives.values()}
    chunks_by_repo = _load_repo_chunks(repos_dir, repo_ids)
    idf = _build_idf(chunks_by_repo)
    _attach_chunk_weights(chunks_by_repo, idf)
    paragraphs_by_paper = _load_paper_paragraphs(structured_dir, positives)
    _attach_paragraph_weights(paragraphs_by_paper, idf)

    rows = build_span_alignment(
        positives,
        paragraphs_by_paper,
        chunks_by_repo,
        idf=idf,
        candidate_pool=args.candidate_pool,
        max_postings=args.max_postings,
        min_shared_terms=args.min_shared_terms,
        min_score=args.min_score,
        max_spans_per_paper=args.max_spans_per_paper,
        negatives_per_positive=args.negatives_per_positive,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    positives_n = sum(1 for row in rows if int(row.get("label") or 0) == 1)
    negatives_n = len(rows) - positives_n
    print(
        f"[done] wrote {len(rows)} rows to {out_path} "
        f"(positives={positives_n}, negatives={negatives_n}, papers={len(paragraphs_by_paper)})"
    )


if __name__ == "__main__":
    main()
