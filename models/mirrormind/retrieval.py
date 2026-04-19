"""
Coarse lane-based retrieval over repos, papers, and paper↔repo span bridges.

This module keeps the lanes separate:
- repo lane: repo-level semantic summaries
- paper lane: coarse paper-level texts from paper↔repo alignment
- bridge: positive paper-span ↔ repo-chunk pairs

Retrieval happens independently per lane, then fuses at the repo level.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import json
import math
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./+-]{2,}")
CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
STOPWORDS = {
    "all",
    "also",
    "and",
    "are",
    "can",
    "code",
    "data",
    "file",
    "for",
    "from",
    "function",
    "functions",
    "image",
    "implementation",
    "into",
    "its",
    "key",
    "keys",
    "main",
    "method",
    "methods",
    "model",
    "models",
    "module",
    "modules",
    "our",
    "paper",
    "repo",
    "repository",
    "system",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "use",
    "uses",
    "using",
    "with",
}


@dataclass
class RepoLaneDoc:
    repo_id: str
    summary_text: str
    key_concepts: List[str]
    time_window: str
    terms: Counter[str]
    concept_terms: set[str]


@dataclass
class PaperLaneDoc:
    paper_id: str
    paper_title: str
    paper_text: str
    repo_id: str
    pdf_path: str
    terms: Counter[str]
    title_terms: set[str]


@dataclass
class BridgeSpanDoc:
    paper_id: str
    paper_title: str
    repo_id: str
    pdf_path: str
    repo_path: str
    repo_offset: int
    paragraph_id: int
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    paper_text: str
    repo_text: str
    shared_terms: List[str]
    score_hint: float
    paper_terms: Counter[str]
    bridge_terms: set[str]


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


def _split_terms(text: str) -> List[str]:
    prepared = CAMEL_RE.sub(r"\1 \2", text or "")
    out: List[str] = []
    for match in WORD_RE.finditer(prepared):
        raw = match.group(0).lower().strip("._/-+")
        for piece in re.split(r"[._/\-+]+", raw):
            if len(piece) < 3 or piece.isdigit() or piece in STOPWORDS:
                continue
            out.append(piece)
    return out


def _term_counts(text: str, *, max_terms: int = 256) -> Counter[str]:
    counts: Counter[str] = Counter()
    total = 0
    for term in _split_terms(text):
        counts[term] += 1
        total += 1
        if total >= max_terms:
            break
    return counts


def _build_idf(term_bags: Sequence[Counter[str]]) -> Dict[str, float]:
    df: Counter[str] = Counter()
    total = max(1, len(term_bags))
    for bag in term_bags:
        df.update(bag.keys())
    return {term: math.log((1.0 + total) / (1.0 + freq)) + 1.0 for term, freq in df.items()}


def _weighted_total(counts: Counter[str], idf: Dict[str, float]) -> float:
    total = 0.0
    for term, count in counts.items():
        total += float(count) * idf.get(term, 1.0)
    return total or 1.0


def _score_bag_overlap(
    query_counts: Counter[str],
    doc_counts: Counter[str],
    *,
    idf: Dict[str, float],
    preferred_terms: Optional[set[str]] = None,
) -> tuple[float, List[str]]:
    shared = [term for term in query_counts if term in doc_counts]
    if not shared:
        return 0.0, []
    shared.sort(key=lambda term: query_counts[term] * idf.get(term, 1.0), reverse=True)
    overlap_weight = sum(float(query_counts[t]) * idf.get(t, 1.0) for t in shared)
    lexical = overlap_weight / _weighted_total(query_counts, idf)
    doc_precision = overlap_weight / _weighted_total(doc_counts, idf)
    preferred_hits = len(set(shared) & (preferred_terms or set()))
    preferred_score = preferred_hits / float(max(1, len(preferred_terms or set()))) if preferred_terms else 0.0
    score = 0.65 * lexical + 0.25 * doc_precision + 0.10 * preferred_score
    score += min(0.10, 0.02 * len(shared))
    return score, shared[:12]


def _load_repo_lane(path: Path) -> List[RepoLaneDoc]:
    docs: List[RepoLaneDoc] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            repo_id = str(obj.get("entity_id") or "")
            summary_text = _normalize_space(str(obj.get("summary_text") or ""))
            if not repo_id or not summary_text:
                continue
            key_concepts = [str(x) for x in (obj.get("key_concepts") or []) if str(x)]
            docs.append(
                RepoLaneDoc(
                    repo_id=repo_id,
                    summary_text=summary_text,
                    key_concepts=key_concepts,
                    time_window=str(obj.get("time_window") or ""),
                    terms=_term_counts(summary_text, max_terms=256),
                    concept_terms=set(_split_terms(" ".join(key_concepts))),
                )
            )
    return docs


def _load_paper_lane(path: Path) -> List[PaperLaneDoc]:
    docs: List[PaperLaneDoc] = []
    seen: set[str] = set()
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
            if not paper_id or not repo_id or paper_id in seen:
                continue
            seen.add(paper_id)
            paper_title = str(obj.get("paper_title") or "")
            paper_text = _normalize_space(str(obj.get("paper_text") or paper_title))
            docs.append(
                PaperLaneDoc(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    paper_text=paper_text,
                    repo_id=repo_id,
                    pdf_path=str(obj.get("pdf_path") or ""),
                    terms=_term_counts(f"{paper_title}\n{paper_text}", max_terms=320),
                    title_terms=set(_split_terms(paper_title)),
                )
            )
    return docs


def _load_bridge_lane(path: Optional[Path]) -> List[BridgeSpanDoc]:
    if path is None or not path.exists():
        return []
    docs: List[BridgeSpanDoc] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("label") or 0) != 1:
                continue
            paper_text = _normalize_space(str(obj.get("paper_text") or ""))
            if not paper_text:
                continue
            docs.append(
                BridgeSpanDoc(
                    paper_id=str(obj.get("paper_id") or ""),
                    paper_title=str(obj.get("paper_title") or ""),
                    repo_id=str(obj.get("repo_id") or ""),
                    pdf_path=str(obj.get("pdf_path") or ""),
                    repo_path=str(obj.get("repo_path") or ""),
                    repo_offset=int(obj.get("repo_offset") or 0),
                    paragraph_id=int(obj.get("paragraph_id") or 0),
                    page_start=int(obj.get("page_start") or 0),
                    page_end=int(obj.get("page_end") or 0),
                    line_start=int(obj.get("line_start") or 0),
                    line_end=int(obj.get("line_end") or 0),
                    paper_text=paper_text,
                    repo_text=_normalize_space(str(obj.get("repo_text") or "")),
                    shared_terms=[str(x) for x in (obj.get("shared_terms") or []) if str(x)],
                    score_hint=float(obj.get("score") or 0.0),
                    paper_terms=_term_counts(paper_text, max_terms=192),
                    bridge_terms=set(_split_terms(" ".join(obj.get("shared_terms") or []))),
                )
            )
    return docs


class CoarseLaneRetriever:
    """
    Late-fusion retriever over repo summaries, paper texts, and paper spans.
    """

    def __init__(
        self,
        *,
        repo_semantic_path: str = "models/exports/semantic_from_chunks.jsonl",
        paper_repo_align_path: str = "exports/paper_repo_align.jsonl",
        paper_repo_span_path: str = "exports/paper_repo_span_align.jsonl",
    ) -> None:
        self.repo_semantic_path = _resolve_data_file(repo_semantic_path)
        self.paper_repo_align_path = _resolve_data_file(paper_repo_align_path)
        self.paper_repo_span_path = _resolve_data_file(paper_repo_span_path)

        self.repo_lane = _load_repo_lane(self.repo_semantic_path)
        self.paper_lane = _load_paper_lane(self.paper_repo_align_path)
        self.bridge_lane = _load_bridge_lane(self.paper_repo_span_path)

        self.repo_idf = _build_idf([doc.terms for doc in self.repo_lane])
        self.paper_idf = _build_idf([doc.terms for doc in self.paper_lane])
        self.bridge_idf = _build_idf([doc.paper_terms for doc in self.bridge_lane]) if self.bridge_lane else {}

        self.paper_by_id = {doc.paper_id: doc for doc in self.paper_lane}
        self.repo_by_id = {doc.repo_id: doc for doc in self.repo_lane}

    def search_repos(self, query: str, *, top_k: int = 5) -> List[Dict[str, object]]:
        query_counts = _term_counts(query, max_terms=96)
        hits: List[Dict[str, object]] = []
        for doc in self.repo_lane:
            score, shared = _score_bag_overlap(
                query_counts,
                doc.terms,
                idf=self.repo_idf,
                preferred_terms=doc.concept_terms,
            )
            if score <= 0:
                continue
            hits.append(
                {
                    "repo_id": doc.repo_id,
                    "score": round(score, 6),
                    "matched_terms": shared,
                    "summary_text": doc.summary_text,
                    "key_concepts": doc.key_concepts,
                    "time_window": doc.time_window,
                }
            )
        hits.sort(key=lambda item: float(item["score"]), reverse=True)
        return hits[:top_k]

    def search_papers(self, query: str, *, top_k: int = 5) -> List[Dict[str, object]]:
        query_counts = _term_counts(query, max_terms=96)
        hits: List[Dict[str, object]] = []
        for doc in self.paper_lane:
            score, shared = _score_bag_overlap(
                query_counts,
                doc.terms,
                idf=self.paper_idf,
                preferred_terms=doc.title_terms,
            )
            if score <= 0:
                continue
            hits.append(
                {
                    "paper_id": doc.paper_id,
                    "paper_title": doc.paper_title,
                    "repo_id": doc.repo_id,
                    "score": round(score, 6),
                    "matched_terms": shared,
                    "paper_text": doc.paper_text,
                    "pdf_path": doc.pdf_path,
                }
            )
        hits.sort(key=lambda item: float(item["score"]), reverse=True)
        return hits[:top_k]

    def search_bridge_spans(
        self,
        query: str,
        *,
        repo_ids: Optional[Sequence[str]] = None,
        paper_ids: Optional[Sequence[str]] = None,
        top_k: int = 6,
    ) -> List[Dict[str, object]]:
        if not self.bridge_lane:
            return []
        query_counts = _term_counts(query, max_terms=96)
        repo_filter = set(repo_ids or [])
        paper_filter = set(paper_ids or [])
        hits: List[Dict[str, object]] = []
        for doc in self.bridge_lane:
            if repo_filter and doc.repo_id not in repo_filter:
                continue
            if paper_filter and doc.paper_id not in paper_filter:
                continue
            score, shared = _score_bag_overlap(
                query_counts,
                doc.paper_terms,
                idf=self.bridge_idf,
                preferred_terms=doc.bridge_terms,
            )
            if score <= 0:
                continue
            score = 0.75 * score + 0.25 * doc.score_hint
            hits.append(
                {
                    "paper_id": doc.paper_id,
                    "paper_title": doc.paper_title,
                    "repo_id": doc.repo_id,
                    "repo_path": doc.repo_path,
                    "repo_offset": doc.repo_offset,
                    "paragraph_id": doc.paragraph_id,
                    "page_start": doc.page_start,
                    "page_end": doc.page_end,
                    "line_start": doc.line_start,
                    "line_end": doc.line_end,
                    "score": round(score, 6),
                    "matched_terms": shared,
                    "shared_terms": doc.shared_terms,
                    "paper_text": doc.paper_text,
                    "repo_text": doc.repo_text,
                    "pdf_path": doc.pdf_path,
                }
            )
        hits.sort(key=lambda item: float(item["score"]), reverse=True)
        return hits[:top_k]

    def retrieve(
        self,
        query: str,
        *,
        top_k_repos: int = 5,
        top_k_papers: int = 5,
        top_k_spans: int = 6,
    ) -> Dict[str, object]:
        repo_hits = self.search_repos(query, top_k=max(8, top_k_repos * 2))
        paper_hits = self.search_papers(query, top_k=max(8, top_k_papers * 2))

        candidate_repo_ids = {str(hit["repo_id"]) for hit in repo_hits}
        candidate_repo_ids.update(str(hit["repo_id"]) for hit in paper_hits)
        candidate_paper_ids = {str(hit["paper_id"]) for hit in paper_hits}
        bridge_hits = self.search_bridge_spans(
            query,
            repo_ids=sorted(candidate_repo_ids) if candidate_repo_ids else None,
            paper_ids=sorted(candidate_paper_ids) if candidate_paper_ids else None,
            top_k=max(10, top_k_spans * 2),
        )

        fused_scores: Dict[str, float] = defaultdict(float)
        evidence: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: {"repo_hits": [], "paper_hits": [], "bridge_hits": []})

        for hit in repo_hits:
            repo_id = str(hit["repo_id"])
            fused_scores[repo_id] += float(hit["score"])
            evidence[repo_id]["repo_hits"].append(hit)
        for hit in paper_hits:
            repo_id = str(hit["repo_id"])
            fused_scores[repo_id] += 0.9 * float(hit["score"])
            evidence[repo_id]["paper_hits"].append(hit)
        for hit in bridge_hits:
            repo_id = str(hit["repo_id"])
            fused_scores[repo_id] += 0.6 * float(hit["score"])
            evidence[repo_id]["bridge_hits"].append(hit)

        fused_repos: List[Dict[str, object]] = []
        for repo_id, fused_score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
            repo_doc = self.repo_by_id.get(repo_id)
            fused_repos.append(
                {
                    "repo_id": repo_id,
                    "score": round(fused_score, 6),
                    "summary_text": repo_doc.summary_text if repo_doc else "",
                    "key_concepts": repo_doc.key_concepts if repo_doc else [],
                    "evidence": evidence[repo_id],
                }
            )
        fused_repos = fused_repos[:top_k_repos]

        chosen_repo_ids = {str(item["repo_id"]) for item in fused_repos}
        chosen_paper_ids = {str(hit["paper_id"]) for hit in paper_hits[:top_k_papers]}
        support_spans = self.search_bridge_spans(
            query,
            repo_ids=sorted(chosen_repo_ids) if chosen_repo_ids else None,
            paper_ids=sorted(chosen_paper_ids) if chosen_paper_ids else None,
            top_k=top_k_spans,
        )

        return {
            "query": query,
            "repo_hits": repo_hits[:top_k_repos],
            "paper_hits": paper_hits[:top_k_papers],
            "fused_repos": fused_repos,
            "support_spans": support_spans,
        }


__all__ = ["CoarseLaneRetriever"]
