"""
Paper ↔ Repo alignment preprocessing.

Builds heuristic alignment pairs between structured PDF text and repo code chunks,
emitting a JSONL at exports/paper_repo_align.jsonl with fields:
  {
    "paper_text": "...",
    "repo_text": "...",
    "label": 1 or 0,
    "score": float
  }

Positives are high-overlap pairs; negatives are random mismatches.

Usage:
  PYTHONPATH=.. python -m models.scripts.preprocess_alignment --max-papers 5000 --max-repos 5000 --out exports/paper_repo_align.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def _load_structured_papers(shard_dir: Path, max_records: int) -> List[str]:
    texts: List[str] = []
    count = 0
    for shard in sorted(shard_dir.glob("pdf_structured_*.jsonl")):
        if count >= max_records:
            break
        try:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    if count >= max_records:
                        break
                    obj = json.loads(line)
                    tokens = obj.get("tokens")
                    if tokens and isinstance(tokens, list):
                        text = " ".join(t.get("text", "") for t in tokens if isinstance(t, dict))
                    else:
                        text = obj.get("text") or obj.get("content") or ""
                    text = str(text).strip()
                    if text:
                        texts.append(text)
                        count += 1
        except Exception:
            continue
    return texts


def _load_repo_chunks(chunks_dir: Path, max_records: int) -> List[str]:
    texts: List[str] = []
    count = 0
    for shard in sorted(chunks_dir.glob("repo_chunks_*.jsonl")):
        if count >= max_records:
            break
        try:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    if count >= max_records:
                        break
                    obj = json.loads(line)
                    code = obj.get("code") or obj.get("text") or ""
                    code = str(code).strip()
                    if code:
                        texts.append(code)
                        count += 1
        except Exception:
            continue
    return texts


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split() if t]


def _overlap_score(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def build_alignment(papers: List[str], repos: List[str], top_k: int, negatives: int) -> List[Dict[str, any]]:
    samples: List[Dict[str, any]] = []
    # Positive sampling: for each paper, pick best repo by overlap.
    for p in papers:
        best_score = 0.0
        best_repo = None
        for r in repos:
            s = _overlap_score(p, r)
            if s > best_score:
                best_score = s
                best_repo = r
        if best_repo is not None and best_score > 0:
            samples.append({"paper_text": p, "repo_text": best_repo, "label": 1, "score": best_score})
    # Keep top_k positives
    samples = sorted(samples, key=lambda x: x["score"], reverse=True)[:top_k]

    # Negatives: random mismatches
    random.seed(42)
    for _ in range(negatives):
        p = random.choice(papers)
        r = random.choice(repos)
        samples.append({"paper_text": p, "repo_text": r, "label": 0, "score": 0.0})
    random.shuffle(samples)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=str, default="exports/pdfs_structured", help="Structured PDF shards dir")
    ap.add_argument("--repos-dir", type=str, default="exports/repos_chunks", help="Repo chunks dir")
    ap.add_argument("--out", type=str, default="exports/paper_repo_align.jsonl", help="Output JSONL path")
    ap.add_argument("--max-papers", type=int, default=5000)
    ap.add_argument("--max-repos", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=5000, help="Keep top-k positives")
    ap.add_argument("--negatives", type=int, default=5000, help="Number of negative pairs")
    args = ap.parse_args()

    def _resolve_dir(path_str: str, fallback_subdir: str, pattern: str) -> Path:
        p = Path(path_str)
        root_fallback = Path("/data/repository_library") / fallback_subdir
        def _has_files(path: Path) -> bool:
            return any(path.glob(pattern))
        if _has_files(p):
            return p
        if _has_files(root_fallback):
            return root_fallback
        # if neither has files, prefer existing dir if present
        if p.exists():
            return p
        if root_fallback.exists():
            return root_fallback
        return p

    papers_dir = _resolve_dir(args.papers_dir, "exports/pdfs_structured", "pdf_structured_*.jsonl")
    repos_dir = _resolve_dir(args.repos_dir, "exports/repos_chunks", "repo_chunks_*.jsonl")

    papers = _load_structured_papers(papers_dir, args.max_papers)
    repos = _load_repo_chunks(repos_dir, args.max_repos)
    print(f"[status] loaded papers={len(papers)} from {papers_dir}, repos={len(repos)} from {repos_dir}")
    if not papers or not repos:
        print("[warn] missing papers or repos; run preprocess.pdfs and preprocess.repos first, or check paths.")
        return
    pairs = build_alignment(papers, repos, top_k=args.top_k, negatives=args.negatives)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in pairs:
            f.write(json.dumps(rec) + "\n")
    print(f"[done] wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
