"""
Unified corpus builder for repos and research papers.

This script stitches together existing preprocessors into a training-ready
corpus layout by reading:

- repo chunks from exports/repos_chunks/ (via preprocess_repos.py),
- structured PDFs from exports/pdfs_structured/ (via preprocess_pdfs.py),
- optional paper↔repo alignment pairs from exports/paper_repo_align.jsonl
  (via preprocess_alignment.py).

It emits sharded JSONL files under exports/corpus/ with a simple, explicit
schema so downstream training code can mix and tokenize examples however it
likes without needing to understand the raw export formats.

Schema
------
Each JSONL line is a JSON object with at least:

  {
    "id": str,             # globally unique example id
    "source": str,         # one of: "repo_chunk", "paper_chunk", "paper_repo_pair"
    "text": str,           # primary text payload (code or prose)
    "meta": { ... }        # small metadata blob, source-dependent
  }

For alignment pairs (source == "paper_repo_pair"), the object instead uses:

  {
    "id": str,
    "source": "paper_repo_pair",
    "paper_text": str,
    "repo_text": str,
    "label": int,          # 1 = positive, 0 = negative
    "score": float,        # overlap score (for positives) or 0.0
    "meta": { ... }
  }

This keeps the corpus builder focused on text+metadata assembly; tokenization
and batching are handled by the training stack (see models.shared.training).

Usage
-----
  PYTHONPATH=.. python -m models.scripts.build_corpus \
      --repos-chunks-dir exports/repos_chunks \
      --pdfs-structured-dir exports/pdfs_structured \
      --alignment-path exports/paper_repo_align.jsonl \
      --out-dir exports/corpus
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set


def _hash_text(text: str) -> str:
    """Stable hash for deduplication."""
    h = hashlib.sha1()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _repo_id_from_path(path: str) -> str:
    """
    Best-effort repo identifier from an absolute file path.

    Expected form (as in preprocess_repos.py):
        /data/repositories/<repo_name>/path/to/file.py
    Falls back to the immediate parent directory if the sentinel is absent.
    """
    p = Path(path)
    parts = list(p.parts)
    if "repositories" in parts:
        idx = parts.index("repositories")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return p.parent.name or "unknown"


def _load_repo_licenses(manifest_path: Path) -> Dict[str, str]:
    """
    Best-effort license map from the library manifest, if present.

    Expected shape (per repo entry, where available):
      {
        "license": "mit" | "apache-2.0" | ...,
        "repo_state": {
          "license": "..."
        },
        ...
      }
    """
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    repos = obj.get("repos") or obj
    licenses: Dict[str, str] = {}
    if isinstance(repos, dict):
        for rid, meta in repos.items():
            if not isinstance(meta, dict):
                continue
            lic = meta.get("license") or meta.get("repo_state", {}).get("license")
            if isinstance(lic, str) and lic:
                licenses[str(rid)] = lic.lower()
    return licenses


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

_DOMAIN_KEYWORDS_CACHE: Optional[Dict[str, List[str]]] = None


def _scrub_pii(text: str) -> str:
    """
    Lightweight PII scrubbing for emails, card-like numbers, and IPv4 addresses.

    This is intentionally simple and conservative; the goal is to avoid
    obviously sensitive strings in the training corpus while keeping the
    surrounding technical content.
    """
    if not text:
        return text
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _CARD_RE.sub("<CARD>", text)
    text = _IP_RE.sub("<IP>", text)
    return text


def _infer_domains_from_text(text: str) -> List[str]:
    """
    Infer coarse domains from raw text using keyword heuristics.

    This is intentionally simple and mirrors the DomainGraph's domain
    pseudo-node assignment so that downstream models can use the same
    domain labels for routing or evaluation.
    """
    if not text:
        return []

    global _DOMAIN_KEYWORDS_CACHE
    if _DOMAIN_KEYWORDS_CACHE is None:
        # Load from the same JSON file used by DomainGraph, with a safe fallback.
        default: Dict[str, List[str]] = {
            "deep-learning": ["transformer", "attention", "llm", "bert", "gpt", "diffusion", "neural", "cnn"],
            "compilers": ["compiler", "llvm", "bytecode", "parser", "interpreter"],
            "rl-control": ["reinforcement", "rl", "reward", "policy", "environment", "gym", "agent", "trajectory", "control"],
        }
        path = Path("models/mirrormind/domain_keywords.json")
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    parsed: Dict[str, List[str]] = {}
                    for dom, kws in obj.items():
                        if not isinstance(dom, str) or not isinstance(kws, list):
                            continue
                        parsed[dom] = [str(k).lower() for k in kws if isinstance(k, (str, bytes))]
                    if parsed:
                        _DOMAIN_KEYWORDS_CACHE = parsed
                    else:
                        _DOMAIN_KEYWORDS_CACHE = default
                else:
                    _DOMAIN_KEYWORDS_CACHE = default
            except Exception:
                _DOMAIN_KEYWORDS_CACHE = default
        else:
            _DOMAIN_KEYWORDS_CACHE = default

    name_l = text.lower()
    domains: List[str] = []
    for dom, kws in (_DOMAIN_KEYWORDS_CACHE or {}).items():
        if any((kw in name_l) for kw in kws):
            domains.append(dom)
    # De-duplicate while preserving order.
    seen: Set[str] = set()
    uniq: List[str] = []
    for d in domains:
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq


def iter_repo_examples(
    chunks_dir: Path,
    *,
    license_allow: Optional[Set[str]] = None,
    license_deny: Optional[Set[str]] = None,
    manifest_path: Path = Path("/data/repository_library/exports/_manifest.json"),
) -> Iterator[Dict]:
    """
    Yield repo code/doc chunks as corpus examples.

    Input: JSONL files repo_chunks_*.jsonl with at least:
      - path: file path
      - offset: character offset
      - code: text payload
    """
    licenses = _load_repo_licenses(manifest_path)
    for shard in sorted(chunks_dir.glob("repo_chunks_*.jsonl")):
        with shard.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = (obj.get("code") or obj.get("text") or "").strip()
                if not text:
                    continue
                path = str(obj.get("path") or "")
                repo_id = _repo_id_from_path(path)
                # License-based inclusion/exclusion if configured.
                lic = licenses.get(repo_id)
                if license_allow and (not lic or lic not in license_allow):
                    continue
                if license_deny and lic and lic in license_deny:
                    continue
                # Scrub basic PII before writing to the corpus.
                text = _scrub_pii(text)
                domains = _infer_domains_from_text(text)
                offset = int(obj.get("offset") or 0)
                meta = {
                    "repo_id": repo_id,
                    "path": path,
                    "offset": offset,
                    "domains": domains,
                }
                yield {
                    "id": f"repo:{repo_id}:{_hash_text(path + '|' + str(offset))}",
                    "source": "repo_chunk",
                    "text": text,
                    "meta": meta,
                }


def _tokens_to_text(tokens: List[dict]) -> str:
    """
    Convert a structured PDF token list into flat text.

    The exact schema comes from models.tier3_pdf.pdf_tokenization; we only
    require that each token is a dict with a "text" field.
    """
    parts: List[str] = []
    for t in tokens:
        if not isinstance(t, dict):
            continue
        s = str(t.get("text") or "").strip()
        if not s:
            continue
        parts.append(s)
    return " ".join(parts)


def iter_paper_examples(pdfs_structured_dir: Path) -> Iterator[Dict]:
    """
    Yield paper chunks as corpus examples.

    Input: JSONL files pdf_structured_*.jsonl with fields:
      - pdf_path: original PDF path
      - tokens: list of token dicts (see _tokens_to_text)
    """
    for shard in sorted(pdfs_structured_dir.glob("pdf_structured_*.jsonl")):
        with shard.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pdf_path = str(obj.get("pdf_path") or "")
                tokens = obj.get("tokens") or []
                text = obj.get("text") or obj.get("content") or ""
                if not text and tokens:
                    text = _tokens_to_text(tokens)
                text = str(text).strip()
                if not text:
                    continue
                text = _scrub_pii(text)
                text = _scrub_pii(text)
                domains = _infer_domains_from_text(text)
                meta = {
                    "pdf_path": pdf_path,
                    "domains": domains,
                }
                yield {
                    "id": f"paper:{_hash_text(pdf_path)}",
                    "source": "paper_chunk",
                    "text": text,
                    "meta": meta,
                }


def iter_alignment_examples(alignment_path: Path) -> Iterator[Dict]:
    """
    Yield paper↔repo alignment pairs as corpus examples (if file exists).

    Input: JSONL records with at least:
      - paper_text: str
      - repo_text: str
      - label: int (1 or 0)
      - score: float
    """
    if not alignment_path.exists():
        return
    with alignment_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            paper_text = str(obj.get("paper_text") or "").strip()
            repo_text = str(obj.get("repo_text") or "").strip()
            if not paper_text or not repo_text:
                continue
            paper_text = _scrub_pii(paper_text)
            repo_text = _scrub_pii(repo_text)
            domains = _infer_domains_from_text(paper_text + "\n" + repo_text)
            label = int(obj.get("label") or 0)
            score = float(obj.get("score") or 0.0)
            meta = {
                "label": label,
                "score": score,
                "domains": domains,
            }
            yield {
                "id": f"pair:{_hash_text(paper_text[:128] + '||' + repo_text[:128])}",
                "source": "paper_repo_pair",
                "paper_text": paper_text,
                "repo_text": repo_text,
                "label": label,
                "score": score,
                "meta": meta,
            }


def _write_sharded(
    examples: Iterable[Dict],
    out_dir: Path,
    prefix: str,
    shard_size: int,
    dedup: bool = True,
    dedup_on: str = "text",
) -> int:
    """
    Write examples to JSONL shards with basic content deduplication.

    Returns the total number of examples written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    shard_idx = 0
    shard: List[Dict] = []
    seen_hashes: set[str] = set()

    for ex in examples:
        payload = ex.get(dedup_on)
        if dedup and isinstance(payload, str):
            h = _hash_text(payload)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        shard.append(ex)
        if len(shard) >= shard_size:
            out_path = out_dir / f"{prefix}_{shard_idx:05d}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for rec in shard:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(shard)
            shard = []
            shard_idx += 1

    if shard:
        out_path = out_dir / f"{prefix}_{shard_idx:05d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in shard:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += len(shard)

    return total


def build_corpus(
    repos_chunks_dir: Path,
    pdfs_structured_dir: Path,
    alignment_path: Optional[Path],
    out_dir: Path,
    shard_size: int = 10_000,
    license_allow: Optional[Set[str]] = None,
    license_deny: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """
    Build sharded corpora for repos, papers, and (optionally) alignment pairs.

    Returns a dict with counts per split, e.g.:
      {"repos": 12345, "papers": 6789, "pairs": 5000}
    """
    counts: Dict[str, int] = {}

    if repos_chunks_dir.exists():
        repos_out = out_dir / "repos"
        n_repos = _write_sharded(
            iter_repo_examples(repos_chunks_dir, license_allow=license_allow, license_deny=license_deny),
            repos_out,
            "repo",
            shard_size,
        )
        counts["repos"] = n_repos

    if pdfs_structured_dir.exists():
        papers_out = out_dir / "papers"
        n_papers = _write_sharded(iter_paper_examples(pdfs_structured_dir), papers_out, "paper", shard_size)
        counts["papers"] = n_papers

    if alignment_path is not None and alignment_path.exists():
        pairs_out = out_dir / "pairs"
        n_pairs = _write_sharded(iter_alignment_examples(alignment_path), pairs_out, "pair", shard_size, dedup=False)
        counts["pairs"] = n_pairs

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified corpora from repos, papers, and alignment pairs.")
    ap.add_argument(
        "--repos-chunks-dir",
        type=str,
        default="exports/repos_chunks",
        help="Directory containing repo_chunks_*.jsonl (from preprocess_repos.py).",
    )
    ap.add_argument(
        "--pdfs-structured-dir",
        type=str,
        default="exports/pdfs_structured",
        help="Directory containing pdf_structured_*.jsonl (from preprocess_pdfs.py).",
    )
    ap.add_argument(
        "--alignment-path",
        type=str,
        default="exports/paper_repo_align.jsonl",
        help="Optional paper↔repo alignment JSONL (from preprocess_alignment.py).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="exports/corpus",
        help="Output directory for sharded corpora.",
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=10_000,
        help="Maximum records per JSONL shard.",
    )
    ap.add_argument(
        "--license-allow",
        nargs="+",
        default=None,
        help="Optional allowlist of repo licenses (lowercased); if set, only repos with these licenses are included.",
    )
    ap.add_argument(
        "--license-deny",
        nargs="+",
        default=None,
        help="Optional denylist of repo licenses (lowercased); if set, repos with these licenses are excluded.",
    )
    args = ap.parse_args()

    repos_chunks_dir = Path(args.repos_chunks_dir)
    pdfs_structured_dir = Path(args.pdfs_structured_dir)
    alignment_path = Path(args.alignment_path) if args.alignment_path else None
    out_dir = Path(args.out_dir)
    license_allow = set(l.lower() for l in (args.license_allow or [])) or None
    license_deny = set(l.lower() for l in (args.license_deny or [])) or None

    counts = build_corpus(
        repos_chunks_dir=repos_chunks_dir,
        pdfs_structured_dir=pdfs_structured_dir,
        alignment_path=alignment_path,
        out_dir=out_dir,
        shard_size=args.shard_size,
        license_allow=license_allow,
        license_deny=license_deny,
    )

    pretty_counts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"[done] built corpus at {out_dir} ({pretty_counts})")


if __name__ == "__main__":
    main()


