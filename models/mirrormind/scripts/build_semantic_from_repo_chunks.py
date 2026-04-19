"""
Build coarse semantic summaries directly from repo_chunks JSONL exports.

This builder no longer stores raw concatenated code as "semantic summaries".
Instead it constructs a compact repo digest from README text, file structure,
imports, and symbol names, then optionally lets a local LLM rewrite that digest
 into a cleaner summary when available.

Usage:
    python -m models.mirrormind.scripts.build_semantic_from_repo_chunks \
        --repo-chunks-dir models/exports/repos_chunks \
        --output models/exports/semantic_from_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from models.mirrormind.llm import safe_build_llm
from models.mirrormind.memory import SemanticMemoryStore, SemanticSummary

REPO_ROOT = Path("/data/repositories")
README_NAMES = ("README.md", "README.rst", "README.txt", "README")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]{2,}")
CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import|import\s+([A-Za-z_][A-Za-z0-9_.]*))",
    re.MULTILINE,
)
GENERIC_FILES = {
    "setup.py",
    "versioneer.py",
    "_version.py",
    "conftest.py",
    "__init__.py",
    "pyproject.toml",
    "requirements.txt",
}
GENERIC_DIRS = {
    "tests",
    "test",
    "docs",
    "doc",
    "examples",
    "example",
    "benchmarks",
    "benchmark",
}
STDLIB_IMPORTS = {
    "abc",
    "argparse",
    "collections",
    "contextlib",
    "copy",
    "csv",
    "dataclasses",
    "functools",
    "gzip",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "shutil",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "typing",
    "uuid",
}
STOPWORDS = {
    "all",
    "also",
    "and",
    "are",
    "build",
    "class",
    "code",
    "data",
    "default",
    "def",
    "example",
    "file",
    "files",
    "for",
    "forward",
    "from",
    "function",
    "functions",
    "get",
    "github",
    "implementation",
    "import",
    "into",
    "key",
    "keys",
    "main",
    "main",
    "model",
    "models",
    "module",
    "modules",
    "one",
    "paper",
    "python",
    "repo",
    "repository",
    "script",
    "set",
    "state",
    "support",
    "system",
    "the",
    "this",
    "torch",
    "using",
    "via",
    "with",
    "work",
    "works",
}
GENERIC_SYMBOLS = {
    "backward",
    "build",
    "evaluate",
    "exists",
    "forward",
    "from_config",
    "load",
    "load_state_dict",
    "log",
    "main",
    "process",
    "reset",
    "save",
    "set_input_tensor",
    "wait",
    "with",
}


@dataclass
class RepoMaterial:
    repo_id: str
    time_window: str
    readme_text: str = ""
    file_counts: Counter[str] = field(default_factory=Counter)
    dir_counts: Counter[str] = field(default_factory=Counter)
    ext_counts: Counter[str] = field(default_factory=Counter)
    imports: Counter[str] = field(default_factory=Counter)
    symbols: Counter[str] = field(default_factory=Counter)
    text_terms: Counter[str] = field(default_factory=Counter)


def _repo_id_from_path(path: str) -> str:
    """
    Extract a repo identifier from a repo_chunks path.
    Expected form: /data/repositories/<repo_name>/...
    Falls back to the immediate parent name if the sentinel is absent.
    """
    p = Path(path)
    parts = list(p.parts)
    if "repositories" in parts:
        idx = parts.index("repositories")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return p.parent.name or "unknown"


def _repo_relative_path(path: str, repo_id: str) -> str:
    p = Path(path)
    parts = list(p.parts)
    if "repositories" in parts:
        idx = parts.index("repositories")
        tail = parts[idx + 2 :]
        if tail:
            return Path(*tail).as_posix()
    return p.name if p.name else repo_id


def _load_manifest_meta(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    repos = obj.get("repos")
    return repos if isinstance(repos, dict) else {}


def _time_window_for_repo(repo_id: str, repos_meta: Dict[str, Any]) -> str:
    meta = repos_meta.get(repo_id, {}) if isinstance(repos_meta, dict) else {}
    t_val = meta.get("last_indexed_at") or meta.get("repo_state", {}).get("snapshot_mtime")
    if isinstance(t_val, (int, float)):
        return str(int(t_val))
    return ""


def _read_repo_readme(repo_id: str) -> str:
    repo_root = REPO_ROOT / repo_id
    if not repo_root.is_dir():
        return ""
    try:
        children = sorted(p for p in repo_root.iterdir() if p.is_file() and p.name.lower().startswith("readme"))
    except Exception:
        children = []
    preferred = [repo_root / name for name in README_NAMES]
    for path in preferred + children:
        if not path.exists() or not path.is_file():
            continue
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            try:
                return path.read_bytes().decode("utf-8", errors="ignore")
            except Exception:
                continue
    return ""


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_markdown(text: str) -> str:
    lines: List[str] = []
    in_code = False
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        if stripped.startswith("[![") or stripped.startswith("![]("):
            continue
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
        stripped = re.sub(r"`([^`]+)`", r"\1", stripped)
        stripped = re.sub(r"^#+\s*", "", stripped)
        stripped = _normalize_space(stripped)
        if not stripped:
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _first_readme_summary(text: str) -> str:
    cleaned = _clean_markdown(text)
    if not cleaned:
        return ""
    candidates: List[str] = []
    for line in cleaned.splitlines():
        lower = line.lower()
        if len(line) < 40:
            continue
        if lower.startswith(("installation", "usage", "license", "contributing", "acknowledg")):
            continue
        candidates.append(line)
        if len(candidates) >= 2:
            break
    if not candidates:
        candidates = [cleaned[:320]]
    joined = " ".join(candidates)
    parts = re.split(r"(?<=[.!?])\s+", joined)
    summary = " ".join(part for part in parts[:2] if part)
    return summary[:360].strip()


def _split_terms(text: str) -> List[str]:
    prepared = CAMEL_RE.sub(r"\1 \2", text or "")
    out: List[str] = []
    for match in WORD_RE.finditer(prepared):
        raw = match.group(0).lower().strip("._+-")
        for piece in re.split(r"[._/+:-]+", raw):
            if len(piece) < 3 or piece.isdigit() or piece in STOPWORDS:
                continue
            out.append(piece)
    return out


def _extract_import_roots(text: str) -> List[str]:
    roots: List[str] = []
    for match in IMPORT_RE.finditer(text or ""):
        module = match.group(1) or match.group(2) or ""
        root = module.split(".", 1)[0]
        if not root or root in STDLIB_IMPORTS:
            continue
        roots.append(root)
    return roots


def _extract_symbols(text: str) -> List[str]:
    out: List[str] = []
    for match in DEF_RE.finditer(text or ""):
        symbol = match.group(1)
        if symbol.startswith("_") or len(symbol) < 3 or symbol in GENERIC_SYMBOLS:
            continue
        out.append(symbol)
    return out


def _looks_like_summary(text: str) -> bool:
    text = (text or "").strip()
    if not text or len(text) > 3000:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    code_like = 0
    for line in lines[:10]:
        if line.startswith(("import ", "from ", "def ", "class ", "if __name__", "@")):
            code_like += 1
    if code_like >= 2:
        return False
    if text.count("{") + text.count("}") > 10:
        return False
    if len(text.split()) < 20:
        return False
    return True


def _top_non_generic(counter: Counter[str], *, limit: int, generic: set[str]) -> List[str]:
    picked: List[str] = []
    fallback: List[str] = []
    for value, _count in counter.most_common():
        if value in picked or value in fallback:
            continue
        if value in generic:
            fallback.append(value)
        else:
            picked.append(value)
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for value in fallback:
            if value not in picked:
                picked.append(value)
            if len(picked) >= limit:
                break
    return picked[:limit]


def _build_key_concepts(material: RepoMaterial) -> List[str]:
    counts: Counter[str] = Counter()
    counts.update({term: 5 for term in _split_terms(material.repo_id)})
    counts.update({term: 4 for term in _split_terms(_first_readme_summary(material.readme_text))})
    for rel_path in _top_non_generic(material.file_counts, limit=6, generic=GENERIC_FILES):
        for term in _split_terms(Path(rel_path).stem):
            counts[term] += 3
    for directory in _top_non_generic(material.dir_counts, limit=5, generic=GENERIC_DIRS):
        for term in _split_terms(directory):
            counts[term] += 2
    for symbol, score in material.symbols.most_common(10):
        for term in _split_terms(symbol):
            counts[term] += min(score, 3)
    for name, score in material.imports.most_common(8):
        for term in _split_terms(name):
            counts[term] += min(score, 3)
    for term, score in material.text_terms.most_common(20):
        counts[term] += min(score, 2)

    concepts: List[str] = []
    for term, _ in counts.most_common():
        if term in concepts:
            continue
        concepts.append(term)
        if len(concepts) >= 8:
            break
    return concepts


def _build_repo_brief(material: RepoMaterial) -> str:
    readme_summary = _first_readme_summary(material.readme_text)
    top_files = _top_non_generic(material.file_counts, limit=5, generic=GENERIC_FILES)
    top_dirs = _top_non_generic(material.dir_counts, limit=4, generic=GENERIC_DIRS)
    top_imports = [name for name, _ in material.imports.most_common(6)]
    top_symbols = [name for name, _ in material.symbols.most_common(8)]
    key_concepts = _build_key_concepts(material)

    lines = [f"Repo: {material.repo_id}"]
    if readme_summary:
        lines.append(f"README summary: {readme_summary}")
    if top_dirs:
        lines.append(f"Top directories: {', '.join(top_dirs)}")
    if top_files:
        lines.append(f"Representative files: {', '.join(top_files)}")
    if top_symbols:
        lines.append(f"Key symbols: {', '.join(top_symbols)}")
    if top_imports:
        lines.append(f"External libraries: {', '.join(top_imports)}")
    if key_concepts:
        lines.append(f"Key concepts: {', '.join(key_concepts)}")
    return "\n".join(lines)


def _deterministic_summary(material: RepoMaterial) -> tuple[str, List[str]]:
    readme_summary = _first_readme_summary(material.readme_text)
    key_concepts = _build_key_concepts(material)
    top_files = _top_non_generic(material.file_counts, limit=4, generic=GENERIC_FILES)
    top_dirs = _top_non_generic(material.dir_counts, limit=3, generic=GENERIC_DIRS)
    top_symbols = [name for name, _ in material.symbols.most_common(6)]
    top_imports = [name for name, _ in material.imports.most_common(5)]
    ext_bits = [f"{ext or '[no_ext]'} x{count}" for ext, count in material.ext_counts.most_common(3)]

    purpose = readme_summary
    if not purpose:
        if key_concepts:
            purpose = f"The repository centers on {', '.join(key_concepts[:4])}."
        else:
            purpose = f"The repository groups implementation code under {material.repo_id}."

    lines = [f"Repo: {material.repo_id}"]
    lines.append(f"- Purpose: {purpose}")
    if top_dirs or top_files:
        detail_bits = []
        if top_dirs:
            detail_bits.append(f"main areas include {', '.join(top_dirs)}")
        if top_files:
            detail_bits.append(f"representative files include {', '.join(top_files)}")
        lines.append(f"- Structure: {'; '.join(detail_bits)}.")
    if top_symbols:
        lines.append(f"- Key symbols: {', '.join(top_symbols)}.")
    if top_imports:
        lines.append(f"- Dependencies/frameworks: {', '.join(top_imports)}.")
    if key_concepts:
        lines.append(f"- Key concepts: {', '.join(key_concepts)}.")
    if ext_bits:
        lines.append(f"- Artifact mix: {', '.join(ext_bits)}.")
    return "\n".join(lines), key_concepts


def _render_summary(material: RepoMaterial, summarize_fn: Optional[Callable[[str], str]]) -> tuple[str, List[str]]:
    fallback, key_concepts = _deterministic_summary(material)
    if summarize_fn is None:
        return fallback, key_concepts

    digest = _build_repo_brief(material)
    try:
        candidate = summarize_fn(digest)
    except Exception:
        candidate = ""
    candidate = _normalize_space(candidate)
    if not _looks_like_summary(candidate):
        return fallback, key_concepts
    if not candidate.lower().startswith("repo:"):
        candidate = f"Repo: {material.repo_id}\n- Summary: {candidate}"
    return candidate[:2200], key_concepts


def _collect_repo_material(paths: Sequence[Path]) -> Dict[str, RepoMaterial]:
    manifest_path = Path("/data/repository_library/exports/_manifest.json")
    repos_meta = _load_manifest_meta(manifest_path)
    materials: Dict[str, RepoMaterial] = {}

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue
                repo_id = _repo_id_from_path(obj.get("path", ""))
                rel_path = _repo_relative_path(obj.get("path", ""), repo_id)
                code = obj.get("code") or ""
                material = materials.get(repo_id)
                if material is None:
                    material = RepoMaterial(
                        repo_id=repo_id,
                        time_window=_time_window_for_repo(repo_id, repos_meta),
                        readme_text=_read_repo_readme(repo_id),
                    )
                    materials[repo_id] = material

                if rel_path:
                    material.file_counts[rel_path] += 1
                    p = Path(rel_path)
                    if p.suffix:
                        material.ext_counts[p.suffix.lower()] += 1
                    else:
                        material.ext_counts[""] += 1
                    if len(p.parts) > 1:
                        material.dir_counts[p.parts[0]] += 1
                    else:
                        material.dir_counts["root"] += 1

                if not code:
                    continue

                for symbol in _extract_symbols(code[:12_000]):
                    material.symbols[symbol] += 1
                for name in _extract_import_roots(code[:12_000]):
                    material.imports[name] += 1
                material.text_terms.update(_split_terms(code[:600]))

    return materials


def build_semantic_from_repo_chunks(
    repo_chunks_dir: Path,
    output_path: Path,
    *,
    summarize_fn: Optional[Callable[[str], str]] = None,
) -> SemanticMemoryStore:
    files = [Path(p) for p in glob(str(repo_chunks_dir / "repo_chunks_*.jsonl"))]
    semantic = SemanticMemoryStore()
    for repo_id, material in sorted(_collect_repo_material(files).items()):
        summary_text, key_concepts = _render_summary(material, summarize_fn=summarize_fn)
        semantic.add(
            SemanticSummary(
                id=f"{repo_id}:repo_chunks",
                entity_id=repo_id,
                time_window=material.time_window,
                scope="repo_chunks",
                summary_text=summary_text,
                key_concepts=key_concepts,
                dense=[float(len(summary_text))],
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic.save(output_path)
    return semantic


def _llm_summarizer() -> Optional[Callable[[str], str]]:
    """
    Build an LLM-based summarizer. Returns None if the model cannot be loaded.
    """
    model, tokenizer, _ = safe_build_llm()
    if model is None or tokenizer is None:
        return None

    def _summarize(text: str) -> str:
        prompt = (
            "You are writing coarse repository notes for retrieval.\n"
            "Given the structured repository brief below, produce 4-6 concise bullets.\n"
            "Mention the repo purpose, main modules, notable symbols, and external frameworks.\n"
            "Do not copy code. Do not echo the prompt. Keep the output under 220 words.\n\n"
            f"{text}"
        )
        inputs = tokenizer(prompt[:6000], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=220)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return _summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic memory from repo_chunks JSONL exports.")
    parser.add_argument("--repo-chunks-dir", type=Path, required=True, help="Directory containing repo_chunks_*.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path for semantic summaries.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Use deterministic repository summaries only.",
    )
    args = parser.parse_args()

    summarize_fn = None if args.skip_llm else _llm_summarizer()
    semantic = build_semantic_from_repo_chunks(
        args.repo_chunks_dir,
        args.output,
        summarize_fn=summarize_fn,
    )
    total = sum(len(v) for v in semantic._by_entity.values())
    print(f"Wrote {total} semantic summaries to {args.output}")


if __name__ == "__main__":
    main()
