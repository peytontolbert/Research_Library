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
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

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
from modules.arxiv_library import search_keyword as arxiv_search_keyword  # type: ignore
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


app = FastAPI(title="Repository Library", version="0.1.0")

logger = logging.getLogger("repository_library.server")


_LLM_MODEL = None
_LLM_TOKENIZER = None
_QA_INDEX_CACHE: Dict[str, Any] = {}


ARXIV_PDF_ROOT = Path("/arxiv/pdfs")


def _download_arxiv_pdf(arxiv_id: str, *, timeout: int = 60) -> bool:
    """
    Download a single Arxiv PDF by id into ARXIV_PDF_ROOT.

    Returns True if a new file was created, False if it already existed.
    """
    ARXIV_PDF_ROOT.mkdir(parents=True, exist_ok=True)

    # Match the downloader script behavior: use the trailing segment.
    norm_id = str(arxiv_id or "").strip().split("/")[-1]
    if not norm_id:
        raise ValueError("invalid arxiv_id")
    pdf_url = f"https://export.arxiv.org/pdf/{norm_id}.pdf"
    out_path = ARXIV_PDF_ROOT / f"{norm_id}.pdf"

    if out_path.exists():
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
  <title>Repository Library</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e5e7eb; }
    header { padding: 1rem 2rem; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
    header h1 { margin: 0; font-size: 1.2rem; }
    main { display: grid; grid-template-columns: 320px 1fr; height: calc(100vh - 64px); }
    aside { border-right: 1px solid #1f2937; overflow-y: auto; padding: 1rem; }
    section { padding: 1rem 1.5rem; overflow-y: auto; }
    .repo-item { padding: 0.35rem 0.5rem; border-radius: 0.25rem; cursor: pointer; margin-bottom: 0.15rem; }
    .repo-item:hover { background: #111827; }
    .repo-item.active { background: #1e293b; }
    .repo-id { font-size: 0.85rem; font-weight: 600; }
    .repo-meta { font-size: 0.75rem; color: #9ca3af; }
    .pill { display: inline-flex; align-items: center; padding: 0.1rem 0.45rem; border-radius: 999px; font-size: 0.7rem; background: #1e293b; color: #9ca3af; margin-left: 0.25rem; }
    label { display: block; font-size: 0.8rem; margin-top: 0.5rem; margin-bottom: 0.1rem; color: #9ca3af; }
    input[type="text"], textarea, select { width: 100%; background: #020617; border: 1px solid #1f2937; border-radius: 0.25rem; color: #e5e7eb; padding: 0.35rem 0.5rem; font-size: 0.85rem; }
    textarea { min-height: 72px; resize: vertical; }
    button { margin-top: 0.5rem; padding: 0.35rem 0.75rem; border-radius: 0.25rem; border: none; cursor: pointer; font-size: 0.8rem; background: #2563eb; color: #e5e7eb; }
    button.secondary { background: #111827; border: 1px solid #1f2937; margin-left: 0.5rem; }
    pre { background: #020617; border-radius: 0.25rem; padding: 0.5rem 0.75rem; font-size: 0.78rem; overflow-x: auto; }
    .row { display: flex; gap: 0.75rem; }
    .row > div { flex: 1; }
    #graph-container { height: 420px; background: radial-gradient(circle at top left, #020617 0, #020617 40%, #020814 100%); border-radius: 0.5rem; border: 1px solid #111827; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    #graph-meta { margin-top: 0.35rem; font-size: 0.75rem; color: #9ca3af; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #graph-controls { display:flex; align-items:center; gap:0.35rem; font-size:0.75rem; }
    #graph-controls select { background:#020617; border:1px solid #1f2937; border-radius:0.25rem; color:#e5e7eb; padding:0.2rem 0.4rem; font-size:0.75rem; }
    #graph-filter { max-width: 220px; }
    .faded { opacity: 0.15; transition opacity 0.15s ease-out; }
  </style>
  <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
</head>
<body>
  <header>
    <div style="display:flex;flex-direction:column;gap:0.2rem;">
      <h1>Repository Library</h1>
      <span style="font-size:0.8rem;color:#9ca3af;">FastAPI · JSON APIs · Skill planning</span>
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.8rem;">
      <span style="color:#9ca3af;">Library</span>
      <select id="library-kind" style="width:auto;" onchange="onLibraryChange()">
        <option value="repositories" selected>Repositories</option>
        <option value="arxiv">Arxiv</option>
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
        <span style="font-size:0.8rem;color:#9ca3af;">Repositories</span>
        <button class="secondary" onclick="loadRepos()">Refresh</button>
      </div>
      <div id="repo-list"></div>
    </aside>
    <section id="repo-main-section">
      <div id="repo-panel" style="margin-bottom:1rem;">
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
    <div id="arxiv-panel-root" style="display:none; padding:1rem 1.5rem; grid-column:1 / span 2;">
      <h2 style="font-size:0.95rem;margin:0 0 0.5rem 0;">Arxiv metadata</h2>
      <p style="font-size:0.8rem;color:#9ca3af;margin:0 0 0.5rem 0;">
        Search the local Arxiv metadata snapshot (titles, abstracts, authors). Results indicate which papers already have a local PDF stored under your library.
      </p>
      <div style="display:flex;gap:0.75rem;align-items:flex-end;flex-wrap:wrap;">
        <div style="flex:1;min-width:220px;">
          <label for="arxiv-query">Query</label>
          <input id="arxiv-query" type="text" placeholder="Search titles, abstracts, or authors…" />
        </div>
        <div style="flex:0.6;min-width:180px;">
          <label for="arxiv-category">Category prefix (optional)</label>
          <input id="arxiv-category" type="text" placeholder="e.g. cs.LG or cs." />
        </div>
        <div style="display:flex;gap:0.5rem;margin-top:0.35rem;">
          <button onclick="runArxivSearch()">Search Arxiv</button>
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
        <div style="flex:1;max-height:420px;overflow-y:auto;border:1px solid #1f2937;border-radius:0.25rem;">
          <ul id="arxiv-list" style="list-style:none;margin:0;padding:0;"></ul>
        </div>
        <div style="flex:1;">
          <h3 style="font-size:0.9rem;margin:0 0 0.35rem 0;">Paper details</h3>
          <div style="display:flex;gap:0.5rem;margin-bottom:0.35rem;align-items:center;flex-wrap:wrap;">
            <button class="secondary" onclick="downloadSingleArxivPdf()">Download selected PDF</button>
            <button id="arxiv-open-pdf" class="secondary" type="button" style="font-size:0.8rem;align-self:center;">
              Open PDF in viewer
            </button>
          </div>
          <pre id="arxiv-detail">{}</pre>
          <div id="arxiv-pdf-viewer-container" style="margin-top:0.5rem;border:1px solid #1f2937;border-radius:0.25rem;overflow:hidden;display:none;height:360px;">
            <iframe id="arxiv-pdf-viewer" src="" style="width:100%;height:100%;border:0;background:#030712;"></iframe>
          </div>
        </div>
      </div>
    </div>
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

    function setActiveLibrary(kind) {
      const aside = document.querySelector('main > aside');
      const repoSection = document.getElementById('repo-main-section');
      const arxivPanel = document.getElementById('arxiv-panel-root');
      const algoPanel = document.getElementById('algorithms-panel-root');
      if (!aside || !repoSection || !arxivPanel || !algoPanel) return;
      if (kind === 'arxiv') {
        aside.style.display = 'none';
        repoSection.style.display = 'none';
        arxivPanel.style.display = 'block';
        algoPanel.style.display = 'none';
        // Clear any stale details prompt when switching in.
        const detailEl = document.getElementById('arxiv-detail');
        if (detailEl) {
          detailEl.textContent = '{}';
        }
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
      }
    }

    function onLibraryChange() {
      const sel = document.getElementById('library-kind');
      const kind = sel ? (sel.value || 'repositories') : 'repositories';
      setActiveLibrary(kind);
    }

    async function loadRepos() {
      const res = await fetch('/api/repos');
      const data = await res.json();
      repos = data.repos || [];
      const listEl = document.getElementById('repo-list');
      listEl.innerHTML = '';
      repos.forEach(r => {
        const div = document.createElement('div');
        div.className = 'repo-item' + (r.repo_id === activeRepo ? ' active' : '');
        div.onclick = () => selectRepo(r.repo_id);
        div.innerHTML = '<div class="repo-id">' + r.repo_id + '</div>' +
                        '<div class="repo-meta">' + (r.branch || '-') + ' · ' + (r.head || '').slice(0,7) + '</div>';
        listEl.appendChild(div);
      });
    }

    // Expose for inline onclick handlers.
    window.loadRepos = loadRepos;

    async function runArxivSearch() {
      const qEl = document.getElementById('arxiv-query');
      const cEl = document.getElementById('arxiv-category');
      const listEl = document.getElementById('arxiv-list');
      const detailEl = document.getElementById('arxiv-detail');
      const statusEl = document.getElementById('arxiv-status');
      const query = qEl ? (qEl.value || '').trim() : '';
      if (!query) {
        alert('Enter a query for Arxiv search.');
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
      if (detailEl) {
        detailEl.textContent = '{}';
      }
      if (statusEl) {
        statusEl.textContent = 'Searching Arxiv…';
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
        const total = typeof data.count === 'number' ? data.count : (arxivResults.length || 0);
        const withPdf = arxivResults.filter(p => p && p.has_pdf).length;
        if (statusEl) {
          statusEl.textContent =
            'Found ' + total + ' article' + (total === 1 ? '' : 's') +
            (total ? ' · ' + withPdf + ' with local PDFs' : '');
        }
        renderArxivList(arxivResults);
      } catch (err) {
        console.error('arxiv search error', err);
        if (listEl) {
          listEl.innerHTML = '<li style="padding:0.5rem;font-size:0.8rem;color:#f97316;">Error running Arxiv search.</li>';
        }
        if (statusEl) {
          statusEl.textContent = 'Search failed – see console for details.';
        }
      }
    }

    function renderArxivList(items) {
      const listEl = document.getElementById('arxiv-list');
      const detailEl = document.getElementById('arxiv-detail');
      const pageMetaEl = document.getElementById('arxiv-page-meta');
      if (!listEl) return;
      listEl.innerHTML = '';
      if (!items.length) {
        listEl.innerHTML = '<li style="padding:0.5rem;font-size:0.8rem;color:#9ca3af;">No results.</li>';
        if (detailEl) {
          detailEl.textContent = '{}';
        }
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
        const li = document.createElement('li');
        li.style.padding = '0.4rem 0.55rem';
        li.style.cursor = 'pointer';
        li.style.borderBottom = '1px solid #111827';
        li.onmouseenter = () => { li.style.background = '#111827'; };
        li.onmouseleave = () => { li.style.background = 'transparent'; };
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
              '<div style="font-size:0.8rem;font-weight:500;">' + title + '</div>' +
              '<div style="font-size:0.75rem;color:#9ca3af;">' + pid + (cats ? ' · ' + cats : '') + '</div>' +
            '</div>' +
            '<div style="font-size:0.7rem;color:' + badgeColor + ';white-space:nowrap;">' +
              '<span style="display:inline-flex;align-items:center;gap:0.25rem;">' +
                '<span style="width:7px;height:7px;border-radius:999px;background:' + badgeColor + ';"></span>' +
                badgeLabel +
              '</span>' +
            '</div>' +
          '</div>';
        listEl.appendChild(li);
      });
      if (detailEl) {
        detailEl.textContent = 'Select a paper from the list to view details and PDF link.';
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
      const pid = paper.id || '';
      const pdfUrl = pid ? ('/api/arxiv/pdf/' + encodeURIComponent(pid)) : null;
       const hasPdf = !!paper.has_pdf;
      activeArxivPaperId = pid || null;
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
      const summary = {
        id: paper.id || '',
        title: paper.title || '',
        authors: paper.authors || '',
        categories: paper.categories || '',
        abstract: paper.abstract || '',
        has_local_pdf: hasPdf,
        pdf_url_if_downloaded: hasPdf && pdfUrl ? pdfUrl : null,
      };
      detailEl.textContent = JSON.stringify(summary, null, 2);
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
          showArxivDetail(paper);
        }
      } catch (err) {
        console.error('arxiv download error', err);
        alert('Download failed due to a network or server error.');
      }
    }

    async function downloadAllArxivPdfs() {
      if (!arxivResults || !arxivResults.length) {
        alert('Run an Arxiv search first.');
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
        el.classList.toggle('active', el.querySelector('.repo-id').textContent === repoId);
      });
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
      document.getElementById('interaction-output').textContent = JSON.stringify(data, null, 2);
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
            }
        )
    return {"repos": out}


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
    results_with_pdf: List[Dict[str, object]] = []
    for rec in results:
        rec_id = str(rec.get("id") or "").strip()
        pdf_id = rec_id.split("/")[-1] if rec_id else ""
        has_pdf = bool(pdf_id and (ARXIV_PDF_ROOT / f"{pdf_id}.pdf").is_file())
        enriched = dict(rec)
        enriched["has_pdf"] = has_pdf
        results_with_pdf.append(enriched)

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

    The downloader script stores PDFs under `/arxiv/pdfs/{id}.pdf`, where
    `id` is typically the trailing segment of the Arxiv identifier.
    """
    # Normalize to the trailing segment to match the downloader's convention.
    pdf_id = str(paper_id or "").strip().split("/")[-1]
    if not pdf_id:
        raise HTTPException(status_code=400, detail="invalid paper_id")
    pdf_path = ARXIV_PDF_ROOT / f"{pdf_id}.pdf"
    if not pdf_path.is_file():
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

        pdf_path = ARXIV_PDF_ROOT / f"{norm_id}.pdf"
        if pdf_path.is_file():
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



