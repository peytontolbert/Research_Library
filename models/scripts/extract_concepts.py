"""
Concept extraction from precomputed repo exports.

Reads <repo>.entities.jsonl and <repo>.edges.jsonl under /data/repository_library/exports
and emits a JSONL catalog of code concepts with lightweight summaries:

{
  "repo_id": "...",
  "id": "py:module.func",
  "name": "func",
  "kind": "function",
  "owner": "py:module",
  "uri": "program://...#Lx-Ly",
  "doc": "optional docstring or description (placeholder)",
  "relations": [{"type": "owns", "target": "..."}]
}

This is heuristic: it uses entity fields and ownership edges only. It does not
invoke an LLM; docstrings would require reading source files (not included here).
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

EXPORT_ROOT = Path("/data/repository_library/exports")
REPO_ROOT = Path("/data/repositories")
_LLM_CACHE = {}


def load_entities(repo_id: str) -> Dict[str, Dict[str, str]]:
    ent_path = EXPORT_ROOT / repo_id / f"{repo_id}.entities.jsonl"
    entities: Dict[str, Dict[str, str]] = {}
    if not ent_path.exists():
        return entities
    with ent_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            eid = obj.get("id")
            if eid:
                entities[eid] = obj
    return entities


def load_edges(repo_id: str) -> List[Dict[str, str]]:
    edge_path = EXPORT_ROOT / repo_id / f"{repo_id}.edges.jsonl"
    edges: List[Dict[str, str]] = []
    if not edge_path.exists():
        return edges
    with edge_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                edges.append(obj)
    return edges


def load_artifacts(repo_id: str) -> List[Path]:
    art_path = EXPORT_ROOT / repo_id / f"{repo_id}.artifacts.jsonl"
    paths: List[Path] = []
    if not art_path.exists():
        return paths
    with art_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            uri = obj.get("uri")
            if not uri or not uri.startswith("program://"):
                continue
            try:
                rel = uri.split("/artifact/", 1)[1]
            except Exception:
                continue
            paths.append(REPO_ROOT / repo_id / rel)
    return paths


def _parse_uri_to_path(uri: str, artifacts: List[Path]) -> Tuple[Optional[Path], Optional[int], Optional[int]]:
    """Convert program://repo/type/path#Lstart-Lend to a local filesystem path, using artifacts as hints."""
    if not uri or not uri.startswith("program://"):
        return None, None, None
    try:
        rest = uri[len("program://") :]
        if "#" in rest:
            rest, span = rest.split("#", 1)
            span = span.replace("L", "")
            if "-" in span:
                start_str, end_str = span.split("-")
                start = int(start_str)
                end = int(end_str)
            else:
                start = int(span)
                end = start
        else:
            start = end = None
        parts = rest.split("/", 2)
        if len(parts) < 3:
            return None, start, end
        repo_name = parts[0]
        rel_path = parts[2]
        candidates = []
        # Direct path
        candidates.append(REPO_ROOT / repo_name / rel_path)
        # Dot-to-slash variant
        candidates.append(REPO_ROOT / repo_name / rel_path.replace(".", "/"))
        # Try adding .py if missing
        if not rel_path.endswith(".py"):
            candidates.append(REPO_ROOT / repo_name / f"{rel_path}.py")
            candidates.append(REPO_ROOT / repo_name / f"{rel_path.replace('.', '/')}.py")
        # Try basename match against artifacts
        rel_base = Path(rel_path).name
        for art in artifacts:
            if art.name == rel_base or art.name == f"{rel_base}.py":
                candidates.append(art)
            elif rel_base and art.name == f"{rel_base.split('.')[-1]}.py":
                candidates.append(art)
        for c in candidates:
            if c.exists():
                return c, start, end
        return candidates[0], start, end
    except Exception:
        return None, None, None


def _read_source_slice(path: Path, start: Optional[int], end: Optional[int]) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        try:
            text = path.read_bytes().decode("latin-1", errors="ignore")
        except Exception:
            return ""
    if start is not None and end is not None:
        lines = text.splitlines()
        # adjust to 0-based
        snippet = "\n".join(lines[max(0, start - 1) : min(len(lines), end)])
        return snippet
    return text


def _build_signature(func: ast.AST) -> str:
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    parts: List[str] = []
    args = func.args
    # positional-only
    posonly = args.posonlyargs or []
    defaults = list(args.defaults) if args.defaults else []
    pos_args = args.args
    # align defaults to args from the end
    def _fmt_arg(arg_node, default_node=None):
        name = arg_node.arg
        ann = ""
        if arg_node.annotation:
            try:
                ann = f": {ast.unparse(arg_node.annotation)}"
            except Exception:
                ann = ""
        default = ""
        if default_node is not None:
            try:
                default = f"={ast.unparse(default_node)}"
            except Exception:
                default = ""
        return f"{name}{ann}{default}"

    # positional-only and positional
    all_pos = posonly + pos_args
    defaults_pad = [None] * (len(all_pos) - len(defaults)) + defaults
    for arg_node, def_node in zip(all_pos, defaults_pad):
        parts.append(_fmt_arg(arg_node, def_node))
    if posonly:
        parts.append("/")
    # vararg
    if args.vararg:
        var_ann = ""
        if args.vararg.annotation:
            try:
                var_ann = f": {ast.unparse(args.vararg.annotation)}"
            except Exception:
                pass
        parts.append(f"*{args.vararg.arg}{var_ann}")
    else:
        parts.append("*") if args.kwonlyargs else None
    # kwonly
    for kw, def_node in zip(args.kwonlyargs, args.kw_defaults or []):
        parts.append(_fmt_arg(kw, def_node))
    # varkw
    if args.kwarg:
        kw_ann = ""
        if args.kwarg.annotation:
            try:
                kw_ann = f": {ast.unparse(args.kwarg.annotation)}"
            except Exception:
                pass
        parts.append(f"**{args.kwarg.arg}{kw_ann}")
    return f"({', '.join([p for p in parts if p])})"


def _regex_signature(snippet: str, name: str) -> Tuple[str, List[str]]:
    """Fallback: regex parse `def name(...)` to get signature/args."""
    sig = ""
    args: List[str] = []
    m = re.search(rf"def\\s+{re.escape(name)}\\s*\\(([^)]*)\\)", snippet)
    if not m:
        return sig, args
    sig_body = m.group(1)
    sig = f"({sig_body})"
    for part in sig_body.split(","):
        part = part.strip()
        if not part:
            continue
        # strip defaults/annotations
        part = part.split("=", 1)[0].strip()
        part = part.split(":", 1)[0].strip()
        if part and part != "self":
            args.append(part)
    return sig, args


def _find_target(full_text: str, name: str, kind: str, owner: Optional[str]) -> Optional[ast.AST]:
    try:
        node = ast.parse(full_text)
    except Exception:
        return None
    owner_leaf = None
    if owner:
        owner_leaf = owner.split(".")[-1].split(":")[-1]
    # If owner is a class, search inside it first for the target.
    if owner_leaf:
        for child in ast.walk(node):
            if isinstance(child, ast.ClassDef) and child.name == owner_leaf:
                for sub in child.body:
                    if kind == "function" and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == name:
                        return sub
                    if kind == "class" and isinstance(sub, ast.ClassDef) and sub.name == name:
                        return sub
    # Fallback: global search.
    for child in ast.walk(node):
        if kind == "function" and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
            return child
        if kind == "class" and isinstance(child, ast.ClassDef) and child.name == name:
            return child
    return None


def _extract_doc_signature(full_text: str, kind: str, name: str, owner: Optional[str]) -> Tuple[str, str, List[str], str, str]:
    """Extract docstring, signature, args, returns, and code snippet from full file text."""
    doc = ""
    signature = ""
    args_list: List[str] = []
    returns = ""
    code_snippet = ""
    if kind not in {"function", "class"}:
        return doc, signature, args_list, returns, code_snippet

    target = _find_target(full_text, name, kind, owner)
    if not target:
        return doc, signature, args_list, returns, code_snippet

    doc = ast.get_docstring(target) or ""
    def _fmt_arg(arg_node, default_node=None) -> str:
        ann = ""
        if getattr(arg_node, "annotation", None):
            try:
                ann = f": {ast.unparse(arg_node.annotation)}"
            except Exception:
                ann = ""
        default = ""
        if default_node is not None:
            try:
                default = f"={ast.unparse(default_node)}"
            except Exception:
                default = ""
        return f"{getattr(arg_node, 'arg', '')}{ann}{default}".strip()

    if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        signature = _build_signature(target)
        # Capture positional-only and standard positional args (drop implicit self).
        pos = list(getattr(target.args, "posonlyargs", [])) + list(target.args.args)
        defaults = list(target.args.defaults or [])
        defaults_pad = [None] * (len(pos) - len(defaults)) + defaults
        for arg_node, def_node in zip(pos, defaults_pad):
            if arg_node.arg != "self":
                args_list.append(_fmt_arg(arg_node, def_node))
        # Keyword-only args with defaults.
        for kw, def_node in zip(target.args.kwonlyargs, target.args.kw_defaults or []):
            args_list.append(_fmt_arg(kw, def_node))
        if target.args.vararg:
            args_list.append(f"*{target.args.vararg.arg}")
        if target.args.kwarg:
            args_list.append(f"**{target.args.kwarg.arg}")
    if target.returns:
        try:
            returns = ast.unparse(target.returns)
        except Exception:
            returns = ""
    if not returns:
        # Infer from explicit return statements when no annotation is present.
        inferred: List[str] = []
        for node in ast.walk(target):
            if isinstance(node, ast.Return):
                if node.value is None:
                    inferred.append("None")
                else:
                    try:
                        inferred.append(ast.unparse(node.value))
                    except Exception:
                        inferred.append(type(node.value).__name__)
            elif isinstance(node, ast.Yield):
                inferred.append("yield")
        if inferred:
            returns = inferred[0]
    if not args_list:
        # Fallback regex-based arg parse on the full file if AST resolution failed.
        sig_fb, args_fb = _regex_signature(full_text, name)
        signature = signature or sig_fb
        if args_fb:
            args_list = args_fb
    try:
        code_snippet = ast.get_source_segment(full_text, target) or ""
    except Exception:
        code_snippet = ""
    # Fallback regex signature/args when AST didn't provide.
    if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not signature or not args_list):
        sig_fallback, args_fb = _regex_signature(code_snippet or full_text, name)
        signature = signature or sig_fallback
        if not args_list:
            args_list = args_fb
    return doc, signature, args_list, returns, code_snippet


def _extract_args_returns(snippet: str, name: str = "") -> Tuple[List[str], str]:
    args: List[str] = []
    returns = ""
    try:
        node = ast.parse(snippet)
    except Exception:
        return args, returns
    target = None
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not name or child.name == name:
                target = child
                break
        if isinstance(child, ast.ClassDef) and name and child.name == name:
            target = child
            break
    func = target if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
    if func:
        pos = list(getattr(func.args, "posonlyargs", [])) + list(func.args.args)
        defaults = list(func.args.defaults or [])
        defaults_pad = [None] * (len(pos) - len(defaults)) + defaults
        for arg_node, def_node in zip(pos, defaults_pad):
            if arg_node.arg != "self":
                try:
                    default_txt = f"={ast.unparse(def_node)}" if def_node is not None else ""
                except Exception:
                    default_txt = "=..." if def_node is not None else ""
                args.append(f"{arg_node.arg}{default_txt}")
        for kw, def_node in zip(func.args.kwonlyargs, func.args.kw_defaults or []):
            try:
                default_txt = f"={ast.unparse(def_node)}" if def_node is not None else ""
            except Exception:
                default_txt = "=..." if def_node is not None else ""
            args.append(f"{kw.arg}{default_txt}")
        if func.args.vararg:
            args.append(f"*{func.args.vararg.arg}")
        if func.args.kwarg:
            args.append(f"**{func.args.kwarg.arg}")
        if func.returns:
            try:
                returns = ast.unparse(func.returns)
            except Exception:
                returns = ""
        if not returns:
            inferred: List[str] = []
            for node in ast.walk(func):
                if isinstance(node, ast.Return):
                    if node.value is None:
                        inferred.append("None")
                    else:
                        try:
                            inferred.append(ast.unparse(node.value))
                        except Exception:
                            inferred.append(type(node.value).__name__)
                elif isinstance(node, ast.Yield):
                    inferred.append("yield")
            if inferred:
                returns = inferred[0]
    if not args:
        # Regex fallback if AST walk fails.
        sig_fb, args_fb = _regex_signature(snippet, name)
        args = args_fb or args
    if not returns:
        # Crude regex-based return capture as last resort.
        for line in snippet.splitlines():
            line_strip = line.strip()
            if line_strip.startswith("return "):
                returns = line_strip[len("return ") :].strip()
                break
    return args, returns


def _summarize_text(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    text = " ".join(text.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _llm_summary(prompt: str, model_name: Optional[str]) -> str:
    if not model_name:
        return ""
    if model_name in _LLM_CACHE:
        pipe = _LLM_CACHE[model_name]
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
            pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
            _LLM_CACHE[model_name] = pipe
        except Exception:
            return ""
    try:
        out = pipe(prompt, max_new_tokens=96, do_sample=False)
        if isinstance(out, list) and out and "generated_text" in out[0]:
            txt = out[0]["generated_text"]
            # strip prompt prefix
            return txt[len(prompt) :].strip() or txt.strip()
    except Exception:
        return ""
    return ""


def build_concepts(repo_id: str, llm_model: Optional[str]) -> List[Dict[str, any]]:
    entities = load_entities(repo_id)
    edges = load_edges(repo_id)
    artifacts = load_artifacts(repo_id)
    relations: Dict[str, List[Dict[str, str]]] = {}
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        etype = e.get("type")
        if src and dst and etype:
            relations.setdefault(src, []).append({"type": etype, "target": dst})

    concepts: List[Dict[str, any]] = []
    for eid, ent in entities.items():
        uri = ent.get("uri", "")
        path, start, end = _parse_uri_to_path(uri, artifacts)
        code = ""
        doc = ent.get("docstring") or ent.get("doc") or ""
        signature = ""
        args = []
        returns = ""
        # If function/class path missing, try owner path.
        if (not path or not path.exists()) and ent.get("kind") in {"function", "class"}:
            owner_id = ent.get("owner")
            owner_uri = entities.get(owner_id, {}).get("uri") if owner_id else None
            if owner_uri:
                owner_path, _, _ = _parse_uri_to_path(owner_uri, artifacts)
                if owner_path and owner_path.exists():
                    path = owner_path
        full_text = ""
        if path and path.exists():
            # Prefer full file parse for signatures/args/returns.
            try:
                full_text = path.read_text(encoding="utf-8")
            except Exception:
                try:
                    full_text = path.read_bytes().decode("latin-1", errors="ignore")
                except Exception:
                    full_text = ""
            if ent.get("kind") in {"function", "class"} and full_text:
                doc2, sig2, args2, ret2, code2 = _extract_doc_signature(
                    full_text, ent.get("kind", ""), ent.get("name", ""), ent.get("owner")
                )
                doc = doc or doc2
                signature = sig2 or signature
                args = args2 or args
                returns = ret2 or returns
                code = code2 or code
            if not code:
                code = _read_source_slice(path, start, end)

        heuristic_summary = _summarize_text(doc) if doc else _summarize_text(code)
        llm_sum = ""
        if llm_model and code:
            prompt = (
                "Summarize the following code in one sentence, focusing on its purpose and behavior.\n"
                f"Signature: {signature or ent.get('name','')}\n"
                f"Code:\n{code[:2000]}"
            )
            llm_sum = _llm_summary(prompt, llm_model)
        concept = {
            "repo_id": repo_id,
            "id": eid,
            "name": ent.get("name"),
            "kind": ent.get("kind"),
            "owner": ent.get("owner"),
            "uri": uri,
            "doc": doc or "",
            "signature": signature,
            "code": code,
            "summary": llm_sum or heuristic_summary,
            "args": args,
            "returns": returns,
            "relations": relations.get(eid, []),
        }
        concepts.append(concept)
    return concepts


def list_repos(limit: int | None) -> List[str]:
    manifest = EXPORT_ROOT / "_manifest.json"
    repos: List[str] = []
    if manifest.exists():
        try:
            obj = json.load(manifest.open())
            repos = list((obj.get("repos") or obj).keys())
        except Exception:
            repos = []
    if not repos:
        repos = [p.name for p in EXPORT_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if limit:
        repos = repos[:limit]
    return repos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="exports/repo_concepts.jsonl", help="Output JSONL path")
    ap.add_argument("--limit-repos", type=int, default=None, help="Max repos to process")
    ap.add_argument("--llm-model", type=str, default=None, help="Optional local LLM for semantic summaries (e.g., meta-llama/Llama-3.2-1B)")
    args = ap.parse_args()

    repos = list_repos(args.limit_repos)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rid in repos:
            concepts = build_concepts(rid, llm_model=args.llm_model)
            for c in concepts:
                f.write(json.dumps(c) + "\n")
                count += 1
    print(f"[done] wrote {count} concepts from {len(repos)} repos to {out_path}")


if __name__ == "__main__":
    main()
