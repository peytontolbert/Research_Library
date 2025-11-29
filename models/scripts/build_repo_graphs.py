#!/usr/bin/env python3
from __future__ import annotations

"""
Build lightweight repo graphs for /data/repositories and write JSONL exports under
/data/repository_library/exports.

Nodes:
  - file: id="{repo}:{rel_path}", kind="file"
  - function: id="{repo}:{rel_path}::{func}", kind="function", owner=file id
  - class: id="{repo}:{rel_path}::{class}", kind="class", owner=file id
  - symbol/import: id="sym:{name}", kind="symbol"

Edges:
  - file -> function/class: type="owns"
  - function/class -> symbol (call/import): type="calls"/"imports"

Manifest:
  updates exports/_manifest.json with repo_root and export_schema_version.

This is intentionally lightweight and dependency-free; it focuses on Python files.
"""

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

EXPORT_ROOT = Path("/data/repository_library/exports")
REPO_ROOT = Path("/data/repositories")
MANIFEST_PATH = EXPORT_ROOT / "_manifest.json"


def iter_repos() -> List[Path]:
    if not REPO_ROOT.exists():
        return []
    return [p for p in REPO_ROOT.iterdir() if p.is_dir()]


def parse_python(path: Path) -> Tuple[List[Dict], List[Dict]]:
    entities: List[Dict] = []
    edges: List[Dict] = []
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        try:
            src = path.read_bytes().decode("latin-1", errors="ignore")
        except Exception:
            return entities, edges
    try:
        tree = ast.parse(src)
    except Exception:
        return entities, edges

    repo_id = path.parts[-len(path.parts) + 1] if len(path.parts) > 1 else "repo"
    rel = path.relative_to(REPO_ROOT / repo_id)
    file_id = f"{repo_id}:{rel}"
    entities.append({"repo_id": repo_id, "id": file_id, "uri": f"program://{repo_id}/file/{rel}", "kind": "file", "name": str(rel), "owner": None})

    def add_edge(src_id: str, dst_id: str, etype: str):
        edges.append({"repo_id": repo_id, "src": src_id, "dst": dst_id, "type": etype})

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope = []

        def visit_FunctionDef(self, node: ast.FunctionDef):
            func_id = f"{file_id}::{node.name}"
            entities.append({"repo_id": repo_id, "id": func_id, "uri": f"program://{repo_id}/function/{rel}#{node.name}", "kind": "function", "name": node.name, "owner": file_id})
            add_edge(file_id, func_id, "owns")
            self.scope.append(func_id)
            self.generic_visit(node)
            self.scope.pop()

        def visit_ClassDef(self, node: ast.ClassDef):
            cls_id = f"{file_id}::{node.name}"
            entities.append({"repo_id": repo_id, "id": cls_id, "uri": f"program://{repo_id}/class/{rel}#{node.name}", "kind": "class", "name": node.name, "owner": file_id})
            add_edge(file_id, cls_id, "owns")
            self.scope.append(cls_id)
            self.generic_visit(node)
            self.scope.pop()

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                sym_id = f"sym:{alias.name}"
                entities.append({"repo_id": repo_id, "id": sym_id, "uri": f"symbol://{alias.name}", "kind": "symbol", "name": alias.name, "owner": None})
                add_edge(self.scope[-1] if self.scope else file_id, sym_id, "imports")

        def visit_ImportFrom(self, node: ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                name = f"{mod}.{alias.name}" if mod else alias.name
                sym_id = f"sym:{name}"
                entities.append({"repo_id": repo_id, "id": sym_id, "uri": f"symbol://{name}", "kind": "symbol", "name": name, "owner": None})
                add_edge(self.scope[-1] if self.scope else file_id, sym_id, "imports")

        def visit_Call(self, node: ast.Call):
            callee = ""
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee:
                sym_id = f"sym:{callee}"
                entities.append({"repo_id": repo_id, "id": sym_id, "uri": f"symbol://{callee}", "kind": "symbol", "name": callee, "owner": None})
                if self.scope:
                    add_edge(self.scope[-1], sym_id, "calls")
                else:
                    add_edge(file_id, sym_id, "calls")
            self.generic_visit(node)

    Visitor().visit(tree)
    return entities, edges


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def update_manifest(manifest: Dict, repo_id: str, repo_root: Path):
    manifest.setdefault("repos", {})
    manifest["repos"][repo_id] = {
        "repo_root": str(repo_root),
        "export_schema_version": 1,
        "last_indexed_at": int(time.time()),
    }


def main(max_files: int):
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.load(MANIFEST_PATH.open())
        except Exception:
            manifest = {}

    for repo_dir in iter_repos():
        repo_id = repo_dir.name
        entities: List[Dict] = []
        edges: List[Dict] = []
        count = 0
        for path in repo_dir.rglob("*.py"):
            if max_files and count >= max_files:
                break
            ent, edg = parse_python(path)
            entities.extend(ent)
            edges.extend(edg)
            count += 1
        if not entities and not edges:
            continue
        out_dir = EXPORT_ROOT / repo_id
        write_jsonl(out_dir / f"{repo_id}.entities.jsonl", entities)
        write_jsonl(out_dir / f"{repo_id}.edges.jsonl", edges)
        update_manifest(manifest, repo_id, repo_dir)
        print(f"[export] {repo_id}: entities={len(entities)} edges={len(edges)}")

    write_jsonl(MANIFEST_PATH, []) if False else None  # placeholder to keep format
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("[done] manifest updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build lightweight repo graphs.")
    parser.add_argument("--max-files", type=int, default=0, help="Max Python files per repo (0 = all).")
    args = parser.parse_args()
    main(max_files=max(0, args.max_files or 0))
