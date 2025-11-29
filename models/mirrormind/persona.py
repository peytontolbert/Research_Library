"""
Persona schema construction for repos and papers.
Uses exported concepts (if present) and lightweight heuristics for style attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json
import random


@dataclass
class PersonaSchema:
    """Compact persona graph plus style attributes."""

    entity_id: str
    concepts: Sequence[str]
    edges: Sequence[Dict[str, str]] = field(default_factory=list)  # {"src": c1, "dst": c2, "type": "co_implemented_with"}
    style: Dict[str, str] = field(default_factory=dict)  # architecture_style, testing_style, etc.

    def to_prompt(self) -> str:
        lines = [f"ENTITY: {self.entity_id}", "CONCEPTS: " + ", ".join(self.concepts)]
        if self.edges:
            rendered = [f"{e.get('src')} -[{e.get('type','rel')}]-> {e.get('dst')}" for e in self.edges]
            lines.append("RELATIONS:\n" + "\n".join(rendered))
        if self.style:
            style_parts = [f"{k}={v}" for k, v in self.style.items()]
            lines.append("STYLE: " + ", ".join(style_parts))
        return "\n".join(lines)


class PersonaBuilder:
    """Helper to load persona schemas from exported concept graphs."""

    def __init__(self, concept_path: Path = Path("models/exports/repo_concepts.jsonl")) -> None:
        self.concept_path = concept_path
        self._cache: Dict[str, PersonaSchema] = {}

    def _load_concepts(self, entity_id: str) -> List[Dict[str, str]]:
        if not self.concept_path.exists():
            return []
        concepts: List[Dict[str, str]] = []
        with self.concept_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("repo_id") == entity_id:
                    concepts.append(obj)
        return concepts

    def _infer_style(self, concepts: List[Dict[str, str]]) -> Dict[str, str]:
        """Lightweight heuristics: look at kinds and signatures to guess style."""
        if not concepts:
            return {}
        kinds = [c.get("kind", "") for c in concepts]
        has_tests = any("test" in (c.get("name") or "").lower() for c in concepts)
        has_cuda = any("cuda" in (c.get("code") or "").lower() for c in concepts)
        has_types = any("->" in (c.get("signature") or "") or ":" in (c.get("signature") or "") for c in concepts)
        doc_tokens = []
        for c in concepts:
            doc = c.get("doc") or c.get("summary") or ""
            doc_tokens.extend(doc.lower().split())
        style = {
            "architecture_style": "modular" if "module" in kinds else "monolith",
            "testing_style": "unit-heavy" if has_tests else "sparse-tests",
            "performance_bias": "cuda-kernels" if has_cuda else "standard-python",
            "type_hints_density": "typed" if has_types else "untyped",
        }
        if any("async" in t for t in doc_tokens):
            style["concurrency_bias"] = "async"
        if any("cuda" in t or "gpu" in t for t in doc_tokens):
            style["hardware_bias"] = "gpu"
        return style

    def build(self, entity_id: str, persona_type: str = "repo") -> PersonaSchema:
        if entity_id in self._cache:
            return self._cache[entity_id]
        concepts = self._load_concepts(entity_id)
        concept_ids = [c.get("id") or c.get("name") for c in concepts if c.get("id") or c.get("name")]
        # Build simple co-occurrence edges between consecutive concepts as a placeholder.
        edges: List[Dict[str, str]] = []
        for i in range(len(concept_ids) - 1):
            edges.append({"src": concept_ids[i], "dst": concept_ids[i + 1], "type": "co_implemented_with"})
        if not edges and len(concept_ids) >= 2:
            edges.append({"src": concept_ids[0], "dst": concept_ids[-1], "type": "co_implemented_with"})
        if persona_type == "paper":
            random.seed(42)
            random.shuffle(concept_ids)
        style = self._infer_style(concepts)
        schema = PersonaSchema(entity_id=entity_id, concepts=concept_ids[:24], edges=edges[:48], style=style)
        self._cache[entity_id] = schema
        return schema
