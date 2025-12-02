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

    def __init__(
        self,
        repo_concept_path: Path = Path("models/exports/repo_concepts.jsonl"),
        paper_concept_path: Path = Path("models/exports/paper_concepts.jsonl"),
    ) -> None:
        # Repo-level concepts (functions, modules, tests, etc.).
        self.repo_concept_path = repo_concept_path
        # Paper-level concepts (sections, method constructs, empirical techniques).
        self.paper_concept_path = paper_concept_path
        # Cache personas by (persona_type, entity_id) to keep repo/paper schemas separate.
        self._cache: Dict[str, PersonaSchema] = {}

    def _load_repo_concepts(self, repo_id: str) -> List[Dict[str, str]]:
        if not self.repo_concept_path.exists():
            return []
        concepts: List[Dict[str, str]] = []
        with self.repo_concept_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("repo_id") == repo_id:
                    concepts.append(obj)
        return concepts

    def _load_paper_concepts(self, paper_id: str) -> List[Dict[str, str]]:
        """Load concepts for a paper from paper_concepts.jsonl if available."""
        if not self.paper_concept_path.exists():
            return []
        concepts: List[Dict[str, str]] = []
        with self.paper_concept_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pid = obj.get("paper_id") or obj.get("id")
                if pid == paper_id:
                    concepts.append(obj)
        return concepts

    def _infer_style(self, concepts: List[Dict[str, str]]) -> Dict[str, str]:
        """Lightweight heuristics for repository style: look at kinds and signatures."""
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

    def _infer_paper_style(self, concepts: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Heuristics for paper style:
        - theoretical_bias: theoretical / empirical / mixed / unspecified
        - engineering_depth: low / medium / high
        - experimentation_pattern: ablation-heavy / single-main-result / unspecified
        """
        if not concepts:
            return {}

        blob_tokens: List[str] = []
        for c in concepts:
            txt = (c.get("summary") or c.get("doc") or c.get("code") or "").lower()
            blob_tokens.extend(txt.split())

        tok_set = set(blob_tokens)
        has_theory = any(t in tok_set for t in ("theorem", "lemma", "proof", "bound", "convergence"))
        has_experiments = any(t in tok_set for t in ("experiment", "dataset", "benchmark", "accuracy", "evaluation"))
        if has_theory and not has_experiments:
            theoretical_bias = "theoretical"
        elif has_experiments and not has_theory:
            theoretical_bias = "empirical"
        elif has_theory and has_experiments:
            theoretical_bias = "mixed"
        else:
            theoretical_bias = "unspecified"

        has_system = any(t in tok_set for t in ("implementation", "system", "deployment", "runtime", "throughput", "latency"))
        has_code = any(t in tok_set for t in ("pytorch", "tensorflow", "library", "framework"))
        if has_system and has_code:
            engineering_depth = "high"
        elif has_system or has_code:
            engineering_depth = "medium"
        else:
            engineering_depth = "low"

        has_ablation = any("ablation" in t for t in tok_set)
        has_sweep = any(t in tok_set for t in ("sweep", "grid-search", "hyperparameter"))
        if has_ablation or has_sweep:
            experimentation_pattern = "ablation-heavy"
        elif has_experiments:
            experimentation_pattern = "single-main-result"
        else:
            experimentation_pattern = "unspecified"

        return {
            "theoretical_bias": theoretical_bias,
            "engineering_depth": engineering_depth,
            "experimentation_pattern": experimentation_pattern,
        }

    def build(self, entity_id: str, persona_type: str = "repo") -> PersonaSchema:
        cache_key = f"{persona_type}:{entity_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if persona_type == "paper":
            concepts = self._load_paper_concepts(entity_id)
        else:
            concepts = self._load_repo_concepts(entity_id)

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
        style = self._infer_paper_style(concepts) if persona_type == "paper" else self._infer_style(concepts)
        schema = PersonaSchema(entity_id=entity_id, concepts=concept_ids[:24], edges=edges[:48], style=style)
        self._cache[cache_key] = schema
        return schema
