"""
Model 13: Equation Reasoning (P3).
Hybrid: HF generative math reasoning when available; deterministic type+steps fallback.
"""

import re
from typing import Any, Dict, Optional, Sequence, List

from models.shared.modeling import GenerativeModel


class EquationReasoningModel(GenerativeModel):
    """HF-backed equation reasoning with rule-based fallback (P3)."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    @staticmethod
    def _extract_equation(example: Dict[str, Any]) -> str:
        return str(example.get("equation") or example.get("expr") or example.get("text") or "")

    @staticmethod
    def _classify_equation(eqn: str) -> str:
        s = eqn.replace(" ", "")
        if not s:
            return "empty"

        if "\\sum" in s or "sum_" in s:
            return "summation"
        if "\\int" in s or "integral" in s:
            return "integral"
        if "lim_" in s or "lim" in s:
            return "limit"
        if "d/d" in s or "\\frac{d" in s:
            return "derivative"

        degree3 = re.search(r"[a-zA-Z]\\^3", s) or ("**3" in s)
        degree2 = re.search(r"[a-zA-Z]\\^2", s) or ("**2" in s)
        if degree3:
            return "polynomial_cubic"
        if degree2:
            return "polynomial_quadratic"

        if "=" in s:
            return "equation_linear_or_general"

        return "expression"

    @staticmethod
    def _suggest_steps(kind: str) -> str:
        if kind == "summation":
            return "Identify index/bounds; try closed form."
        if kind == "integral":
            return "Check substitution or parts; compare to standard forms."
        if kind == "limit":
            return "Simplify; if indeterminate, consider L'Hôpital."
        if kind == "derivative":
            return "Apply product/quotient/chain rules and simplify."
        if kind == "polynomial_quadratic":
            return "Rewrite as ax^2+bx+c=0; use quadratic formula or factorization."
        if kind == "polynomial_cubic":
            return "Search simple roots to factor, reduce to quadratic."
        if kind == "equation_linear_or_general":
            return "Isolate unknown variable with inverse operations, simplify."
        if kind == "expression":
            return "Collect like terms, factor common elements, reduce fractions."
        return "Clarify equation and simplify algebraically."

    def generate(self, batch: Sequence[Dict[str, Any]]) -> List[str]:
        if self._ensure_ready():
            prompts = []
            for ex in batch:
                eqn = self._extract_equation(ex)
                ctx = str(ex.get("context") or "")
                prompts.append(f"EQUATION:\n{eqn}\nCONTEXT:\n{ctx}\nExplain, simplify, or derive next steps.")
            return super().generate(prompts, max_new_tokens=256, temperature=0.1)

        outputs: List[str] = []
        for ex in batch:
            eqn = self._extract_equation(ex).strip()
            if not eqn:
                outputs.append("No equation provided.")
                continue
            kind = self._classify_equation(eqn)
            steps = self._suggest_steps(kind)
            explanation = f"Equation: {eqn}\nType: {kind.replace('_', ' ')}.\nNext steps: {steps}"
            outputs.append(explanation)
        return outputs
