"""
Cognitive Controller: selects a high-level intent from AgentState.
Acts as the coherence bottleneck across concurrent modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

Intent = Literal[
    "talk",
    "inspect_graph",
    "edit_code",
    "run_tests",
    "read_paper",
    "plan_more",
    "spawn_specialist",
    "run_benchmarks",
    "update_persona",
    "add_concept_node",
    "apply_lora",
    "fine_tune",
    "run_inference",
]


@dataclass
class CognitiveController:
    """Heuristic or policy-driven intent selector."""

    policy: Optional[Callable[[str], Tuple[Intent, str]]] = None
    last_rationale: Optional[str] = None

    def decide(self, state_summary: str) -> Intent:
        if self.policy:
            try:
                raw_intent, rationale = self.policy(state_summary)
                if raw_intent in Intent.__args__:  # type: ignore[attr-defined]
                    self.last_rationale = rationale
                    return raw_intent  # type: ignore[return-value]
            except Exception:
                pass
        summary = state_summary.lower()
        self.last_rationale = None
        if "benchmark" in summary:
            return "run_benchmarks"
        if "new concept" in summary:
            return "add_concept_node"
        if "persona" in summary:
            return "update_persona"
        if "adapter" in summary or "lora" in summary:
            return "apply_lora"
        if "fine-tune" in summary or "finetune" in summary:
            return "fine_tune"
        if "inference" in summary:
            return "run_inference"
        if "spawn" in summary or "delegate" in summary:
            return "spawn_specialist"
        if "test fail" in summary or "assert" in summary:
            return "edit_code"
        if "plan" in summary and len(summary) < 320:
            return "plan_more"
        if "paper" in summary:
            return "read_paper"
        if "concept" in summary or "graph" in summary:
            return "inspect_graph"
        return "talk"
