"""
LLM-driven intent policy with guardrails.
The policy prompts an LLM to pick an intent from a fixed allowlist and returns
both the intent and a short rationale.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from models.piano.controller import Intent


class LLMIntentPolicy:
    """Wraps an LLM (models.shared.modeling.GenerativeModel) to choose intents."""

    def __init__(self, llm, allowed: Optional[List[Intent]] = None) -> None:
        self.llm = llm
        self.allowed = allowed or list(Intent.__args__)  # type: ignore[attr-defined]

def __call__(self, state_summary: str) -> Tuple[Intent, str]:
        prompt = (
            "Choose EXACTLY one intent from the list and answer ONLY in JSON.\n"
            f"intents: {', '.join(self.allowed)}\n"
            f"STATE:\n{state_summary}\n"
            'Return: {"intent":"<one_intent>","why":"<short reason>"}\n'
            "Answer:"
        )
        try:
            out = self.llm.generate([prompt], max_new_tokens=48, temperature=0.0)[0]
        except Exception:
            return "talk", "fallback_policy"
        intent: Intent = "talk"  # type: ignore[assignment]
        rationale = out
        for cand in self.allowed:
            if f'"intent":"{cand}"' in out or f'"intent": "{cand}"' in out or f"intent:{cand}" in out:
                intent = cand  # type: ignore[assignment]
                break
        return intent, rationale
