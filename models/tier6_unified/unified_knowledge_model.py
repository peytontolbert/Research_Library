"""
Model 27: Unified Knowledge Model (U1).
HF-backed generative model fusing papers, repos, and metadata contexts.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GenerativeModel


class UnifiedKnowledgeModel(GenerativeModel):
    """HF-backed unified reasoning model (U1) with context-concat fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def generate(self, batch):
        if self._ensure_ready():
            prompts = []
            for ex in batch:
                parts = []
                for key in ("question", "goal", "abstract", "code", "repo_context", "graph_context"):
                    if ex.get(key):
                        parts.append(f"{key.upper()}:\n{ex[key]}")
                prompts.append("\n".join(parts) + "\nProvide a unified reasoning answer.")
            return super().generate(prompts, max_new_tokens=400, temperature=0.2)

        outputs = []
        for ex in batch:
            parts = []
            for key in ("question", "goal", "abstract", "code", "repo_context", "graph_context"):
                if ex.get(key):
                    parts.append(f"{key.upper()}: {ex[key]}")
            outputs.append("\n".join(parts) if parts else "UNIFIED_REASONING: [no context]")
        return outputs
