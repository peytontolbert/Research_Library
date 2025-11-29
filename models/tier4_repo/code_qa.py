"""
Model 17: Code Q&A Model (R3).
HF-backed generative QA over code context + question.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GenerativeModel


class CodeQAModel(GenerativeModel):
    """HF-backed code Q&A model (R3) with summarization fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def generate(self, batch):
        if self._ensure_ready():
            prompts = []
            for ex in batch:
                q = ex.get("question") or ex.get("query") or ""
                code = ex.get("code") or ex.get("context") or ""
                prompts.append(f"QUESTION:\n{q}\nCODE:\n{code}\nAnswer grounded in code.")
            return super().generate(prompts, max_new_tokens=256, temperature=0.0)

        outputs = []
        for ex in batch:
            code = str(ex.get("code") or ex.get("context") or "")
            question = str(ex.get("question") or ex.get("query") or "")
            # Crude summary: first docstring or function signature
            summary = ""
            import re
            doc = re.search(r'"""(.*?)"""', code, re.DOTALL)
            if doc:
                summary = doc.group(1).strip()
            else:
                sigs = re.findall(r"^(def\\s+[a-zA-Z_][a-zA-Z0-9_]*\\(.*\\):)", code, re.MULTILINE)
                if sigs:
                    summary = " ".join(sigs[:3])
            answer = summary or code[:400]
            outputs.append(f"Q: {question}\nA: {answer}")
        return outputs
