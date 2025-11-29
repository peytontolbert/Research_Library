"""
Model 18: Bug Localization Model (R4).
HF-backed classifier to score files/lines given failing tests/descriptions.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import ClassifierModel


class BugLocalizationModel(ClassifierModel):
    """HF-backed bug localization classifier (R4) with heuristic fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def predict(self, batch):
        if self._ensure_ready():
            texts = []
            for ex in batch:
                failure = ex.get("failure") or ex.get("test_failure") or ""
                code = ex.get("code") or ex.get("context") or ""
                texts.append(f"FAILURE:\n{failure}\nCODE:\n{code}\nPredict file/line.")
            return super().predict(texts)

        outputs = []
        for ex in batch:
            candidates = ex.get("candidates") or []
            failure = str(ex.get("failure") or ex.get("test_failure") or "")
            # Pick the first candidate that shares a keyword with the failure.
            chosen = candidates[0] if candidates else "unknown"
            for cand in candidates:
                if any(tok in str(cand).lower() for tok in failure.lower().split()):
                    chosen = cand
                    break
            outputs.append(str(chosen))
        return outputs
