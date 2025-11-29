"""
Model 3: Citation Prediction (graph/link prediction).
HF-backed graph/contrastive wrapper to score citation edges.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GraphModel


class CitationPredictionModel(GraphModel):
    """HF-backed link predictor for citations (M3) with lexical fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def predict(self, batch):
        """Return ranked citation candidates per example."""
        # If HF ready, use score_edges on provided pairs.
        if self._ensure_ready():
            results = []
            for ex in batch:
                edges = []
                src = str(ex.get("source") or "")
                for cand in ex.get("candidates") or []:
                    edges.append({"src": src, "dst": str(cand)})
                sims = self.score_edges(edges)
                ranked = sorted(zip(ex.get("candidates") or [], sims), key=lambda x: x[1], reverse=True)
                results.append(ranked)
            return results

        # Fallback: string overlap similarity.
        results = []
        for ex in batch:
            src = str(ex.get("source") or "")
            candidates = ex.get("candidates") or []
            scored = []
            for cand in candidates:
                c = str(cand)
                inter = len(set(src) & set(c))
                union = len(set(src) | set(c) or {""})
                scored.append((cand, inter / union if union else 0.0))
            scored.sort(key=lambda x: x[1], reverse=True)
            results.append(scored)
        return results
