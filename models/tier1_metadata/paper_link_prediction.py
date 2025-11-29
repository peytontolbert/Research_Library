"""
Model 4: Paper Similarity / Link Prediction (Graph).
HF-backed graph scorer over co-author/citation nodes, aligned to plan.
"""

from typing import Any, Dict, Optional

from models.shared.modeling import GraphModel


class PaperLinkPredictionModel(GraphModel):
    """HF-backed graph link predictor for papers (M4) with title-overlap fallback."""

    def __init__(self, tokenizer: Any = None, backbone: Any = None, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(tokenizer=tokenizer, backbone=backbone, config=config)

    def predict(self, batch):
        """Rank candidate paper edges."""
        if self._ensure_ready():
            results = []
            for ex in batch:
                source = ex.get("source") or {}
                candidates = ex.get("candidates") or []
                edges = [{"src": str(source.get("title", source)), "dst": str(c.get("title", c))} for c in candidates]
                sims = self.score_edges(edges)
                ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
                results.append(ranked)
            return results

        def _title_tokens(paper: Dict[str, Any]) -> set:
            title = (paper.get("title") or "").lower()
            return set(title.replace(",", " ").split())

        results = []
        for ex in batch:
            source = ex.get("source") or {}
            candidates = ex.get("candidates") or []
            src_tokens = _title_tokens(source)
            scored = []
            for cand in candidates:
                cand_tokens = _title_tokens(cand if isinstance(cand, dict) else {"title": str(cand)})
                inter = len(src_tokens & cand_tokens)
                union = len(src_tokens | cand_tokens) or 1
                scored.append((cand, inter / union))
            scored.sort(key=lambda x: x[1], reverse=True)
            results.append(scored)
        return results
