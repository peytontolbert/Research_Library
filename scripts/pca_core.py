from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable


# Canonical PCA Interface (portable utility helpers)

def select_subgraph(goal: Dict[str, Any], program_facts: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Select(): question-aware selection of subgraph/segments/windows from the program.
	Heuristic: prefer recently changed files, failing spans, or symbols mentioned in goal text.
	"""
	text = (goal.get("text") or goal.get("prompt") or "").lower()
	return {"windows": [], "symbols": [], "goal_terms": text.split()[:16]}


def pack_with_anchors(sources: Iterable[Path], windows: List[Tuple[str, int, int]]) -> Dict[str, Any]:
	"""
	Pack(): deterministic context packaging with anchored snippets (path:line windows).
	"""
	packed = {"anchors": [], "tokens_budget": 0}
	for rel, a, b in windows:
		try:
			packed["anchors"].append({"path": rel, "start_line": int(a), "end_line": int(b)})
		except Exception:
			continue
	return packed


def embed_multifactor(features: Dict[str, Any]) -> List[float]:
	"""
	Embed(): multi-factor embedding of program-specific features (schemas, contracts, graphs, traces).
	Placeholder returns a small fixed-length vector for compatibility.
	"""
	return [0.0, 0.0, 0.0, 1.0]


def adapt_lora_deltas(targets: List[str], rank: int = 8) -> Dict[str, Any]:
	"""
	Adapt(): synthesize/mix LoRA-like deltas for LM layer targets (attention/MLP) with a stable gating schedule.
	"""
	return {"targets": list(targets), "rank": int(rank), "gating": "stable"}


def verify_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Verify(): run program-native checks (lints/compile/tests/SQL dry-run/schema).
	Here we only check structure is non-empty for demo purposes.
	"""
	ok = isinstance(outputs, dict) and len(outputs) > 0
	return {"ok": bool(ok), "summary": "ok" if ok else "empty"}


def cite_outputs(outputs: Dict[str, Any], anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Cite(): append anchors to every claim; here we attach under a standard field if missing.
	"""
	with_cites = dict(outputs)
	with_cites.setdefault("citations", anchors)
	return with_cites


def log_minimal(goal: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Log(): emit minimal, privacy-respecting telemetry for reproducibility and distillation.
	"""
	return {"goal_hash": hash(str(goal)) % (10**9), "outputs_keys": list(outputs.keys())}


@dataclass(frozen=True)
class Budget:
	tokens: int = 64000
	wall_sec: int = 120
	ci_min: int = 10


