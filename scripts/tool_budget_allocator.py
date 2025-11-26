from __future__ import annotations

from typing import Any, Dict


def allocate(budget: Dict[str, Any], expected_gain: float) -> Dict[str, Any]:
	"""
	Greedily allocate tokens/time by expected verifier-gain per unit cost (placeholder).
	"""
	out = dict(budget)
	out["allocated"] = {"tokens": int(budget.get("tokens", 64000) * 0.5)}
	out["expected_gain"] = float(expected_gain)
	return out


