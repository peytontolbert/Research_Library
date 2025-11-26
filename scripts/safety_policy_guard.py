from __future__ import annotations

from typing import Any, Dict, List


def enforce(policy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Apply simple safety gates like 'apply_patch requires tests_green' etc.
	Returns {'ok': bool, 'missing': [..]}.
	"""
	reqs: List[str] = list(policy.get("requires", []))
	passed = []
	missing = []
	for r in reqs:
		if context.get(r, False):
			passed.append(r)
		else:
			missing.append(r)
	return {"ok": len(missing) == 0, "missing": missing}


