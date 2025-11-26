from __future__ import annotations

from typing import Any, Dict, List


def require_citations(outputs: Dict[str, Any], min_count: int = 1) -> Dict[str, Any]:
	"""
	Ensure outputs contain at least min_count citation anchors.
	"""
	cites = outputs.get("citations") or []
	ok = isinstance(cites, list) and len(cites) >= int(min_count)
	return {"ok": ok, "count": len(cites) if isinstance(cites, list) else 0}


