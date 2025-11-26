from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional


def run_benchmark_verifier(cmd: Optional[List[str]], env: Optional[Dict[str, str]] = None, timeout_sec: int = 120) -> bool:
	"""
	Runs an external verifier command (official checker) and returns True on success.
	When cmd is None, returns True (no-op) for smoke runs.
	"""
	if not cmd:
		return True
	try:
		rc = subprocess.call(cmd, env=env, timeout=timeout_sec)  # type: ignore
	except Exception:
		return False
	return rc == 0



