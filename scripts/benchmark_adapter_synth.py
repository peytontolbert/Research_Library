from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any


def synth_benchmark_heads(adapters_dir: str, datasets: List[str]) -> List[str]:
	"""
	Smoke-level synthesis of benchmark-aware adapter heads.
	Creates tiny shard stubs under adapters/shards/* so run.py can pick them up.
	"""
	adir = Path(adapters_dir)
	shards_dir = adir / "shards"
	shards_dir.mkdir(parents=True, exist_ok=True)
	created: List[str] = []
	for spec in datasets:
		ds_name = spec.replace(":", "_").replace("/", "_")
		path = shards_dir / f"benchmark_{ds_name}_head.json"
		obj: Dict[str, Any] = {
			"schema_version": 1,
			"type": "benchmark_head",
			"dataset": spec,
			"rank": 8,
			"prior": {"recent_pass_boost": 0.2},
		}
		path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
		created.append(str(path))
	return created


def main() -> None:
	if len(sys.argv) < 3:
		print("usage: python benchmark_adapter_synth.py <adapters_dir> <datasets_csv>", file=sys.stderr)
		sys.exit(2)
	adapters_dir = sys.argv[1]
	datasets_csv = sys.argv[2]
	datasets = [s.strip() for s in datasets_csv.split(",") if s.strip()]
	paths = synth_benchmark_heads(adapters_dir, datasets)
	print("\n".join(paths))


if __name__ == "__main__":
	main()



