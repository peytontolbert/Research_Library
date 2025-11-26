from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as fh:
		return json.loads(fh.read())


def run_program_training(adapters_dir: str, plan_json: str, out_json: str) -> str:
	"""
	Smoke-level trainer hook used by examples: simulates a training pass
	grounded to a ProgramGraph-based TrainingPlan by producing a summary
	and a trained marker under adapters_dir.
	"""
	adapters_dir_abs = os.path.abspath(adapters_dir)
	plan = _read_json(plan_json)
	plan_obj = plan.get("DatasetTrainingPlan") or {}
	datasets = (plan_obj.get("plan") or {}).get("datasets", [])
	schedule = (plan_obj.get("plan") or {}).get("schedule", [])

	# Simulate "training" by writing a marker file
	Path(adapters_dir_abs).mkdir(parents=True, exist_ok=True)
	marker = Path(adapters_dir_abs) / "TRAINED.OK"
	marker.write_text("trained=1\n", encoding="utf-8")

	summary = {
		"trained": True,
		"datasets": datasets,
		"schedule": schedule,
		"artifacts": {"marker": str(marker)},
	}
	obj = {"schema_version": 1, "TrainingSummary": summary}
	os.makedirs(os.path.dirname(out_json), exist_ok=True)
	with open(out_json, "w", encoding="utf-8") as fh:
		fh.write(json.dumps(obj, indent=2))
	return out_json


def main() -> None:
	if len(sys.argv) < 4:
		print("usage: python program_trainer.py <adapters_dir> <plan_json> <out_json>", file=sys.stderr)
		sys.exit(2)
	adapters_dir = sys.argv[1]
	plan_json = sys.argv[2]
	out_json = sys.argv[3]
	path = run_program_training(adapters_dir, plan_json, out_json)
	print(path)


if __name__ == "__main__":
	main()



