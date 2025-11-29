"""
Run a multi-step PianoAgent via Coordinator with file/Neo4j graph backing.
Usage: python -m models.mirrormind.scripts.demo_coordinator --task "fix failing optimizer tests" --steps 3 --repo AgentLab
"""

import argparse
import json
import os

from models.mirrormind.coordinator import Coordinator, TaskDescriptor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--paper", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    # Allow overriding HF commands via env for demo
    os.environ.setdefault("PIANO_FINE_TUNE_CMD", '["echo","fine_tune_stub"]')
    os.environ.setdefault("PIANO_INFERENCE_CMD", '["echo","inference_stub"]')
    os.environ.setdefault("PIANO_APPLY_LORA_CMD", '["echo","apply_lora_stub"]')
    coord = Coordinator()
    result = coord.run_multi_step(task_text=args.task, steps=args.steps, repo_id=args.repo, paper_id=args.paper)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
