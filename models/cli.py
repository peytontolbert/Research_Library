"""
CLI scaffold for training/evaluating the 32-model substrate.
Usage:
  python -m models.cli --experiment models/experiments/m1_metadata_embedding.json --dry-run
"""

import argparse
import pprint
from pathlib import Path

from models.shared.config import load_experiment, validate_cache_dirs
from models.shared.training import (
    Trainer,
    build_tokenizer,
    build_backbone,
    apply_peft_if_needed,
)
from models.shared.registry import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to an experiment JSON under models/experiments/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Run training or evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.experiment)
    config = validate_cache_dirs(load_experiment(str(config_path)))
    model_id = config.get("model_id")
    tokenizer = None
    backbone = None
    use_hf = config.get("training", {}).get("use_hf_trainer", False)
    try:
        tokenizer = build_tokenizer(config) if use_hf else None
        backbone = build_backbone(config) if use_hf else None
        if backbone is not None:
            backbone = apply_peft_if_needed(backbone, config)
    except Exception as exc:
        print(f"[warn] Skipping HF/PEFT construction (offline or missing deps): {exc}")
    # Build the model stub after HF objects are ready so the wrapper can bind them.
    model_stub = build_model(model_id, tokenizer=tokenizer, backbone=backbone, config=config)

    if args.dry_run:
        print(f"[dry-run] Loaded {model_id} from {config_path}")
        pprint.pprint(config)
        return

    trainer = Trainer(config=config, model_stub=model_stub, tokenizer=tokenizer, backbone=backbone)
    if args.mode == "train":
        trainer.train()
    else:
        trainer.evaluate()


if __name__ == "__main__":
    main()
