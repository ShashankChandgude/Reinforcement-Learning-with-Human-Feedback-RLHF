# eval_runner.py

import argparse
import json
from datasets import load_dataset
from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from evaluation.sft_evaluator import SFTEvaluator
from evaluation.ppo_evaluator import PPOEvaluator
from evaluation.ood_evaluator import OODEvaluator

PHASE = "evaluate"
EVALUATOR_MAP = {
    "sft": SFTEvaluator,
    "ppo": PPOEvaluator,
    "ood": OODEvaluator,
}

def main():
    parser = argparse.ArgumentParser(description="RLHF for All - Evaluation Runner")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=[PHASE],
        help="Phase to run: evaluate"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file for evaluation"
    )
    args = parser.parse_args()

    if args.phase != PHASE:
        raise ValueError(f"Unsupported phase: {args.phase}. Only '{PHASE}' is supported.")

    config = load_config(args.config)
    log_dir = config.get("output", {}).get("log_dir", "logs")
    logger = setup_logger(name="evaluate", log_dir=log_dir)
    logger.info(f"Starting evaluation with config '{args.config}'")

    # Choose evaluator based on dataset.loader
    ds_cfg = config.get("dataset", {})
    loader = ds_cfg.get("loader")
    evaluator_cls = EVALUATOR_MAP.get(loader)
    if evaluator_cls is None:
        raise ValueError(f"Unknown evaluator for loader '{loader}'")

    # Instantiate and load model
    evaluator = evaluator_cls(config.get("generation", {}))
    evaluator.load_model(config["model_dir"])

    # Load raw prompts & references from HF dataset
    raw = load_dataset(ds_cfg["name"])
    split = raw.get("test") or raw.get("validation") or raw.get("train")
    subset = split.select(range(min(ds_cfg.get("subset_size", len(split)), len(split))))
    # Assume standard column names
    prompts    = subset["instruction"] if "instruction" in subset.column_names else subset["prompt"]
    references = subset["response"]    if "response"    in subset.column_names else subset["completion"]

    # Generate and evaluate
    predictions = evaluator.generate(prompts)
    metrics     = evaluator.compute_metrics(predictions, references)

    # Write out report
    report_file = config["output"]["report_file"]
    with open(report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation complete. Report saved to '{report_file}'")

if __name__ == "__main__":
    main()
