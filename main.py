# main.py

import subprocess
from utils.logging_utils import setup_logger
from scripts.run_training import run_training

def main():
    """Run the full RLHF pipeline: SFT, Reward, PPO training and evaluation."""
    logger = setup_logger("pipeline")
    logger.info("=== Starting full pipeline ===")

    # Training phases
    run_training(logger)
    logger.info("=== Training complete ===")

    # Evaluation phases
    eval_configs = [
        "configs/evaluate_indist.yaml",
        "configs/evaluate_ood.yaml",
    ]
    for cfg in eval_configs:
        logger.info(f"Running evaluation with config: {cfg}")
        subprocess.run(
            ["python", "eval_runner.py", "--phase", "evaluate", "--config", cfg],
            check=True
        )
    logger.info("=== Full pipeline complete ===")

if __name__ == "__main__":
    main()
