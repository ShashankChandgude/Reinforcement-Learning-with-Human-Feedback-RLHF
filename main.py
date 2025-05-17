import argparse
from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from training.sft_trainer import SFTTrainer
from training.reward_trainer import RewardTrainer
from training.ppo_trainer import PPOTrainer

# Map phase names to their corresponding trainer classes
PHASE_MAP = {
    "sft": SFTTrainer,
    "reward": RewardTrainer,
    "ppo": PPOTrainer,
}

def main():
    parser = argparse.ArgumentParser(description="RLHF for All - Phase Runner")
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=PHASE_MAP.keys(),
        help="Phase to run: sft (Supervised FT), reward (Reward Model), or ppo (PPO Training)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file for the selected phase"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files"
    )
    args = parser.parse_args()

    logger = setup_logger(name="main", log_dir=args.log_dir)
    logger.info(f"Starting phase '{args.phase}' with config '{args.config}'")

    config = load_config(args.config)

    trainer_cls = PHASE_MAP.get(args.phase)
    if trainer_cls is None:
        logger.error(f"Unsupported phase: {args.phase}")
        raise ValueError(f"Unsupported phase: {args.phase}")

    trainer = trainer_cls(config)
    trainer.train()
    logger.info("Training complete")

if __name__ == "__main__":
    main()
