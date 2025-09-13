# scripts/run_training.py

from utils.config_loader import load_config
from training.sft_trainer import SFTTrainer
from training.reward_trainer import RewardTrainer
from training.preference_reward_trainer import PreferenceRewardTrainer
from training.ppo_trainer import PPOTrainer

TRAINING_STEPS = [
    ("sft", SFTTrainer, "configs/sft.yaml"),
    ("reward_preference", PreferenceRewardTrainer, "configs/reward_preference.yaml"),
    ("ppo", PPOTrainer, "configs/ppo.yaml"),
]

def run_training(logger=None):
    """Run SFT → Reward → PPO sequentially."""
    for phase_name, TrainerCls, cfg_path in TRAINING_STEPS:
        if logger:
            logger.info(f"=== Starting {phase_name.upper()} phase")
        config = load_config(cfg_path)
        trainer = TrainerCls(config)
        trainer.train()
