"""
PPO Training Module

This module contains the PPO (Proximal Policy Optimization) trainer implementation
for Reinforcement Learning with Human Feedback (RLHF).
"""

from .ppo_preference_trainer import PPOPreferenceTrainer, PPOConfig, build_config_from_yaml

__all__ = ['PPOPreferenceTrainer', 'PPOConfig', 'build_config_from_yaml']
