"""
Supervised Training Module

This module contains supervised learning approaches for RLHF,
including reward-ranked supervised training as an alternative to PPO.
"""

from .supervised_rlhf_trainer import SupervisedRLHFTrainer, RankedResponseDataset

__all__ = ['SupervisedRLHFTrainer', 'RankedResponseDataset']
