"""
Reward Training Module

This module contains various reward model training implementations
for Reinforcement Learning with Human Feedback (RLHF).
"""

from .old_reward_trainer import PreferenceRewardTrainer
from .balanced_reward_trainer import BalancedRewardTrainer
from .refactored_balanced_trainer import RefactoredBalancedRewardTrainer, run_refactored_training

# Import all concrete implementations
from .loss_strategies import (
    BradleyTerryLoss, HingeLoss, LogSigmoidLoss, LossStrategyFactory
)
from .model_factory import RewardModelFactory, ModelFactoryRegistry
from .training_monitor import (
    BasicTrainingObserver, MetricsCollector, ConsoleLoggingStrategy,
    FileLoggingStrategy, ProgressTracker, CompositeTrainingObserver, TrainingMonitorFactory
)
from .dataset_manager import (
    PreferenceDatasetManager, PreferenceDataProcessor, PreferenceDataValidator,
    NoOpDataAugmentation, StandardDataIterator, DatasetManagerFactory
)
from .optimization_manager import (
    RewardOptimizationManager, CUDAMemoryManager, DeviceManager,
    PerformanceOptimizer, StandardOptimizationStrategy, OptimizationManagerFactory
)
from .model_persistence import (
    RewardModelPersistence, ConfigurationPersistence, MetadataManager,
    FileManager, ModelValidator, ModelPersistenceFactory
)

__all__ = [
    # Original trainers
    'PreferenceRewardTrainer', 
    'BalancedRewardTrainer',
    
    # Refactored trainer
    'RefactoredBalancedRewardTrainer',
    'run_refactored_training',
    
    # Loss strategies
    'BradleyTerryLoss',
    'HingeLoss', 
    'LogSigmoidLoss',
    'LossStrategyFactory',
    
    # Model factory
    'RewardModelFactory',
    'ModelFactoryRegistry',
    
    # Training monitor
    'BasicTrainingObserver',
    'MetricsCollector',
    'ConsoleLoggingStrategy',
    'FileLoggingStrategy',
    'ProgressTracker',
    'CompositeTrainingObserver',
    'TrainingMonitorFactory',
    
    # Dataset manager
    'PreferenceDatasetManager',
    'PreferenceDataProcessor',
    'PreferenceDataValidator',
    'NoOpDataAugmentation',
    'StandardDataIterator',
    'DatasetManagerFactory',
    
    # Optimization manager
    'RewardOptimizationManager',
    'CUDAMemoryManager',
    'DeviceManager',
    'PerformanceOptimizer',
    'StandardOptimizationStrategy',
    'OptimizationManagerFactory',
    
    # Model persistence
    'RewardModelPersistence',
    'ConfigurationPersistence',
    'MetadataManager',
    'FileManager',
    'ModelValidator',
    'ModelPersistenceFactory'
]
