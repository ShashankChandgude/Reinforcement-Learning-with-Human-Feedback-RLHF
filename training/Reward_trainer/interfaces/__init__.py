"""
Interfaces Module

This module contains all the abstract base classes and interfaces
for the RLHF reward trainer components. These interfaces define
the contracts that concrete implementations must follow.

Interfaces:
- IModelFactory: Model creation and management
- ILossStrategy: Loss computation strategies
- ITrainingObserver: Training monitoring and observation
- IDatasetManager: Dataset management and data loading
- IOptimizationManager: Optimization and memory management
- IModelPersistence: Model saving and loading
"""

# Model Factory Interfaces
from .model_factory_interface import (
    IModelFactory,
    IModelManager
)

# Loss Strategy Interfaces
from .loss_strategy_interface import (
    ILossStrategy,
    ILossConfigurable,
    ILossMetrics
)

# Training Observer Interfaces
from .training_observer_interface import (
    ITrainingObserver,
    IMetricsCollector,
    ILoggingStrategy,
    IProgressTracker
)

# Dataset Manager Interfaces
from .dataset_manager_interface import (
    IDatasetManager,
    IDataProcessor,
    IDataValidator,
    IDataAugmentation,
    IDataIterator
)

# Optimization Manager Interfaces
from .optimization_manager_interface import (
    IOptimizationManager,
    IMemoryManager,
    IDeviceManager,
    IPerformanceOptimizer,
    IOptimizationStrategy
)

# Model Persistence Interfaces
from .model_persistence_interface import (
    IModelPersistence,
    IConfigurationPersistence,
    IMetadataManager,
    IFileManager,
    IModelValidator
)

__all__ = [
    # Model Factory
    'IModelFactory',
    'IModelManager',
    
    # Loss Strategy
    'ILossStrategy',
    'ILossConfigurable',
    'ILossMetrics',
    
    # Training Observer
    'ITrainingObserver',
    'IMetricsCollector',
    'ILoggingStrategy',
    'IProgressTracker',
    
    # Dataset Manager
    'IDatasetManager',
    'IDataProcessor',
    'IDataValidator',
    'IDataAugmentation',
    'IDataIterator',
    
    # Optimization Manager
    'IOptimizationManager',
    'IMemoryManager',
    'IDeviceManager',
    'IPerformanceOptimizer',
    'IOptimizationStrategy',
    
    # Model Persistence
    'IModelPersistence',
    'IConfigurationPersistence',
    'IMetadataManager',
    'IFileManager',
    'IModelValidator'
]
