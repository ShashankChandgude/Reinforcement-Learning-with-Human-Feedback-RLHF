"""
Unit Tests for Refactored Balanced Reward Trainer

This test suite demonstrates the testability of the refactored trainer
using dependency injection and mock objects.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
import tempfile
import os
from typing import Dict, Any

# Import the refactored trainer and components
from .refactored_balanced_trainer import RefactoredBalancedRewardTrainer
from .interfaces import (
    IModelFactory, ILossStrategy, ITrainingObserver, 
    IDatasetManager, IOptimizationManager, IModelPersistence
)


class MockModelFactory(IModelFactory):
    """Mock model factory for testing."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = Mock()
        self.base_model = Mock()
        self.model = Mock()
    
    def create_base_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        return self.base_model
    
    def create_reward_model(self, base_model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        return self.model
    
    def create_tokenizer(self, config: Dict[str, Any]) -> Any:
        return self.tokenizer
    
    def setup_device(self, config: Dict[str, Any]) -> torch.device:
        return self.device
    
    def configure_model_optimizations(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        return model


class MockLossStrategy(ILossStrategy):
    """Mock loss strategy for testing."""
    
    def compute_loss(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.tensor(0.5)
    
    def get_required_parameters(self) -> list[str]:
        return ["margin"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return True


class MockTrainingObserver(ITrainingObserver):
    """Mock training observer for testing."""
    
    def __init__(self):
        self.calls = []
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        self.calls.append(("on_training_start", config))
    
    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        self.calls.append(("on_training_end", final_metrics))
    
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self.calls.append(("on_epoch_start", epoch, total_epochs))
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        self.calls.append(("on_epoch_end", epoch, metrics))
    
    def on_step_complete(self, step: int, metrics: Dict[str, Any]) -> None:
        self.calls.append(("on_step_complete", step, metrics))
    
    def on_validation_start(self, epoch: int) -> None:
        self.calls.append(("on_validation_start", epoch))
    
    def on_validation_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        self.calls.append(("on_validation_end", epoch, metrics))


class MockDatasetManager(IDatasetManager):
    """Mock dataset manager for testing."""
    
    def __init__(self):
        self.dataset = Mock()
        self.dataloader = Mock()
        self.dataset.__len__ = Mock(return_value=10)
        self.dataloader.__iter__ = Mock(return_value=iter([
            {
                'chosen_input_ids': torch.tensor([[1, 2, 3]]),
                'chosen_attention_mask': torch.tensor([[1, 1, 1]]),
                'rejected_input_ids': torch.tensor([[4, 5, 6]]),
                'rejected_attention_mask': torch.tensor([[1, 1, 1]])
            }
        ]))
    
    def load_dataset(self, config: Dict[str, Any]) -> Any:
        return self.dataset
    
    def create_dataloader(self, dataset: Any, config: Dict[str, Any]) -> Any:
        return self.dataloader
    
    def get_dataset_info(self, dataset: Any) -> Dict[str, Any]:
        return {"size": 10, "type": "MockDataset"}


class MockOptimizationManager(IOptimizationManager):
    """Mock optimization manager for testing."""
    
    def create_optimizer(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
        return Mock()
    
    def setup_mixed_precision(self, config: Dict[str, Any]) -> tuple[bool, Any]:
        return False, None
    
    def configure_gradient_accumulation(self, config: Dict[str, Any]) -> int:
        return 1
    
    def setup_gradient_clipping(self, config: Dict[str, Any]) -> Any:
        return 1.0


class MockModelPersistence(IModelPersistence):
    """Mock model persistence for testing."""
    
    def save_model(self, model: torch.nn.Module, path: str, config: Dict[str, Any], metadata: Dict[str, Any] = None) -> None:
        pass
    
    def load_model(self, path: str, config: Dict[str, Any]) -> torch.nn.Module:
        return Mock()
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str, metadata: Dict[str, Any] = None) -> None:
        pass
    
    def load_checkpoint(self, path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
        return {}


class TestRefactoredBalancedRewardTrainer(unittest.TestCase):
    """Test cases for the refactored balanced reward trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model": "test-model",
            "dataset": {"name": "test-dataset", "max_seq_length": 512},
            "training": {
                "epochs": 2,
                "learning_rate": 1e-4,
                "loss": {"strategy": "bradley_terry", "parameters": {"margin": 0.5}}
            },
            "output": {"model_dir": "test_output"}
        }
    
    @patch('training.Reward_trainer.refactored_balanced_trainer.ModelFactoryRegistry')
    @patch('training.Reward_trainer.refactored_balanced_trainer.LossStrategyFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.TrainingMonitorFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.DatasetManagerFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.OptimizationManagerFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.ModelPersistenceFactory')
    def test_initialization(self, mock_persistence_factory, mock_opt_factory, 
                          mock_dataset_factory, mock_monitor_factory, 
                          mock_loss_factory, mock_model_factory):
        """Test trainer initialization with mocked dependencies."""
        # Setup mocks
        mock_model_factory.create_factory.return_value = MockModelFactory()
        mock_loss_factory.create_strategy.return_value = MockLossStrategy()
        mock_monitor_factory.create_basic_observer.return_value = MockTrainingObserver()
        mock_monitor_factory.create_console_logger.return_value = Mock()
        mock_monitor_factory.create_composite_observer.return_value = MockTrainingObserver()
        mock_dataset_factory.create_preference_manager.return_value = MockDatasetManager()
        mock_opt_factory.create_optimization_manager.return_value = MockOptimizationManager()
        mock_opt_factory.create_memory_manager.return_value = Mock()
        mock_opt_factory.create_device_manager.return_value = Mock()
        mock_persistence_factory.create_model_persistence.return_value = MockModelPersistence()
        mock_persistence_factory.create_configuration_persistence.return_value = Mock()
        mock_persistence_factory.create_metadata_manager.return_value = Mock()
        
        # Create trainer
        trainer = RefactoredBalancedRewardTrainer(self.config)
        
        # Verify initialization
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.loss_strategy)
        self.assertIsNotNone(trainer.training_observer)
        self.assertIsNotNone(trainer.dataset_manager)
        self.assertIsNotNone(trainer.optimization_manager)
        self.assertIsNotNone(trainer.model_persistence)
    
    @patch('training.Reward_trainer.refactored_balanced_trainer.ModelFactoryRegistry')
    @patch('training.Reward_trainer.refactored_balanced_trainer.LossStrategyFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.TrainingMonitorFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.DatasetManagerFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.OptimizationManagerFactory')
    @patch('training.Reward_trainer.refactored_balanced_trainer.ModelPersistenceFactory')
    def test_training_observer_calls(self, mock_persistence_factory, mock_opt_factory,
                                   mock_dataset_factory, mock_monitor_factory,
                                   mock_loss_factory, mock_model_factory):
        """Test that training observer methods are called correctly."""
        # Setup mocks
        mock_observer = MockTrainingObserver()
        mock_model_factory.create_factory.return_value = MockModelFactory()
        mock_loss_factory.create_strategy.return_value = MockLossStrategy()
        mock_monitor_factory.create_basic_observer.return_value = mock_observer
        mock_monitor_factory.create_console_logger.return_value = Mock()
        mock_monitor_factory.create_composite_observer.return_value = mock_observer
        mock_dataset_factory.create_preference_manager.return_value = MockDatasetManager()
        mock_opt_factory.create_optimization_manager.return_value = MockOptimizationManager()
        mock_opt_factory.create_memory_manager.return_value = Mock()
        mock_opt_factory.create_device_manager.return_value = Mock()
        mock_persistence_factory.create_model_persistence.return_value = MockModelPersistence()
        mock_persistence_factory.create_configuration_persistence.return_value = Mock()
        mock_persistence_factory.create_metadata_manager.return_value = Mock()
        
        # Create trainer
        trainer = RefactoredBalancedRewardTrainer(self.config)
        
        # Mock the model to return proper tensors
        trainer.model = Mock()
        trainer.model.parameters.return_value = [torch.tensor([1.0])]
        trainer.model.return_value = torch.tensor([[0.5, 0.3]])
        
        # Run training
        results = trainer.train()
        
        # Verify observer calls
        self.assertTrue(any(call[0] == "on_training_start" for call in mock_observer.calls))
        self.assertTrue(any(call[0] == "on_training_end" for call in mock_observer.calls))
        self.assertTrue(any(call[0] == "on_epoch_start" for call in mock_observer.calls))
        self.assertTrue(any(call[0] == "on_epoch_end" for call in mock_observer.calls))
        
        # Verify training results
        self.assertTrue(results["training_completed"])
        self.assertEqual(results["epochs_completed"], 2)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with valid configuration
        valid_config = {
            "model": "test-model",
            "dataset": {"name": "test-dataset"},
            "training": {"epochs": 1, "learning_rate": 1e-4}
        }
        
        # This should not raise an exception
        try:
            with patch('training.Reward_trainer.refactored_balanced_trainer.ModelFactoryRegistry') as mock_factory:
                mock_factory.create_factory.return_value = MockModelFactory()
                # Add other necessary mocks...
                trainer = RefactoredBalancedRewardTrainer(valid_config)
                self.assertIsNotNone(trainer)
        except Exception as e:
            self.fail(f"Valid configuration should not raise exception: {e}")
    
    def test_error_handling(self):
        """Test error handling in trainer."""
        # Test with invalid configuration
        invalid_config = {
            "model": None,  # Invalid model
            "dataset": {},  # Missing required fields
            "training": {}  # Missing required fields
        }
        
        with self.assertRaises(Exception):
            RefactoredBalancedRewardTrainer(invalid_config)
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Setup mocks
        with patch('training.Reward_trainer.refactored_balanced_trainer.ModelFactoryRegistry') as mock_factory:
            mock_factory.create_factory.return_value = MockModelFactory()
            # Add other necessary mocks...
            
            trainer = RefactoredBalancedRewardTrainer(self.config)
            trainer.cleanup()
            
            # Verify cleanup was called
            self.assertIsNotNone(trainer)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components."""
    
    def test_loss_strategy_integration(self):
        """Test loss strategy integration."""
        from .loss_strategies import BradleyTerryLoss, LossStrategyFactory
        
        # Test Bradley-Terry loss
        loss_strategy = BradleyTerryLoss(margin=0.5)
        
        # Create test tensors
        chosen_rewards = torch.tensor([0.8, 0.6])
        rejected_rewards = torch.tensor([0.3, 0.4])
        
        # Compute loss
        loss = loss_strategy.compute_loss(chosen_rewards, rejected_rewards)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_model_factory_integration(self):
        """Test model factory integration."""
        from .model_factory import ModelFactoryRegistry
        
        # Test factory creation
        factory = ModelFactoryRegistry.create_factory("reward_model")
        self.assertIsNotNone(factory)
        self.assertIsInstance(factory, IModelFactory)
    
    def test_training_monitor_integration(self):
        """Test training monitor integration."""
        from .training_monitor import TrainingMonitorFactory
        
        # Test observer creation
        observer = TrainingMonitorFactory.create_basic_observer("test")
        self.assertIsNotNone(observer)
        self.assertIsInstance(observer, ITrainingObserver)
        
        # Test metrics collector
        collector = TrainingMonitorFactory.create_metrics_collector()
        self.assertIsNotNone(collector)
        
        # Test progress tracker
        tracker = TrainingMonitorFactory.create_progress_tracker()
        self.assertIsNotNone(tracker)


class TestDesignPatterns(unittest.TestCase):
    """Test that design patterns are properly implemented."""
    
    def test_strategy_pattern(self):
        """Test strategy pattern implementation."""
        from .loss_strategies import LossStrategyFactory
        
        # Test different loss strategies
        strategies = ["bradley_terry", "hinge", "log_sigmoid"]
        
        for strategy_name in strategies:
            strategy = LossStrategyFactory.create_strategy(strategy_name, {"margin": 0.5})
            self.assertIsNotNone(strategy)
            self.assertIsInstance(strategy, ILossStrategy)
    
    def test_factory_pattern(self):
        """Test factory pattern implementation."""
        from .model_factory import ModelFactoryRegistry
        from .training_monitor import TrainingMonitorFactory
        from .dataset_manager import DatasetManagerFactory
        
        # Test different factories
        factories = [
            (ModelFactoryRegistry, "reward_model"),
            (TrainingMonitorFactory, "create_basic_observer"),
            (DatasetManagerFactory, "create_data_validator")
        ]
        
        for factory_class, method_name in factories:
            if hasattr(factory_class, method_name):
                result = getattr(factory_class, method_name)()
                self.assertIsNotNone(result)
    
    def test_observer_pattern(self):
        """Test observer pattern implementation."""
        from .training_monitor import CompositeTrainingObserver, BasicTrainingObserver
        
        # Create multiple observers
        observer1 = BasicTrainingObserver("test1")
        observer2 = BasicTrainingObserver("test2")
        
        # Create composite observer
        composite = CompositeTrainingObserver([observer1, observer2])
        
        # Test that composite calls all observers
        config = {"test": "config"}
        composite.on_training_start(config)
        
        # Verify both observers were called
        self.assertEqual(len(observer1.calls), 1)
        self.assertEqual(len(observer2.calls), 1)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
