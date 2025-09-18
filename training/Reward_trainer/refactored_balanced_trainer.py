"""
Refactored Balanced Reward Trainer

A completely refactored version of the BalancedRewardTrainer that uses design patterns,
dependency injection, and modular components for better maintainability and testability.

This trainer demonstrates proper SOLID principles implementation and clean architecture.
"""

import os
import time
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader

# Import all our interfaces and implementations
from .interfaces import (
    IModelFactory, IModelManager,
    ILossStrategy, ILossConfigurable, ILossMetrics,
    ITrainingObserver, IMetricsCollector, ILoggingStrategy, IProgressTracker,
    IDatasetManager, IDataProcessor, IDataValidator,
    IOptimizationManager, IMemoryManager, IDeviceManager,
    IModelPersistence, IConfigurationPersistence, IMetadataManager
)

# Import concrete implementations
from .model_factory import RewardModelFactory, ModelFactoryRegistry
from .loss_strategies import LossStrategyFactory
from .training_monitor import (
    BasicTrainingObserver, MetricsCollector, ConsoleLoggingStrategy,
    ProgressTracker, CompositeTrainingObserver, TrainingMonitorFactory
)
from .dataset_manager import (
    PreferenceDatasetManager, PreferenceDataProcessor, PreferenceDataValidator,
    DatasetManagerFactory
)
from .optimization_manager import (
    RewardOptimizationManager, CUDAMemoryManager, DeviceManager,
    PerformanceOptimizer, StandardOptimizationStrategy, OptimizationManagerFactory
)
from .model_persistence import (
    RewardModelPersistence, ConfigurationPersistence, MetadataManager,
    ModelPersistenceFactory
)

from utils.logging_utils import setup_logger


class RefactoredBalancedRewardTrainer:
    """
    Refactored Balanced Reward Trainer using design patterns and dependency injection.
    
    This trainer demonstrates proper SOLID principles implementation with:
    - Single Responsibility: Each component has one clear purpose
    - Open/Closed: Easy to extend without modification
    - Liskov Substitution: All components are substitutable
    - Interface Segregation: Focused, non-bloated interfaces
    - Dependency Inversion: Depends on abstractions, not concretions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the refactored trainer with dependency injection.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = setup_logger("refactored_balanced_trainer")
        
        # Initialize all components using dependency injection
        self._initialize_components()
        
        # Training state
        self.training_start_time = None
        self.current_epoch = 0
        self.current_step = 0
        self.is_training = False
        
        self.logger.info("Refactored Balanced Reward Trainer initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all components using dependency injection."""
        try:
            # 1. Model Factory and Manager
            self.model_factory = ModelFactoryRegistry.create_factory("reward_model")
            self.device = self.model_factory.setup_device(self.config)
            
            # 2. Loss Strategy
            loss_config = self.config.get("training", {}).get("loss", {})
            loss_strategy_name = loss_config.get("strategy", "bradley_terry")
            self.loss_strategy = LossStrategyFactory.create_strategy(
                loss_strategy_name, 
                loss_config.get("parameters", {})
            )
            
            # 3. Training Monitor
            self.training_observer = self._create_training_observer()
            self.metrics_collector = MetricsCollector()
            self.progress_tracker = ProgressTracker()
            
            # 4. Dataset Manager
            self.dataset_manager = self._create_dataset_manager()
            self.data_validator = PreferenceDataValidator()
            
            # 5. Optimization Manager
            self.optimization_manager = OptimizationManagerFactory.create_optimization_manager()
            self.memory_manager = OptimizationManagerFactory.create_memory_manager()
            self.device_manager = OptimizationManagerFactory.create_device_manager()
            
            # 6. Model Persistence
            self.model_persistence = ModelPersistenceFactory.create_model_persistence()
            self.config_persistence = ModelPersistenceFactory.create_configuration_persistence()
            self.metadata_manager = ModelPersistenceFactory.create_metadata_manager()
            
            # 7. Create and configure model
            self._create_and_configure_model()
            
            # 8. Create dataset and dataloader
            self._create_dataset_and_dataloader()
            
            # 9. Create optimizer
            self._create_optimizer()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise RuntimeError(f"Component initialization failed: {e}") from e
    
    def _create_training_observer(self) -> ITrainingObserver:
        """Create training observer with multiple strategies."""
        try:
            # Create individual observers
            basic_observer = TrainingMonitorFactory.create_basic_observer("refactored_trainer")
            
            # Create logging strategies
            console_logger = TrainingMonitorFactory.create_console_logger("refactored_console")
            
            # Add file logging if specified
            log_file = self.config.get("output", {}).get("log_file")
            observers = [basic_observer]
            
            if log_file:
                file_observer = TrainingMonitorFactory.create_file_observer(log_file, "refactored_file")
                observers.append(file_observer)
            
            # Create composite observer
            return TrainingMonitorFactory.create_composite_observer(observers)
            
        except Exception as e:
            self.logger.error(f"Failed to create training observer: {e}")
            raise
    
    def _create_dataset_manager(self) -> IDatasetManager:
        """Create dataset manager with tokenizer."""
        try:
            # Create tokenizer first
            tokenizer = self.model_factory.create_tokenizer(self.config)
            
            # Create dataset manager
            return DatasetManagerFactory.create_preference_manager(tokenizer)
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset manager: {e}")
            raise
    
    def _create_and_configure_model(self) -> None:
        """Create and configure the reward model."""
        try:
            # Create base model
            self.base_model = self.model_factory.create_base_model(self.config)
            
            # Create reward model
            self.model = self.model_factory.create_reward_model(self.base_model, self.config)
            
            # Move to device
            self.model = self.model_factory.move_to_device(self.model, self.device)
            
            # Apply optimizations
            self.model = self.model_factory.configure_model_optimizations(self.model, self.config)
            
            self.logger.info("Model created and configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create and configure model: {e}")
            raise
    
    def _create_dataset_and_dataloader(self) -> None:
        """Create dataset and dataloader."""
        try:
            # Load dataset
            dataset_config = self.config.get("dataset", {})
            self.dataset = self.dataset_manager.load_dataset(dataset_config)
            
            # Validate dataset
            if not self.data_validator.validate_dataset(self.dataset):
                errors = self.data_validator.get_validation_errors()
                raise ValueError(f"Dataset validation failed: {errors}")
            
            # Create dataloader
            dataloader_config = self.config.get("training", {})
            self.dataloader = self.dataset_manager.create_dataloader(self.dataset, dataloader_config)
            
            self.logger.info(f"Dataset and dataloader created: {len(self.dataset)} examples")
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset and dataloader: {e}")
            raise
    
    def _create_optimizer(self) -> None:
        """Create optimizer and configure mixed precision."""
        try:
            # Create optimizer
            training_config = self.config.get("training", {})
            self.optimizer = self.optimization_manager.create_optimizer(self.model, training_config)
            
            # Setup mixed precision
            self.use_mixed_precision, self.scaler = self.optimization_manager.setup_mixed_precision(training_config)
            
            # Configure gradient accumulation
            self.gradient_accumulation_steps = self.optimization_manager.configure_gradient_accumulation(training_config)
            
            # Setup gradient clipping
            self.max_grad_norm = self.optimization_manager.setup_gradient_clipping(training_config)
            
            self.logger.info("Optimizer created and configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create optimizer: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """
        Train the reward model using the refactored architecture.
        
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            self.logger.info("Starting refactored training...")
            self.is_training = True
            self.training_start_time = time.time()
            
            # Notify training start
            self.training_observer.on_training_start(self.config)
            
            # Get training parameters
            training_config = self.config.get("training", {})
            epochs = int(training_config.get("epochs", 5))
            logging_steps = int(training_config.get("logging_steps", 15))
            
            # Training loop
            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch
                self._train_epoch(epoch, epochs, logging_steps)
            
            # Training completed
            final_metrics = self.metrics_collector.get_training_summary()
            self.training_observer.on_training_end(final_metrics)
            
            # Save model
            self._save_model()
            
            self.is_training = False
            total_time = time.time() - self.training_start_time
            
            self.logger.info(f"Training completed successfully in {total_time:.2f} seconds")
            
            # Get final accuracy from the last epoch
            final_accuracy = final_metrics.get("final_accuracy", 0.0)
            
            return {
                "training_completed": True,
                "total_time": total_time,
                "final_metrics": final_metrics,
                "epochs_completed": epochs,
                "final_accuracy": final_accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.is_training = False
            raise RuntimeError(f"Training failed: {e}") from e
    
    def _train_epoch(self, epoch: int, total_epochs: int, logging_steps: int) -> None:
        """Train a single epoch."""
        try:
            self.training_observer.on_epoch_start(epoch, total_epochs)
            
            self.model.train()
            epoch_start_time = time.time()
            total_loss = 0.0
            step_count = 0
            
            for step, batch in enumerate(self.dataloader):
                self.current_step = step
                
                # Validate batch
                if not self.data_validator.validate_batch(batch):
                    errors = self.data_validator.get_validation_errors()
                    self.logger.warning(f"Invalid batch at step {step}: {errors}")
                    continue
                
                # Training step
                step_result = self._train_step(batch, step)
                loss = step_result['loss']
                accuracy = step_result['accuracy']
                total_loss += loss
                step_count += 1
                
                # Collect metrics
                step_metrics = {
                    "loss": loss,
                    "accuracy": accuracy,
                    "epoch": epoch,
                    "step": step
                }
                self.metrics_collector.collect_metrics(step_metrics)
                
                # Logging
                if step % logging_steps == 0:
                    self._log_step_metrics(epoch, step, loss, step_metrics)
                
                # Update progress
                self.progress_tracker.update_progress(
                    step, len(self.dataloader), f"Epoch {epoch}"
                )
            
            # Epoch completed
            epoch_metrics = self.metrics_collector.get_aggregated_metrics(epoch)
            epoch_time = time.time() - epoch_start_time
            avg_accuracy = epoch_metrics.get("accuracy_mean", 0.0)
            epoch_metrics.update({
                "epoch_time": epoch_time,
                "avg_loss": total_loss / step_count if step_count > 0 else 0.0,
                "avg_accuracy": avg_accuracy,
                "final_accuracy": avg_accuracy * 100  # Convert to percentage
            })
            
            self.training_observer.on_epoch_end(epoch, epoch_metrics)
            
            # Memory cleanup
            self.memory_manager.clear_cache()
            
        except Exception as e:
            self.logger.error(f"Epoch {epoch} failed: {e}")
            raise
    
    def _train_step(self, batch: Dict[str, torch.Tensor], step: int) -> float:
        """Execute a single training step."""
        try:
            # Move batch to device
            device = next(self.model.parameters()).device
            chosen_input_ids = batch['chosen_input_ids'].to(device, non_blocking=True)
            chosen_attention_mask = batch['chosen_attention_mask'].to(device, non_blocking=True)
            rejected_input_ids = batch['rejected_input_ids'].to(device, non_blocking=True)
            rejected_attention_mask = batch['rejected_attention_mask'].to(device, non_blocking=True)
            
            # Forward pass
            if self.use_mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    chosen_rewards = self.model(chosen_input_ids, attention_mask=chosen_attention_mask)
                    rejected_rewards = self.model(rejected_input_ids, attention_mask=rejected_attention_mask)
                    loss = self.loss_strategy.compute_loss(chosen_rewards, rejected_rewards)
                    loss = loss / self.gradient_accumulation_steps
            else:
                chosen_rewards = self.model(chosen_input_ids, attention_mask=chosen_attention_mask)
                rejected_rewards = self.model(rejected_input_ids, attention_mask=rejected_attention_mask)
                loss = self.loss_strategy.compute_loss(chosen_rewards, rejected_rewards)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Compute accuracy for monitoring
            with torch.no_grad():
                accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            
            return {
                'loss': loss.item() * self.gradient_accumulation_steps,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            raise
    
    def _log_step_metrics(self, epoch: int, step: int, loss: float, metrics: Dict[str, Any]) -> None:
        """Log step metrics."""
        try:
            # Compute additional metrics using loss strategy
            if hasattr(self.loss_strategy, 'compute_metrics'):
                # Get rewards for metrics computation
                device = next(self.model.parameters()).device
                with torch.no_grad():
                    # This is a simplified version - in practice, you'd need to recompute
                    # or store the rewards from the forward pass
                    additional_metrics = {}
                
                metrics.update(additional_metrics)
            
            # Notify training observer
            self.training_observer.on_step_complete(step, metrics)
            
        except Exception as e:
            self.logger.warning(f"Failed to log step metrics: {e}")
    
    def _save_model(self) -> None:
        """Save the trained model and configuration."""
        try:
            output_dir = self.config.get("output", {}).get("model_dir", "models/refactored_reward_model")
            
            # Create metadata
            metadata = self.metadata_manager.create_metadata(
                self.model, 
                self.config,
                {"training_completed": True, "epochs": self.current_epoch}
            )
            
            # Save model
            self.model_persistence.save_model(
                self.model, 
                output_dir, 
                self.config, 
                metadata
            )
            
            # Save configuration
            config_path = os.path.join(output_dir, "training_config.json")
            self.config_persistence.save_configuration(self.config, config_path)
            
            self.logger.info(f"Model saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'memory_manager'):
                self.memory_manager.cleanup_resources()
            
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'base_model'):
                del self.base_model
            
            self.logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")


def run_refactored_training(config_path: str) -> bool:
    """
    Run refactored reward model training.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if training succeeded, False otherwise
    """
    try:
        from utils.config_loader import load_config
        
        # Load configuration
        config = load_config(config_path)
        
        # Create trainer
        trainer = RefactoredBalancedRewardTrainer(config)
        
        try:
            # Train model
            results = trainer.train()
            
            # Log results
            trainer.logger.info(f"Training completed successfully: {results}")
            
            return True
            
        finally:
            # Cleanup
            trainer.cleanup()
    
    except Exception as e:
        print(f"Training failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/reward_preference_balanced.yaml"
    
    success = run_refactored_training(config_path)
    exit(0 if success else 1)
