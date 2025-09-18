"""
Training Monitor Implementations

Concrete implementations of training monitoring and observation for RLHF training.
Supports different monitoring strategies and logging backends.
"""

import time
import json
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from .interfaces.training_observer_interface import (
    ITrainingObserver, 
    IMetricsCollector, 
    ILoggingStrategy, 
    IProgressTracker
)
from utils.logging_utils import setup_logger


class BasicTrainingObserver(ITrainingObserver):
    """
    Basic training observer implementation.
    
    Provides standard training monitoring with logging and metrics collection.
    """
    
    def __init__(self, logger_name: str = "training_observer"):
        """
        Initialize the training observer.
        
        Args:
            logger_name: Name for the logger
        """
        self.logger = setup_logger(logger_name)
        self.training_start_time = None
        self.epoch_start_time = None
        self.step_times = deque(maxlen=100)  # Keep last 100 step times for averaging
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Called when training starts."""
        self.training_start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            self.logger.info("=" * 60)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"Total training time: {total_time:.2f} seconds")
            self.logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called when an epoch starts."""
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called when an epoch ends."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
            self.logger.info(f"Epoch metrics: {json.dumps(metrics, indent=2)}")
    
    def on_step_complete(self, step: int, metrics: Dict[str, Any]) -> None:
        """Called when a training step completes."""
        current_time = time.time()
        if self.epoch_start_time:
            step_time = current_time - self.epoch_start_time
            self.step_times.append(step_time)
        
        # Log step metrics
        self.logger.info(f"Step {step} completed - {json.dumps(metrics, indent=2)}")
    
    def on_validation_start(self, epoch: int) -> None:
        """Called when validation starts."""
        self.logger.info(f"Validation started for epoch {epoch}")
    
    def on_validation_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called when validation ends."""
        self.logger.info(f"Validation completed for epoch {epoch}")
        self.logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")


class MetricsCollector(IMetricsCollector):
    """
    Metrics collector implementation.
    
    Collects and aggregates training metrics across epochs and steps.
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.step_metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.training_metrics = {}
        self.current_epoch = 0
        self.current_step = 0
    
    def collect_metrics(self, metrics: Dict[str, Any]) -> None:
        """Collect metrics from a training step or epoch."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.step_metrics[key].append(value)
            else:
                # Store non-numeric metrics separately
                self.step_metrics[f"{key}_raw"].append(value)
    
    def get_aggregated_metrics(self, epoch: int) -> Dict[str, Any]:
        """Get aggregated metrics for an epoch."""
        aggregated = {}
        
        for key, values in self.step_metrics.items():
            if values and isinstance(values[0], (int, float)):
                aggregated[f"{key}_mean"] = sum(values) / len(values)
                aggregated[f"{key}_min"] = min(values)
                aggregated[f"{key}_max"] = max(values)
                aggregated[f"{key}_count"] = len(values)
            else:
                # For non-numeric metrics, just take the last value
                aggregated[key] = values[-1] if values else None
        
        # Store epoch metrics
        self.epoch_metrics[epoch] = aggregated.copy()
        
        # Clear step metrics for next epoch
        self.step_metrics.clear()
        
        return aggregated
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of entire training run."""
        summary = {
            "total_epochs": len(self.epoch_metrics),
            "epoch_metrics": dict(self.epoch_metrics)
        }
        
        # Calculate overall statistics
        if self.epoch_metrics:
            all_losses = []
            all_accuracies = []
            
            for epoch_data in self.epoch_metrics.values():
                if "loss_mean" in epoch_data:
                    all_losses.append(epoch_data["loss_mean"])
                if "accuracy_mean" in epoch_data:
                    all_accuracies.append(epoch_data["accuracy_mean"])
            
            if all_losses:
                summary["overall_loss_mean"] = sum(all_losses) / len(all_losses)
                summary["overall_loss_min"] = min(all_losses)
                summary["overall_loss_max"] = max(all_losses)
            
            if all_accuracies:
                summary["overall_accuracy_mean"] = sum(all_accuracies) / len(all_accuracies)
                summary["overall_accuracy_min"] = min(all_accuracies)
                summary["overall_accuracy_max"] = max(all_accuracies)
        
        return summary
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.step_metrics.clear()
        self.epoch_metrics.clear()
        self.training_metrics.clear()
        self.current_epoch = 0
        self.current_step = 0


class ConsoleLoggingStrategy(ILoggingStrategy):
    """
    Console logging strategy implementation.
    
    Logs all messages to the console using the standard logger.
    """
    
    def __init__(self, logger_name: str = "console_logger"):
        """
        Initialize console logging strategy.
        
        Args:
            logger_name: Name for the logger
        """
        self.logger = setup_logger(logger_name)
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def log_error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log training metrics."""
        if step is not None:
            message = f"Step {step} | {json.dumps(metrics, indent=2)}"
        else:
            message = f"Metrics | {json.dumps(metrics, indent=2)}"
        self.logger.info(message)


class FileLoggingStrategy(ILoggingStrategy):
    """
    File logging strategy implementation.
    
    Logs all messages to a file in addition to console.
    """
    
    def __init__(self, log_file: str, logger_name: str = "file_logger"):
        """
        Initialize file logging strategy.
        
        Args:
            log_file: Path to the log file
            logger_name: Name for the logger
        """
        self.logger = setup_logger(logger_name)
        self.log_file = log_file
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def log_error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log training metrics."""
        if step is not None:
            message = f"Step {step} | {json.dumps(metrics, indent=2)}"
        else:
            message = f"Metrics | {json.dumps(metrics, indent=2)}"
        self.logger.info(message)


class FileTrainingObserver(ITrainingObserver):
    """
    File-based training observer implementation.
    
    Logs all training events to a file in addition to console.
    """
    
    def __init__(self, log_file: str, logger_name: str = "file_training_observer"):
        """
        Initialize file training observer.
        
        Args:
            log_file: Path to the log file
            logger_name: Name for the logger
        """
        self.logger = setup_logger(logger_name)
        self.log_file = log_file
        self.training_start_time = None
        self.epoch_start_time = None
        self.step_times = deque(maxlen=100)
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Called when training starts."""
        self.training_start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED (FILE OBSERVER)")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            self.logger.info("=" * 60)
            self.logger.info("TRAINING COMPLETED (FILE OBSERVER)")
            self.logger.info("=" * 60)
            self.logger.info(f"Total training time: {total_time:.2f} seconds")
            self.logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called when an epoch starts."""
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch}/{total_epochs} started (FILE OBSERVER)")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called when an epoch ends."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds (FILE OBSERVER)")
            self.logger.info(f"Epoch metrics: {json.dumps(metrics, indent=2)}")
    
    def on_step_complete(self, step: int, metrics: Dict[str, Any]) -> None:
        """Called when a training step completes."""
        current_time = time.time()
        if self.epoch_start_time:
            step_time = current_time - self.epoch_start_time
            self.step_times.append(step_time)
        
        # Log step metrics
        self.logger.info(f"Step {step} completed (FILE OBSERVER) - {json.dumps(metrics, indent=2)}")
    
    def on_validation_start(self, epoch: int) -> None:
        """Called when validation starts."""
        self.logger.info(f"Validation started for epoch {epoch} (FILE OBSERVER)")
    
    def on_validation_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called when validation ends."""
        self.logger.info(f"Validation completed for epoch {epoch} (FILE OBSERVER)")
        self.logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")


class ProgressTracker(IProgressTracker):
    """
    Progress tracker implementation.
    
    Tracks training progress and provides time estimates.
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.start_time = None
        self.current_progress = 0
        self.total_progress = 0
        self.description = ""
        self.step_times = deque(maxlen=10)  # Keep last 10 step times
        self.last_update_time = None
    
    def update_progress(self, current: int, total: int, description: str = "") -> None:
        """Update progress tracking."""
        self.current_progress = current
        self.total_progress = total
        self.description = description
        
        if self.start_time is None:
            self.start_time = time.time()
        
        current_time = time.time()
        if self.last_update_time:
            step_time = current_time - self.last_update_time
            self.step_times.append(step_time)
        
        self.last_update_time = current_time
    
    def get_progress_percentage(self) -> float:
        """Get current progress as percentage."""
        if self.total_progress == 0:
            return 0.0
        return (self.current_progress / self.total_progress) * 100.0
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """Get estimated time remaining for training."""
        if not self.step_times or self.total_progress == 0:
            return None
        
        # Calculate average step time
        avg_step_time = sum(self.step_times) / len(self.step_times)
        
        # Calculate remaining steps
        remaining_steps = self.total_progress - self.current_progress
        
        # Estimate time remaining
        estimated_time = remaining_steps * avg_step_time
        
        return estimated_time


class CompositeTrainingObserver(ITrainingObserver):
    """
    Composite training observer that combines multiple observers.
    
    This allows for multiple monitoring strategies to be used simultaneously.
    """
    
    def __init__(self, observers: List[ITrainingObserver]):
        """
        Initialize composite observer.
        
        Args:
            observers: List of training observers to combine
        """
        self.observers = observers
        self.logger = setup_logger("composite_observer")
        self.logger.info(f"Initialized composite observer with {len(observers)} observers")
    
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """Called when training starts."""
        for observer in self.observers:
            try:
                observer.on_training_start(config)
            except Exception as e:
                self.logger.error(f"Error in observer {type(observer).__name__}: {e}")
    
    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        for observer in self.observers:
            try:
                observer.on_training_end(final_metrics)
            except Exception as e:
                self.logger.error(f"Error in observer {type(observer).__name__}: {e}")
    
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called when an epoch starts."""
        for observer in self.observers:
            try:
                observer.on_epoch_start(epoch, total_epochs)
            except Exception as e:
                self.logger.error(f"Error in observer {type(observer).__name__}: {e}")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called when an epoch ends."""
        for observer in self.observers:
            try:
                observer.on_epoch_end(epoch, metrics)
            except Exception as e:
                self.logger.error(f"Error in observer {type(observer).__name__}: {e}")
    
    def on_step_complete(self, step: int, metrics: Dict[str, Any]) -> None:
        """Called when a training step completes."""
        for observer in self.observers:
            try:
                observer.on_step_complete(step, metrics)
            except Exception as e:
                self.logger.error(f"Error in observer {type(observer).__name__}: {e}")
    
    def on_validation_start(self, epoch: int) -> None:
        """Called when validation starts."""
        for observer in self.observers:
            try:
                observer.on_validation_start(epoch)
            except Exception as e:
                self.logger.error(f"Error in observer {type(observer).__name__}: {e}")
    
    def on_validation_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called when validation ends."""
        for observer in self.observers:
            try:
                observer.on_validation_end(epoch, metrics)
            except Exception as e:
                self.logger.error(f"Error in observer {type(observer).__name__}: {e}")


class TrainingMonitorFactory:
    """
    Factory for creating training monitor components.
    
    Provides a centralized way to create different types of training monitors.
    """
    
    @staticmethod
    def create_basic_observer(logger_name: str = "training_observer") -> ITrainingObserver:
        """Create a basic training observer."""
        return BasicTrainingObserver(logger_name)
    
    @staticmethod
    def create_metrics_collector() -> IMetricsCollector:
        """Create a metrics collector."""
        return MetricsCollector()
    
    @staticmethod
    def create_console_logger(logger_name: str = "console_logger") -> ILoggingStrategy:
        """Create a console logging strategy."""
        return ConsoleLoggingStrategy(logger_name)
    
    @staticmethod
    def create_file_logger(log_file: str, logger_name: str = "file_logger") -> ILoggingStrategy:
        """Create a file logging strategy."""
        return FileLoggingStrategy(log_file, logger_name)
    
    @staticmethod
    def create_file_observer(log_file: str, logger_name: str = "file_training_observer") -> ITrainingObserver:
        """Create a file training observer."""
        return FileTrainingObserver(log_file, logger_name)
    
    @staticmethod
    def create_progress_tracker() -> IProgressTracker:
        """Create a progress tracker."""
        return ProgressTracker()
    
    @staticmethod
    def create_composite_observer(observers: List[ITrainingObserver]) -> ITrainingObserver:
        """Create a composite observer."""
        return CompositeTrainingObserver(observers)
