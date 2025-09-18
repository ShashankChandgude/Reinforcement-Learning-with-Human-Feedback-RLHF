"""
Training Observer Interface

Defines the contract for training monitoring and observation in RLHF training.
Supports different monitoring strategies and logging backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time


class ITrainingObserver(ABC):
    """Interface for training observation and monitoring."""
    
    @abstractmethod
    def on_training_start(self, config: Dict[str, Any]) -> None:
        """
        Called when training starts.
        
        Args:
            config: Training configuration dictionary
        """
        pass
    
    @abstractmethod
    def on_training_end(self, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training ends.
        
        Args:
            final_metrics: Final training metrics
        """
        pass
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """
        Called when an epoch starts.
        
        Args:
            epoch: Current epoch number (1-indexed)
            total_epochs: Total number of epochs
        """
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Called when an epoch ends.
        
        Args:
            epoch: Completed epoch number
            metrics: Epoch metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def on_step_complete(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Called when a training step completes.
        
        Args:
            step: Current step number
            metrics: Step metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def on_validation_start(self, epoch: int) -> None:
        """
        Called when validation starts.
        
        Args:
            epoch: Current epoch number
        """
        pass
    
    @abstractmethod
    def on_validation_end(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Called when validation ends.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
        """
        pass


class IMetricsCollector(ABC):
    """Interface for metrics collection and aggregation."""
    
    @abstractmethod
    def collect_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Collect metrics from a training step or epoch.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        pass
    
    @abstractmethod
    def get_aggregated_metrics(self, epoch: int) -> Dict[str, Any]:
        """
        Get aggregated metrics for an epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Dictionary of aggregated metrics
        """
        pass
    
    @abstractmethod
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of entire training run.
        
        Returns:
            Dictionary of training summary metrics
        """
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        pass


class ILoggingStrategy(ABC):
    """Interface for different logging strategies."""
    
    @abstractmethod
    def log_info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
            **kwargs: Additional context
        """
        pass
    
    @abstractmethod
    def log_warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
            **kwargs: Additional context
        """
        pass
    
    @abstractmethod
    def log_error(self, message: str, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
            **kwargs: Additional context
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        pass


class IProgressTracker(ABC):
    """Interface for training progress tracking."""
    
    @abstractmethod
    def update_progress(self, current: int, total: int, description: str = "") -> None:
        """
        Update progress tracking.
        
        Args:
            current: Current progress value
            total: Total progress value
            description: Optional description of current activity
        """
        pass
    
    @abstractmethod
    def get_progress_percentage(self) -> float:
        """
        Get current progress as percentage.
        
        Returns:
            Progress percentage (0.0 to 100.0)
        """
        pass
    
    @abstractmethod
    def get_estimated_time_remaining(self) -> Optional[float]:
        """
        Get estimated time remaining for training.
        
        Returns:
            Estimated time remaining in seconds, or None if not available
        """
        pass
