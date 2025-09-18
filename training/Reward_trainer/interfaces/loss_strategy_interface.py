"""
Loss Strategy Interface

Defines the contract for different loss computation strategies in RLHF training.
Supports various preference learning loss functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class ILossStrategy(ABC):
    """Interface for loss computation strategies."""
    
    @abstractmethod
    def compute_loss(self, 
                    chosen_rewards: torch.Tensor, 
                    rejected_rewards: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """
        Compute the loss between chosen and rejected rewards.
        
        Args:
            chosen_rewards: Reward scores for chosen responses [batch_size]
            rejected_rewards: Reward scores for rejected responses [batch_size]
            **kwargs: Additional parameters specific to the loss strategy
            
        Returns:
            Computed loss tensor
            
        Raises:
            ValueError: If input tensors have incompatible shapes
            RuntimeError: If loss computation fails
        """
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> list[str]:
        """
        Get the list of required parameters for this loss strategy.
        
        Returns:
            List of required parameter names
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate that all required parameters are present and valid.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass


class ILossConfigurable(ABC):
    """Interface for configurable loss strategies."""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the loss strategy with given parameters.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration of the loss strategy.
        
        Returns:
            Current configuration dictionary
        """
        pass


class ILossMetrics(ABC):
    """Interface for loss strategy metrics and monitoring."""
    
    @abstractmethod
    def compute_metrics(self, 
                       chosen_rewards: torch.Tensor, 
                       rejected_rewards: torch.Tensor,
                       loss: torch.Tensor) -> Dict[str, float]:
        """
        Compute additional metrics for monitoring.
        
        Args:
            chosen_rewards: Reward scores for chosen responses
            rejected_rewards: Reward scores for rejected responses
            loss: Computed loss value
            
        Returns:
            Dictionary of metric names and values
        """
        pass
    
    @abstractmethod
    def get_metric_names(self) -> list[str]:
        """
        Get the list of metric names this strategy provides.
        
        Returns:
            List of metric names
        """
        pass
