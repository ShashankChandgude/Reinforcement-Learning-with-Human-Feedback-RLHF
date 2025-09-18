"""
Loss Strategy Implementations

Concrete implementations of loss computation strategies for RLHF training.
Supports various preference learning loss functions with proper configuration.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from .interfaces.loss_strategy_interface import (
    ILossStrategy, 
    ILossConfigurable, 
    ILossMetrics
)


class BradleyTerryLoss(ILossStrategy, ILossConfigurable, ILossMetrics):
    """
    Bradley-Terry preference loss implementation.
    
    This loss function models the probability that a chosen response is preferred
    over a rejected response using the Bradley-Terry model.
    """
    
    def __init__(self, margin: float = 0.0):
        """
        Initialize Bradley-Terry loss.
        
        Args:
            margin: Margin for margin-based loss (0.0 disables margin)
        """
        self.margin = margin
        self.config = {"margin": margin}
    
    def compute_loss(self, 
                    chosen_rewards: torch.Tensor, 
                    rejected_rewards: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """
        Compute Bradley-Terry preference loss.
        
        Args:
            chosen_rewards: Reward scores for chosen responses [batch_size]
            rejected_rewards: Reward scores for rejected responses [batch_size]
            **kwargs: Additional parameters (unused for this strategy)
            
        Returns:
            Computed loss tensor
            
        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        # Validate inputs
        if chosen_rewards.shape != rejected_rewards.shape:
            raise ValueError(f"Shape mismatch: chosen_rewards {chosen_rewards.shape} != rejected_rewards {rejected_rewards.shape}")
        
        # Ensure rewards are scalars
        if chosen_rewards.dim() > 1:
            chosen_rewards = chosen_rewards.squeeze()
        if rejected_rewards.dim() > 1:
            rejected_rewards = rejected_rewards.squeeze()
        
        # Compute reward difference
        reward_diff = chosen_rewards - rejected_rewards
        
        if self.margin > 0.0:
            # Margin-based loss (hinge loss)
            loss = F.relu(self.margin - reward_diff).mean()
        else:
            # Standard Bradley-Terry loss via BCE
            targets = torch.ones_like(reward_diff)
            loss = F.binary_cross_entropy_with_logits(reward_diff, targets)
        
        return loss
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this loss strategy."""
        return ["margin"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for this loss strategy."""
        if "margin" not in parameters:
            return False
        margin = parameters["margin"]
        return isinstance(margin, (int, float)) and margin >= 0.0
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the loss strategy."""
        if not self.validate_parameters(config):
            raise ValueError(f"Invalid parameters for BradleyTerryLoss: {config}")
        self.margin = float(config["margin"])
        self.config = config.copy()
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def compute_metrics(self, 
                       chosen_rewards: torch.Tensor, 
                       rejected_rewards: torch.Tensor,
                       loss: torch.Tensor) -> Dict[str, float]:
        """Compute additional metrics for monitoring."""
        with torch.no_grad():
            # Preference accuracy
            correct_preferences = (chosen_rewards > rejected_rewards).sum().item()
            total_preferences = chosen_rewards.size(0)
            accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0.0
            
            # Reward statistics
            chosen_mean = chosen_rewards.mean().item()
            rejected_mean = rejected_rewards.mean().item()
            reward_diff_mean = (chosen_rewards - rejected_rewards).mean().item()
            
            return {
                "preference_accuracy": accuracy,
                "chosen_reward_mean": chosen_mean,
                "rejected_reward_mean": rejected_mean,
                "reward_diff_mean": reward_diff_mean,
                "loss_value": loss.item()
            }
    
    def get_metric_names(self) -> List[str]:
        """Get metric names this strategy provides."""
        return [
            "preference_accuracy",
            "chosen_reward_mean", 
            "rejected_reward_mean",
            "reward_diff_mean",
            "loss_value"
        ]


class HingeLoss(ILossStrategy, ILossConfigurable, ILossMetrics):
    """
    Hinge loss implementation for preference learning.
    
    This loss function enforces a margin between chosen and rejected rewards.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize hinge loss.
        
        Args:
            margin: Required margin between chosen and rejected rewards
        """
        self.margin = margin
        self.config = {"margin": margin}
    
    def compute_loss(self, 
                    chosen_rewards: torch.Tensor, 
                    rejected_rewards: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """Compute hinge loss."""
        # Validate inputs
        if chosen_rewards.shape != rejected_rewards.shape:
            raise ValueError(f"Shape mismatch: chosen_rewards {chosen_rewards.shape} != rejected_rewards {rejected_rewards.shape}")
        
        # Ensure rewards are scalars
        if chosen_rewards.dim() > 1:
            chosen_rewards = chosen_rewards.squeeze()
        if rejected_rewards.dim() > 1:
            rejected_rewards = rejected_rewards.squeeze()
        
        # Compute hinge loss
        reward_diff = chosen_rewards - rejected_rewards
        loss = F.relu(self.margin - reward_diff).mean()
        
        return loss
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this loss strategy."""
        return ["margin"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for this loss strategy."""
        if "margin" not in parameters:
            return False
        margin = parameters["margin"]
        return isinstance(margin, (int, float)) and margin > 0.0
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the loss strategy."""
        if not self.validate_parameters(config):
            raise ValueError(f"Invalid parameters for HingeLoss: {config}")
        self.margin = float(config["margin"])
        self.config = config.copy()
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def compute_metrics(self, 
                       chosen_rewards: torch.Tensor, 
                       rejected_rewards: torch.Tensor,
                       loss: torch.Tensor) -> Dict[str, float]:
        """Compute additional metrics for monitoring."""
        with torch.no_grad():
            # Preference accuracy
            correct_preferences = (chosen_rewards > rejected_rewards).sum().item()
            total_preferences = chosen_rewards.size(0)
            accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0.0
            
            # Margin satisfaction
            reward_diff = chosen_rewards - rejected_rewards
            margin_satisfied = (reward_diff >= self.margin).sum().item()
            margin_satisfaction_rate = margin_satisfied / total_preferences if total_preferences > 0 else 0.0
            
            return {
                "preference_accuracy": accuracy,
                "margin_satisfaction_rate": margin_satisfaction_rate,
                "loss_value": loss.item()
            }
    
    def get_metric_names(self) -> List[str]:
        """Get metric names this strategy provides."""
        return [
            "preference_accuracy",
            "margin_satisfaction_rate", 
            "loss_value"
        ]


class LogSigmoidLoss(ILossStrategy, ILossConfigurable, ILossMetrics):
    """
    Log-sigmoid loss implementation for preference learning.
    
    This loss function uses the log-sigmoid of the reward difference.
    """
    
    def __init__(self, margin: float = 0.0):
        """
        Initialize log-sigmoid loss.
        
        Args:
            margin: Margin for the loss computation
        """
        self.margin = margin
        self.config = {"margin": margin}
    
    def compute_loss(self, 
                    chosen_rewards: torch.Tensor, 
                    rejected_rewards: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """Compute log-sigmoid loss."""
        # Validate inputs
        if chosen_rewards.shape != rejected_rewards.shape:
            raise ValueError(f"Shape mismatch: chosen_rewards {chosen_rewards.shape} != rejected_rewards {rejected_rewards.shape}")
        
        # Ensure rewards are scalars
        if chosen_rewards.dim() > 1:
            chosen_rewards = chosen_rewards.squeeze()
        if rejected_rewards.dim() > 1:
            rejected_rewards = rejected_rewards.squeeze()
        
        # Compute log-sigmoid loss
        reward_diff = chosen_rewards - rejected_rewards - self.margin
        loss = -F.logsigmoid(reward_diff).mean()
        
        return loss
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this loss strategy."""
        return ["margin"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for this loss strategy."""
        if "margin" not in parameters:
            return False
        margin = parameters["margin"]
        return isinstance(margin, (int, float)) and margin >= 0.0
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the loss strategy."""
        if not self.validate_parameters(config):
            raise ValueError(f"Invalid parameters for LogSigmoidLoss: {config}")
        self.margin = float(config["margin"])
        self.config = config.copy()
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def compute_metrics(self, 
                       chosen_rewards: torch.Tensor, 
                       rejected_rewards: torch.Tensor,
                       loss: torch.Tensor) -> Dict[str, float]:
        """Compute additional metrics for monitoring."""
        with torch.no_grad():
            # Preference accuracy
            correct_preferences = (chosen_rewards > rejected_rewards).sum().item()
            total_preferences = chosen_rewards.size(0)
            accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0.0
            
            # Reward statistics
            chosen_mean = chosen_rewards.mean().item()
            rejected_mean = rejected_rewards.mean().item()
            reward_diff_mean = (chosen_rewards - rejected_rewards).mean().item()
            
            return {
                "preference_accuracy": accuracy,
                "chosen_reward_mean": chosen_mean,
                "rejected_reward_mean": rejected_mean,
                "reward_diff_mean": reward_diff_mean,
                "loss_value": loss.item()
            }
    
    def get_metric_names(self) -> List[str]:
        """Get metric names this strategy provides."""
        return [
            "preference_accuracy",
            "chosen_reward_mean",
            "rejected_reward_mean", 
            "reward_diff_mean",
            "loss_value"
        ]


class LossStrategyFactory:
    """
    Factory for creating loss strategy instances.
    
    This factory provides a centralized way to create loss strategies
    based on configuration.
    """
    
    _strategies = {
        "bradley_terry": BradleyTerryLoss,
        "hinge": HingeLoss,
        "log_sigmoid": LogSigmoidLoss
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> ILossStrategy:
        """
        Create a loss strategy instance.
        
        Args:
            strategy_name: Name of the strategy to create
            config: Configuration for the strategy
            
        Returns:
            Loss strategy instance
            
        Raises:
            ValueError: If strategy name is not supported
        """
        if strategy_name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unknown loss strategy '{strategy_name}'. Available: {available}")
        
        strategy_class = cls._strategies[strategy_name]
        strategy = strategy_class()
        strategy.configure(config)
        
        return strategy
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """
        Register a new loss strategy.
        
        Args:
            name: Name for the strategy
            strategy_class: Strategy class that implements ILossStrategy
        """
        if not issubclass(strategy_class, ILossStrategy):
            raise ValueError(f"Strategy class must implement ILossStrategy")
        cls._strategies[name] = strategy_class
