"""
Model Factory Interface

Defines the contract for model creation and management in RLHF training.
Supports different model types and configurations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
from torch.nn import Module


class IModelFactory(ABC):
    """Interface for model creation and management."""
    
    @abstractmethod
    def create_base_model(self, config: Dict[str, Any]) -> Module:
        """
        Create the base language model.
        
        Args:
            config: Configuration dictionary containing model parameters
            
        Returns:
            The base language model
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If model creation fails
        """
        pass
    
    @abstractmethod
    def create_reward_model(self, base_model: Module, config: Dict[str, Any]) -> Module:
        """
        Create the reward model wrapper.
        
        Args:
            base_model: The base language model
            config: Configuration dictionary
            
        Returns:
            The reward model wrapper
            
        Raises:
            ValueError: If reward model configuration is invalid
        """
        pass
    
    @abstractmethod
    def create_tokenizer(self, config: Dict[str, Any]) -> Any:
        """
        Create the tokenizer.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            The tokenizer instance
            
        Raises:
            ValueError: If tokenizer configuration is invalid
        """
        pass
    
    @abstractmethod
    def setup_device(self, config: Dict[str, Any]) -> torch.device:
        """
        Setup and configure the device (GPU/CPU).
        
        Args:
            config: Configuration dictionary
            
        Returns:
            The configured device
            
        Raises:
            RuntimeError: If device setup fails
        """
        pass
    
    @abstractmethod
    def configure_model_optimizations(self, model: Module, config: Dict[str, Any]) -> Module:
        """
        Apply model-specific optimizations.
        
        Args:
            model: The model to optimize
            config: Configuration dictionary
            
        Returns:
            The optimized model
        """
        pass


class IModelManager(ABC):
    """Interface for model lifecycle management."""
    
    @abstractmethod
    def move_to_device(self, model: Module, device: torch.device) -> Module:
        """
        Move model to specified device.
        
        Args:
            model: The model to move
            device: Target device
            
        Returns:
            Model on target device
        """
        pass
    
    @abstractmethod
    def enable_gradient_checkpointing(self, model: Module) -> Module:
        """
        Enable gradient checkpointing for memory optimization.
        
        Args:
            model: The model to optimize
            
        Returns:
            Model with gradient checkpointing enabled
        """
        pass
    
    @abstractmethod
    def configure_mixed_precision(self, model: Module, config: Dict[str, Any]) -> Tuple[Module, Optional[Any]]:
        """
        Configure mixed precision training.
        
        Args:
            model: The model to configure
            config: Configuration dictionary
            
        Returns:
            Tuple of (model, scaler) where scaler is None if mixed precision disabled
        """
        pass
