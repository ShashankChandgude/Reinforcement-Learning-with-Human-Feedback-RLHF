"""
Optimization Manager Interface

Defines the contract for optimization and memory management in RLHF training.
Supports different optimization strategies and memory optimization techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch.optim import Optimizer
from torch.nn import Module


class IOptimizationManager(ABC):
    """Interface for optimization management."""
    
    @abstractmethod
    def create_optimizer(self, model: Module, config: Dict[str, Any]) -> Optimizer:
        """
        Create an optimizer for the model.
        
        Args:
            model: Model to optimize
            config: Optimization configuration
            
        Returns:
            Configured optimizer
            
        Raises:
            ValueError: If optimization configuration is invalid
        """
        pass
    
    @abstractmethod
    def setup_mixed_precision(self, config: Dict[str, Any]) -> Tuple[bool, Optional[torch.cuda.amp.GradScaler]]:
        """
        Setup mixed precision training.
        
        Args:
            config: Mixed precision configuration
            
        Returns:
            Tuple of (enabled, scaler) where scaler is None if disabled
        """
        pass
    
    @abstractmethod
    def configure_gradient_accumulation(self, config: Dict[str, Any]) -> int:
        """
        Configure gradient accumulation.
        
        Args:
            config: Gradient accumulation configuration
            
        Returns:
            Number of gradient accumulation steps
        """
        pass
    
    @abstractmethod
    def setup_gradient_clipping(self, config: Dict[str, Any]) -> Optional[float]:
        """
        Setup gradient clipping.
        
        Args:
            config: Gradient clipping configuration
            
        Returns:
            Gradient clipping threshold, or None if disabled
        """
        pass


class IMemoryManager(ABC):
    """Interface for memory management."""
    
    @abstractmethod
    def optimize_memory_usage(self, config: Dict[str, Any]) -> None:
        """
        Apply memory optimization settings.
        
        Args:
            config: Memory optimization configuration
        """
        pass
    
    @abstractmethod
    def clear_cache(self) -> None:
        """Clear GPU cache if available."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary of memory usage metrics
        """
        pass
    
    @abstractmethod
    def cleanup_resources(self) -> None:
        """Clean up allocated resources."""
        pass


class IDeviceManager(ABC):
    """Interface for device management."""
    
    @abstractmethod
    def setup_device(self, config: Dict[str, Any]) -> torch.device:
        """
        Setup and configure the device.
        
        Args:
            config: Device configuration
            
        Returns:
            Configured device
        """
        pass
    
    @abstractmethod
    def optimize_device_settings(self, device: torch.device, config: Dict[str, Any]) -> None:
        """
        Optimize device settings for performance.
        
        Args:
            device: Target device
            config: Device optimization configuration
        """
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the current device.
        
        Returns:
            Dictionary of device information
        """
        pass


class IPerformanceOptimizer(ABC):
    """Interface for performance optimization."""
    
    @abstractmethod
    def enable_compilation(self, model: Module, config: Dict[str, Any]) -> Module:
        """
        Enable model compilation for performance.
        
        Args:
            model: Model to compile
            config: Compilation configuration
            
        Returns:
            Compiled model
        """
        pass
    
    @abstractmethod
    def optimize_dataloader(self, dataloader: torch.utils.data.DataLoader, config: Dict[str, Any]) -> torch.utils.data.DataLoader:
        """
        Optimize DataLoader for performance.
        
        Args:
            dataloader: DataLoader to optimize
            config: Optimization configuration
            
        Returns:
            Optimized DataLoader
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        pass


class IOptimizationStrategy(ABC):
    """Interface for different optimization strategies."""
    
    @abstractmethod
    def apply_optimization(self, model: Module, config: Dict[str, Any]) -> Module:
        """
        Apply optimization strategy to the model.
        
        Args:
            model: Model to optimize
            config: Optimization configuration
            
        Returns:
            Optimized model
        """
        pass
    
    @abstractmethod
    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Get information about the optimization strategy.
        
        Returns:
            Dictionary of optimization information
        """
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate optimization configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
