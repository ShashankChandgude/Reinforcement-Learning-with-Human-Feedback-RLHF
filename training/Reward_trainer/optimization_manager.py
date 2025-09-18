"""
Optimization Manager Implementations

Concrete implementations of optimization and memory management for RLHF training.
Supports different optimization strategies and memory optimization techniques.
"""

import torch
from typing import Dict, Any, Optional, Tuple, List
from torch.optim import Optimizer, AdamW
from torch.nn import Module
from .interfaces.optimization_manager_interface import (
    IOptimizationManager, 
    IMemoryManager, 
    IDeviceManager, 
    IPerformanceOptimizer, 
    IOptimizationStrategy
)
from utils.logging_utils import setup_logger


class RewardOptimizationManager(IOptimizationManager):
    """
    Optimization manager for reward model training.
    
    Handles optimizer creation, mixed precision, and gradient management.
    """
    
    def __init__(self):
        """Initialize the optimization manager."""
        self.logger = setup_logger("optimization_manager")
    
    def create_optimizer(self, model: Module, config: Dict[str, Any]) -> Optimizer:
        """
        Create an optimizer for the model.
        
        Args:
            model: Model to optimize
            config: Optimization configuration
            
        Returns:
            Configured optimizer
        """
        try:
            # Extract optimizer configuration
            lr = float(config.get("learning_rate", 3e-5))
            weight_decay = float(config.get("weight_decay", 0.01))
            betas = config.get("betas", (0.9, 0.999))
            eps = float(config.get("eps", 1e-8))
            
            self.logger.info(f"Creating AdamW optimizer: lr={lr}, weight_decay={weight_decay}")
            
            # Create optimizer
            optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
            
            self.logger.info("Optimizer created successfully")
            return optimizer
            
        except Exception as e:
            self.logger.error(f"Failed to create optimizer: {e}")
            raise ValueError(f"Optimizer creation failed: {e}") from e
    
    def setup_mixed_precision(self, config: Dict[str, Any]) -> Tuple[bool, Optional[torch.cuda.amp.GradScaler]]:
        """
        Setup mixed precision training.
        
        Args:
            config: Mixed precision configuration
            
        Returns:
            Tuple of (enabled, scaler)
        """
        try:
            enabled = config.get("mixed_precision", False)
            
            if enabled and torch.cuda.is_available():
                scaler = torch.cuda.amp.GradScaler()
                self.logger.info("Mixed precision training enabled")
                return True, scaler
            else:
                self.logger.info("Mixed precision training disabled")
                return False, None
                
        except Exception as e:
            self.logger.error(f"Failed to setup mixed precision: {e}")
            return False, None
    
    def configure_gradient_accumulation(self, config: Dict[str, Any]) -> int:
        """
        Configure gradient accumulation.
        
        Args:
            config: Gradient accumulation configuration
            
        Returns:
            Number of gradient accumulation steps
        """
        steps = int(config.get("gradient_accumulation_steps", 1))
        self.logger.info(f"Gradient accumulation steps: {steps}")
        return steps
    
    def setup_gradient_clipping(self, config: Dict[str, Any]) -> Optional[float]:
        """
        Setup gradient clipping.
        
        Args:
            config: Gradient clipping configuration
            
        Returns:
            Gradient clipping threshold, or None if disabled
        """
        max_grad_norm = config.get("max_grad_norm")
        if max_grad_norm is not None:
            threshold = float(max_grad_norm)
            self.logger.info(f"Gradient clipping enabled: max_norm={threshold}")
            return threshold
        else:
            self.logger.info("Gradient clipping disabled")
            return None


class CUDAMemoryManager(IMemoryManager):
    """
    CUDA memory manager for GPU optimization.
    
    Handles GPU memory optimization and cleanup.
    """
    
    def __init__(self):
        """Initialize the memory manager."""
        self.logger = setup_logger("memory_manager")
    
    def optimize_memory_usage(self, config: Dict[str, Any]) -> None:
        """Apply memory optimization settings."""
        try:
            if not torch.cuda.is_available():
                self.logger.info("CUDA not available, skipping memory optimization")
                return
            
            # Clear cache
            if config.get("clear_cache_on_start", True):
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared")
            
            # Set memory fraction if specified
            memory_fraction = config.get("memory_fraction")
            if memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                self.logger.info(f"Memory fraction set to: {memory_fraction}")
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear GPU cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not torch.cuda.is_available():
            return {"gpu_memory_allocated": 0.0, "gpu_memory_reserved": 0.0}
        
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            return {
                "gpu_memory_allocated": allocated,
                "gpu_memory_reserved": reserved
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return {"gpu_memory_allocated": 0.0, "gpu_memory_reserved": 0.0}
    
    def cleanup_resources(self) -> None:
        """Clean up allocated resources."""
        self.clear_cache()
        self.logger.info("Memory resources cleaned up")


class DeviceManager(IDeviceManager):
    """
    Device manager for GPU/CPU configuration.
    
    Handles device setup and optimization.
    """
    
    def __init__(self):
        """Initialize the device manager."""
        self.logger = setup_logger("device_manager")
        self.current_device = None
    
    def setup_device(self, config: Dict[str, Any]) -> torch.device:
        """
        Setup and configure the device.
        
        Args:
            config: Device configuration
            
        Returns:
            Configured device
        """
        try:
            device_name = config.get("device", "auto")
            force_cpu = config.get("force_cpu", False)
            
            if force_cpu:
                device = torch.device("cpu")
                self.logger.info("Forced CPU usage")
            elif device_name == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                else:
                    device = torch.device("cpu")
                    self.logger.warning("GPU not available, using CPU")
            else:
                device = torch.device(device_name)
                self.logger.info(f"Using specified device: {device}")
            
            self.current_device = device
            return device
            
        except Exception as e:
            self.logger.error(f"Failed to setup device: {e}")
            raise RuntimeError(f"Device setup failed: {e}") from e
    
    def optimize_device_settings(self, device: torch.device, config: Dict[str, Any]) -> None:
        """Optimize device settings for performance."""
        try:
            if device.type == "cuda":
                # CUDA optimizations
                torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", True)
                torch.backends.cudnn.deterministic = config.get("cudnn_deterministic", False)
                self.logger.info("CUDA optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"Device optimization failed: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        info = {
            "device": str(self.current_device) if self.current_device else "unknown",
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_device_capability": torch.cuda.get_device_capability()
            })
        
        return info


class PerformanceOptimizer(IPerformanceOptimizer):
    """
    Performance optimizer for training efficiency.
    
    Handles model compilation and DataLoader optimization.
    """
    
    def __init__(self):
        """Initialize the performance optimizer."""
        self.logger = setup_logger("performance_optimizer")
    
    def enable_compilation(self, model: Module, config: Dict[str, Any]) -> Module:
        """Enable model compilation for performance."""
        try:
            if config.get("compile_model", False):
                model = torch.compile(model)
                self.logger.info("Model compilation enabled")
            return model
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}")
            return model
    
    def optimize_dataloader(self, dataloader: torch.utils.data.DataLoader, config: Dict[str, Any]) -> torch.utils.data.DataLoader:
        """Optimize DataLoader for performance."""
        # DataLoader is already optimized in dataset manager
        return dataloader
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        # Placeholder for performance metrics
        return {"optimization_enabled": True}


class StandardOptimizationStrategy(IOptimizationStrategy):
    """
    Standard optimization strategy for reward model training.
    
    Implements standard optimization techniques for RLHF training.
    """
    
    def __init__(self):
        """Initialize the optimization strategy."""
        self.logger = setup_logger("optimization_strategy")
        self.config = {}
    
    def apply_optimization(self, model: Module, config: Dict[str, Any]) -> Module:
        """Apply optimization strategy to the model."""
        try:
            self.config = config.copy()
            
            # Gradient checkpointing
            if config.get("gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
            
            # Model compilation
            if config.get("compile_model", False):
                try:
                    model = torch.compile(model)
                    self.logger.info("Model compilation enabled")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Optimization application failed: {e}")
            return model
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the optimization strategy."""
        return {
            "strategy": "standard",
            "config": self.config,
            "gradient_checkpointing": self.config.get("gradient_checkpointing", False),
            "model_compilation": self.config.get("compile_model", False)
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate optimization configuration."""
        try:
            # Validate required parameters
            required_params = ["learning_rate"]
            for param in required_params:
                if param not in config:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate parameter types and ranges
            lr = config.get("learning_rate")
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.logger.error(f"Invalid learning rate: {lr}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


class OptimizationManagerFactory:
    """
    Factory for creating optimization manager components.
    
    Provides a centralized way to create different types of optimization managers.
    """
    
    @staticmethod
    def create_optimization_manager() -> IOptimizationManager:
        """Create an optimization manager."""
        return RewardOptimizationManager()
    
    @staticmethod
    def create_memory_manager() -> IMemoryManager:
        """Create a memory manager."""
        return CUDAMemoryManager()
    
    @staticmethod
    def create_device_manager() -> IDeviceManager:
        """Create a device manager."""
        return DeviceManager()
    
    @staticmethod
    def create_performance_optimizer() -> IPerformanceOptimizer:
        """Create a performance optimizer."""
        return PerformanceOptimizer()
    
    @staticmethod
    def create_optimization_strategy() -> IOptimizationStrategy:
        """Create an optimization strategy."""
        return StandardOptimizationStrategy()
