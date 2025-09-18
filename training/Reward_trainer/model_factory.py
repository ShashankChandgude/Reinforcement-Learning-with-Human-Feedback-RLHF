"""
Model Factory Implementations

Concrete implementations of model creation and management for RLHF training.
Supports different model types and configurations with proper error handling.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from .interfaces.model_factory_interface import IModelFactory, IModelManager
from .interfaces import IModelFactory, IModelManager
from training.Reward_Model.reward_model import RewardModel
from utils.logging_utils import setup_logger


class RewardModelFactory(IModelFactory, IModelManager):
    """
    Factory for creating reward models and related components.
    
    This factory handles the creation of base models, reward models,
    tokenizers, and device configuration for RLHF training.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self.logger = setup_logger("model_factory")
        self._device_cache = None
    
    def create_base_model(self, config: Dict[str, Any]) -> torch.nn.Module:
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
        try:
            # Extract model configuration
            model_id = config.get("model") or config.get("base_model")
            if not model_id:
                raise ValueError("Model ID not specified in configuration")
            
            # Get model parameters
            torch_dtype = config.get("torch_dtype", torch.float32)
            device_map = config.get("device_map", "auto" if torch.cuda.is_available() else None)
            use_cache = config.get("use_cache", False)
            
            self.logger.info(f"Creating base model: {model_id}")
            self.logger.info(f"Model dtype: {torch_dtype}, device_map: {device_map}")
            
            # Create base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                use_cache=use_cache
            )
            
            # Configure model settings
            base_model.config.use_cache = use_cache
            
            self.logger.info(f"Base model created successfully: {type(base_model).__name__}")
            return base_model
            
        except Exception as e:
            self.logger.error(f"Failed to create base model: {e}")
            raise RuntimeError(f"Model creation failed: {e}") from e
    
    def create_reward_model(self, base_model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
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
        try:
            self.logger.info("Creating reward model wrapper")
            
            # Create reward model
            reward_model = RewardModel(base_model)
            
            # Apply any reward model specific configurations
            reward_config = config.get("reward_model", {})
            if reward_config:
                self.logger.info(f"Applying reward model configuration: {reward_config}")
                # Add any reward model specific configuration here
            
            self.logger.info("Reward model created successfully")
            return reward_model
            
        except Exception as e:
            self.logger.error(f"Failed to create reward model: {e}")
            raise ValueError(f"Reward model creation failed: {e}") from e
    
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
        try:
            # Extract tokenizer configuration
            model_id = config.get("model") or config.get("base_model")
            if not model_id:
                raise ValueError("Model ID not specified for tokenizer")
            
            use_fast = config.get("use_fast_tokenizer", True)
            pad_token = config.get("pad_token", "eos")
            
            self.logger.info(f"Creating tokenizer for model: {model_id}")
            self.logger.info(f"Use fast tokenizer: {use_fast}")
            
            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=use_fast
            )
            
            # Configure pad token
            if pad_token == "eos" and tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            elif pad_token and pad_token != "eos":
                tokenizer.pad_token = pad_token
            
            # Ensure pad token is set
            if not tokenizer.pad_token:
                self.logger.warning("No pad token found, using eos token")
                tokenizer.pad_token = tokenizer.eos_token
            
            self.logger.info(f"Tokenizer created successfully: {type(tokenizer).__name__}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to create tokenizer: {e}")
            raise ValueError(f"Tokenizer creation failed: {e}") from e
    
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
        try:
            # Get device configuration
            device_name = config.get("device", "auto")
            force_cpu = config.get("force_cpu", False)
            
            # Determine device
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
            
            # Cache device for reuse
            self._device_cache = device
            
            # Apply device optimizations
            self._optimize_device_settings(device, config)
            
            return device
            
        except Exception as e:
            self.logger.error(f"Failed to setup device: {e}")
            raise RuntimeError(f"Device setup failed: {e}") from e
    
    def configure_model_optimizations(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        """
        Apply model-specific optimizations.
        
        Args:
            model: The model to optimize
            config: Configuration dictionary
            
        Returns:
            The optimized model
        """
        try:
            optimization_config = config.get("optimization", {})
            
            # Gradient checkpointing
            if optimization_config.get("gradient_checkpointing", False):
                model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
            
            # Mixed precision configuration
            if optimization_config.get("mixed_precision", False):
                self.logger.info("Mixed precision training enabled")
                # Note: Mixed precision is handled in the training loop
            
            # Model compilation (if supported)
            if optimization_config.get("compile_model", False):
                try:
                    model = torch.compile(model)
                    self.logger.info("Model compilation enabled")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to configure model optimizations: {e}")
            return model  # Return unoptimized model on failure
    
    def move_to_device(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """
        Move model to specified device.
        
        Args:
            model: The model to move
            device: Target device
            
        Returns:
            Model on target device
        """
        try:
            self.logger.info(f"Moving model to device: {device}")
            model = model.to(device)
            return model
        except Exception as e:
            self.logger.error(f"Failed to move model to device: {e}")
            raise RuntimeError(f"Device move failed: {e}") from e
    
    def enable_gradient_checkpointing(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Enable gradient checkpointing for memory optimization.
        
        Args:
            model: The model to optimize
            
        Returns:
            Model with gradient checkpointing enabled
        """
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
            else:
                self.logger.warning("Model does not support gradient checkpointing")
            return model
        except Exception as e:
            self.logger.error(f"Failed to enable gradient checkpointing: {e}")
            return model
    
    def configure_mixed_precision(self, model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[torch.nn.Module, Optional[torch.cuda.amp.GradScaler]]:
        """
        Configure mixed precision training.
        
        Args:
            model: The model to configure
            config: Configuration dictionary
            
        Returns:
            Tuple of (model, scaler) where scaler is None if mixed precision disabled
        """
        try:
            use_mixed_precision = config.get("mixed_precision", False)
            
            if use_mixed_precision and torch.cuda.is_available():
                scaler = torch.cuda.amp.GradScaler()
                self.logger.info("Mixed precision training configured")
                return model, scaler
            else:
                self.logger.info("Mixed precision training disabled")
                return model, None
                
        except Exception as e:
            self.logger.error(f"Failed to configure mixed precision: {e}")
            return model, None
    
    def _optimize_device_settings(self, device: torch.device, config: Dict[str, Any]) -> None:
        """
        Apply device-specific optimizations.
        
        Args:
            device: Target device
            config: Configuration dictionary
        """
        try:
            if device.type == "cuda":
                # CUDA optimizations
                optimization_config = config.get("optimization", {})
                
                # CuDNN settings
                if optimization_config.get("cudnn_benchmark", True):
                    torch.backends.cudnn.benchmark = True
                    self.logger.info("CuDNN benchmark enabled")
                
                if optimization_config.get("cudnn_deterministic", False):
                    torch.backends.cudnn.deterministic = True
                    self.logger.info("CuDNN deterministic mode enabled")
                
                # Clear cache
                if optimization_config.get("clear_cache_on_start", True):
                    torch.cuda.empty_cache()
                    self.logger.info("GPU cache cleared")
            
        except Exception as e:
            self.logger.warning(f"Device optimization failed: {e}")


class ModelFactoryRegistry:
    """
    Registry for different model factory types.
    
    This registry allows for easy extension with different model factory
    implementations for different model types or frameworks.
    """
    
    _factories = {
        "reward_model": RewardModelFactory
    }
    
    @classmethod
    def create_factory(cls, factory_type: str) -> IModelFactory:
        """
        Create a model factory instance.
        
        Args:
            factory_type: Type of factory to create
            
        Returns:
            Model factory instance
            
        Raises:
            ValueError: If factory type is not supported
        """
        if factory_type not in cls._factories:
            available = ", ".join(cls._factories.keys())
            raise ValueError(f"Unknown factory type '{factory_type}'. Available: {available}")
        
        factory_class = cls._factories[factory_type]
        return factory_class()
    
    @classmethod
    def register_factory(cls, factory_type: str, factory_class: type) -> None:
        """
        Register a new model factory type.
        
        Args:
            factory_type: Name for the factory type
            factory_class: Factory class that implements IModelFactory
        """
        if not issubclass(factory_class, IModelFactory):
            raise ValueError(f"Factory class must implement IModelFactory")
        cls._factories[factory_type] = factory_class
    
    @classmethod
    def get_available_factories(cls) -> list[str]:
        """Get list of available factory types."""
        return list(cls._factories.keys())
