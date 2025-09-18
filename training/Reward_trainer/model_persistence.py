"""
Model Persistence Implementations

Concrete implementations of model saving and loading for RLHF training.
Supports different persistence strategies and formats.
"""

import os
import json
import torch
from typing import Dict, Any, Optional, List, Union
from torch.nn import Module
from .interfaces.model_persistence_interface import (
    IModelPersistence, 
    IConfigurationPersistence, 
    IMetadataManager, 
    IFileManager, 
    IModelValidator
)
from utils.logging_utils import setup_logger


class RewardModelPersistence(IModelPersistence):
    """
    Model persistence implementation for reward models.
    
    Handles saving and loading of reward models with proper configuration management.
    """
    
    def __init__(self):
        """Initialize the model persistence manager."""
        self.logger = setup_logger("model_persistence")
    
    def save_model(self, 
                  model: Module, 
                  path: str, 
                  config: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model
            config: Model configuration
            metadata: Optional metadata to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            self.logger.info(f"Saving model to: {path}")
            
            # Save model state dict
            model_path = os.path.join(path, "reward_model.pth")
            torch.save(model.state_dict(), model_path)
            
            # Save base model if it has save_pretrained method
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'save_pretrained'):
                model.base_model.save_pretrained(path)
                self.logger.info("Base model saved")
            
            # Save metadata
            if metadata:
                metadata_path = os.path.join(path, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info("Metadata saved")
            
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Model saving failed: {e}") from e
    
    def load_model(self, 
                  path: str, 
                  config: Dict[str, Any]) -> Module:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            config: Model configuration
            
        Returns:
            Loaded model
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model path does not exist: {path}")
            
            self.logger.info(f"Loading model from: {path}")
            
            # Load base model first
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(path)
            
            # Create reward model wrapper
            from training.Reward_Model.reward_model import RewardModel
            model = RewardModel(base_model)
            
            # Load reward model weights
            model_path = os.path.join(path, "reward_model.pth")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                self.logger.info("Reward model weights loaded")
            else:
                self.logger.warning("Reward model weights not found, using base model")
            
            self.logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def save_checkpoint(self, 
                       model: Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       path: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a training checkpoint."""
        try:
            checkpoint_dir = os.path.join(path, f"checkpoint-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.logger.info(f"Saving checkpoint for epoch {epoch}")
            
            # Save model state
            model_path = os.path.join(checkpoint_dir, "reward_model.pth")
            torch.save(model.state_dict(), model_path)
            
            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizer_path)
            
            # Save metadata
            checkpoint_metadata = {
                "epoch": epoch,
                "model_type": type(model).__name__,
                "optimizer_type": type(optimizer).__name__
            }
            if metadata:
                checkpoint_metadata.update(metadata)
            
            metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(checkpoint_metadata, f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Checkpoint saving failed: {e}") from e
    
    def load_checkpoint(self, 
                       path: str, 
                       model: Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load a training checkpoint."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
            
            self.logger.info(f"Loading checkpoint from: {path}")
            
            # Load model state
            model_path = os.path.join(path, "reward_model.pth")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                self.logger.info("Model state loaded")
            
            # Load optimizer state
            if optimizer:
                optimizer_path = os.path.join(path, "optimizer.pt")
                if os.path.exists(optimizer_path):
                    optimizer_state = torch.load(optimizer_path, map_location='cpu')
                    optimizer.load_state_dict(optimizer_state)
                    self.logger.info("Optimizer state loaded")
            
            # Load metadata
            metadata_path = os.path.join(path, "checkpoint_metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.info("Checkpoint metadata loaded")
            
            self.logger.info("Checkpoint loaded successfully")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e


class ConfigurationPersistence(IConfigurationPersistence):
    """
    Configuration persistence implementation.
    
    Handles saving and loading of training configurations.
    """
    
    def __init__(self):
        """Initialize the configuration persistence manager."""
        self.logger = setup_logger("config_persistence")
    
    def save_configuration(self, 
                          config: Dict[str, Any], 
                          path: str) -> None:
        """Save configuration to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Configuration saved to: {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise RuntimeError(f"Configuration saving failed: {e}") from e
    
    def load_configuration(self, path: str) -> Dict[str, Any]:
        """Load configuration from disk."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Configuration file not found: {path}")
            
            with open(path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Configuration loaded from: {path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}") from e
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        try:
            required_sections = ["model", "training", "dataset"]
            
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


class MetadataManager(IMetadataManager):
    """
    Metadata manager for model and training metadata.
    
    Handles creation and management of metadata for models and training runs.
    """
    
    def __init__(self):
        """Initialize the metadata manager."""
        self.logger = setup_logger("metadata_manager")
    
    def create_metadata(self, 
                       model: Module, 
                       config: Dict[str, Any],
                       training_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create metadata for a model."""
        try:
            metadata = {
                "model_type": type(model).__name__,
                "model_config": config.get("model", {}),
                "training_config": config.get("training", {}),
                "dataset_config": config.get("dataset", {}),
                "created_at": torch.utils.data.get_worker_info(),
                "pytorch_version": torch.__version__
            }
            
            if training_info:
                metadata["training_info"] = training_info
            
            # Add model-specific metadata
            if hasattr(model, 'config'):
                metadata["model_parameters"] = {
                    "hidden_size": getattr(model.config, 'hidden_size', None),
                    "num_layers": getattr(model.config, 'num_hidden_layers', None),
                    "vocab_size": getattr(model.config, 'vocab_size', None)
                }
            
            self.logger.info("Metadata created successfully")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create metadata: {e}")
            return {}
    
    def save_metadata(self, 
                     metadata: Dict[str, Any], 
                     path: str) -> None:
        """Save metadata to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Metadata saved to: {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise RuntimeError(f"Metadata saving failed: {e}") from e
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load metadata from disk."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Metadata file not found: {path}")
            
            with open(path, 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Metadata loaded from: {path}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            raise RuntimeError(f"Metadata loading failed: {e}") from e


class FileManager(IFileManager):
    """
    File manager for handling file operations.
    
    Provides utilities for file and directory management.
    """
    
    def __init__(self):
        """Initialize the file manager."""
        self.logger = setup_logger("file_manager")
    
    def create_directory(self, path: str) -> None:
        """Create a directory if it doesn't exist."""
        try:
            os.makedirs(path, exist_ok=True)
            self.logger.debug(f"Directory created: {path}")
        except Exception as e:
            self.logger.error(f"Failed to create directory: {e}")
            raise RuntimeError(f"Directory creation failed: {e}") from e
    
    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(path)
    
    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        """List files in a directory."""
        try:
            if not os.path.exists(directory):
                return []
            
            files = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    if pattern is None or pattern in item:
                        files.append(item_path)
            
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            return []
    
    def cleanup_old_files(self, directory: str, max_files: int) -> None:
        """Clean up old files in a directory."""
        try:
            files = self.list_files(directory)
            if len(files) <= max_files:
                return
            
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove oldest files
            files_to_remove = files[:-max_files]
            for file_path in files_to_remove:
                os.remove(file_path)
                self.logger.info(f"Removed old file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files: {e}")


class ModelValidator(IModelValidator):
    """
    Model validator for validating model structure and weights.
    
    Provides validation utilities for models and checkpoints.
    """
    
    def __init__(self):
        """Initialize the model validator."""
        self.logger = setup_logger("model_validator")
        self.validation_errors = []
    
    def validate_model_structure(self, model: Module) -> bool:
        """Validate model structure."""
        try:
            self.validation_errors.clear()
            
            # Check if model has required methods
            required_methods = ['forward', 'parameters', 'state_dict']
            for method in required_methods:
                if not hasattr(model, method):
                    self.validation_errors.append(f"Missing required method: {method}")
                    return False
            
            # Check if model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                self.validation_errors.append("Model has no parameters")
                return False
            
            self.logger.info(f"Model structure validation passed: {param_count} parameters")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Model structure validation error: {e}")
            self.logger.error(f"Model structure validation failed: {e}")
            return False
    
    def validate_model_weights(self, model: Module) -> bool:
        """Validate model weights."""
        try:
            self.validation_errors.clear()
            
            # Check for NaN or Inf values
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    self.validation_errors.append(f"NaN values found in parameter: {name}")
                    return False
                
                if torch.isinf(param).any():
                    self.validation_errors.append(f"Inf values found in parameter: {name}")
                    return False
            
            self.logger.info("Model weights validation passed")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Model weights validation error: {e}")
            self.logger.error(f"Model weights validation failed: {e}")
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()


class ModelPersistenceFactory:
    """
    Factory for creating model persistence components.
    
    Provides a centralized way to create different types of persistence managers.
    """
    
    @staticmethod
    def create_model_persistence() -> IModelPersistence:
        """Create a model persistence manager."""
        return RewardModelPersistence()
    
    @staticmethod
    def create_configuration_persistence() -> IConfigurationPersistence:
        """Create a configuration persistence manager."""
        return ConfigurationPersistence()
    
    @staticmethod
    def create_metadata_manager() -> IMetadataManager:
        """Create a metadata manager."""
        return MetadataManager()
    
    @staticmethod
    def create_file_manager() -> IFileManager:
        """Create a file manager."""
        return FileManager()
    
    @staticmethod
    def create_model_validator() -> IModelValidator:
        """Create a model validator."""
        return ModelValidator()
